from __future__ import annotations

import os
from typing import Any, Callable, Literal, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typeguard import typechecked


def _select_metric(df, score_col: str, metric: Literal["mean", "max", "weighted"]) -> float:
    if metric == "mean":
        return float(df[score_col].mean())
    if metric == "max":
        return float(df.groupby("jb_index")[score_col].max().mean())
    if metric == "weighted":
        return (float(df[score_col].mean()) + float(df.groupby("jb_index")[score_col].max().mean())) / 2
    raise ValueError(f"Unknown metric: {metric!r}. Choose from 'mean', 'max', 'weighted'.")


@typechecked
class DPOTrainer:
    """Direct Preference Optimization trainer for IHO LoRA adapters.

    The trainer intentionally avoids experiment-tracker dependencies so the
    package can run in lightweight reproduction and smoke-test settings.
    """

    def __init__(
        self,
        wrapper: Any,
        learning_rate: float = 1e-5,
        beta: float = 0.1,
        run_name: str = "dpo_run",
    ) -> None:
        if not hasattr(wrapper, "has_lora") or not wrapper.has_lora:
            raise RuntimeError("DPO training requires LoRA adapters.")
        for attr in ("model", "encode", "mask_tokens", "compute_log_likelihood"):
            if not hasattr(wrapper, attr):
                raise TypeError(f"Wrapper must expose `{attr}`.")

        self.wrapper = wrapper
        self.beta = beta
        self.run_name = run_name
        self.learning_rate = learning_rate
        self.global_step = 0
        self.optimizer = torch.optim.AdamW(
            (p for p in self.wrapper.model.parameters() if p.requires_grad),
            lr=learning_rate,
        )

    def save_optimizer_state(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))

    def load_optimizer_state(self, path: str) -> None:
        state_path = os.path.join(path, "optimizer.pt")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"No optimizer state found at {state_path}")
        self.optimizer.load_state_dict(torch.load(state_path, map_location="cpu"))

    def _make_warmup_scheduler(self, warmup_epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
        def lr_lambda(epoch: int) -> float:
            if warmup_epochs <= 0:
                return 1.0
            if epoch == 0:
                return 0.0
            if epoch <= warmup_epochs:
                return epoch / warmup_epochs
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def compute_dpo_loss(
        self,
        chosen: list[str],
        rejected: list[str],
        masking_mode: Literal["all", "prompt", "attack"] = "prompt",
        mask_all: bool = False,
    ) -> torch.Tensor:
        if not self.wrapper.model.training:
            raise RuntimeError("Model must be in training mode. Call .train() first.")
        if not self.wrapper.has_lora:
            raise RuntimeError("DPO requires LoRA adapters to compare against base model.")

        chosen_ids = self.wrapper.encode(chosen)
        rejected_ids = self.wrapper.encode(rejected)

        chosen_masking = self.wrapper.mask_tokens(chosen_ids, masking_mode, mask_all)
        rejected_masking = self.wrapper.mask_tokens(rejected_ids, masking_mode, mask_all)

        chosen_log_pi = self.wrapper.compute_log_likelihood(
            chosen_masking.masked_ids, chosen_masking.mask_positions, chosen_ids, use_base_model=False
        )
        rejected_log_pi = self.wrapper.compute_log_likelihood(
            rejected_masking.masked_ids, rejected_masking.mask_positions, rejected_ids, use_base_model=False
        )

        with torch.no_grad():
            chosen_log_ref = self.wrapper.compute_log_likelihood(
                chosen_masking.masked_ids, chosen_masking.mask_positions, chosen_ids, use_base_model=True
            )
            rejected_log_ref = self.wrapper.compute_log_likelihood(
                rejected_masking.masked_ids, rejected_masking.mask_positions, rejected_ids, use_base_model=True
            )

        logits = self.beta * (chosen_log_pi - rejected_log_pi - chosen_log_ref + rejected_log_ref)
        return -F.logsigmoid(logits).mean()

    def train_step(
        self,
        chosen: list[str],
        rejected: list[str],
        masking_mode: Literal["all", "prompt", "attack"] = "prompt",
        mask_all: bool = False,
        max_grad_norm: Optional[float] = 1.0,
    ) -> float:
        self.optimizer.zero_grad()
        loss = self.compute_dpo_loss(chosen, rejected, masking_mode, mask_all)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.wrapper.model.parameters(), max_grad_norm)
        self.optimizer.step()
        self.global_step += 1
        return float(loss.detach().item())

    def train(
        self,
        dataloader: DataLoader,
        n_epochs: int = 100,
        masking_mode: Literal["all", "prompt", "attack"] = "prompt",
        mask_all: bool = False,
        repeats_per_example: int = 1,
        cycle_id: int = 0,
        run_path: str = "",
        checkpoint_every: int = 5,
        save_checkpoints: bool = False,
        patience: int = 1,
        sample_fn: Optional[Callable] = None,
        warmup_epochs: int = 0,
        load_optimizer_state: bool = False,
        metric: Literal["mean", "max", "weighted"] = "mean",
    ) -> list[float]:
        if load_optimizer_state and cycle_id > 0:
            self.load_optimizer_state(os.path.join(run_path, "checkpoints", f"optimizer_cycle_{cycle_id - 1}"))

        scheduler = self._make_warmup_scheduler(warmup_epochs)
        checkpoint_root = os.path.join(run_path, "checkpoints")
        if save_checkpoints:
            os.makedirs(checkpoint_root, exist_ok=True)

        best_metric: Optional[float] = None
        bad_checks = 0
        losses: list[float] = []
        self.wrapper.train()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_steps = 0
            for batch in dataloader:
                chosen = list(batch["chosen"])
                rejected = list(batch["rejected"])
                for _ in range(repeats_per_example):
                    epoch_loss += self.train_step(chosen, rejected, masking_mode, mask_all)
                    n_steps += 1

            losses.append(epoch_loss / max(n_steps, 1))
            scheduler.step()

            if save_checkpoints and checkpoint_every and epoch % checkpoint_every == 0:
                if sample_fn is None:
                    # Public support repo fallback: save epoch checkpoint without early-stop sampling.
                    self.wrapper.save_lora(os.path.join(checkpoint_root, f"epoch_{epoch + 1}"))
                    continue

                self.wrapper.eval()
                df = sample_fn(mode="dpo", save_df=False, cycle_id=cycle_id)
                current_metric = _select_metric(df, "judge_score_training", metric)
                improved = best_metric is None or current_metric > best_metric
                if improved:
                    best_metric = current_metric
                    bad_checks = 0
                    self.wrapper.save_lora(os.path.join(checkpoint_root, f"best_cycle_{cycle_id}"))
                else:
                    bad_checks += 1
                    if bad_checks > patience:
                        break
                self.wrapper.train()

        if load_optimizer_state:
            self.save_optimizer_state(os.path.join(checkpoint_root, f"optimizer_cycle_{cycle_id}"))

        if save_checkpoints:
            best_path = os.path.join(checkpoint_root, f"best_cycle_{cycle_id}")
            if not os.path.isdir(best_path):
                self.wrapper.save_lora(best_path)

        self.wrapper.eval()
        self.global_step = 0
        return losses
