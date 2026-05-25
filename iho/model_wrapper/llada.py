from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class MaskingResult:
    masked_ids: torch.Tensor
    mask_positions: torch.Tensor


class LLaDAWrapper:
    def __init__(
        self,
        model_name: str = "GSAI-ML/LLaDA-8B-Base",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        mask_token_id: int = 126336,
        attacker_input_size: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        lora_checkpoint: Optional[str] = None,
    ) -> None:
        self.device = device
        self.mask_token_id = mask_token_id
        self.dtype = dtype
        self.attacker_input_size = attacker_input_size

        logger.info("Loading base model %s on %s", model_name, device)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            cache_dir=cache_dir,
        ).to(device)

        self.has_lora = False
        if lora_checkpoint is not None:
            logger.info("Loading IHO adapter %s", lora_checkpoint)
            self.model = PeftModel.from_pretrained(self.model, lora_checkpoint)
            self.has_lora = True

        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        self.tokenizer.mask_token = "<|mdm_mask|>"
        self.padding_token_id = self.tokenizer.pad_token_id

    def eval(self) -> "LLaDAWrapper":
        self.model.eval()
        return self

    @contextmanager
    def disable_adapter(self):
        if not self.has_lora:
            yield
            return

        self.model.disable_adapter_layers()
        try:
            yield
        finally:
            self.model.enable_adapter_layers()

    def encode(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return encoded["input_ids"].to(self.device)

    def decode(self, token_ids: torch.Tensor) -> list[str]:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)

    def mask_tokens(
        self,
        token_ids: torch.Tensor,
        masking_mode: Literal["all", "prompt", "attack"],
        mask_all: bool = False,
    ) -> MaskingResult:
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        if masking_mode == "all":
            eligible_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        elif masking_mode == "prompt":
            eligible_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
            eligible_mask[:, : self.attacker_input_size] = True
        elif masking_mode == "attack":
            attack_len = self.attacker_input_size
            mask_block = token_ids.new_full((batch_size, attack_len), self.mask_token_id)
            token_ids = torch.cat([mask_block, token_ids], dim=1)

            eligible_mask = torch.zeros(token_ids.shape, dtype=torch.bool, device=device)
            eligible_mask[:, :attack_len] = True
        else:
            raise ValueError(f"Unknown masking_mode: {masking_mode}")

        if mask_all:
            mask_positions = eligible_mask.clone()
        else:
            rand = torch.rand(token_ids.shape, device=device)
            p = torch.rand(token_ids.shape[0], device=device).view(-1, 1)
            mask_positions = (rand < p) & eligible_mask

        masked_ids = token_ids.clone()
        masked_ids[mask_positions] = self.mask_token_id
        return MaskingResult(masked_ids=masked_ids, mask_positions=mask_positions)

    def compute_log_likelihood(
        self,
        masked_ids: torch.Tensor,
        mask_positions: torch.Tensor,
        target_ids: torch.Tensor,
        use_base_model: bool = False,
    ) -> torch.Tensor:
        context = self.disable_adapter() if use_base_model else torch.no_grad() if not self.model.training else nullcontext()

        with context:
            logits = self.model(masked_ids).logits

        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)
        return (target_log_probs * mask_positions).sum(dim=1)

    @torch.no_grad()
    def _add_gumbel_noise(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature == 0:
            return logits

        eps = 1e-20
        noise = -torch.log(-torch.log(torch.rand_like(logits, dtype=torch.bfloat16) + eps) + eps)
        return (logits + noise) / temperature

    @torch.no_grad()
    def _forward_process_batched(
        self,
        batch: torch.Tensor,
        fixed_mask: torch.Tensor,
        mask_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = batch.shape
        device = batch.device

        x = torch.randint(1, seq_len + 1, (batch_size,), device=device)
        indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        is_mask = indices < x.unsqueeze(1)

        randperm = torch.argsort(torch.rand(batch_size, seq_len, device=device), dim=1)
        is_mask = torch.gather(is_mask, 1, randperm)
        is_mask = is_mask & fixed_mask

        noisy_batch = torch.where(is_mask, mask_id, batch)
        mask_ratio = (x / seq_len).unsqueeze(1).expand(-1, seq_len)
        return noisy_batch, mask_ratio

    def _get_num_transfer_tokens(self, mask_index: torch.Tensor, steps: int) -> torch.Tensor:
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.zeros(
            mask_num.size(0),
            steps,
            device=mask_index.device,
            dtype=torch.int64,
        ) + base

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, : remainder[i]] += 1

        return num_transfer_tokens

    @torch.no_grad()
    def predict_masked(
        self,
        masked_ids: torch.Tensor,
        steps: int = 32,
        temperature: float = 0.0,
        remasking: Literal["low_confidence", "random"] = "low_confidence",
        mask_padding: bool = True,
        number_global_remask: int = 8,
        global_remasking: Literal["random", "low_confidence"] = "random",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model.training:
            raise RuntimeError("predict_masked must be run in evaluation mode.")

        x = masked_ids.to(self.device)
        batch_size, seq_len = x.shape

        known_mask = x != self.mask_token_id
        known_tokens = x.clone()
        global_conf = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=self.device)
        total_loglikelihood = torch.zeros(batch_size, device=self.device)

        special_token_ids = [
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        ]
        special_token_ids = list({int(s) for s in special_token_ids if s is not None})

        mask_index_initial = x == self.mask_token_id
        num_transfer_tokens = self._get_num_transfer_tokens(mask_index_initial, steps)

        for step in range(steps):
            mask_index = x == self.mask_token_id
            x_l, _ = self._forward_process_batched(x, known_mask, mask_id=self.mask_token_id)

            attention_mask = None
            if mask_padding:
                attention_mask = (x_l != self.padding_token_id).long()

            logits = self.model(x_l, attention_mask=attention_mask).logits
            logits_with_noise = self._add_gumbel_noise(logits, temperature=temperature)

            if special_token_ids:
                logits_with_noise[:, :, special_token_ids] = -float("inf")

            x0 = torch.argmax(logits_with_noise, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            chosen_logp = torch.gather(log_probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            if remasking == "low_confidence":
                probs = F.softmax(logits, dim=-1)
                x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1).to(self.device)
            elif remasking == "random":
                x0_p = torch.rand((batch_size, seq_len), device=self.device)
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking}")

            x0 = torch.where(known_mask, known_tokens, x0)

            neg_inf = torch.tensor(-float("inf"), device=self.device)
            confidence = torch.where(mask_index, x0_p, neg_inf)
            confidence = torch.where(known_mask, neg_inf, confidence)

            transfer_index = torch.zeros_like(x, dtype=torch.bool, device=self.device)
            new_mask_index = torch.zeros_like(x, dtype=torch.bool, device=self.device)

            for b in range(batch_size):
                k = int(num_transfer_tokens[b, step].item())

                if (
                    k == 0
                    and step < steps - 1 - number_global_remask
                    and number_global_remask > 0
                    and step % number_global_remask == 0
                ):
                    unknown_indices = (~known_mask[b]).nonzero(as_tuple=True)[0]

                    if len(unknown_indices) >= number_global_remask:
                        if global_remasking == "random":
                            rnd = torch.randperm(len(unknown_indices), device=self.device)[:number_global_remask]
                            selected_indices = unknown_indices[rnd]
                        elif global_remasking == "low_confidence":
                            unknown_confidence = global_conf[b][unknown_indices]
                            _, local_indices = unknown_confidence.topk(number_global_remask, largest=False)
                            selected_indices = unknown_indices[local_indices]
                        else:
                            raise ValueError(f"Unknown global remasking strategy: {global_remasking}")
                        new_mask_index[b, selected_indices] = True

                k = max(k, 1)
                _, select_index = torch.topk(confidence[b], k=k)
                transfer_index[b, select_index] = True

            x[transfer_index] = x0[transfer_index]
            total_loglikelihood += (chosen_logp * transfer_index).sum(dim=1)
            global_conf[transfer_index] = confidence[transfer_index].float()
            x[new_mask_index] = self.mask_token_id

        return x, total_loglikelihood
