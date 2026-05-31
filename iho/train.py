from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from peft import LoraConfig
from torch.utils.data import DataLoader, Dataset

from iho.model_wrapper import LLaDAWrapper
from iho.trainer import DPOTrainer

DEFAULT_BASE_MODEL = "GSAI-ML/LLaDA-8B-Base"


class PreferencePairsDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = []
        for idx, row in enumerate(rows):
            chosen = str(row.get("chosen", "")).strip()
            rejected = str(row.get("rejected", "")).strip()
            if not chosen or not rejected:
                raise ValueError(f"Preference row {idx} must have non-empty chosen and rejected fields.")
            self.rows.append({"chosen": chosen, "rejected": rejected})
        if not self.rows:
            raise ValueError("Preference dataset is empty.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.rows[idx]


def load_preference_pairs(path: Path | str) -> PreferencePairsDataset:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported preference format. Use JSONL, JSON, CSV, or Parquet.")

    missing = {"chosen", "rejected"}.difference(df.columns)
    if missing:
        raise ValueError(f"Preference data is missing required columns: {sorted(missing)}")
    return PreferencePairsDataset(df[["chosen", "rejected"]].to_dict("records"))


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype}") from exc


def default_lora_config(r: int = 8, alpha: int = 16, dropout: float = 0.05, target_modules: list[str] | None = None) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules or ["q_proj", "v_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def train(
    data: Path | str,
    *,
    output_dir: Path | str,
    checkpoint: str | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
    device: str = "auto",
    dtype: str | torch.dtype = "bfloat16",
    cache_dir: Path | str | None = None,
    epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    beta: float = 0.25,
    attack_size: int = 32,
    masking_mode: Literal["all", "prompt", "attack"] = "prompt",
    mask_all: bool = False,
    seed: int = 42,
    lora_config: LoraConfig | None = None,
) -> tuple[LLaDAWrapper, list[float]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = load_preference_pairs(data)
    fresh_lora = lora_config or default_lora_config()
    wrapper = LLaDAWrapper(
        model_name=base_model,
        device=resolve_device(device),
        cache_dir=str(cache_dir) if cache_dir else None,
        attacker_input_size=attack_size,
        dtype=resolve_dtype(dtype),
        lora_config=None if checkpoint else fresh_lora,
        lora_checkpoint=checkpoint,
        lora_trainable=checkpoint is not None,
    )

    trainer = DPOTrainer(wrapper, learning_rate=learning_rate, beta=beta)
    losses = trainer.train(
        DataLoader(dataset, batch_size=batch_size, shuffle=True),
        n_epochs=epochs,
        masking_mode=masking_mode,
        mask_all=mask_all,
        save_checkpoints=False,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper.save_lora(str(output_dir))
    return wrapper, losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or finetune an IHO LoRA adapter from preference pairs.")
    parser.add_argument("data", type=Path, help="JSONL/JSON/CSV/Parquet file with chosen and rejected columns.")
    parser.add_argument("--output-dir", type=Path, default=Path("iho_adapter"), help="Directory for the saved LoRA adapter.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base diffusion LM to adapt.")
    parser.add_argument("--checkpoint", "--lora-checkpoint", dest="lora_checkpoint", default=None, help="Existing LoRA adapter to finetune. Omit to train a fresh adapter.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"], help="Model dtype.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of DPO epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="DPO batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="DPO learning rate.")
    parser.add_argument("--beta", type=float, default=0.25, help="DPO beta.")
    parser.add_argument("--attack-size", type=int, default=32, help="Number of leading attack tokens to train.")
    parser.add_argument("--masking-mode", default="prompt", choices=["all", "prompt", "attack"], help="Tokens to mask during DPO.")
    parser.add_argument("--mask-all", action="store_true", help="Mask all eligible tokens instead of a random subset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank for fresh adapters.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha for fresh adapters.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout for fresh adapters.")
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"], help="LoRA target modules for fresh adapters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, losses = train(
        args.data,
        output_dir=args.output_dir,
        checkpoint=args.lora_checkpoint,
        base_model=args.base_model,
        device=args.device,
        dtype=args.dtype,
        cache_dir=args.cache_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        attack_size=args.attack_size,
        masking_mode=args.masking_mode,
        mask_all=args.mask_all,
        seed=args.seed,
        lora_config=default_lora_config(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.target_modules,
        ),
    )
    print(json.dumps({"output_dir": str(args.output_dir), "checkpoint": args.lora_checkpoint, "epoch_losses": losses}, indent=2))


if __name__ == "__main__":
    main()
