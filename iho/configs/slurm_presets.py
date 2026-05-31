from __future__ import annotations

import argparse

from slurm.slurm_wrapper import parse_rows


def _wrapper_namespace(**overrides: object) -> argparse.Namespace:
    values = {
        "training_rows": None,
        "validation_rows": None,
        "testing_rows": None,
        "dpo_rows": None,
        "attacker_input_size": None,
        "attacker_step_number": None,
        "attacker_temperature": None,
        "mask_padding_tokens": False,
        "no_mask_padding_tokens": False,
        "no_logprobs": False,
        "no_checkpoints": False,
        "lora_checkpoint_path": None,
        "lora_r": None,
        "lora_alpha": None,
        "lora_dropout": None,
        "lora_target_modules": None,
        "lora_use_rslora": None,
        "model": None,
        "max_new_tokens": None,
        "greedy": False,
        "use_detector": False,
        "judge_training": None,
        "judge_validation": None,
        "judge_validation_2": None,
        "learning_rate": None,
        "beta": None,
        "dpo_epochs": None,
        "checkpoint_every": None,
        "save_checkpoints": None,
        "percent_chosen": None,
        "harmfulness_threshold": None,
        "expanding_dpo_dataset": None,
        "patience": None,
        "warmup_epochs": None,
        "load_optimizer_state": False,
        "dpo_metric": None,
        "num_training_samples": None,
        "num_validation_samples": None,
        "num_test_samples": None,
        "num_dpo_samples": None,
        "n_cycles": None,
        "embed_attack_prompts": None,
        "seed": None,
        "cache_dir": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def baseline_fixed_args(*, model: str, n_cycles: int, cache_dir: str | None) -> argparse.Namespace:
    return _wrapper_namespace(
        training_rows=parse_rows("THE_TRAINING_ONES_BIG_STRATIFIED"),
        validation_rows=parse_rows("THE_VALIDATION_ONES_STRATIFIED"),
        dpo_rows=parse_rows("THE_TRAINING_ONES_BIG_STRATIFIED"),
        model=model,
        learning_rate=0.0002,
        beta=0.15,
        dpo_epochs=361,
        checkpoint_every=40,
        save_checkpoints=True,
        patience=3,
        num_training_samples=15360,
        num_validation_samples=2048,
        num_dpo_samples=1024,
        percent_chosen=0.0625,
        harmfulness_threshold=0.35,
        n_cycles=n_cycles,
        cache_dir=cache_dir,
    )


def full_sampling_new_args(
    *,
    model: str,
    rows: str,
    id_index: int,
    cache_dir: str | None,
) -> argparse.Namespace:
    return _wrapper_namespace(
        training_rows=parse_rows(rows),
        no_logprobs=True,
        model=model,
        judge_validation_2="llama_guard_3",
        num_training_samples=1280,
        seed=id_index,
        cache_dir=cache_dir,
    )
