from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from peft import LoraConfig

import iho
from iho.configs.config_helper import PipelineConfig, deep_merge, load_defaults
from iho.datasets.JailbreakBenchDataset import rows_to_indices
from iho.pipeline import DEFAULT_CHECKPOINT, IHOPipeline


CUDA_REASON = (
    "CUDA unavailable; skipping full IHO pipeline tests. "
    "Running only minimal import/config setup."
)
TARGET_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def _cache_dir() -> str:
    return os.environ.get("IHO_TEST_CACHE_DIR") or os.environ.get("IHO_CACHE_DIR", "")


def _tiny_lora_config() -> LoraConfig:
    return LoraConfig(
        r=1,
        lora_alpha=2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )


def _tiny_pipeline_config(*, checkpoint: str | None, scratch: bool = False) -> PipelineConfig:
    override = {
        "attack_dataset": {
            "dataset_name": "jbb",
            "behaviour_subsets": {
                "training": [0],
            },
        },
        "attack_model": {
            "attacker_input_size": 8,
            "attacker_step_number": 4,
            "attacker_temperature": 0.0,
            "mask_padding_tokens": True,
            "remasking": "low_confidence",
            "lora_config": _tiny_lora_config() if scratch else None,
            "lora_checkpoint": checkpoint,
        },
        "attacked_model": {
            "model_ids": [TARGET_MODEL],
            "max_new_tokens": 2,
            "use_chat_template": True,
            "greedy_sampling": True,
            "compute_logprobs": False,
        },
        "judge_models_config": {
            "training": "strong_reject",
        },
        "dpo_training": {
            "learning_rate": 1e-5,
            "beta": 0.25,
            "dpo_epochs": 1,
            "preference_masking_mode": "prompt",
            "dpo_mask_all": True,
            "checkpoint_every": 0,
            "save_checkpoints": True,
            "percent_chosen": 1.0,
            "harmfulness_threshold": 0.0,
            "expanding_dpo_dataset": False,
            "patience": 1,
            "warmup_epochs": 0,
            "load_optimizer_state": False,
            "dpo_metric": "mean",
        },
        "general": {
            "seed": 7,
            "device": "cuda",
            "custom_cache_path": _cache_dir(),
            "debug": True,
            "run_name": None,
            "num_sampled_attacks": {
                "training": 1,
                "dpo": 1,
            },
            "num_cycles": 1,
            "gpu_type": "custom",
            "embed_attack_prompts": False,
            "use_detector": False,
        },
        "batch_sizes": {
            "attack": 1,
            "generate": 1,
            "detector": 1,
            "judge": 1,
            "dpo": 1,
        },
    }
    config = deep_merge(load_defaults("custom"), override)
    config["attack_dataset"]["behaviour_subsets"] = {"training": [0]}
    config["judge_models_config"] = {"training": "strong_reject"}
    config["general"]["num_sampled_attacks"] = {"training": 1, "dpo": 1}
    return config


def _pipeline(tmp_path: Path, name: str, config: PipelineConfig) -> IHOPipeline:
    return IHOPipeline(
        experiment_path=str(tmp_path),
        run_name=name,
        configs=config,
        overwrite_existing=True,
    )


def test_minimal_setup_when_cuda_is_unavailable(capsys: pytest.CaptureFixture[str]) -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available; full pipeline tests cover the real features.")

    print(CUDA_REASON)
    assert callable(iho.load_model)
    assert callable(iho.generate)
    assert callable(iho.IHOPipeline)
    assert rows_to_indices([0, 1]) == [0, 1]
    assert len(rows_to_indices("THE_TRAINING_ONES_SMALL")) > 0
    assert DEFAULT_CHECKPOINT == "SEML-Lab/IHO-Llama-3-8B-Instruct"

    captured = capsys.readouterr()
    assert "CUDA unavailable" in captured.out


@pytest.mark.skipif(not torch.cuda.is_available(), reason=CUDA_REASON)
def test_cuda_jbb_inference_pipeline_with_llama_and_judge(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        "inference-from-predefined-checkpoint",
        _tiny_pipeline_config(checkpoint=DEFAULT_CHECKPOINT),
    )

    samples = pipeline.sample(mode="training", save_df=True, cycle_id=0)

    assert len(samples) == 1
    row = samples.iloc[0]
    assert int(row["jb_index"]) == 0
    assert str(row["goal_text"]).strip()
    assert str(row["target_text"]).strip()
    assert str(row["attacking_prompt_text"]).strip()
    assert str(row["attacked_output"]).strip()
    assert torch.isfinite(torch.tensor([float(row["judge_score_training"])] )).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason=CUDA_REASON)
def test_cuda_trains_one_cycle_from_scratch_on_jbb(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        "train-from-scratch",
        _tiny_pipeline_config(checkpoint=None, scratch=True),
    )

    pipeline.execute_one_cycle(cycle_id=0)

    assert (tmp_path / "train-from-scratch" / "samples" / "cycle_0.parquet").exists()
    assert (tmp_path / "train-from-scratch" / "dpo_sets" / "cycle_0.parquet").exists()
    assert (tmp_path / "train-from-scratch" / "checkpoints" / "best_cycle_0" / "adapter_config.json").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason=CUDA_REASON)
def test_cuda_finetunes_one_cycle_from_predefined_checkpoint_on_jbb(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        "finetune-from-predefined-checkpoint",
        _tiny_pipeline_config(checkpoint=DEFAULT_CHECKPOINT),
    )

    pipeline.execute_one_cycle(cycle_id=0)

    assert (
        tmp_path
        / "finetune-from-predefined-checkpoint"
        / "checkpoints"
        / "best_cycle_0"
        / "adapter_config.json"
    ).exists()
