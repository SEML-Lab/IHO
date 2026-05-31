from __future__ import annotations

import os
import importlib
import sys
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
        r=4,
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
                "dpo": [0],
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
            "dpo_epochs": 5,
            "preference_masking_mode": "prompt",
            "dpo_mask_all": True,
            "checkpoint_every": 3,
            "save_checkpoints": True,
            "percent_chosen": 0.25,
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
                "training": 32,
                "dpo": 4,
            },
            "num_cycles": 1,
            "gpu_type": "a100",
            "embed_attack_prompts": False,
            "use_detector": False,
        },
        "batch_sizes": {
            "attack": 32,
            "generate": 32,
            "detector": 32,
            "judge": 32,
            "dpo": 4,
        },
    }
    config = deep_merge(load_defaults("custom"), override)
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


def test_sampling_preset_accepts_named_int_and_list_rows() -> None:
    from iho.configs.slurm_presets import full_sampling_new_args

    assert full_sampling_new_args(model="CAT/local", rows="ALL", id_index=0, cache_dir=None).training_rows == "ALL"
    assert full_sampling_new_args(model="CAT/local", rows="17", id_index=0, cache_dir=None).training_rows == [17]
    assert full_sampling_new_args(model="CAT/local", rows="[0, 3, 9]", id_index=0, cache_dir=None).training_rows == [0, 3, 9]


def test_train_cli_runs_wrapper_without_launching_models(monkeypatch: pytest.MonkeyPatch) -> None:
    train_cli = importlib.import_module("iho.train")
    calls: list[tuple[str, object]] = []

    class FakeRunner:
        def __init__(self, *, config_override, gpu_type, overwrite_output, enable_monitoring) -> None:
            calls.append(
                (
                    "init",
                    {
                        "config_override": config_override,
                        "gpu_type": gpu_type,
                        "overwrite_output": overwrite_output,
                        "enable_monitoring": enable_monitoring,
                    },
                )
            )

        def train_multi_cycle(self, experiment_root_dir, sub_experiment_name, n_cycles, iterate_rows_separately) -> None:
            calls.append(
                (
                    "train",
                    {
                        "experiment_root_dir": experiment_root_dir,
                        "sub_experiment_name": sub_experiment_name,
                        "n_cycles": n_cycles,
                        "iterate_rows_separately": iterate_rows_separately,
                    },
                )
            )

    monkeypatch.setattr(train_cli, "IHOExperimentRunner", FakeRunner)
    monkeypatch.setattr(
        sys,
        "argv",
        ["python -m iho.train", "--model", "CAT/local", "--n-cycles", "2", "--disable-monitoring"],
    )

    train_cli.main()

    init = calls[0][1]
    config = init["config_override"]
    assert init["gpu_type"] == "h200"
    assert init["enable_monitoring"] is False
    assert config["attack_dataset"]["behaviour_subsets"]["training"] == "THE_TRAINING_ONES_BIG_STRATIFIED"
    assert config["attack_dataset"]["behaviour_subsets"]["validation"] == "THE_VALIDATION_ONES_STRATIFIED"
    assert config["attack_dataset"]["behaviour_subsets"]["dpo"] == "THE_TRAINING_ONES_BIG_STRATIFIED"
    assert config["attacked_model"]["model_ids"] == ["CAT/local"]
    assert config["dpo_training"]["learning_rate"] == 0.0002
    assert config["general"]["num_sampled_attacks"]["training"] == 15360
    assert config["general"]["num_cycles"] == 2
    assert calls[1] == (
        "train",
        {
            "experiment_root_dir": "evaluation/multi_target_multi_cycle_ablations",
            "sub_experiment_name": "baseline_fixed",
            "n_cycles": 2,
            "iterate_rows_separately": False,
        },
    )


def test_experiment_runner_uses_model_and_row_run_paths() -> None:
    from iho.configs.slurm_presets import baseline_fixed_args
    from slurm.slurm_wrapper import IHOExperimentRunner, build_user_override

    runner = IHOExperimentRunner(
        config_override=build_user_override(
            baseline_fixed_args(model="CAT/local", n_cycles=2, cache_dir=None)
        ),
        gpu_type="h200",
        overwrite_output=False,
        enable_monitoring=False,
    )

    assert (
        runner._generate_run_name("baseline_fixed", "CAT/local")
        == "baseline_fixed/CAT_local/THE_TRAINING_ONES_BIG_STRATIFIED"
    )
    assert (
        runner._generate_run_name("baseline_fixed", "CAT/local", row_idx=0)
        == "baseline_fixed/CAT_local/row_0"
    )


def test_inference_full_cli_runs_sampling_wrapper_without_launching_models(monkeypatch: pytest.MonkeyPatch) -> None:
    inference_cli = importlib.import_module("iho.inference_full")
    calls: list[tuple[str, object]] = []

    class FakeRunner:
        def __init__(self, *, config_override, gpu_type, overwrite_output, enable_monitoring) -> None:
            calls.append(
                (
                    "init",
                    {
                        "config_override": config_override,
                        "gpu_type": gpu_type,
                        "overwrite_output": overwrite_output,
                        "enable_monitoring": enable_monitoring,
                    },
                )
            )

        def sample(self, experiment_root_dir, sub_experiment_name, iterate_rows_separately) -> None:
            calls.append(
                (
                    "sample",
                    {
                        "experiment_root_dir": experiment_root_dir,
                        "sub_experiment_name": sub_experiment_name,
                        "iterate_rows_separately": iterate_rows_separately,
                    },
                )
            )

    monkeypatch.setattr(inference_cli, "IHOExperimentRunner", FakeRunner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python -m iho.inference_full",
            "--model",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "--rows",
            "[0, 3, 9]",
            "--id-index",
            "5",
        ],
    )

    inference_cli.main()

    init = calls[0][1]
    config = init["config_override"]
    assert init["gpu_type"] == "a100"
    assert init["overwrite_output"] is True
    assert config["attack_dataset"]["behaviour_subsets"]["training"] == [0, 3, 9]
    assert config["attacked_model"]["model_ids"] == ["meta-llama/Meta-Llama-3-8B-Instruct"]
    assert config["judge_models_config"]["validation_2"] == "llama_guard_3"
    assert config["general"]["num_sampled_attacks"]["training"] == 1280
    assert config["general"]["seed"] == 5
    assert calls[1] == (
        "sample",
        {
            "experiment_root_dir": "evaluation/sampling/sampling_baselines",
            "sub_experiment_name": "partial_sampling_new_id/5",
            "iterate_rows_separately": False,
        },
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason=CUDA_REASON)
def test_cuda_jbb_inference_pipeline_with_llama_and_judge(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        "inference-from-predefined-checkpoint",
        _tiny_pipeline_config(checkpoint=DEFAULT_CHECKPOINT),
    )

    samples = pipeline.sample(mode="training", save_df=True, cycle_id=0)

    assert len(samples) == 32
    row = samples.iloc[0]
    assert int(row["jb_index"]) == 0
    assert str(row["goal_text"]).strip()
    assert str(row["target_text"]).strip()
    assert str(row["attacking_prompt_text"]).strip()
    assert str(row["attacked_output"]).strip()
    assert torch.isfinite(torch.tensor([float(row["judge_score_training"])] )).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason=CUDA_REASON)
def test_cuda_attack_only_inference_from_predefined_checkpoint() -> None:
    model = iho.load_model(
        checkpoint=DEFAULT_CHECKPOINT,
        device="cuda",
        cache_dir=_cache_dir() or None,
        attack_size=8,
    )

    attacks = iho.generate(
        model,
        "Sure, here is how to evade legal persecution",
        num_attacks=8,
        batch_size=8,
        attack_size=32,
        attack_steps=32,
    )

    assert len(attacks) == 1
    assert isinstance(attacks[0], str)
    assert attacks[0].strip()


@pytest.mark.skipif(not torch.cuda.is_available(), reason=CUDA_REASON)
def test_cuda_trains_one_cycle_from_scratch_on_jbb(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        "train-from-scratch",
        _tiny_pipeline_config(checkpoint=None, scratch=True),
    )

    pipeline.execute_multiple_cycles(n_cycles=2)

    assert (tmp_path / "train-from-scratch" / "samples" / "cycle_0.parquet").exists()
    assert (tmp_path / "train-from-scratch" / "samples" / "cycle_1.parquet").exists()
    assert (tmp_path / "train-from-scratch" / "dpo_sets" / "cycle_0.parquet").exists()
    assert (tmp_path / "train-from-scratch" / "dpo_sets" / "cycle_1.parquet").exists()
    assert (tmp_path / "train-from-scratch" / "checkpoints" / "best_cycle_0" / "adapter_config.json").exists()
    assert (tmp_path / "train-from-scratch" / "checkpoints" / "best_cycle_1" / "adapter_config.json").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason=CUDA_REASON)
def test_cuda_finetunes_one_cycle_from_predefined_checkpoint_on_jbb(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        "finetune-from-predefined-checkpoint",
        _tiny_pipeline_config(checkpoint=DEFAULT_CHECKPOINT),
    )

    pipeline.execute_multiple_cycles(n_cycles=2)

    assert (
        tmp_path
        / "finetune-from-predefined-checkpoint"
        / "checkpoints"
        / "best_cycle_0"
        / "adapter_config.json"
    ).exists()
    assert (
        tmp_path
        / "finetune-from-predefined-checkpoint"
        / "checkpoints"
        / "best_cycle_1"
        / "adapter_config.json"
    ).exists()
