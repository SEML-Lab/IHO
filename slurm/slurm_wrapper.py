from __future__ import annotations

import argparse
import ast
import copy
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, cast

import pandas as pd
from peft import LoraConfig
from transformers import logging as hf_logging

from iho.configs.config_helper import (
    ALL,
    ALL_MODELS,
    GPUType,
    JudgeModelsConfigOverride,
    NamedBehaviourSubsets,
    PipelineConfig,
    PipelineConfigOverride,
    SamplingMode,
    THE_TEST_HELD_OUT_STRATIFIED,
    THE_TRAINING_ONES_BIG,
    THE_TRAINING_ONES_BIG_STRATIFIED,
    THE_TRAINING_ONES_SMALL,
    THE_TRAINING_ONES_SMALL_STRATIFIED,
    THE_VALIDATION_ONES,
    THE_VALIDATION_ONES_STRATIFIED,
    BehaviourSubsetsConfigOverride,
    deep_merge,
    load_defaults,
)
from iho.datasets.JailbreakBenchDataset import rows_to_indices
from iho.pipeline import IHOPipeline
from iho.utils.MemoryMonitor import MemoryMonitor
from iho.utils.general_utils import attach_file_logger

hf_logging.set_verbosity_warning()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

NAMED_ROW_SETS = {
    "ALL",
    "THE_TRAINING_ONES_SMALL",
    "THE_TRAINING_ONES_BIG",
    "THE_VALIDATION_ONES",
    "THE_TRAINING_ONES_SMALL_STRATIFIED",
    "THE_TRAINING_ONES_BIG_STRATIFIED",
    "THE_VALIDATION_ONES_STRATIFIED",
    "THE_TEST_HELD_OUT_STRATIFIED",
}


def parse_rows(value: str | None) -> Union[NamedBehaviourSubsets, List[int], None]:
    if value is None:
        return None

    value = value.strip()
    if value in NAMED_ROW_SETS:
        return cast(NamedBehaviourSubsets, value)
    if value.isdigit():
        return [int(value)]

    try:
        parsed = ast.literal_eval(value)
    except Exception as exc:
        raise ValueError(f"Invalid row specification: {value}") from exc

    if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
        return parsed
    raise ValueError("Rows must be a named row set, an integer, or a Python list of integers.")


def get_models(model_arg: str, use_detector: bool) -> List[str]:
    models = ALL_MODELS if model_arg.lower() == "all" else [model_arg]
    if not use_detector:
        return models

    detector_models = []
    for model in models:
        detector_models.append(model if model.endswith("-with-detector") else f"{model}-with-detector")
    return detector_models


def build_user_override(args: argparse.Namespace) -> PipelineConfigOverride:
    override: PipelineConfigOverride = {}

    behaviour_subsets: BehaviourSubsetsConfigOverride = {}
    if args.training_rows is not None:
        behaviour_subsets["training"] = args.training_rows
    if args.validation_rows is not None:
        behaviour_subsets["validation"] = args.validation_rows
    if args.testing_rows is not None:
        behaviour_subsets["testing"] = args.testing_rows
    if args.dpo_rows is not None:
        behaviour_subsets["dpo"] = args.dpo_rows
    if behaviour_subsets:
        override.setdefault("attack_dataset", {})["behaviour_subsets"] = behaviour_subsets

    if args.attacker_input_size is not None:
        override.setdefault("attack_model", {})["attacker_input_size"] = args.attacker_input_size
    if args.attacker_step_number is not None:
        override.setdefault("attack_model", {})["attacker_step_number"] = args.attacker_step_number
    if args.attacker_temperature is not None:
        override.setdefault("attack_model", {})["attacker_temperature"] = args.attacker_temperature
    if args.mask_padding_tokens:
        override.setdefault("attack_model", {})["mask_padding_tokens"] = True
    if args.no_mask_padding_tokens:
        override.setdefault("attack_model", {})["mask_padding_tokens"] = False
    if args.lora_checkpoint_path is not None:
        override.setdefault("attack_model", {})["lora_checkpoint"] = args.lora_checkpoint_path
        override.setdefault("attack_model", {})["lora_config"] = None

    lora_args = [args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules, args.lora_use_rslora]
    if any(x is not None for x in lora_args):
        if args.lora_checkpoint_path is not None:
            raise ValueError("Cannot specify both --lora-checkpoint-path and LoRA hyperparameters.")
        if not all(x is not None for x in lora_args):
            raise ValueError("LoRA hyperparameters require --lora-r, --lora-alpha, --lora-dropout, --lora-target-modules, and --lora-use-rslora.")
        override.setdefault("attack_model", {})["lora_config"] = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[m.strip() for m in args.lora_target_modules.split(",")],
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=args.lora_use_rslora,
        )
        override.setdefault("attack_model", {})["lora_checkpoint"] = None

    if args.model is not None:
        override.setdefault("attacked_model", {})["model_ids"] = get_models(args.model, args.use_detector)
    if args.max_new_tokens is not None:
        override.setdefault("attacked_model", {})["max_new_tokens"] = args.max_new_tokens
    if args.greedy:
        override.setdefault("attacked_model", {})["greedy_sampling"] = True
    if args.no_logprobs:
        override.setdefault("attacked_model", {})["compute_logprobs"] = False

    judge_overrides: JudgeModelsConfigOverride = {}
    if args.judge_training is not None:
        judge_overrides["training"] = args.judge_training
    if args.judge_validation is not None:
        judge_overrides["validation"] = args.judge_validation
    if args.judge_validation_2 is not None:
        judge_overrides["validation_2"] = args.judge_validation_2
    if judge_overrides:
        override["judge_models_config"] = judge_overrides

    if args.learning_rate is not None:
        override.setdefault("dpo_training", {})["learning_rate"] = args.learning_rate
    if args.beta is not None:
        override.setdefault("dpo_training", {})["beta"] = args.beta
    if args.dpo_epochs is not None:
        override.setdefault("dpo_training", {})["dpo_epochs"] = args.dpo_epochs
    if args.checkpoint_every is not None:
        override.setdefault("dpo_training", {})["checkpoint_every"] = args.checkpoint_every
    if args.save_checkpoints is not None:
        override.setdefault("dpo_training", {})["save_checkpoints"] = args.save_checkpoints
    if args.no_checkpoints:
        override.setdefault("dpo_training", {})["save_checkpoints"] = False
    if args.percent_chosen is not None:
        override.setdefault("dpo_training", {})["percent_chosen"] = args.percent_chosen
    if args.harmfulness_threshold is not None:
        override.setdefault("dpo_training", {})["harmfulness_threshold"] = args.harmfulness_threshold
    if args.expanding_dpo_dataset is not None:
        override.setdefault("dpo_training", {})["expanding_dpo_dataset"] = args.expanding_dpo_dataset
    if args.patience is not None:
        override.setdefault("dpo_training", {})["patience"] = args.patience
    if args.warmup_epochs is not None:
        override.setdefault("dpo_training", {})["warmup_epochs"] = args.warmup_epochs
    if args.load_optimizer_state:
        override.setdefault("dpo_training", {})["load_optimizer_state"] = True
    if args.dpo_metric is not None:
        override.setdefault("dpo_training", {})["dpo_metric"] = args.dpo_metric

    num_sampled: Dict[SamplingMode, int] = {}
    if args.num_training_samples is not None:
        num_sampled["training"] = args.num_training_samples
    if args.num_validation_samples is not None:
        num_sampled["validation"] = args.num_validation_samples
    if args.num_test_samples is not None:
        num_sampled["testing"] = args.num_test_samples
    if args.num_dpo_samples is not None:
        num_sampled["dpo"] = args.num_dpo_samples
    if num_sampled:
        override.setdefault("general", {}).setdefault("num_sampled_attacks", {}).update(num_sampled)

    if args.n_cycles is not None:
        override.setdefault("general", {})["num_cycles"] = args.n_cycles
    if args.embed_attack_prompts is not None:
        override.setdefault("general", {})["embed_attack_prompts"] = args.embed_attack_prompts
    if args.seed is not None:
        override.setdefault("general", {})["seed"] = args.seed
    if args.cache_dir is not None:
        override.setdefault("general", {})["custom_cache_path"] = args.cache_dir
    if args.use_detector:
        override.setdefault("general", {})["use_detector"] = True

    return override


class IHOExperimentRunner:
    def __init__(
        self,
        *,
        config_override: Optional[PipelineConfigOverride],
        gpu_type: GPUType,
        overwrite_output: bool,
        enable_monitoring: bool = True,
        monitor_poll_interval: float = 0.1,
        verbose_monitoring: bool = False,
    ) -> None:
        self.overwrite_output = overwrite_output
        self.pipeline_config = deep_merge(load_defaults(gpu_type), config_override or {})
        self.monitor = (
            MemoryMonitor(poll_interval=monitor_poll_interval, verbose=verbose_monitoring)
            if enable_monitoring
            else None
        )

    def _generate_run_name(self, sub_experiment_name: str, attacked_model_id: str, row_idx: Optional[int] = None) -> str:
        model_string = attacked_model_id.replace("/", "_")
        if row_idx is not None:
            row_string = f"row_{row_idx}"
        else:
            training_rows = sorted(rows_to_indices(self.pipeline_config["attack_dataset"]["behaviour_subsets"]["training"]))
            known_sets = [
                (THE_TRAINING_ONES_SMALL, "THE_TRAINING_ONES_SMALL"),
                (THE_TRAINING_ONES_BIG, "THE_TRAINING_ONES_BIG"),
                (THE_VALIDATION_ONES, "THE_VALIDATION_ONES"),
                (THE_TRAINING_ONES_SMALL_STRATIFIED, "THE_TRAINING_ONES_SMALL_STRATIFIED"),
                (THE_TRAINING_ONES_BIG_STRATIFIED, "THE_TRAINING_ONES_BIG_STRATIFIED"),
                (THE_VALIDATION_ONES_STRATIFIED, "THE_VALIDATION_ONES_STRATIFIED"),
                (THE_TEST_HELD_OUT_STRATIFIED, "THE_TEST_HELD_OUT_STRATIFIED"),
                (ALL, "ALL"),
            ]
            row_string = next((name for rows, name in known_sets if training_rows == sorted(rows)), None)
            if row_string is None:
                row_string = f"row_{training_rows[0]}" if len(training_rows) == 1 else "non_standard_rows"
        return f"{sub_experiment_name}/{model_string}/{row_string}"

    def _create_pipeline(self, experiment_root_dir: str, run_name: str, model_id: str) -> tuple[IHOPipeline, int]:
        run_config = copy.deepcopy(self.pipeline_config)
        run_config["attacked_model"]["model_ids"] = [model_id]
        pipeline = IHOPipeline(
            experiment_path=experiment_root_dir,
            run_name=run_name,
            configs=run_config,
            overwrite_existing=self.overwrite_output,
        )
        execution_num = attach_file_logger(Path(pipeline.run_path) / "logs")
        logger.info("Logger attached to logs/execution_%02d.log", execution_num)
        return pipeline, execution_num

    def _with_monitor(self, pipeline: IHOPipeline, experiment_root_dir: str, execution_num: int):
        if self.monitor is None:
            return None
        meta_dir = os.path.join(pipeline.run_path, "meta")
        os.makedirs(meta_dir, exist_ok=True)
        return self.monitor.monitor(
            operation_name=os.path.basename(experiment_root_dir.rstrip(os.sep)) or "iho_experiment",
            log_summary=True,
            save_path=os.path.join(meta_dir, "monitoring.json"),
            execution_num=execution_num,
        )

    def sample(self, experiment_root_dir: str, sub_experiment_name: str, iterate_rows_separately: bool) -> None:
        for model_id in self.pipeline_config["attacked_model"]["model_ids"]:
            row_indices = rows_to_indices(self.pipeline_config["attack_dataset"]["behaviour_subsets"]["training"])
            rows_to_run: List[Optional[int]] = row_indices if iterate_rows_separately else [None]
            original_config = self.pipeline_config

            for row_idx in rows_to_run:
                if row_idx is not None:
                    self.pipeline_config = copy.deepcopy(original_config)
                    self.pipeline_config["attack_dataset"]["behaviour_subsets"]["training"] = [row_idx]

                run_name = self._generate_run_name(sub_experiment_name, model_id, row_idx=row_idx)
                pipeline, execution_num = self._create_pipeline(experiment_root_dir, run_name, model_id)
                monitor = self._with_monitor(pipeline, experiment_root_dir, execution_num)

                def run_sampling() -> int:
                    total = len(pipeline.sample(mode="training", save_df=True))
                    if "validation" in pipeline.configs["attack_dataset"]["behaviour_subsets"]:
                        total += len(pipeline.sample(mode="validation", save_df=True))
                    if "testing" in pipeline.configs["attack_dataset"]["behaviour_subsets"]:
                        total += len(pipeline.sample(mode="testing", save_df=True))
                    return total

                if monitor is None:
                    total_samples = run_sampling()
                else:
                    with monitor:
                        total_samples = run_sampling()
                        self.monitor.set_num_training_samples(total_samples)
                logger.info("Sampling complete for %s | samples=%d", run_name, total_samples)

            self.pipeline_config = original_config

    def train_multi_cycle(self, experiment_root_dir: str, sub_experiment_name: str, n_cycles: int, iterate_rows_separately: bool) -> None:
        for model_id in self.pipeline_config["attacked_model"]["model_ids"]:
            row_indices = rows_to_indices(self.pipeline_config["attack_dataset"]["behaviour_subsets"]["training"])
            rows_to_run: List[Optional[int]] = row_indices if iterate_rows_separately else [None]
            original_config = self.pipeline_config

            for row_idx in rows_to_run:
                if row_idx is not None:
                    self.pipeline_config = copy.deepcopy(original_config)
                    self.pipeline_config["attack_dataset"]["behaviour_subsets"]["training"] = [row_idx]

                run_name = self._generate_run_name(sub_experiment_name, model_id, row_idx=row_idx)
                pipeline, execution_num = self._create_pipeline(experiment_root_dir, run_name, model_id)
                monitor = self._with_monitor(pipeline, experiment_root_dir, execution_num)
                if monitor is None:
                    pipeline.execute_multiple_cycles(n_cycles=n_cycles)
                else:
                    with monitor:
                        pipeline.execute_multiple_cycles(n_cycles=n_cycles)
                        self.monitor.set_num_training_samples(0)
                logger.info("Training complete for %s", run_name)

            self.pipeline_config = original_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IHO experiments via SLURM-friendly CLI")
    parser.add_argument("--operation", required=True, choices=["sample", "train"])
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--sub-experiment-name", required=True)
    parser.add_argument("--iterate-rows-separately", action="store_true")
    parser.add_argument("--gpu-type", default="a100", choices=["a100", "a100_small", "h100", "h200", "custom"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache-dir", default=os.environ.get("IHO_CACHE_DIR"))
    parser.add_argument("--disable-monitoring", action="store_true")

    parser.add_argument("--rows")
    parser.add_argument("--training-rows")
    parser.add_argument("--validation-rows")
    parser.add_argument("--testing-rows")
    parser.add_argument("--dpo-rows")

    parser.add_argument("--attacker-input-size", type=int)
    parser.add_argument("--attacker-step-number", type=int)
    parser.add_argument("--attacker-temperature", type=float)
    parser.add_argument("--mask-padding-tokens", action="store_true")
    parser.add_argument("--no-mask-padding-tokens", action="store_true")
    parser.add_argument("--no-logprobs", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--lora-checkpoint-path")
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--lora-target-modules")
    parser.add_argument("--lora-use-rslora", type=lambda x: x.lower() == "true")

    parser.add_argument("--model", default="all", choices=ALL_MODELS + ["all"])
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--use-detector", action="store_true")

    judge_choices = ["strong_reject", "harmbench", "llama_guard_3", "llama_guard_4", "jail_judge", "aegis_guard"]
    parser.add_argument("--judge-training", choices=judge_choices)
    parser.add_argument("--judge-validation", choices=judge_choices)
    parser.add_argument("--judge-validation-2", choices=judge_choices)

    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--dpo-epochs", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--save-checkpoints", type=lambda x: x.lower() == "true")
    parser.add_argument("--percent-chosen", type=float)
    parser.add_argument("--harmfulness-threshold", type=float)
    parser.add_argument("--expanding-dpo-dataset", type=lambda x: x.lower() == "true")
    parser.add_argument("--patience", type=int)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--load-optimizer-state", action="store_true")
    parser.add_argument("--dpo-metric", choices=["mean", "max", "weighted"])

    parser.add_argument("--num-training-samples", type=int)
    parser.add_argument("--num-validation-samples", type=int)
    parser.add_argument("--num-test-samples", type=int)
    parser.add_argument("--num-dpo-samples", type=int)
    parser.add_argument("--n-cycles", type=int)
    parser.add_argument("--embed-attack-prompts", type=lambda x: x.lower() == "true")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.operation == "train" and args.n_cycles is None:
        logger.error("--n-cycles is required when --operation train")
        sys.exit(1)

    has_global_rows = args.rows is not None
    has_specific_rows = any(x is not None for x in [args.training_rows, args.validation_rows, args.testing_rows, args.dpo_rows])
    if has_global_rows and has_specific_rows:
        logger.error("Use either --rows or split-specific row options, not both.")
        sys.exit(1)

    if has_global_rows:
        parsed = parse_rows(args.rows)
        args.training_rows = parsed
        args.validation_rows = parsed
        args.testing_rows = parsed
        args.dpo_rows = parsed
    else:
        args.training_rows = parse_rows(args.training_rows)
        args.validation_rows = parse_rows(args.validation_rows)
        args.testing_rows = parse_rows(args.testing_rows)
        args.dpo_rows = parse_rows(args.dpo_rows)

    runner = IHOExperimentRunner(
        config_override=build_user_override(args),
        gpu_type=cast(GPUType, args.gpu_type),
        overwrite_output=args.overwrite,
        enable_monitoring=not args.disable_monitoring,
    )

    if args.operation == "sample":
        runner.sample(args.experiment_root, args.sub_experiment_name, args.iterate_rows_separately)
    else:
        runner.train_multi_cycle(args.experiment_root, args.sub_experiment_name, args.n_cycles, args.iterate_rows_separately)
    logger.info("Operation completed successfully")


if __name__ == "__main__":
    main()
