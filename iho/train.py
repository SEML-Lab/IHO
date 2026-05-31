from __future__ import annotations

import argparse
import logging
import os
from typing import cast

from iho.configs.config_helper import ALL_MODELS, GPUType
from iho.configs.slurm_presets import baseline_fixed_args
from slurm.slurm_wrapper import IHOExperimentRunner, build_user_override

logger = logging.getLogger(__name__)


def _gpu_type_for_model(model: str) -> GPUType:
    if model == "Qwen/Qwen2.5-32B-Instruct":
        return "custom"
    return "h200"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the baseline_fixed full-cycle IHO training setup.")
    parser.add_argument("--n-cycles", type=int, required=True)
    parser.add_argument("--model", default="CAT/local", choices=ALL_MODELS)
    parser.add_argument("--experiment-root", default="evaluation/multi_target_multi_cycle_ablations")
    parser.add_argument("--run-name", "--sub-experiment-name", dest="run_name", default="baseline_fixed")
    parser.add_argument("--gpu-type", choices=["a100", "a100_small", "h100", "h200", "custom"])
    parser.add_argument("--cache-dir", default=os.environ.get("IHO_CACHE_DIR"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-monitoring", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu_type = cast(GPUType, args.gpu_type or _gpu_type_for_model(args.model))
    runner = IHOExperimentRunner(
        config_override=build_user_override(
            baseline_fixed_args(model=args.model, n_cycles=args.n_cycles, cache_dir=args.cache_dir)
        ),
        gpu_type=gpu_type,
        overwrite_output=args.overwrite,
        enable_monitoring=not args.disable_monitoring,
    )
    runner.train_multi_cycle(args.experiment_root, args.run_name, args.n_cycles, iterate_rows_separately=False)
    logger.info("Training operation completed successfully")


if __name__ == "__main__":
    main()
