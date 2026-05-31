from __future__ import annotations

import argparse
import logging
import os
from typing import cast

from iho.configs.config_helper import ALL_MODELS, GPUType
from iho.configs.slurm_presets import full_sampling_new_args
from slurm.slurm_wrapper import IHOExperimentRunner, build_user_override

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full_sampling_new sampling setup.")
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument("--rows", default="ALL")
    parser.add_argument("--id-index", type=int, default=0)
    parser.add_argument("--experiment-root", default="evaluation/sampling/sampling_baselines")
    parser.add_argument("--sub-experiment-prefix", default="partial_sampling_new_id")
    parser.add_argument("--gpu-type", default="a100", choices=["a100", "a100_small", "h100", "h200", "custom"])
    parser.add_argument("--cache-dir", default=os.environ.get("IHO_CACHE_DIR"))
    parser.add_argument("--disable-monitoring", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = IHOExperimentRunner(
        config_override=build_user_override(
            full_sampling_new_args(
                model=args.model,
                rows=args.rows,
                id_index=args.id_index,
                cache_dir=args.cache_dir,
            )
        ),
        gpu_type=cast(GPUType, args.gpu_type),
        overwrite_output=not args.no_overwrite,
        enable_monitoring=not args.disable_monitoring,
    )
    runner.sample(
        args.experiment_root,
        f"{args.sub_experiment_prefix}/{args.id_index}",
        iterate_rows_separately=False,
    )
    logger.info("Sampling operation completed successfully")


if __name__ == "__main__":
    main()
