# SLURM launch helpers

This directory contains optional SLURM helpers for running IHO experiments. The
wrapper translates command-line flags into an `IHOPipeline` config and launches
either sampling or multi-cycle DPO training.

Run from the repository root with pixi, for example:

```bash
pixi run python -m slurm.slurm_wrapper \
  --operation train \
  --experiment-root evaluation/example \
  --sub-experiment-name smoke \
  --gpu-type a100 \
  --rows THE_TRAINING_ONES_SMALL \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-training-samples 128 \
  --num-dpo-samples 32 \
  --n-cycles 1
```

The `slurm_scripts/` templates cover sampling ablations, single-target
ablations, multi-target ablations, cross-model sampling, and sanity runs. They
assume the repository root is the working directory and use
`pixi run python -m slurm.slurm_wrapper`.

Environment variables:

- `IHO_CACHE_DIR`: optional Hugging Face cache directory.
- `IHO_CAT_MODEL_PATH`: required only when using the `CAT/local` model alias.
- `IHO_CHECKPOINT_ROOT`: optional base directory for checkpoint paths in
  cross-model sampling templates.
- `IHO_PROJECT_ROOT`: optional repository root for templates that need to inspect
  previous outputs before launching follow-up jobs.
- `IHO_EVALUATION_ROOT`: optional base directory replacing old absolute
  evaluation roots in migrated templates.

Cluster names, partitions, wall times, and array limits are site-specific; edit
the `#SBATCH` headers before submitting on a different cluster.
