# IHO

Reference implementation for Indirect Harm Optimization (IHO).

This repository provides:

- `iho.inference`: sample attacks from released or local IHO checkpoints.
- `iho.inference_attacks_only`: generate attack strings from an attacker checkpoint.
- `iho.inference_full`: run the full sampling pipeline against a defender model.
- `iho.train`: run multi-cycle IHO training.
- `slurm/`: scripts for reproducing the experiment runs.

## Install

```bash
pixi install
```

## Attack-Only Sampling

Sample attacks from an IHO checkpoint:

```bash
pixi run python -m iho.inference_attacks_only \
  "Sure, here is a target response." \
  --checkpoint SEML-Lab/IHO-Llama-3-8B-Instruct \
  --device cuda \
  --num-attacks 32
```

## Full Sampling

Run the full sampling pipeline against a defender model:

```bash
pixi run python -m iho.inference_full \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --rows ALL
```

## Training

Run multi-cycle IHO training:

```bash
pixi run python -m iho.train \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --n-cycles 3
```

## Reproducing Experiments

The experiment setups are in `slurm/`. Use the SLURM scripts there to reproduce
the sampling and multi-cycle training experiments.

```bash
pixi run python -m slurm.slurm_wrapper --help
```

## Environment

- `IHO_CACHE_DIR`: optional Hugging Face cache directory.
- `IHO_CAT_MODEL_PATH`: required when using the `CAT/local` target alias.

## License

MIT
