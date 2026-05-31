# IHO

Reference implementation for Indirect Harm Optimization (IHO).

This repository provides:

- `iho.inference`: sample attacks from released or local IHO checkpoints.
- `iho.train`: run multi-cycle IHO training.
- `slurm/`: scripts for reproducing the experiment runs.

## Install

```bash
pixi install
```

## Sampling

Sample attacks from an IHO checkpoint:

```bash
pixi run python -m iho.inference --model CAT/local --id-index 0
```

## Training

Run multi-cycle IHO training:

```bash
pixi run python -m iho.train --model CAT/local --n-cycles 3
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
