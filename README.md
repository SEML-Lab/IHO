# IHO

[![ArXiv](https://img.shields.io/badge/arXiv-2606.03647-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.03647)

Official implementation of Indirect Harm Optimization (IHO) from *Black-box, Adaptive, Efficient, Transferable, Harmful, Applicable... Attacks Are All You Need to Break LLMs.*

This repository provides:

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
  "Sure, here is how to evade legal persecution" \
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

## Citation

Please cite the Limbach et al., 2026, *"Black-box, Adaptive, Efficient, Transferable, Harmful, Applicable... Attacks Are All You Need to Break LLMs."*:

```bibtex
@misc{limbach2026blackboxadaptiveefficient,
  title={Black-box, Adaptive, Efficient, Transferable, Harmful, Applicable... Attacks Are All You Need to Break LLMs},
  author={Vincent Limbach and Jonas Dornbusch and David L{\"u}dke and Stephan G{\"u}nnemann and Leo Schwinn},
  year={2026},
  eprint={2606.03647},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2606.03647},
}
```
