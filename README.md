<div align="center">

# IHO: Indirect Harm Optimization

**Black-box, adaptive, efficient, and transferable jailbreak optimization for LLM safety evaluation.**

[![arXiv](https://img.shields.io/badge/arXiv-2606.03647-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.03647)
[![Hugging Face](https://img.shields.io/badge/Models-Hugging%20Face-fcd34d?logo=huggingface&logoColor=black)](https://huggingface.co/collections/SEML-Lab/iho)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Pixi](https://img.shields.io/badge/Env-pixi-5C6AC4)](https://pixi.sh/)

[Paper](https://arxiv.org/abs/2606.03647) | [Models and checkpoints](https://huggingface.co/collections/SEML-Lab/iho)

</div>

<p align="center">
  <img src="assets/iho_paper_figure.png" width="680" alt="Conceptual comparison figure from the IHO paper">
</p>

This repository contains the official implementation of **Indirect Harm Optimization (IHO)** from *Black-box, Adaptive, Efficient, Transferable, Harmful, Applicable... Attacks Are All You Need to Break LLMs*.

IHO trains a diffusion attacker model using indirect feedback from a target or defender model and a harmfulness judge. The codebase supports attack-only sampling from released checkpoints, full black-box sampling against a defender model, and multi-cycle IHO training.

## Highlights

- Attack-only sampling from released IHO attacker checkpoints.
- Full black-box sampling pipeline against defender models.
- Multi-cycle IHO training with explicit, reproducible configs.
- SLURM helpers for reproducing paper-style experiment runs.

## Quickstart

Install the Pixi environment:

```bash
pixi install
```

Sample attack strings from a released IHO checkpoint:

```bash
pixi run python -m iho.inference_attacks_only \
  "Sure, here is how to evade legal persecution" \
  --checkpoint SEML-Lab/IHO-Llama-3-8B-Instruct \
  --device cuda \
  --num-attacks 32
```

## Models / Checkpoints

Released models and checkpoints are available in the Hugging Face collection:

- `SEML-Lab/iho`: <https://huggingface.co/collections/SEML-Lab/iho>
- Default attack-only checkpoint in this repo: `SEML-Lab/IHO-Llama-3-8B-Instruct`

## Repository Overview

| Component | Description |
| --- | --- |
| `iho.inference_attacks_only` | Generate attack strings from an attacker checkpoint. |
| `iho.inference_full` | Run the full sampling pipeline against a defender model. |
| `iho.train` | Run multi-cycle IHO training. |
| `slurm/` | Helpers and templates for reproducing experiment runs. |
| `tests/` | Small regression checks for repo functionality. |

## Repository Structure

```text
.
├── iho/
│   ├── inference_attacks_only.py
│   ├── inference_full.py
│   ├── train.py
│   ├── pipeline.py
│   ├── configs/
│   ├── datasets/
│   ├── model_wrapper/
│   ├── trainer/
│   └── utils/
├── slurm/
│   ├── slurm_wrapper.py
│   └── slurm_scripts/
├── tests/
├── assets/
└── pyproject.toml
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

## Method Overview

The paper’s method diagram is shown below for quick reference. For the full derivation and design rationale, refer to the paper.

<p align="center">
  <img src="assets/iho_method_figure_original.png" width="760" alt="Original IHO method overview figure from the paper">
</p>

## Reproducing Experiments

The experiment setups live in `slurm/`. The wrapper translates command-line flags into an `IHOPipeline` config and can launch sampling or multi-cycle DPO training runs.

```bash
pixi run python -m slurm.slurm_wrapper --help
```

See [slurm/README.md](slurm/README.md) for an example launch and the cluster-specific environment variables used by the templates.

## Environment Variables

| Variable | Description |
| --- | --- |
| `IHO_CACHE_DIR` | Optional Hugging Face cache directory. |
| `IHO_CAT_MODEL_PATH` | Required when using the `CAT/local` target alias. |
| `IHO_CHECKPOINT_ROOT` | Optional base directory for checkpoint paths in cross-model SLURM templates. |
| `IHO_PROJECT_ROOT` | Optional repository root used by some SLURM follow-up templates. |
| `IHO_EVALUATION_ROOT` | Optional base directory replacing older absolute evaluation roots in templates. |

## Results

The table below reproduces the main adaptive-attack and held-out-behavior comparison from the paper. The metric is **EVUS** (Expected Value Under the Surface of $\mathrm{ASR}(n, \tau)$) under **StrongREJECT** (higher is better). The left block reports performance on the 60 training behaviors; the right block reports the same checkpoints on 40 held-out behaviors.

For metric definitions, evaluation protocol, and the corresponding ASR tables, see the paper and Appendix G.

<p align="center">
  <img src="assets/iho_table1_paper.png" width="100%" alt="Table 1 from the IHO paper showing EVUS results on training and held-out behaviors">
</p>

## Responsible Use

This repository is released for research on LLM safety evaluation, robustness, and red-teaming methodology. Use it only in controlled settings where you have authorization to evaluate the target systems.

## Citation

If you use this repository, please cite:

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

## License

MIT. See [LICENSE](LICENSE).
