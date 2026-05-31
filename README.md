# IHO

Reference implementation for Indirect Harm Optimization (IHO), a masked
diffusion-language-model attacker trained with preference optimization.

This repository provides the public IHO package used to generate attacks,
finetune LoRA attacker adapters, and run the core iterative IHO pipeline. It
does not bundle model weights or generated experiment outputs.

## Installation

Install dependencies with [pixi](https://pixi.sh):

```bash
pixi install
```


Alternatively, start an interactive shell inside the environment:

```bash
pixi shell
```

## Command Line

Run the default checkpoint with pixi:

```bash
pixi run python -m iho.inference "Sure, here is a placeholder target response." \
  --device cuda \
  --num-attacks 32 \
  --batch-size 32 \
  --attack-size 32
```

## Python API

```python
import iho

model = iho.load_model(device="cuda")

target_response = "Sure, here is a harmless placeholder response."
attacks = iho.generate(model, target_response, num_attacks=32, batch_size=32)
```

The wrapper is also importable directly:

```python
from iho.model_wrapper.LLaDAWrapper import LLaDAWrapper

wrapper = LLaDAWrapper(
    model_name="GSAI-ML/LLaDA-8B-Base",
    lora_checkpoint="SEML-Lab/IHO-Llama-3-8B-Instruct",
    device="cuda",  # or "cpu"
)
```


## Training And Finetuning

IHO supports three training modes:

* train a fresh LoRA adapter from scratch
* train from a predefined IHO checkpoint on Hugging Face
* continue training from a local adapter checkpoint you saved earlier

The training data must contain `chosen` and `rejected` columns, where each row is a preference pair for DPO. Supported input formats are JSONL, JSON, CSV, and Parquet.

Example JSONL row:

```json
{"chosen": "<preferred full prompt or completion>", "rejected": "<dispreferred full prompt or completion>"}
```

Train a fresh adapter:

```bash
pixi run python -m iho.train preferences.jsonl \
  --output-dir checkpoints/my-iho-adapter \
  --device cuda \
  --epochs 3 \
  --batch-size 4 \
  --attack-size 32
```

Train from a predefined checkpoint:

```bash
pixi run python -m iho.train preferences.jsonl \
  --checkpoint SEML-Lab/IHO-Llama-3-8B-Instruct \
  --output-dir checkpoints/my-finetuned-iho-adapter \
  --device cuda
```

Train from a local checkpoint:

```bash
pixi run python -m iho.train preferences.jsonl \
  --checkpoint checkpoints/previous-adapter \
  --output-dir checkpoints/resumed-adapter \
  --device cuda
```


Python API:

```python
import iho

wrapper, losses = iho.train(
    "preferences.parquet",
    output_dir="checkpoints/my-iho-adapter",
    device="cuda",
    epochs=3,
    batch_size=4,
)

# Train from any local or Hugging Face adapter checkpoint.
wrapper, losses = iho.train(
    "preferences.parquet",
    checkpoint="SEML-Lab/IHO-Llama-3-8B-Instruct",
    output_dir="checkpoints/my-finetuned-iho-adapter",
    device="cuda",
)

wrapper, losses = iho.train(
    "preferences.parquet",
    checkpoint="checkpoints/my-iho-adapter",
    output_dir="checkpoints/resumed-iho-adapter",
    device="cuda",
)
```

The saved adapter directory can be passed back to `iho.load_model(checkpoint=...)` or `--checkpoint` for generation.

## Full Pipeline

The package also exposes `iho.IHOPipeline` for iterative sample collection,
judge scoring, preference-pair construction, and DPO training cycles. A minimal
CUDA smoke test is available in `tests/test_iho_features.py`; larger paper
experiments require the corresponding Hugging Face model access, enough GPU
memory for the selected target and judge models, and locally chosen output/cache
directories.

Set `IHO_CACHE_DIR` or pass `cache_dir`/`custom_cache_path` in configuration if
you want Hugging Face assets stored outside the default cache. If you use the
`CAT/local` target alias, set `IHO_CAT_MODEL_PATH` to the merged CAT model
directory.

## SLURM Experiments

Optional SLURM launch helpers live in `slurm/`. The main entrypoint is:

```bash
pixi run python -m slurm.slurm_wrapper --help
```

The templates in `slurm/slurm_scripts/` cover sampling ablations, single-target
ablations, multi-target ablations, cross-model sampling, and sanity runs. They
are meant to be submitted from the repository root and may need cluster-specific
`#SBATCH` headers adjusted before use.

## Checkpoints

* `SEML-Lab/IHO-Llama-3-8B-Instruct`
* `SEML-Lab/IHO-CircuitBreaker-Llama-3-8B-PolyGuard`
* `SEML-Lab/IHO-Qwen2.5-32B-Instruct`
* `SEML-Lab/IHO-Qwen2.5-7B-Instruct-PolyGuard`
* `SEML-Lab/IHO-CircuitBreaker-Llama-3-8B`
* `SEML-Lab/IHO-Qwen2.5-7B-Instruct`
* `SEML-Lab/IHO-CAT-Llama-3-8B`
* `SEML-Lab/IHO-LAT-Llama-3-8B`

## License

MIT
