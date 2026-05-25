# IHO

Minimal inference package for IHO attack generation.

## Installation

Install dependencies with [pixi](https://pixi.sh):

```bash
pixi install
```

Then run commands inside the pixi environment:

```bash
pixi run python -m iho.main "etc" --device cuda -n 32 -b 32 --attack_size 32
```

Alternatively, start an interactive shell inside the environment:

```bash
pixi shell
```

## Command Line

Run the default checkpoint with pixi:

```bash
pixi run python -m iho.main "etc" --device cuda -n 32 -b 32 --attack_size 32
```

## Python API

```python
from iho.pipeline import load_model, generate

model = load_model(
    "SEML-Lab/IHO-Llama-3-8B-Instruct",
    device="cuda",
    attack_size=32,
    attack_steps=32,
)

target_response = "Sure, here is a harmless placeholder response."
attacks = generate(model, target_response, num_attacks=32, batch_size=32)
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
