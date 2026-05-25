# IHO

Minimal inference package for IHO attack generation.

```python
from iho.pipeline import load_model, generate

model = load_model("SEML-Lab/IHO-Llama-3-8B-Instruct")
target_response = "Sure, here is a harmless placeholder response."

attacks = generate(model, target_response)
```

The public API is intentionally small for evaluation use:

- `load_model(checkpoint)` loads an IHO PEFT adapter from Hugging Face or a local path.
- `generate(model, target_response)` returns attack prompts targeting the supplied response.

Known checkpoint names include:

- `SEML-Lab/IHO-Llama-3-8B-Instruct`
- `SEML-Lab/IHO-Llama-3-8B-Instruct-RR`
- `SEML-Lab/IHO-Llama-3-8B-Instruct-RR-with-detector`
- `SEML-Lab/IHO-LAT-Llama-3-8B`
- `SEML-Lab/IHO-Qwen2.5-7B-Instruct`
- `SEML-Lab/IHO-Gemma-3-1B-IT`

By default, models are loaded on CUDA when available and use Hugging Face's normal cache resolution. Pass `cache_dir=...` to `load_model` if you want an explicit cache location.
