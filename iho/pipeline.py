from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

import torch

from iho.model_wrapper import LLaDAWrapper

DEFAULT_BASE_MODEL = "GSAI-ML/LLaDA-8B-Base"

SUPPORTED_CHECKPOINTS = (
    "SEML-Lab/IHO-Llama-3-8B-Instruct",
    "SEML-Lab/IHO-Llama-3-8B-Instruct-RR",
    "SEML-Lab/IHO-Llama-3-8B-Instruct-RR-with-detector",
    "SEML-Lab/IHO-LAT-Llama-3-8B",
    "SEML-Lab/IHO-Qwen2.5-7B-Instruct",
    "SEML-Lab/IHO-Gemma-3-1B-IT",
)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype

    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype}") from exc


@dataclass
class IHOModel:
    """Loaded IHO attack generator."""

    checkpoint: str
    wrapper: LLaDAWrapper
    steps: int = 32
    temperature: float = 0.0
    remasking: Literal["low_confidence", "random"] = "low_confidence"
    mask_padding: bool = True

    @torch.no_grad()
    def generate(
        self,
        affirmative_response: str | Sequence[str],
        *,
        num_attacks: int = 1,
        batch_size: int = 8,
        return_dict: bool = False,
        steps: Optional[int] = None,
        temperature: Optional[float] = None,
        remasking: Optional[Literal["low_confidence", "random"]] = None,
        mask_padding: Optional[bool] = None,
    ) -> list[str] | list[dict[str, Any]]:
        if num_attacks < 1:
            raise ValueError("num_attacks must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if isinstance(affirmative_response, str):
            responses = [affirmative_response] * num_attacks
        else:
            responses = [
                response
                for response in affirmative_response
                for _ in range(num_attacks)
            ]

        attack_steps = steps if steps is not None else self.steps
        attack_temperature = temperature if temperature is not None else self.temperature
        attack_remasking = remasking if remasking is not None else self.remasking
        attack_mask_padding = mask_padding if mask_padding is not None else self.mask_padding

        self.wrapper.eval()
        outputs: list[str] | list[dict[str, Any]] = []

        for start in range(0, len(responses), batch_size):
            batch_responses = responses[start : start + batch_size]
            target_sequences = [f"\nAnswer: {response}" for response in batch_responses]
            target_ids = self.wrapper.encode(target_sequences)

            masking_result = self.wrapper.mask_tokens(
                token_ids=target_ids,
                masking_mode="attack",
                mask_all=True,
            )

            inpainted_ids, loglikelihood = self.wrapper.predict_masked(
                masked_ids=masking_result.masked_ids,
                steps=attack_steps,
                temperature=attack_temperature,
                remasking=attack_remasking,
                mask_padding=attack_mask_padding,
            )

            attack_ids = inpainted_ids[masking_result.mask_positions].view(
                inpainted_ids.size(0),
                -1,
            )
            attack_texts = self.wrapper.decode(attack_ids)

            if return_dict:
                full_texts = self.wrapper.decode(inpainted_ids)
                outputs.extend(
                    {
                        "attack": attack_text,
                        "affirmative_response": response,
                        "attack_ids": attack_id.cpu().tolist(),
                        "inpainted_ids": inpainted_id.cpu().tolist(),
                        "inpainted_text": full_text,
                        "attack_loglikelihood": float(score.item()),
                        "checkpoint": self.checkpoint,
                    }
                    for attack_text, response, attack_id, inpainted_id, full_text, score in zip(
                        attack_texts,
                        batch_responses,
                        attack_ids,
                        inpainted_ids,
                        full_texts,
                        loglikelihood,
                    )
                )
            else:
                outputs.extend(attack_texts)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return outputs


def load_model(
    checkpoint: str,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    device: str = "auto",
    cache_dir: Optional[str] = None,
    dtype: str | torch.dtype = torch.bfloat16,
    attack_prefix_length: int = 32,
    steps: int = 32,
    temperature: float = 0.0,
    remasking: Literal["low_confidence", "random"] = "low_confidence",
    mask_padding: bool = True,
    strict_checkpoint_names: bool = False,
) -> IHOModel:
    """Load an IHO generator checkpoint from Hugging Face or a local path."""

    if strict_checkpoint_names and checkpoint not in SUPPORTED_CHECKPOINTS:
        supported = ", ".join(SUPPORTED_CHECKPOINTS)
        raise ValueError(f"Unsupported checkpoint {checkpoint!r}. Known checkpoints: {supported}")

    wrapper = LLaDAWrapper(
        model_name=base_model,
        device=_resolve_device(device),
        cache_dir=cache_dir,
        attacker_input_size=attack_prefix_length,
        dtype=_resolve_dtype(dtype),
        lora_checkpoint=checkpoint,
    )
    return IHOModel(
        checkpoint=checkpoint,
        wrapper=wrapper,
        steps=steps,
        temperature=temperature,
        remasking=remasking,
        mask_padding=mask_padding,
    )


def generate(
    model: IHOModel,
    affirmative_response: str | Sequence[str],
    **kwargs: Any,
) -> list[str] | list[dict[str, Any]]:
    """Generate IHO attack prompts for one or more target responses."""

    return model.generate(affirmative_response, **kwargs)
