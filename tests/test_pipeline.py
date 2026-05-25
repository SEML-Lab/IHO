import pytest
import torch

from iho.pipeline import _resolve_device, _resolve_dtype, load_model


def test_resolve_device_auto(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert _resolve_device("auto") == "cpu"
    assert _resolve_device("cuda:0") == "cuda:0"


def test_resolve_dtype_aliases():
    assert _resolve_dtype("bf16") is torch.bfloat16
    assert _resolve_dtype("float16") is torch.float16
    assert _resolve_dtype(torch.float32) is torch.float32


def test_strict_checkpoint_validation():
    with pytest.raises(ValueError):
        load_model("SEML-Lab/not-an-iho-checkpoint", strict_checkpoint_names=True)


def test_capitalized_llada_wrapper_import():
    from iho.model_wrapper.LLaDAWrapper import LLaDAWrapper as PublicWrapper
    from iho.model_wrapper.llada import LLaDAWrapper

    assert PublicWrapper is LLaDAWrapper


def test_load_model_signature_has_requested_options():
    import inspect

    params = inspect.signature(load_model).parameters

    assert "attack_size" in params
    assert "attack_steps" in params
    assert "cache_dir" in params
