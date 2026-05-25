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
