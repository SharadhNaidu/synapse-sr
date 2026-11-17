"""ONNX Runtime backend: conversion and parity with PyTorch (skipped when absent)."""

import pathlib
import sys

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from synapse_sr.backends.onnxruntime import ONNXRuntimeNetwork  # noqa: E402
from synapse_sr.models.flash import SynapseFlashX5  # noqa: E402


def test_onnxruntime_flash_matches_pytorch():
    torch.manual_seed(0)
    model = SynapseFlashX5.v2().eval().reparameterise()
    x = torch.randn(1, 10, 32, 32)
    with torch.no_grad():
        ref = model(x)

    net = ONNXRuntimeNetwork(model, "cpu")
    got = net(x)

    assert got["delta"].shape == ref["delta"].shape
    assert got["conf_logit"].shape == ref["conf_logit"].shape
    assert torch.allclose(got["delta"], ref["delta"], atol=1e-4, rtol=1e-4)
    assert torch.allclose(got["conf_logit"], ref["conf_logit"], atol=1e-3, rtol=1e-3)
