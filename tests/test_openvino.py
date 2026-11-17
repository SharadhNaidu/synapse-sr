"""OpenVINO backend: conversion and parity with PyTorch (skipped when OpenVINO is absent)."""

import pathlib
import sys

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

pytest.importorskip("openvino")

from synapse_sr.backends.openvino import OpenVINONetwork  # noqa: E402
from synapse_sr.models.flash import SynapseFlashX5  # noqa: E402


@pytest.mark.skipif(not hasattr(torch, "export"), reason="torch.export needs PyTorch 2.0+")
def test_openvino_flash_matches_pytorch():
    torch.manual_seed(0)
    model = SynapseFlashX5.v2().eval().reparameterise()
    x = torch.randn(1, 10, 32, 32)
    with torch.no_grad():
        ref = model(x)

    net = OpenVINONetwork(model, "CPU")
    got = net(x)

    assert got["delta"].shape == ref["delta"].shape
    assert got["conf_logit"].shape == ref["conf_logit"].shape
    assert torch.allclose(got["delta"], ref["delta"], atol=1e-4, rtol=1e-4)
    assert torch.allclose(got["conf_logit"], ref["conf_logit"], atol=1e-3, rtol=1e-3)
