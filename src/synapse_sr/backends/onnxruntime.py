"""ONNX Runtime backend for the Flash detail network.

The network is exported to ONNX once per model instance and run through
ONNX Runtime. CPU, NVIDIA CUDA, AMD ROCm and the TensorRT execution
provider are supported, depending on the ONNX Runtime build.

Only the network forward is accelerated. The physics baseline and
null-space projector stay in PyTorch, so the output is unchanged apart
from floating-point rounding.
"""

from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import torch


def _ort():
    try:
        import onnxruntime as ort
    except ImportError as exc:                     # pragma: no cover - optional dependency
        raise ImportError(
            "The ONNX Runtime backend requires `onnxruntime` (CPU) or "
            "`onnxruntime-gpu` (CUDA, TensorRT), or a ROCm build of ONNX Runtime."
        ) from exc
    return ort


class _Tuple(torch.nn.Module):
    """Return the two tensors as a tuple so the ONNX graph has two outputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["delta"], out["conf_logit"]


_PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "rocm": ["ROCMExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
}


class ONNXRuntimeNetwork:
    """Compiled ONNX Runtime model for a Flash network.

    Parameters
    ----------
    model:
        A ``SynapseFlashX5`` in eval mode, already re-parameterised.
    device:
        ``"cpu"``, ``"cuda"``, ``"rocm"`` or ``"tensorrt"``.
    cache_dir:
        TensorRT engine cache directory. Ignored for the other providers.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu",
                 cache_dir: str | None = None):
        self.device = device.lower()
        if self.device not in _PROVIDERS:
            raise ValueError(f"unknown ONNX Runtime device {device!r}; "
                             f"choose from {sorted(_PROVIDERS)}")
        self._ort = _ort()
        self._model = _Tuple(model.eval())
        self._tmp = tempfile.TemporaryDirectory(prefix="synapse-onnx-")
        self._path = os.path.join(self._tmp.name, "flash.onnx")
        self._export()
        self._cache_dir = cache_dir
        self._session = self._session()

    def _export(self):
        self._model.eval()
        example = torch.zeros(1, 10, 32, 32)
        try:
            torch.onnx.export(self._model, (example,), self._path,
                              input_names=["image"], output_names=["delta", "conf"],
                              opset_version=18, do_constant_folding=True,
                              dynamo=True, verbose=False,
                              dynamic_shapes=({2: "height", 3: "width"},))
        except TypeError:
            # Older PyTorch without the torch.export-based ONNX exporter.
            torch.onnx.export(self._model, (example,), self._path,
                              input_names=["image"], output_names=["delta", "conf"],
                              opset_version=18, do_constant_folding=True,
                              dynamo=False,
                              dynamic_axes={"image": {2: "height", 3: "width"},
                                            "delta": {2: "height5", 3: "width5"},
                                            "conf": {2: "height5", 3: "width5"}})

    def _session(self):
        available = self._ort.get_available_providers()
        providers = _PROVIDERS[self.device]
        if providers[0] not in available:
            warnings.warn(
                f"{providers[0]} is not available in this ONNX Runtime build "
                f"({', '.join(available)}); falling back to the first available provider. "
                "Install onnxruntime-gpu for CUDA or TensorRT, or a ROCm build of ONNX Runtime.",
                RuntimeWarning, stacklevel=3,
            )
        options = self._ort.SessionOptions()
        options.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        provider_options = [{} for _ in providers]
        if self.device == "tensorrt" and self._cache_dir:
            provider_options[0] = {"trt_engine_cache_enable": True,
                                   "trt_engine_cache_path": self._cache_dir}
        return self._ort.InferenceSession(self._path, sess_options=options,
                                          providers=providers,
                                          provider_options=provider_options)

    def __call__(self, x: torch.Tensor) -> dict:
        name = self._session.get_inputs()[0].name
        out = self._session.run(None, {name: x.detach().cpu().numpy()})
        # Copy: the session owns the output buffers.
        delta = torch.as_tensor(np.asarray(out[0])).clone().to(x.device)
        conf = torch.as_tensor(np.asarray(out[1])).clone().to(x.device)
        return {"delta": delta, "conf_logit": conf}
