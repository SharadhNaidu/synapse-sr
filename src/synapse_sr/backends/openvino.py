"""OpenVINO backend for the Flash detail network.

Only the network forward runs in OpenVINO. The physics baseline and null-space
projector stay in PyTorch, so the output and the observation-consistency
guarantee are unchanged.

Install the optional dependency with ``pip install openvino``.
"""

from __future__ import annotations

import numpy as np
import torch


def _openvino():
    try:
        import openvino as ov
    except ImportError as exc:                     # pragma: no cover - optional dependency
        raise ImportError(
            "The OpenVINO backend requires the `openvino` package. "
            "Install it with `pip install openvino`."
        ) from exc
    return ov


class _Tuple(torch.nn.Module):
    """Return the two tensors as a tuple so the export is a plain tensor pair."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["delta"], out["conf_logit"]


class OpenVINONetwork:
    """Compiled OpenVINO model for a Flash network.

    Parameters
    ----------
    model:
        A ``SynapseFlashX5`` in eval mode, already re-parameterised.
    device:
        Any OpenVINO device string: ``"CPU"``, ``"GPU"``, ``"GPU.0"``, ...
    """

    def __init__(self, model: torch.nn.Module, device: str = "CPU"):
        ov = _openvino()
        self.device = device
        self._core = ov.Core()
        self._model = _Tuple(model.eval())
        self._compiled = self._compile()
        self._out_keys = None

    def _compile(self):
        ov = _openvino()
        example = torch.zeros(1, 10, 32, 32)
        dynamic = (
            {2: torch.export.Dim("height", min=16),
             3: torch.export.Dim("width", min=16)},
        )
        exported = torch.export.export(self._model, (example,), dynamic_shapes=dynamic)
        ov_model = ov.convert_model(exported)
        return self._core.compile_model(ov_model, self.device)

    def __call__(self, x: torch.Tensor) -> dict:
        result = self._compiled(x.detach().cpu().numpy())
        if self._out_keys is None:
            self._out_keys = list(result.keys())
        # Copy: OpenVINO may reuse the output buffer on the next inference.
        delta = torch.as_tensor(np.asarray(result[self._out_keys[0]])).clone().to(x.device)
        conf = torch.as_tensor(np.asarray(result[self._out_keys[1]])).clone().to(x.device)
        return {"delta": delta, "conf_logit": conf}
