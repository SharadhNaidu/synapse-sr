"""synapse-sr: observation-consistent Sentinel-2 super-resolution (10 m -> 2.0 m RGBN)."""

from synapse_sr.apps import Change, boundaries, change
from synapse_sr.data import fetch_sentinel2
from synapse_sr.pro import Flash, Pro
from synapse_sr.result import Result

__version__ = "0.4.2"
__all__ = ["Pro", "Flash", "Result", "load", "super_resolve", "super_resolve_folder", "fetch_sentinel2", "change", "boundaries", "Change", "clear_cache", "__version__"]

_default = {}

_BACKENDS = ("openvino", "onnx", "onnxruntime", "tensorrt")


def _check_backend(backend):
    if backend is not None and backend.lower() not in _BACKENDS:
        raise ValueError("backend must be 'openvino', 'onnx', 'onnxruntime' or 'tensorrt'")


def _apply_backend(model, backend, backend_device):
    """Attach an optional accelerator to a loaded model."""
    if backend is None:
        return model
    name = backend.lower()
    if name == "openvino":
        return model.enable_openvino(backend_device or "CPU")
    if name in ("onnx", "onnxruntime"):
        if backend_device is None:
            import torch
            backend_device = "cuda" if torch.cuda.is_available() else "cpu"
        return model.enable_onnxruntime(backend_device)
    if name == "tensorrt":
        return model.enable_tensorrt()
    raise ValueError("backend must be 'openvino', 'onnx', 'onnxruntime' or 'tensorrt'")


def load(model="flash", weights=None, device=None, backend=None, backend_device=None):
    """Load a model by short name (``"pro"``, ``"flash"``) or registered name (``"pro-v2"`` ...), or a local
    checkpoint with ``weights=``. Returns a :class:`Pro` or :class:`Flash`.

    ``backend`` optionally attaches an accelerator: ``"openvino"``, ``"onnx"``/``"onnxruntime"`` or
    ``"tensorrt"``. ``backend_device`` is the OpenVINO device (``"CPU"``, ``"GPU"``) or the ONNX Runtime
    provider device (``"cpu"``, ``"cuda"``, ``"rocm"``). Both are optional; the PyTorch path is the default.
    """
    from synapse_sr.cli import ALIASES
    _check_backend(backend)
    m = Pro.from_pretrained(ALIASES.get(model, model), weights=weights, device=device)
    return _apply_backend(m, backend, backend_device)


def super_resolve(src, model="flash", weights=None, device=None, backend=None, backend_device=None, **kwargs):
    """One call: load (once, then cached until :func:`clear_cache`) a model and super-resolve ``src``.

    ``model`` is ``"flash"`` (default: fast on any machine), ``"pro"`` (highest detail, best on a GPU) or a
    registered name; ``weights`` / ``device`` go to
    ``from_pretrained``; ``backend`` / ``backend_device`` optionally attach OpenVINO, ONNX Runtime or TensorRT;
    every other keyword goes to :meth:`Pro.super_resolve` (``tile``, ``batch``, ``scl``, ``progress`` ...).
    """
    key = (model, str(weights), str(device), backend, backend_device)
    if key not in _default:
        _default[key] = load(model, weights, device, backend, backend_device)
    return _default[key].super_resolve(src, **kwargs)


def clear_cache():
    """Release the models :func:`super_resolve` keeps loaded (and their GPU memory). Returns how many were dropped."""
    n = len(_default)
    _default.clear()
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                    # cleanup is best effort
        pass
    return n


def super_resolve_folder(src_dir, out_dir, model="flash", pattern=None, overwrite=False, cog=False,
                         weights=None, device=None, backend=None, backend_device=None, with_confidence=True,
                         preview_dir=None, verbose=True, **kwargs):
    """Super-resolve every GeoTIFF in ``src_dir`` into ``out_dir`` (same file names), loading the model once.
    By default every ``.tif`` / ``.tiff`` file (any case) is processed; ``pattern`` (a glob) narrows that.

    Scene-classification sidecars (``*_scl.tif``) are used as cloud masks, not processed. Existing outputs are
    skipped unless ``overwrite=True``. A file that fails is reported and the rest continue. Returns one summary dict
    per input (``"output"``, ``"seconds"``, consistency, support fractions, or ``"error"``).

    ``backend`` / ``backend_device`` attach an optional accelerator, as in :func:`load`. ``with_confidence=False``
    writes the four reflectance bands only, ``preview_dir`` also saves a ``<name>.png`` quicklook per scene and
    ``verbose=False`` silences the per-file lines (written to stderr).
    """
    import pathlib
    import sys
    src_dir, out_dir = pathlib.Path(src_dir), pathlib.Path(out_dir)
    found = src_dir.glob(pattern) if pattern else (f for f in src_dir.iterdir() if f.suffix.lower() in (".tif", ".tiff"))
    files = sorted(f for f in found if f.is_file() and not f.stem.lower().endswith("_scl"))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern or '*.tif / *.tiff'!r} in {src_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir = pathlib.Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
    m = load(model, weights, device, backend, backend_device)
    report = []
    for i, f in enumerate(files, 1):
        dst = out_dir / f.name
        if dst.exists() and not overwrite:
            report.append({"input": str(f), "output": str(dst), "skipped": "exists"})
            continue
        try:
            r = m.super_resolve(f, **kwargs)
            r.save(dst, with_confidence=with_confidence, cog=cog)
            if preview_dir is not None:
                import matplotlib.pyplot as plt
                plt.close(r.quicklook(preview_dir / (f.stem + ".png")))
            report.append(dict(r.summary(print_=False), input=str(f), output=str(dst)))
        except Exception as e:                           # keep going: one bad file must not stop a batch
            report.append({"input": str(f), "error": f"{type(e).__name__}: {e}"})
        if verbose:
            print(f"synapse-sr: {i}/{len(files)} {f.name} -> {report[-1].get('error') or dst}", file=sys.stderr, flush=True)
    return report
