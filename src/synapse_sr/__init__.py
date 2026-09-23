"""synapse-sr: observation-consistent Sentinel-2 super-resolution (10 m -> 2.0 m RGBN)."""

from synapse_sr.data import fetch_sentinel2
from synapse_sr.pro import Pro
from synapse_sr.result import Result

__version__ = "0.1.0"
__all__ = ["Pro", "Result", "super_resolve", "fetch_sentinel2", "__version__"]

_default = {}


def super_resolve(src, weights=None, device=None, **kwargs):
    """One-call convenience: load (and cache) the default model, then :meth:`Pro.super_resolve`.

    ``weights`` / ``device`` are passed to :meth:`Pro.from_pretrained`; every other keyword goes to
    :meth:`Pro.super_resolve`.
    """
    key = (str(weights), str(device))
    if key not in _default:
        _default[key] = Pro.from_pretrained(weights=weights, device=device)
    return _default[key].super_resolve(src, **kwargs)
