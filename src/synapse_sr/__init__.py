"""synapse-sr: observation-consistent Sentinel-2 super-resolution (10 m -> 2.0 m RGBN)."""

from synapse_sr.apps import Change, boundaries, change
from synapse_sr.data import fetch_sentinel2
from synapse_sr.pro import Flash, Pro
from synapse_sr.result import Result

__version__ = "0.3.0"
__all__ = ["Pro", "Flash", "Result", "load", "super_resolve", "fetch_sentinel2", "change", "boundaries", "Change", "__version__"]

_default = {}


def load(model="pro", weights=None, device=None):
    """Load a model by short name (``"pro"``, ``"flash"``) or registered name (``"pro-v2"`` ...), or a local
    checkpoint with ``weights=``. Returns a :class:`Pro` or :class:`Flash`."""
    from synapse_sr.cli import ALIASES
    return Pro.from_pretrained(ALIASES.get(model, model), weights=weights, device=device)


def super_resolve(src, model="pro", weights=None, device=None, **kwargs):
    """One call: load (once, then cached) a model and super-resolve ``src``.

    ``model`` is ``"pro"`` (default), ``"flash"`` or a registered name; ``weights`` / ``device`` go to
    ``from_pretrained``; every other keyword goes to :meth:`Pro.super_resolve` (``tile``, ``batch``, ``scl``,
    ``progress`` ...).
    """
    key = (model, str(weights), str(device))
    if key not in _default:
        _default[key] = load(model, weights, device)
    return _default[key].super_resolve(src, **kwargs)
