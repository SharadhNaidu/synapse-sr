"""Application helpers on super-resolved results: change detection and boundary maps."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

from synapse_sr.result import Result

CHANGE_INDICES = {"ndvi": "vegetation loss / gain", "nbr": "burn severity (needs context)",
                  "ndwi": "flooding / water change", "mndwi": "flooding / water change (needs context)",
                  "ndbi": "construction / demolition (needs context)", "brightness": "debris, collapse, bare soil"}


@dataclass
class Change:
    """Output of :func:`change`. ``delta`` = after - before of the chosen index; ``mask`` = |delta| > threshold on
    reliable pixels; ``reliable`` = valid on both dates and not prior-dominated on either."""

    delta: np.ndarray
    mask: np.ndarray
    reliable: np.ndarray
    index: str
    threshold: float
    gsd: float

    @property
    def area_km2(self) -> float:
        return float(self.mask.sum() * self.gsd ** 2 / 1e6)

    @property
    def unreliable_fraction(self) -> float:
        return float(1 - self.reliable.mean())


def _index(r: Result, name: str) -> np.ndarray:
    if name == "brightness":
        v = r.image[:3].mean(0)
        return np.where(r.valid, v, np.nan) if r.valid is not None else v
    idx = r.indices()
    if name not in idx:
        raise KeyError(f"index {name!r} unavailable; choose from {sorted(idx)} (SWIR / red-edge indices need context)")
    return idx[name]


def _same_grid(a: Result, b: Result):
    if a.image.shape != b.image.shape:
        raise ValueError(f"before and after must be on the same grid: shapes {a.image.shape} and {b.image.shape}")
    if (a.profile is None) != (b.profile is None):
        raise ValueError("before and after must both be georeferenced (GeoTIFF input) or both be arrays")
    if a.profile is None:
        return
    if a.profile.get("crs") != b.profile.get("crs"):
        raise ValueError(f"before and after use different CRS: {a.profile.get('crs')} and {b.profile.get('crs')}")
    ta, tb = a.profile["transform"], b.profile["transform"]
    tol = 0.01 * abs(ta.a)
    if any(abs(u - v) > tol for u, v in zip(tuple(ta)[:6], tuple(tb)[:6])):
        raise ValueError(f"before and after cover different areas: transforms {tuple(ta)[:6]} and {tuple(tb)[:6]}; "
                         "clip both scenes to the same 10 m grid first")


def change(before: Result, after: Result, index: str = "ndvi", threshold: Optional[float] = None,
           min_support: int = 1) -> Change:
    """Change map between two results on the same grid (e.g. before / after a flood, fire or earthquake).
    Georeferenced results must share CRS and geotransform; a mismatch raises instead of comparing different places.

    ``index``: one of ``CHANGE_INDICES``. ``threshold`` defaults to 0.2 for normalised indices and 0.03 reflectance
    for brightness. Pixels invalid on either date, or prior-dominated (support < ``min_support``) on either date,
    are excluded from ``mask`` and reported through ``reliable``, so detected change rests on observed evidence."""
    _same_grid(before, after)
    d = _index(after, index) - _index(before, index)
    thr = threshold if threshold is not None else (0.03 if index == "brightness" else 0.2)
    rel = np.isfinite(d)                                   # index undefined on either date (e.g. dark pixels)
    for r in (before, after):
        if r.valid is not None:
            rel &= r.valid
        if r.support is not None:
            rel &= r.support >= min_support
    mask = rel & (np.abs(np.nan_to_num(d)) > thr)
    return Change(delta=d, mask=mask, reliable=rel, index=index, threshold=thr, gsd=before.gsd)


def boundaries(r: Result, kind: str = "field", sigma: float = 1.0) -> np.ndarray:
    """Boundary strength in [0, 1] on the output grid: ``"field"`` (NDVI edges: crop parcels), ``"water"`` (NDWI
    edges: shorelines, flood fronts) or ``"urban"`` (brightness edges: buildings, roads). Scaled by the 99th
    percentile; NaN where the input was invalid."""
    src = {"field": lambda: r.indices()["ndvi"], "water": lambda: r.indices()["ndwi"],
           "urban": lambda: r.image[:3].mean(0)}
    if kind not in src:
        raise KeyError(f"kind must be one of {sorted(src)}")
    v = ndimage.gaussian_filter(np.nan_to_num(src[kind]().astype(np.float32)), sigma)
    g = np.hypot(ndimage.sobel(v, 0), ndimage.sobel(v, 1))
    g = np.clip(g / (np.percentile(g, 99) + 1e-12), 0, 1)
    return np.where(r.valid, g, np.nan) if r.valid is not None else g
