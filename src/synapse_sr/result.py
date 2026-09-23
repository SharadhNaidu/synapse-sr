"""Container for a super-resolved scene."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from synapse_sr.io import geotiff

BANDS = ("B04", "B03", "B02", "B08")


@dataclass
class Result:
    """Output of :meth:`synapse_sr.Pro.super_resolve`.

    Attributes
    ----------
    image:
        ``(4, 5H, 5W)`` float32 surface reflectance, bands B04 B03 B02 B08, on the output grid.
    confidence:
        ``(4, 5H, 5W)`` predicted absolute error scale in reflectance. A learned estimate, not calibrated.
    consistency:
        Per-band ``RMS(A(image) - y) / tau_b`` on the fully supported block, where ``A`` is the nominal
        Sentinel-2 forward model and ``tau_b`` the band's noise level.
    gsd:
        Output grid spacing in metres (2.0 for Sentinel-2 input).
    metadata:
        Processing details: scan backend, precision, tiling, regularisation weights, support thresholds.
    profile:
        Input rasterio profile when the input was a GeoTIFF, else ``None``.
    support:
        ``(5H, 5W)`` uint8: 2 HIGH (observation-determined), 1 MEDIUM, 0 LOW (prior-dominated) or invalid input.
    valid:
        ``(5H, 5W)`` bool, False where the input was NoData, cloud, cloud shadow, cirrus or saturated.
    x_base:
        ``(4, 5H, 5W)`` the observation-determined baseline.
    prior:
        ``(4, 5H, 5W)`` structure contributed by the learned prior; ``x_base + prior == image``.
    """

    image: np.ndarray
    confidence: np.ndarray
    consistency: dict
    gsd: float
    metadata: dict = field(default_factory=dict)
    profile: Optional[dict] = None
    support: Optional[np.ndarray] = None
    valid: Optional[np.ndarray] = None
    x_base: Optional[np.ndarray] = None
    prior: Optional[np.ndarray] = None

    @property
    def shape(self):
        return self.image.shape

    def ndvi(self) -> np.ndarray:
        """NDVI from the super-resolved B08 and B04, NaN where the input was invalid."""
        r, n = self.image[0], self.image[3]
        v = (n - r) / (n + r + 1e-6)
        if self.valid is not None:
            v = np.where(self.valid, v, np.nan)
        return v

    def rgb(self, percentiles=(2, 98), gamma: float = 1.0) -> np.ndarray:
        """``(5H, 5W, 3)`` uint8 true-colour quicklook (B04 B03 B02), percentile-stretched jointly."""
        img = self.image[:3]
        ok = np.isfinite(img).all(0) & (self.valid if self.valid is not None else True)
        lo, hi = np.percentile(img[:, ok], percentiles) if ok.any() else (0.0, 1.0)
        v = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1) ** (1.0 / gamma)
        return (np.nan_to_num(v).transpose(1, 2, 0) * 255).astype(np.uint8)

    def to_xarray(self):
        """``xarray.DataArray`` (band, y, x) with coordinates when the input was georeferenced.
        Requires ``xarray``."""
        import xarray as xr
        coords = {"band": list(BANDS)}
        if self.profile is not None:
            t = self.profile["transform"]
            s = self.metadata["scale"]
            h, w = self.image.shape[1:]
            coords["x"] = t.c + (np.arange(w) + 0.5) * t.a / s
            coords["y"] = t.f + (np.arange(h) + 0.5) * t.e / s
        attrs = {"gsd_m": self.gsd, "crs": str(self.profile["crs"]) if self.profile else None}
        return xr.DataArray(self.image, dims=("band", "y", "x"), coords=coords, attrs=attrs, name="reflectance")

    def save(self, path, with_confidence: bool = True):
        """Write a float32 GeoTIFF (or ``.npz`` when the input was an array).

        Bands: B04 B03 B02 B08, then ERRSCALE_* x4 and SUPPORT unless ``with_confidence=False``.
        Invalid pixels are written as NaN.
        """
        if self.profile is None:
            np.savez_compressed(path, image=self.image, confidence=self.confidence, support=self.support,
                                valid=self.valid)
            return path
        img = self.image.copy()
        if self.valid is not None:
            img[:, ~self.valid] = np.nan
        names = list(BANDS)
        extra = None
        if with_confidence:
            names += ["ERRSCALE_" + b for b in BANDS] + ["SUPPORT"]
            extra = np.concatenate([self.confidence, self.support[None].astype(np.float32)])
        geotiff.write(path, img, self.profile, self.metadata["scale"], names, extra=extra)
        return path
