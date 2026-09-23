"""Container for a super-resolved scene."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from synapse_sr.io import geotiff

BANDS = ("B04", "B03", "B02", "B08")
CONTEXT_BANDS = ("B05", "B06", "B07", "B8A", "B11", "B12")


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
    context:
        ``(6, 5H, 5W)`` float16 B05 B06 B07 B8A B11 B12 replicated from their native 20 m grid (not
        super-resolved), or ``None``.
    calibration:
        The checkpoint's calibrated error model (coefficients, noise levels, conformal quantiles), used by
        :meth:`uncertainty` and :meth:`interval`; ``None`` when the checkpoint ships without one.
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
    context: Optional[np.ndarray] = None
    calibration: Optional[dict] = None

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

    def band(self, name: str) -> np.ndarray:
        """One band by name: B04 B03 B02 B08 (super-resolved) or a 20 m context band (replicated)."""
        if name in BANDS:
            return self.image[BANDS.index(name)]
        if self.context is not None and name in CONTEXT_BANDS:
            return self.context[CONTEXT_BANDS.index(name)].astype(np.float32)
        raise KeyError(f"band {name!r} not available (context bands present: {self.context is not None})")

    def indices(self) -> dict:
        """Application indices on the output grid, NaN where the input was invalid.

        From the super-resolved bands: NDVI, SAVI, EVI, GNDVI (vegetation / crops) and NDWI (water). With 20 m
        context: NDRE (crop stress), NDBI (built-up), NBR (burn severity) and MNDWI (water / flood); these carry the
        20 m spatial detail of their red-edge / SWIR band."""
        r, g, b, n = (self.image[i].astype(np.float32) for i in range(4))
        nd = lambda a, c: (a - c) / (a + c + 1e-6)
        out = {"ndvi": nd(n, r), "ndwi": nd(g, n), "gndvi": nd(n, g),
               "savi": 1.5 * (n - r) / (n + r + 0.5),
               "evi": 2.5 * (n - r) / (n + 6 * r - 7.5 * b + 1.0)}
        if self.context is not None:
            re1, sw1, sw2 = self.band("B05"), self.band("B11"), self.band("B12")
            out.update({"ndre": nd(n, re1), "ndbi": nd(sw1, n), "nbr": nd(n, sw2), "mndwi": nd(g, sw1)})
        if self.valid is not None:
            out = {k: np.where(self.valid, v, np.nan) for k, v in out.items()}
        return out

    def uncertainty(self) -> np.ndarray:
        """Expected absolute error per pixel and band (reflectance), ``(4, 5H, 5W)``, from the checkpoint's calibrated
        error model: log|error| regressed on the learned error scale, the prior's magnitude relative to sensor noise,
        local edge strength and variance, brightness, NDVI and band, fitted against a held-out HR reference."""
        em = (self.calibration or {}).get("error_model")
        if not em:
            raise RuntimeError("this checkpoint ships no calibrated error model; use `confidence` as a relative scale")
        from scipy import ndimage
        x = self.image.astype(np.float32)
        br = x[:3].mean(0)
        pad = np.pad(br, 1, mode="edge")
        gx = ndimage.sobel(pad, 1)[1:-1, 1:-1]; gy = ndimage.sobel(pad, 0)[1:-1, 1:-1]
        mu = ndimage.uniform_filter(br, 5, mode="nearest"); sd = np.sqrt(np.maximum(ndimage.uniform_filter(br * br, 5, mode="nearest") - mu * mu, 0))
        ndvi = np.clip((x[3] - x[0]) / (x[3] + x[0] + 1e-6), -1, 1)   # physical range; guards near-zero denominators
        tau = np.asarray(em["tau"], np.float32)[:, None, None]
        feats = [np.log(self.confidence.astype(np.float32)), np.log(np.abs(self.prior) / tau + 1e-3),
                 np.broadcast_to(np.hypot(gx, gy), x.shape), np.broadcast_to(sd, x.shape), np.broadcast_to(br, x.shape),
                 np.broadcast_to(ndvi, x.shape), np.broadcast_to(np.arange(4)[:, None, None] == 3, x.shape).astype(np.float32)]
        w = np.asarray(em["weights"], np.float32)
        z = w[0] + sum(wi * f for wi, f in zip(w[1:], feats))
        return np.exp(np.clip(z, np.log(1e-4), np.log(0.3)))       # bounded to physically possible errors

    def interval(self, level: float = 0.9) -> np.ndarray:
        """Calibrated error half-width (reflectance), ``(4, 5H, 5W)``: with probability ``level`` the reference value
        lies within ``image +/- half-width``. Levels 0.80, 0.90 and 0.95 are calibrated by split conformal prediction on
        held-out HR reference data; see the model card for the measured coverage."""
        em = (self.calibration or {}).get("error_model")
        if not em:
            raise RuntimeError("this checkpoint ships no uncertainty calibration; use `confidence` as a relative scale")
        q = em["quantiles"].get(f"{level:.2f}")
        if q is None:
            raise KeyError(f"calibrated levels: {sorted(em['quantiles'])}")
        return q * self.uncertainty()

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
