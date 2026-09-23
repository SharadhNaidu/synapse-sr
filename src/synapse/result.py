from dataclasses import dataclass, field

import numpy as np

from synapse.io import geotiff


@dataclass
class Result:
    """image: (4, H, W) reflectance B04 B03 B02 B08 at `gsd` m. confidence: (4, H, W) predicted absolute
    complement-error scale in reflectance (a support map, lower = better supported; not calibrated).
    consistency: per-band RMS(A(image) - y) / tau_b on the supported block (nominal S2 forward model)."""

    image: np.ndarray
    confidence: np.ndarray
    consistency: dict
    gsd: float
    metadata: dict = field(default_factory=dict)
    profile: dict = None
    support: np.ndarray = None       # (H, W) uint8: 2 HIGH, 1 MEDIUM, 0 LOW / invalid
    valid: np.ndarray = None         # (H, W) bool: input not NoData / cloud / shadow (SCL)
    x_base: np.ndarray = None        # observation-determined baseline
    prior: np.ndarray = None         # P_N(delta): structure supplied by the learned prior

    def save(self, path, with_confidence=True):
        if self.profile is None:
            np.savez_compressed(path, image=self.image, confidence=self.confidence, support=self.support)
            return path
        img = self.image.copy()
        if self.valid is not None:
            img[:, ~self.valid] = np.nan
        names = ["B04", "B03", "B02", "B08"]
        extra = None
        if with_confidence:
            names += ["ERRSCALE_" + b for b in ("B04", "B03", "B02", "B08")] + ["SUPPORT"]
            extra = np.concatenate([self.confidence, self.support[None].astype(np.float32)])
        geotiff.write(path, img, self.profile, self.metadata["scale"], names, extra=extra)
        return path
