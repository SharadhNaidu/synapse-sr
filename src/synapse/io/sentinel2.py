"""Sentinel-2 band bookkeeping for SYNAPSE inputs.

The model consumes 10 bands in INPUT_BANDS order, as surface reflectance (L2A DN / 10000). 20 m bands
(B05 B06 B07 B8A B11 B12) are expected resampled onto the 10 m grid; they carry native 20 m information only.
"""

import numpy as np

from synapse.models.pro import INPUT_BANDS

S2_12 = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
S2_13 = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")


def select_bands(arr, names=None):
    """(C, H, W) array -> (10, H, W) in INPUT_BANDS order.

    names: band names of arr's channels (e.g. GeoTIFF descriptions). Without names, 10-, 12- and 13-channel
    stacks are interpreted as INPUT_BANDS, the L2A 12-band and the L1C 13-band orders respectively."""
    if names and all(n in names for n in INPUT_BANDS):
        return arr[[names.index(b) for b in INPUT_BANDS]]
    order = {10: INPUT_BANDS, 12: S2_12, 13: S2_13}.get(arr.shape[0])
    if order is None:
        raise ValueError(f"cannot map {arr.shape[0]} channels to Sentinel-2 bands; pass band names "
                         f"or a stack in one of the orders {INPUT_BANDS}, {S2_12}, {S2_13}")
    return arr[[order.index(b) for b in INPUT_BANDS]]


def to_reflectance(arr, scale=10000.0, offset=0.0):
    """DN -> reflectance: (DN + offset) / scale. L2A processing baseline >= 04.00 uses offset -1000."""
    arr = arr.astype(np.float32)
    if arr.max() <= 2.0:                     # already reflectance
        return arr
    return (arr + offset) / scale
