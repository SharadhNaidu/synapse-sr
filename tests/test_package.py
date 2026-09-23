"""Package contract: arbitrary geometry, georeferencing, band mapping, preprocessing, outputs, seams.

Needs SYNAPSE_TEST_WEIGHTS (a synapse-pro .safetensors). Set SYNAPSE_TEST_LOG to a path to write the measurements.
"""

import json
import os
import pathlib

import numpy as np
import pytest
import torch

rasterio = pytest.importorskip("rasterio")
from affine import Affine  # noqa: E402
from rasterio.crs import CRS  # noqa: E402

from synapse import Pro  # noqa: E402
from synapse.io.sentinel2 import S2_12, select_bands  # noqa: E402
from synapse.models.pro import INPUT_BANDS  # noqa: E402

W = os.environ.get("SYNAPSE_TEST_WEIGHTS")
pytestmark = pytest.mark.skipif(not W, reason="SYNAPSE_TEST_WEIGHTS not set")
LOG = {}


def scene(h, w, seed=0):
    rng = np.random.default_rng(seed)
    from scipy import ndimage
    base = ndimage.gaussian_filter(rng.random((h, w)), 2) * 0.3 + 0.05
    arr = np.stack([base * (0.8 + 0.1 * k) for k in range(12)])
    for k in (4, 5, 6, 8, 10, 11):                                     # 20 m bands: 2x2 replicas
        m = arr[k][: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2).mean((1, 3))
        arr[k][: h // 2 * 2, : w // 2 * 2] = np.repeat(np.repeat(m, 2, 0), 2, 1)
    return (arr * 10000).astype(np.uint16)


def write_tif(path, arr, names=None):
    t = Affine(10.0, 0, 500000.0, 0, -10.0, 4200000.0)
    with rasterio.open(path, "w", driver="GTiff", count=arr.shape[0], height=arr.shape[1], width=arr.shape[2],
                       dtype=arr.dtype, crs=CRS.from_epsg(32643), transform=t) as d:
        d.write(arr)
        if names:
            for i, n in enumerate(names, 1):
                d.set_band_description(i, n)
    return t


@pytest.fixture(scope="module")
def model():
    return Pro.from_pretrained(weights=W)


@pytest.mark.parametrize("hw", [(128, 128), (200, 300), (300, 200), (513, 677)])
def test_arbitrary_geometry(model, tmp_path, hw):
    arr = scene(*hw)
    src = tmp_path / "in.tif"
    t = write_tif(src, arr, list(S2_12))
    r = model.super_resolve(str(src))
    assert r.image.shape == (4, 5 * hw[0], 5 * hw[1])
    assert np.isfinite(r.image).all()
    out = tmp_path / "out.tif"
    r.save(str(out))
    with rasterio.open(out) as d:
        assert (d.height, d.width) == (5 * hw[0], 5 * hw[1])
        assert d.crs == CRS.from_epsg(32643)
        assert d.transform == Affine(2.0, 0, t.c, 0, -2.0, t.f)
        with rasterio.open(src) as s:
            assert np.allclose(d.bounds, s.bounds)
    LOG[f"geometry_{hw[0]}x{hw[1]}"] = {"shape": list(r.image.shape), "consistency": r.consistency, "backend": r.metadata["scan_backend"]}


def test_tiling_matches_single_window(model):
    arr = scene(96, 96, seed=3)
    one = model.super_resolve(arr, band_names=list(S2_12), tile=96, halo=0).image
    tiled = model.super_resolve(arr, band_names=list(S2_12), tile=32, halo=16).image
    core = (slice(None), slice(80, -80), slice(80, -80))
    d = one[core] - tiled[core]
    seam = float(np.sqrt((d ** 2).mean()) / (one[core].std() + 1e-12))
    LOG["tiling"] = {"rel_rms_full_vs_tiled_interior": seam, "max_abs": float(np.abs(d).max())}
    assert seam < 0.05, seam


def test_band_mapping():
    arr = np.arange(12)[:, None, None] * np.ones((12, 2, 2))
    out = select_bands(arr)
    assert [int(v) for v in out[:, 0, 0]] == [S2_12.index(b) for b in INPUT_BANDS]
    named = select_bands(arr, list(S2_12))
    assert np.array_equal(out, named)
    with pytest.raises(ValueError):
        select_bands(np.zeros((7, 2, 2)))


def test_preprocessing_scl_and_nodata(model):
    arr = scene(64, 64, seed=5)
    arr[:, :8, :8] = 0
    scl = np.full((32, 32), 4, np.uint8); scl[16:, 16:] = 9            # 20 m SCL, cloud high probability
    r = model.super_resolve(arr, band_names=list(S2_12), scl=scl)
    assert not r.valid[:40, :40].any() and not r.valid[160:, 160:].any() and r.valid[100, 50]
    assert (r.support[~r.valid] == 0).all()
    LOG["preprocessing"] = {"invalid_input_fraction": r.metadata["invalid_input_fraction"]}


def test_outputs_and_decomposition(model):
    arr = scene(64, 64, seed=7)
    r = model.super_resolve(arr, band_names=list(S2_12))
    assert r.confidence.shape == r.image.shape and (r.confidence > 0).all()
    assert set(np.unique(r.support)) <= {0, 1, 2}
    assert np.allclose(r.x_base + r.prior, r.image, atol=1e-5)
    assert max(r.consistency.values()) < 5.0
    LOG["outputs"] = {"consistency": r.consistency, "support_fraction": {int(k): float((r.support == k).mean()) for k in (0, 1, 2)}}


def teardown_module(_):
    if os.environ.get("SYNAPSE_TEST_LOG"):
        LOG["torch"] = torch.__version__
        pathlib.Path(os.environ["SYNAPSE_TEST_LOG"]).write_text(json.dumps(LOG, indent=1))
