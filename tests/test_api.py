"""Public API contract, runnable on CPU without trained weights."""

import json

import numpy as np
import pytest
import torch

import synapse_sr
from synapse_sr import Pro, Result
from synapse_sr.cli import main as cli
from synapse_sr.io.sentinel2 import S2_12, select_bands, to_reflectance
from synapse_sr.models.pro import INPUT_BANDS

from conftest import scene, write_tif

rasterio = pytest.importorskip("rasterio")


# ------------------------------------------------------------------ geometry and georeferencing
@pytest.mark.parametrize("hw", [(24, 24), (20, 36), (36, 20), (23, 29)])
def test_any_shape_gives_exact_5x_grid_and_preserves_georeference(model, tmp_path, hw):
    src = tmp_path / "in.tif"
    t = write_tif(src, scene(*hw))
    r = model.super_resolve(str(src), tile=16, halo=6)
    assert r.image.shape == (4, 5 * hw[0], 5 * hw[1]) and np.isfinite(r.image).all()
    assert r.gsd == 2.0
    out = r.save(str(tmp_path / "out.tif"))
    with rasterio.open(out) as d, rasterio.open(src) as s:
        assert (d.height, d.width) == (5 * hw[0], 5 * hw[1])
        assert d.crs == s.crs
        assert d.transform.a == 2.0 and d.transform.e == -2.0 and (d.transform.c, d.transform.f) == (t.c, t.f)
        assert np.allclose(d.bounds, s.bounds)
        assert d.descriptions[:4] == ("B04", "B03", "B02", "B08") and d.descriptions[-1] == "SUPPORT"


def test_tiling_is_seamless(model):
    arr = scene(40, 40, seed=3)
    one = model.super_resolve(arr, band_names=S2_12, tile=40, halo=0).image
    tiled = model.super_resolve(arr, band_names=S2_12, tile=16, halo=12).image
    core = (slice(None), slice(60, -60), slice(60, -60))
    rel = np.sqrt(((one[core] - tiled[core]) ** 2).mean()) / one[core].std()
    assert rel < 0.05, rel


# ------------------------------------------------------------------ input forms
def test_numpy_torch_and_reflectance_inputs_agree(model):
    dn = scene(20, 20, seed=1)
    a = model.super_resolve(dn, band_names=S2_12, tile=20).image
    b = model.super_resolve(torch.from_numpy(dn.astype(np.int32)), band_names=S2_12, tile=20).image
    c = model.super_resolve(dn.astype(np.float32) / 1e4, band_names=S2_12, tile=20).image
    assert np.allclose(a, b, atol=1e-6) and np.allclose(a, c, atol=1e-5)


def test_xarray_input_uses_band_coordinate_and_drops_singleton_time(model):
    xr = pytest.importorskip("xarray")
    dn = scene(20, 20, seed=2)
    da = xr.DataArray(dn[None], dims=("time", "band", "y", "x"), coords={"band": list(S2_12)})
    r = model.super_resolve(da, tile=20)
    ref = model.super_resolve(dn, band_names=S2_12, tile=20)
    assert np.allclose(r.image, ref.image, atol=1e-6)
    with pytest.raises(ValueError, match="select one"):
        model.super_resolve(xr.concat([da, da], "time"))


def test_band_mapping_orders_and_names():
    arr = np.arange(12)[:, None, None] * np.ones((12, 2, 2))
    assert [int(v) for v in select_bands(arr)[:, 0, 0]] == [S2_12.index(b) for b in INPUT_BANDS]
    assert np.array_equal(select_bands(arr), select_bands(arr, [b.lower() for b in S2_12]))
    short = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
    assert np.array_equal(select_bands(arr), select_bands(arr, short))
    with pytest.raises(ValueError):
        select_bands(np.zeros((7, 2, 2)))


def test_reflectance_scaling_and_offset():
    assert np.allclose(to_reflectance(np.array([2000], np.uint16)), 0.2)
    assert np.allclose(to_reflectance(np.array([2000], np.uint16), offset=-1000), 0.1)
    assert np.allclose(to_reflectance(np.array([0.2], np.float32)), 0.2)


def test_offset_tag_is_applied_automatically(model, tmp_path):
    dn = scene(20, 20, seed=4)
    plain, tagged = tmp_path / "a.tif", tmp_path / "b.tif"
    write_tif(plain, dn)
    write_tif(tagged, (dn.astype(np.int32) + 1000).astype(np.uint16), tags={"BOA_ADD_OFFSET": "-1000"})
    a = model.super_resolve(str(plain), tile=20)
    b = model.super_resolve(str(tagged), tile=20)
    assert b.metadata["offset"] == -1000.0 and np.allclose(a.image, b.image, atol=1e-5)


# ------------------------------------------------------------------ preprocessing
def test_nodata_and_scl_masking_including_auto_sidecar(model, tmp_path):
    dn = scene(24, 24, seed=5)
    dn[:, :4, :4] = 0
    src = tmp_path / "s.tif"
    write_tif(src, dn)
    scl = np.full((12, 12), 4, np.uint8); scl[6:, 6:] = 9
    write_tif(tmp_path / "s_scl.tif", scl[None], names=None)
    r = model.super_resolve(str(src), tile=24)
    assert r.metadata["scl"].endswith("s_scl.tif")
    assert not r.valid[:20, :20].any() and not r.valid[60:, 60:].any() and r.valid[50, 10]
    assert (r.support[~r.valid] == 0).all()
    none = model.super_resolve(str(src), tile=24, scl=None)
    assert none.metadata["scl"] is None and none.valid[60:, 60:].all()
    with rasterio.open(r.save(str(tmp_path / "o.tif"))) as d:
        assert np.isnan(d.read(1)[70, 70]) and np.isfinite(d.read(1)[50, 10])


def test_rejects_non_sentinel_grids(model, tmp_path):
    from affine import Affine
    from rasterio.crs import CRS
    p = tmp_path / "r.tif"
    with rasterio.open(p, "w", driver="GTiff", count=12, height=8, width=8, dtype="uint16", crs=CRS.from_epsg(32643),
                       transform=Affine(20.0, 0, 0, 0, -20.0, 0)) as d:
        d.write(scene(8, 8))
    with pytest.raises(ValueError, match="10 m grid"):
        model.super_resolve(str(p))


# ------------------------------------------------------------------ outputs
def test_outputs_decomposition_support_and_helpers(model):
    r = model.super_resolve(scene(20, 20, seed=7), band_names=S2_12, tile=20)
    assert isinstance(r, Result)
    assert r.confidence.shape == r.image.shape and (r.confidence > 0).all()
    assert set(np.unique(r.support)) <= {0, 1, 2}
    assert np.allclose(r.x_base + r.prior, r.image, atol=1e-5)
    assert set(r.consistency) == {"B04", "B03", "B02", "B08"} and all(np.isfinite(list(r.consistency.values())))
    q = r.rgb()
    assert q.shape == (100, 100, 3) and q.dtype == np.uint8
    assert r.ndvi().shape == (100, 100)
    assert r.metadata["scan_backend"] in ("fused", "triton", "pytorch") and r.metadata["precision"] in ("bfloat16", "float32")


def test_projection_leaves_the_observation_unchanged(model):
    """A x_hat must equal A x_base: the learned part lies in the null space of the forward operator."""
    r = model.super_resolve(scene(24, 24, seed=8), band_names=S2_12, tile=24, halo=0)
    x = torch.tensor(r.image)[None]; xb = torch.tensor(r.x_base)[None]
    d = model.op(x) - model.op(xb)
    assert float(d.abs().max()) < 1e-4 * float(model.op(xb).abs().max())


def test_to_xarray_coordinates(model, tmp_path):
    pytest.importorskip("xarray")
    src = tmp_path / "x.tif"
    t = write_tif(src, scene(20, 20))
    da = model.super_resolve(str(src), tile=20).to_xarray()
    assert da.dims == ("band", "y", "x") and float(da.x[0]) == t.c + 1.0 and float(da.y[0]) == t.f - 1.0


def test_array_input_saves_npz(model, tmp_path):
    r = model.super_resolve(scene(16, 16), band_names=S2_12, tile=16)
    r.save(str(tmp_path / "o.npz"))
    z = np.load(tmp_path / "o.npz")
    assert z["image"].shape == (4, 80, 80)


# ------------------------------------------------------------------ checkpoints, registry, CLI
def test_save_and_load_roundtrip(model, tmp_path):
    p = model.save_pretrained(str(tmp_path / "m.safetensors"))
    m2 = Pro.from_pretrained(weights=p, device="cpu")
    arr = scene(16, 16, seed=9)
    assert np.allclose(model.super_resolve(arr, band_names=S2_12, tile=16).image,
                       m2.super_resolve(arr, band_names=S2_12, tile=16).image, atol=1e-6)
    assert "Pro(" in repr(m2)


def test_unpublished_registry_entry_gives_actionable_error():
    from synapse_sr.pretrained import registry
    m = registry.MANIFEST_DIR / "pro-v1.json"
    if json.loads(m.read_text())["url"].startswith("http"):
        pytest.skip("weights published")
    with pytest.raises(RuntimeError, match="weights="):
        Pro.from_pretrained()
    with pytest.raises(KeyError):
        registry.manifest("no-such-model")


def test_one_call_function_and_cli(model, tmp_path, capsys):
    p = model.save_pretrained(str(tmp_path / "m.safetensors"))
    src = tmp_path / "in.tif"
    write_tif(src, scene(16, 16))
    r = synapse_sr.super_resolve(str(src), weights=p, device="cpu", tile=16)
    assert r.image.shape == (4, 80, 80)
    assert cli([str(src), str(tmp_path / "o.tif"), "--weights", p, "--device", "cpu", "--tile", "16", "--no-confidence", "--json"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["shape"] == [4, 80, 80]
    with rasterio.open(tmp_path / "o.tif") as d:
        assert d.count == 4
    assert cli(["--version"]) == 0 and synapse_sr.__version__ in capsys.readouterr().out
    assert cli(["--env", "--json"]) == 0 and '"torch"' in capsys.readouterr().out


# ------------------------------------------------------------------ applications
def test_indices_context_bands_and_applications(model):
    import synapse_sr
    arr = scene(20, 20, seed=11)
    r = model.super_resolve(arr, band_names=S2_12, tile=20)
    idx = r.indices()
    assert {"ndvi", "ndwi", "savi", "evi", "gndvi", "ndre", "ndbi", "nbr", "mndwi"} <= set(idx)
    assert all(v.shape == (100, 100) for v in idx.values())
    assert r.context.shape == (6, 100, 100) and np.allclose(r.band("B11")[0, :5], r.band("B11")[0, 0], atol=1e-3)
    no_ctx = model.super_resolve(arr, band_names=S2_12, tile=20, context=False)
    assert no_ctx.context is None and "nbr" not in no_ctx.indices()
    for kind in ("field", "water", "urban"):
        b = synapse_sr.boundaries(r, kind)
        assert b.shape == (100, 100) and np.nanmin(b) >= 0 and np.nanmax(b) <= 1


def test_change_is_zero_on_identical_input_and_detects_a_change(model):
    import synapse_sr
    arr = scene(24, 24, seed=12)
    a = model.super_resolve(arr, band_names=S2_12, tile=24)
    same = synapse_sr.change(a, a, "ndvi")
    assert same.mask.sum() == 0 and same.area_km2 == 0.0
    burnt = arr.copy(); burnt[7, 8:16, 8:16] = burnt[7, 8:16, 8:16] // 4          # NIR collapse over 80 m
    b = model.super_resolve(burnt, band_names=S2_12, tile=24)
    ch = synapse_sr.change(a, b, "ndvi")
    assert ch.mask[45:75, 45:75].mean() > 0.3 and ch.mask[:20, :20].mean() < 0.05
    with pytest.raises(KeyError):
        synapse_sr.change(a, b, "not_an_index")


def test_uncertainty_and_interval_require_and_use_calibration(model):
    r = model.super_resolve(scene(16, 16), band_names=S2_12, tile=16)
    with pytest.raises(RuntimeError, match="calibrat"):
        r.interval(0.9)
    r.calibration = {"error_model": {"weights": [-4.0, 0.3, 0.1, 2.0, 1.0, 0.5, 0.1, 0.2],
                                     "tau": [0.00068, 0.00084, 0.00086, 0.00192],
                                     "quantiles": {"0.80": 1.2, "0.90": 1.6, "0.95": 2.0}}}
    u = r.uncertainty()
    assert u.shape == r.image.shape and (u > 0).all() and np.isfinite(u).all()
    assert np.allclose(r.interval(0.9), 1.6 * u)
    with pytest.raises(KeyError):
        r.interval(0.5)
