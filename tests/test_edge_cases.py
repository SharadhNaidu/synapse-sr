"""Edge cases behind the 0.4.3 fixes: masks, NoData, input scaling, grids, persistence, memory and downloads."""

import hashlib
import http.server
import math
import threading

import numpy as np
import pytest
import torch

import synapse_sr
from synapse_sr import Result, change, memory
from synapse_sr.data import check_area
from synapse_sr.io.sentinel2 import S2_12, to_reflectance
from synapse_sr.pretrained import download
from synapse_sr.pro import _scl_invalid

from conftest import scene, write_tif

rasterio = pytest.importorskip("rasterio")


# ------------------------------------------------------------------ cloud masks
def test_scl_coarser_finer_and_mismatched():
    coarse = np.full((12, 15), 4)
    coarse[0, 0] = 9
    m = _scl_invalid(coarse, (23, 29))
    assert m.shape == (23, 29) and m[:2, :2].all() and m.sum() == 4
    fine = np.full((46, 58), 4)
    fine[1, 1] = 8
    m = _scl_invalid(fine, (23, 29))
    assert m[0, 0] and m.sum() == 1
    with pytest.raises(ValueError, match="aspect"):
        _scl_invalid(np.zeros((10, 40)), (23, 29))


# ------------------------------------------------------------------ outputs
def test_geotiff_declares_nan_nodata(model, tmp_path):
    arr = scene(16, 16)
    arr[:, :2, :2] = 0
    write_tif(tmp_path / "in.tif", arr)
    r = model.super_resolve(str(tmp_path / "in.tif"), tile=16)
    for cog in (False, True):
        r.save(tmp_path / "o.tif", cog=cog)
        with rasterio.open(tmp_path / "o.tif") as d:
            assert math.isnan(d.nodata)
            x = d.read(masked=True)
            assert x.mask[0].sum() == 100 and x.mask[4].sum() == 100 and not x.mask[-1].any()


@pytest.mark.parametrize("name", ["o.tif", "o.npz"])
def test_result_round_trip(model, tmp_path, name):
    write_tif(tmp_path / "in.tif", scene(16, 16))
    r = model.super_resolve(str(tmp_path / "in.tif"), tile=16)
    r.save(tmp_path / name)
    q = Result.load(tmp_path / name)
    assert np.allclose(q.image[:, q.valid], r.image[:, r.valid]) and (q.support == r.support).all()
    assert q.consistency == r.consistency and q.metadata["model"] == r.metadata["model"]
    assert q.profile["transform"] == r.profile["transform"] and q.profile["crs"] == r.profile["crs"]
    assert (q.x_base is not None) == name.endswith(".npz")
    q.save(tmp_path / ("again_" + name))


def test_array_result_saves_npz_only(model, tmp_path):
    r = model.super_resolve(scene(16, 16), band_names=S2_12, tile=16)
    with pytest.raises(ValueError, match="npz"):
        r.save(tmp_path / "o.tif")
    q = Result.load(r.save(tmp_path / "o.npz"))
    assert q.profile is None and np.allclose(q.prior, r.prior)
    with pytest.raises(ValueError, match="not written by synapse-sr"):
        write_tif(tmp_path / "x.tif", scene(8, 8))
        Result.load(tmp_path / "x.tif")


def test_xarray_dataset_layers(model, tmp_path):
    pytest.importorskip("xarray")
    write_tif(tmp_path / "in.tif", scene(16, 16))
    r = model.super_resolve(str(tmp_path / "in.tif"), tile=16)
    ds = r.to_xarray(dataset=True)
    assert {"reflectance", "x_base", "prior", "confidence", "support", "valid", "context"} <= set(ds.data_vars)
    assert ds["context"].dims == ("context_band", "y", "x") and ds["reflectance"].sizes["band"] == 4
    assert r.to_xarray().shape == (4, 80, 80)


# ------------------------------------------------------------------ inputs
def test_hot_pixel_does_not_flip_reflectance():
    a = np.full((4, 50, 50), 0.2, np.float32)
    a[0, 0, 0] = 3.5
    assert np.isclose(to_reflectance(a)[1, 0, 0], 0.2)
    assert np.isclose(to_reflectance(np.array([2000.0], np.float32))[0], 0.2)
    assert np.isclose(to_reflectance(np.array([2], np.uint16))[0], 2e-4)


def test_window_needs_a_path(model):
    with pytest.raises(ValueError, match="GeoTIFF path"):
        model.super_resolve(scene(16, 16), band_names=S2_12, window=(0, 0, 8, 8))


def test_window_reads_scl_over_the_same_area(model, tmp_path):
    from affine import Affine
    t = write_tif(tmp_path / "in.tif", scene(40, 40))
    scl = np.full((20, 20), 4, np.uint8)
    scl[5, 5] = 9
    with rasterio.open(tmp_path / "in_scl.tif", "w", driver="GTiff", count=1, height=20, width=20, dtype="uint8",
                       crs="EPSG:32643", transform=Affine(20, 0, t.c, 0, -20, t.f)) as d:
        d.write(scl, 1)
    r = model.super_resolve(str(tmp_path / "in.tif"), window=(8, 8, 16, 16), tile=16)
    assert np.argwhere(~r.valid[::5, ::5]).tolist() == [[2, 2], [2, 3], [3, 2], [3, 3]]


def test_change_rejects_different_grids():
    from affine import Affine
    from rasterio.crs import CRS
    img = np.random.default_rng(0).random((4, 10, 10)).astype(np.float32)
    prof = {"crs": CRS.from_epsg(32643), "transform": Affine(10, 0, 5e5, 0, -10, 4.2e6)}

    def res(p):
        return Result(img, img, {}, 2.0, {"scale": 5}, p, np.full((10, 10), 2, np.uint8), np.ones((10, 10), bool))
    change(res(prof), res(dict(prof)))
    with pytest.raises(ValueError, match="different areas"):
        change(res(prof), res(dict(prof, transform=Affine(10, 0, 5e5 + 100, 0, -10, 4.2e6))))
    with pytest.raises(ValueError, match="different CRS"):
        change(res(prof), res(dict(prof, crs=CRS.from_epsg(32644))))
    with pytest.raises(ValueError, match="both be georeferenced"):
        change(res(prof), res(None))


@pytest.mark.parametrize("args, message", [
    ((95, 0, "2025-01-01", "2025-02-01"), "latitude"),
    ((0, 0, "2025-1-1x", "2025-02-01"), "YYYY-MM-DD"),
    ((0, 0, "2025-03-01", "2025-02-01"), "after"),
    ((0, 0, "2014-01-01", "2014-02-01"), "2015"),
    ((0, 0, "2025-01-01", "2025-02-01", 1.0), "size_m"),
])
def test_fetch_argument_checks(args, message):
    with pytest.raises(ValueError, match=message):
        check_area(*args)


# ------------------------------------------------------------------ engine behaviour
def test_accelerator_failure_is_reported(model):
    def boom(y):
        raise RuntimeError("bad shape")
    model._accel, model._accel_name, model._warned = boom, "onnx:test", False
    try:
        with pytest.warns(RuntimeWarning, match="onnx:test failed"):
            r = model.super_resolve(scene(32, 32), band_names=S2_12, tile=16, halo=4)
        assert r.metadata["detail_backend_fallbacks"] == 4
    finally:
        model._accel, model._accel_name = None, "pytorch"


def test_auto_mps_falls_back_to_cpu(model):
    model.device, model._auto_device = torch.device("mps"), True
    try:
        if torch.backends.mps.is_available():
            pytest.skip("real MPS present")
        with pytest.warns(RuntimeWarning, match="MPS could not run"):
            r = model.super_resolve(scene(16, 16), band_names=S2_12, tile=16)
        assert r.image.shape == (4, 80, 80) and model.device.type == "cpu"
    finally:
        model.to("cpu")
        model._auto_device = False


def test_memory_override_and_preflight(monkeypatch):
    monkeypatch.setenv("SYNAPSE_SR_MEMORY", "1.5")
    assert memory.cpu_budget() == 1.5e9
    monkeypatch.setenv("SYNAPSE_SR_MEMORY", "lots")
    with pytest.raises(ValueError, match="number of GB"):
        memory.cpu_budget()
    monkeypatch.delenv("SYNAPSE_SR_MEMORY")
    monkeypatch.setattr(memory, "available_ram", lambda: 4e9)
    memory.preflight(500, 500, torch.device("cpu"))
    with pytest.raises(MemoryError, match="window="):
        memory.preflight(10980, 10980, torch.device("cpu"))
    monkeypatch.setenv("SYNAPSE_SR_PREFLIGHT", "0")
    memory.preflight(10980, 10980, torch.device("cpu"))


def test_clear_cache(model, tmp_path):
    p = model.save_pretrained(str(tmp_path / "m.safetensors"))
    synapse_sr.super_resolve(scene(16, 16), weights=p, device="cpu", band_names=S2_12, tile=16)
    assert synapse_sr.clear_cache() >= 1 and synapse_sr.clear_cache() == 0


# ------------------------------------------------------------------ downloads
def test_download_resumes_after_a_dropped_connection(tmp_path, monkeypatch):
    data = np.random.default_rng(0).bytes(300_000)
    calls = []

    class Flaky(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            start = int(self.headers.get("Range", "bytes=0-")[6:].rstrip("-") or 0)
            calls.append(start)
            body = data[start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[:100_000] if len(calls) == 1 else body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Flaky)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("SYNAPSE_CACHE", str(tmp_path))
    monkeypatch.setattr(download.time, "sleep", lambda s: None)
    try:
        p = download.fetch(f"http://127.0.0.1:{srv.server_port}/w", hashlib.sha256(data).hexdigest(), "w.safetensors")
    finally:
        srv.shutdown()
    assert p.read_bytes() == data and calls == [0, 100_000]
    assert (tmp_path / "w.safetensors.sha256").exists()
    assert download.file_sha256(p) == hashlib.sha256(data).hexdigest()


def test_download_mirror_and_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com/")
    assert download.resolve("https://huggingface.co/a/b") == "https://hf-mirror.com/a/b"
    assert download.resolve("https://example.com/a") == "https://example.com/a"
    monkeypatch.setenv("SYNAPSE_CACHE", str(tmp_path))
    monkeypatch.setenv("SYNAPSE_OFFLINE", "1")
    with pytest.raises(FileNotFoundError, match="offline"):
        download.fetch("https://huggingface.co/x", "0" * 64, "missing.safetensors")


def test_checksum_sidecar_invalidates_on_change(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"one")
    assert download.file_sha256(f) == hashlib.sha256(b"one").hexdigest()
    f.write_bytes(b"three")
    assert download.file_sha256(f) == hashlib.sha256(b"three").hexdigest()

