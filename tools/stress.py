"""Stress test with the published weights: every input form, size, layout and failure mode, checked for correctness.

    python tools/stress.py [--model flash|pro] [--device cpu|cuda] [--big 1000]

Each case asserts properties of the output (shape, georeferencing, finiteness, physical range, masks, consistency),
not merely that no exception was raised. Prints one line per case and a summary; exit code 1 on any failure.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import warnings

import numpy as np

warnings.simplefilter("ignore")
import synapse_sr  # noqa: E402
from synapse_sr.io.sentinel2 import S2_12  # noqa: E402

TEN = ["B04", "B03", "B02", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]
L1C = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
RESULTS = []


def scene(h, w, seed=0, bands=12, dn=True):
    rng = np.random.default_rng(seed)
    from numpy.fft import irfft2, rfft2
    noise = rng.random((h, w))
    ky = np.fft.fftfreq(h)[:, None]; kx = np.fft.rfftfreq(w)[None]
    base = irfft2(rfft2(noise) * np.exp(-(ky ** 2 + kx ** 2) * 60), s=(h, w))
    base = (base - base.min()) / (np.ptp(base) + 1e-9) * 0.3 + 0.05
    arr = np.stack([base * (0.8 + 0.05 * k) for k in range(bands)])
    return (arr * 10000).astype(np.uint16) if dn else arr.astype(np.float32)


def write_tif(path, arr, names, crs="EPSG:32643", res=10.0, tags=None, nodata=None, compress=None):
    import rasterio
    from affine import Affine
    t = Affine(res, 0, 500000.0, 0, -res, 4200000.0)
    kw = dict(driver="GTiff", count=arr.shape[0], height=arr.shape[1], width=arr.shape[2], dtype=arr.dtype, crs=crs,
              transform=t, nodata=nodata)
    if compress:
        kw.update(compress=compress, tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **kw) as d:
        d.write(arr)
        for i, n in enumerate(names or [], 1):
            d.set_band_description(i, n)
        if tags:
            d.update_tags(**tags)
    return t


def check(r, h, w):
    assert r.image.shape == (4, 5 * h, 5 * w), r.image.shape
    v = r.image[:, r.valid] if r.valid.any() else np.zeros((4, 0))
    assert np.isfinite(v).all(), "non-finite output on valid pixels"
    assert (v >= 0).all() and (v <= 1.5).all(), (float(v.min()), float(v.max()))
    assert r.support.shape == (5 * h, 5 * w) and set(np.unique(r.support)) <= {0, 1, 2}


def case(name):
    def deco(fn):
        def run(ctx):
            t = time.time()
            try:
                info = fn(ctx) or ""
                RESULTS.append((name, "PASS", round(time.time() - t, 2), str(info)))
            except Exception as e:
                RESULTS.append((name, "FAIL", round(time.time() - t, 2), f"{type(e).__name__}: {e}"))
                if ctx["verbose"]:
                    traceback.print_exc()
            print("%-46s %-4s %6.2fs  %s" % RESULTS[-1], flush=True)
        run.name = name
        return run
    return deco


CASES = []


def register(name):
    def deco(fn):
        CASES.append(case(name)(fn))
        return fn
    return deco


# ------------------------------------------------------------------ sizes and shapes
for hw in [(8, 8), (17, 23), (23, 17), (60, 60), (64, 300), (300, 64), (129, 131)]:
    def _f(ctx, hw=hw):
        r = ctx["m"].super_resolve(scene(*hw), band_names=S2_12, progress=False)
        check(r, *hw)
        return f"consistency max {max(r.consistency.values()):.2f}"
    register(f"size {hw[0]}x{hw[1]}")(_f)


@register("size 1x1 -> clear error")
def _(ctx):
    try:
        ctx["m"].super_resolve(scene(1, 1), band_names=S2_12, progress=False)
    except ValueError as e:
        return str(e)[:60]
    raise AssertionError("1x1 should raise a ValueError")


@register("big scene (default 1000x1000 = 10 km)")
def _(ctx):
    n = ctx["big"]
    import psutil
    p = psutil.Process()
    rss0 = p.memory_info().rss
    t = time.time()
    r = ctx["m"].super_resolve(scene(n, n, seed=3), band_names=S2_12, progress=False)
    check(r, n, n)
    return f"{n}x{n} -> {5 * n}x{5 * n} in {time.time() - t:.1f} s, RSS +{(p.memory_info().rss - rss0) / 2**30:.1f} GB, tile {r.metadata['tile']}"


# ------------------------------------------------------------------ input forms
@register("dtypes uint16 / int32 / float32 / float64 agree")
def _(ctx):
    dn = scene(40, 40, seed=5)
    a = ctx["m"].super_resolve(dn, band_names=S2_12, progress=False).image
    for arr in (dn.astype(np.int32), dn.astype(np.float32) / 1e4, dn.astype(np.float64) / 1e4):
        b = ctx["m"].super_resolve(arr, band_names=S2_12, progress=False).image
        assert np.allclose(a, b, atol=2e-4), float(np.abs(a - b).max())


@register("torch tensor (C,H,W) and (1,C,H,W)")
def _(ctx):
    import torch
    dn = scene(30, 30, seed=6).astype(np.int32)
    a = ctx["m"].super_resolve(torch.from_numpy(dn), band_names=S2_12, progress=False).image
    b = ctx["m"].super_resolve(torch.from_numpy(dn)[None], band_names=S2_12, progress=False).image
    assert np.allclose(a, b)


@register("xarray with band coordinate")
def _(ctx):
    import xarray as xr
    dn = scene(30, 30, seed=7)
    da = xr.DataArray(dn[None], dims=("time", "band", "y", "x"), coords={"band": list(S2_12)})
    r = ctx["m"].super_resolve(da, progress=False)
    assert r.to_xarray().dims == ("band", "y", "x")


@register("band layouts 10 / 12 / 13 / shuffled named")
def _(ctx):
    base = scene(30, 30, seed=8, bands=13)
    by = dict(zip(L1C, base))
    ref = ctx["m"].super_resolve(np.stack([by[b] for b in TEN]), progress=False).image
    for names in (S2_12, L1C, list(reversed(L1C))):
        arr = np.stack([by[b] for b in names])
        out = ctx["m"].super_resolve(arr, band_names=names, progress=False).image
        assert np.allclose(out, ref, atol=1e-6), names[:3]
    arr = np.stack([by[b] for b in L1C])
    out = ctx["m"].super_resolve(arr, progress=False).image        # 13 channels, L1C order by count
    assert np.allclose(out, ref, atol=1e-6)


@register("missing bands -> clear error")
def _(ctx):
    try:
        ctx["m"].super_resolve(scene(20, 20, bands=7), progress=False)
    except ValueError as e:
        return str(e)[:60]
    raise AssertionError("7 bands should raise")


# ------------------------------------------------------------------ bad data
@register("NaN / inf pixels masked, rest finite")
def _(ctx):
    arr = scene(40, 40, seed=9, dn=False)
    arr[:, 3:8, 3:8] = np.nan; arr[:, 30, 30] = np.inf
    r = ctx["m"].super_resolve(arr, band_names=S2_12, offset=0.0, progress=False)
    check(r, 40, 40)
    assert not r.valid[20:35, 20:35].any() and not r.valid[150:155, 150:155].any() and r.valid[160:, 160:].all()


@register("all nodata -> NaN consistency, all LOW")
def _(ctx):
    r = ctx["m"].super_resolve(np.zeros((12, 30, 30), np.uint16), band_names=S2_12, progress=False)
    assert all(np.isnan(v) for v in r.consistency.values()) and (r.support == 0).all()


@register("saturated + negative reflectance stay physical")
def _(ctx):
    arr = scene(40, 40, seed=10, dn=False)
    arr[:, :10] = 1.6; arr[:, -5:] = -0.05
    r = ctx["m"].super_resolve(arr, band_names=S2_12, offset=0.0, progress=False)
    check(r, 40, 40)


@register("salt-and-pepper noise bounded")
def _(ctx):
    arr = scene(40, 40, seed=11, dn=False)
    rng = np.random.default_rng(0); sp = rng.random(arr.shape[1:]) < 0.05
    arr[:, sp] = rng.choice([0.0, 3.0], sp.sum())
    r = ctx["m"].super_resolve(arr, band_names=S2_12, offset=0.0, progress=False)
    check(r, 40, 40)
    return f"clamped {100 * r.metadata['clamped_fraction']:.2f} %"


# ------------------------------------------------------------------ GeoTIFF I/O
@register("GeoTIFF: CRS, bounds, 5x grid, tags, COG-style input")
def _(ctx):
    import rasterio
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.tif")
        t = write_tif(src, scene(48, 40, seed=12), S2_12, crs="EPSG:32743", compress="deflate",
                      tags={"BOA_ADD_OFFSET": "0"})
        r = ctx["m"].super_resolve(src, progress=False)
        out = r.save(os.path.join(d, "o.tif"))
        with rasterio.open(out) as o, rasterio.open(src) as s:
            assert o.crs == s.crs and np.allclose(o.bounds, s.bounds)
            assert o.res == (2.0, 2.0) and o.count == 9
            assert o.descriptions[:4] == ("B04", "B03", "B02", "B08")


@register("GeoTIFF: SCL sidecar masks clouds")
def _(ctx):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.tif")
        write_tif(src, scene(40, 40, seed=13), S2_12)
        scl = np.full((1, 20, 20), 4, np.uint8); scl[0, 10:, 10:] = 9
        write_tif(os.path.join(d, "s_scl.tif"), scl, None, res=20.0)
        r = ctx["m"].super_resolve(src, progress=False)
        assert not r.valid[120:, 120:].any() and r.valid[:80, :80].all()


@register("GeoTIFF: offset tag applied (baseline >= 04.00)")
def _(ctx):
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.tif"), os.path.join(d, "b.tif")
        dn = scene(30, 30, seed=14)
        write_tif(a, dn, S2_12)
        write_tif(b, (dn.astype(np.int32) + 1000).astype(np.uint16), S2_12, tags={"BOA_ADD_OFFSET": "-1000"})
        ra, rb = ctx["m"].super_resolve(a, progress=False), ctx["m"].super_resolve(b, progress=False)
        assert np.allclose(ra.image, rb.image, atol=1e-5)


@register("GeoTIFF: non-10 m grid rejected")
def _(ctx):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.tif")
        write_tif(src, scene(20, 20), S2_12, res=20.0)
        try:
            ctx["m"].super_resolve(src, progress=False)
        except ValueError as e:
            return str(e)[:60]
        raise AssertionError("20 m grid should be rejected")


# ------------------------------------------------------------------ determinism and outputs
@register("deterministic across runs")
def _(ctx):
    arr = scene(40, 40, seed=15)
    a = ctx["m"].super_resolve(arr, band_names=S2_12, progress=False).image
    b = ctx["m"].super_resolve(arr, band_names=S2_12, progress=False).image
    assert np.array_equal(a, b)


@register("outputs: indices, uncertainty, interval, npz, summary")
def _(ctx):
    r = ctx["m"].super_resolve(scene(40, 40, seed=16), band_names=S2_12, progress=False)
    idx = r.indices()
    assert {"ndvi", "ndwi", "savi", "evi", "ndre", "ndbi", "nbr", "mndwi"} <= set(idx)
    for v in idx.values():
        f = v[np.isfinite(v)]
        assert f.size and f.min() >= -1.5 and f.max() <= 1.5
    u = r.uncertainty(); assert u.shape == r.image.shape and np.isfinite(u).all() and (u > 0).all()
    assert (r.interval(0.9) >= u).all()
    with tempfile.TemporaryDirectory() as d:
        r.save(os.path.join(d, "o.npz"))
    s = r.summary(print_=False); json.dumps(s)


@register("applications: change + boundaries")
def _(ctx):
    a = ctx["m"].super_resolve(scene(40, 40, seed=17), band_names=S2_12, progress=False)
    arr = scene(40, 40, seed=17); arr[[S2_12.index("B08")], 10:30, 10:30] //= 3
    b = ctx["m"].super_resolve(arr, band_names=S2_12, progress=False)
    ch = synapse_sr.change(a, b, "ndvi")
    assert ch.mask.any() and 0 <= ch.unreliable_fraction <= 1
    for k in ("field", "water", "urban"):
        e = synapse_sr.boundaries(b, k); f = e[np.isfinite(e)]
        assert f.min() >= 0 and f.max() <= 1
    return f"changed {ch.area_km2:.4f} km2"


@register("tta option runs and stays physical")
def _(ctx):
    r = ctx["m"].super_resolve(scene(30, 30, seed=18), band_names=S2_12, progress=False, tta=True)
    check(r, 30, 30)


# ------------------------------------------------------------------ CLI
@register("CLI: run, --json, --env, --models, errors")
def _(ctx):
    exe = [sys.executable, "-m", "synapse_sr"]
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.tif")
        write_tif(src, scene(32, 32, seed=19), S2_12)
        p = subprocess.run(exe + [src, os.path.join(d, "o.tif"), "--json", "--model", ctx["model"], "--device", ctx["device"]],
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-300:]
        out = json.loads(p.stdout.strip().splitlines()[-1]); assert out["shape"] == [4, 160, 160]
        for extra in (["--env", "--json"], ["--models", "--json"], ["--version"]):
            q = subprocess.run(exe + extra, capture_output=True, text=True); assert q.returncode == 0, q.stderr[-200:]
        q = subprocess.run(exe + [os.path.join(d, "missing.tif"), "o.tif"], capture_output=True, text=True)
        assert q.returncode == 1 and "Traceback" not in q.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flash")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--big", type=int, default=1000)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    m = synapse_sr.load(a.model, device=a.device)
    print(f"synapse-sr {synapse_sr.__version__}  {m!r}  load {time.time() - t0:.1f} s", flush=True)
    ctx = {"m": m, "model": a.model, "device": a.device, "big": a.big, "verbose": a.verbose}
    for c in CASES:
        c(ctx)
    fails = [r for r in RESULTS if r[1] != "PASS"]
    print(f"\nSTRESS SUMMARY: {len(RESULTS) - len(fails)}/{len(RESULTS)} passed in {time.time() - t0:.0f} s  ({a.model}, {a.device})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
