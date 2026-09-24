# Verify it yourself

You should not have to take anything in these docs on trust. Each claim below comes with a few lines of code you
can run on your own scene. All of them were run on the RV University scene from the [Quick start](../quickstart.md)
before being published here.

```python
import numpy as np, torch, synapse_sr
from synapse_sr import Pro

model = Pro.from_pretrained()
scene = synapse_sr.fetch_sentinel2(lat=12.9237, lon=77.4987, start="2025-01-01", end="2025-03-15", size_m=1280)
r = model.super_resolve(scene)
```

## 1. The output is where the input was

The output GeoTIFF must have the same CRS and the same bounds, with pixels exactly 5x smaller.

```python
import rasterio

r.save("out_2m.tif")
with rasterio.open(scene) as a, rasterio.open("out_2m.tif") as b:
    assert a.crs == b.crs
    assert np.allclose(a.bounds, b.bounds)
    assert b.res == (a.res[0] / 5, a.res[1] / 5)
    print(a.bounds, b.bounds, a.res, b.res)
```

Open both files in QGIS as well: roads and field edges line up, with no offset.

## 2. Re-observing the output reproduces the measurement

`r.consistency` reports `RMS(A(image) - y) / tau` per band, where `A` is the Sentinel-2 forward model (the
instrument's point-spread function, then sampling at 10 m) and `tau` is the band's noise level. You can
recompute it from scratch:

```python
from synapse_sr.io import geotiff
from synapse_sr.io.sentinel2 import select_bands, to_reflectance
from synapse_sr.models import baseline

arr, profile, names, tags = geotiff.read(scene)
y = torch.tensor(to_reflectance(select_bands(arr, names), offset=float(tags.get("BOA_ADD_OFFSET", 0))))[None, :4]
x = torch.tensor(r.image)[None]
A = model.op.cpu()
(o, q), (o2, q2) = A.block(x.shape[-2]), A.block(x.shape[-1])
resid = A(x) - y[..., o:o + q, o2:o2 + q2]
print((resid.pow(2).mean((0, 2, 3)).sqrt() / baseline.tau_for(A)).tolist())   # ~1: at sensor-noise level
```

Values near 1 mean the output agrees with the satellite measurement to within sensor noise. A generic
upsampler or GAN has no such constraint. Run the same check on `bicubic = torch.nn.functional.interpolate(y, scale_factor=5, mode="bicubic")`
to compare.

Your numbers can differ slightly from `r.consistency`, which is measured per tile on the tiled mosaic. They are
of the same size.

## 3. The network cannot change what the satellite saw

The learned detail (`r.prior`) lies in the null space of `A`, so it is invisible to the sensor:

```python
xb = torch.tensor(r.x_base)[None]
d = A(x) - A(xb)
print(float(d.abs().max()), float(A(xb).abs().max()))   # e.g. 0.00035 vs 1.22: a few parts in 10,000
```

This holds tile by tile. The difference comes from floating-point error and tile seams, not from the model.

## 4. Where the detail comes from

```python
r.show(["image", "x_base", "prior", "support"])
print(r.summary(print_=False)["support_fraction"])      # {'high': ..., 'medium': ..., 'low': ...}
```

- `x_base` depends only on the measurement: there is no network in it.
- `prior` is everything the network added.
- `support` says, pixel by pixel, which of the two dominates.

For decisions, use only pixels with `r.support == 2`, or weight them by `r.uncertainty()`.

## 5. Accuracy against your own high-resolution reference

If you have a reference image of the same area and date (drone, aerial, or commercial imagery), co-register it
and resample it to the 2 m output grid first (`gdalwarp -tr 2 2 -r average -te <output bounds>`). Then:

```python
with rasterio.open("reference_2m.tif") as f:
    ref = f.read([1, 2, 3, 4]).astype("float32") / 10000     # same bands and units: B04 B03 B02 B08 reflectance

def rmse(a, b, m):
    return float(np.sqrt(np.nanmean((a[:, m] - b[:, m]) ** 2)))

valid = r.valid & np.isfinite(ref).all(0)
bic = torch.nn.functional.interpolate(y, scale_factor=5, mode="bicubic")[0].numpy()
for name, img in [("bicubic", bic), ("x_base", r.x_base), ("synapse-sr", r.image)]:
    print(f"{name:11s} RMSE {rmse(img, ref, valid):.4f}  on HIGH-support pixels {rmse(img, ref, valid & (r.support == 2)):.4f}")
```

Also check that the uncertainty is honest. About 90 % of pixels should fall inside the 90 % interval:

```python
half = r.interval(0.9)
inside = np.abs(r.image - ref) <= half
print("coverage at 90 %:", float(inside[:, valid].mean()))
```

The shipped calibration was fitted against 2 m references derived from US aerial imagery. If coverage on your
region is clearly below the nominal level, treat the intervals as optimistic there.

!!! warning "Reference data pitfalls"
    Differences in acquisition date, viewing angle, sensor spectral response and co-registration all show up as
    "error". Use a reference acquired within days of the Sentinel-2 image, match the bands spectrally, and
    register it to sub-pixel accuracy before judging any super-resolution model.

## 6. Same numbers on every machine

The CPU, CUDA, Triton and mamba-ssm paths compute the same model. Compare them yourself:

```python
cpu = Pro.from_pretrained(device="cpu").super_resolve(scene, progress=False).image
gpu = Pro.from_pretrained(device="cuda").super_resolve(scene, progress=False, tile=32).image
print(np.abs(cpu - gpu).max())      # small: bfloat16 rounding on the GPU, identical physics
```

Use the same `tile` on both, because the tile size changes the context each tile sees.

## 7. The weights are the published ones

Every download is checked against the SHA-256 in the package's registry, on every load. To check by hand:

```python
from synapse_sr.pretrained import download, registry
m = registry.manifest("pro-v2")
print(download.sha256(download.cache_dir() / m["file"]) == m["sha256"])
```

## 8. Run the test suite

```bash
git clone https://github.com/SharadhNaidu/synapse-sr && cd synapse-sr
pip install -e ".[test]"
pytest -q
```

The suite needs no weights or network. It checks each fast path against the exact reference it replaces:

- the forward operator against its 0.5 m fine-grid form;
- the scan against a float64 sequential recurrence, including extreme decay rates, and its gradients;
- the preconditioned solvers against plain conjugate gradients;
- Triton against the PyTorch scan (on CUDA);
- the Flash re-parameterisation against its training form.

It also checks georeferencing, tiling seams, input formats, masking, the null-space property, the CLI and the
application helpers.
