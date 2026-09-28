# Cheat sheet

Everything you usually need, on one page. Every snippet runs as written once you have a Sentinel-2 GeoTIFF
called `scene.tif` (or use `fetch_sentinel2` to get one).

## Install

```bash
pip install synapse-sr                 # core
pip install "synapse-sr[stac]"         # + download scenes with fetch_sentinel2 / --fetch
pip install "synapse-sr[all]"          # + xarray
synapse-sr --env                       # what hardware and kernels will be used
```

## Command line

| Task | Command |
|---|---|
| Super-resolve a GeoTIFF (Flash, the default) | `synapse-sr scene.tif scene_2m.tif` |
| Use Pro (most detail, best on a GPU) | `synapse-sr scene.tif scene_2m.tif --model pro` |
| Download a scene and super-resolve it | `synapse-sr --fetch 12.92,77.50 --dates 2025-01-01:2025-03-15 out.tif` |
| Every GeoTIFF in a folder | `synapse-sr scenes/ scenes_2m/` |
| Cloud-Optimised GeoTIFF for web maps | `synapse-sr scene.tif out.tif --cog` |
| Also save a preview image | `synapse-sr scene.tif out.tif --preview preview.png` |
| Four bands only, for a GIS | `synapse-sr scene.tif out.tif --no-confidence` |
| Machine-readable output for scripts | `synapse-sr scene.tif out.tif --json` |
| Force CPU | `synapse-sr scene.tif out.tif --device cpu` |
| List models / versions | `synapse-sr --models`, `synapse-sr --version` |

## Python: the essentials

```python
import synapse_sr

r = synapse_sr.super_resolve("scene.tif")                 # Flash; model="pro" for Pro
r.save("scene_2m.tif")                                    # georeferenced GeoTIFF (cog=True for a COG)
r.summary()                                               # size, consistency, support, time
```

## Choose a model and a device

```python
from synapse_sr import Flash, Pro

flash = Flash.from_pretrained()                           # ~1 s per 1.28 km scene, any CPU or GPU
pro = Pro.from_pretrained(device="cuda")                  # most detail
r = flash.super_resolve("scene.tif")
```

## Get data

```python
import synapse_sr

scene = synapse_sr.fetch_sentinel2(lat=12.9237, lon=77.4987, start="2025-01-01", end="2025-03-15", size_m=2000)
r = synapse_sr.super_resolve(scene)                       # cloud mask and radiometric offset handled automatically
```

## Look at it

```python
import synapse_sr

r = synapse_sr.super_resolve("scene.tif")
r.quicklook("preview.png")                                # 10 m | 2 m | support, side by side
r.show(["image", "x_base", "prior", "support", "uncertainty", "ndvi"])
rgb = r.rgb()                                             # (5H, 5W, 3) uint8 true colour
```

## Trust layers

```python
import synapse_sr

r = synapse_sr.super_resolve("scene.tif")
r.x_base          # what the 10 m measurement determines
r.prior           # what the network added (x_base + prior == image)
r.support         # 2 measured, 1 medium, 0 inferred or invalid
r.uncertainty()   # calibrated expected error per pixel and band
r.interval(0.9)   # half-width of the 90 % interval
r.valid           # False on NoData, cloud, shadow, cirrus, saturation
```

## Applications

```python
import synapse_sr

r = synapse_sr.super_resolve("scene.tif")
idx = r.indices()                                         # ndvi savi evi gndvi ndwi ndre ndbi nbr mndwi
fields = synapse_sr.boundaries(r, "field")                # also "water", "urban"

before = synapse_sr.super_resolve("before.tif")
after = synapse_sr.super_resolve("after.tif")
flood = synapse_sr.change(before, after, "ndwi")          # also "ndvi", "nbr", "ndbi", "brightness"
print(flood.area_km2, flood.unreliable_fraction)
```

## Batches

```python
import synapse_sr

report = synapse_sr.super_resolve_folder("scenes/", "scenes_2m/", cog=True)   # skips *_scl.tif and existing outputs
failed = [r for r in report if "error" in r]
```

## Other inputs

```python
import numpy as np
import synapse_sr

arr = np.random.randint(500, 3000, (10, 64, 64)).astype(np.uint16)       # (C, H, W) DN or reflectance
r = synapse_sr.super_resolve(arr, band_names=["B04", "B03", "B02", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"])
r.save("result.npz")                                                     # arrays have no georeference
```

torch tensors `(C, H, W)` and xarray `DataArray`s with a `band` coordinate work the same way.

## Useful options

| Option | Meaning |
|---|---|
| `model="flash"` / `"pro"` | which network (one call API) |
| `device="cpu"`, `"cuda"`, `"mps"` | where to run |
| `tta=True` | average over 8 flips / rotations; small accuracy gain, 8x network time |
| `scl="scene_SCL.tif"`, `scl=None` | cloud mask file, or no masking |
| `offset=-1000` | radiometric offset for raw baseline 04.00+ DN without a tag |
| `tile=`, `batch=` | override the automatic memory-aware tiling |
| `progress=False` | silence the progress bar (`SYNAPSE_SR_QUIET=1` everywhere) |
| `discrepancy=` | how tightly the physics fits the measurement, in sensor-noise units (Pro 0.5, Flash 4) |
| `restore_mean=` | restore each 10 m pixel's measured mean reflectance (Flash: on, Pro: off) |
