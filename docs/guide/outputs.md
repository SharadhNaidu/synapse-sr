# Outputs

`super_resolve` returns a [`Result`](../api.md#synapse_sr.Result).

| Attribute | Shape / type | Meaning |
|---|---|---|
| `image` | `(4, 5H, 5W)` float32 | surface reflectance, B04 B03 B02 B08, 2.0 m grid |
| `confidence` | `(4, 5H, 5W)` float32 | predicted absolute error scale (reflectance); learned, not calibrated |
| `support` | `(5H, 5W)` uint8 | 2 HIGH, 1 MEDIUM, 0 LOW or invalid; see [Support](support.md) |
| `valid` | `(5H, 5W)` bool | False where the input was NoData, cloud, shadow, cirrus or saturated |
| `consistency` | dict | per-band RMS of `A(image) - y` in noise units |
| `x_base` | `(4, 5H, 5W)` | the observation-determined baseline |
| `prior` | `(4, 5H, 5W)` | structure added by the learned prior; `x_base + prior == image` |
| `gsd` | float | output grid spacing in metres |
| `metadata` | dict | scan backend, precision, tiling, offset, SCL source, regularisation weights |

## Helpers

```python
import synapse_sr
result = synapse_sr.super_resolve("scene.tif")

result.rgb()          # (5H, 5W, 3) uint8 true-colour quicklook, 2-98 % stretch
result.ndvi()         # (5H, 5W) NDVI, NaN where invalid
result.to_xarray()    # DataArray (band, y, x) with map coordinates and CRS (needs xarray; rioxarray optional)
result.to_xarray(dataset=True)   # Dataset with every layer: reflectance, x_base, prior, support, valid, context ...
result.save("out.tif")
result.save("out.npz")           # everything, including x_base, prior and context
```

## Reopening a result

```python
from synapse_sr import Result

r = Result.load("out.tif")       # image, error scale, support, valid, georeference, consistency, metadata
r = Result.load("out.npz")       # all of the above plus x_base, prior and context
r.indices()["ndvi"]; r.interval(0.9)
```

## Saved GeoTIFF

| Band | Name | Content |
|---|---|---|
| 1 to 4 | `B04` `B03` `B02` `B08` | reflectance, float32, NaN where invalid |
| 5 to 8 | `ERRSCALE_B04` ... `ERRSCALE_B08` | predicted error scale |
| 9 | `SUPPORT` | 2 / 1 / 0 |

NaN is declared as the file's NoData value, so QGIS, GDAL and rasterio mask invalid pixels automatically
(`ERRSCALE_*` is NaN there too). The run's consistency, metadata and calibration are stored in the `SYNAPSE_SR`
tag, which `Result.load` reads back.

`save(path, with_confidence=False)` (CLI `--no-confidence`) writes bands 1 to 4 only. A path ending in `.npz`
writes a compressed NumPy archive with every layer; array inputs without georeferencing can only be saved that way.

The CRS is copied from the input. The geotransform keeps the input origin and divides the pixel size by five,
so the output covers exactly the input bounds. Output pixel `5i + r` lies inside input pixel `i`, and `r = 2` is
centred on it.
