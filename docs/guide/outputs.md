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
result.rgb()          # (5H, 5W, 3) uint8 true-colour quicklook, 2-98 % stretch
result.ndvi()         # (5H, 5W) NDVI, NaN where invalid
result.to_xarray()    # DataArray (band, y, x) with map coordinates (needs xarray)
result.save("out.tif")
```

## Saved GeoTIFF

| Band | Name | Content |
|---|---|---|
| 1 to 4 | `B04` `B03` `B02` `B08` | reflectance, float32, NaN where invalid |
| 5 to 8 | `ERRSCALE_B04` ... `ERRSCALE_B08` | predicted error scale |
| 9 | `SUPPORT` | 2 / 1 / 0 |

`save(path, with_confidence=False)` (CLI `--no-confidence`) writes bands 1 to 4 only. For array inputs without
georeferencing, `save` writes a compressed `.npz`.

The CRS is copied from the input. The geotransform keeps the input origin and divides the pixel size by five,
so the output covers exactly the input bounds. Output pixel `5i + r` lies inside input pixel `i`, and `r = 2` is
centred on it.
