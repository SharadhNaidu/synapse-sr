# Inputs and preprocessing

## Accepted inputs

| Input | Example |
|---|---|
| GeoTIFF path | `model.super_resolve("scene.tif")` |
| numpy array `(C, H, W)` | `model.super_resolve(arr, band_names=[...])` |
| torch tensor `(C, H, W)` or `(1, C, H, W)` | `model.super_resolve(tensor)` |
| xarray DataArray with a `band` coordinate | `model.super_resolve(cube.isel(time=0))` |

DN (L2A integer counts) and reflectance (floats up to about 1) are both accepted; reflectance is detected
automatically. NaN and infinite values are treated as NoData.

## Bands

The model uses ten bands:

| Role | Bands | Native resolution |
|---|---|---|
| Super-resolved | B04, B03, B02, B08 | 10 m |
| Spectral context | B05, B06, B07, B8A, B11, B12 | 20 m |

The bands are identified as follows:

1. From `band_names`, or from the GeoTIFF band descriptions. Names are case-insensitive, and `B2` and `B02` are equivalent.
2. Otherwise, from the channel count:
    - 10 channels are read as the order above;
    - 12 channels as the L2A order B01 to B12 without B10;
    - 13 channels as the L1C order.

Any other layout raises a `ValueError` that names the accepted orders.

The 20 m bands should be on the 10 m grid with nearest-neighbour resampling. This is how Earth Engine,
`fetch_sentinel2` and `gdalwarp -r near` deliver them. They inform the network but are not returned at 2 m.

## Radiometry

Reflectance is `(DN + offset) / 10000`. Products from processing baseline 04.00 onward, from January 2022,
carry `BOA_ADD_OFFSET = -1000`:

| Source | What to pass |
|---|---|
| `fetch_sentinel2` output | nothing: the offset is stored in the `BOA_ADD_OFFSET` tag and applied automatically |
| Earth Engine `COPERNICUS/S2_SR_HARMONIZED` | nothing: the offset is already removed |
| SAFE / JP2 from the Copernicus Data Space, baseline 04.00 or later | `offset=-1000` (CLI: `--offset -1000`) |

## Masking

| Source | Masked |
|---|---|
| NoData | the raster NoData value, NaN / inf, all-zero pixels |
| Scene classification (SCL) | classes 0 no data, 1 saturated or defective, 3 cloud shadow, 8 and 9 cloud, 10 cirrus |

```python
result = model.super_resolve("scene.tif", scl="scene_SCL.tif")   # 10 m or 20 m SCL
result = model.super_resolve("scene.tif")                        # uses scene_scl.tif next to the input if present
result = model.super_resolve("scene.tif", scl=None)              # no SCL masking
```

Masked pixels are written as NaN and receive support class 0. No heuristic cloud detection is performed.

## Grid checks

GeoTIFF inputs must be north-up with square pixels of about 10 m. Rotated or sheared transforms, non-square
pixels and other resolutions are rejected with an explanatory error. They are not silently resampled.
