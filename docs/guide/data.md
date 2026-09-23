# Getting Sentinel-2 data

## fetch_sentinel2

```bash
pip install "synapse-sr[stac]"
```

```python
from synapse_sr import fetch_sentinel2

path = fetch_sentinel2(lat=30.935, lon=75.800, start="2024-11-01", end="2025-01-31", size_m=3000)
```

`fetch_sentinel2` searches the public [Earth Search](https://earth-search.aws.element84.com/v1) STAC catalogue.
It picks the least-cloudy L2A scene over the point and writes:

- a 10-band GeoTIFF in model order, with named bands, on the 10 m grid (20 m bands nearest-resampled);
- the offset in the `BOA_ADD_OFFSET` tag, plus the STAC item id and cloud cover;
- `<name>_scl.tif`, the scene classification layer.

`super_resolve` picks up the offset and the SCL file automatically. Pass `api=` to use another STAC API that
serves the `sentinel-2-l2a` collection with Earth Search asset names.

## Other sources

| Source | Notes |
|---|---|
| Copernicus Data Space (SAFE) | stack B02-B12 onto the 10 m grid with nearest resampling; pass `offset=-1000` for baseline 04.00 or later |
| Google Earth Engine `COPERNICUS/S2_SR_HARMONIZED` | export at `scale=10`; the offset is already removed |
| `cubo` or `stackstac` cubes | pass `cube.isel(time=k)` directly; band names are read from the `band` coordinate |
| `odc-stac` datasets | pass `ds.isel(time=k).to_array("band")` |
