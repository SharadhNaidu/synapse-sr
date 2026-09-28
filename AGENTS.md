# AGENTS.md

Guidance for AI coding agents working with or on synapse-sr.

## Using the package

```python
import synapse_sr
r = synapse_sr.super_resolve("scene.tif")              # Sentinel-2 L2A GeoTIFF -> 2 m RGBN Result (Flash model)
r.save("scene_2m.tif")                                 # GeoTIFF; r.save(..., cog=True) for a COG
```

- `model="pro"` selects Pro (most detail, GPU recommended); the default Flash runs anywhere, CPU included.
- Inputs: GeoTIFF path, numpy / torch `(C, H, W)`, or xarray with a `band` coordinate; DN or reflectance; band order
  from names or channel count (10, 12, 13). Pass `band_names=` for arrays with an unusual order.
- Outputs (`Result`): `image` (4, 5H, 5W) reflectance B04 B03 B02 B08; `support`, `valid`, `x_base`, `prior`,
  `uncertainty()`, `interval()`, `indices()`, `summary()`, `quicklook()`.
- Applications: `synapse_sr.change(before, after, "ndwi")`, `synapse_sr.boundaries(r, "field")`.
- Batches: `synapse_sr.super_resolve_folder(src_dir, out_dir)`; CLI `synapse-sr in/ out/`.
- Data: `synapse_sr.fetch_sentinel2(lat, lon, start, end)` (needs `pip install synapse-sr[stac]`).
- Full docs in one file: https://sharadhnaidu.github.io/synapse-sr/llms-full.txt

## Developing

```bash
pip install -e ".[test,all]"
pytest -q                                   # no weights or network needed
python tools/stress.py --model flash --device cpu   # published weights, ~30 s
```

- Every fast path has a test against its exact reference (`tests/test_engine.py`).
- Commit titles: three lowercase words, no body.
