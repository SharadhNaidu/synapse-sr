# Quick start

## One call

```python
import synapse_sr

result = synapse_sr.super_resolve("sentinel2_l2a.tif")
result.save("sentinel2_2m.tif")
```

`synapse_sr.super_resolve` loads the default model once, caches it, and processes the scene. Keyword arguments
go to [`Pro.super_resolve`](api.md#synapse_sr.Pro.super_resolve).

## Keep a model object

For several scenes, or to choose the device and weights explicitly:

```python
from synapse_sr import Pro

model = Pro.from_pretrained(device="cuda")        # or weights="synapse-pro-v2.safetensors"
for path in ["a.tif", "b.tif", "c.tif"]:
    model.super_resolve(path).save(path.replace(".tif", "_2m.tif"))
```

## From download to result

```python
from synapse_sr import Pro, fetch_sentinel2          # pip install "synapse-sr[stac]"

scene = fetch_sentinel2(lat=12.9237, lon=77.4987, start="2025-01-01", end="2025-03-15", size_m=2000)
result = Pro.from_pretrained().super_resolve(scene)  # offset and cloud mask picked up automatically

result.image        # (4, 1000, 1000) reflectance, B04 B03 B02 B08
result.support      # (1000, 1000) 2 HIGH, 1 MEDIUM, 0 LOW / invalid
result.consistency  # {"B04": 0.95, ...} round trip against the input, in noise units
```

## Look at it

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(result.rgb());                         ax[0].set_title("synapse-sr 2 m")
ax[1].imshow(result.ndvi(), cmap="RdYlGn");        ax[1].set_title("NDVI")
ax[2].imshow(result.support, cmap="gray", vmin=0, vmax=2); ax[2].set_title("support")
for a in ax: a.axis("off")
```

## Command line

```bash
synapse-sr scene.tif scene_2m.tif
synapse-sr scene.tif scene_2m.tif --device cpu --weights synapse-pro-v2.safetensors
```

Next: [Inputs and preprocessing](guide/inputs.md).
