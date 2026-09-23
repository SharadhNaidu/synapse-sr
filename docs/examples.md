# Examples

Every example below was produced with the package itself, using `tools/make_gifs.py`:
`fetch_sentinel2` to get the scene, `Pro.super_resolve` to process it. Left: the Sentinel-2 L2A 10 m input,
shown with nearest-neighbour pixels and no smoothing. Right: the synapse-sr 2 m output. Both halves use the
same true-colour stretch. Each area is 1.28 km x 1.28 km.

!!! info "Research preview"
    These images come from the v0.1 preview checkpoint. They show what the package does. They do not
    validate an effective resolution; see [Limitations](limitations.md).

## RV University, Bengaluru

<img src="../assets/gifs/rv_university.gif" alt="RV University, Bengaluru" width="480">

Campus and surrounding blocks on Mysore Road, 12.924 N 77.499 E. Sentinel-2B, 6 February 2025, cloud cover
below 0.01 %.

## Bengaluru city centre: urban

<img src="../assets/gifs/bengaluru_urban.gif" alt="Bengaluru city centre" width="480">

Dense urban fabric around 12.972 N 77.595 E. Sentinel-2B, 6 February 2025.

## Ludhiana, Punjab: agriculture

<img src="../assets/gifs/punjab_fields.gif" alt="Ludhiana fields" width="480">

Field parcels and boundaries around 30.935 N 75.800 E. Sentinel-2A, 2 December 2024.

## Wayanad, Kerala: landslide-affected hills

<img src="../assets/gifs/wayanad_landslide.gif" alt="Wayanad landslide" width="480">

Terrain in the area of the 30 July 2024 Wayanad landslide, 11.475 N 76.135 E. Sentinel-2C,
24 February 2025.

## Reproduce

```bash
pip install "synapse-sr[stac]" pillow
python tools/make_gifs.py --weights synapse-pro-v2.safetensors --out gifs
```

## Recipes

### NDVI at 2 m for a field survey

```python
import numpy as np
from synapse_sr import Pro, fetch_sentinel2

scene = fetch_sentinel2(30.935, 75.800, "2024-11-01", "2025-01-31", size_m=3000)
r = Pro.from_pretrained().super_resolve(scene)
ndvi = r.ndvi()
trusted = ndvi[r.support == 2]            # observation-determined pixels only
print(np.nanmean(trusted))
```

### Change between two dates

```python
before = model.super_resolve("before.tif")
after = model.super_resolve("after.tif")
change = after.ndvi() - before.ndvi()
loss = (change < -0.2) & before.valid & after.valid
print("vegetation loss km2:", loss.sum() * (before.gsd ** 2) / 1e6)
```

### Export only reflectance for a GIS

```python
model.super_resolve("scene.tif").save("scene_2m.tif", with_confidence=False)
```
