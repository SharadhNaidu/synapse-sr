# Quick start

!!! tip "No installation needed to try it"
    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharadhNaidu/synapse-sr/blob/main/notebooks/quickstart.ipynb)
    [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/SharadhNaidu/synapse-sr/blob/main/notebooks/quickstart.ipynb)
    runs the whole tour below on a free GPU. See [Colab and Kaggle](guide/notebooks.md).

## 1. Install

```bash
pip install "synapse-sr[stac]"      # [stac] adds the Sentinel-2 downloader used below
synapse-sr --env                    # what hardware and kernels will be used
```

## 2. Get a scene

```python
import synapse_sr

scene = synapse_sr.fetch_sentinel2(lat=12.9237, lon=77.4987,          # RV University, Bengaluru
                                   start="2025-01-01", end="2025-03-15", size_m=1280)
```

`scene` is a 10-band GeoTIFF on the 10 m grid, with its cloud mask saved next to it and applied automatically.
Any Sentinel-2 L2A GeoTIFF of your own works the same way (see [Inputs](guide/inputs.md)).

## 3. Super-resolve

```python
r = synapse_sr.super_resolve(scene)
```

In a terminal or notebook you see a live progress bar and then a one-line summary:

```text
✓ synapse-pro-v2: 640×640 px at 2 m in 80.3 s · consistency ≤ 0.99 τ · 22% observation-determined
```

That line was measured on a laptop GPU (RTX 4070, Windows, PyTorch scan). The same scene takes 5.2 s on an
A100 slice with the Triton kernel, which is the kind of GPU runtime Colab and Kaggle give you.

`synapse_sr.super_resolve` loads the default model (Pro) once and caches it. To choose the model and the device:

```python
from synapse_sr import Pro, Flash

model = Pro.from_pretrained(device="cuda")          # most accurate; GPU recommended
model = Flash.from_pretrained(device="cpu")         # fast on any CPU, laptop or ARM machine
r = model.super_resolve(scene)
```

## 4. Look at it

```python
r.summary()                                          # table: size, backend, consistency, support, time
r.show(["image", "x_base", "prior", "support"])      # needs matplotlib
```

| Panel | Meaning |
|---|---|
| `image` | the 2 m result |
| `x_base` | what the 10 m observation alone determines |
| `prior` | what the network added; invisible to the sensor, so the observation neither confirms nor contradicts it |
| `support` | green: observation-determined · amber: medium · red: prior-dominated or masked |

## 5. Save it

```python
r.save("rvu_2m.tif")     # GeoTIFF, same CRS and bounds: B04 B03 B02 B08, ERRSCALE x4, SUPPORT
```

It opens directly in QGIS or ArcGIS, or with `rasterio`.

## 6. Use it

```python
idx = r.indices()                                    # ndvi savi evi gndvi ndwi, + ndre ndbi nbr mndwi
fields = synapse_sr.boundaries(r, "field")           # crop-parcel edges
urban = synapse_sr.boundaries(r, "urban")            # building and road edges
flood = synapse_sr.change(before, after, "ndwi")     # two dates -> changed area in km²
err = r.uncertainty()                                # calibrated expected error per pixel
```

See [Applications](applications.md) for crop monitoring, urban analysis, water and disaster assessment.

## Command line

```bash
synapse-sr scene.tif scene_2m.tif                    # Pro, progress bar, summary table
synapse-sr scene.tif scene_2m.tif --model flash --device cpu
synapse-sr scene.tif scene_2m.tif --json             # machine-readable summary on stdout
synapse-sr --models                                  # what is published
```

Next: [Choosing a model and a device](guide/models.md) · [Inputs and preprocessing](guide/inputs.md).
