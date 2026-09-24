<p align="center">
  <img src="assets/logo-wordmark.svg#only-light" alt="synapse-sr" width="440">
  <img src="assets/logo-wordmark-dark.svg#only-dark" alt="synapse-sr" width="440">
</p>

<p align="center"><b>Sentinel-2 at 2 m, with every pixel accounted for.</b></p>

<p align="center">
  <img src="assets/gifs/rv_university.gif" alt="RV University, Bengaluru: Sentinel-2 10 m versus synapse-sr 2 m" width="460">
</p>

**synapse-sr** turns a Sentinel-2 L2A scene into a 2.0 m red, green, blue and near-infrared GeoTIFF. A physical
model of the instrument pins everything the satellite measured. The network only adds what 10 m pixels cannot
show, and every result says which is which.

```python
import synapse_sr

r = synapse_sr.super_resolve("sentinel2_l2a.tif")
r.save("sentinel2_2m.tif")
```

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharadhNaidu/synapse-sr/blob/main/notebooks/quickstart.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/SharadhNaidu/synapse-sr/blob/main/notebooks/quickstart.ipynb)

## What you can do with it

<div class="grid cards" markdown>

- **Crop monitoring**

    NDVI, SAVI, EVI and red-edge NDRE at 2 m, field-boundary maps, statistics restricted to
    observation-determined pixels.

    `r.indices()["ndvi"]` · `synapse_sr.boundaries(r, "field")`

- **Urban analysis**

    Building and road edges, built-up index, per-building detection.

    `synapse_sr.boundaries(r, "urban")` · `r.indices()["ndbi"]`

- **Water and floods**

    2 m water masks, shoreline strength, flooded area between two dates in km².

    `synapse_sr.change(before, after, "ndwi")`

- **Disaster assessment**

    Burn scars, landslides and debris. Change is flagged only where both dates are trustworthy.

    `synapse_sr.change(before, after, "nbr")`

</div>

Worked examples: [Applications](applications.md).

## Pro or Flash

| | **Pro** (default) | **Flash** |
|---|---|---|
| Network | 14.4 M-parameter Mamba state-space model | ~0.6 M-parameter CNN |
| Runs best on | GPU (Colab, Kaggle, workstations) | any CPU, laptops, integrated graphics, Apple silicon, ARM |
| Guarantees | observation-consistent, support map, calibrated uncertainty | the same |

More: [Choosing a model and a device](guide/models.md).

## Why trust it

| | |
|---|---|
| **Observation-consistent by construction** | `x_hat = x_base + P_N(delta)`: the network's contribution lies in the null space of the Sentinel-2 forward model, so re-observing the output reproduces the measurement. |
| **Accountable** | per-pixel support (observation-determined versus prior-dominated), calibrated uncertainty, validity mask and a measured round-trip consistency on every result. |
| **Geospatially exact** | GeoTIFF in, GeoTIFF out, CRS and bounds preserved, direct x5 onto a grid that shares the input origin. |
| **Runs anywhere** | CUDA, Triton, or pure PyTorch on CPU; live progress bars in terminals and notebooks; fully offline once the weights are local. |

!!! note "Grid spacing is not effective resolution"
    The output grid is 2.0 m. How fine a structure is genuinely resolved is a separate, measured property;
    see [Limitations](limitations.md).
