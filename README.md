<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/SharadhNaidu/synapse-sr/main/docs/assets/logo-wordmark-dark.svg">
    <img src="https://raw.githubusercontent.com/SharadhNaidu/synapse-sr/main/docs/assets/logo-wordmark-card.png" alt="synapse-sr" width="440">
  </picture>
</p>

<h3 align="center">Sentinel-2 at 2 m, with every pixel accounted for</h3>

<p align="center">
  <a href="https://pypi.org/project/synapse-sr/"><img src="https://img.shields.io/pypi/v/synapse-sr?color=black" alt="PyPI"></a>
  <a href="https://github.com/SharadhNaidu/synapse-sr/actions/workflows/tests.yml"><img src="https://github.com/SharadhNaidu/synapse-sr/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <a href="https://sharadhnaidu.github.io/synapse-sr/"><img src="https://img.shields.io/badge/docs-online-black" alt="docs"></a>
  <a href="https://colab.research.google.com/github/SharadhNaidu/synapse-sr/blob/main/notebooks/quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
  <a href="https://kaggle.com/kernels/welcome?src=https://github.com/SharadhNaidu/synapse-sr/blob/main/notebooks/quickstart.ipynb"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/SharadhNaidu/synapse-sr/main/docs/assets/gifs/rv_university.gif" alt="RV University, Bengaluru: Sentinel-2 10 m and synapse-sr 2 m" width="440">
</p>

**synapse-sr** turns a Sentinel-2 L2A scene into a 2.0 m red / green / blue / near-infrared GeoTIFF. A physical model
of the instrument pins everything the satellite measured. The network only adds what 10 m pixels cannot show, and
every output says which pixels came from the measurement and which came from the learned prior.

```bash
pip install synapse-sr
```

```python
import synapse_sr

r = synapse_sr.super_resolve("sentinel2_l2a.tif")   # progress bar, then a 5x larger result
r.save("sentinel2_2m.tif")                           # georeferenced, same CRS and bounds
r.summary()                                          # size, consistency, support, time
```

```bash
synapse-sr sentinel2_l2a.tif sentinel2_2m.tif        # the same from the shell
```

## What it is for

| | One line | |
|---|---|---|
| **Crop monitoring** | `r.indices()["ndvi"]`, `synapse_sr.boundaries(r, "field")` | NDVI, SAVI, EVI, red-edge NDRE, field edges at 2 m |
| **Urban analysis** | `synapse_sr.boundaries(r, "urban")`, `r.indices()["ndbi"]` | buildings, roads, built-up index |
| **Water and floods** | `synapse_sr.change(before, after, "ndwi")` | flooded area in km², shoreline strength |
| **Disaster assessment** | `synapse_sr.change(before, after, "nbr")` | burn scars, landslides, debris; changes flagged only where both dates are trustworthy |

Worked examples for each: [Applications](https://sharadhnaidu.github.io/synapse-sr/applications/).

## Pro or Flash

| | **Pro** (default) | **Flash** |
|---|---|---|
| Network | 14.4 M-parameter state-space (Mamba) model | ~0.6 M-parameter convolutional model |
| Best on | GPU: Colab, Kaggle, workstations | any CPU, laptops, integrated graphics, Apple silicon, ARM |
| Load | `Pro.from_pretrained()` | `Flash.from_pretrained()` (weights not released yet) |
| Physics guarantees | identical | identical |

Both share the same observation-consistent pipeline, so the same guarantees hold for either one. **Flash weights
are not released yet.** Until they are, use Pro, which also runs on CPU (more slowly). `synapse-sr --models` shows
what is published.

## Why trust the output

```python
r.x_base        # what the 10 m observation determines
r.prior         # what the network added (x_base + prior == image), invisible to the sensor
r.support       # per pixel: 2 observation-determined, 1 medium, 0 prior-dominated or invalid
r.uncertainty() # calibrated expected error per pixel and band
r.consistency   # re-observing the output reproduces the input, in sensor-noise units
```

`x_hat = x_base + P_N(delta)`: `P_N` removes every component the sensor could have seen from the network's
output, so the network cannot contradict the measurement. See [How it works](https://sharadhnaidu.github.io/synapse-sr/method/).

## Runs everywhere

| Platform | What you get |
|---|---|
| Colab / Kaggle GPU | Pro with a Triton scan kernel, compiled on first use (no extra installs) |
| Linux GPU + `mamba-ssm` | fused CUDA kernel (`pip install "synapse-sr[cuda]"`) |
| Windows / macOS / CPU-only / ARM | exact PyTorch path; Flash recommended |
| Offline / air-gapped | `Pro.from_pretrained(weights="model.safetensors")` |

`synapse-sr --env` shows what is active on your machine.

## Documentation

[Quick start](https://sharadhnaidu.github.io/synapse-sr/quickstart/) ·
[Colab and Kaggle](https://sharadhnaidu.github.io/synapse-sr/guide/notebooks/) ·
[Applications](https://sharadhnaidu.github.io/synapse-sr/applications/) ·
[Examples](https://sharadhnaidu.github.io/synapse-sr/examples/) ·
[API](https://sharadhnaidu.github.io/synapse-sr/api/) ·
[Limitations](https://sharadhnaidu.github.io/synapse-sr/limitations/)

**Limitations, in short.** A 2 m grid is not 2 m effective resolution; no effective-resolution figure is claimed
for this release. Only B04, B03, B02 and B08 are super-resolved; the 20 m bands are spectral context. The training
references are from the United States.

<details>
<summary>Citation and acknowledgements</summary>

```bibtex
@software{synapse_sr,
  title  = {synapse-sr: observation-consistent super-resolution of Sentinel-2 imagery},
  author = {Naidu, Sharadh},
  year   = {2026},
  url    = {https://github.com/SharadhNaidu/synapse-sr}
}
```

- The Mamba backbone and the Flash building blocks derive from [SEN2SR](https://github.com/ESAOpenSR/SEN2SR)
  (ESA OpenSR, CC0-1.0); see `THIRD_PARTY_NOTICES`.
- The optional fused kernel comes from [mamba-ssm](https://github.com/state-spaces/mamba) (Apache-2.0).
- Sentinel-2 data: Copernicus programme, European Space Agency.

</details>
