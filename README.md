<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/SharadhNaidu/synapse-sr/main/docs/assets/logo-wordmark-dark.svg">
    <img src="https://raw.githubusercontent.com/SharadhNaidu/synapse-sr/main/docs/assets/logo-wordmark-card.png" alt="SYNAPSE-SR, Sentinel-2 super-resolution" width="440">
  </picture>
</p>

<h1 align="center">SYNAPSE-SR: Sentinel-2 super-resolution to 2 m</h1>

<h3 align="center">Sentinel-2 at 2 m, with every pixel accounted for</h3>

<p align="center">
  <a href="https://pypi.org/project/synapse-sr/"><img src="https://img.shields.io/pypi/v/synapse-sr?color=black" alt="PyPI"></a>
  <a href="https://github.com/SharadhNaidu/synapse-sr/actions/workflows/tests.yml"><img src="https://github.com/SharadhNaidu/synapse-sr/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <a href="https://sharadhnaidu.github.io/synapse-sr/"><img src="https://img.shields.io/badge/docs-online-black" alt="docs"></a>
  <a href="https://huggingface.co/SharadhNaiduTrains/synapse-sr"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20weights-Hugging%20Face-black" alt="Hugging Face weights"></a>
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
result = synapse_sr.super_resolve("scene.tif")

import synapse_sr

r = synapse_sr.super_resolve("sentinel2_l2a.tif")   # progress bar, then a 5x larger result
r.save("sentinel2_2m.tif")                           # georeferenced, same CRS and bounds
r.summary()                                          # size, consistency, support, time
```

```bash
synapse-sr sentinel2_l2a.tif sentinel2_2m.tif        # the same from the shell
synapse-sr --fetch 12.92,77.50 --dates 2025-01-01:2025-03-15 out.tif --preview preview.png   # download + run
synapse-sr scenes/ scenes_2m/ --cog                  # a whole folder, as Cloud-Optimised GeoTIFFs
```

Every common task on one page: **[cheat sheet](https://sharadhnaidu.github.io/synapse-sr/cheatsheet/)**. AI assistants
and agents can read the whole documentation from [`llms.txt`](https://sharadhnaidu.github.io/synapse-sr/llms.txt).

## Benchmarks

Official [opensr-test](https://github.com/ESAOpenSR/opensr-test) protocol (Aybar et al.): Sentinel-2 L2A input, harmonised
high-resolution references, `opensr_test.Metrics()` defaults, mean over its five datasets (NAIP, SPOT, Spain urban,
Spain crops, VENµS; 178 scenes). Every model at its native scale, compared on the reference grid.

| Model | Improvement ↑ | Omission ↓ | Hallucination ↓ | Detail corr. ↑ | RMSE ↓ | Spectral error ↓ | Reflectance error ↓ |
|---|---|---|---|---|---|---|---|
| **SYNAPSE Flash** | 0.155 | 0.748 | 0.097 | **0.300** | **0.0234** | 0.401 | 0.0018 |
| **SYNAPSE Pro** | **0.199** | 0.631 | 0.171 | 0.289 | 0.0254 | **0.223** | **0.0011** |
| SEN2SR | 0.150 | 0.759 | 0.091 | 0.284 | 0.0235 | 0.665 | 0.0025 |
| SEN2SR-Lite | 0.152 | 0.749 | 0.099 | 0.290 | **0.0234** | 0.463 | 0.0019 |
| LDSR-S2 | 0.197 | 0.599 | 0.204 | 0.206 | 0.0240 | 1.015 | 0.0036 |
| Satlas ESRGAN | 0.129 | **0.181** | 0.690 | 0.089 | 0.0443 | 7.787 | 0.0242 |
| Bicubic | 0.102 | 0.830 | **0.068** | 0.279 | **0.0234** | 0.601 | 0.0028 |

**Best** in bold. SYNAPSE runs with the package defaults (`synapse_sr.super_resolve`). SYNAPSE is best on five of the
seven columns: Pro on improvement, spectral and reflectance error, Flash on detail correlation and (tied) RMSE. Flash processes
a 1.28 km scene in about 1 s on a laptop CPU; Pro in about 5 s on a GPU.

## What it is for

| | One line | |
|---|---|---|
| **Crop monitoring** | `r.indices()["ndvi"]`, `synapse_sr.boundaries(r, "field")` | NDVI, SAVI, EVI, red-edge NDRE, field edges at 2 m |
| **Urban analysis** | `synapse_sr.boundaries(r, "urban")`, `r.indices()["ndbi"]` | buildings, roads, built-up index |
| **Water and floods** | `synapse_sr.change(before, after, "ndwi")` | flooded area in km², shoreline strength |
| **Disaster assessment** | `synapse_sr.change(before, after, "nbr")` | burn scars, landslides, debris; changes flagged only where both dates are trustworthy |

Worked examples for each: [Applications](https://sharadhnaidu.github.io/synapse-sr/applications/).

## Flash or Pro

| | **Flash** (default) | **Pro** |
|---|---|---|
| Network | ~0.6 M-parameter re-parameterised CNN, distilled from Pro | 14.4 M-parameter state-space (Mamba) model |
| Speed (1.28 km scene) | ~1 s on a laptop CPU; 10 km x 10 km in ~14 s | ~5 s on a GPU (Colab / Kaggle T4, workstations) |
| Runs on | any CPU, laptops, integrated graphics, Apple silicon, ARM, any GPU | GPU recommended; CPU works, slower |
| Choose it | `synapse_sr.super_resolve(src)` | `synapse_sr.super_resolve(src, model="pro")` |
| Load | `Flash.from_pretrained()` | `Pro.from_pretrained()` |
| Tuned for | balanced fidelity: best detail correlation, lowest RMSE (tied), measured 10 m mean reflectance restored exactly | maximum detail: best improvement, spectral and reflectance error |
| Physics, support map, calibrated uncertainty | yes | yes |

Both run the same observation-consistent pipeline. Use **Flash** for speed, anywhere-deployment and the most faithful
output, **Pro** for the most recovered detail on a GPU. `synapse-sr --models` lists
every published checkpoint.

## Why trust the output

```python
import synapse_sr
r = synapse_sr.super_resolve("scene.tif")

r.x_base        # what the 10 m observation determines
r.prior         # what the network added (x_base + prior == image), invisible to the sensor
r.support       # per pixel: 2 observation-determined, 1 medium, 0 prior-dominated or invalid
r.uncertainty() # calibrated expected error per pixel and band
r.consistency   # how closely re-observing the output reproduces the input, in sensor-noise units
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
[API](https://sharadhnaidu.github.io/synapse-sr/api/)

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

- Third-party components and their licences are listed in `THIRD_PARTY_NOTICES`.
- The optional fused kernel comes from [mamba-ssm](https://github.com/state-spaces/mamba) (Apache-2.0).
- Sentinel-2 data: Copernicus programme, European Space Agency.

</details>
