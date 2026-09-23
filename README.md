<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/SharadhNaidu/synapse-sr/main/docs/assets/logo-wordmark-dark.svg">
    <img src="https://raw.githubusercontent.com/SharadhNaidu/synapse-sr/main/docs/assets/logo-wordmark-card.png" alt="synapse-sr" width="480">
  </picture>
</p>

<h3 align="center">Observation-consistent super-resolution of Sentinel-2 imagery to a 2 m grid</h3>

<p align="center">
  <a href="https://github.com/SharadhNaidu/synapse-sr/actions/workflows/tests.yml"><img src="https://github.com/SharadhNaidu/synapse-sr/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <a href="https://sharadhnaidu.github.io/synapse-sr/"><img src="https://img.shields.io/badge/docs-online-black" alt="docs"></a>
  <a href="https://pypi.org/project/synapse-sr/"><img src="https://img.shields.io/pypi/v/synapse-sr?color=black" alt="PyPI"></a>
  <a href="https://huggingface.co/SharadhNaiduTrains/synapse-sr"><img src="https://img.shields.io/badge/weights-Hugging%20Face-black" alt="weights"></a>
  <img src="https://img.shields.io/badge/python-3.8%2B-black" alt="python">
  <a href="https://github.com/SharadhNaidu/synapse-sr/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-CC0--1.0-black" alt="license"></a>
</p>

<p align="center">
  <b><a href="https://sharadhnaidu.github.io/synapse-sr/">Documentation</a></b> |
  <b><a href="https://sharadhnaidu.github.io/synapse-sr/quickstart/">Quick start</a></b> |
  <b><a href="https://sharadhnaidu.github.io/synapse-sr/examples/">Examples</a></b> |
  <b><a href="https://sharadhnaidu.github.io/synapse-sr/api/">API</a></b>
</p>

---

**synapse-sr** turns a Sentinel-2 L2A scene into a 2.0 m red, green, blue and near-infrared product. The network
adds only the structure the 10 m measurement cannot see. Everything the satellite did observe is pinned by a
physical model of the instrument and cannot be changed. Every result reports how much of each pixel came from
the observation and how much from the learned prior.

<p align="center">
  <img src="https://raw.githubusercontent.com/SharadhNaidu/synapse-sr/main/docs/assets/gifs/rv_university.gif" alt="RV University, Bengaluru" width="480">
  <br><sub>RV University, Bengaluru. Sentinel-2 L2A 10 m (left) and synapse-sr 2 m (right), 6 February 2025.</sub>
</p>

## Table of contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [From download to 2 m](#from-download-to-2-m)
- [Numpy, torch and xarray inputs](#numpy-torch-and-xarray-inputs)
- [Large scenes](#large-scenes)
- [Support, confidence and consistency](#support-confidence-and-consistency)
- [Applications](#applications)
- [Calibrated uncertainty](#calibrated-uncertainty)
- [Command line](#command-line)
- [Offline use](#offline-use)
- [Examples](#examples)
- [How it works](#how-it-works)
- [Limitations](#limitations)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Overview

| | |
|---|---|
| **Observation-consistent** | `x_hat = x_base + P_N(delta)`: the learned correction lives in the null space of the Sentinel-2 forward operator, so re-observing the output reproduces the input |
| **Accountable** | per-pixel error scale, HIGH / MEDIUM / LOW support classes, validity mask, and a measured round-trip consistency on every result |
| **Direct x5** | one network from 10 m to a 2 m grid; the centre sub-pixel sits on the source-pixel centre, and the grid origin is preserved |
| **Geospatial I/O** | GeoTIFF in, GeoTIFF out; CRS and bounds preserved; any scene size via seamless tiling; NoData and SCL cloud masking |
| **Easy** | one Python call or one shell command; numpy, torch or xarray inputs; automatic band mapping and radiometric offset; fully offline once the weights are local |

## Installation

```bash
pip install synapse-sr
```

| Extra | Adds |
|---|---|
| `pip install "synapse-sr[stac]"` | `fetch_sentinel2` scene download |
| `pip install "synapse-sr[xarray]"` | xarray input and `Result.to_xarray()` |
| `pip install "synapse-sr[cuda]"` | fused `mamba-ssm` CUDA kernel (optional; must match your PyTorch/CUDA build) |

Without the fused kernel, a PyTorch implementation is used. It matches the fused kernel to a relative error
of 1.7e-7 and is slower. Run `synapse-sr --env` to see what is active. Full details:
[Installation](https://sharadhnaidu.github.io/synapse-sr/installation/).

## Quick start

```python
import synapse_sr

result = synapse_sr.super_resolve("sentinel2_l2a.tif")
result.save("sentinel2_2m.tif")
```

With an explicit model object:

```python
from synapse_sr import Pro

model = Pro.from_pretrained(device="cuda")        # weights="model.safetensors" for a local file
result = model.super_resolve("sentinel2_l2a.tif")

result.image          # (4, 5H, 5W) reflectance, B04 B03 B02 B08, 2.0 m grid
result.support        # (5H, 5W) 2 HIGH, 1 MEDIUM, 0 LOW / invalid
result.consistency    # {"B04": 0.95, ...} round trip against the input, in noise units
```

## From download to 2 m

```python
from synapse_sr import Pro, fetch_sentinel2       # pip install "synapse-sr[stac]"

scene = fetch_sentinel2(lat=12.9237, lon=77.4987, start="2025-01-01", end="2025-03-15", size_m=2000)
result = Pro.from_pretrained().super_resolve(scene)   # radiometric offset and cloud mask applied automatically

import matplotlib.pyplot as plt
plt.imshow(result.rgb()); plt.axis("off")
```

## Numpy, torch and xarray inputs

```python
model.super_resolve(array)                                   # (C, H, W) DN or reflectance, 10/12/13-band order
model.super_resolve(array, band_names=["B04", "B03", ...])   # any order, by name
model.super_resolve(tensor)                                  # torch.Tensor (C, H, W) or (1, C, H, W)
model.super_resolve(cube.isel(time=0))                       # xarray with a "band" coordinate (cubo, stackstac)
```

The model uses B04, B03 and B02 (10 m), B08 (10 m), and B05, B06, B07, B8A, B11 and B12 (20 m, as spectral
context). See [Inputs and preprocessing](https://sharadhnaidu.github.io/synapse-sr/guide/inputs/).

## Large scenes

Scenes of any size and shape are tiled with a real-context halo and assembled on the output grid. The output
is always exactly `5H x 5W`, with no dropped regions. Tiled and single-window results agree to 0.4 % relative
RMS.

```python
result = model.super_resolve("large_scene.tif", tile=64, halo=16)
```

## Support, confidence and consistency

```python
result.x_base         # determined by the Sentinel-2 observation
result.prior          # added by the learned prior, invisible to the sensor; x_base + prior == image
result.support        # 2 HIGH (observation-determined), 1 MEDIUM, 0 LOW (prior-dominated) or invalid
result.confidence     # learned per-pixel error scale (raw; use uncertainty() for decisions)
result.valid          # False on NoData, cloud, cloud shadow, cirrus, saturation
```

See [Support, confidence and consistency](https://sharadhnaidu.github.io/synapse-sr/guide/support/).

## Applications

```python
idx = result.indices()                                   # ndvi savi evi gndvi ndwi (+ ndre ndbi nbr mndwi)
fields = synapse_sr.boundaries(result, "field")          # also "water", "urban"
flood = synapse_sr.change(before, after, "ndwi")         # also "ndvi", "nbr", "brightness"
flood.mask, flood.area_km2, flood.unreliable_fraction
```

`change` flags only pixels that are valid and observation-supported on both dates. See
[Applications](https://sharadhnaidu.github.io/synapse-sr/applications/) for crop monitoring, urban analysis,
water mapping and disaster assessment, with benchmarks.

## Calibrated uncertainty

```python
err = result.uncertainty()     # expected absolute error per pixel and band (reflectance)
half = result.interval(0.9)    # the HR reference lies within image +/- half with probability 0.9
```

Pro v1 measured coverage on held-out patches: 82 / 91 / 96 % at the 80 / 90 / 95 % levels.

## Command line

```bash
synapse-sr scene.tif scene_2m.tif
synapse-sr scene.tif scene_2m.tif --device cpu --weights synapse-pro-v1.safetensors --scl scene_SCL.tif
synapse-sr --env
```

The output GeoTIFF has nine bands: `B04 B03 B02 B08`, `ERRSCALE_*` for each of the four, and `SUPPORT`.
Use `--no-confidence` to write the four reflectance bands only.

## Offline use

```python
model = Pro.from_pretrained(weights="/opt/models/synapse-pro-v1.safetensors")
```

No network access is needed apart from an optional one-time weight download, which is verified by SHA-256. See
[Offline and on-premises use](https://sharadhnaidu.github.io/synapse-sr/guide/offline/).

## Examples

Produced with the package itself (`tools/make_gifs.py`). Left: Sentinel-2 L2A at 10 m. Right: synapse-sr at
2 m. Each area is 1.28 km x 1.28 km.

| RV University, Bengaluru | Bengaluru city centre |
|---|---|
| <img src="https://raw.githubusercontent.com/SharadhNaidu/synapse-sr/main/docs/assets/gifs/rv_university.gif" width="400"> | <img src="https://raw.githubusercontent.com/SharadhNaidu/synapse-sr/main/docs/assets/gifs/bengaluru_urban.gif" width="400"> |
| **Ludhiana, Punjab: fields** | **Wayanad, Kerala: landslide-affected hills** |
| <img src="https://raw.githubusercontent.com/SharadhNaidu/synapse-sr/main/docs/assets/gifs/punjab_fields.gif" width="400"> | <img src="https://raw.githubusercontent.com/SharadhNaidu/synapse-sr/main/docs/assets/gifs/wayanad_landslide.gif" width="400"> |

More in the [Examples](https://sharadhnaidu.github.io/synapse-sr/examples/) section of the documentation.

## How it works

```
x_hat = x_base + P_N(delta)
```

- `x_base`: a Tikhonov-regularised inversion of the Sentinel-2 forward model `A`. Its per-band weight is set
  so the residual matches sensor noise.
- `delta`: predicted by `SynapseProX5`, a 14.4 M-parameter network. It has:
  - a state-space (Mamba) backbone;
  - a gated 20 m spectral-context stem;
  - a frequency mixer;
  - a direct x5 PixelShuffle head.
- `P_N = I - A^T (A A^T)^+ A`: removes everything the sensor could have observed. `A x_hat = A x_base`, whatever
  the network predicts.

Details: [How it works](https://sharadhnaidu.github.io/synapse-sr/method/).

## Limitations

- **Grid spacing is not effective resolution.** In the controlled bar-pair test, the v0.1 preview checkpoint
  does not yet certify pairs at 8 m or finer; the official SEN2SR model certifies 6 m. No effective-resolution
  figure is claimed for this release.
- **Only red, green, blue and near-infrared are super-resolved.** The six 20 m bands are used as context only.
- **Consistency assumes the nominal sensor model and correct geolocation.** It degrades with PSF or
  registration error.
- **The confidence map is not calibrated.** The support thresholds are heuristic.
- **The training reference imagery comes from the United States.** Other regions are not separately
  validated.

Full list: [Limitations](https://sharadhnaidu.github.io/synapse-sr/limitations/).

## Citation

```bibtex
@software{synapse_sr,
  title  = {synapse-sr: observation-consistent super-resolution of Sentinel-2 imagery},
  author = {Naidu, Sharadh},
  year   = {2026},
  url    = {https://github.com/SharadhNaidu/synapse-sr}
}
```

## Acknowledgements
- The optional fused kernel comes from [mamba-ssm](https://github.com/state-spaces/mamba) (Apache-2.0).
- Sentinel-2 data: Copernicus programme, European Space Agency.

See [THIRD_PARTY_NOTICES](https://github.com/SharadhNaidu/synapse-sr/blob/main/src/synapse_sr/THIRD_PARTY_NOTICES).
