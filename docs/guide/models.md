# Choosing a model and a device

synapse-sr ships two networks behind one pipeline. Both produce `x_hat = x_base + P_N(delta)`: the same physics
baseline, the same null-space projection, and the same support, consistency and uncertainty outputs. They differ
only in the network that predicts `delta`.

| | **Flash** (default) | **Pro** |
|---|---|---|
| Network | `SynapseFlashX5`: re-parameterised SPAN-style CNN on all ten bands, ~0.6 M parameters at inference, distilled from Pro | `SynapseProX5`: Mamba state-space backbone with a 20 m spectral-context stem, 14.4 M parameters |
| Context | local convolutions (dilated receptive field over the whole tile) | whole tile (global scan) |
| Speed, 1.28 km scene | ~1 s on a laptop CPU, ~0.7 s on a laptop GPU | ~5 s on an A100 slice, ~25 s on a laptop GPU without Triton |
| Best hardware | anything: CPU, laptop, integrated graphics, Apple silicon, ARM, any GPU | NVIDIA GPU |
| One call | `synapse_sr.super_resolve(src)` | `synapse_sr.super_resolve(src, model="pro")` |
| Load | `Flash.from_pretrained()` | `Pro.from_pretrained()` |
| CLI | `synapse-sr in.tif out.tif` (default) | `synapse-sr in.tif out.tif --model pro` |

```python
import synapse_sr

from synapse_sr import Pro, Flash

flash = Flash.from_pretrained()                   # default: fast on any machine
pro = Pro.from_pretrained()                       # most detail; best on a GPU
```

`Pro.from_pretrained(weights=...)` also accepts a Flash checkpoint and returns a `Flash` object, so code that
loads local files does not need to know which kind of checkpoint it has.

Flash was trained to reproduce Pro's output (knowledge distillation) on real Sentinel-2 / NAIP pairs, a streamed NAIP
corpus and ISRO Cartosat-derived pairs. The two ship with different physics defaults, and on the official opensr-test
benchmark (README) each leads different columns:

| | Flash default | Pro default |
|---|---|---|
| Physics fit (`discrepancy`) | 4 noise units: absorbs forward-model error | 0.5: tight fit, the most recovered detail |
| `restore_mean` | on: each 10 m pixel's measured mean reflectance restored exactly | off |
| Leads on | detail correlation, RMSE | improvement, spectral and reflectance error |

Both are arguments of `super_resolve`, so either model can run with the other's settings.

## Test-time augmentation

`super_resolve(..., tta=True)` averages the network over the 8 flips and 90-degree rotations of each tile before
the physics projection. On the benchmark it changes the metrics by a few thousandths (slightly better spectral angle
and detail correlation); it costs 8x the network time, which is cheap for Flash.

## Devices

| `device=` | Used for |
|---|---|
| `None` (default) | CUDA when available, else CPU |
| `"cuda"`, `"cuda:1"` | NVIDIA GPUs; bfloat16 autocast on GPUs that support it |
| `"cpu"` | any machine; results equal the GPU path within floating-point rounding |
| `"mps"` | Apple silicon GPU (experimental; `"cpu"` is the tested path on macOS) |

## How Pro's scan runs

Pro's Mamba layers need a selective scan. synapse-sr picks the fastest exact implementation available:

| Backend | When | |
|---|---|---|
| `fused` | `mamba-ssm` installed and importable | CUDA kernel from the Mamba authors |
| `triton` | CUDA GPU with Triton (Colab, Kaggle, most Linux PyTorch installs) | synapse-sr's own kernel; it self-tests against the exact scan once per process and is skipped if the test fails |
| `pytorch` | everything else (CPU, Windows, macOS) | exact chunked scan in plain PyTorch |

All three agree to about 1e-5 relative error, and `result.metadata["scan_backend"]` records which one ran.
`SYNAPSE_SR_DISABLE_FUSED=1` and `SYNAPSE_SR_DISABLE_TRITON=1` switch the first two off.

## Throughput tips

- **Several scenes:** create the model once and reuse it. `synapse_sr.super_resolve` already caches it.
- **GPU memory:** `batch` sets how many tiles run together (default 8 on CUDA, 1 on CPU). Lower it if memory is short.
- **Containers with a CPU quota** (Docker, Kubernetes, Kubeflow): PyTorch sizes its thread pool from the host's
  core count, not the quota. Set `OMP_NUM_THREADS` (or call `torch.set_num_threads`) to the number of cores you
  actually have. Oversubscription can make CPU runs many times slower.
