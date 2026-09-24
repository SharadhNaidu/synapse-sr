# Choosing a model and a device

synapse-sr ships two networks behind one pipeline. Both produce `x_hat = x_base + P_N(delta)`: the same physics
baseline, the same null-space projection, and the same support, consistency and uncertainty outputs. They differ
only in the network that predicts `delta`.

| | **Pro** | **Flash** |
|---|---|---|
| Network | `SynapseProX5`: Mamba state-space backbone with a 20 m spectral-context stem, 14.4 M parameters | `SynapseFlashX5`: re-parameterised SPAN-style CNN, ~0.6 M parameters at inference |
| Context | whole tile (global scan) | local convolutions |
| Best hardware | NVIDIA GPU | anything: CPU, laptop, integrated graphics, Apple silicon, ARM |
| Load | `Pro.from_pretrained()` | `Flash.from_pretrained()` |
| CLI | `--model pro` (default) | `--model flash` |

```python
from synapse_sr import Pro, Flash

pro = Pro.from_pretrained()                       # most accurate
flash = Flash.from_pretrained(device="cpu")       # fastest on any machine
```

`Pro.from_pretrained(weights=...)` also accepts a Flash checkpoint and returns a `Flash` object, so code that
loads local files does not need to know which kind of checkpoint it has.

!!! note "Flash weights"
    Flash is in training. `synapse-sr --models` lists what is published. Until Flash weights are released,
    `Flash.from_pretrained()` raises an error that explains how to load a local checkpoint.

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
