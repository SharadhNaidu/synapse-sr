# Installation

## Requirements

| | |
|---|---|
| Python | 3.8 or newer |
| PyTorch | 1.13 or newer (CPU or CUDA build) |
| Other | `numpy`, `scipy`, `safetensors`, `rasterio`, `affine`, `rich`, installed automatically |

## Install

=== "pip (PyPI)"

    ```bash
    pip install synapse-sr
    ```

=== "pip (GitHub)"

    ```bash
    pip install git+https://github.com/SharadhNaidu/synapse-sr.git
    ```

=== "From source"

    ```bash
    git clone https://github.com/SharadhNaidu/synapse-sr.git
    cd synapse-sr
    pip install -e ".[all,test]"
    ```

Install the PyTorch build that matches your hardware first if you need a specific CUDA version. See
[pytorch.org](https://pytorch.org/get-started/locally/).

## Optional extras

| Extra | Installs | Enables |
|---|---|---|
| `synapse-sr[stac]` | `pystac-client` | [`fetch_sentinel2`](guide/data.md) for downloading a scene |
| `synapse-sr[xarray]` | `xarray` | `Result.to_xarray()` and xarray inputs |
| `synapse-sr[all]` | both of the above | |
| `synapse-sr[cuda]` | `mamba-ssm` | the fused CUDA selective-scan kernel |

## Platforms

| Platform | Pro scan backend | Recommended model |
|---|---|---|
| Google Colab, Kaggle (GPU) | `triton`, compiled on first use; nothing extra to install | Pro |
| Linux + NVIDIA GPU | `triton`; `fused` if `mamba-ssm` is installed | Pro |
| Windows + NVIDIA GPU | `pytorch` (exact, slower) | Pro or Flash |
| CPU only, laptops, integrated graphics | `pytorch` | Flash |
| macOS (Intel or Apple silicon), ARM Linux | `pytorch` | Flash |

All backends give the same numbers to about 1e-5 relative error; only the speed differs. See
[Choosing a model and a device](guide/models.md).

### Optional: the fused mamba-ssm kernel

`mamba-ssm` compiles CUDA code and must match your PyTorch and CUDA versions:

```bash
pip install mamba-ssm --no-build-isolation
```

If it is installed but was built for a different PyTorch, synapse-sr warns and uses Triton or the PyTorch path
instead of failing. `SYNAPSE_SR_DISABLE_FUSED=1` and `SYNAPSE_SR_DISABLE_TRITON=1` switch those backends off.

## Check the installation

```bash
synapse-sr --env
```

This prints the synapse-sr, Python, PyTorch, numpy, rasterio and GDAL versions, the CPU thread count, whether
CUDA, bfloat16 and Apple MPS are available, and which scan backend Pro will use (`--json` for a machine-readable
version). `synapse-sr --models` lists the registered checkpoints and whether they are published.
