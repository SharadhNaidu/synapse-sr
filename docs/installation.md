# Installation

## Requirements

| | |
|---|---|
| Python | 3.8 or newer |
| PyTorch | 1.13 or newer (CPU or CUDA build) |
| Other | `numpy`, `safetensors`, `rasterio`, `affine`, installed automatically |

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

## GPU fast path

On NVIDIA GPUs, synapse-sr uses the fused selective-scan kernel from
[mamba-ssm](https://github.com/state-spaces/mamba) when it can import it. Otherwise it falls back to a
pure PyTorch implementation. The fallback agrees with the fused kernel to a relative error of 1.7e-7, so the
results are the same and only the speed differs.

`mamba-ssm` compiles CUDA code and must match your PyTorch and CUDA versions:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # your CUDA version
pip install mamba-ssm --no-build-isolation
```

If the kernel is installed but was built for a different PyTorch, synapse-sr warns and uses the PyTorch path
instead of failing. To force the PyTorch path, set `SYNAPSE_SR_DISABLE_FUSED=1`.

## Check the installation

```bash
synapse-sr --env
```

This prints the synapse-sr, Python, PyTorch, numpy, rasterio and GDAL versions, whether CUDA and bfloat16 are
available, and whether the fused kernel is in use.
