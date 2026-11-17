---
description: Supported hardware and inference runtimes for synapse-sr, including NVIDIA CUDA, AMD ROCm, Apple MPS, Intel OpenVINO, ONNX Runtime and TensorRT.
---

# Hardware compatibility

Synapse SR runs on any device PyTorch supports. Flash can also run its detail network through OpenVINO or ONNX Runtime.

The pipeline has two parts: the detail network and the physics baseline plus null-space projector. The accelerated backends replace the network forward only; the physics stays in PyTorch, so the output is unchanged apart from floating-point rounding. On small scenes the physics can dominate, so a faster network does not always mean a faster scene.

## Matrix

| Runtime | Device | Flash | Pro | Status | Install |
|---|---|---|---|---|---|
| PyTorch | CPU | Yes | Yes | Tested | `pip install synapse-sr` |
| PyTorch | NVIDIA CUDA | Yes | Yes | Preview | CUDA PyTorch build |
| PyTorch | AMD ROCm | Yes | Yes | Preview | ROCm PyTorch build |
| PyTorch | Apple MPS | Yes | Experimental | Preview | MPS PyTorch build |
| OpenVINO | Intel CPU / GPU | Yes | No | Tested | `pip install "synapse-sr[openvino]"` |
| ONNX Runtime | CPU | Yes | No | Tested | `pip install "synapse-sr[onnx]"` |
| ONNX Runtime | NVIDIA CUDA | Yes | No | Preview | `pip install "synapse-sr[onnx-gpu]"` |
| ONNX Runtime | AMD ROCm | Yes | No | Preview | ROCm build of ONNX Runtime |
| ONNX Runtime | NVIDIA TensorRT | Yes | No | Preview | `onnx-gpu` plus TensorRT |

Pro's state-space scan is not exportable to ONNX, OpenVINO or TensorRT yet, so those backends support Flash only. Pro runs through native PyTorch on CPU, CUDA, ROCm and MPS.

**Preview** means the code path is implemented and uses the same interface, but the maintainers do not have the hardware to validate it. PyTorch CPU, OpenVINO CPU/Intel GPU and ONNX Runtime CPU are tested. The CUDA, ROCm, MPS and TensorRT rows need community validation.

The OpenVINO and ONNX Runtime backends need PyTorch 2.1 or newer for `torch.export`; the ONNX exporter is tested with PyTorch 2.6 and newer.

The CLI uses the same switches:

```bash
synapse-sr in.tif out.tif --backend openvino --backend-device CPU
synapse-sr in.tif out.tif --backend onnx --backend-device cuda
```

`synapse-sr --env` lists the installed runtimes and their devices/providers.

## What stays the same

Only the detail network moves. The physics baseline, null-space projector, support thresholds, calibration and every `Result` method stay in PyTorch, so these are backend-independent:

- `x_base`, `prior`, `context`, `valid`
- `consistency`
- `support`, `confidence` (up to network precision)
- `uncertainty()`, `interval()`
- `indices()`, `boundaries()`, `change()`
- `save()` and the GeoTIFF layout

On a 320 × 320 urban scene, the full-pipeline comparison against PyTorch was:

| Backend | Image max diff | Support agreement | Index max diff |
|---|---|---|---|
| OpenVINO CPU | 1.5e-7 | 100 % | 6e-7 |
| OpenVINO Intel GPU | 6.2e-4 | 99.5 % | 2.5e-3 |
| ONNX Runtime CPU | 1.2e-7 | 100 % | 6e-7 |

The GPU difference is FP16 rounding, not a change in the pipeline. The support map is a thresholded quantity, so a 0.5 % class change on the GPU is expected; the underlying reflectance difference is still sub-milli-reflectance.

## NVIDIA CUDA

No extra step. Pass `device="cuda"`; the package uses the native PyTorch path, and Pro picks the Triton or fused scan when available.

```python
import synapse_sr

result = synapse_sr.super_resolve("scene.tif", model="pro", device="cuda")
```

## AMD ROCm

Install the ROCm build of PyTorch. PyTorch exposes it through the same `device="cuda"` name. The selective scan is pure PyTorch, so it runs on ROCm without a separate kernel.

## Apple MPS

Install the MPS build of PyTorch and pass `device="mps"`. Flash is the recommended path. Pro runs, but its scan is slow on MPS; use Flash on a laptop.

```python
result = synapse_sr.super_resolve("scene.tif", device="mps")
```

## Intel OpenVINO

OpenVINO accelerates the Flash network on Intel CPUs and Intel GPUs. Install the optional dependency:

```bash
pip install "synapse-sr[openvino]"
```

```python
import synapse_sr

flash = synapse_sr.load("flash")
flash.enable_openvino("CPU")      # or "GPU" for an Intel GPU
result = flash.super_resolve("scene.tif")
```

`device` accepts any OpenVINO device string (`"CPU"`, `"GPU"`, `"GPU.0"`, ...). The model is exported when `enable_openvino` is called and reused for every tile. The export uses dynamic height and width, so tiles of different sizes share one compiled model.

## ONNX Runtime

ONNX Runtime runs the Flash network from an ONNX graph. Install the CPU or GPU build:

```bash
pip install "synapse-sr[onnx]"       # CPU
pip install "synapse-sr[onnx-gpu]"   # CUDA, TensorRT
```

For ROCm, use an ONNX Runtime build that includes `ROCMExecutionProvider`; the standard PyPI wheel does not.

```python
import synapse_sr

flash = synapse_sr.load("flash")
flash.enable_onnxruntime("cpu")      # "cuda", "rocm" or "tensorrt"
result = flash.super_resolve("scene.tif")
```

The model is exported to ONNX when `enable_onnxruntime` is called. The graph uses dynamic height and width, so one session handles every tile size. CPU speed depends on the model and hardware, so measure it on your machine; the main reason to use ONNX Runtime is deployment flexibility and the CUDA, ROCm and TensorRT providers, not a guaranteed CPU win.

## NVIDIA TensorRT

TensorRT runs through the ONNX Runtime TensorRT execution provider. Install `onnxruntime-gpu` (the `onnx-gpu` extra) and the TensorRT libraries for your CUDA version, then:

```python
flash.enable_tensorrt(cache_dir="trt-cache")
```

The first run builds an engine for each tile shape. `cache_dir` keeps the engines so later runs skip the build. TensorRT needs an NVIDIA GPU; the provider falls back to CUDA or CPU with a warning if it is not available.

## Not currently supported

Intel XPU, Rockchip RKNN, Qualcomm QNN, Apple CoreML and Windows DirectML are not implemented. The physics solver is PyTorch, so an NPU backend would only accelerate the network, not the full pipeline.

If you need one of these, the backend interface is small: a class that takes the Flash model, exposes `__call__(x) -> {"delta", "conf_logit"}`, and is wired into `Pro.enable_*`. Open an issue to discuss a pull request before writing a large backend.

## Preview backends: help us validate them

The CUDA, ROCm, MPS and TensorRT rows are implemented but not validated by the maintainers, because we do not have that hardware in the test loop. If you run one of them:

- **It works?** Tell us the device, driver, package versions and whether the output matches the PyTorch path. A short note in an issue is enough.
- **It fails?** Open an issue with the command, the traceback, the output of `synapse-sr --env`, and the backend and device you used.
- **You have a fix?** Send a pull request. The backend files are small and the test files skip cleanly when the runtime is missing.
- **You need help?** Reach out to the maintainers.

- Issues: <https://github.com/SharadhNaidu/synapse-sr/issues>
- Pull requests: <https://github.com/SharadhNaidu/synapse-sr/pulls>
