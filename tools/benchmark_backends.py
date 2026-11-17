"""Benchmark the PyTorch, OpenVINO and ONNX Runtime backends.

Run from the repository root:

    python tools/benchmark_backends.py --device cpu --models flash
    python tools/benchmark_backends.py --device cuda --models flash pro --json out.json

The physics pipeline is included in the end-to-end numbers. The network-only
numbers isolate the part the accelerated backends replace.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings

import numpy as np
import torch

import synapse_sr


def scene(n=121, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.random((10, n, n)).astype(np.float32)
    # cheap smoothing so the physics solve is not pathological
    for _ in range(3):
        raw = (raw + np.roll(raw, 1, -1) + np.roll(raw, -1, -1)) / 3.0
    return (raw * 0.45 + 0.02).astype(np.float32)


BANDS = ["B04", "B03", "B02", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]


def timed(fn, reps):
    fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--models", nargs="+", default=["flash"], choices=["flash", "pro"])
    ap.add_argument("--tile", type=int, default=32)
    ap.add_argument("--halo", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    out = {"device": args.device, "torch": torch.__version__, "models": {}}
    if torch.cuda.is_available():
        out["cuda"] = torch.cuda.get_device_name(0)
    try:
        import openvino as ov
        out["openvino"] = ov.__version__
        out["openvino_devices"] = ov.Core().available_devices
    except Exception:
        out["openvino"] = None
    try:
        import onnxruntime as ort
        out["onnxruntime"] = ort.__version__
        out["onnxruntime_providers"] = ort.get_available_providers()
    except Exception:
        out["onnxruntime"] = None
    print(json.dumps(out, indent=1), flush=True)

    img = scene()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    for name in args.models:
        model = synapse_sr.load(name, device=args.device)
        x = torch.from_numpy(img[:, :32, :32]).unsqueeze(0).contiguous()
        row = {"params": int(sum(p.numel() for p in model.model.parameters()))}
        with torch.no_grad():
            ref = model.model(x)
            row["network_torch_ms"] = round(timed(lambda: model.model(x), 20) * 1000, 2)
        row["e2e_torch_s"] = round(timed(
            lambda: model.super_resolve(img, band_names=BANDS, tile=args.tile,
                                        halo=args.halo, progress=False, scl=None,
                                        offset=0.0), args.reps), 3)
        row["consistency"] = model.super_resolve(
            img, band_names=BANDS, tile=args.tile, halo=args.halo,
            progress=False, scl=None, offset=0.0).consistency

        if name == "flash":
            backends = []
            try:
                import openvino  # noqa: F401
                backends += [("openvino-cpu", lambda: model.enable_openvino("CPU")),
                             ("openvino-gpu", lambda: model.enable_openvino("GPU"))]
            except Exception:
                pass
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if "CPUExecutionProvider" in providers:
                    backends.append(("onnx-cpu", lambda: model.enable_onnxruntime("cpu")))
                if "CUDAExecutionProvider" in providers:
                    backends.append(("onnx-cuda", lambda: model.enable_onnxruntime("cuda")))
                if "TensorrtExecutionProvider" in providers:
                    backends.append(("onnx-tensorrt", lambda: model.enable_tensorrt()))
            except Exception:
                pass

            for label, enable in backends:
                try:
                    enable()
                except Exception as exc:
                    print(f"{label}: skipped ({exc})", flush=True)
                    continue
                accel = model._accel
                with torch.no_grad():
                    got = accel(x)
                parity = float((got["delta"] - ref["delta"]).abs().max())
                row[label] = {
                    "network_ms": round(timed(lambda: accel(x), 20) * 1000, 2),
                    "parity_delta": parity,
                }
                try:
                    row[label]["e2e_s"] = round(timed(
                        lambda: model.super_resolve(img, band_names=BANDS, tile=args.tile,
                                                    halo=args.halo, progress=False, scl=None,
                                                    offset=0.0), args.reps), 3)
                except Exception:
                    pass
                print(f"{label}: {row[label]}", flush=True)
                # reset for the next backend
                model._accel = None
        out["models"][name] = row
        print(name, json.dumps(row, indent=1), flush=True)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print("wrote", args.json, flush=True)


if __name__ == "__main__":
    main()
