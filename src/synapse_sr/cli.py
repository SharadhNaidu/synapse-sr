import argparse
import json
import platform
import sys

from .pretrained.registry import DEFAULT


def _env():
    import numpy
    import torch

    from synapse_sr import __version__
    from synapse_sr.models import scan
    info = {"synapse-sr": __version__, "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": numpy.__version__, "cuda_available": torch.cuda.is_available(),
            "fused_scan_kernel": scan.selective_scan_cuda is not None}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["bf16"] = torch.cuda.is_bf16_supported()
    try:
        import rasterio
        info["rasterio"] = rasterio.__version__
        info["gdal"] = rasterio.__gdal_version__
    except ImportError:
        info["rasterio"] = None
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(prog="synapse-sr",
                                 description="Sentinel-2 10 m -> 2.0 m RGBN super-resolution (SYNAPSE Pro)")
    ap.add_argument("input", nargs="?", help="Sentinel-2 GeoTIFF (10, 12 or 13 bands, or named bands)")
    ap.add_argument("output", nargs="?", help="output GeoTIFF: B04 B03 B02 B08, ERRSCALE x4, SUPPORT")
    ap.add_argument("--weights", help="local .safetensors checkpoint (no network access needed)")
    ap.add_argument("--model", default=DEFAULT, help=f"registered model name (default: {DEFAULT})")
    ap.add_argument("--device", help="cuda, cpu, cuda:1, ... (default: cuda when available)")
    ap.add_argument("--scl", default="auto", help="scene classification raster; 'auto' uses <input>_scl.tif, 'none' disables")
    ap.add_argument("--offset", type=float, help="DN offset, e.g. -1000 for baseline >= 04.00 (default: BOA_ADD_OFFSET tag or 0)")
    ap.add_argument("--tile", type=int, help="tile size in source pixels")
    ap.add_argument("--halo", type=int, default=16, help="tile context halo in source pixels (default 16)")
    ap.add_argument("--no-confidence", action="store_true", help="write the four reflectance bands only")
    ap.add_argument("--env", action="store_true", help="print environment diagnostics and exit")
    ap.add_argument("--version", action="store_true", help="print the version and exit")
    a = ap.parse_args(argv)
    if a.version:
        from synapse_sr import __version__
        print(__version__)
        return 0
    if a.env:
        print(json.dumps(_env(), indent=1))
        return 0
    if not a.input or not a.output:
        ap.error("input and output are required")
    from synapse_sr import Pro
    model = Pro.from_pretrained(a.model, weights=a.weights, device=a.device)
    r = model.super_resolve(a.input, scl=None if str(a.scl).lower() == "none" else a.scl, offset=a.offset,
                            tile=a.tile, halo=a.halo)
    r.save(a.output, with_confidence=not a.no_confidence)
    print(json.dumps({"output": a.output, "shape": list(r.image.shape), "gsd_m": r.gsd,
                      "consistency_rms_over_tau": r.consistency, "scan_backend": r.metadata["scan_backend"],
                      "precision": r.metadata["precision"], "invalid_input_fraction": r.metadata["invalid_input_fraction"],
                      "scl": r.metadata["scl"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
