import argparse
import json
import os
import platform
import sys

from .pretrained import registry

ALIASES = {"pro": registry.DEFAULT, "flash": registry.DEFAULT_FLASH}


def _env():
    import numpy
    import torch

    from synapse_sr import __version__
    from synapse_sr.models import scan
    info = {"synapse-sr": __version__, "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "torch": torch.__version__, "numpy": numpy.__version__,
            "cpu_threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "fused_scan_kernel": scan.selective_scan_cuda is not None}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["bf16"] = torch.cuda.is_bf16_supported()
        info["pro_scan_backend"] = scan.backend(torch.zeros(1, device="cuda"))
    else:
        info["pro_scan_backend"] = "pytorch"
    try:
        import rasterio
        info["rasterio"] = rasterio.__version__
        info["gdal"] = rasterio.__gdal_version__
    except ImportError:
        info["rasterio"] = None
    return info


def _models():
    rows = []
    for p in sorted(registry.MANIFEST_DIR.glob("*.json")):
        m = json.loads(p.read_text())
        rows.append({"name": p.stem, "arch": m.get("arch", "pro"), "status": m.get("status", ""),
                     "published": str(m.get("url", "")).startswith("http"),
                     "default": p.stem in (registry.DEFAULT, registry.DEFAULT_FLASH)})
    return rows


def _table(title, rows):
    from synapse_sr import ui
    con = ui.console()
    if con is None:
        print(json.dumps(rows, indent=1))
        return
    from rich.table import Table
    if isinstance(rows, dict):
        t = Table(title=title, title_style=f"bold {ui.ACCENT}", show_header=False, box=None)
        t.add_column(style="dim")
        t.add_column()
        for k, v in rows.items():
            t.add_row(k, str(v))
    else:
        t = Table(title=title, title_style=f"bold {ui.ACCENT}")
        for k in rows[0]:
            t.add_column(k)
        for r in rows:
            t.add_row(*[("yes" if v is True else "" if v is False else str(v)) for v in r.values()])
    con.print(t)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="synapse-sr",
                                 description="Sentinel-2 10 m -> 2.0 m RGBN super-resolution (SYNAPSE Pro / Flash)",
                                 epilog="examples:\n  synapse-sr scene.tif scene_2m.tif\n"
                                        "  synapse-sr scene.tif scene_2m.tif --model flash --device cpu\n"
                                        "  synapse-sr --env\n  synapse-sr --models",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="Sentinel-2 GeoTIFF (10, 12 or 13 bands, or named bands)")
    ap.add_argument("output", nargs="?", help="output GeoTIFF: B04 B03 B02 B08, ERRSCALE x4, SUPPORT")
    ap.add_argument("--model", default="pro",
                    help="'pro' (default, most accurate), 'flash' (fast, CPU friendly) or a registered name")
    ap.add_argument("--weights", help="local .safetensors checkpoint (no network access needed)")
    ap.add_argument("--device", help="cuda, cpu, mps, cuda:1, ... (default: cuda when available)")
    ap.add_argument("--scl", default="auto", help="scene classification raster; 'auto' uses <input>_scl.tif, 'none' disables")
    ap.add_argument("--offset", type=float, help="DN offset, e.g. -1000 for baseline >= 04.00 (default: BOA_ADD_OFFSET tag or 0)")
    ap.add_argument("--tile", type=int, help="tile size in source pixels")
    ap.add_argument("--halo", type=int, default=16, help="tile context halo in source pixels (default 16)")
    ap.add_argument("--batch", type=int, help="tiles per forward pass (lower it if memory is short)")
    ap.add_argument("--no-confidence", action="store_true", help="write the four reflectance bands only")
    ap.add_argument("--json", action="store_true", help="machine-readable JSON on stdout instead of tables")
    ap.add_argument("--quiet", action="store_true", help="no progress bar or summary")
    ap.add_argument("--env", action="store_true", help="print environment diagnostics and exit")
    ap.add_argument("--models", action="store_true", help="list registered models and exit")
    ap.add_argument("--version", action="store_true", help="print the version and exit")
    ap.add_argument("--debug", action="store_true", help="show full tracebacks on errors")
    a = ap.parse_args(argv)
    if a.version:
        from synapse_sr import __version__
        print(__version__)
        return 0
    if a.env or a.models:
        rows = _env() if a.env else _models()
        if a.json:
            print(json.dumps(rows, indent=1))
        else:
            _table("synapse-sr environment" if a.env else "registered models", rows)
        return 0
    if not a.input or not a.output:
        ap.error("input and output are required (examples: synapse-sr --help)")
    if not os.path.exists(a.input):
        return _fail(f"input not found: {a.input}")
    try:
        return _run(a)
    except KeyboardInterrupt:
        return _fail("interrupted", 130)
    except Exception as e:
        if a.debug:
            raise
        return _fail(f"{type(e).__name__}: {e}  (rerun with --debug for the traceback; see "
                     "https://sharadhnaidu.github.io/synapse-sr/troubleshooting/)")


def _fail(msg, code=1):
    print(f"synapse-sr: error: {msg}", file=sys.stderr)
    return code


def _run(a):
    from synapse_sr import load, ui
    model = load(a.model, weights=a.weights, device=a.device)
    r = model.super_resolve(a.input, scl=None if str(a.scl).lower() == "none" else a.scl, offset=a.offset,
                            tile=a.tile, halo=a.halo, batch=a.batch,
                            progress=False if (a.quiet or a.json) else "auto")
    r.save(a.output, with_confidence=not a.no_confidence)
    if a.json:
        print(json.dumps(dict(r.summary(print_=False), output=a.output)))
    elif not a.quiet:
        con = ui.console()
        if con is not None:
            con.print(ui.summary_table(r))
            con.print(f"[green]done[/green] wrote [bold]{a.output}[/bold]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
