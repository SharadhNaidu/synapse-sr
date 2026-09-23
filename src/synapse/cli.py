import argparse
import json

from synapse import Pro


def main(argv=None):
    ap = argparse.ArgumentParser(prog="synapse-sr", description="Sentinel-2 10 m -> 2.0 m RGBN (SYNAPSE Pro)")
    ap.add_argument("input", help="Sentinel-2 GeoTIFF (10, 12 or 13 bands, or band descriptions)")
    ap.add_argument("output", help="output GeoTIFF: B04 B03 B02 B08, error-scale x4, SUPPORT")
    ap.add_argument("--weights", help="local synapse-pro .safetensors (air-gapped use)")
    ap.add_argument("--model", default="pro-v1")
    ap.add_argument("--device")
    ap.add_argument("--scl", help="Sentinel-2 SCL raster for cloud / shadow masking")
    ap.add_argument("--offset", type=float, default=0.0, help="L2A BOA_ADD_OFFSET (-1000 for baseline >= 04.00)")
    ap.add_argument("--tile", type=int)
    ap.add_argument("--no-confidence", action="store_true")
    a = ap.parse_args(argv)
    model = Pro.from_pretrained(a.model, weights=a.weights, device=a.device)
    r = model.super_resolve(a.input, tile=a.tile, offset=a.offset, scl=a.scl)
    r.save(a.output, with_confidence=not a.no_confidence)
    print(json.dumps({"output": a.output, "gsd_m": r.gsd, "shape": list(r.image.shape),
                      "consistency_rms_over_tau": r.consistency, "backend": r.metadata["scan_backend"]}))


if __name__ == "__main__":
    main()
