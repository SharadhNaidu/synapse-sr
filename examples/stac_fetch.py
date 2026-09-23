"""Optional helper: fetch a Sentinel-2 L2A scene subset from a public STAC API into a SYNAPSE-ready GeoTIFF.
Not a package dependency - SYNAPSE runs fully offline on local GeoTIFFs.

    pip install pystac-client rasterio
    python stac_fetch.py 76.135 11.475 2025-01-01 2025-03-31 out.tif --half 3000

Writes 10 bands in SYNAPSE order (B04 B03 B02 B08 B05 B06 B07 B8A B11 B12, band descriptions set) on the 10 m
grid of B04; 20 m bands nearest-resampled (native values replicated, as in training); plus out_scl.tif.
"""

import argparse

import numpy as np
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

API = "https://earth-search.aws.element84.com/v1"
ASSETS = {"B04": "red", "B03": "green", "B02": "blue", "B08": "nir", "B05": "rededge1", "B06": "rededge2",
          "B07": "rededge3", "B8A": "nir08", "B11": "swir16", "B12": "swir22", "SCL": "scl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lon", type=float); ap.add_argument("lat", type=float)
    ap.add_argument("start"); ap.add_argument("end"); ap.add_argument("out")
    ap.add_argument("--half", type=float, default=3000, help="half-size of the square in metres")
    ap.add_argument("--max-cloud", type=float, default=20)
    a = ap.parse_args()
    items = Client.open(API).search(collections=["sentinel-2-l2a"], intersects={"type": "Point", "coordinates": [a.lon, a.lat]},
                                    datetime=f"{a.start}/{a.end}", query={"eo:cloud_cover": {"lt": a.max_cloud}}).item_collection()
    if not items:
        raise SystemExit("no scene found")
    it = min(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))
    with rasterio.open(it.assets[ASSETS["B04"]].href) as ref:
        x, y = rasterio.warp.transform("EPSG:4326", ref.crs, [a.lon], [a.lat])
        b = (x[0] - a.half, y[0] - a.half, x[0] + a.half, y[0] + a.half)
        win = from_bounds(*b, ref.transform).round_offsets().round_lengths()
        h, w = int(win.height), int(win.width)
        tr = ref.window_transform(win)
        prof = dict(driver="GTiff", crs=ref.crs, transform=tr, height=h, width=w, compress="deflate")
    bands, names = [], []
    for name, key in ASSETS.items():
        with rasterio.open(it.assets[key].href) as src:
            wb = from_bounds(*transform_bounds(prof["crs"], src.crs, *rasterio.transform.array_bounds(h, w, tr)), src.transform)
            arr = src.read(1, window=wb, out_shape=(h, w), resampling=Resampling.nearest, boundless=True, fill_value=0)
        if name == "SCL":
            with rasterio.open(a.out.replace(".tif", "_scl.tif"), "w", count=1, dtype="uint8", **prof) as d:
                d.write(arr.astype(np.uint8), 1)
        else:
            bands.append(arr); names.append(name)
    offset = -1000 if float(it.properties.get("s2:processing_baseline", "0")) >= 4.0 else 0
    with rasterio.open(a.out, "w", count=len(bands), dtype="uint16", nodata=0, **prof) as d:
        d.write(np.stack(bands))
        for i, n in enumerate(names, 1):
            d.set_band_description(i, n)
    print(f"{it.id} cloud {it.properties.get('eo:cloud_cover')} -> {a.out} ({h}x{w}); "
          f"run: synapse-sr {a.out} out_2m.tif --scl {a.out.replace('.tif', '_scl.tif')} --offset {offset}")


if __name__ == "__main__":
    main()
