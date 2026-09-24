"""Optional helper: fetch a Sentinel-2 L2A subset from a public STAC catalogue (``pip install synapse-sr[stac]``).

SYNAPSE itself never needs the network; this is a convenience for getting a first scene.
"""

import pathlib
from typing import Optional

import numpy as np

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
ASSETS = {"B04": "red", "B03": "green", "B02": "blue", "B08": "nir", "B05": "rededge1", "B06": "rededge2",
          "B07": "rededge3", "B8A": "nir08", "B11": "swir16", "B12": "swir22"}


def boa_offset(properties: dict) -> int:
    """DN offset still to be applied to a STAC item's L2A assets: -1000 for processing baseline 04.00 or later,
    unless the provider already removed it (Earth Search v1 flags this with ``earthsearch:boa_offset_applied``;
    applying it twice pushes dark surfaces such as vegetation red and water below zero reflectance)."""
    baseline = float(properties.get("s2:processing_baseline", "0") or 0)
    applied = bool(properties.get("earthsearch:boa_offset_applied", False))
    return -1000 if baseline >= 4.0 and not applied else 0


def fetch_sentinel2(lat: float, lon: float, start: str, end: str, size_m: float = 2000.0,
                    out: Optional[str] = None, max_cloud: float = 20.0, api: str = EARTH_SEARCH) -> str:
    """Download the least-cloudy Sentinel-2 L2A scene over a point into a SYNAPSE-ready GeoTIFF.

    Parameters
    ----------
    lat, lon:
        Centre of the area, WGS84 degrees.
    start, end:
        Date range, ``"YYYY-MM-DD"``.
    size_m:
        Edge length of the square area in metres (on the scene's UTM grid).
    out:
        Output path; defaults to ``s2_<item id>.tif`` in the current directory.
    max_cloud:
        Maximum scene cloud cover in percent.
    api:
        STAC API root; default Element 84 Earth Search.

    Returns
    -------
    str
        Path of the 10-band GeoTIFF (B04 B03 B02 B08 B05 B06 B07 B8A B11 B12, named bands, 10 m grid, 20 m bands
        nearest-resampled). The scene classification layer is written next to it as ``*_scl.tif`` and the
        radiometric offset is stored in the ``BOA_ADD_OFFSET`` tag, which :meth:`Pro.super_resolve` applies
        automatically.
    """
    try:
        import rasterio
        from pystac_client import Client
        from rasterio.enums import Resampling
        from rasterio.transform import array_bounds
        from rasterio.warp import transform as warp_xy, transform_bounds
        from rasterio.windows import from_bounds
    except ImportError as e:
        raise ImportError("fetch_sentinel2 needs pystac-client: pip install 'synapse-sr[stac]'") from e

    items = Client.open(api).search(collections=["sentinel-2-l2a"], intersects={"type": "Point", "coordinates": [lon, lat]},
                                    datetime=f"{start}/{end}", query={"eo:cloud_cover": {"lt": max_cloud}}).item_collection()
    if len(items) == 0:
        raise LookupError(f"no Sentinel-2 L2A scene with cloud cover < {max_cloud}% at ({lat}, {lon}) in {start}..{end}")
    item = min(items, key=lambda i: i.properties.get("eo:cloud_cover", 100.0))
    out = str(out or f"s2_{item.id}.tif")
    half = size_m / 2.0
    with rasterio.open(item.assets["red"].href) as ref:
        x, y = warp_xy("EPSG:4326", ref.crs, [lon], [lat])
        win = from_bounds(x[0] - half, y[0] - half, x[0] + half, y[0] + half, ref.transform).round_offsets().round_lengths()
        h, w = int(win.height), int(win.width)
        tr, crs = ref.window_transform(win), ref.crs
    bounds = array_bounds(h, w, tr)

    def read(key):
        with rasterio.open(item.assets[key].href) as src:
            wb = from_bounds(*transform_bounds(crs, src.crs, *bounds), src.transform)
            return src.read(1, window=wb, out_shape=(h, w), resampling=Resampling.nearest, boundless=True, fill_value=0)

    stack = np.stack([read(k) for k in ASSETS.values()])
    offset = boa_offset(item.properties)
    prof = dict(driver="GTiff", crs=crs, transform=tr, height=h, width=w, compress="deflate")
    with rasterio.open(out, "w", count=len(ASSETS), dtype="uint16", nodata=0, **prof) as d:
        d.write(stack.astype(np.uint16))
        for i, name in enumerate(ASSETS, 1):
            d.set_band_description(i, name)
        d.update_tags(BOA_ADD_OFFSET=str(offset), STAC_ITEM=item.id, CLOUD_COVER=str(item.properties.get("eo:cloud_cover")))
    scl = pathlib.Path(out).with_name(pathlib.Path(out).stem + "_scl.tif")
    with rasterio.open(scl, "w", count=1, dtype="uint8", **prof) as d:
        d.write(read("scl").astype(np.uint8), 1)
    return out
