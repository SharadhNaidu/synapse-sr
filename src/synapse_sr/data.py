"""Optional helper: fetch a Sentinel-2 L2A subset from a public STAC catalogue (``pip install synapse-sr[stac]``).

SYNAPSE itself never needs the network; this is a convenience for getting a first scene.
"""

import datetime as dt
import pathlib
from typing import Optional

import numpy as np

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
ASSETS = {"B04": "red", "B03": "green", "B02": "blue", "B08": "nir", "B05": "rededge1", "B06": "rededge2",
          "B07": "rededge3", "B8A": "nir08", "B11": "swir16", "B12": "swir22"}
SCL_INVALID = (0, 1, 3, 8, 9, 10)          # no data, saturated/defective, cloud shadow, cloud medium/high, cirrus
CANDIDATES = 6                             # scenes whose local cloud is checked with rank="local"


def boa_offset(properties: dict) -> int:
    """DN offset still to be applied to a STAC item's L2A assets: -1000 for processing baseline 04.00 or later,
    unless the provider already removed it (Earth Search v1 flags this with ``earthsearch:boa_offset_applied``;
    applying it twice pushes dark surfaces such as vegetation red and water below zero reflectance)."""
    baseline = float(properties.get("s2:processing_baseline", "0") or 0)
    applied = bool(properties.get("earthsearch:boa_offset_applied", False))
    return -1000 if baseline >= 4.0 and not applied else 0


def offset_present(dn, valid=None) -> bool:
    """True if L2A DN still carry the +1000 baseline-04.00 offset. With it, surface reflectance 0 is DN 1000, so real
    pixels almost never fall below it; when more than 5 % of valid blue/green/red pixels do, the offset has already
    been removed (whatever the product metadata says)."""
    vis = np.asarray(dn[:3], dtype=np.float32)
    ok = (vis > 0).all(0) if valid is None else valid & (vis > 0).all(0)
    if ok.sum() < 16:
        return True
    return float((vis[:, ok] < 1000).mean()) <= 0.05


def check_area(lat, lon, start, end, size_m=2000.0):
    """Validate fetch arguments; returns (start, end) as ISO dates."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"latitude must be in [-90, 90] and longitude in [-180, 180], got ({lat}, {lon}); "
                         "the order is LAT,LON")
    if not 10 <= size_m <= 50000:
        raise ValueError(f"size_m must be between 10 m and 50 km, got {size_m}")
    try:
        a, b = dt.date.fromisoformat(str(start)), dt.date.fromisoformat(str(end))
    except ValueError:
        raise ValueError(f"dates must be YYYY-MM-DD, got {start!r} and {end!r}") from None
    if a > b:
        raise ValueError(f"start {a} is after end {b}")
    if b < dt.date(2015, 6, 23):
        raise ValueError(f"Sentinel-2 data starts in June 2015; the range {a}..{b} is earlier")
    return str(a), str(b)


def fetch_sentinel2(lat: float, lon: float, start: str, end: str, size_m: float = 2000.0,
                    out: Optional[str] = None, max_cloud: float = 20.0, api: str = EARTH_SEARCH,
                    rank: str = "local") -> str:
    """Download the least-cloudy Sentinel-2 L2A scene over a point into a SYNAPSE-ready GeoTIFF.

    Parameters
    ----------
    lat, lon:
        Centre of the area, WGS84 degrees.
    start, end:
        Date range, ``"YYYY-MM-DD"``.
    size_m:
        Edge length of the square area in metres (on the scene's UTM grid), 10 m to 50 km.
    out:
        Output path; defaults to ``s2_<item id>.tif`` in the current directory.
    max_cloud:
        Maximum scene cloud cover in percent.
    api:
        STAC API root; default Element 84 Earth Search.
    rank:
        ``"local"`` (default) reads the scene classification of the few least-cloudy scenes over the requested area
        and keeps the one with the fewest cloud, shadow and no-data pixels there; ``"scene"`` keeps the scene with
        the lowest whole-tile cloud cover (one scene read, but a mostly clear tile can be cloudy over the area).

    Returns
    -------
    str
        Path of the 10-band GeoTIFF (B04 B03 B02 B08 B05 B06 B07 B8A B11 B12, named bands, 10 m grid, 20 m bands
        nearest-resampled). The scene classification layer is written next to it as ``*_scl.tif`` and the
        radiometric offset is stored in the ``BOA_ADD_OFFSET`` tag, which :meth:`Pro.super_resolve` applies
        automatically. The tags also record the item id, the scene cloud cover and the area's invalid fraction.
    """
    if rank not in ("local", "scene"):
        raise ValueError(f"rank must be 'local' or 'scene', got {rank!r}")
    start, end = check_area(lat, lon, start, end, size_m)
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
    half = size_m / 2.0

    def area(item):
        with rasterio.open(item.assets["red"].href) as ref:
            x, y = warp_xy("EPSG:4326", ref.crs, [lon], [lat])
            win = from_bounds(x[0] - half, y[0] - half, x[0] + half, y[0] + half, ref.transform).round_offsets().round_lengths()
            return ref.window_transform(win), ref.crs, int(win.height), int(win.width)

    def read(item, key, tr, crs, h, w):
        bounds = array_bounds(h, w, tr)
        with rasterio.open(item.assets[key].href) as src:
            wb = from_bounds(*transform_bounds(crs, src.crs, *bounds), src.transform)
            return src.read(1, window=wb, out_shape=(h, w), resampling=Resampling.nearest, boundless=True, fill_value=0)

    ranked = sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 100.0))
    best = None
    for item in ranked[:CANDIDATES if rank == "local" else 1]:
        grid = area(item)
        scl = read(item, "scl", *grid)
        bad = float(np.isin(scl, SCL_INVALID).mean())
        if best is None or bad < best[0]:
            best = (bad, item, grid, scl)
        if bad == 0.0:
            break
    bad, item, (tr, crs, h, w), scl = best
    out = str(out or f"s2_{item.id}.tif")
    stack = np.stack([read(item, k, tr, crs, h, w) for k in ASSETS.values()])
    offset = boa_offset(item.properties)
    if offset and not offset_present(stack[:3]):            # provider flag disagrees with the data: trust the data
        offset = 0
    prof = dict(driver="GTiff", crs=crs, transform=tr, height=h, width=w, compress="deflate")
    with rasterio.open(out, "w", count=len(ASSETS), dtype="uint16", nodata=0, **prof) as d:
        d.write(stack.astype(np.uint16))
        for i, name in enumerate(ASSETS, 1):
            d.set_band_description(i, name)
        d.update_tags(BOA_ADD_OFFSET=str(offset), STAC_ITEM=item.id, CLOUD_COVER=str(item.properties.get("eo:cloud_cover")),
                      AREA_INVALID_FRACTION=f"{bad:.4f}")
    side = pathlib.Path(out).with_name(pathlib.Path(out).stem + "_scl.tif")
    with rasterio.open(side, "w", count=1, dtype="uint8", **prof) as d:
        d.write(scl.astype(np.uint8), 1)
    return out
