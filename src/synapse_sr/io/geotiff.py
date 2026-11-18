import numpy as np


def _rasterio():
    try:
        import rasterio
    except ImportError as e:
        raise ImportError("GeoTIFF IO needs rasterio: pip install rasterio") from e
    return rasterio


def window(path, window=None, bounds=None):
    """A rasterio Window inside ``path`` from ``window`` ((row_off, col_off, height, width) or a Window) or
    ``bounds`` ((left, bottom, right, top) in the raster's CRS), clipped to the raster; None when neither is given."""
    if window is None and bounds is None:
        return None
    if window is not None and bounds is not None:
        raise ValueError("pass window= or bounds=, not both")
    rio = _rasterio()
    from rasterio.windows import Window, from_bounds
    with rio.open(path) as src:
        if bounds is not None:
            left, bottom, right, top = bounds
            if right <= left or top <= bottom:
                raise ValueError(f"bounds must be (left, bottom, right, top), got {tuple(bounds)}")
            w = from_bounds(left, bottom, right, top, src.transform)
        else:
            w = window if isinstance(window, Window) else Window(window[1], window[0], window[3], window[2])
        w = w.round_offsets().round_lengths()
        try:
            w = w.intersection(Window(0, 0, src.width, src.height))
        except Exception:
            raise ValueError(f"the requested area {w} lies outside the {src.height} x {src.width} px raster") from None
    if w.width < 1 or w.height < 1:
        raise ValueError(f"the requested area {w} is empty")
    return w


def read(path, window=None, bounds=None):
    """-> (array (C, H, W), profile, band names or None, dataset tags). ``window`` / ``bounds`` read a sub-area
    (see :func:`window`); ``bounds`` may be in another raster's CRS only if both share it."""
    rio = _rasterio()
    win = globals()["window"](path, window, bounds)
    with rio.open(path) as src:
        arr = src.read(window=win)
        prof = src.profile.copy()
        if win is not None:
            prof.update(height=int(win.height), width=int(win.width), transform=src.window_transform(win))
        names = [d.upper() if d else None for d in src.descriptions]
        return arr, prof, (names if all(names) else None), src.tags()


def scaled_transform(transform, scale):
    """Same origin and orientation, pixel size divided by `scale`: output px (5i + r) lies inside source px i."""
    from affine import Affine
    return transform * Affine.scale(1.0 / scale)


def write(path, image, profile, scale, band_names=None, extra=None, cog=False, nodata=None, tags=None):
    rio = _rasterio()
    image = np.asarray(image, np.float32)
    if extra is not None:
        image = np.concatenate([image, np.asarray(extra, np.float32)])
    p = profile.copy()
    p.update(driver="GTiff", count=image.shape[0], height=image.shape[1], width=image.shape[2], dtype="float32",
             transform=scaled_transform(profile["transform"], scale), compress="deflate", nodata=nodata)
    p.pop("blockxsize", None); p.pop("blockysize", None); p.pop("tiled", None)
    with rio.open(path, "w", **p) as dst:
        dst.write(image)
        if band_names:
            for i, n in enumerate(band_names, 1):
                dst.set_band_description(i, n)
        if tags:
            dst.update_tags(**tags)
    if cog:
        to_cog(path)


def to_cog(path):
    """Rewrite a GeoTIFF in place as a Cloud-Optimised GeoTIFF (internal tiles + overviews), for web maps and GIS."""
    import os
    import shutil
    rio = _rasterio()
    from rasterio.enums import Resampling
    tmp = str(path) + ".tmp.tif"
    shutil.move(str(path), tmp)
    try:
        with rio.open(tmp, "r+") as d:
            n = max(d.width, d.height)
            levels = [f for f in (2, 4, 8, 16, 32) if n // f >= 256]
            if levels:
                d.build_overviews(levels, Resampling.average)
        from rasterio.shutil import copy
        copy(tmp, str(path), driver="COG", compress="DEFLATE", overview_resampling="AVERAGE")
    except Exception:                                   # older GDAL without the COG driver: tiled GeoTIFF + overviews
        copy_kw = dict(driver="GTiff", tiled=True, blockxsize=512, blockysize=512, compress="deflate", copy_src_overviews=True)
        from rasterio.shutil import copy
        copy(tmp, str(path), **copy_kw)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

