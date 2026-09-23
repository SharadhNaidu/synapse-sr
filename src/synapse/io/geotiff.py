import numpy as np


def _rasterio():
    try:
        import rasterio
    except ImportError as e:
        raise ImportError("GeoTIFF IO needs rasterio: pip install rasterio") from e
    return rasterio


def read(path):
    """-> (array (C, H, W), profile, band names or None)."""
    rio = _rasterio()
    with rio.open(path) as src:
        arr = src.read()
        names = [d.upper() if d else None for d in src.descriptions]
        return arr, src.profile.copy(), (names if all(names) else None)


def scaled_transform(transform, scale):
    """Same origin and orientation, pixel size divided by `scale`: output px (5i + r) lies inside source px i."""
    from affine import Affine
    return transform * Affine.scale(1.0 / scale)


def write(path, image, profile, scale, band_names=None, extra=None):
    rio = _rasterio()
    image = np.asarray(image, np.float32)
    if extra is not None:
        image = np.concatenate([image, np.asarray(extra, np.float32)])
    p = profile.copy()
    p.update(driver="GTiff", count=image.shape[0], height=image.shape[1], width=image.shape[2], dtype="float32",
             transform=scaled_transform(profile["transform"], scale), compress="deflate", nodata=None)
    p.pop("blockxsize", None); p.pop("blockysize", None); p.pop("tiled", None)
    with rio.open(path, "w", **p) as dst:
        dst.write(image)
        if band_names:
            for i, n in enumerate(band_names, 1):
                dst.set_band_description(i, n)
