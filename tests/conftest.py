import numpy as np
import pytest
import torch

from synapse_sr import Pro
from synapse_sr.models.forward import S2Forward
from synapse_sr.models.pro import SynapseProX5
from synapse_sr.io.sentinel2 import S2_12


def gaussian_operator(sigma=6.0, khalf=20):
    k = np.arange(-khalf, khalf + 1)
    g = np.exp(-(k[:, None] ** 2 + k[None] ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return S2Forward(np.repeat(g[None, None], 4, 0), target_m=2.0)


@pytest.fixture(scope="session")
def model():
    """Randomly initialised network with a small Gaussian PSF: exercises every code path without weights."""
    torch.manual_seed(0)
    net = SynapseProX5()
    torch.nn.init.normal_(net.to_delta.weight, std=1e-3)
    return Pro(net, gaussian_operator(), device="cpu", meta={"name": "test"})


def scene(h, w, seed=0, dn=True):
    rng = np.random.default_rng(seed)
    from numpy.fft import irfft2, rfft2
    noise = rng.random((h, w))
    ky = np.fft.fftfreq(h)[:, None]; kx = np.fft.rfftfreq(w)[None]
    base = irfft2(rfft2(noise) * np.exp(-(ky ** 2 + kx ** 2) * 60), s=(h, w))
    base = (base - base.min()) / (np.ptp(base) + 1e-9) * 0.3 + 0.05
    arr = np.stack([base * (0.8 + 0.05 * k) for k in range(12)])
    for k in (4, 5, 6, 8, 10, 11):
        m = arr[k][: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2).mean((1, 3))
        arr[k][: h // 2 * 2, : w // 2 * 2] = np.repeat(np.repeat(m, 2, 0), 2, 1)
    return (arr * 10000).astype(np.uint16) if dn else arr.astype(np.float32)


def write_tif(path, arr, names=tuple(S2_12), tags=None, nodata=None):
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine
    from rasterio.crs import CRS
    t = Affine(10.0, 0, 500000.0, 0, -10.0, 4200000.0)
    with rasterio.open(path, "w", driver="GTiff", count=arr.shape[0], height=arr.shape[1], width=arr.shape[2],
                       dtype=arr.dtype, crs=CRS.from_epsg(32643), transform=t, nodata=nodata) as d:
        d.write(arr)
        if names:
            for i, n in enumerate(names, 1):
                d.set_band_description(i, n)
        if tags:
            d.update_tags(**tags)
    return t
