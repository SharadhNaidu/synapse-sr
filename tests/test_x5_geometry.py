"""x5 head geometry: phase balance, cell-period energy, grid centring, flat field, translation."""

import pathlib
import sys

import torch
import torch.nn as nn

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from synapse_sr.models.pro import SynapseProX5, icnr_  # noqa: E402

torch.manual_seed(0)


def phase_means(h, s=5):
    b, c, H, W = h.shape
    return h.view(b, c, H // s, s, W // s, s).mean((0, 1, 2, 4))          # (s, s)


def test_icnr_makes_all_25_phases_identical_on_any_input():
    s = 5
    conv = nn.Conv2d(8, 8 * s * s, 3, padding=1)
    icnr_(conv, s)
    h = nn.PixelShuffle(s)(conv(torch.randn(2, 8, 7, 9)))
    cells = h.view(2, 8, 7, s, 9, s)
    assert torch.allclose(cells, cells[:, :, :, :1, :, :1].expand_as(cells), atol=1e-6)


def test_default_init_would_have_phase_imbalance():
    s = 5
    conv = nn.Conv2d(8, 8 * s * s, 3, padding=1)
    h = nn.PixelShuffle(s)(conv(torch.ones(1, 8, 6, 6)))
    assert float(phase_means(h).std()) > 1e-3


def test_pixelshuffle_cell_lies_inside_its_source_pixel_and_is_centred():
    # out px 5I+r spans [10I+2r, 10I+2r+2] m: inside source px I; r = 2 holds the source centre 10I+5
    s, src_m, dst_m = 5, 10.0, 2.0
    for I in range(4):
        for r in range(s):
            lo = (s * I + r) * dst_m
            assert I * src_m <= lo and lo + dst_m <= (I + 1) * src_m
        centre = (s * I + 2) * dst_m + dst_m / 2
        assert centre == I * src_m + src_m / 2
    x = torch.zeros(1, 25, 3, 3); x[0, :, 1, 1] = 1.0
    y = nn.PixelShuffle(5)(x)
    nz = y[0, 0].nonzero()
    assert nz.min(0).values.tolist() == [5, 5] and nz.max(0).values.tolist() == [9, 9]


def _model():
    m = SynapseProX5().eval()
    nn.init.normal_(m.to_delta.weight, std=1e-2)
    return m


def test_model_is_identity_at_construction_and_flat_field_is_phase_flat():
    m = SynapseProX5().eval()
    with torch.no_grad():
        out = m(torch.full((1, 10, 8, 8), 0.1))
    assert out["delta"].abs().max() == 0
    # backbone features of a flat field are not flat (scan start, zero padding), so the check is the
    # imbalance BETWEEN the 25 sub-pixel phases, measured 80x lower with ICNR than default init (0.006 vs 0.49)
    m = _model()
    with torch.no_grad():
        d = m(torch.full((1, 10, 16, 16), 0.1))["delta"][..., 20:-20, 20:-20]
    assert float(phase_means(d).std() / d.abs().mean()) < 0.05


def test_translation_by_one_source_pixel_moves_output_by_five():
    m = _model()
    y = torch.rand(1, 10, 14, 14)
    with torch.no_grad():
        a = m(y)["delta"]
        b = m(torch.roll(y, 1, -1))["delta"]
    a_i, b_i = a[..., 25:-25, 25:-30], b[..., 25:-25, 30:-25]
    rel = float((a_i - b_i).norm() / a_i.norm())
    # the raster scan is not exactly translation-equivariant (row wrap / scan start); record the bound
    assert rel < 0.5, rel
    print("translation_rel_err", rel)
