"""Numerical contracts of the fast paths: each must reproduce the exact reference it replaces."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from synapse_sr import Flash, Pro
from synapse_sr.io.sentinel2 import S2_12
from synapse_sr.models import baseline, projector
from synapse_sr.models.fastphys import FastPhysics
from synapse_sr.models.flash import SynapseFlashX5
from synapse_sr.models.forward import S2Forward
from synapse_sr.models.scan import chunked_selective_scan

from conftest import gaussian_operator, scene

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def rel(a, b):
    return float((a - b).detach().abs().max() / b.detach().abs().max())


def random_operator(khalf=23, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(4, 1, 2 * khalf + 1, 2 * khalf + 1, generator=g) ** 3
    return S2Forward(w / w.sum((-2, -1), keepdim=True), target_m=2.0)


# ------------------------------------------------------------------ forward operator
@pytest.mark.parametrize("op", [gaussian_operator(), random_operator()], ids=["gaussian", "asymmetric"])
@pytest.mark.parametrize("hw", [(60, 60), (61, 77), (95, 40), (150, 150)])
def test_target_grid_operator_equals_fine_grid_operator(op, hw):
    x = torch.rand(2, 4, *hw, dtype=torch.float64)
    op = op.double()
    a, b = op(x), op.forward_fine(x)
    assert a.shape == b.shape and rel(a, b) < 1e-10


# ------------------------------------------------------------------ selective scan
def sequential_scan(u, delta, A, B, C, D, delta_bias):
    u, delta, A, B, C = [t.double() for t in (u, delta, A, B, C)]
    delta = F.softplus(delta + delta_bias.double()[:, None])
    per = u.shape[1] // B.shape[1]
    Bx, Cx = B.repeat_interleave(per, 1), C.repeat_interleave(per, 1)
    h = torch.zeros(*u.shape[:2], A.shape[1], dtype=torch.float64)
    ys = []
    for t in range(u.shape[-1]):
        h = h * (delta[..., t, None] * A).exp() + (delta[..., t] * u[..., t])[..., None] * Bx[..., t]
        ys.append((h * Cx[..., t]).sum(-1))
    return torch.stack(ys, -1) + u * D.double()[None, :, None]


def scan_problem(scale, L, seed=0, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    b, k, per, n = 2, 2, 4, 8
    d = k * per
    t = lambda *s: torch.randn(*s, generator=g)
    p = dict(u=t(b, d, L), delta=t(b, d, L), A=-torch.rand(d, n, generator=g) * scale, B=t(b, k, n, L),
             C=t(b, k, n, L), D=t(d), delta_bias=t(d))
    if requires_grad:
        p["u"].requires_grad_(True); p["A"].requires_grad_(True)
    return p


@pytest.mark.parametrize("scale,L", [(1.0, 67), (8.0, 200), (40.0, 150), (400.0, 90)])
def test_chunked_scan_matches_sequential_at_every_decay_rate(scale, L):
    p = scan_problem(scale, L)
    y = chunked_selective_scan(**p, delta_softplus=True)
    assert torch.isfinite(y).all()
    assert rel(y.double(), sequential_scan(**p)) < 2e-6


def test_chunked_scan_ungrouped_B_and_gradients():
    p = scan_problem(1.0, 130)
    y3 = chunked_selective_scan(p["u"], p["delta"], p["A"], p["B"][:, 0], p["C"][:, 0], D=p["D"],
                                delta_bias=p["delta_bias"], delta_softplus=True)
    ref = sequential_scan(p["u"], p["delta"], p["A"], p["B"][:, :1], p["C"][:, :1], p["D"], p["delta_bias"])
    assert rel(y3.double(), ref) < 2e-6
    q = scan_problem(4.0, 70, requires_grad=True)
    qd = {k: (v.detach().double().requires_grad_(True) if k in ("u", "A") else v) for k, v in q.items()}
    g = torch.randn(2, 8, 70)
    (chunked_selective_scan(**q, delta_softplus=True) * g).sum().backward()
    (sequential_scan(**qd) * g).sum().backward()
    assert rel(q["u"].grad.double(), qd["u"].grad) < 1e-5 and rel(q["A"].grad.double(), qd["A"].grad) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_triton_scan_matches_chunked_when_available():
    from synapse_sr.models import scan_triton
    u = torch.zeros(1, device="cuda")
    if not scan_triton.available(u):
        pytest.skip("triton unavailable")
    p = {k: v.cuda() for k, v in scan_problem(2.0, 300).items()}
    a = scan_triton.triton_scan(**p, delta_softplus=True)
    b = chunked_selective_scan(**p, delta_softplus=True)
    assert rel(a, b) < 1e-4


# ------------------------------------------------------------------ physics solvers
@pytest.fixture(scope="module")
def physics():
    op = gaussian_operator()
    return op, FastPhysics(op)


def test_periodic_physics_is_exact_on_periodic_images(physics):
    op, fp = physics
    M = 24
    x = torch.rand(1, 4, 5 * M, 5 * M)
    big = x.repeat(1, 1, 3, 3)
    off, q = op.block(15 * M)
    idx = torch.arange(off, off + q) % M
    assert rel(fp.A(x)[:, :, idx][:, :, :, idx], op(big)) < 1e-5
    assert float(fp.A(fp.PN(x)).norm() / fp.A(x).norm()) < 1e-5


def test_preconditioning_changes_speed_not_answers(physics):
    op, fp = physics
    y = torch.tensor(scene(40, 40, dn=False)[[3, 2, 1, 7]])[None]
    lam = baseline.select_lambda(op, y)
    assert torch.equal(lam, baseline.select_lambda(op, y, fast=fp))
    a, b = baseline.window(op, y, lam), baseline.window(op, y, lam, fast=fp)
    assert float((a - b).abs().max()) < 1e-4 * float(a.abs().max())
    d = torch.randn(2, 4, 200, 200) * 0.01
    pa, pb = projector.apply(op, d), projector.apply(op, d, fast=fp)
    assert rel(pb, pa) < 1e-4
    assert float(op(pb).norm() / op(d).norm()) < 1e-4


# ------------------------------------------------------------------ Flash
@pytest.fixture(scope="module")
def flash():
    torch.manual_seed(1)
    net = SynapseFlashX5()
    torch.nn.init.normal_(net.to_delta[0].weight, std=1e-3)
    return Flash(net, gaussian_operator(), device="cpu", meta={"name": "flash-test"})


def test_reparameterised_flash_equals_training_form():
    torch.manual_seed(2)
    net = SynapseFlashX5().eval()
    y = torch.rand(2, 10, 24, 24)
    a = net(y)["delta"]
    b = net.reparameterise()(y)["delta"]
    assert rel(b, a) < 1e-4


def test_flash_checkpoint_roundtrip_and_class_dispatch(flash, model, tmp_path):
    p = flash.save_pretrained(str(tmp_path / "f.safetensors"))
    g = Pro.from_pretrained(weights=p, device="cpu")
    assert type(g) is Flash and "Flash(" in repr(g)
    arr = scene(24, 24, seed=3)
    a = flash.super_resolve(arr, band_names=S2_12, tile=24).image
    b = g.super_resolve(arr, band_names=S2_12, tile=24).image
    assert np.allclose(a, b, atol=1e-5)
    q = model.save_pretrained(str(tmp_path / "p.safetensors"))
    with pytest.raises(ValueError, match="Pro checkpoint"):
        Flash.from_pretrained(weights=q, device="cpu")


def test_flash_output_is_observation_consistent_and_tiles_seamlessly(flash):
    arr = scene(48, 48, seed=4)
    r = flash.super_resolve(arr, band_names=S2_12, tile=48, halo=0)
    assert r.metadata["scan_backend"] == "cnn" and r.image.shape == (4, 240, 240)
    x, xb = torch.tensor(r.image)[None], torch.tensor(r.x_base)[None]
    assert float((flash.op(x) - flash.op(xb)).abs().max()) < 1e-4 * float(flash.op(xb).abs().max())
    tiled = flash.super_resolve(arr, band_names=S2_12, tile=16, halo=12).image
    core = (slice(None), slice(60, -60), slice(60, -60))
    assert np.sqrt(((r.image[core] - tiled[core]) ** 2).mean()) / r.image[core].std() < 0.05


def test_progress_callback_and_summary(flash):
    seen = []
    r = flash.super_resolve(scene(40, 40), band_names=S2_12, tile=16, progress=lambda d, t: seen.append((d, t)))
    assert seen and seen[-1][0] == seen[-1][1]
    s = r.summary(print_=False)
    assert s["shape"] == [4, 200, 200] and set(s["support_fraction"]) == {"high", "medium", "low"}
    assert "Result(" in repr(r) and r.metadata["seconds"] >= 0


# ------------------------------------------------------------------ data and indices
def test_stac_offset_is_applied_exactly_once():
    from synapse_sr.data import boa_offset
    assert boa_offset({"s2:processing_baseline": "05.10", "earthsearch:boa_offset_applied": True}) == 0
    assert boa_offset({"s2:processing_baseline": "05.10"}) == -1000
    assert boa_offset({"s2:processing_baseline": "03.01"}) == 0
    assert boa_offset({}) == 0


def test_indices_stay_physical_on_dark_and_negative_reflectance(flash):
    arr = scene(24, 24, seed=6, dn=False)
    arr[:, :6, :6] = -0.01                                     # L2A below zero (dark water / shadow)
    r = flash.super_resolve(arr, band_names=S2_12, tile=24, offset=0.0)
    for k, v in r.indices().items():
        ok = v[np.isfinite(v)]
        assert ok.size and ok.min() >= -1.5 and ok.max() <= 1.5, k
    nd = r.indices()["ndvi"]
    assert np.nanmin(nd) >= -1 and np.nanmax(nd) <= 1 and np.isnan(nd[:20, :20]).any()


def test_one_call_model_selection(flash, tmp_path):
    import json
    import synapse_sr
    from synapse_sr.pretrained import registry
    p = flash.save_pretrained(str(tmp_path / "f.safetensors"))
    r = synapse_sr.super_resolve(scene(16, 16), band_names=S2_12, model="flash", weights=p, device="cpu", tile=16)
    assert r.metadata["scan_backend"] == "cnn"
    assert type(synapse_sr.load("flash", weights=p, device="cpu")) is Flash
    if not json.loads((registry.MANIFEST_DIR / "flash-v1.json").read_text())["url"].startswith("http"):
        with pytest.raises(RuntimeError, match="weights="):
            synapse_sr.load("flash")


def test_cli_errors_are_clean_messages(tmp_path, capsys):
    from synapse_sr.cli import main as cli
    assert cli([str(tmp_path / "missing.tif"), str(tmp_path / "o.tif")]) == 1
    assert "input not found" in capsys.readouterr().err
    src = tmp_path / "in.tif"
    src.write_text("not a tiff")
    assert cli([str(src), str(tmp_path / "o.tif"), "--model", "no-such-model"]) == 1
    err = capsys.readouterr().err
    assert "no-such-model" in err and "Traceback" not in err
