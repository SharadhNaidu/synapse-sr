"""Deterministic observation-consistent baseline x_base (Tikhonov-Morozov), no learned parameters.

anchor: bicubic sample at target-pixel centres (border padding). Correction: x = x_bic + A^T z with
(A A^T + lam_b I) z = y - A x_bic, lam_b chosen per band so RMS(A x - y)_b = tau_b (Morozov), a band already
within tau_b is left as bicubic. lam_b is one scene-level value, the median over deterministic calibration
windows (60 source px, the patch size the training baseline was computed on); the scene is solved in tiles
with a real-context halo.
"""

import torch
import torch.nn.functional as F

from synapse.models.forward import S

TAU_L2A = {"B04": 0.00068, "B03": 0.00084, "B02": 0.00086, "B08": 0.00192}
NO_CORRECTION = 1e6


def tau_for(op):
    return torch.tensor([TAU_L2A[b] for b in op.bands], dtype=torch.float32, device=op.weight.device)


def n_target(op, n_src):
    v = n_src * S / op.R
    if abs(v - round(v)) > 1e-9:
        raise ValueError(f"{n_src} source px is not a whole number of target px (needs a multiple of {op.R})")
    return int(round(v))


def anchor(op, y, n_th, n_tw):
    b, c, nh, nw = y.shape

    def axis(n_t, ns):
        pos = (torch.arange(n_t, device=y.device, dtype=torch.float32) + 0.5) * op.R / S
        return (pos / ns) * 2 - 1
    yy, xx = torch.meshgrid(axis(n_th, nh), axis(n_tw, nw), indexing="ij")
    grid = torch.stack([xx, yy], -1)[None].expand(b, -1, -1, -1).to(y.dtype)
    return F.grid_sample(y, grid, mode="bicubic", padding_mode="border", align_corners=False)


def tikhonov(op, x_bic, y_block, lam, max_iters=80, tol=1e-4):
    x0 = x_bic.detach()
    lam = torch.as_tensor(lam, dtype=x0.dtype, device=x0.device).view(1, -1, 1, 1)

    def AT(r):
        v = torch.zeros_like(x0, requires_grad=True)
        return torch.autograd.grad((op(v) * r).sum(), v)[0]

    with torch.enable_grad():
        rhs = y_block - op(x0)
        z = torch.zeros_like(rhs); r = rhs.clone(); p = r.clone()
        rs = (r * r).sum(dim=(-2, -1), keepdim=True)
        n0 = rhs.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt() + 1e-30
        for _ in range(max_iters):
            Ap = op(AT(p)) + lam * p
            alpha = rs / ((p * Ap).sum(dim=(-2, -1), keepdim=True) + 1e-30)
            z = z + alpha * p
            r = r - alpha * Ap
            rs_new = (r * r).sum(dim=(-2, -1), keepdim=True)
            if float((rs_new.sqrt() / n0).max()) < tol:
                break
            p = r + (rs_new / (rs + 1e-30)) * p
            rs = rs_new
        x = (x0 + AT(z)).detach()
    skip = (lam >= NO_CORRECTION).view(-1)
    if bool(skip.any()):
        x[:, skip] = x0[:, skip]
    return x


def morozov_lambda(op, x_bic, y_block, tau, n_bisect=16, lo=-9.0, hi=2.0):
    lo = torch.full_like(tau, lo); hi = torch.full_like(tau, hi)
    for _ in range(n_bisect):
        mid = (lo + hi) / 2
        x = tikhonov(op, x_bic, y_block, torch.exp(mid), max_iters=40)
        with torch.no_grad():
            big = (op(x) - y_block).pow(2).mean(dim=(0, -2, -1)).sqrt() > tau
        hi = torch.where(big, mid, hi); lo = torch.where(big, lo, mid)
    lam = torch.exp((lo + hi) / 2)
    with torch.no_grad():
        r0 = (op(x_bic) - y_block).pow(2).mean(dim=(0, -2, -1)).sqrt()
    return torch.where(r0 <= tau, torch.full_like(lam, NO_CORRECTION), lam)


def window(op, y, lam):
    h, w = y.shape[-2:]
    n_th, n_tw = n_target(op, h), n_target(op, w)
    with torch.no_grad():
        xb = anchor(op, y, n_th, n_tw)
    (oh, qh), (ow, qw) = op.block(n_th), op.block(n_tw)
    if qh <= 0 or qw <= 0:
        return xb
    return tikhonov(op, xb, y[..., oh:oh + qh, ow:ow + qw].to(xb.dtype), lam)


def calibration_windows(H, W, win=60, n=5):
    if H < win or W < win:
        return [(0, 0, H, W)]
    cands = [((H - win) // 2, (W - win) // 2)]
    for fr in ((1, 1), (2, 2), (1, 2), (2, 1)):
        cands.append(((H - win) * fr[0] // 3, (W - win) * fr[1] // 3))
    out = []
    for r, c in cands[:n]:
        r -= r % 3; c -= c % 3
        out.append((r, c, r + win, c + win))
    return out


def select_lambda(op, y, n_bisect=16):
    tau = tau_for(op)
    lams = []
    for r0, c0, r1, c1 in calibration_windows(y.shape[-2], y.shape[-1]):
        yw = y[..., r0:r1, c0:c1]
        n_th, n_tw = n_target(op, r1 - r0), n_target(op, c1 - c0)
        with torch.no_grad():
            xb = anchor(op, yw, n_th, n_tw)
        (oh, qh), (ow, qw) = op.block(n_th), op.block(n_tw)
        lams.append(morozov_lambda(op, xb, yw[..., oh:oh + qh, ow:ow + qw].to(xb.dtype), tau, n_bisect))
    return torch.stack(lams).median(dim=0).values if len(lams) > 1 else lams[0]
