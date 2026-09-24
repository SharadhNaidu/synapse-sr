"""Triton selective scan (inference): the Mamba recurrence with the state held in registers.

Used on CUDA GPUs where the mamba-ssm kernel is not installed (Colab, Kaggle, most pip installs). Each program owns
BLOCK_D channels of one sample and walks the sequence once:

    h[t] = exp(dt[t] * A) * h[t-1] + dt[t] * B[t] * u[t],   y[t] = C[t] . h[t] + D * u[t],   dt = softplus(delta + bias)

so nothing of size (d, n, L) is ever materialised. A self-test against the exact PyTorch scan runs once per
process before the kernel is used; any error or mismatch disables it silently.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:                                  # triton is optional (absent on CPU-only and Windows installs)
    triton = None

_STATE = {"ok": None}


if triton is not None:
    @triton.jit
    def _scan_kernel(u_ptr, dt_ptr, a_ptr, b_ptr, c_ptr, d_ptr, bias_ptr, y_ptr,
                     L, DIM, K, PER,
                     HAS_D: tl.constexpr, HAS_BIAS: tl.constexpr, SOFTPLUS: tl.constexpr,
                     NSTATE: tl.constexpr, BLOCK_D: tl.constexpr, TCH: tl.constexpr):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        d_idx = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        n_idx = tl.arange(0, NSTATE)
        dmask = d_idx < DIM
        k = (pid_d * BLOCK_D) // PER                                          # BLOCK_D divides PER: one group per program
        A = tl.load(a_ptr + d_idx[:, None] * NSTATE + n_idx[None, :], mask=dmask[:, None], other=0.0)
        bias = tl.zeros([BLOCK_D], tl.float32)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + d_idx, mask=dmask, other=0.0)
        Dv = tl.zeros([BLOCK_D], tl.float32)
        if HAS_D:
            Dv = tl.load(d_ptr + d_idx, mask=dmask, other=0.0)
        h = tl.zeros([BLOCK_D, NSTATE], tl.float32)
        ub = u_ptr + pid_b * L * DIM                                           # (b, L, d) layout
        tb = dt_ptr + pid_b * L * DIM
        bb = b_ptr + (pid_b * K + k) * L * NSTATE                              # (b, k, L, n) layout
        cb = c_ptr + (pid_b * K + k) * L * NSTATE
        yb = y_ptr + pid_b * L * DIM
        for t0 in range(0, L, TCH):                                            # unrolled: loads do not depend on h,
            for tt in tl.static_range(TCH):                                    # so they are issued ahead of the recurrence
                t = t0 + tt
                tm = t < L
                u = tl.load(ub + t * DIM + d_idx, mask=dmask & tm, other=0.0)
                dt = tl.load(tb + t * DIM + d_idx, mask=dmask & tm, other=0.0) + bias
                if SOFTPLUS:
                    dt = tl.where(dt > 20.0, dt, tl.log(1.0 + tl.exp(dt)))
                Bv = tl.load(bb + t * NSTATE + n_idx, mask=(n_idx < NSTATE) & tm, other=0.0)
                Cv = tl.load(cb + t * NSTATE + n_idx, mask=(n_idx < NSTATE) & tm, other=0.0)
                h = h * tl.exp(dt[:, None] * A) + (dt * u)[:, None] * Bv[None, :]
                y = tl.sum(h * Cv[None, :], axis=1) + Dv * u
                tl.store(yb + t * DIM + d_idx, y, mask=dmask & tm)


BLOCK_D, TCH = 32, 8


def _block_d(per):
    for bd in (BLOCK_D, 16, 8, 4, 2, 1):
        if per % bd == 0:
            return bd
    return 1


def triton_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
    """u, delta (b, d, L); A (d, n); B, C (b, k, n, L) grouped. Returns y (b, d, L) in u's dtype."""
    b, d, L = u.shape
    n = A.shape[1]
    k = B.shape[1]
    per = d // k
    bd = _block_d(per)
    ut = u.float().transpose(1, 2).contiguous()
    dtt = delta.float().transpose(1, 2).contiguous()
    Bt = B.float().transpose(2, 3).contiguous()
    Ct = C.float().transpose(2, 3).contiguous()
    y = torch.empty(b, L, d, device=u.device, dtype=torch.float32)
    Af = A.float().contiguous()
    Df = D.float().contiguous() if D is not None else Af
    bf = delta_bias.float().contiguous() if delta_bias is not None else Af
    _scan_kernel[(b, triton.cdiv(d, bd))](ut, dtt, Af, Bt, Ct, Df, bf, y, L, d, k, per,
                                          HAS_D=D is not None, HAS_BIAS=delta_bias is not None,
                                          SOFTPLUS=bool(delta_softplus), NSTATE=n, BLOCK_D=bd, TCH=TCH)
    return y.transpose(1, 2).to(u.dtype)


def available(u):
    """True once, per process, if Triton runs here and matches the exact scan on a random problem."""
    if triton is None or not u.is_cuda:
        return False
    if _STATE["ok"] is None:
        try:
            from .scan import chunked_selective_scan
            g = torch.Generator(device=u.device).manual_seed(0)
            b, k, per, n, L = 1, 2, 8, 16, 67
            d = k * per
            uu = torch.randn(b, d, L, device=u.device, generator=g)
            dd = torch.randn(b, d, L, device=u.device, generator=g)
            AA = -torch.rand(d, n, device=u.device, generator=g) * 4
            BB = torch.randn(b, k, n, L, device=u.device, generator=g)
            CC = torch.randn(b, k, n, L, device=u.device, generator=g)
            DD = torch.randn(d, device=u.device, generator=g)
            bb = torch.randn(d, device=u.device, generator=g)
            ref = chunked_selective_scan(uu, dd, AA, BB, CC, D=DD, delta_bias=bb, delta_softplus=True)
            out = triton_scan(uu, dd, AA, BB, CC, D=DD, delta_bias=bb, delta_softplus=True)
            _STATE["ok"] = bool(torch.isfinite(out).all()) and float((out - ref).abs().max() / (ref.abs().max() + 1e-12)) < 1e-4
        except Exception:
            _STATE["ok"] = False
    return _STATE["ok"]
