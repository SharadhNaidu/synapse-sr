"""Chunked selective scan: the Mamba recurrence in PyTorch, without a Python loop over the sequence.

The recurrence h[t] = a[t] h[t-1] + b[t] has the closed form h[t] = sum_{s<=t} exp(c[t] - c[s]) b[s] with
c = cumsum(log a), so within a chunk

    h[t] = exp(c[t] - o) * cumsum_s( exp(o - c[s]) b[s] )[t]

-- elementwise ops and one cumsum, no (chunk x chunk) matrix. o is half the chunk's total decay, which bounds
both exponents by half a chunk of decay; the state is then carried across chunks by an elementwise loop over the
(few) chunk indices. When some chunk decays too fast for float32 (> 120) the chunk is halved, down to one
step, where no exponential of an accumulated decay is formed at all. Forward error against a float64 sequential scan ~1.3e-7.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["chunked_selective_scan", "selective_scan_fn", "backend"]

SAFE_DECAY = 120.0


def chunked_selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, chunk=64):
    """u, delta (b, d, l); A (d, n); B, C (b, k, n, l) grouped or (b, n, l); D (d,)."""
    dtype_in = u.dtype
    b, d, L = u.shape
    n = A.shape[1]
    dt = delta.float()
    if delta_bias is not None:
        dt = dt + delta_bias.float()[:, None]
    if delta_softplus:
        dt = F.softplus(dt)
    uf = u.float()
    amax = A.detach().float().abs().amax(1)[None, :, None]                  # worst chunk decay, exactly, per chunk size
    while chunk > 1 and float((F.pad(dt.detach(), (0, (chunk - L % chunk) % chunk)).unflatten(-1, (-1, chunk)).sum(-1) * amax).max()) > SAFE_DECAY:
        chunk //= 2
    if B.dim() == 3:
        B, C = B[:, None], C[:, None]
    k = B.shape[1]
    per = d // k
    pad = (chunk - L % chunk) % chunk
    T = L + pad
    nc = T // chunk
    dt = F.pad(dt, (0, pad)).view(b, k, per, 1, nc, chunk)
    du = dt * F.pad(uf, (0, pad)).view(b, k, per, 1, nc, chunk)
    Bl = F.pad(B.float(), (0, pad)).view(b, k, 1, n, nc, chunk)
    Cl = F.pad(C.float(), (0, pad)).view(b, k, 1, n, nc, chunk)
    cum = (dt * A.float().view(1, k, per, n, 1, 1)).cumsum(-1)             # (b, k, per, n, nc, chunk), <= 0
    tot = cum[..., -1:]
    if chunk == 1:
        h = Bl * du
    else:
        off = tot * 0.5
        h = (cum - off).exp() * ((off - cum).exp() * (Bl * du)).cumsum(-1)
    decay = tot[..., 0].exp()                                                # (b, k, per, n, nc)
    last = h[..., -1]
    carry = torch.zeros_like(last[..., 0])
    prev = []
    for c in range(nc):
        prev.append(carry)
        carry = carry * decay[..., c] + last[..., c]
    h = h + cum.exp() * torch.stack(prev, -1)[..., None]
    y = (Cl * h).sum(3).reshape(b, d, T)[..., :L]
    if D is not None:
        y = y + uf * D.float()[None, :, None]
    return y.to(dtype_in)


import os
import warnings

selective_scan_cuda = None
if os.environ.get("SYNAPSE_SR_DISABLE_FUSED", "0") != "1":
    try:
        import selective_scan_cuda
    except ImportError:
        pass
    except Exception as e:                   # built against another torch / CUDA: undefined symbols, OSError
        warnings.warn(f"mamba-ssm selective_scan_cuda is installed but unusable ({e}); using the PyTorch scan",
                      RuntimeWarning)


class _FusedScan(torch.autograd.Function):
    """The mamba_ssm CUDA kernel called directly, so causal_conv1d is not required."""

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, delta_bias, delta_softplus):
        u, delta, B, C = u.contiguous(), delta.contiguous(), B.contiguous(), C.contiguous()
        D = D.contiguous() if D is not None else None
        out, x, *_ = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    def backward(ctx, dout):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        du, ddelta, dA, dB, dC, dD, ddb, *_ = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout.contiguous(), x, None, None, ctx.delta_softplus, False)
        return du, ddelta, dA, dB, dC, (dD if D is not None else None), (ddb if delta_bias is not None else None), None


def backend(u):
    """'fused' (mamba-ssm CUDA kernel), else 'triton' (CUDA GPU with Triton, self-tested once), else 'pytorch'."""
    if selective_scan_cuda is not None and u.is_cuda:
        return "fused"
    if u.is_cuda and os.environ.get("SYNAPSE_SR_DISABLE_TRITON", "0") != "1":
        from . import scan_triton
        if scan_triton.available(u):
            return "triton"
    return "pytorch"


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """Drop-in for mamba_ssm selective_scan_fn (options SYNAPSE uses): fused CUDA kernel when available,
    otherwise the chunked PyTorch scan (forward relative error 1.7e-7, gradients <= 5.8e-7 against the fused kernel)."""
    if z is not None or return_last_state:
        raise NotImplementedError("z gating / last-state return are not used by SYNAPSE")
    kind = backend(u)
    if kind == "fused" and B.dim() == 4:
        return _FusedScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus)
    if kind == "triton" and B.dim() == 4 and not torch.is_grad_enabled():
        from .scan_triton import triton_scan
        return triton_scan(u, delta, A, B, C, D=D, delta_bias=delta_bias, delta_softplus=delta_softplus)
    return chunked_selective_scan(u, delta, A, B, C, D=D, delta_bias=delta_bias, delta_softplus=delta_softplus)
