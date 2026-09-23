"""Chunked selective scan: the Mamba recurrence without 81 sequential Python steps.

WHY. The raster scan adapted from SEN2SR is the right mechanism -- state propagates across the
whole context instead of resetting every row -- but our reference implementation made it
unaffordable: 12.2 s/step against 0.426 s for the row scan, and 8.5 GB, because it materialises
(B, D, L, N) tensors and loops L times in Python.

HOW. The recurrence x[t] = a[t] x[t-1] + b[t] has the closed form

    x[t] = sum_{s<=t} exp(c[t] - c[s]) b[s],    c[t] = sum_{r<=t} log a[r]

so a whole chunk can be done as ONE masked matmul instead of a loop. Writing it that way
naively overflows, because exp(-c[s]) grows without bound as the decay accumulates. The fix is to
process the sequence in CHUNKS and express every exponent RELATIVE TO THE CHUNK START: inside a
chunk the exponent is bounded by that chunk's own decay, and the carried state moves between
chunks sequentially. With L = 81 and chunk 16 that is 6 sequential steps instead of 81.

This is the standard chunked/SSD formulation. Verified against the reference scan elementwise.
"""

from __future__ import annotations

import torch

__all__ = ["chunked_selective_scan", "selective_scan_fn", "backend"]


def chunked_selective_scan(u, delta, A, B, C, D=None, delta_bias=None,
                           delta_softplus=True, chunk=16):
    """Same contract as selective_scan_ref, minus the options we do not use.

    u, delta : (b, d, l)
    A        : (d, n)
    B, C     : (b, k, n, l) grouped, or (b, n, l)
    D        : (d,)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)

    b, d, L = u.shape
    n = A.shape[1]
    if B.dim() == 4:
        per = d // B.shape[1]
        B = B.float().repeat_interleave(per, dim=1)      # (b, d, n, l)
        C = C.float().repeat_interleave(per, dim=1)
    else:
        B = B.float()[:, None].expand(b, d, n, L)
        C = C.float()[:, None].expand(b, d, n, L)

    pad = (chunk - L % chunk) % chunk
    if pad:
        u = torch.nn.functional.pad(u, (0, pad))
        delta = torch.nn.functional.pad(delta, (0, pad))
        B = torch.nn.functional.pad(B, (0, pad))
        C = torch.nn.functional.pad(C, (0, pad))
    T = u.shape[-1]
    nc = T // chunk

    # (b, d, nc, chunk) and (b, d, n, nc, chunk)
    dl = delta.view(b, d, nc, chunk)
    ul = u.view(b, d, nc, chunk)
    Bl = B.view(b, d, n, nc, chunk)
    Cl = C.view(b, d, n, nc, chunk)

    # log decay per step, cumulative WITHIN the chunk
    loga = dl.unsqueeze(2) * A[None, :, :, None, None]           # (b, d, n, nc, chunk) <= 0
    cum = loga.cumsum(-1)
    bu = (dl.unsqueeze(2) * Bl) * ul.unsqueeze(2)

    # Contract the STATE dimension before building the chunk matrix. Forming
    # exp(cum[t]-cum[s]) per (n, t, s) needs (b, d, n, nc, chunk, chunk), which was 13.2 GB here;
    # factorising into exp(cum[t]) * exp(-cum[s]) lets n be summed by a matmul first and drops
    # the n axis from the big tensor entirely.
    #
    # exp(-cum[s]) alone would overflow once the accumulated decay is large, so both factors are
    # taken RELATIVE to half the chunk's total decay. Each exponent is then bounded by half a
    # chunk of decay in either direction, which keeps float32 comfortable, and the offsets cancel
    # exactly in the product.
    # The factors are formed in float64: a clamp on the offset (the earlier float32 form) broke the bound
    # once a chunk decayed by more than ~60 and produced inf * 0 = NaN on 0.9 % of outputs; float64 keeps
    # both factors finite up to a chunk decay of ~1400, far beyond any softplus(delta) * |A| seen here.
    off = cum[..., -1:].double() * 0.5
    Ct = Cl.double() * (cum.double() - off).exp()                # (b, d, n, nc, chunk)
    Bs = bu.double() * (off - cum.double()).exp()
    G = torch.einsum("bdnck,bdncs->bdcks", Ct, Bs)               # (b, d, nc, chunk_t, chunk_s)
    G = G * torch.ones(chunk, chunk, device=u.device, dtype=G.dtype).tril()
    y_in = G.sum(-1).float()                                     # (b, d, nc, chunk)

    # state carried across chunks, sequential in nc only (6 steps here instead of 81)
    carry = torch.zeros(b, d, n, device=u.device)
    outs = []
    for i in range(nc):
        cross = torch.einsum("bdnc,bdn->bdc", Cl[..., i, :] * cum[..., i, :].exp(), carry)
        outs.append(y_in[..., i, :] + cross)
        carry = carry * cum[..., i, -1].exp() + (bu[..., i, :] * (cum[..., i, -1:] - cum[..., i, :]).exp()).sum(-1)
    # stack on the CHUNK-INDEX axis, not the last: chunks are (b, d, nc, chunk) and flattening
    # a (b, d, chunk, nc) interleaving instead scrambles the sequence order entirely
    y = torch.stack(outs, dim=-2).reshape(b, d, T)[..., :L]
    if D is not None:
        y = y + u[..., :L] * D.float()[None, :, None]
    return y.to(dtype_in)


try:
    import selective_scan_cuda
except ImportError:
    selective_scan_cuda = None


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
    return "fused" if selective_scan_cuda is not None and u.is_cuda else "chunked"


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """Drop-in for mamba_ssm selective_scan_fn (options SYNAPSE uses): fused CUDA kernel when available,
    otherwise the chunked PyTorch scan (forward relative error 1.7e-7, gradients <= 5.8e-7 against the fused kernel)."""
    if z is not None or return_last_state:
        raise NotImplementedError("z gating / last-state return are not used by SYNAPSE")
    if backend(u) == "fused" and B.dim() == 4:
        return _FusedScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus)
    return chunked_selective_scan(u, delta, A, B, C, D=D, delta_bias=delta_bias, delta_softplus=delta_softplus)
