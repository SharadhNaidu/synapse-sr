"""Observation-complement projector P_N = I - A^T (A A^T)^+ A and the affine data-consistency step.

x = x_base + P_N(delta) changes nothing A can see, so A x = A x_base. P_N is linear and symmetric: its
vector-Jacobian product is P_N itself (one more solve), constant in memory.
"""

import torch


def adjoint(op, ref, r):
    with torch.enable_grad():
        v = torch.zeros_like(ref).requires_grad_(True)
        return torch.autograd.grad((op(v) * r.detach()).sum(), v)[0].detach()


def gram_solve(op, ref, rhs, iters=200, tol=1e-5):
    """(A A^T) z = rhs by conjugate gradients, per sample and band."""
    rhs = rhs.detach()
    z = torch.zeros_like(rhs); r = rhs.clone(); p = r.clone()
    rs = (r * r).sum(dim=(-2, -1), keepdim=True)
    n0 = rhs.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt() + 1e-30
    for _ in range(iters):
        with torch.no_grad():
            Ap = op(adjoint(op, ref, p))
        alpha = rs / ((p * Ap).sum(dim=(-2, -1), keepdim=True) + 1e-30)
        z = z + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum(dim=(-2, -1), keepdim=True)
        if float((rs_new.sqrt() / n0).max()) < tol:
            break
        p = r + (rs_new / (rs + 1e-30)) * p
        rs = rs_new
    return z


def apply(op, d, iters=200, tol=1e-5):
    d = d.detach().float()
    with torch.no_grad():
        z = gram_solve(op, d, op(d), iters, tol)
    return d - adjoint(op, d, z)


class _Project(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, op, iters, tol):
        ctx.op, ctx.iters, ctx.tol = op, iters, tol
        return apply(op, d, iters, tol)

    @staticmethod
    def backward(ctx, g):
        return apply(ctx.op, g.contiguous(), ctx.iters, ctx.tol), None, None, None


def project(op, d, iters=200, tol=1e-5):
    return _Project.apply(d, op, iters, tol)
