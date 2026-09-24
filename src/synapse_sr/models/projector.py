"""Observation-complement projector P_N = I - A^T (A A^T)^+ A and the affine data-consistency step.

x = x_base + P_N(delta) changes nothing A can see, so A x = A x_base. P_N is linear and symmetric: its
vector-Jacobian product is P_N itself (one more solve), constant in memory.
"""

import torch


def adjoint(op, ref, r):
    with torch.enable_grad():
        v = torch.zeros_like(ref).requires_grad_(True)
        return torch.autograd.grad((op(v) * r.detach()).sum(), v)[0].detach()


def gram_solve(op, ref, rhs, iters=200, tol=1e-5, precond=None):
    """(A A^T) z = rhs by (preconditioned) conjugate gradients, per sample and band. Stops on the true residual
    ||rhs - A A^T z|| / ||rhs|| < tol, so a preconditioner changes the iteration count, not the answer."""
    rhs = rhs.detach()
    M = precond or (lambda v: v)
    z = torch.zeros_like(rhs); r = rhs.clone(); s = M(r); p = s.clone()
    rs = (r * s).sum(dim=(-2, -1), keepdim=True)
    n0 = rhs.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt() + 1e-30
    for _ in range(iters):
        with torch.no_grad():
            Ap = op(adjoint(op, ref, p))
        alpha = rs / ((p * Ap).sum(dim=(-2, -1), keepdim=True) + 1e-30)
        z = z + alpha * p
        r = r - alpha * Ap
        if float((r.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt() / n0).max()) < tol:
            break
        s = M(r)
        rs_new = (r * s).sum(dim=(-2, -1), keepdim=True)
        p = s + (rs_new / (rs + 1e-30)) * p
        rs = rs_new
    return z


def apply(op, d, iters=200, tol=1e-5, fast=None):
    d = d.detach().float()
    with torch.no_grad():
        y = op(d)
        pre = fast.gram_precond(y.shape[-2], y.shape[-1]) if fast is not None else None
        z = gram_solve(op, d, y, iters, tol, pre)
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
