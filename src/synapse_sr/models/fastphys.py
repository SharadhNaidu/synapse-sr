"""Closed-form Sentinel-2 physics on periodic windows: A, A^T, the observation-complement projector P_N and the
Tikhonov baseline, each one FFT and a per-frequency formula instead of conjugate gradients.

The nominal operator is equivariant under shifts of one 10 m pixel (5 target px). On a periodic window of
Mh x Mw source px it is a polyphase filter bank, y_hat(w) = h(w) . x_poly(w) with a 25-vector h(w) per band, so

    A A^T (w) = ||h(w)||^2,
    P_N x      = x - A^T [ A x / ||h||^2 ],
    Tikhonov   : x = x_bic + A^T [ (y - A x_bic) / (||h||^2 + lam) ].

The 25 lag kernels are measured once by pushing impulses through the exact operator. Windows are processed with
a real-context halo, which absorbs the wrap-around at their edges.
"""

import torch

S = 5


class FastPhysics:
    def __init__(self, op, lag=12):
        self.op = op
        self.lag = lag
        self.k = self._lag_kernels(op, lag)          # (C, 5, 5, 2*lag, 2*lag), lag 0 at index [lag, lag]
        self._cache = {}

    @staticmethod
    @torch.no_grad()
    def _lag_kernels(op, lag):
        pad = lag + 20
        n = 5 * (2 * pad)
        off, q = op.block(n)
        dev, nb = op.weight.device, op.weight.shape[0]
        k = torch.zeros(nb, S, S, 2 * lag, 2 * lag, device=dev)
        c0 = off + q // 2
        idx = torch.arange(c0 - off - lag, c0 - off + lag, device=dev)
        for p in range(S):
            x = torch.zeros(S, nb, n, n, device=dev)
            for r in range(S):
                x[r, :, 5 * c0 + p, 5 * c0 + r] = 1.0
            y = op(x)                                    # (5, nb, q, q)
            for r in range(S):
                k[:, p, r] = y[r][:, idx][:, :, idx]
        return k

    def _H(self, mh, mw, dtype):
        key = (mh, mw, dtype)
        if key not in self._cache:
            L = self.lag
            per = torch.zeros(*self.k.shape[:3], mh, mw, device=self.k.device)
            for i in range(2 * L):
                for j in range(2 * L):
                    per[..., (i - L) % mh, (j - L) % mw] += self.k[..., i, j]
            H = torch.fft.fft2(per)
            self._cache[key] = (H, (H.abs() ** 2).sum((1, 2)).clamp(min=1e-12))
        return self._cache[key]

    @staticmethod
    def _poly(x, mh, mw):
        b, c = x.shape[:2]
        return x.reshape(b, c, mh, S, mw, S).permute(0, 1, 3, 5, 2, 4)

    @staticmethod
    def _unpoly(xp):
        b, c, _, _, mh, mw = xp.shape
        return xp.permute(0, 1, 4, 2, 5, 3).reshape(b, c, S * mh, S * mw)

    def _A_hat(self, x):
        mh, mw = x.shape[-2] // S, x.shape[-1] // S
        H, HH = self._H(mh, mw, x.dtype)
        X = torch.fft.fft2(self._poly(x, mh, mw))
        return (H[None] * X).sum((2, 3)), H, HH

    def _AT_hat(self, Yh, H):
        return self._unpoly(torch.fft.ifft2(H[None].conj() * Yh[:, :, None, None]).real)

    def A(self, x):
        Yh, _, _ = self._A_hat(x)
        return torch.fft.ifft2(Yh).real

    def PN(self, x):
        Yh, H, HH = self._A_hat(x)
        return x - self._AT_hat(Yh / HH[None], H)

    def tikhonov(self, x_bic, y, lam):
        """x_bic (B, C, 5Mh, 5Mw), y (B, C, Mh, Mw) on the same periodic window, lam (C,) or (B, C)."""
        Yh, H, HH = self._A_hat(x_bic)
        R = torch.fft.fft2(y) - Yh
        lam = torch.as_tensor(lam, dtype=R.real.dtype, device=R.device)
        lam = lam.view(-1, lam.shape[-1], 1, 1) if lam.dim() == 2 else lam.view(1, -1, 1, 1)
        return x_bic + self._AT_hat(R / (HH[None] + lam), H)

    def gram_precond(self, qh, qw, lam=0.0):
        """Returns r -> (A A^T + lam)^-1 r for the periodic (qh, qw) source grid: a preconditioner for CG on the
        exact, finite Gram operator. It changes the number of iterations, never the solution."""
        _, HH = self._H(qh, qw, torch.float32)
        def apply(r):
            lam_ = torch.as_tensor(lam, dtype=r.dtype, device=r.device)
            lam_ = lam_.view(-1, lam_.shape[-1], 1, 1) if lam_.dim() == 2 else lam_.reshape(1, -1, 1, 1) if lam_.dim() == 1 else lam_
            return torch.fft.ifft2(torch.fft.fft2(r) / (HH[None] + lam_ + 1e-12)).real
        return apply
