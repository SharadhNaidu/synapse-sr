"""Sentinel-2 forward operator A: target-grid reflectance -> 10 m observation on the fully supported block.

Per-band PSF on a 0.5 m fine grid (VALID convolution, no wrap), sampled at source-pixel centres (the 2 x 2
fine-cell mean around 10k + 5 m). The kernels ship inside the checkpoint (`operator.weight`), so the package
needs no PSF tables at run time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

FINE_M = 0.5
S = 20                                    # fine cells per 10 m source pixel


class S2Forward(nn.Module):
    def __init__(self, weight, target_m=2.0, bands=("B04", "B03", "B02", "B08")):
        super().__init__()
        self.bands = tuple(bands)
        self.R = int(round(target_m / FINE_M))
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float32))
        self.khalf = self.weight.shape[-1] // 2

    def block(self, n_target):
        """(source offset, count) of the fully supported source pixels along one axis."""
        n_fine = n_target * self.R
        off = -(-self.khalf // S)
        q = (n_fine - off * S - self.khalf) // S
        return off, max(q, 0)

    def _valid_conv(self, fine):
        b, c, H, W = fine.shape
        K = self.weight.shape[-1]
        if b * H * W * K * K * fine.element_size() <= 1_000_000_000:
            return F.conv2d(fine, self.weight.to(fine.dtype), groups=c)
        w = self.weight[:, 0].to(fine.dtype)
        Hf, Wf = H + K - 1, W + K - 1
        full = torch.fft.irfft2(torch.fft.rfft2(fine, s=(Hf, Wf)) * torch.fft.rfft2(torch.flip(w, (-2, -1)), s=(Hf, Wf))[None],
                                s=(Hf, Wf))
        return full[..., K - 1:H, K - 1:W]

    def _target_kernel(self, dtype):
        """The operator as one strided convolution on the target grid.

        The fine image is x replicated over R x R cells and the output keeps only the 2 x 2 cell mean at a fixed
        phase of every S-th fine position, so summing the (2 x 2-averaged) PSF over the fine cells that fall in
        each target pixel gives an exactly equivalent kernel on the target grid with stride S / R."""
        key = (self.weight.device, dtype)
        if getattr(self, "_tk", None) is None or self._tk[0] != key:
            w = self.weight.double()
            w2 = 0.25 * (F.pad(w, (0, 1, 0, 1)) + F.pad(w, (1, 0, 0, 1)) + F.pad(w, (0, 1, 1, 0)) + F.pad(w, (1, 0, 1, 0)))
            base = -(-self.khalf // S) * S - self.khalf + 9
            rho, t0 = base % self.R, base // self.R
            k = w2.shape[-1] + rho
            nt = -(-k // self.R)
            w2 = F.pad(w2, (rho, nt * self.R - k, rho, nt * self.R - k))
            kt = w2.view(w2.shape[0], 1, nt, self.R, nt, self.R).sum((3, 5))
            self._tk = (key, kt.to(dtype), t0)
        return self._tk[1], self._tk[2]

    def forward(self, x):
        b, c, n_th, n_tw = x.shape
        (offh, qh), (offw, qw) = self.block(n_th), self.block(n_tw)
        if qh <= 0 or qw <= 0:
            raise ValueError(f"target patch {n_th}x{n_tw} leaves no fully supported source pixel")
        kt, t0 = self._target_kernel(x.dtype)
        st = S // self.R
        return F.conv2d(x[..., t0:, t0:], kt, stride=st, groups=c)[..., :qh, :qw]

    def forward_fine(self, x):
        b, c, n_th, n_tw = x.shape
        blur = self._valid_conv(x.repeat_interleave(self.R, -2).repeat_interleave(self.R, -1))
        (offh, qh), (offw, qw) = self.block(n_th), self.block(n_tw)
        if qh <= 0 or qw <= 0:
            raise ValueError(f"target patch {n_th}x{n_tw} leaves no fully supported source pixel")
        oh, ow = offh * S - self.khalf, offw * S - self.khalf
        i = blur[..., oh:oh + qh * S, ow:ow + qw * S]
        return 0.25 * (i[..., 9::S, 9::S][..., :qh, :qw] + i[..., 9::S, 10::S][..., :qh, :qw]
                       + i[..., 10::S, 9::S][..., :qh, :qw] + i[..., 10::S, 10::S][..., :qh, :qw])
