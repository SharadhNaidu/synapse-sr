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

    def forward(self, x):
        b, c, n_th, n_tw = x.shape
        blur = self._valid_conv(x.repeat_interleave(self.R, -2).repeat_interleave(self.R, -1))
        (offh, qh), (offw, qw) = self.block(n_th), self.block(n_tw)
        if qh <= 0 or qw <= 0:
            raise ValueError(f"target patch {n_th}x{n_tw} leaves no fully supported source pixel")
        oh, ow = offh * S - self.khalf, offw * S - self.khalf
        i = blur[..., oh:oh + qh * S, ow:ow + qw * S]
        return 0.25 * (i[..., 9::S, 9::S][..., :qh, :qw] + i[..., 9::S, 10::S][..., :qh, :qw]
                       + i[..., 10::S, 9::S][..., :qh, :qw] + i[..., 10::S, 10::S][..., :qh, :qw])
