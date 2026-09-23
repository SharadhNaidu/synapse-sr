"""SYNAPSE Pro X5: 10 m Sentinel-2 -> 2.0 m RGBN, x5, observation-complement constrained.

    y10 (10 bands)  ->  RGBN spatial stem  +  tanh(g_aux) * spectral-context stem
                    ->  state-space structural backbone (6 residual groups x 8 VSS blocks)
                    ->  frequency-split mixer (low / high band re-fused, zero-initialised)
                    ->  direct x5 reconstruction head (PixelShuffle 5, zero-initialised output)
                    ->  delta_theta (2 m, RGBN)
    x_hat = x_base + P_N(delta_theta)             (projector applied by the caller / SynapsePro wrapper)
    confidence head: per-pixel support logits from the reconstruction features (NOT calibrated uncertainty)

Initialisation contract: delta_theta == 0 exactly at construction (zero output conv), and the auxiliary
spectral stem and frequency mixer contribute exactly zero (bounded gate at 0, zero-initialised mixer output),
so x_hat == x_base at step 0. The backbone may be initialised from compatible third-party MambaSR weights
(`load_backbone`) - a training-initialisation option only; the trained SYNAPSE checkpoint is self-contained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from synapse_sr.models.mamba import MambaSR

RGBN = ("B04", "B03", "B02", "B08")
AUX = ("B05", "B06", "B07", "B8A", "B11", "B12")
INPUT_BANDS = RGBN + AUX

BACKBONE = dict(img_size=(64, 64), in_channels=4, out_channels=4, embed_dim=96, depths=[8] * 6,
                num_heads=[8] * 6, mlp_ratio=4, upscale=4, attention_type="sigmoid_02",
                upsampler="pixelshuffle", resi_connection="1conv", operation_attention="sum")


def icnr_(conv, scale):
    """All scale^2 PixelShuffle phases start from the same kernel, so the x5 head has no cell-period
    (checkerboard) pattern at initialisation; phases may diverge only as the data demands."""
    out = conv.out_channels // (scale * scale)
    w = torch.empty(out, conv.in_channels, *conv.kernel_size)
    nn.init.kaiming_normal_(w, a=0.2)
    with torch.no_grad():
        conv.weight.copy_(w.repeat_interleave(scale * scale, 0))
        conv.bias.zero_()


class FrequencyMixer(nn.Module):
    """Split features into a local low band and its high-band complement, mix each, re-fuse residually."""

    def __init__(self, dim, k=5):
        super().__init__()
        self.k = k
        self.low = nn.Conv2d(dim, dim, 1)
        self.high = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.Conv2d(dim, dim, 1))
        self.out = nn.Conv2d(2 * dim, dim, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, f):
        lo = F.avg_pool2d(F.pad(f, [self.k // 2] * 4, mode="replicate"), self.k, stride=1)
        return f + self.out(torch.cat([self.low(lo), self.high(f - lo)], 1))


class SynapseProX5(nn.Module):
    def __init__(self, scale=5, n_aux=len(AUX), feat=64):
        super().__init__()
        self.scale = scale
        bb = MambaSR(**BACKBONE)
        self.rgbn_stem = bb.conv_first                     # 4 -> 96
        self.patch_embed, self.patch_unembed, self.pos_drop = bb.patch_embed, bb.patch_unembed, bb.pos_drop
        self.layers, self.norm, self.conv_after_body = bb.layers, bb.norm, bb.conv_after_body
        self.conv_before_upsample = bb.conv_before_upsample   # 96 -> 64 + LeakyReLU
        E = BACKBONE["embed_dim"]
        self.aux_stem = nn.Conv2d(n_aux, E, 3, padding=1)
        self.aux_gate = nn.Parameter(torch.zeros(1))       # tanh(0) = 0: auxiliary context off at init
        self.freq = FrequencyMixer(E)
        self.up = nn.Sequential(nn.Conv2d(feat, feat * scale * scale, 3, padding=1), nn.PixelShuffle(scale),
                                nn.LeakyReLU(0.2, inplace=True))
        icnr_(self.up[0], scale)
        self.to_delta = nn.Conv2d(feat, 4, 3, padding=1)
        nn.init.zeros_(self.to_delta.weight); nn.init.zeros_(self.to_delta.bias)
        self.to_conf = nn.Sequential(nn.Conv2d(feat, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 4, 1))

    def load_backbone(self, state_dict):
        """Initialise the backbone from compatible MambaSR weights (training-initialisation option)."""
        own = self.state_dict()
        remap = {"conv_first.": "rgbn_stem."}
        take = {}
        for k, v in state_dict.items():
            k2 = k
            for a, b in remap.items():
                if k.startswith(a):
                    k2 = b + k[len(a):]
            if k2 in own and own[k2].shape == v.shape:
                take[k2] = v
        own.update(take)
        self.load_state_dict(own)
        return {"transferred_tensors": len(take), "transferred_params": int(sum(v.numel() for v in take.values()))}

    def features(self, y10):
        rgbn, aux = y10[:, :4], y10[:, 4:]
        f = self.rgbn_stem(rgbn) + torch.tanh(self.aux_gate) * self.aux_stem(aux)
        size = (f.shape[2], f.shape[3])
        z = self.pos_drop(self.patch_embed(f))
        for layer in self.layers:
            z = layer(z, size)
        z = self.patch_unembed(self.norm(z), size)
        f = self.conv_after_body(z) + f
        f = self.freq(f)
        return self.up(self.conv_before_upsample(f))

    def forward(self, y10):
        """y10: (B, 10, H, W) reflectance in INPUT_BANDS order -> dict(delta, conf_logit) at (5H, 5W)."""
        h = self.features(y10)
        # detached: the support-map NLL must not steer the reconstruction trunk (at step 51-76 it dominated the
        # clipped gradient, NLL -3.6 vs Charbonnier 0.014)
        return {"delta": self.to_delta(h), "conf_logit": self.to_conf(h.detach())}
