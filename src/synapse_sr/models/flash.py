"""SynapseFlashX5: a fast convolutional detail network for x_hat = x_base + P_N(delta).

The body is the SPAN-style network of SEN2SR-Lite (ESAOpenSR/SEN2SR, CC0-1.0; see THIRD_PARTY_NOTICES) and was
initialised from its public weights. Heads are new: a x5 PixelShuffle detail head and a x5 error-scale head. The
re-parameterisable Conv3XC blocks train as three-branch convolutions and are folded into one 3x3 convolution each
for inference (`reparameterise`).
"""

import torch
import torch.nn.functional as F
from torch import nn

SCALE = 5


def conv(cin, cout, k=3):
    return nn.Conv2d(cin, cout, k, padding=(k - 1) // 2)


class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain=2):
        super().__init__()
        self.sk = nn.Conv2d(c_in, c_out, 1)
        self.conv = nn.Sequential(nn.Conv2d(c_in, c_in * gain, 1), nn.Conv2d(c_in * gain, c_out * gain, 3),
                                  nn.Conv2d(c_out * gain, c_out, 1))
        self.eval_conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.eval_conv.weight.requires_grad_(False)
        self.eval_conv.bias.requires_grad_(False)
        self.folded = False

    @torch.no_grad()
    def fold(self):
        w1, b1 = self.conv[0].weight, self.conv[0].bias
        w2, b2 = self.conv[1].weight, self.conv[1].bias
        w3, b3 = self.conv[2].weight, self.conv[2].bias
        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
        wc = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3).flip(2, 3).permute(1, 0, 2, 3)
        bc = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
        self.eval_conv.weight.copy_(wc + F.pad(self.sk.weight, [1, 1, 1, 1]))
        self.eval_conv.bias.copy_(bc + self.sk.bias)
        self.folded = True

    def forward(self, x):
        if self.folded:
            return self.eval_conv(x)
        return self.conv(F.pad(x, (1, 1, 1, 1))) + self.sk(x)


class SPAB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1_r, self.c2_r, self.c3_r = Conv3XC(c, c), Conv3XC(c, c), Conv3XC(c, c)

    def forward(self, x):
        out1 = F.silu(self.c1_r(x))                 # SEN2SR-Lite applies SiLU in place, so its returned out1 is activated
        out3 = self.c3_r(F.silu(self.c2_r(out1)))
        return (out3 + x) * (torch.sigmoid(out3) - 0.5), out1


class SynapseFlashX5(nn.Module):
    def __init__(self, feat=24, blocks=6, scale=SCALE):
        super().__init__()
        self.scale = scale
        self.conv_1 = Conv3XC(4, feat)
        self.blocks = nn.ModuleList([SPAB(feat) for _ in range(blocks)])
        self.conv_cat = conv(feat * 4, feat, 1)
        self.conv_2 = Conv3XC(feat, feat)
        self.to_delta = nn.Sequential(conv(feat, 4 * scale * scale), nn.PixelShuffle(scale))
        self.to_conf = nn.Sequential(conv(feat, 16), nn.GELU(), conv(16, 4 * scale * scale), nn.PixelShuffle(scale))
        with torch.no_grad():
            self.to_delta[0].weight.mul_(0.1)
            self.to_delta[0].bias.zero_()

    def features(self, y):
        f0 = self.conv_1(y)
        for i, blk in enumerate(self.blocks):                 # SEN2SR-Lite: every block reads f0 (parallel, not chained)
            out, o1 = blk(f0)
            if i == 0:
                b1 = out
        return self.conv_cat(torch.cat([f0, self.conv_2(out), b1, o1], 1))

    def forward(self, y10):
        f = self.features(y10[:, :4])
        return {"delta": self.to_delta(f), "conf_logit": self.to_conf(f)}

    def load_lite_body(self, state_dict):
        """Body weights from a SEN2SR-Lite RGBN checkpoint (keys conv_1, blocks, conv_cat, conv_2)."""
        body = {k: v for k, v in state_dict.items() if k.startswith(("conv_1.", "blocks.", "conv_cat.", "conv_2."))
                and not k.endswith(("eval_conv.weight", "eval_conv.bias"))}
        missing, unexpected = self.load_state_dict(body, strict=False)
        return {"loaded": len(body), "unexpected": list(unexpected)}

    def reparameterise(self):
        for m in self.modules():
            if isinstance(m, Conv3XC):
                m.fold()
        return self
