"""SYNAPSE Pro: Sentinel-2 10 m -> 2.0 m RGBN, x_hat = x_base + P_N(delta).

    from synapse import Pro
    model = Pro.from_pretrained()
    result = model.super_resolve("sentinel2.tif")
    result.save("sentinel2_2m.tif")
"""

import numpy as np
import safetensors.torch as st
import torch
import torch.nn.functional as F

from synapse.io import geotiff
from synapse.io.sentinel2 import select_bands, to_reflectance
from synapse.models import baseline, projector
from synapse.models.forward import S2Forward
from synapse.models.pro import SynapseProX5
from synapse.models.scan import backend
from synapse.pretrained import download, registry
from synapse.result import Result

SCALE = 5
SCL_INVALID = (0, 1, 3, 8, 9, 10)          # no data, saturated/defective, cloud shadow, cloud medium/high, cirrus
SUPPORT_T = (3.0, 10.0)                    # |P_N delta| / tau_b: below 3 HIGH, 3-10 MEDIUM, above LOW (heuristic)


def _validate_profile(profile):
    t = profile["transform"]
    if abs(t.b) > 1e-9 or abs(t.d) > 1e-9:
        raise ValueError("rotated / sheared geotransforms are not supported; warp to a north-up grid first")
    if abs(abs(t.a) - abs(t.e)) > 1e-6 * abs(t.a):
        raise ValueError(f"non-square pixels {t.a} x {t.e}")
    if abs(abs(t.a) - 10.0) > 0.5 and profile.get("crs") is not None and profile["crs"].is_projected:
        raise ValueError(f"expected the Sentinel-2 10 m grid, got {abs(t.a)} m pixels")


class Pro:
    def __init__(self, model, op, device, meta=None):
        self.model, self.op, self.device, self.meta = model.eval(), op, device, meta or {}

    @classmethod
    def from_pretrained(cls, name=registry.DEFAULT, weights=None, device=None):
        """weights: a local .safetensors path (skips download); otherwise the named manifest is fetched and
        its SHA-256 verified."""
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        meta = {}
        if weights is None:
            m = registry.manifest(name)
            weights = download.fetch(m["url"], m["sha256"], m["file"])
            meta = m
        sd = st.load_file(str(weights))
        op = S2Forward(sd.pop("operator.weight"), target_m=10.0 / SCALE)
        model = SynapseProX5()
        model.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")})
        return cls(model.to(device), op.to(device), device, meta)

    def save_pretrained(self, path):
        sd = {"model." + k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items()}
        sd["operator.weight"] = self.op.weight.cpu().contiguous()
        st.save_file(sd, str(path))
        return path

    @torch.no_grad()
    def super_resolve(self, src, band_names=None, tile=None, halo=16, offset=0.0, scl=None, nodata=None):
        """src: GeoTIFF path or (C, H, W) array; scl: optional Sentinel-2 scene classification (path or (H, W)).
        Returns a Result on the 2.0 m grid (origin preserved)."""
        profile = None
        if isinstance(src, (str, bytes)) or hasattr(src, "__fspath__"):
            arr, profile, names = geotiff.read(src)
            band_names = band_names or names
            _validate_profile(profile)
            nodata = profile.get("nodata") if nodata is None else nodata
        else:
            arr = np.asarray(src)
        arr = select_bands(arr, band_names)
        valid = np.ones(arr.shape[1:], bool)
        if nodata is not None:
            valid &= ~(arr == nodata).any(0)
        valid &= ~(arr == 0).all(0)
        if scl is not None:
            sc = geotiff.read(scl)[0][0] if isinstance(scl, (str, bytes)) or hasattr(scl, "__fspath__") else np.asarray(scl)
            if sc.shape != valid.shape:              # SCL is 20 m: nearest-replicate onto the 10 m grid
                f = valid.shape[0] // sc.shape[0]
                sc = np.repeat(np.repeat(sc, f, 0), f, 1)[:valid.shape[0], :valid.shape[1]]
            valid &= ~np.isin(sc, SCL_INVALID)
        y10 = torch.tensor(to_reflectance(arr, offset=offset), device=self.device)[None]
        tile = tile or (64 if backend(y10) == "fused" else 32)
        H, W = y10.shape[-2:]
        y4 = y10[:, :4]
        lam = baseline.select_lambda(self.op, y4)
        x = torch.zeros(1, 4, SCALE * H, SCALE * W, device=self.device)
        conf = torch.zeros_like(x)
        base = torch.zeros_like(x)
        wins = []
        for i in range(0, H, tile):
            for j in range(0, W, tile):
                i1, j1 = min(i + tile, H), min(j + tile, W)
                r0, c0, r1, c1 = max(i - halo, 0), max(j - halo, 0), min(i1 + halo, H), min(j1 + halo, W)
                xb = baseline.window(self.op, y4[..., r0:r1, c0:c1], lam)
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    o = self.model(y10[..., r0:r1, c0:c1])
                xw = xb + projector.apply(self.op, o["delta"].float())
                base[..., SCALE * i:SCALE * i1, SCALE * j:SCALE * j1] = xb[..., SCALE * (i - r0):SCALE * (i - r0) + SCALE * (i1 - i), SCALE * (j - c0):SCALE * (j - c0) + SCALE * (j1 - j)]
                cw = F.softplus(o["conf_logit"].float()) + 1e-4
                a, b = SCALE * (i - r0), SCALE * (j - c0)
                h, w = SCALE * (i1 - i), SCALE * (j1 - j)
                x[..., SCALE * i:SCALE * i1, SCALE * j:SCALE * j1] = xw[..., a:a + h, b:b + w]
                conf[..., SCALE * i:SCALE * i1, SCALE * j:SCALE * j1] = cw[..., a:a + h, b:b + w]
                wins.append((r0, c0, r1, c1))
        tau = baseline.tau_for(self.op)
        num = torch.zeros(4, device=self.device); cnt = 0
        for r0, c0, r1, c1 in wins:               # measured on the MOSAIC, so seams count
            xw = x[..., SCALE * r0:SCALE * r1, SCALE * c0:SCALE * c1]
            (oh, qh), (ow, qw) = self.op.block(xw.shape[-2]), self.op.block(xw.shape[-1])
            if qh <= 0 or qw <= 0:
                continue
            r = self.op(xw) - y4[..., r0 + oh:r0 + oh + qh, c0 + ow:c0 + ow + qw]
            num += r.pow(2).sum((0, 2, 3)); cnt += qh * qw
        rms = (num / max(cnt, 1)).sqrt() / tau
        gsd = abs(profile["transform"].a) / SCALE if profile else 10.0 / SCALE
        meta = {"scale": SCALE, "model": self.meta.get("name", "synapse-pro"), "scan_backend": backend(y10),
                "tile": tile, "halo": halo, "lambda": lam.tolist(), "bands": list(self.op.bands),
                "consistency_note": "RMS(A x - y)/tau under the nominal S2 forward model; not a registration check"}
        prior = (x - base)[0]
        s = (prior.abs() / tau.view(-1, 1, 1)).amax(0).cpu().numpy()
        support = np.where(s < SUPPORT_T[0], 2, np.where(s < SUPPORT_T[1], 1, 0)).astype(np.uint8)
        v2 = np.repeat(np.repeat(valid, SCALE, 0), SCALE, 1)
        support[~v2] = 0
        meta["support_classes"] = {"2": "HIGH: observation-determined", "1": "MEDIUM", "0": "LOW: prior-dominated or invalid input",
                                   "thresholds_tau": SUPPORT_T}
        meta["invalid_input_fraction"] = float(1 - valid.mean())
        return Result(image=x[0].cpu().numpy(), confidence=conf[0].cpu().numpy(),
                      consistency=dict(zip(self.op.bands, rms.tolist())), gsd=gsd, metadata=meta, profile=profile,
                      support=support, valid=v2, x_base=base[0].cpu().numpy(), prior=prior.cpu().numpy())
