"""High-level interface: load a SYNAPSE Pro checkpoint and super-resolve Sentinel-2 scenes."""

import contextlib
import os
import time
import warnings
from typing import Callable, Optional, Sequence, Union

import numpy as np
import safetensors.torch as st
import torch
import torch.nn.functional as F

from synapse_sr import ui
from synapse_sr.data import offset_present
from synapse_sr.io import geotiff
from synapse_sr.io.sentinel2 import select_bands, to_reflectance
from synapse_sr.models import baseline, projector
from synapse_sr.models.fastphys import FastPhysics
from synapse_sr.models.flash import SynapseFlashX5
from synapse_sr.models.forward import S2Forward
from synapse_sr.models.pro import SynapseProX5, fold_v1_gate
from synapse_sr.models.scan import backend
from synapse_sr.pretrained import download, registry
from synapse_sr.result import Result

SCALE = 5
SCL_INVALID = (0, 1, 3, 8, 9, 10)          # no data, saturated/defective, cloud shadow, cloud medium/high, cirrus
DISCREPANCY = 4.0                           # tau units; Flash default, see super_resolve(discrepancy=)
CLAMP = (0.0, 1.5)                          # physical reflectance range of the output
SUPPORT_T = (3.0, 10.0)                    # |prior| / tau_b: below 3 HIGH, 3-10 MEDIUM, above LOW (heuristic)

PathLike = Union[str, "os.PathLike[str]"]


def _is_path(x):
    return isinstance(x, (str, bytes)) or hasattr(x, "__fspath__")


def _as_array(src, band_names):
    """numpy / torch / xarray -> (C, H, W) numpy array and band names."""
    if hasattr(src, "dims") and hasattr(src, "values"):            # xarray.DataArray (e.g. from cubo)
        da = src
        extra = [d for d in da.dims if d not in ("band", "y", "x")]
        for d in extra:
            if da.sizes[d] != 1:
                raise ValueError(f"select one {d!r} first (e.g. da.isel({d}=0)); got {da.sizes[d]}")
            da = da.isel({d: 0})
        da = da.transpose("band", "y", "x")
        if band_names is None and "band" in da.coords:
            band_names = [str(b).upper() for b in da.coords["band"].values]
        return np.asarray(da.values), band_names
    if hasattr(src, "detach") and hasattr(src, "cpu"):              # torch.Tensor
        src = src.detach().cpu().numpy()
    arr = np.asarray(src)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"expected a (C, H, W) array, got shape {arr.shape}")
    return arr, band_names


def _validate_profile(profile):
    t = profile["transform"]
    if abs(t.b) > 1e-9 or abs(t.d) > 1e-9:
        raise ValueError("rotated or sheared geotransforms are not supported; warp to a north-up grid first")
    if abs(abs(t.a) - abs(t.e)) > 1e-6 * abs(t.a):
        raise ValueError(f"non-square pixels {t.a} x {t.e}")
    crs = profile.get("crs")
    if crs is not None and crs.is_projected and abs(abs(t.a) - 10.0) > 0.5:
        raise ValueError(f"expected the Sentinel-2 10 m grid, got {abs(t.a)} m pixels")


def _amp_dtype(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


class Pro:
    """SYNAPSE Pro super-resolution model (Sentinel-2 10 m -> 2.0 m RGBN).

    The output is ``x_hat = x_base + P_N(delta)``: a deterministic, observation-consistent baseline plus a
    learned correction restricted to the null space of the Sentinel-2 forward operator, so the network can add
    structure the sensor could not observe but cannot change what it did observe.

    Use :meth:`from_pretrained` to construct.
    """

    scale = SCALE
    DEFAULTS = {"discrepancy": 0.5, "restore_mean": False}   # maximum detail: fit the measurement tightly

    def __init__(self, model: SynapseProX5, op: S2Forward, device: Union[str, torch.device] = "cpu",
                 meta: Optional[dict] = None):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.op = op.to(self.device)
        self.meta = meta or {}
        self._fast = None

    def __repr__(self):
        return (f"{type(self).__name__}(name={self.meta.get('name', 'local')!r}, device={str(self.device)!r}, "
                f"backend={self._backend(torch.zeros(1, device=self.device))!r})")

    def _backend(self, y):
        return backend(y)

    def _predict(self, yb, amp, tta):
        """Network output; with tta, the mean over the 8 flips / 90-degree rotations of the input, each mapped back.
        Every transform keeps the 10 m lattice, so the averaged detail is still a valid input to P_N."""
        def run(y):
            if amp is not None:
                with torch.autocast("cuda", dtype=amp):
                    return self.model(y)
            return self.model(y)
        if not tta:
            return run(yb)
        acc = None
        for k in range(4):
            for flip in (False, True):
                y = torch.rot90(yb, k, (-2, -1))
                y = y.flip(-1) if flip else y
                o = run(y)
                outs = {}
                for key, v in o.items():
                    v = v.float()
                    v = v.flip(-1) if flip else v
                    outs[key] = torch.rot90(v, -k, (-2, -1))
                acc = outs if acc is None else {key: acc[key] + outs[key] for key in acc}
        return {key: v / 8 for key, v in acc.items()}

    BYTES_PER_PX = {"pytorch": 3.0e5, "triton": 1.2e5, "fused": 1.2e5, "cnn": 2.0e4}   # measured peak per window px

    def _plan(self, kind, H, W, halo, tile, batch):
        """Largest tile, then largest batch, that fit the memory budget: fewer, larger windows waste less of
        the context halo (tile 32 + halo 16 computes 4x the output area; tile 96 1.8x)."""
        if self.device.type == "cuda":
            idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
            free, _ = torch.cuda.mem_get_info(idx)       # torch < 2.1 needs an index, not a bare 'cuda'
            budget = 0.6 * free
        else:
            budget = 6e9
        per = self.BYTES_PER_PX.get(kind, 3.0e5)
        if tile is None:
            tile = 32
            for t in (256, 192, 128, 96, 64, 48, 32):
                if (t + 2 * halo) ** 2 * per <= budget:
                    tile = t
                    break
        tile = max(8, min(tile, max(H, W)))
        if batch is None:
            fit = int(budget // ((min(tile, H) + 2 * halo) * (min(tile, W) + 2 * halo) * per))
            batch = max(1, min(8, fit)) if self.device.type == "cuda" else 1
        return tile, batch

    @classmethod
    def from_pretrained(cls, name: str = registry.DEFAULT, weights: Optional[PathLike] = None,
                        device: Optional[Union[str, torch.device]] = None) -> "Pro":
        """Load a SYNAPSE Pro checkpoint.

        Parameters
        ----------
        name:
            Registered model name (see ``synapse_sr/pretrained/*.json``). Its checkpoint is downloaded once
            into the cache (``$SYNAPSE_CACHE`` or ``~/.cache/synapse``) and its SHA-256 verified on every load.
        weights:
            Path to a local ``.safetensors`` checkpoint. Skips the registry and the network entirely
            (air-gapped use).
        device:
            ``"cuda"``, ``"cpu"``, ``"cuda:1"`` ... Defaults to CUDA when available.
        """
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        meta = {}
        if weights is not None:
            known = registry.match(download.sha256(weights))
            meta = dict(known, weights=str(weights)) if known else {"name": "local", "weights": str(weights)}
        else:
            m = registry.manifest(name)
            weights = download.fetch(m["url"], m["sha256"], m["file"])
            meta = dict(m)
        sd = st.load_file(str(weights))
        if "operator.weight" not in sd:
            raise ValueError(f"{weights} is not a SYNAPSE checkpoint (no operator.weight)")
        op = S2Forward(sd.pop("operator.weight"), target_m=10.0 / SCALE)
        net = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        if "conv_1.sk.weight" in net:
            klass, model = Flash, SynapseFlashX5.from_state_dict(net)
            model.load_state_dict(net)
            model.reparameterise()
        else:
            klass, model = Pro, SynapseProX5()
            model.load_state_dict(fold_v1_gate(net))
        if cls is not Pro and klass is not cls:
            raise ValueError(f"{weights} is a {klass.__name__} checkpoint; load it with {klass.__name__}.from_pretrained")
        return klass(model, op, device, meta)

    def save_pretrained(self, path: PathLike) -> PathLike:
        """Write model and operator to a single ``.safetensors`` file loadable with ``weights=``."""
        sd = {"model." + k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items()}
        sd["operator.weight"] = self.op.weight.detach().cpu().contiguous()
        st.save_file(sd, str(path))
        return path

    def to(self, device: Union[str, torch.device]) -> "Pro":
        """Move the model and operator to another device, in place."""
        self.device = torch.device(device)
        self.model.to(self.device)
        self.op.to(self.device)
        self._fast = None
        return self

    def _physics(self):
        """Periodic closed-form physics, used only to precondition the exact solves (same answers, fewer steps)."""
        if self._fast is None:
            self._fast = FastPhysics(self.op)
        return self._fast

    @torch.no_grad()
    def super_resolve(self, src: Union[PathLike, np.ndarray], band_names: Optional[Sequence[str]] = None,
                      scl: Optional[Union[PathLike, np.ndarray]] = "auto", nodata: Optional[float] = None,
                      offset: Optional[float] = None, tile: Optional[int] = None, halo: Optional[int] = None,
                      context: bool = True, batch: Optional[int] = None,
                      progress: Union[bool, str, Callable[[int, int], None]] = "auto",
                      discrepancy: Optional[float] = None, tta: bool = False,
                      restore_mean: Optional[bool] = None) -> Result:
        """Super-resolve one Sentinel-2 scene.

        Parameters
        ----------
        src:
            GeoTIFF path, a ``(C, H, W)`` numpy array or torch tensor, or an ``xarray.DataArray`` with a ``band``
            coordinate (for example a single time step of a ``cubo`` cube), in L2A DN or reflectance. Accepted band layouts: the 10 model
            bands (B04 B03 B02 B08 B05 B06 B07 B8A B11 B12), the 12-band L2A order, the 13-band L1C order, or any
            stack whose ``band_names`` (or GeoTIFF band descriptions) include the 10 model bands.
        band_names:
            Names of the channels of ``src``; overrides GeoTIFF descriptions.
        scl:
            Sentinel-2 scene classification (path or ``(h, w)`` array, 10 m or 20 m); classes 0, 1, 3, 8, 9 and 10
            are masked. ``"auto"`` (default) uses ``<input>_scl.tif`` next to a GeoTIFF input when it exists;
            ``None`` disables masking.
        nodata:
            Input NoData value; defaults to the GeoTIFF's. All-zero pixels are always masked.
        offset:
            Added to DN before dividing by 10000 (``-1000`` for L2A processing baseline 04.00 and later).
            ``None`` (default) reads the GeoTIFF ``BOA_ADD_OFFSET`` tag, else 0. Ignored for reflectance input.
        discrepancy:
            How closely the physics baseline fits the measurement, in units of the Sentinel-2 L2A noise level.
            Pro defaults to 0.5 (tight fit, the most recovered detail); Flash to 4 (a looser fit that absorbs
            forward-model error, with fewer spurious edges). ``4`` on Pro reproduces 0.4.0.
        tile, halo:
            Source-pixel tile size and context halo. By default the largest tile (and batch) that fits the free GPU
            memory, or a 6 GB RAM budget on CPU, is chosen: larger tiles waste less computation on the halo.
        batch:
            Tiles processed together (default: as many as fit the free GPU memory, up to 8; 1 on CPU).
        progress:
            ``"auto"`` (default) shows a live progress bar in terminals and notebooks and stays silent when output
            is piped or logged; ``True`` / ``False`` force it on / off; a callable ``f(done, total)`` receives the
            tile count instead. ``SYNAPSE_SR_QUIET=1`` silences everything.
        context:
            Carry the six native 20 m bands (B05 B06 B07 B8A B11 B12) onto the output grid by pixel replication,
            so red-edge and SWIR indices (NDRE, NDBI, NBR, MNDWI) are available. They are NOT super-resolved.
        restore_mean:
            Make every 10 m pixel's mean reflectance equal the measurement with a smooth (bicubic) correction that
            adds no fine structure; it is counted in ``x_base``, not ``prior``. On by default for Flash (lower spectral
            and reflectance error, higher detail correlation); off by default for Pro, whose tight fit already
            matches the measurement.

        Returns
        -------
        Result
            2.0 m RGBN image with error-scale map, support classes, validity mask, consistency and the
            observed / inferred decomposition.
        """
        discrepancy = self.DEFAULTS["discrepancy"] if discrepancy is None else discrepancy
        restore_mean = self.DEFAULTS["restore_mean"] if restore_mean is None else restore_mean
        show = progress is True or (progress == "auto" and ui.interactive())
        callback = progress if callable(progress) else None
        t0 = time.time()
        with (ui.RunProgress(type(self).__name__) if show else contextlib.nullcontext()) as bar:
            r = self._run(src, band_names, scl, nodata, offset, tile, halo, context, batch, bar, callback, discrepancy, tta,
                          restore_mean)
        r.metadata["seconds"] = round(time.time() - t0, 3)
        if show:
            bar.done(r, r.metadata["seconds"])
        return r

    def _run(self, src, band_names, scl, nodata, offset, tile, halo, context, batch, bar, callback, discrepancy, tta, restore_mean):
        if bar:
            bar.stage("reading input")
        profile, tags = None, {}
        if _is_path(src):
            arr, profile, names, tags = geotiff.read(src)
            band_names = band_names or names
            _validate_profile(profile)
            nodata = profile.get("nodata") if nodata is None else nodata
            if isinstance(scl, str) and scl == "auto":
                side = os.path.splitext(os.fspath(src))[0] + "_scl.tif"
                scl = side if os.path.exists(side) else None
        else:
            arr, band_names = _as_array(src, band_names)
        if isinstance(scl, str) and scl == "auto":
            scl = None
        tagged = offset is None
        if offset is None:
            offset = float(tags.get("BOA_ADD_OFFSET", 0.0))
        arr = select_bands(arr, list(band_names) if band_names else None)
        if tagged and offset < 0 and not np.issubdtype(arr.dtype, np.floating) and not offset_present(arr):
            warnings.warn(f"the input carries BOA_ADD_OFFSET={offset:g}, but its values show the offset was already "
                          f"removed (many pixels below DN 1000); using offset 0. Pass offset= explicitly to override.",
                          RuntimeWarning, stacklevel=3)
            offset = 0.0
        valid = np.ones(arr.shape[1:], bool)
        if np.issubdtype(arr.dtype, np.floating):
            valid &= np.isfinite(arr).all(0)                  # NaN / inf inputs are invalid pixels, not zeros
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if nodata is not None:
            valid &= ~(arr == nodata).any(0)
        valid &= ~(arr == 0).all(0)
        scl_used = None
        if scl is not None:
            sc = geotiff.read(scl)[0][0] if _is_path(scl) else np.asarray(scl)
            scl_used = os.fspath(scl) if _is_path(scl) else "array"
            if sc.shape != valid.shape:
                f = int(round(valid.shape[0] / sc.shape[0]))
                sc = np.repeat(np.repeat(sc, f, 0), f, 1)[:valid.shape[0], :valid.shape[1]]
                if sc.shape != valid.shape:
                    raise ValueError(f"SCL shape {sc.shape} does not tile the {valid.shape} scene")
            valid &= ~np.isin(sc, SCL_INVALID)

        y10 = torch.tensor(to_reflectance(arr, offset=offset), dtype=torch.float32, device=self.device)[None]
        kind = self._backend(y10)
        if halo is not None and tile is not None and halo >= tile:
            raise ValueError(f"halo ({halo}) must be smaller than tile ({tile}); a halo only adds context around a tile")
        if halo is None:
            halo = 16 if tile is None else min(16, tile - 1)
        tile, batch = self._plan(kind, *y10.shape[-2:], halo, tile, batch)
        amp = _amp_dtype(self.device)
        H, W = y10.shape[-2:]
        y4 = y10[:, :4]
        if bar:
            bar.stage("calibrating the physics baseline")
        fast = self._physics()
        lam = baseline.select_lambda(self.op, y4, fast=fast, discrepancy=discrepancy, common=True)
        x = torch.zeros(1, 4, SCALE * H, SCALE * W, device=self.device)
        conf = torch.zeros_like(x)
        base = torch.zeros_like(x)
        jobs = []
        for i in range(0, H, tile):
            for j in range(0, W, tile):
                i1, j1 = min(i + tile, H), min(j + tile, W)
                r0, c0, r1, c1 = max(i - halo, 0), max(j - halo, 0), min(i1 + halo, H), min(j1 + halo, W)
                jobs.append((i, j, i1, j1, r0, c0, r1, c1))
        groups = {}
        for jb in jobs:                                   # equal-shape windows run as one batch (all solves are per sample)
            groups.setdefault((jb[6] - jb[4], jb[7] - jb[5]), []).append(jb)
        bs = batch
        wins = []
        for shape, members in groups.items():
            for k in range(0, len(members), bs):
                part = members[k:k + bs]
                yb = torch.cat([y10[..., r0:r1, c0:c1] for _, _, _, _, r0, c0, r1, c1 in part])
                xb = baseline.window(self.op, yb[:, :4], lam, fast=fast)
                o = self._predict(yb, amp, tta)
                xw = xb + projector.apply(self.op, o["delta"].float(), fast=fast)
                cw = F.softplus(o["conf_logit"].float()) + 1e-4
                for n, (i, j, i1, j1, r0, c0, r1, c1) in enumerate(part):
                    a, b = SCALE * (i - r0), SCALE * (j - c0)
                    h, w = SCALE * (i1 - i), SCALE * (j1 - j)
                    dst = (Ellipsis, slice(SCALE * i, SCALE * i1), slice(SCALE * j, SCALE * j1))
                    x[dst] = xw[n, :, a:a + h, b:b + w]
                    conf[dst] = cw[n, :, a:a + h, b:b + w]
                    base[dst] = xb[n, :, a:a + h, b:b + w]
                    wins.append((r0, c0, r1, c1))
                if bar:
                    bar.tiles(len(wins), len(jobs))
                if callback:
                    callback(len(wins), len(jobs))

        if restore_mean:                                  # smooth, measurement-derived: part of the baseline, not the prior
            vm = torch.tensor(valid, device=self.device)[None, None]
            xr = baseline.restore_mean(x, y4, vm)
            base = base + (xr - x)
            x = xr
        if bar:
            bar.stage("checking observation consistency")
        tau = baseline.tau_for(self.op)
        clamped = (x < CLAMP[0]) | (x > CLAMP[1])                 # unphysical values only arise from bad inputs
        x = x.clamp(*CLAMP)
        vt = torch.tensor(valid, device=self.device, dtype=torch.float32)[None, None]
        num = torch.zeros(4, device=self.device)
        cnt = 0.0
        for r0, c0, r1, c1 in wins:                       # measured on the mosaic over valid pixels, so seams count
            xw = x[..., SCALE * r0:SCALE * r1, SCALE * c0:SCALE * c1]
            (oh, qh), (ow, qw) = self.op.block(xw.shape[-2]), self.op.block(xw.shape[-1])
            if qh <= 0 or qw <= 0:
                continue
            r = self.op(xw) - y4[..., r0 + oh:r0 + oh + qh, c0 + ow:c0 + ow + qw]
            m = vt[..., r0 + oh:r0 + oh + qh, c0 + ow:c0 + ow + qw]
            num += (r.pow(2) * m).sum((0, 2, 3))
            cnt += float(m.sum())
        rms = (num / cnt).sqrt() / tau if cnt > 0 else torch.full((4,), float("nan"), device=self.device)
        bad = [b for b, v in zip(self.op.bands, rms.tolist()) if v > 5.0 * max(discrepancy, DISCREPANCY)]
        if bad:
            warnings.warn(f"the output disagrees with the input beyond sensor noise in {bad} (RMS well above the {discrepancy:g} tau target): check the "
                          f"band order, the radiometric offset and that the input is Sentinel-2 L2A on the 10 m grid",
                          RuntimeWarning, stacklevel=3)

        prior = (x - base)[0]
        s = (prior.abs() / tau.view(-1, 1, 1)).amax(0).cpu().numpy()
        support = np.where(s < SUPPORT_T[0], 2, np.where(s < SUPPORT_T[1], 1, 0)).astype(np.uint8)
        v2 = np.repeat(np.repeat(valid, SCALE, 0), SCALE, 1)
        support[~v2] = 0
        support[clamped[0].any(0).cpu().numpy()] = 0
        gsd = abs(profile["transform"].a) / SCALE if profile else 10.0 / SCALE
        meta = {"scale": SCALE, "model": self.meta.get("name", "local"), "scan_backend": kind,
                "precision": "bfloat16" if amp is not None else "float32", "tile": tile, "halo": halo,
                "lambda": lam.tolist(), "discrepancy_tau": discrepancy, "tta": bool(tta), "restore_mean": bool(restore_mean), "bands": list(self.op.bands),
                "support_classes": {"2": "HIGH: observation-determined", "1": "MEDIUM",
                                    "0": "LOW: prior-dominated or invalid input", "thresholds_tau": SUPPORT_T},
                "invalid_input_fraction": float(1 - valid.mean()),
                "clamped_fraction": float(clamped.float().mean()), "clamp_range": CLAMP, "scl": scl_used, "offset": offset,
                "consistency_note": "RMS(A x - y) / tau under the nominal Sentinel-2 forward model"}
        ctx = None
        if context:
            ctx = y10[0, 4:].repeat_interleave(SCALE, -2).repeat_interleave(SCALE, -1).cpu().numpy().astype(np.float16)
            meta["context_note"] = "B05 B06 B07 B8A B11 B12 replicated from their native 20 m grid; not super-resolved"
        return Result(image=x[0].cpu().numpy(), confidence=conf[0].cpu().numpy(),
                      consistency=dict(zip(self.op.bands, rms.tolist())), gsd=gsd, metadata=meta, profile=profile,
                      support=support, valid=v2, x_base=base[0].cpu().numpy(), prior=prior.cpu().numpy(),
                      context=ctx, calibration=self.meta.get("calibration"))


class Flash(Pro):
    """SYNAPSE Flash: the same observation-consistent pipeline as :class:`Pro` (``x_base + P_N(delta)``) with a
    0.6 M-parameter convolutional detail network instead of the Mamba network. No sequence scan, so it is fast on
    CPUs, laptops, integrated graphics and Apple silicon. Its body was initialised from SEN2SR-Lite (ESAOpenSR,
    CC0-1.0) and fine-tuned for the x5 null-space task.
    """

    @classmethod
    def from_pretrained(cls, name: str = registry.DEFAULT_FLASH, weights: Optional[PathLike] = None,
                        device: Optional[Union[str, torch.device]] = None) -> "Flash":
        """Load a SYNAPSE Flash checkpoint; arguments as :meth:`Pro.from_pretrained`."""
        return super().from_pretrained(name, weights, device)

    DEFAULTS = {"discrepancy": DISCREPANCY, "restore_mean": True}

    def _backend(self, y):
        return "cnn"

