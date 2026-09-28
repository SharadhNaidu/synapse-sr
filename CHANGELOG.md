# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.4.1] - 2026-09-28

### Changed

- **Benchmark: SYNAPSE is now best on five of the seven official opensr-test columns** (means over NAIP, SPOT,
  Spain-urban, Spain-crops, VENuS; 178 scenes). Pro: improvement 0.199, spectral error 0.223, reflectance error 0.0011.
  Flash: detail correlation 0.300, RMSE 0.0234.
- Flash restores each 10 m pixel's measured mean reflectance with a smooth bicubic correction after the physics
  projection (`super_resolve(..., restore_mean=True)`, its new default). The correction is counted in `x_base`, so
  `prior` is still exactly the network's contribution.
- Pro fits the measurement tightly by default (`discrepancy=0.5`) for the most recovered detail. `discrepancy=4,
  restore_mean=False` reproduces 0.4.0 for either model.
- Both checkpoints recalibrated for the new defaults: Pro 82 / 91 / 96 % and Flash 82 / 91 / 95 % coverage at
  80 / 90 / 95 % on held-out references.

### Fixed

- Intel Macs: numpy is pinned below 2 there, because the last PyTorch built for Intel macOS (2.2.2) needs numpy 1.
  Continuous integration now runs the published-weights suite on an Intel Mac runner.

## [0.4.0] - 2026-09-28

### Added

- **SYNAPSE Flash v1 released and now the default model.** It runs a 1.28 km scene in about 1 s on a laptop CPU and a
  10 km x 10 km scene in about 14 s. It matches Pro on the official opensr-test benchmark (knowledge distillation from Pro;
  trained on real Sentinel-2 / NAIP pairs, a streamed NAIP corpus and ISRO Cartosat-derived pairs). It ships with
  calibrated uncertainty (82 / 91 / 95 % coverage at 80 / 90 / 95 %). Pro stays one argument away: `model="pro"`.
- `synapse_sr.super_resolve_folder(src_dir, out_dir)` and `synapse-sr in_dir/ out_dir/`: batch a folder, skip cloud
  sidecars and existing outputs, and keep going past a bad file.
- `synapse-sr --fetch LAT,LON --dates START:END out.tif`: download and super-resolve in one command.
- `Result.save(..., cog=True)` / `--cog`: Cloud-Optimised GeoTIFF with overviews.
- `Result.quicklook("preview.png")` / `--preview`: side-by-side 10 m / 2 m / support preview.
- `super_resolve(..., tta=True)`: test-time augmentation over 8 flips / rotations (small accuracy gain).
- `super_resolve(..., discrepancy=)`: how tightly the physics baseline fits the measurement.
- Documentation: cheat sheet, benchmark table in the README, `llms.txt` / `llms-full.txt` for AI assistants,
  `AGENTS.md`, `CITATION.cff`; every documentation example is executed in CI-like runs.
- `tools/stress.py`: 27-case stress suite with the published weights.

### Changed

- **Physics baseline:** one regularisation weight for all bands, fitted to 4x the L2A sensor noise (it absorbs
  forward-model error). This strongly reduces hallucinated detail and spectral-angle error on the official benchmark
  (VENuS hallucination 0.351 -> 0.198). The consistency readout is correspondingly a few noise units.
- **Memory-aware tiling:** tile and batch sizes follow free GPU memory (or a RAM budget on CPU). Pro is about 2.5x faster.
- Uncertainty calibration refitted for the new physics defaults (Pro 82 / 91 / 96 %).

### Fixed

- Some Earth Search Sentinel-2 items flag the baseline-04.00 offset as not applied when it already was. Such scenes
  were read about 0.1 reflectance too dark. The offset is now decided from the pixel values (with a warning),
  in `fetch_sentinel2` and for tagged GeoTIFFs; an explicit `offset=` is always respected.
- NaN / inf inputs are masked instead of zeroed. Outputs are clamped to [0, 1.5] reflectance, and clamped pixels
  get LOW support. Consistency is NaN when no pixel is valid. `halo >= tile` raises. Saving array input to `.tif`
  raises instead of silently writing `.npz`. `torch.cuda.mem_get_info` works on torch < 2.1.
- Flash: the confidence head no longer steers the network body.

## [0.3.0] - 2026-09-24

### Added

- **SYNAPSE Flash** (`synapse_sr.Flash`, `--model flash`): a re-parameterised SPAN-style CNN on the same
  observation-consistent pipeline, for CPUs, laptops, integrated graphics, Apple silicon and ARM. Weights are not
  released yet; `Pro.from_pretrained(weights=...)` recognises Flash checkpoints automatically.
- **Triton selective-scan kernel**: Pro runs its Mamba layers at GPU speed on Colab, Kaggle and any CUDA machine
  with Triton, with no `mamba-ssm` build. It is self-tested against the exact scan once per process.
- **Live feedback** (Rich): progress bar with stages and tile counts, download bar, one-line result summary;
  automatic in terminals and notebooks, silent when piped (`progress=`, `SYNAPSE_SR_QUIET=1`).
- `Result.summary()`, `Result.show()` (matplotlib panels: image, x_base, prior, support, uncertainty, NDVI) and a
  concise `repr`.
- CLI: `--models`, `--json`, `--batch`, `--quiet`; `--env` shows CPU threads, MPS, and the scan backend Pro will use.
- Quick-start notebook for Colab and Kaggle; new guides: *Colab and Kaggle*, *Choosing a model and a device*.
- Tests for every fast path against its exact reference (operator, scan at extreme decay rates, preconditioned
  solvers, Triton, Flash re-parameterisation and checkpoint dispatch).

### Changed

- **Pro is 3-4x faster with unchanged outputs.**
  - The Sentinel-2 forward operator is evaluated as one strided convolution on the 2 m grid. This is
    algebraically identical to the 0.5 m fine-grid form and about 100x faster.
  - The physics solves (lambda calibration, Tikhonov baseline, null-space projection) use conjugate gradients
    preconditioned with the operator's closed-form periodic inverse. The answers are the same and fewer iterations
    are needed.
  - The PyTorch scan is rewritten as elementwise operations plus a cumulative sum in float32: no chunk matrices, and
    memory bounded at any batch size.
  - Equal-size tiles and calibration windows are batched.
  - End-to-end check against 0.2.0 on a real scene: x_base within 3e-5, the image within bf16 rounding (0.002).
- The CLI prints a summary table; JSON output is now behind `--json`.
- `rich` is a dependency.

### Fixed

- `fetch_sentinel2` applied the processing-baseline 04.00 offset to Earth Search scenes that already had it removed
  (`earthsearch:boa_offset_applied`). Scenes came out 0.1 reflectance too dark and dark surfaces went negative.
  Re-download scenes fetched with 0.2.0.
- Normalised-difference indices are NaN where both bands are essentially dark and are clipped to [-1, 1]; EVI is
  clipped to [-1, 1]; negative L2A reflectance counts as zero. `change()` excludes pixels whose index is undefined.
- Windows consoles without UTF-8 get ASCII symbols instead of garbled characters.

### Corrected claim

- The effective-resolution statement made in 0.2.0 is withdrawn. No effective-resolution figure is claimed.

## [0.2.0] - 2026-09-23

### Added

- Application layer: `Result.indices()` (NDVI, SAVI, EVI, GNDVI, NDWI; with 20 m context NDRE, NDBI, NBR, MNDWI),
  `Result.band()`, 20 m context bands on the output grid (`context=True`, labelled not super-resolved),
  `synapse_sr.change()` (support-aware change detection) and `synapse_sr.boundaries()` (field, water, urban).
- Calibrated uncertainty: `Result.uncertainty()` and `Result.interval(level)` from a checkpoint-shipped error model
  with split-conformal coverage (Pro v1: 82 / 91 / 96 % at 80 / 90 / 95 %).
- Local checkpoints matching a registered model (by SHA-256) receive that model's calibration.
- Documentation: Applications page; calibrated-uncertainty guide.

### Changed

- **Default model is now SYNAPSE Pro v2** (`pro-v2`, 5000 training steps): best field / urban / water edge F1 on the development benchmark, calibrated uncertainty shipped.
  Known limitation: low-contrast wide strips (about 24 m) can be split by a false gap. `pro-v1` remains available.
- The CLI `--model` default follows the registry default.
- The 20 m context stem is zero-initialised without a scalar gate; Pro v1 checkpoints load unchanged (gate folded).
- The frequency mixer pads in float32.
- `scipy` is now a dependency.

## [0.1.0] - 2026-09-23

### Added

- `Pro` model interface: `from_pretrained`, `super_resolve`, `save_pretrained`, `to`.
- `synapse_sr.super_resolve` one-call function with a cached default model.
- Inputs: GeoTIFF paths, numpy arrays, torch tensors and xarray DataArrays; DN or reflectance; automatic band
  mapping (10-, 12- and 13-band layouts or band names) and `BOA_ADD_OFFSET` handling.
- Preprocessing: NoData, NaN and Sentinel-2 SCL masking (automatic `<input>_scl.tif` pick-up), grid validation,
  seamless tiling with context halo for any scene size.
- `Result` with reflectance, error scale, support classes, validity mask, round-trip consistency, observed /
  inferred decomposition, `rgb()`, `ndvi()`, `to_xarray()` and GeoTIFF / npz export.
- `fetch_sentinel2` STAC helper (`synapse-sr[stac]`).
- `synapse-sr` command line and `python -m synapse_sr`, with `--env` diagnostics and `--version`.
- Fused `mamba-ssm` selective scan when available, with a verified PyTorch fallback.
- SYNAPSE Pro v1 research-preview checkpoint on Hugging Face.
