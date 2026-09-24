# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-09-24

### Added

- **SYNAPSE Flash** (`synapse_sr.Flash`, `--model flash`): a re-parameterised SPAN-style CNN on the same
  observation-consistent pipeline, for CPUs, laptops, integrated graphics, Apple silicon and ARM. Weights are in
  training; `Pro.from_pretrained(weights=...)` recognises Flash checkpoints automatically.
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

- 0.2.0 stated that Pro v2 certifies 6 m in the two-bar resolution test. That test lacked matched equal-flux
  negatives. Under the bias-controlled version, no released checkpoint certifies finer than 8 m. No
  effective-resolution figure is claimed for this release.

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

- **Default model is now SYNAPSE Pro v2** (`pro-v2`, 5000 training steps): 6 m certified in the two-bar resolution
  test (v1: none), best field / urban / water edge F1 on the development benchmark, calibrated uncertainty shipped.
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
