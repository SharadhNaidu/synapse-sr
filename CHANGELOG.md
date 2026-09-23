# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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
