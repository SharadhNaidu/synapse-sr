# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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
