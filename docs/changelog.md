# Changelog

## 0.1.0

First public release.

- `Pro` model interface: `from_pretrained`, `super_resolve`, `save_pretrained`, `to`.
- One-call `synapse_sr.super_resolve`.
- Inputs: GeoTIFF, numpy, torch and xarray; DN or reflectance; automatic band mapping and offset handling.
- Preprocessing: NoData and SCL masking, grid validation, tiling with halo.
- Outputs: reflectance, error scale, support classes, validity, consistency, observed / inferred decomposition,
  `rgb()`, `ndvi()`, `to_xarray()`, GeoTIFF export.
- `fetch_sentinel2` STAC helper (`synapse-sr[stac]`).
- `synapse-sr` command line with `--env` diagnostics.
- Fused CUDA selective scan when available, verified PyTorch fallback otherwise.

## Acknowledgements

- The state-space backbone is adapted from [ESAOpenSR / SEN2SR](https://github.com/ESAOpenSR/sen2sr)
  (CC0-1.0). Training initialised it from the public SEN2SR weights. The synapse-sr checkpoint is trained
  separately, and SEN2SR is not required at run time.
- The optional fused kernel comes from [mamba-ssm](https://github.com/state-spaces/mamba) (Apache-2.0).
- Sentinel-2 data: Copernicus programme, European Space Agency.
