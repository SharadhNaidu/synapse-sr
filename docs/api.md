# API reference

## Top level

::: synapse_sr.super_resolve

::: synapse_sr.Pro
    options:
      members:
        - from_pretrained
        - super_resolve
        - save_pretrained
        - to

::: synapse_sr.Result
    options:
      members:
        - rgb
        - ndvi
        - to_xarray
        - save

::: synapse_sr.fetch_sentinel2

## Lower level

These are stable, but most users do not need them.

::: synapse_sr.io.sentinel2.select_bands

::: synapse_sr.io.sentinel2.to_reflectance

::: synapse_sr.models.scan.backend

## Environment variables

| Variable | Effect |
|---|---|
| `SYNAPSE_CACHE` | weight cache directory (default `~/.cache/synapse`) |
| `SYNAPSE_SR_DISABLE_FUSED` | `1` forces the PyTorch selective scan even when `mamba-ssm` is installed |
