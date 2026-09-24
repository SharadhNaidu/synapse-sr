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

::: synapse_sr.Flash
    options:
      members:
        - from_pretrained

::: synapse_sr.Result
    options:
      members:
        - summary
        - show
        - indices
        - ndvi
        - band
        - uncertainty
        - interval
        - rgb
        - to_xarray
        - save

## Applications

::: synapse_sr.change

::: synapse_sr.Change

::: synapse_sr.boundaries

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
| `SYNAPSE_SR_DISABLE_FUSED` | `1` ignores an installed `mamba-ssm` kernel |
| `SYNAPSE_SR_DISABLE_TRITON` | `1` skips the Triton scan kernel on CUDA |
| `SYNAPSE_SR_QUIET` | `1` silences progress bars and summaries everywhere |
