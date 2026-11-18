# API reference

## Top level

::: synapse_sr.super_resolve

::: synapse_sr.super_resolve_folder

::: synapse_sr.load

::: synapse_sr.clear_cache

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
        - load

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
| `SYNAPSE_OFFLINE` / `HF_HUB_OFFLINE` | `1` never downloads; a missing checkpoint raises with the fix |
| `HF_ENDPOINT` | Hugging Face mirror for the weight download, e.g. `https://hf-mirror.com` |
| `SYNAPSE_SR_MEMORY` | working-memory budget for tiling in GB, on any device (default: free GPU memory, or half the available RAM up to 6 GB) |
| `SYNAPSE_SR_PREFLIGHT` | `0` skips the up-front check that the scene's output fits in memory |
| `SYNAPSE_SR_DISABLE_FUSED` | `1` ignores an installed `mamba-ssm` kernel |
| `SYNAPSE_SR_DISABLE_TRITON` | `1` skips the Triton scan kernel on CUDA |
| `SYNAPSE_SR_QUIET` | `1` silences progress bars and summaries everywhere |
