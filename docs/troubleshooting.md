# Troubleshooting

Start with the environment report:

```bash
synapse-sr --env
```

| Message | Cause | Fix |
|---|---|---|
| `pretrained weights for 'pro-v2' are not published yet` | registry entry without a download URL | pass `weights="path/to/model.safetensors"` |
| `checksum mismatch for ...` | incomplete or modified download | delete the file named in the message and retry |
| `cannot map N channels to Sentinel-2 bands` | unknown band layout | pass `band_names=[...]` or write band descriptions into the GeoTIFF |
| `expected the Sentinel-2 10 m grid, got 20.0 m pixels` | input not on the 10 m grid | resample to 10 m first (`gdalwarp -tr 10 10 -r near`) |
| `rotated or sheared geotransforms are not supported` | non north-up raster | `gdalwarp` to a north-up grid |
| `select one 'time' first` | multi-date xarray cube | pass `cube.isel(time=k)` |
| `mamba-ssm selective_scan_cuda is installed but unusable` (warning) | kernel built for another PyTorch or CUDA | reinstall `mamba-ssm` with `--no-build-isolation`; results are unaffected, only slower |
| `CUDA out of memory` | tile too large for the GPU | lower `tile` (for example `tile=32`) |
| `GeoTIFF IO needs rasterio` | rasterio missing | `pip install rasterio` (conda-forge on Windows if pip fails) |
| output much darker or brighter than expected | offset mismatch | check the product's processing baseline; pass `offset=-1000` or `offset=0` |
| scenes from `fetch_sentinel2` in v0.2.0 look too dark, NDVI near 1 everywhere | v0.2.0 applied the baseline 04.00 offset to Earth Search data that already had it removed | upgrade to 0.3.0, re-download the scene (the `BOA_ADD_OFFSET` tag is now correct) |
| `pretrained weights for 'flash-v1' are not published yet` | Flash weights are not released yet | use Pro, or a local Flash checkpoint via `weights=` |
| CPU run far slower than expected inside Docker / Kubernetes | PyTorch starts one thread per host core, not per allowed core | `OMP_NUM_THREADS=<cores you have>` or `torch.set_num_threads(n)` |
| first GPU scene slow on Colab / Kaggle, later ones fast | Triton compiles its kernel once per session | expected; nothing to do |
| GPU memory full, or very slow on Windows laptops | too many tiles per batch for the GPU | `batch=2` (or `--batch 2`); on Windows, exhausted GPU memory spills into system RAM instead of failing |
| odd symbols in the progress bar or summary | console without UTF-8 | harmless; synapse-sr falls back to ASCII; `set PYTHONUTF8=1` on Windows for the full symbols |
| no progress bar | output is piped, logged or run under a test runner | pass `progress=True` to force it, or `SYNAPSE_SR_QUIET=1` to silence it everywhere |

Still stuck? Open an issue at
[github.com/SharadhNaidu/synapse-sr/issues](https://github.com/SharadhNaidu/synapse-sr/issues) and include
the `synapse-sr --env` output.
