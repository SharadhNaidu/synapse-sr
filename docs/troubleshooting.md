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
| `CUDA out of memory` | tile too large for the GPU | lower `tile` (for example `tile=32`) or set `SYNAPSE_SR_MEMORY=2` |
| `MemoryError: a H x W px scene needs about N GB for its 2 m output` | the whole scene does not fit | process parts with `window=` / `bounds=` (CLI `--window` / `--bounds`) of the size the message gives |
| `onnx:cpu failed on a ... window; running it in PyTorch instead` (warning) | the accelerator rejected a window | the result is correct (same model); `metadata['detail_backend_fallbacks']` counts such windows; report it if it happens on every window |
| `MPS could not run this model; using the CPU` (warning) | an operation is missing on Apple MPS | nothing to do; pass `device="mps"` to see the error |
| `could not download ... after 4 attempts` | network or firewall | rerun to resume, set `HF_ENDPOINT` to a mirror, or pass `weights=` |
| `before and after cover different areas` | `change()` on two scenes with different grids | clip both scenes to the same 10 m grid first |
| QGIS shows a black frame around the scene | file from synapse-sr 0.4.2 or older, NoData not declared | set NoData to `nan` in the layer properties, or re-save with 0.4.3 |
| `GeoTIFF IO needs rasterio` | rasterio missing | `pip install rasterio` (conda-forge on Windows if pip fails) |
| output much darker or brighter than expected | offset mismatch | check the product's processing baseline; pass `offset=-1000` or `offset=0` |
| scenes from `fetch_sentinel2` in v0.2.0 look too dark, NDVI near 1 everywhere | v0.2.0 applied the baseline 04.00 offset to Earth Search data that already had it removed | upgrade to 0.3.0, re-download the scene (the `BOA_ADD_OFFSET` tag is now correct) |
| CPU run far slower than expected inside Docker / Kubernetes | PyTorch starts one thread per host core, not per allowed core | `OMP_NUM_THREADS=<cores you have>` or `torch.set_num_threads(n)` |
| first GPU scene slow on Colab / Kaggle, later ones fast | Triton compiles its kernel once per session | expected; nothing to do |
| GPU memory full, or very slow on Windows laptops | too many tiles per batch for the GPU | `batch=2` (or `--batch 2`); on Windows, exhausted GPU memory spills into system RAM instead of failing |
| odd symbols in the progress bar or summary | console without UTF-8 | harmless; synapse-sr falls back to ASCII; `set PYTHONUTF8=1` on Windows for the full symbols |
| no progress bar | output is piped, logged or run under a test runner | pass `progress=True` to force it, or `SYNAPSE_SR_QUIET=1` to silence it everywhere |

Still stuck? Open an issue at
[github.com/SharadhNaidu/synapse-sr/issues](https://github.com/SharadhNaidu/synapse-sr/issues) and include
the `synapse-sr --env` output.
