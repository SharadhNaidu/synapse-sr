# Troubleshooting

Start with the environment report:

```bash
synapse-sr --env
```

| Message | Cause | Fix |
|---|---|---|
| `pretrained weights for 'pro-v1' are not published yet` | registry entry without a download URL | pass `weights="path/to/model.safetensors"` |
| `checksum mismatch for ...` | incomplete or modified download | delete the file named in the message and retry |
| `cannot map N channels to Sentinel-2 bands` | unknown band layout | pass `band_names=[...]` or write band descriptions into the GeoTIFF |
| `expected the Sentinel-2 10 m grid, got 20.0 m pixels` | input not on the 10 m grid | resample to 10 m first (`gdalwarp -tr 10 10 -r near`) |
| `rotated or sheared geotransforms are not supported` | non north-up raster | `gdalwarp` to a north-up grid |
| `select one 'time' first` | multi-date xarray cube | pass `cube.isel(time=k)` |
| `mamba-ssm selective_scan_cuda is installed but unusable` (warning) | kernel built for another PyTorch or CUDA | reinstall `mamba-ssm` with `--no-build-isolation`; results are unaffected, only slower |
| `CUDA out of memory` | tile too large for the GPU | lower `tile` (for example `tile=32`) |
| `GeoTIFF IO needs rasterio` | rasterio missing | `pip install rasterio` (conda-forge on Windows if pip fails) |
| output much darker or brighter than expected | offset mismatch | check the product's processing baseline; pass `offset=-1000` or `offset=0` |

Still stuck? Open an issue at
[github.com/SharadhNaidu/synapse-sr/issues](https://github.com/SharadhNaidu/synapse-sr/issues) and include
the `synapse-sr --env` output.
