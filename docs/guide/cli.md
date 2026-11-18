# Command line

```bash
synapse-sr INPUT.tif OUTPUT.tif [options]
```

In a terminal you get a live progress bar (stages, tile count, elapsed time and ETA), then a summary table:

```text
                    synapse-sr result
model          synapse-pro-v2
output         640 × 640 px · 2 m · bands B04 B03 B02 B08
backend        pytorch · bfloat16 · tile 32 halo 16
consistency    B04 0.91τ  B03 0.80τ  B02 0.68τ  B08 0.99τ
support        HIGH 22%  MEDIUM 48%  LOW 29%
invalid input  0.0%
time           80.3 s
done wrote OUTPUT.tif
```

## Common tasks

| Task | Command |
|---|---|
| Default (Flash, best available device) | `synapse-sr scene.tif scene_2m.tif` |
| Highest detail (GPU) | `synapse-sr scene.tif scene_2m.tif --model pro` |
| Offline, local weights | `synapse-sr scene.tif scene_2m.tif --weights synapse-pro-v2.safetensors` |
| Explicit cloud mask | `synapse-sr scene.tif scene_2m.tif --scl scene_SCL_20m.tif` |
| No cloud masking | `synapse-sr scene.tif scene_2m.tif --scl none` |
| Raw DN from baseline ≥ 04.00 without a tag | `synapse-sr scene.tif scene_2m.tif --offset -1000` |
| Four bands only (for a GIS) | `synapse-sr scene.tif scene_2m.tif --no-confidence` |
| Low GPU memory | `synapse-sr scene.tif scene_2m.tif --batch 2` |
| Part of a large tile | `synapse-sr T43PGQ.tif part.tif --window 2000,3000,1000,1000` |
| A whole folder, four bands, previews | `synapse-sr scenes/ out/ --no-confidence --preview previews/` |
| Scripts and pipelines | `synapse-sr scene.tif scene_2m.tif --json` (JSON on stdout, no bars; per-file lines of folder runs go to stderr) |
| Diagnose the machine | `synapse-sr --env` |
| What models exist | `synapse-sr --models` |

## All options

| Option | Default | Meaning |
|---|---|---|
| `--model` | `flash` | `flash`, `pro`, or a registered name such as `pro-v1` |
| `--weights` | | local `.safetensors` checkpoint; no network access needed |
| `--device` | CUDA, then MPS (Flash), then CPU | `cuda`, `cuda:1`, `cpu`, `mps` |
| `--backend`, `--backend-device` | | Flash accelerators, see [Hardware](hardware.md) |
| `--window ROW,COL,H,W` | | process only this pixel window |
| `--bounds L,B,R,T` | | process only this area, in the input CRS |
| `--scl` | `auto` | scene classification raster; `auto` uses `<input>_scl.tif` when it exists; `none` disables |
| `--offset` | tag or 0 | DN offset added before dividing by 10 000 |
| `--tile` | largest that fits memory | tile size in 10 m pixels |
| `--halo` | 16 | context around each tile, in 10 m pixels |
| `--batch` | as many as fit, up to 8, on CUDA / 1 CPU | tiles per forward pass |
| `--no-confidence` | | write B04 B03 B02 B08 only (also in folder mode) |
| `--preview PNG` | | save a 10 m / 2 m / support quicklook; a folder of `<name>.png` in folder mode |
| `--cog` | | Cloud-Optimised GeoTIFF |
| `--fetch LAT,LON`, `--dates START:END`, `--size-m`, `--rank` | | download a scene first, see [Data](data.md) |
| `--json` | | machine-readable summary on stdout |
| `--quiet` | | no progress bar, summary or per-file lines |
| `--env`, `--models`, `--version` | | print and exit (`--env --json` for JSON) |

## Output file

| Bands | Content |
|---|---|
| 1-4 | B04 B03 B02 B08 surface reflectance, float32, NaN (the declared NoData) where the input was invalid |
| 5-8 | `ERRSCALE_*`: learned per-pixel error scale for each band |
| 9 | `SUPPORT`: 2 observation-determined, 1 medium, 0 prior-dominated or invalid |

Same CRS and bounds as the input, with a 2 m pixel.

## Exit codes

`0` success, `1` an error (unreadable file, unsupported grid, missing weights, a bad `--window` / `--fetch`
argument, or any failed file in folder mode), `2` a usage error, `130` interrupted. Every error prints a message
that says what to change; see [Troubleshooting](../troubleshooting.md).
