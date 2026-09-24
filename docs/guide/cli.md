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
| Default (Pro, best available device) | `synapse-sr scene.tif scene_2m.tif` |
| Fast model on a CPU | `synapse-sr scene.tif scene_2m.tif --model flash --device cpu` |
| Offline, local weights | `synapse-sr scene.tif scene_2m.tif --weights synapse-pro-v2.safetensors` |
| Explicit cloud mask | `synapse-sr scene.tif scene_2m.tif --scl scene_SCL_20m.tif` |
| No cloud masking | `synapse-sr scene.tif scene_2m.tif --scl none` |
| Raw DN from baseline ≥ 04.00 without a tag | `synapse-sr scene.tif scene_2m.tif --offset -1000` |
| Four bands only (for a GIS) | `synapse-sr scene.tif scene_2m.tif --no-confidence` |
| Low GPU memory | `synapse-sr scene.tif scene_2m.tif --batch 2` |
| Scripts and pipelines | `synapse-sr scene.tif scene_2m.tif --json` (one JSON line on stdout, no bars) |
| Diagnose the machine | `synapse-sr --env` |
| What models exist | `synapse-sr --models` |

## All options

| Option | Default | Meaning |
|---|---|---|
| `--model` | `pro` | `pro`, `flash`, or a registered name such as `pro-v1` |
| `--weights` | | local `.safetensors` checkpoint; no network access needed |
| `--device` | CUDA if available | `cuda`, `cuda:1`, `cpu`, `mps` |
| `--scl` | `auto` | scene classification raster; `auto` uses `<input>_scl.tif` when it exists; `none` disables |
| `--offset` | tag or 0 | DN offset added before dividing by 10 000 |
| `--tile` | 64 GPU scan / 32 PyTorch / 128 Flash | tile size in 10 m pixels |
| `--halo` | 16 | context around each tile, in 10 m pixels |
| `--batch` | 8 CUDA / 1 CPU | tiles per forward pass |
| `--no-confidence` | | write B04 B03 B02 B08 only |
| `--json` | | machine-readable summary on stdout |
| `--quiet` | | no progress bar or summary |
| `--env`, `--models`, `--version` | | print and exit (`--env --json` for JSON) |

## Output file

| Bands | Content |
|---|---|
| 1-4 | B04 B03 B02 B08 surface reflectance, float32, NaN where the input was invalid |
| 5-8 | `ERRSCALE_*`: learned per-pixel error scale for each band |
| 9 | `SUPPORT`: 2 observation-determined, 1 medium, 0 prior-dominated or invalid |

Same CRS and bounds as the input, with a 2 m pixel.

## Exit codes

`0` success. Any error (unreadable file, unsupported grid, missing weights) exits non-zero with a message that
says what to change; see [Troubleshooting](../troubleshooting.md).
