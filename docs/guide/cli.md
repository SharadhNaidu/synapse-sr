# Command line

```
synapse-sr INPUT OUTPUT [options]
python -m synapse_sr INPUT OUTPUT [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--weights PATH` | | local `.safetensors` checkpoint; no network access |
| `--model NAME` | `pro-v1` | registered model to download when `--weights` is not given |
| `--device DEV` | `cuda` if available | `cuda`, `cpu`, `cuda:1`, ... |
| `--scl PATH` | `auto` | scene classification raster; `auto` uses `<input>_scl.tif`, `none` disables |
| `--offset DN` | tag or 0 | DN offset, e.g. `-1000` for processing baseline 04.00 or later |
| `--tile N` | 64 GPU / 32 CPU | tile size in 10 m pixels |
| `--halo N` | 16 | tile context in 10 m pixels |
| `--no-confidence` | | write the four reflectance bands only |
| `--env` | | print environment diagnostics and exit |
| `--version` | | print the version and exit |

After each run the command prints a one-line JSON summary:

```json
{"output": "scene_2m.tif", "shape": [4, 640, 640], "gsd_m": 2.0,
 "consistency_rms_over_tau": {"B04": 0.95, "B03": 0.89, "B02": 0.81, "B08": 1.0},
 "scan_backend": "fused", "precision": "bfloat16", "invalid_input_fraction": 0.0, "scl": "scene_scl.tif"}
```

Batch processing from the shell:

```bash
for f in scenes/*.tif; do synapse-sr "$f" "out/$(basename "${f%.tif}")_2m.tif"; done
```
