# Large scenes and tiling

Scenes of any height and width are processed in tiles with a halo of real context and assembled on the output
grid. The output is always exactly `5H x 5W`: square, portrait, landscape and odd sizes are all supported, and
no region is dropped.

```python
from synapse_sr import Pro
model = Pro.from_pretrained()

result = model.super_resolve("large_scene.tif", tile=64, halo=16)
```

| Parameter | Default | Meaning |
|---|---|---|
| `tile` | automatic: the largest tile that fits the free GPU memory (or half the available RAM, at most 6 GB, on CPU) | tile edge in 10 m source pixels |
| `batch` | automatic on GPU (up to 8), 1 on CPU | tiles per forward pass |
| `halo` | 16 (below `tile`) | extra context on each side, discarded after processing |

The regularisation weight of the baseline is chosen once per scene, so it is identical across tiles. Tiled and
single-window results agree to 0.4 % relative RMS in the interior. The package tests check this
(`tests/test_api.py::test_tiling_is_seamless`).

## Memory

Peak memory grows with `(tile + 2 * halo)^2`. The automatic choice leaves headroom; if memory is still short, pass a
smaller `tile` or `batch`, or cap the budget with `SYNAPSE_SR_MEMORY=2` (GB). Larger tiles are faster: a 32-pixel
tile with a 16-pixel halo computes four times its output area, a 96-pixel tile under twice.

The whole 2 m output is held in memory: about 3.2 KB per 10 m input pixel (2.9 KB with `context=False`), so a
1000 x 1000 px (10 km) scene needs about 3 GB and a full 110 km tile several hundred GB. Before any work,
`super_resolve` compares this with the available RAM (and free GPU memory) and raises a `MemoryError` that says
how large a part fits. `SYNAPSE_SR_PREFLIGHT=0` skips the check.

## Part of a large scene

Process one area of a large GeoTIFF without reading the rest of it:

```python
import synapse_sr

part = synapse_sr.super_resolve("T43PGQ.tif", window=(2000, 3000, 1000, 1000))     # row, col, height, width
area = synapse_sr.super_resolve("T43PGQ.tif", bounds=(770000, 1430000, 780000, 1440000))  # left, bottom, right, top
```

```bash
synapse-sr T43PGQ.tif part_2m.tif --window 2000,3000,1000,1000
synapse-sr T43PGQ.tif area_2m.tif --bounds 770000,1430000,780000,1440000
```

`bounds` are in the raster's CRS. The output is georeferenced to that part, and the `_scl.tif` sidecar (or an
`scl=` path) is read over the same area, whatever its resolution.

## Throughput

| Input (10 m px) | Output (2 m px) | Backend | Time |
|---|---|---|---|
| 128 x 128 | 640 x 640 | fused CUDA, A100 (shared) | 29 s |
| 513 x 677 | 2565 x 3385 | fused CUDA, A100 (shared) | 272 s |
| 64 x 80 | 320 x 400 | PyTorch, laptop CPU | 5.4 min |

Timings include the per-scene baseline calibration and were taken while other jobs shared the GPU.
