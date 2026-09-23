# Large scenes and tiling

Scenes of any height and width are processed in tiles with a halo of real context and assembled on the output
grid. The output is always exactly `5H x 5W`: square, portrait, landscape and odd sizes are all supported, and
no region is dropped.

```python
result = model.super_resolve("large_scene.tif", tile=64, halo=16)
```

| Parameter | Default | Meaning |
|---|---|---|
| `tile` | 64 with the fused CUDA kernel, 32 otherwise | tile edge in 10 m source pixels |
| `halo` | 16 | extra context on each side, discarded after processing |

The regularisation weight of the baseline is chosen once per scene, so it is identical across tiles. Tiled and
single-window results agree to 0.4 % relative RMS in the interior. The package tests check this
(`tests/test_api.py::test_tiling_is_seamless`).

## Memory

Peak memory grows with `(tile + 2 * halo)^2`. If a GPU runs out of memory, lower `tile` first. On CPU, keep
`tile` at 32 or below.

## Throughput

| Input (10 m px) | Output (2 m px) | Backend | Time |
|---|---|---|---|
| 128 x 128 | 640 x 640 | fused CUDA, A100 (shared) | 29 s |
| 513 x 677 | 2565 x 3385 | fused CUDA, A100 (shared) | 272 s |
| 64 x 80 | 320 x 400 | PyTorch, laptop CPU | 5.4 min |

Timings include the per-scene baseline calibration and were taken while other jobs shared the GPU.
