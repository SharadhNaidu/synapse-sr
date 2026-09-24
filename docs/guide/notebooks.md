# Colab and Kaggle

The quick-start notebook runs unchanged on both platforms:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharadhNaidu/synapse-sr/blob/main/notebooks/quickstart.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/SharadhNaidu/synapse-sr/blob/main/notebooks/quickstart.ipynb)

It downloads a real Sentinel-2 scene, super-resolves it, and walks through crop monitoring, urban analysis and
flood / landslide change detection.

## Google Colab

1. Open the notebook with the badge above.
2. *Runtime → Change runtime type → T4 GPU* (optional; the CPU runtime works too, more slowly).
3. Run the first cell:

    ```python
    %pip install -q "synapse-sr[stac,xarray]" matplotlib
    ```

    No restart is needed. Colab already ships PyTorch and Triton.
4. Check the environment:

    ```python
    !synapse-sr --env
    ```

    On a GPU runtime `pro_scan_backend` reads `triton`. Triton compiles its kernel once per session, which
    takes a few seconds on the first scene.

## Kaggle

1. Open the notebook with the badge above, or *Create → New notebook → File → Import notebook* and paste the
   GitHub URL of `notebooks/quickstart.ipynb`.
2. In the right-hand panel: *Session options → Accelerator → GPU T4 x2 or P100*, and *Internet → On*. Internet
   access is needed for the one-time weight download and for `fetch_sentinel2`.
3. Run the cells in order.

!!! tip "Without internet on Kaggle"
    Add the weights as a Kaggle dataset: download `synapse-pro-v2.safetensors` from
    [Hugging Face](https://huggingface.co/SharadhNaiduTrains/synapse-sr) and upload it. Then load it with
    `Pro.from_pretrained(weights="/kaggle/input/<dataset>/synapse-pro-v2.safetensors")`. No network access is
    needed after that.

## Using your own files

=== "Colab"

    ```python
    from google.colab import files
    up = files.upload()                                  # pick your Sentinel-2 GeoTIFF
    r = model.super_resolve(next(iter(up)))
    r.save("result_2m.tif"); files.download("result_2m.tif")
    ```

    Or mount Google Drive: `from google.colab import drive; drive.mount("/content/drive")`.

=== "Kaggle"

    Add the GeoTIFF as a dataset (*Add data → Upload*), then:

    ```python
    r = model.super_resolve("/kaggle/input/<dataset>/scene.tif")
    r.save("/kaggle/working/scene_2m.tif")              # appears under Output
    ```

## Speed and memory

| Runtime | Model | Notes |
|---|---|---|
| Colab / Kaggle GPU | Pro | Triton scan kernel; the default of 8 tiles per batch fits a T4 (16 GB) |
| Colab / Kaggle CPU | Flash | recommended; Pro works but is much slower on CPU |

If the GPU runs out of memory, lower `batch` (tiles per forward pass) or `tile`:

```python
r = model.super_resolve(scene, batch=2)
```
