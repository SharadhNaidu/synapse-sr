# FAQ

### Is the 2 m output "real" 2 m detail?

Partly, and synapse-sr tells you which part. A 10 m pixel cannot uniquely determine the 25 pixels of 2 m inside
it. The output therefore has two components:

- `x_base` is fixed by the measurement;
- `prior` is inferred by the network and is invisible to the sensor.

`support` labels each pixel by which component dominates, and `uncertainty()` gives a calibrated expected error.
The output grid is 2 m. The effective resolution, meaning how fine a structure is genuinely resolved, is coarser
and is stated in [Limitations](limitations.md).

### How is this different from bicubic, SEN2SR or a GAN?

Bicubic adds no information. Learned models, including SEN2SR and GAN-based methods, add plausible detail but do
not tell you where it came from, and most can alter what the satellite measured. synapse-sr constrains the
network so that re-observing the output reproduces the measurement (`consistency ≈ 1` noise unit). It also returns
the observed / inferred split with every result. You can check all of this yourself:
[Verify it yourself](guide/verify.md).

### Which bands come out at 2 m?

B04, B03, B02 and B08 (red, green, blue, near-infrared). The six 20 m bands (B05, B06, B07, B8A, B11 and B12)
help the network as spectral context. They are available on the output grid via `r.band("B11")` and in the
red-edge and SWIR indices, but they are **not** super-resolved.

### Pro or Flash?

Use Pro on a GPU (Colab, Kaggle, a workstation). Flash, the CPU-oriented model, is not released yet. Until it is, Pro runs on CPU too, only more slowly.
The physics guarantees are identical. See [Choosing a model and a device](guide/models.md).

### How long does it take?

Measured on a 1.28 km x 1.28 km scene (128 x 128 input pixels, 640 x 640 output):

| Hardware | Pro |
|---|---|
| A100 slice, Triton kernel | 5.2 s (first call 22 s, including the one-time compile and weight download) |
| Laptop RTX 4070, Windows, PyTorch scan | 80 s |
| Kaggle free CPU runtime (2 threads) | more than 30 min: use a GPU runtime, or a smaller area |

Time grows with area: a 10 km x 10 km scene has about 61x as many pixels.

### Does it need the internet?

Only to download the weights once (~58 MB, SHA-256 checked) and, if you use it, for `fetch_sentinel2`. After that
everything runs offline, or from a local file with `from_pretrained(weights=...)`. See
[Offline use](guide/offline.md).

### What input do I need?

A Sentinel-2 **L2A** (surface reflectance) scene on the 10 m grid that contains the 10 model bands, as a GeoTIFF,
a numpy or torch array, or xarray. DN or reflectance are both fine, and band order is detected from names. L1C
top-of-atmosphere data is accepted as input, but the model was trained on L2A. See
[Inputs](guide/inputs.md).

### Why are some pixels NaN or red in the support map?

NaN means the input was invalid there (NoData, cloud, cloud shadow, cirrus or saturation, from the SCL layer).
Red, or support 0, means the prior dominates, which is often the case at sharp edges and along the scene border,
where tiles have no context on one side. Use `support == 2` for decisions that must rest on the measurement.

### My result looks darker or brighter than expected.

This is almost always a radiometric offset problem. Products of processing baseline 04.00 and later store DN
with +1000. synapse-sr reads the `BOA_ADD_OFFSET` tag. If your file lacks it, pass `offset=-1000` for raw DN, or
`offset=0` for data where the offset was already removed (for example GEE `S2_SR_HARMONIZED` and Earth Search).
See also [Troubleshooting](troubleshooting.md).

### Can I use it commercially? How do I cite it?

The package is CC0-1.0. Parts derive from SEN2SR (CC0-1.0); see `THIRD_PARTY_NOTICES`. The citation is in the
README.
