# How it works

```
x_hat = x_base + P_N(delta)
```

| Component | Role |
|---|---|
| `A` | Sentinel-2 forward operator: per-band point-spread function on a 0.5 m grid, sampled at source-pixel centres |
| `x_base` | deterministic baseline: bicubic anchor corrected by Tikhonov regularisation, with the weight chosen per band so the residual matches the sensor noise level (Morozov discrepancy principle) |
| `delta` | structural correction predicted by the network |
| `P_N` | projector onto the null space of `A`, `I - A^T (A A^T)^+ A`: removes every component the sensor could have observed |

Because `A P_N = 0`, re-observing `x_hat` gives exactly what re-observing `x_base` gives, whatever the network
predicts. The network can add detail below the sensor's resolving power. It cannot contradict the
measurement.

## Network

`SynapseProX5`, 14.4 M parameters:

- **State-space backbone**: 6 residual groups of 8 visual state-space blocks, with four-direction scanning so
  that no orientation is preferred, followed by channel attention.
- **Two stems**: an RGBN spatial stem, plus a gated stem for the six 20 m bands. The gate starts closed, so the
  20 m context is used only as far as training finds it useful.
- **Frequency mixer**: splits features into a local low band and its high-frequency complement, mixes each,
  and re-fuses them.
- **Direct x5 head**: PixelShuffle x5 with ICNR initialisation, so all 25 sub-pixel phases start identical and
  no 10 m cell pattern is imprinted.
- **Two outputs**: `delta` (2 m, RGBN) and a confidence map computed from detached features, so the confidence
  loss cannot steer the reconstruction.

## Geometry

x5 is odd. Output pixel `5i + r` spans `[10i + 2r, 10i + 2r + 2]` m inside source pixel `i`, and `r = 2` is
centred on the source-pixel centre. The output grid therefore shares the input origin exactly.

## Training (summary)

The network was trained on Sentinel-2 L2A paired with 0.6 m aerial reference imagery. The reference was
registered to Sentinel-2, integrated onto the 2 m grid, and gain-matched per band to Sentinel-2 through the
forward operator. Every loss term acts only on the null-space component `P_N(x_hat - target)`. The model is
therefore never trained to reproduce cross-sensor radiometric differences that the Sentinel-2 measurement
already determines. The backbone was initialised from the public SEN2SR weights (CC0); see
[Acknowledgements](changelog.md#acknowledgements).
