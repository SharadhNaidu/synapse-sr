# Support, confidence and consistency

Every pixel of a super-resolved image mixes two things. The first is what the 10 m measurement determines. The
second is what the model infers from learned priors. synapse-sr keeps the two apart and reports them.

## Decomposition

```python
import synapse_sr
result = synapse_sr.super_resolve("scene.tif")

result.x_base   # determined by the Sentinel-2 observation
result.prior    # contributed by the learned prior; invisible to the sensor
```

`x_base` is a deterministic, regularised inversion of the sensor model. `prior` lies in the null space of the
forward operator: re-observing the image with the Sentinel-2 model gives the same measurement with or without
it.

## Support classes

`result.support` measures the prior's contribution against each band's sensor noise level `tau_b`:

| Class | Value | Rule | Reading |
|---|---|---|---|
| HIGH | 2 | max over bands of `|prior| / tau_b < 3` | essentially observation-determined |
| MEDIUM | 1 | between 3 and 10 | the prior adds noticeable structure |
| LOW | 0 | above 10, or invalid input | prior-dominated: treat as inferred |

The thresholds are heuristic and are reported in `result.metadata["support_classes"]`.

## Calibrated uncertainty

```python
import synapse_sr
result = synapse_sr.super_resolve("scene.tif")

err = result.uncertainty()        # (4, 5H, 5W) expected absolute error, reflectance
half = result.interval(0.9)       # reference within image +/- half with probability 0.9
```

`uncertainty()` comes from an error model shipped with the checkpoint. It regresses log |error| on the learned
error scale, the prior's magnitude relative to sensor noise, local edge strength and variance, brightness, NDVI and
band, against a 2 m HR reference (GeoSR development split). `interval()` scales it with split-conformal quantiles.
Measured on development patches not used for fitting or calibration, for Pro v2 (Pro v1: 82 / 91 / 96 %):

| Level | Measured coverage |
|---|---|
| 80 % | 82 % |
| 90 % | 91 % |
| 95 % | 96 % |

The error map also ranks error. Keeping the 50 % of pixels it marks as most reliable roughly halves the mean
absolute error, and its rank correlation with the actual error is 0.58.

`result.confidence` is the network's raw error-scale output. Use `uncertainty()` or `interval()` for decisions.

## Consistency

`result.consistency` re-observes the output with the Sentinel-2 forward model and compares it with the input,
per band, in units of that band's noise level:

```python
{"B04": 5.7, "B03": 3.5, "B02": 3.0, "B08": 3.2}
```

The physics baseline deliberately fits the measurement to about 4 noise units, not 1: a real scene also carries
forward-model error (point-spread function, registration), and fitting it to sensor noise alone turns that error into
false fine detail. Because one regularisation weight is shared by all bands, individual bands can sit a few units
above or below 4. Values far beyond that (above 5 x `discrepancy`) trigger a warning: check the band order, the
radiometric offset and that the input is L2A on the 10 m grid. `super_resolve(..., discrepancy=1)` fits to sensor
noise alone. The learned detail itself is invisible to this measurement by construction (it lies in the null space).
The measurement is taken on the assembled mosaic, so tiling seams would show up here.

