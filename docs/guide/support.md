# Support, confidence and consistency

Every pixel of a super-resolved image mixes two things. The first is what the 10 m measurement determines. The
second is what the model infers from learned priors. synapse-sr keeps the two apart and reports them.

## Decomposition

```python
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

## Confidence

`result.confidence` is a learned per-pixel estimate of the absolute error scale in reflectance units. It is
useful for ranking regions, not as a probability. It has not been calibrated.

## Consistency

`result.consistency` re-observes the output with the Sentinel-2 forward model and compares it with the input,
per band, in units of that band's noise level:

```python
{"B04": 0.95, "B03": 0.89, "B02": 0.81, "B08": 1.00}
```

Values near 1 mean the output explains the input to within sensor noise. The measurement is taken on the
assembled mosaic, so tiling seams would show up here.

!!! warning "What consistency does not prove"
    Agreement holds under the nominal sensor model and correct geolocation. It does not show that the added
    detail is real. That question is answered by reference validation and controlled resolution tests; see
    [Limitations](../limitations.md).
