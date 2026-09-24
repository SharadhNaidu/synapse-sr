# Limitations

These are the known limits of the current release. Read them before relying on the output.

**Grid spacing is not effective resolution.**
The output grid is 2.0 m. Whether structures at that scale are genuinely resolved is measured separately with
two-bar resolution targets. Each pair is matched with an equal-flux merged-bar negative, so a model that splits
every wide bar is not credited with resolving narrow pairs. Under that bias-controlled protocol, neither the
released Pro checkpoints nor the other learned models we tested (including SEN2SR) certify bar pairs finer than
8 m. No effective-resolution figure is claimed for this release. Treat fine detail in the output as plausible,
not measured, and use `support` and `uncertainty()` to see where it is observation-determined.

**Only red, green, blue and near-infrared are super-resolved.**
B05, B06, B07, B8A, B11 and B12 inform the network as native 20 m context. They are not returned at 2 m.

**Consistency depends on the sensor model and geolocation.**
Exact agreement with the observation holds under the modelled point-spread functions and correct registration.
In simulation:
- with a 10 % error in PSF width, the residual rises to about 1 to 3 noise units;
- with a quarter-pixel registration error, it rises to about 5 to 13 noise units.

**The raw confidence map is not calibrated; `uncertainty()` is.**
`Result.confidence` ranks regions by expected error but is not in physical units. `Result.uncertainty()` and
`Result.interval()` apply the checkpoint's calibrated error model. They were fitted and checked against a 2 m
reference derived from US aerial imagery, and coverage elsewhere is not guaranteed. The support thresholds are
heuristic.

**Training domain.**
The reference imagery used for training is aerial imagery over the United States. Performance on other
landscapes, including Indian urban and agricultural scenes, has not been separately validated.

**CPU speed.**
Pro's Mamba scan is fast on NVIDIA GPUs (Triton or `mamba-ssm`). Elsewhere it runs an exact but slower PyTorch
path. Use Flash on CPU-only machines, or a GPU runtime (Colab, Kaggle) for Pro.
