# Limitations

These are the known limits of the current release. Read them before relying on the output.

**Grid spacing is not effective resolution.**
The output grid is 2.0 m. Whether structures at that scale are genuinely resolved is measured separately,
with two-bar resolution targets and merged-bar negative controls under a fixed protocol. In that test the
v0.1 preview checkpoint does **not** yet certify bar pairs at 8 m or finer, while the official SEN2SR model
certifies 6 m. No effective-resolution figure is claimed for this release. Treat fine detail in the output as
plausible, not measured.

**Only red, green, blue and near-infrared are super-resolved.**
B05, B06, B07, B8A, B11 and B12 inform the network as native 20 m context. They are not returned at 2 m.

**Consistency depends on the sensor model and geolocation.**
Exact agreement with the observation holds under the modelled point-spread functions and correct registration.
In simulation:
- with a 10 % error in PSF width, the residual rises to about 1 to 3 noise units;
- with a quarter-pixel registration error, it rises to about 5 to 13 noise units.

**Confidence is not calibrated.**
The error-scale map ranks regions by expected error. It is not a probability, and the support thresholds are
heuristic.

**Training domain.**
The reference imagery used for training is aerial imagery over the United States. Performance on other
landscapes, including Indian urban and agricultural scenes, has not been separately validated.

**CPU speed.**
The PyTorch path works everywhere but is slow: minutes per square kilometre on a laptop CPU. Use a GPU for
production volumes.
