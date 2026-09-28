---
description: What 2 m Sentinel-2 imagery from SYNAPSE-SR is used for: building detection in Indian cities, field boundaries, flood and water mapping, disaster change detection.
---

# Applications

synapse-sr is built for the uses named in the SIH problem statement: crop monitoring, urban analysis, water
mapping and disaster assessment. Each needs detail finer than 10 m and an honest statement of how far that detail
can be trusted. Every example below works on any `Result`.

## Crop monitoring

```python
import synapse_sr

r = synapse_sr.super_resolve("fields.tif")
idx = r.indices()
ndvi, savi, evi, ndre = idx["ndvi"], idx["savi"], idx["evi"], idx["ndre"]   # NDRE uses the 20 m red edge

fields = synapse_sr.boundaries(r, "field")        # 0..1 parcel-boundary strength at 2 m
trusted = r.support == 2                          # observation-determined pixels
print("mean NDVI on trusted pixels:", float(ndvi[trusted].mean()))
```

On the development split (128 GeoSR pairs with a 2 m NAIP-derived reference), the NDVI edge F1 (field
boundaries) is 0.602 for SYNAPSE Pro and 0.599 for SYNAPSE Flash (synapse-sr 0.4.1 defaults). The comparison figures
are LDSR-S2 0.570, SEN2SR 0.549, SEN2SR-Lite 0.536 and bicubic 0.478.

## Urban analysis

```python
import synapse_sr

r = synapse_sr.super_resolve("city.tif")
edges = synapse_sr.boundaries(r, "urban")         # buildings, roads
built = r.indices()["ndbi"]                       # built-up index (20 m SWIR context)
```

The urban benchmark covers 12 Indian cities (Google Open Buildings v3), with the same classifier for every product
and leave-one-city-out evaluation. The table gives the fraction of individual buildings detected (synapse-sr 0.4.1):

| Building size | Sentinel-2 10 m | Bicubic | SEN2SR | LDSR-S2 | Satlas | SYNAPSE Flash | SYNAPSE Pro |
|---|---|---|---|---|---|---|---|
| < 40 m² | 0.556 | 0.581 | 0.599 | 0.620 | **0.636** | 0.591 | 0.607 |
| 40–100 m² | 0.677 | 0.725 | 0.709 | **0.740** | 0.715 | 0.710 | 0.728 |
| 100–400 m² | 0.812 | 0.855 | 0.831 | 0.867 | 0.783 | 0.857 | **0.872** |
| > 400 m² | 0.932 | **0.961** | 0.952 | 0.960 | 0.841 | 0.957 | 0.953 |

Exact 2 m footprint outlines remain beyond every product. Pixel IoU is 0.35–0.37 for all of them, including native
10 m (Satlas 0.28).

## Water and flood mapping

```python
import synapse_sr
r = synapse_sr.super_resolve("scene.tif")

water = r.indices()["ndwi"] > 0                   # 2 m water mask
shore = synapse_sr.boundaries(r, "water")         # shoreline / flood-front strength
```

## Disaster and change assessment

```python
import synapse_sr

before = synapse_sr.super_resolve("before.tif")
after = synapse_sr.super_resolve("after.tif")

flood = synapse_sr.change(before, after, "ndwi")          # water advance
burn = synapse_sr.change(before, after, "nbr")            # burn scar (20 m SWIR context)
damage = synapse_sr.change(before, after, "brightness")   # debris, collapse, bare soil
print(damage.area_km2, "km2 changed;", damage.unreliable_fraction, "of pixels excluded as unreliable")
```

`change` only flags pixels that are valid and observation-supported on both dates, so detected change rests on
measured evidence.

In a known-truth benchmark (collapsed structures, debris strips, flood advance), every product was held at the same
0.5 % false-alarm rate on unchanged ground. SYNAPSE Pro had the best F1: 0.355, against 0.319 for SEN2SR, 0.309 for
SYNAPSE Flash and 0.303 for native 10 m (synapse-sr 0.4.1). LDSR-S2 and Satlas ESRGAN vary much more between two looks
at unchanged ground, so they need a higher threshold and miss events: LDSR-S2 missed 25 % of debris events, Satlas
ESRGAN detected no collapses and a quarter of the floods.

## Available indices

| Index | Formula | Bands | Use |
|---|---|---|---|
| NDVI | (N − R) / (N + R) | 2 m | vegetation, crop vigour |
| SAVI | 1.5 (N − R) / (N + R + 0.5) | 2 m | sparse vegetation |
| EVI | 2.5 (N − R) / (N + 6R − 7.5B + 1) | 2 m | dense canopy |
| GNDVI | (N − G) / (N + G) | 2 m | chlorophyll |
| NDWI | (G − N) / (G + N) | 2 m | open water |
| NDRE | (N − B05) / (N + B05) | 2 m × 20 m | crop stress, nitrogen |
| NDBI | (B11 − N) / (B11 + N) | 20 m context | built-up |
| NBR | (N − B12) / (N + B12) | 20 m context | burn severity |
| MNDWI | (G − B11) / (G + B11) | 20 m context | water, flood |

Indices that use a 20 m context band inherit that band's 20 m spatial detail.
