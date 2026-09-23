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
boundaries) is 0.597 for SYNAPSE Pro v1. The comparison figures are SEN2SR 0.548, LDSR-S2 0.570 and bicubic
0.477. An independent held-out evaluation is pending.

## Urban analysis

```python
r = synapse_sr.super_resolve("city.tif")
edges = synapse_sr.boundaries(r, "urban")         # buildings, roads
built = r.indices()["ndbi"]                       # built-up index (20 m SWIR context)
```

The urban benchmark covers 12 Indian cities (Google Open Buildings v3), with the same classifier for every product
and leave-one-city-out evaluation. The table gives the fraction of individual buildings detected:

| Building size | Sentinel-2 10 m | Bicubic | SEN2SR | LDSR-S2 | Satlas | SYNAPSE Pro v1 |
|---|---|---|---|---|---|---|
| < 40 m² | 0.556 | 0.581 | 0.599 | 0.620 | **0.636** | 0.635 |
| 40–100 m² | 0.677 | 0.725 | 0.709 | 0.740 | 0.715 | **0.766** |
| 100–400 m² | 0.812 | 0.855 | 0.831 | 0.867 | 0.783 | **0.894** |
| > 400 m² | 0.932 | 0.961 | 0.952 | 0.960 | 0.841 | **0.970** |

Exact 2 m footprint outlines remain beyond every product. Pixel IoU is 0.35–0.37 for all of them, including native
10 m (Satlas 0.28).

## Water and flood mapping

```python
water = r.indices()["ndwi"] > 0                   # 2 m water mask
shore = synapse_sr.boundaries(r, "water")         # shoreline / flood-front strength
```

## Disaster and change assessment

```python
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
0.5 % false-alarm rate on unchanged ground. SYNAPSE had the best F1: 0.331, against 0.319 for SEN2SR and 0.303
for native 10 m. That lead comes from precision, not stability: SYNAPSE varies slightly more than SEN2SR between
two looks at unchanged ground. LDSR-S2 and Satlas ESRGAN vary much more, so they need a higher threshold and
miss events. LDSR-S2 missed 25 % of debris events. Satlas ESRGAN detected no collapses and
a quarter of the floods.

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
