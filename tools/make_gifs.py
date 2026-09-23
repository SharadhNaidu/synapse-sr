"""Before / after GIFs for the documentation, produced with the synapse-sr package itself.

    python tools/make_gifs.py --weights synapse-pro-v2.safetensors --out assets/gifs

Each scene is fetched from Earth Search (``synapse_sr.fetch_sentinel2``), super-resolved, and rendered as a wipe
between the Sentinel-2 10 m input (nearest-neighbour, no smoothing) and the 2 m output, both with the same
true-colour stretch.
"""

import argparse
import json
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import synapse_sr
from synapse_sr import Pro, fetch_sentinel2
from synapse_sr.io import geotiff
from synapse_sr.io.sentinel2 import select_bands, to_reflectance

SCENES = [
    {"name": "rv_university", "title": "RV University, Bengaluru", "lat": 12.9237, "lon": 77.4987,
     "start": "2025-01-01", "end": "2025-03-15"},
    {"name": "bengaluru_urban", "title": "Bengaluru, India - urban", "lat": 12.9716, "lon": 77.5946,
     "start": "2025-01-01", "end": "2025-03-15"},
    {"name": "punjab_fields", "title": "Ludhiana, Punjab - agriculture", "lat": 30.935, "lon": 75.800,
     "start": "2024-11-01", "end": "2025-01-31"},
    {"name": "wayanad_landslide", "title": "Wayanad, Kerala - landslide-affected hills", "lat": 11.475, "lon": 76.135,
     "start": "2025-01-01", "end": "2025-03-31"},
]


def font(size):
    for f in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def stretch(img, lo, hi):
    return (np.clip((img - lo) / (hi - lo), 0, 1) ** (1 / 1.2) * 255).astype(np.uint8).transpose(1, 2, 0)


def label(im, text, xy, anchor):
    d = ImageDraw.Draw(im)
    f = font(17)
    box = d.textbbox(xy, text, font=f, anchor=anchor)
    d.rectangle((box[0] - 8, box[1] - 6, box[2] + 8, box[3] + 6), fill=(0, 0, 0))
    d.text(xy, text, font=f, fill=(255, 255, 255), anchor=anchor)


def wipe_gif(lr, sr, path, size=480, steps=14):
    a = Image.fromarray(lr).resize((size, size), Image.NEAREST)
    b = Image.fromarray(sr).resize((size, size), Image.LANCZOS)
    xs = list(np.linspace(0.08, 0.92, steps)) + [0.92] * 6 + list(np.linspace(0.92, 0.08, steps)) + [0.08] * 6
    frames = []
    for f in xs:
        x = int(f * size)
        im = a.copy()
        im.paste(b.crop((x, 0, size, size)), (x, 0))
        d = ImageDraw.Draw(im)
        d.line((x, 0, x, size), fill=(255, 255, 255), width=3)
        label(im, "Sentinel-2  10 m", (14, size - 14), "ld")
        label(im, "synapse-sr  2 m", (size - 14, size - 14), "rd")
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=160))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="assets/gifs")
    ap.add_argument("--size-m", type=float, default=1280)
    ap.add_argument("--device")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = Pro.from_pretrained(weights=a.weights, device=a.device)
    record = {"synapse_sr": synapse_sr.__version__, "weights": str(a.weights), "scenes": []}
    for s in SCENES:
        tif = str(out / f"{s['name']}.tif")
        if not pathlib.Path(tif).exists():
            fetch_sentinel2(s["lat"], s["lon"], s["start"], s["end"], size_m=a.size_m, out=tif)
        r = model.super_resolve(tif)
        sr = r.image[:3]
        lo, hi = np.percentile(sr[np.isfinite(sr)], (2, 98))
        arr, _, names, tags = geotiff.read(tif)
        y = to_reflectance(select_bands(arr, names), offset=float(tags.get("BOA_ADD_OFFSET", 0)))[:3]
        wipe_gif(stretch(y, lo, hi), stretch(sr, lo, hi), out / f"{s['name']}.gif")
        Image.fromarray(stretch(sr, lo, hi)).save(out / f"{s['name']}_2m.png")
        record["scenes"].append({**s, "stac_item": tags.get("STAC_ITEM"), "cloud": tags.get("CLOUD_COVER"),
                                 "consistency": r.consistency, "support_fraction":
                                 {int(k): float((r.support == k).mean()) for k in (0, 1, 2)}})
        print(s["name"], r.consistency, flush=True)
    (out / "gifs.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
