import json
import pathlib

MANIFEST_DIR = pathlib.Path(__file__).parent
DEFAULT = "pro-v1"


def manifest(name=DEFAULT):
    p = MANIFEST_DIR / f"{name}.json"
    if not p.exists():
        known = sorted(q.stem for q in MANIFEST_DIR.glob("*.json"))
        raise KeyError(f"unknown pretrained model {name!r}; available: {known}")
    m = json.loads(p.read_text())
    if not str(m.get("url", "")).startswith("http"):
        raise RuntimeError(f"pretrained weights for {name!r} are not published yet; "
                           f"pass a local checkpoint: Pro.from_pretrained(weights='path/to/model.safetensors')")
    return m
