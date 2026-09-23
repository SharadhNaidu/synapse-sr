import json
import pathlib

MANIFEST_DIR = pathlib.Path(__file__).parent
DEFAULT = "pro-v1"


def manifest(name=DEFAULT):
    p = MANIFEST_DIR / f"{name}.json"
    if not p.exists():
        known = sorted(q.stem for q in MANIFEST_DIR.glob("*.json"))
        raise KeyError(f"unknown pretrained model {name!r}; available: {known}")
    return json.loads(p.read_text())
