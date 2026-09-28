"""Concatenate the documentation into docs/llms-full.txt (one plain-text file that LLMs and agents can read whole)."""
import pathlib
import re
import yaml

root = pathlib.Path(__file__).resolve().parents[1]
nav = yaml.safe_load(re.sub(r"!!python/\S+", "''", (root / "mkdocs.yml").read_text(encoding="utf-8")))["nav"]


def pages(items):
    for it in items:
        for v in it.values():
            yield from (pages(v) if isinstance(v, list) else [v])


parts = [(root / "docs" / "llms.txt").read_text(encoding="utf-8")]
for p in pages(nav):
    f = root / "docs" / p
    if f.exists() and p != "api.md":
        parts.append(f"\n\n---\n\n<!-- {p} -->\n\n" + f.read_text(encoding="utf-8"))
(root / "docs" / "llms-full.txt").write_text("".join(parts), encoding="utf-8")
print("llms-full.txt", sum(len(x) for x in parts), "chars")
