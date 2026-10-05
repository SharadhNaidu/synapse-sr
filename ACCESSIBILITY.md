# Accessibility Statement

`synapse-sr` is a developer tool (Python package, CLI, notebooks, docs site).
This statement covers what works, what is known not to, and how to report
new barriers. It will be updated as each item below is addressed.

## What Works Today

- **Docs site**: built with MkDocs Material, which provides keyboard
  navigation, skip links, semantic landmarks, and a color-contrast-checked
  default palette. Report any page that breaks keyboard-only use.
- **CLI without a terminal UI**: all progress output degrades to plain
  `stderr` lines when Rich cannot render (piped output, logs, `SYNAPSE_SR_QUIET=1`),
  so screen-reader and scripted use keeps working. Prefer `--json` output and
  `SYNAPSE_SR_QUIET=1` in automation.
- **Headless figures**: every visualization has a non-visual equivalent —
  `Result.summary()` returns the same facts as a dict, `quicklook(path)`
  writes PNGs but all underlying arrays (`image`, `support`, `uncertainty`,
  indices) are plain NumPy, and numerical checks in `docs/guide/verify.md`
  never require looking at an image.

## Known Limitations

- **Support-map colors are not colorblind-safe.** The red/amber/green
  (`#ef4444`/`#f59e0b`/`#22c55e`) support palette in `show()`/`quicklook()`
  is ambiguous under red-green color blindness. Until it is replaced, use the
  `support` array directly (`0`/`1`/`2`) or the `summary()` dict.
- **`show()` needs a display.** In headless environments use the `Agg`
  backend or `quicklook(path)` to a file instead; `show()` itself has no
  text fallback.
- **Progress bars and screen readers.** The Rich live display is verbose
  under screen readers; set `SYNAPSE_SR_QUIET=1` or pass a `progress=False`
  / callback API instead.
- **Notebooks** assume a running kernel with image output; all notebook
  workflows have a script equivalent (`tools/stress.py`, CLI) — ask if one
  is missing.

## Feedback

Accessibility barriers are bugs: open a regular issue (not the security
channel) with your assistive technology, the command or page involved, and
what blocked you. The target for docs pages is WCAG 2.1 AA; CLI/API output
targets machine-readable parity for every visual (see "What Works Today").
