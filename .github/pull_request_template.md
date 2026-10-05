## What and why

<!-- One or two sentences: what changes for users, and why. Link issues with
"Fixes #NNN" or "Relates to #NNN". -->

## Verification

<!-- Paste evidence, not adjectives. At least one of: -->

- [ ] `pytest -q` green locally (no weights or network needed)
- [ ] `python tools/stress.py --model flash --device cpu` green (needs weights)
- [ ] New/changed behavior covered by a test or stress case
- [ ] Docs updated (`docs/`, docstrings, or `CHANGELOG.md` as appropriate)

## Checklist

- [ ] No model weights (`*.safetensors`, `*.ckpt`, `*.pt`), datasets, or large rasters in the diff
- [ ] No secrets, tokens, or machine-local paths
- [ ] Public API/error-message changes are documented where users will look first
- [ ] Follows the existing style (lazy optional imports, actionable errors naming the pip extra)

## Notes for reviewers (optional)

<!-- Anything surprising: tradeoffs taken, alternatives rejected, follow-ups left open. -->
