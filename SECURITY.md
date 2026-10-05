# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.4.x   | :white_check_mark: |
| < 0.4   | Best effort — please upgrade first |

## Reporting a Vulnerability

**Do not open a public issue** for anything that could put users at risk.
Report privately via **GitHub Security Advisories**
(`Security` tab → `Report a vulnerability`) on
[SharadhNaidu/synapse-sr](https://github.com/SharadhNaidu/synapse-sr).

Please include:

- What you ran (command or API call, package version from `synapse-sr --env`)
- What you expected vs what happened (logs, traceback)
- Whether it needs crafted input (a file, a scene, a checkpoint) to trigger —
  attach the smallest reproducer you can
- Your assessment of impact (who is affected, what an attacker gains)

We will acknowledge receipt, investigate against the current release, and
credit reporters in the release notes unless you ask otherwise.

## Scope Notes (What We Treat as Security-Relevant Here)

`synapse-sr` is a local-first scientific tool: no accounts, no servers, no
telemetry. The trust boundaries that matter are:

1. **Model weights.** Shipped checkpoints are SHA-256 pinned in
   `src/synapse_sr/pretrained/*.json` and verified on every load. Never
   replace a pinned hash to silence a mismatch — a mismatch means the file is
   not the published artifact. Third-party loaders (e.g. the LDSR reference
   path) may use pickle checkpoints; treat those files as executable content
   and only load ones you fetched yourself.
2. **Downloaded inputs.** `fetch_sentinel2` and the Hugging Face / STAC fetches
   retrieve remote data over HTTPS into local files. Review anything you did
   not request before executing it; do not pipe fetched content into shells.
3. **Local file writes.** The CLI and `Result.save()` write GeoTIFFs/NPZs/PNGs
   to paths you give them (overwriting without asking unless a folder-mode
   flag says otherwise). Point outputs at empty directories, especially in
   shared or automated runs.
4. **Untrusted scenes.** Super-resolution output is derived data, not ground
   truth: never feed `r.image` into safety-relevant decisions without the
   `support`/`valid` masks and the consistency check in
   `docs/guide/verify.md`.

General hardening (dependency CVEs, CI secret handling) is welcome at any
time; the `security` CI job (`pip-audit`, blocking) is the standing gate.
