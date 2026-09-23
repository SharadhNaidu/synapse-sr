# Contributing to synapse-sr

Thank you for helping improve synapse-sr. Bug reports, documentation fixes and code contributions are all
welcome.

## Reporting a bug

Open an issue at https://github.com/SharadhNaidu/synapse-sr/issues and include:

- the output of `synapse-sr --env`;
- the command or code you ran, and the full error message;
- if possible, a small input that reproduces the problem (for example a 64 x 64 crop).

## Development setup

```bash
git clone https://github.com/SharadhNaidu/synapse-sr.git
cd synapse-sr
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all,test,docs]"
```

## Running the checks

```bash
pytest                          # full API contract on CPU, no weights needed (about 5 minutes)
pytest --cov=synapse_sr         # with coverage
mkdocs serve                    # documentation at http://127.0.0.1:8000
mkdocs build --strict           # what CI runs
python -m build && twine check --strict dist/*
```

The tests use a randomly initialised network and a small Gaussian point-spread function, so every code path
runs without downloading the trained checkpoint.

## Pull requests

1. Branch from `main`.
2. Keep changes focused. Add or update tests for any behaviour change, and update `docs/` and `CHANGELOG.md`.
3. Make sure `pytest` and `mkdocs build --strict` pass.
4. Describe what changed and why in the pull request.

## Code style

- Clear names and small functions; comments only where the code cannot say it.
- Public functions and classes carry numpy-style docstrings. The API reference is generated from them.
- The code must stay compatible with Python 3.8 and PyTorch 1.13.

## Releases

@SharadhNaidu publish by creating a GitHub release. The `publish` workflow builds the package and uploads it to
PyPI through trusted publishing.
