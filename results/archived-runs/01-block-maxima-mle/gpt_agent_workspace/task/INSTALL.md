# INSTALL.md

YOUR INSTALLATION GUIDE GOES HERE

## Platform

- Windows 11 10.0.26200, AMD64, 64-bit
- Python 3.12.5 (`python -V`)
- No GPU or CUDA is used.

## Environment

The estimator uses only the Python standard library. There are no third-party
packages to install or pin, so a fresh virtual environment has an empty
`pip freeze`.

From the workspace root, create a clean environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -V
python -m pip freeze
```

Expected version and dependency output:

```text
Python 3.12.5
```

`python -m pip freeze` should print nothing.

## Reproduce the result

With the virtual environment active and from the workspace root:

```powershell
python task/solution/estimate.py
```

This reads only `data/train.csv` and writes
`task/results/estimate.json`.
