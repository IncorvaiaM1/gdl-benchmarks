# Installation

## Platform

- Windows 11 build 26200, AMD64
- Python 3.12.5 (`python -V`)
- No GPU or CUDA is required.

## Dependencies

The estimator uses only the Python 3.12 standard library. There are no packages
to install and therefore no `pip freeze` entries.

## Rebuild and run

From `geo&GenDL/gdlTests/grok_agent_workspace/grok_agent_workspace`:

```powershell
python -m venv .venv-clean
.\.venv-clean\Scripts\python.exe task\solution\estimate.py
```

The script reads `data/train.csv` and writes `task/results/estimate.json`.

## Verification

These commands were run successfully on 2026-09-06, after which the temporary
environment was removed. The run and its output are recorded in `STATUS.md`.
