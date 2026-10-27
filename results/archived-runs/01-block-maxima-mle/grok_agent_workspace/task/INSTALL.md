# INSTALL.md

## Platform

- OS: Windows 11 (10.0.26200), AMD64
- Python: 3.12.5 (`C:\Python312\python.exe`)
- GPU: not used

## Environment creation

From the workspace root (`grok_agent_workspace`):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy==2.5.3 scipy==1.18.1
```

`pip` itself was upgraded to `26.2.1` during this install. That pin is incidental; numpy and scipy are the load-bearing runtime pins.

## Pinned packages (`pip freeze` of `.venv`, 2026-09-06)

```
numpy==2.5.3
scipy==1.18.1
```

These versions were taken from `.\.venv\Scripts\python.exe -m pip freeze` after a clean install, not written from memory.

## Reproduce the estimate

```powershell
.\.venv\Scripts\python.exe task\solution\estimate_lambda.py
```

This writes `task/results/estimate.json`.

## Notes

- No system packages, CUDA, or extra index URLs.
- `data/train.csv` must be present; do not use `data/holdout.csv`.

## Verification

Followed this file from a second clean venv (`.venv_verify`) on 2026-09-06:

```powershell
python -m venv .venv_verify
.\.venv_verify\Scripts\python.exe -m pip install --upgrade pip
.\.venv_verify\Scripts\python.exe -m pip install numpy==2.5.3 scipy==1.18.1
.\.venv_verify\Scripts\python.exe task\solution\estimate_lambda.py
```

Reproduced `lambda_hat=1.9947851486676018` identically. Recorded in `STATUS.md` under the 17:53 entries. `.venv` is the working environment; `.venv_verify` was only for this check.
