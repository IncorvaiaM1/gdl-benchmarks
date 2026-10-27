# Install guide

Rebuild the environment used to compute the block-maxima MLE.

## Platform

- OS: Microsoft Windows NT 10.0.26200.0 (Windows 11), X64
- Host: MAIThinkpad2
- Python: 3.12.5 (`C:\Python312\python.exe`)
- GPU: not used

## Create the environment

From the workspace root (`geo&GenDL/grok_agent_workspace`):

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip numpy scipy
```

On Unix the activate/python path is `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.

## Pinned packages

Generated from `.\.venv\Scripts\python.exe -m pip freeze` after the install above:

```
numpy==2.5.3
scipy==1.18.1
```

pip itself is `26.2.1` (upgraded from the venv default 24.2). Only numpy and scipy are required to run the estimator.

Equivalent one-liner if you already have a venv:

```bash
.\.venv\Scripts\python.exe -m pip install numpy==2.5.3 scipy==1.18.1
```

## Reproduce the estimate

```bash
.\.venv\Scripts\python.exe task\solution\estimate_lambda.py
```

This writes `task/results/estimate.json`.

## Notes

- No system packages, CUDA, or extra index URLs.
- Pins are incidental (latest wheels that installed cleanly on CPython 3.12 / win_amd64 on 2026-09-06). Nothing here works around a known bug.

## Last verified

2026-09-06 against the workspace `.venv` created by the commands above. `task/solution/estimate_lambda.py` exited 0 and wrote `task/results/estimate.json` with `lambda_hat = 1.9947851486676018`.
