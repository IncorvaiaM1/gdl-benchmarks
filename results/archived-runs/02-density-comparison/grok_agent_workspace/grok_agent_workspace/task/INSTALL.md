# Install guide

Rebuild the environment from a clean prompt in the workspace root
(`grok_agent_workspace/`). No GPU. No extra system packages.

## Platform (verified 2026-09-23)

- OS: Windows 11 AMD64 (`Windows-11-10.0.26200-SP0`)
- Python: 3.12.5 (`python -V` → `Python 3.12.5`; creator is `C:\Python312\python.exe`)
- CUDA / GPU: not used

## Create the environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy==2.5.3 scipy==1.18.1
```

On Unix the same pins, with `.venv/bin/python` instead of `.venv\Scripts\python.exe`.

`pip` itself ended at `pip==26.2.1` after the upgrade step. That pin is incidental
(whatever 3.12.5's ensurepip shipped was 24.2; the upgrade is not load-bearing).

## Pinned packages (`pip freeze` of this `.venv`)

```
numpy==2.5.3
scipy==1.18.1
```

- `numpy==2.5.3` is load-bearing: histogram of `data/images.csv` and the closed-form
  `E_u[-log s(1-s)]` for Model B's logit Jacobian.
- `scipy==1.18.1` is only used for `scipy.integrate.quad` as a self-check of that
  closed form. It does not enter `answer.json`. An environment with only numpy can
  still reproduce the numbers if the quadrature check is skipped.

No other pip or non-pip dependencies. Dataset is already at `data/images.csv`
(SHA-256 `98b5d6a6d90bc2bc1c22df168e45e63da97fa2cce5d75001650f7f391a8b69c4`;
see `data/DOWNLOAD.md`).

## Reproduce the answer

```powershell
.\.venv\Scripts\python.exe task\solution\solve.py
```

Writes `task/results/answer.json`. Last verified 2026-09-23 against this `.venv`
(created from scratch with the commands above, then the solver run end-to-end).

## Known-bad combinations

None hit. The first solver self-check used a uniform Riemann sum and failed at
`lambda_min` because `-log(1-s)` is singular for white pixels; that is a checker
bug, not an install issue. Adaptive quadrature plus the analytic antiderivative
is stable on numpy 2.5.3 / scipy 1.18.1 / Python 3.12.5.
