# INSTALL.md — rebuilding the environment

Environment for the block-maxima MLE task (`task/TASK.md`). CPU only, no GPU,
no CUDA. Total install is two wheels and takes about a minute.

## Platform

| | |
|---|---|
| OS | Microsoft Windows 11 Pro, `Microsoft Windows NT 10.0.26200.0` |
| Architecture | AMD64 (x86-64) |
| Shell | PowerShell (commands below are PowerShell; bash equivalents noted) |
| `python -V` | `Python 3.12.5` (system interpreter used to create the venv) |
| GPU / CUDA | not used |

Nothing outside pip is required: no system packages, no environment variables,
no dataset download (`data/train.csv` ships with the repo), no cache configuration.

## Install commands, verbatim and in order

Run from the repository root
(`.../gdlTests/claude_agent_workspace`, the directory containing `data/` and `task/`).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade "pip==24.3.1"
.\.venv\Scripts\python.exe -m pip install "numpy==2.2.1" "scipy==1.15.0"
```

On Linux/macOS the same three commands are:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade "pip==24.3.1"
./.venv/bin/python -m pip install "numpy==2.2.1" "scipy==1.15.0"
```

`pip install pip==24.3.1` prints a `[notice] A new release of pip is available`
line on stderr. PowerShell surfaces that as a `NativeCommandError` record even
though the exit status is 0. It is cosmetic — check the exit code, not the
presence of red text.

## Exact package versions

Generated with `.\.venv\Scripts\python.exe -m pip freeze` (do not retype from
memory), plus the pinned pip:

```
numpy==2.2.1
scipy==1.15.0
```

```
pip==24.3.1   # not in pip freeze output; installed explicitly by the command above
```

That is the complete list — `venv` creates no other packages on Python 3.12.5,
and neither wheel pulls further runtime dependencies.

### Which pins are load-bearing

None of these pins work around a bug; they are the versions the reported result
was produced with, pinned for reproducibility rather than necessity.

- **`scipy==1.15.0`** — used only for `scipy.optimize.brentq` and
  `minimize_scalar`, both long-stable APIs. Any SciPy ≥ 1.10 should work.
  Incidental, not load-bearing.
- **`numpy==2.2.1`** — used for array arithmetic, `expm1`, and
  `numpy.random.default_rng`. Incidental for `estimate_lambda.py`. **Mildly
  load-bearing for `selfcheck.py`:** that script seeds `default_rng(20260906)`,
  and NumPy only guarantees stream compatibility within the Generator API's
  versioning policy, so simulated numbers printed by the self-checks can differ
  on another NumPy version. The checks are all tolerance-based and will still
  pass; only the exact digits move. The reported `lambda_hat` does not depend on
  any RNG at all.
- **`pip==24.3.1`** — pinned purely so the install transcript is reproducible.
  Any pip that can install the two wheels is fine.

## Running the project

```powershell
.\.venv\Scripts\python.exe task\solution\estimate_lambda.py   # writes task/results/estimate.json
.\.venv\Scripts\python.exe task\solution\selfcheck.py          # validation, exits non-zero on failure
```

Both scripts resolve `data/train.csv` and `task/results/estimate.json` from
their own location, so they can be invoked from any working directory.

## Known-bad combinations

- **Python < 3.9 will not run the solution.** `estimate_lambda.py` uses PEP 585
  builtin generics in annotations (`dict[str, float]`). `from __future__ import
  annotations` is present, which defers evaluation, but 3.9+ is still the floor.
  Developed and verified on 3.12.5.
- No other incompatibilities were encountered; nothing had to be worked around
  during install.

## Verification status

Last verified **2026-09-06** by creating a throwaway virtualenv (`.venv-verify`)
from the system Python 3.12.5, running only the three install commands above
verbatim, and then running both scripts end to end. `estimate.json` was
byte-identical to the one produced by `.venv`, and all self-checks passed. See
the corresponding `STATUS.md` entry for the captured output.
