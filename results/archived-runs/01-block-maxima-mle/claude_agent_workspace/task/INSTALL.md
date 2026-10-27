# INSTALL.md — how to rebuild the environment

Environment for `task/solution/` (maximum likelihood estimation from block
maxima of an Exponential distribution).

Repository root referred to below as `$REPO`:

```
C:\Users\micha\MAIStuff\randLab\geo&GenDL\gdlTests\claude_agent_workspace
```

Note the literal `&` in the path. In PowerShell any command that mentions this
path must quote it; in `cmd.exe` an unquoted `&` splits the command line.

---

## 1. Platform

| Item | Value |
|---|---|
| OS | Windows 10.0.26200 (win32), x86_64 / AMD64 |
| Shell | PowerShell 5.1 (the default `powershell.exe`) |
| System Python | `C:\Python312\python.exe`, `Python 3.12.5` |
| Venv Python | `$REPO\.venv\Scripts\python.exe`, `Python 3.12.5` |
| GPU / CUDA | not used — this is a CPU-only, single-threaded numerics task |

`python -V` output, verbatim:

```
Python 3.12.5
```

---

## 2. Install commands, verbatim and in order

Run from `$REPO`. These are the exact commands that were used (PowerShell):

```powershell
Set-Location "C:\Users\micha\MAIStuff\randLab\geo&GenDL\gdlTests\claude_agent_workspace"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade "pip==25.2"
.\.venv\Scripts\python.exe -m pip install -r task\solution\requirements.txt
```

On Linux/macOS the equivalent is:

```bash
cd "$REPO"
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade "pip==25.2"
./.venv/bin/python -m pip install -r task/solution/requirements.txt
```

No system packages, no conda, no compilers are needed: `numpy` and `scipy`
install from prebuilt wheels on this platform. No dataset download is needed —
`data/train.csv` is already in the repo (see `data/DOWNLOAD.md`). No environment
variables are required. Pip's wheel/HTTP cache stays at its default location
(`%LOCALAPPDATA%\pip\Cache` on this machine).

## 3. Pinned package versions

`task/solution/requirements.txt` holds the pins:

```
numpy==2.5.3
scipy==1.18.1
mpmath==1.3.0
```

Full `pip freeze` of the venv after installing (generated, not written from
memory):

```
mpmath==1.3.0
numpy==2.5.3
scipy==1.18.1
```

Build/bootstrap tooling in the venv, for completeness:

```
pip==25.2
```

(`python -m venv` on 3.12 does not install `setuptools`/`wheel`, and nothing
here needs them.)

### Which pins are load-bearing

- **None of the three pins is load-bearing for correctness.** The estimator uses
  only `numpy.loadtxt`/`expm1`, `scipy.optimize.brentq`, and
  `mpmath.mp`/`mpmath.findroot`, all of which are long-stable APIs. The pins
  exist so that the reported digits are byte-reproducible, not because any
  specific version is required.
- `mpmath==1.3.0` is a *deliberate redundancy*, not a dependency of the answer.
  It is used only for the 50-decimal-digit independent re-solve that confirms
  the float64 answer is correct well beyond the 8 significant figures the task
  asks for. `mpmath` also ships as a `scipy` dependency-adjacent pure-Python
  package, so it adds no binary risk.
- `scipy` could be dropped entirely: `run_estimate.py` also solves the score
  equation with a stdlib-only bisection + Newton routine and asserts the three
  independent solvers agree to `<= 1e-12` relative. `scipy` is kept because
  `brentq` is the conventional, auditable choice for a bracketed root.

### Known-bad combinations hit

**Nothing version-related broke.** All three failures below are environmental,
and are recorded so the next person does not lose time on them:

- `python -m venv` can fail on this machine with
  `Error: [Errno 13] Permission denied: '...\.venv-verify\Scripts\python.exe'`
  while bootstrapping pip. This is a transient file lock (on-access antivirus
  scanning the freshly written `python.exe`), not a permissions problem. Fix:
  delete the half-built directory, wait a moment, and retry —
  `Remove-Item -Recurse -Force .venv-verify; Start-Sleep -Seconds 2; python -m venv .venv-verify`
  succeeded on the first retry. Confirm success with `$LASTEXITCODE` and
  `Test-Path .venv\Scripts\pip.exe` rather than by eye, because the failed
  attempt still leaves a working `python.exe` behind and looks half-alive.
- `cmd`/`bash`-style `&&` chaining fails in PowerShell 5.1 with
  `The token '&&' is not a valid statement separator in this version.`
  Use `;` between statements.
- PowerShell surfaces anything pip writes to stderr (its "A new release of pip
  is available" notice) as a red `NativeCommandError` block. That is cosmetic:
  check the exit code, not the presence of red text.

---

## 4. Reproducing the result

```powershell
Set-Location "C:\Users\micha\MAIStuff\randLab\geo&GenDL\gdlTests\claude_agent_workspace"
.\.venv\Scripts\python.exe task\solution\run_estimate.py
```

The script verifies the SHA-256 of `data/train.csv` against the value published
in `data/DOWNLOAD.md` before it estimates anything, then writes
`task/results/estimate.json`. It is deterministic: no RNG, no threading-sensitive
reductions, no seeds to record.

Optional validation step (`verify_model.py`, KS goodness of fit + 500-replicate
simulation recovery, seed `20260906`):

```powershell
.\.venv\Scripts\python.exe task\solution\verify_model.py
```

## 5. Verification history

- **2026-09-06** — `INSTALL.md` was followed from scratch into a *second*,
  clean virtualenv (`.venv-verify`, created with the same
  `C:\Python312\python.exe`) on Windows 10.0.26200, with
  `task/results/estimate.json` deleted first so nothing stale could be mistaken
  for a fresh result. `pip freeze` in that venv matched the pins exactly, and
  both `run_estimate.py` and `verify_model.py` ran end to end (exit 0),
  reproducing `lambda_hat = 1.9947851486676023` and every diagnostic digit
  identically to the `.venv` run. `.venv-verify` was then deleted. See the
  matching `STATUS.md` entry for the verbatim commands and full output.
