# STATUS.md — append-only work log

Task: estimate the rate λ of an Exponential distribution observed only through
block maxima of `m = 4` i.i.d. draws, from `data/train.csv`. See `task/TASK.md`.

Append only. Entries are never edited or deleted; corrections are added as new
entries further down.

---

## 2026-09-06 17:48 — initial state

**Machine:** Windows `Microsoft Windows NT 10.0.26200.0`, `AMD64`, PowerShell 5.1.
System Python `C:\Python312\python.exe` → `Python 3.12.5`.

**Repo:** enclosing git repository is `C:/Users/micha/MAIStuff/randLab` at commit
`1ff2b5353b57f9f440c4219ddd980f0cb047decb`. The project directory itself
(`geo&GenDL/gdlTests/claude_agent_workspace`) is **untracked** in that repo
(`git status --porcelain` reports `?? geo&GenDL/`), so the commit hash above
identifies the surrounding checkout, not this project's contents.

**Files present at start (6):**

```
task/TASK.md
task/STATUS.md              (stub: "THE AGENT LOG GOES HERE.")
task/INSTALL.md             (stub: "YOUR INSTALLATION GUIDE GOES HERE")
task/PROJECT_INSTRUCTION.md
data/DOWNLOAD.md
data/train.csv
```

`data/holdout.csv` is **not** present; per `task/TASK.md` it must not be read
even if it appears. Nothing under `tests/` or `gt_code/` exists here either, and
both are likewise off limits.

**Result:** No environment yet, no solution code, no results.

**Next:** verify the data hash, then build the environment.

---

## 2026-09-06 17:50 — verify interpreter and `data/train.csv` integrity

**Command:**
```powershell
Set-Location "C:\Users\micha\MAIStuff\randLab\geo&GenDL\gdlTests\claude_agent_workspace"; python -V; python -c "import sys; print(sys.executable)"; (Get-FileHash -Algorithm SHA256 'data\train.csv').Hash.ToLower()
```

**Exit status:** 0

**Key output:**
```
Python 3.12.5
C:\Python312\python.exe
7fc1dade9370cab98f675808ed0c7b9cea2b17bf974ec4d1860435db26c74587
```

**Result:** SHA-256 matches the value published in `data/DOWNLOAD.md`
(`7fc1dade9370cab98f675808ed0c7b9cea2b17bf974ec4d1860435db26c74587`) exactly, so
this is the intended `train.csv`. Every number below is conditional on that
hash; if it ever changes, all of them are invalid.

**Next:** create a virtualenv.

---

## 2026-09-06 17:52 — create virtualenv, pin pip

**Command:**
```powershell
Set-Location "C:\Users\micha\MAIStuff\randLab\geo&GenDL\gdlTests\claude_agent_workspace"; python -m venv .venv; .\.venv\Scripts\python.exe -V; .\.venv\Scripts\python.exe -m pip install --upgrade "pip==25.2"
```

**Exit status:** 0

**Key output:**
```
Python 3.12.5
Successfully installed pip-25.2
[notice] A new release of pip is available: 25.2 -> 26.2.1
```

**Result:** Created `$REPO\.venv` (Python 3.12.5) with `pip==25.2`. Created
`task/INSTALL.md` as a stub declaring the base environment at this point.

**Note on a non-failure that looks like one:** PowerShell renders pip's stderr
notice as a red `NativeCommandError` block. The exit code was 0; the install
succeeded. Judge these commands by exit code, not by red text.

**Note on a real failure earlier in the session:** `bash`-style chaining does not
work in this shell.
```
At ...ps1:7 char:80
+ ... claude_agent_workspace" && ls -la  ...
+                                ~~
The token '&&' is not a valid statement separator in this version.
```
Use `;` as the statement separator in PowerShell 5.1.

**Next:** install numpy/scipy/mpmath from a pinned requirements file.

---

## 2026-09-06 17:54 — install pinned dependencies

**Command:**
```powershell
Set-Location "C:\Users\micha\MAIStuff\randLab\geo&GenDL\gdlTests\claude_agent_workspace"; .\.venv\Scripts\python.exe -m pip install -r task\solution\requirements.txt
```

with `task/solution/requirements.txt`:
```
numpy==2.5.3
scipy==1.18.1
mpmath==1.3.0
```

**Exit status:** 0

**Key output:**
```
Successfully installed mpmath-1.3.0 numpy-2.5.3 scipy-1.18.1
```

Versions were chosen as the latest available on PyPI on 2026-09-06
(`pip index versions numpy` → `2.5.3`; `pip index versions scipy` → `1.18.1`)
and then frozen. All three came from prebuilt wheels; no compiler was invoked.

**Result:** Environment complete. Wrote `task/solution/requirements.txt` and
filled in `task/INSTALL.md` with the platform table, verbatim install commands,
the `pip freeze` output, and which pins are load-bearing (answer: none of them
are required for correctness — `mpmath` is deliberate redundancy and `scipy` is
replaceable by the stdlib fallback in the entry point).

**Next:** derive the likelihood and implement the estimator.

---

## 2026-09-06 18:05 — derivation written down (no state change, recorded for the record)

No command run; this entry exists so the numbers below are interpretable.

`Y = max(X_1..X_m)`, `X_j ~ Exp(λ)` i.i.d., so
`F(y) = (1 - exp(-λy))^m` and

```
f(y;λ)  = m λ exp(-λy) (1 - exp(-λy))^(m-1)
l(λ)    = n log m + n log λ - λ Σ y_i + (m-1) Σ log(1 - exp(-λ y_i))
l'(λ)   = n/λ - Σ y_i + (m-1) Σ y_i / (exp(λ y_i) - 1)
l''(λ)  = -n/λ² - (m-1) Σ y_i² exp(-λ y_i) / (1 - exp(-λ y_i))²   < 0  for all λ>0
```

`l''< 0` everywhere ⇒ strictly concave ⇒ the stationary point is the unique
global max; `l'(0+) = +∞` and `l'(∞) = -Σ y_i < 0` ⇒ exactly one sign change, so
a bracketing root-finder cannot land on the wrong point. No closed form for
`m > 1`. Start value from moments: `E[Y] = H_m/λ`, so `λ ≈ H_m/ȳ` with
`H_4 = 25/12 = 2.0833333333`.

Written up in full in `task/solution/README.md` and the module docstring of
`task/solution/blockmax_exp_mle.py`.

**Result:** Wrote `task/solution/blockmax_exp_mle.py` (model + three solvers),
`task/solution/run_estimate.py` (entry point), `task/solution/README.md`.

**Next:** run the entry point.

---

## 2026-09-06 18:08 — first run of the estimator

**Command:**
```powershell
Set-Location "C:\Users\micha\MAIStuff\randLab\geo&GenDL\gdlTests\claude_agent_workspace"; .\.venv\Scripts\python.exe task\solution\run_estimate.py
```

**Exit status:** 0

**Key output:**
```
sha256 actual   : 7fc1dade9370cab98f675808ed0c7b9cea2b17bf974ec4d1860435db26c74587  (match)
n = 2000   min/max 0.045721 / 5.399904   mean 1.0431360315
scipy brentq    : 1.9947851486676023
stdlib bis+newt : 1.994785148667602
mpmath (50 dps) : 1.99478514866760195042823544293
|brentq - stdlib| / stdlib = 1.113e-16
|brentq - mpmath| / mpmath = 1.113e-16
score l'(lam)   : -9.094947e-13
curvature l''   : -1.565005e+03
logL(lam_hat)   : -1493.4216022051
asymptotic SE   : 0.02527797
95% Wald CI     : [1.94524124, 2.04432906]
moments H_m/ybar: 1.9971827934440716   (rel. gap to MLE 1.202e-03)
naive 1/ybar    : 0.9586477408531543   (rel. gap to MLE 5.194e-01)
```

**Result:** Wrote `task/results/estimate.json` = `{"lambda_hat": 1.9947851486676023}`.

The three solvers agree to `1.113e-16` relative — one ulp of a float64, i.e.
the reported double is the correctly rounded value of the exact root
`1.99478514866760195042823544293…`. The task tolerance is `3e-4` relative, so
there are ~12 orders of magnitude of headroom; all 17 printed digits are real.

Sanity numbers worth keeping: the method-of-moments estimator `H_4/ȳ` sits
`1.2e-03` away from the MLE (right order of magnitude for `n = 2000`,
`SE/λ ≈ 1.3e-2`), while the naive `1/ȳ = 0.9587` — the estimator you get by
wrongly treating `y` as Exponential — is off by 52%. That 52% gap is the whole
point of the task, and `run_estimate.py` now asserts the two are *not* close as
a guard against silently reverting to the wrong model.

**Invalidates:** nothing (first result).

**Next:** validate the *derivation*, not just the arithmetic — three solvers on
one wrong score function would agree just as happily.

---

## 2026-09-06 18:14 — independent model validation (goodness of fit + simulation recovery)

**Command:**
```powershell
Set-Location "C:\Users\micha\MAIStuff\randLab\geo&GenDL\gdlTests\claude_agent_workspace"; .\.venv\Scripts\python.exe task\solution\verify_model.py
```

**Exit status:** 0

**Key output:**
```
lam_hat from estimate.json : 1.9947851486676023
n = 2000, m = 4, seed = 20260906, replicates = 500

[A] Kolmogorov-Smirnov goodness of fit on data/train.csv
    max-of-4 Exponential(lam_hat) : D = 0.014603, p = 0.781464
    plain Exponential(1/ybar)     : D = 0.237484, p = 1.004e-99

[B] recovery of lam_0 = lam_hat = 1.9947851487 from simulated data
    correct MLE   mean 1.9945499876  sd 0.02486364  bias -2.35e-04 = -0.21 MC SE
    naive 1/ybar  mean 0.9573670150  bias -1.0374
    simulated sd 0.02486364  vs  observed-information SE 0.02527797  (ratio 0.9836)
```

**Determinism:** `numpy.random.default_rng(20260906)`, 500 replicates of
`n = 2000 × m = 4` draws. Changing the seed, the replicate count, or the numpy
version will move the third decimal of the simulated mean/sd but not the
conclusions.

**Result:** Wrote `task/solution/verify_model.py`. The derivation holds up from
outside the algebra:

- The fitted `(1 - e^{-λ̂y})^4` CDF is **not** rejected by KS (`p = 0.78`), while
  the plain Exponential fit is destroyed (`p = 1.0e-99`). So `m = 4` block
  maxima is the right family for this file.
- Re-estimating from data simulated at `λ_0 = λ̂` recovers `λ_0` with a bias of
  `-0.21` Monte-Carlo standard errors — statistically indistinguishable from
  zero. A misderived likelihood surfaces here as a bias many MC SE wide; for
  scale, the naive estimator on the *same* samples is biased by `-1.0374`.
- The simulated spread of the MLE (`0.02486364`) matches the
  observed-information SE at the reported point (`0.02527797`) to 1.7%, so the
  curvature formula `l''` is consistent with the sampling distribution too.

Taken together: `λ̂ = 1.99479 ± 0.02528`, 95% Wald `[1.94524, 2.04433]`, which
contains `2.0` — plausibly the round number the data was generated with, though
nothing in the pipeline assumes that.

**Invalidates:** nothing.

**Next:** verify `INSTALL.md` from a clean virtualenv.

---

## 2026-09-06 18:22 — FAILED: clean-venv creation hit a transient permission error

**Command:**
```powershell
Set-Location "C:\Users\micha\MAIStuff\randLab\geo&GenDL\gdlTests\claude_agent_workspace"; C:\Python312\python.exe -m venv .venv-verify; .\.venv-verify\Scripts\python.exe -V; .\.venv-verify\Scripts\python.exe -m pip install --upgrade "pip==25.2" 2>&1 | Select-String "Successfully"; .\.venv-verify\Scripts\python.exe -m pip install -r task\solution\requirements.txt 2>&1 | Select-String "Successfully"; .\.venv-verify\Scripts\python.exe -m pip freeze
```

**Exit status:** 0 (misleading — see below)

**Key output (verbatim error text):**
```
Error: [Errno 13] Permission denied: 'C:\Users\micha\MAIStuff\randLab\geo&GenDL\gdlTests\claude_agent_workspace\.venv-verify\Scripts\python.exe'
Python 3.12.5
--- freeze ---
mpmath==1.3.0
numpy==2.5.3
scipy==1.18.1
```

**Result:** `python -m venv` failed while bootstrapping pip — a transient file
lock, almost certainly on-access antivirus scanning the just-written
`python.exe`. Note the trap: the venv was left half-built with a *working*
`python.exe`, neither `Successfully` line was printed, yet the pipeline's
overall `$LASTEXITCODE` was 0 and `pip freeze` still listed the right versions,
so at a glance this looks like a success. It was not a clean install and does
not count as a verification.

**Lesson recorded in `INSTALL.md` §"Known-bad combinations hit":** check
`$LASTEXITCODE` of the `venv` step itself and `Test-Path
.venv\Scripts\pip.exe`, do not judge by eye.

**Next:** delete the half-built venv and retry.

---

## 2026-09-06 18:25 — clean-venv creation, retry (succeeded)

**Command:**
```powershell
Set-Location "C:\Users\micha\MAIStuff\randLab\geo&GenDL\gdlTests\claude_agent_workspace"; Remove-Item -Recurse -Force .venv-verify -ErrorAction SilentlyContinue; Start-Sleep -Seconds 2; C:\Python312\python.exe -m venv .venv-verify 2>&1; echo "=== venv exit: $LASTEXITCODE ==="; Test-Path .venv-verify\Scripts\python.exe; Test-Path .venv-verify\Scripts\pip.exe
```

**Exit status:** 0

**Key output:**
```
=== venv exit: 0 ===
True
True
```

**Result:** Clean `.venv-verify` built on the first retry, no error output. The
earlier failure was transient, not reproducible.

**Next:** install the pinned dependencies into it.

---

## 2026-09-06 18:27 — install dependencies into the clean venv

**Command:**
```powershell
Set-Location "C:\Users\micha\MAIStuff\randLab\geo&GenDL\gdlTests\claude_agent_workspace"; .\.venv-verify\Scripts\python.exe -V; .\.venv-verify\Scripts\python.exe -m pip install --upgrade "pip==25.2" 2>&1 | Select-String "Successfully|ERROR"; echo "=== pip exit: $LASTEXITCODE ==="; .\.venv-verify\Scripts\python.exe -m pip install -r task\solution\requirements.txt 2>&1 | Select-String "Successfully|ERROR"; echo "=== deps exit: $LASTEXITCODE ==="; .\.venv-verify\Scripts\python.exe -m pip freeze
```

**Exit status:** 0

**Key output:**
```
Python 3.12.5
      Successfully uninstalled pip-24.2
Successfully installed pip-25.2
=== pip exit: 0 ===
Successfully installed mpmath-1.3.0 numpy-2.5.3 scipy-1.18.1
=== deps exit: 0 ===
mpmath==1.3.0
numpy==2.5.3
scipy==1.18.1
```

**Result:** `pip freeze` in the clean venv matches the pins in
`task/solution/requirements.txt` exactly. Recorded: `python -m venv` on 3.12.5
ships `pip 24.2`, which the documented step upgrades to `25.2`.

**Next:** run the whole pipeline in this venv, with `estimate.json` deleted
first so a stale file cannot masquerade as a fresh result.

---

## 2026-09-06 18:30 — VERIFIED `INSTALL.md`: full pipeline in the clean venv

**Command:**
```powershell
Set-Location "C:\Users\micha\MAIStuff\randLab\geo&GenDL\gdlTests\claude_agent_workspace"; Remove-Item -Force task\results\estimate.json; Test-Path task\results\estimate.json; .\.venv-verify\Scripts\python.exe task\solution\run_estimate.py; echo "=== run exit: $LASTEXITCODE ==="; Get-Content task\results\estimate.json; .\.venv-verify\Scripts\python.exe task\solution\verify_model.py; echo "=== verify exit: $LASTEXITCODE ==="
```

**Exit status:** 0

**Key output:**
```
False                                   <- estimate.json really was gone first
scipy brentq    : 1.9947851486676023
stdlib bis+newt : 1.994785148667602
mpmath (50 dps) : 1.99478514866760195042823544293
score l'(lam)   : -9.094947e-13    curvature l'' : -1.565005e+03
logL(lam_hat)   : -1493.4216022051    asymptotic SE : 0.02527797
=== run exit: 0 ===
{
  "lambda_hat": 1.9947851486676023
}
[A] max-of-4 Exponential(lam_hat) : D = 0.014603, p = 0.781464
    plain Exponential(1/ybar)     : D = 0.237484, p = 1.004e-99
[B] correct MLE   mean 1.9945499876  sd 0.02486364  bias -2.35e-04 = -0.21 MC SE
    simulated sd 0.02486364  vs  observed-information SE 0.02527797 (ratio 0.9836)
=== verify exit: 0 ===
```

**Result:** `INSTALL.md` is verified end to end against a clean virtualenv on
Windows 10.0.26200 / Python 3.12.5. Every printed digit — `lambda_hat`, the
score, the curvature, the log-likelihood, the KS statistics, and the simulated
mean and sd — is identical to the `.venv` run, so the pipeline is reproducible
and the simulation is seed-deterministic across environments as claimed.

**Verified against:** `.venv-verify`, built only from the commands in
`INSTALL.md` §2, no leftovers from `.venv`.

**Next:** delete `.venv-verify`, then final review of deliverables.

---

## 2026-09-06 18:33 — remove the verification venv

**Command:**
```powershell
Set-Location "C:\Users\micha\MAIStuff\randLab\geo&GenDL\gdlTests\claude_agent_workspace"; Remove-Item -Recurse -Force .venv-verify; Test-Path .venv-verify
```

**Exit status:** 0

**Key output:**
```
False
```

**Result:** `.venv-verify` deleted. `$REPO\.venv` remains as the working
environment. This deletes the *evidence* but not the *record*: the entry above
holds the commands and output, and the check is re-runnable from `INSTALL.md`
§2 at any time.

**Invalidates:** nothing.

---

## 2026-09-06 18:35 — final state

**Deliverables:**

| Path | Contents |
|---|---|
| `task/results/estimate.json` | `{"lambda_hat": 1.9947851486676023}` |
| `task/solution/run_estimate.py` | entry point (deterministic, no RNG) |
| `task/solution/blockmax_exp_mle.py` | log-likelihood, score, curvature, three solvers |
| `task/solution/verify_model.py` | KS goodness of fit + simulation recovery, seed `20260906` |
| `task/solution/requirements.txt` | `numpy==2.5.3`, `scipy==1.18.1`, `mpmath==1.3.0` |
| `task/solution/README.md` | full derivation and results write-up |
| `task/INSTALL.md` | environment rebuild guide, verified 2026-09-06 |

**The number, for the record:**

```
lambda_hat (float64, reported) : 1.9947851486676023
exact root  (50-digit mpmath)  : 1.99478514866760195042823544293
asymptotic SE                  : 0.02527797
95% Wald CI                    : [1.94524124, 2.04432906]
sha256 data/train.csv          : 7fc1dade9370cab98f675808ed0c7b9cea2b17bf974ec4d1860435db26c74587
```

The reported double is the correctly rounded exact root (agreement `1.1e-16`,
one ulp), against a grading tolerance of `3e-4` relative.

**Prohibitions honoured:** `data/holdout.csv` was never read (it is not even
present in this checkout); nothing under `tests/` or `gt_code/` was read (neither
exists here); no web search was performed at any point in this session — the
derivation is the standard order-statistic argument, written out in
`task/solution/README.md`.

**Reproduce from scratch:** follow `task/INSTALL.md` §2, then §4.
