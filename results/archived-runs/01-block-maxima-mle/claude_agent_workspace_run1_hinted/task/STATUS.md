# STATUS.md — append-only work log

Task: estimate the rate λ of an Exponential(λ) from block maxima of `m = 4`
draws (`task/TASK.md`). Append only; never edit or delete an entry above.

---

## 2026-09-06 15:18 — initial state

**Command:**
```powershell
python -V; python -c "import numpy, scipy; print('numpy', numpy.__version__); print('scipy', scipy.__version__)"; git -C . rev-parse HEAD
```

**Exit status:** 0

**Key output:**
```
Python 3.12.5
numpy 2.2.1
scipy 1.15.0
1ff2b5353b57f9f440c4219ddd980f0cb047decb   (short: 1ff2b53)
```

**Result:** Starting point recorded, nothing changed.

- Repo: `C:\Users\micha\MAIStuff\randLab`, commit `1ff2b5353b57f9f440c4219ddd980f0cb047decb`.
  Working directory for all of this task's work is the subdirectory
  `geo&GenDL/gdlTests/claude_agent_workspace`, which git reports as entirely
  untracked (`?? geo&GenDL/gdlTests/claude_agent_workspace/`) — so the commit
  hash pins the surrounding repo, not this task's files.
- Machine: Windows 11 Pro, `Microsoft Windows NT 10.0.26200.0`, AMD64. CPU only.
- Contents at start: `data/DOWNLOAD.md`, `data/train.csv`, `task/*.md`, and empty
  `task/results/` + `task/solution/` (each holding only `.gitkeep`).
  `STATUS.md` was the placeholder line `THE AGENT LOG GOES HERE.`, `INSTALL.md`
  the placeholder `YOUR INSTALLATION GUIDE GOES HERE`.
- `data/holdout.csv` is **not** present in this workspace, so the prohibition on
  reading it is trivially satisfied. `tests/` and `gt_code/` are likewise absent
  and were never accessed.

**Next:** verify the data hash, then derive and implement the estimator.

---

## 2026-09-06 15:19 — verify data integrity

**Command:**
```powershell
(Get-FileHash -Algorithm SHA256 data\train.csv).Hash.ToLower(); Get-Content data\train.csv -TotalCount 5; (Get-Content data\train.csv | Measure-Object -Line).Lines
```

**Exit status:** 0

**Key output:**
```
7fc1dade9370cab98f675808ed0c7b9cea2b17bf974ec4d1860435db26c74587
y
1.366774
1.020233
0.806559
1.823957
2001
```

**Result:** SHA-256 matches the value published in `data/DOWNLOAD.md` exactly, so
this is the intended `train.csv`. 2001 lines = 1 header + 2000 observations,
single column `y`, as documented. No state changed.

Numbers to keep: **expected train.csv sha256
`7fc1dade9370cab98f675808ed0c7b9cea2b17bf974ec4d1860435db26c74587`**, n = 2000,
m = 4. If the hash ever stops matching, every result below is invalid.

**Next:** write the solution.

---

## 2026-09-06 15:23 — write solution files

**Command:** no shell command; created three files with the editor:
`task/solution/estimate_lambda.py`, `task/solution/selfcheck.py`,
`task/solution/README.md`.

**Result:** Derivation and implementation in place.

The derivation actually used (full version in `task/solution/README.md`): with
`X_j ~ Exp(λ)` and `Y = max_j X_j` over `m` draws,
`F_Y(t) = (1 - e^{-λt})^m`, hence
`f_Y(t) = m λ e^{-λt} (1 - e^{-λt})^{m-1}` and

```
loglik(λ) = n log m + n log λ − λ Σ y_i + (m−1) Σ log(1 − e^{−λ y_i})
score(λ)  = n/λ − Σ y_i + (m−1) Σ y_i e^{−λ y_i} / (1 − e^{−λ y_i})
```

`score` is strictly decreasing, `+∞` at `0+` and `−Σ y_i` at `∞`, so the root is
unique. Solved with `scipy.optimize.brentq` (`xtol=1e-14`, `rtol=1e-15`) on a
bracket grown outward from the method-of-moments start `H_m / mean(y)`
(`H_4 = 25/12 ≈ 2.0833`, since `E[Y] = H_m/λ`), then cross-checked against
`minimize_scalar` on `−loglik`; the script raises if the two disagree by more
than `rel_tol=1e-6`.

Implementation note worth keeping: the first draft of `score` used
`y / np.expm1(λ y)` and `observed_information` used
`y² e^{λy} / (e^{λy} − 1)²`. Both are algebraically right but overflow to
`inf` (and then `inf/inf = nan`) for large `λ y`, which the bracket-doubling loop
can reach. Rewrote both in the negative-exponent form
`y e^{−u}/(1 − e^{−u})` and `y² e^{−u}/(1 − e^{−u})²` with `expm1(−u)` in the
denominator, which cannot overflow and stays accurate as `u → 0`. Caught by
inspection before any run, so no failing output was produced.

**Next:** build a clean environment and run it.

---

## 2026-09-06 15:24 — create venv, install numpy + scipy

**Command:**
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade "pip==24.3.1"
.\.venv\Scripts\python.exe -m pip install "numpy==2.2.1" "scipy==1.15.0"
```

**Exit status:** 0, 0, 0

**Key output:**
```
Python 3.12.5
Successfully installed pip-24.3.1
Successfully installed numpy-2.2.1 scipy-1.15.0
```

**Result:** `.venv/` created at the workspace root and populated. Versions chosen
to match what the system interpreter already had, so the pins are the
as-run versions, not an upgrade. `INSTALL.md` written at this point (replacing
its placeholder) with platform, the three commands verbatim, and the `pip freeze`
output.

Non-failure worth recording so it is not re-debugged: both `pip install`
invocations print pip's `[notice] A new release of pip is available: 24.3.1 ->
26.2.1` to stderr, and PowerShell renders that as

```
.\.venv\Scripts\python.exe : 
    + CategoryInfo          : NotSpecified: (:String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
```

This is **not** an error — exit status is 0 and the installs succeeded. Check the
exit code, not the red text.

**Next:** run the estimator.

---

## 2026-09-06 15:25 — run estimator, write estimate.json

**Command:**
```powershell
.\.venv\Scripts\python.exe task\solution\estimate_lambda.py
```

**Exit status:** 0

**Key output:**
```
sha256               7fc1dade9370cab98f675808ed0c7b9cea2b17bf974ec4d1860435db26c74587
n                    2000
m                    4
mean(y)              1.043136
moment start         1.9971827934
lambda_hat (MLE)     1.9947851487
  cross-check (opt)  1.9947851450
  score at root      -9.095e-13
  loglik at root     -1493.421602
  approx std err     0.025278  (95% CI 1.9452, 2.0443)
naive 1/mean(y)      0.9586477409   <- wrong model, not reported
```

**Result:** Wrote `task/results/estimate.json`:
```json
{"lambda_hat": 1.9947851486676023}
```

Numbers to keep: **λ̂ = 1.9947851486676023**, analytic s.e. 0.025278,
loglik −1493.421602, score at root −9.1e−13. The two independent solvers agree
to 8 significant figures. The naive plain-exponential MLE `1/mean(y) = 0.95865`
is 2.081× smaller — consistent with the `H_4 = 2.0833` bias factor and far
outside the ±0.05 scale of the correct estimate's confidence interval, which is
the discrimination `tests/test_estimator.py` is described as making.

**Next:** validate the numerics against known ground truth.

---

## 2026-09-06 15:26 — run self-checks

**Command:**
```powershell
.\.venv\Scripts\python.exe task\solution\selfcheck.py
```

**Exit status:** 0

**Key output:**
```
[PASS] score matches finite difference: max rel err 1.39e-08
[PASS] m=1 equals 1/mean(y): 1.6671890079 vs 1.6671890079
[PASS] recovers lambda=1.9 at n=2000: est 1.940283, |err| 0.040283 < tol 0.134164
[PASS] recovers lambda=1.9 at n=200000: est 1.897926, |err| 0.002074 < tol 0.013416
[PASS] naive 1/mean(y) low by factor H_m: lam_true / naive = 2.0823, H_4 = 2.0833
[PASS] split-half estimates agree within 4 s.e.: 1.976637 vs 2.013212 (full 1.994785, s.e. 0.025278)
[PASS] bootstrap s.d. matches observed information: bootstrap sd 0.023520 vs analytic s.e. 0.025278
lambda_hat on train.csv: 1.9947851487 (s.e. 0.025278)
bootstrap 95% percentile CI: (1.951152, 2.034496)
all checks passed
```

**Result:** No files written; validation only. The important one is the
consistency check: with simulated data at a known `λ = 1.9`, the estimator
returns 1.897926 at n = 200000, i.e. the derivation is right and not merely
self-consistent.

Determinism: `selfcheck.py` seeds `numpy.random.default_rng(20260906)`; the
simulated figures above are reproducible under numpy 2.2.1 and may shift
slightly on another numpy version (Generator stream guarantees are
version-scoped). All checks are tolerance-based, so they still pass; the reported
`lambda_hat` uses no RNG and is unaffected.

**Next:** verify `INSTALL.md` from a genuinely clean environment.

---

## 2026-09-06 15:27 — verify INSTALL.md in a throwaway venv

**Command:**
```powershell
python -m venv .venv-verify
.\.venv-verify\Scripts\python.exe -m pip install --upgrade "pip==24.3.1"
.\.venv-verify\Scripts\python.exe -m pip install "numpy==2.2.1" "scipy==1.15.0"
.\.venv-verify\Scripts\python.exe -m pip freeze
.\.venv-verify\Scripts\python.exe task\solution\estimate_lambda.py --out task\results\_verify.json
.\.venv-verify\Scripts\python.exe task\solution\selfcheck.py
(Get-FileHash task\results\estimate.json).Hash; (Get-FileHash task\results\_verify.json).Hash
```

**Exit status:** 0 throughout

**Key output:**
```
Successfully uninstalled pip-24.2
Successfully installed pip-24.3.1
Successfully installed numpy-2.2.1 scipy-1.15.0
numpy==2.2.1
scipy==1.15.0
lambda_hat (MLE)     1.9947851487
all checks passed
A8A4FACDDF7B36568E4A0674195FC646EA97BB67F828CF421C8E6B69B7D12305
A8A4FACDDF7B36568E4A0674195FC646EA97BB67F828CF421C8E6B69B7D12305
```

**Result:** `INSTALL.md` followed verbatim from a fresh virtualenv reproduces the
result exactly — the two SHA-256 hashes above are `estimate.json` (from `.venv`)
and `_verify.json` (from `.venv-verify`), and they are identical, so the output
is byte-for-byte reproducible. All self-checks passed identically, including the
simulated figures. `INSTALL.md` is therefore a followed guide, not a guess; its
"Verification status" section records this run.

Detail: the fresh venv bootstrapped `pip 24.2`, which the pin then replaced with
24.3.1 — confirming the pin command does real work on a clean machine rather
than silently no-opping as it did in `.venv`.

**Next:** clean up the throwaway environment.

---

## 2026-09-06 15:28 — clean up verification artifacts

**Command:**
```powershell
Remove-Item -Recurse -Force .venv-verify; Remove-Item -Force task\results\_verify.json
```

**Exit status:** 0

**Result:** Deleted `.venv-verify/` and the temporary `task/results/_verify.json`
so `task/results/` contains only the graded `estimate.json` (plus `.gitkeep`).
`.venv/` is kept as the working environment and is git-ignored via a new
`.gitignore` at the workspace root.

**Invalidates:** nothing. `estimate.json` was produced by `.venv` and was not
touched; its hash is the `A8A4FACD...` recorded in the previous entry and can be
re-checked at any time.

**Next:** nothing outstanding. Final state below.

---

## 2026-09-06 15:29 — final state

**Deliverables**

| Path | Contents |
|---|---|
| `task/results/estimate.json` | `{"lambda_hat": 1.9947851486676023}` — finite, strictly positive, exactly the required shape |
| `task/solution/estimate_lambda.py` | entry point; reproduces the estimate |
| `task/solution/selfcheck.py` | validation harness, exits non-zero on failure |
| `task/solution/README.md` | full derivation |
| `INSTALL.md` | environment, verified 2026-09-06 |

**Reproduce from scratch** (three install commands from `INSTALL.md`, then):
```powershell
.\.venv\Scripts\python.exe task\solution\estimate_lambda.py
.\.venv\Scripts\python.exe task\solution\selfcheck.py
```

**Prohibitions observed:** `data/holdout.csv` was never read (it is not even
present in this workspace); `tests/` and `gt_code/` were never read (also absent);
no internet search was performed at any point — the likelihood was derived from
the order-statistic CDF, as the entries above record.
