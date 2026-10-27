# STATUS.md — append-only work log

## 2026-09-06 17:49 — initial state

**Command:**
```powershell
python -V; python -c "import platform,sys; print(platform.platform()); print(platform.machine()); print(sys.executable)"; Get-Location; git -C "C:\Users\micha\MAIStuff\randLab" rev-parse --short HEAD; Get-Date -Format "yyyy-MM-dd HH:mm:ss"; Get-FileHash -Algorithm SHA256 "C:\Users\micha\MAIStuff\randLab\geo&GenDL\gdlTests\grok_agent_workspace\data\train.csv"
```

**Exit status:** 0

**Key output:**
```
Python 3.12.5
Windows-11-10.0.26200-SP0
AMD64
C:\Python312\python.exe
cwd: C:\Users\micha\MAIStuff\randLab\geo&GenDL\gdlTests\grok_agent_workspace
git HEAD: 1ff2b53
date: 2026-09-06 17:49:10
train.csv sha256: 7FC1DADE9370CAB98F675808ED0C7B9CEA2B17BF974EC4D1860435DB26C74587
```

**Result:** Workspace is a fresh task checkout. `data/train.csv` hash matches `data/DOWNLOAD.md`. No venv, no `task/solution/`, no `task/results/` yet. `task/STATUS.md` and `task/INSTALL.md` are stubs.

**Invalidates:** nothing — first recorded state.

**Next:** create a project venv, pin numpy/scipy, then implement the block-maxima MLE.

## 2026-09-06 17:49 — create venv and install numpy/scipy

**Command:**
```powershell
python -m venv .venv; .\.venv\Scripts\python.exe -m pip install --upgrade pip; .\.venv\Scripts\python.exe -m pip install numpy scipy
```

**Exit status:** 0

**Key output:**
```
Successfully installed pip-26.2.1
Successfully installed numpy-2.5.3 scipy-1.18.1
pip freeze:
numpy==2.5.3
scipy==1.18.1
```

**Result:** Project venv at `.venv`. Runtime is Python 3.12.5 + numpy 2.5.3 + scipy 1.18.1. No GPU packages.

**Invalidates:** nothing yet — no estimates computed.

**Next:** implement the MLE entry point and write `task/results/estimate.json`.

## 2026-09-06 17:51 — compute train-set MLE

**Command:**
```powershell
.\.venv\Scripts\python.exe task\solution\estimate_lambda.py
```

**Exit status:** 0

**Key output:**
```
n=2000
mean(y)=1.043136031500
lambda_mom=1.997182793444
lambda_hat=1.9947851486676018
score=0.000000e+00
loglik=-1493.421602205054
wrote C:\Users\micha\MAIStuff\randLab\geo&GenDL\gdlTests\grok_agent_workspace\task\results\estimate.json
```

**Result:** Wrote `{"lambda_hat": 1.9947851486676018}` to `task/results/estimate.json`. Score of the block-maxima log-likelihood is 0 at this root. Brentq and Newton agreed to 1e-12 relative. Observed Hessian ≈ -1565.0.

**Invalidates:** any earlier unwritten estimate (none). Downstream grading should use this file.

**Next:** finite-difference / synthetic-recovery checks, then clean-venv INSTALL verification.

## 2026-09-06 17:51 — score / maximum / synthetic checks

**Command:**
```powershell
.\.venv\Scripts\python.exe -c @"
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, r'task\solution')
from estimate_lambda import load_y, score, score_prime, log_likelihood, mle_lambda, TRAIN_PATH

y = load_y(TRAIN_PATH)
lam = mle_lambda(y)
eps = 1e-8
fd = (log_likelihood(lam+eps, y) - log_likelihood(lam-eps, y)) / (2*eps)
print('lam', f'{lam:.16f}')
print('score', score(lam, y))
print('fd_score', fd)
print('rel_fd_err', abs(fd - score(lam, y)))
fd2 = (score(lam+eps, y) - score(lam-eps, y)) / (2*eps)
print('score_prime', score_prime(lam, y))
print('fd2', fd2)
for d in (-1e-3, -1e-4, 0.0, 1e-4, 1e-3):
    print(f'll({lam+d:.10f})', f'{log_likelihood(lam+d, y):.12f}')
print('n', y.size, 'min', y.min(), 'max', y.max(), 'mean', y.mean())
"@
```

**Exit status:** 0

**Key output:**
```
lam 1.9947851486676018
score 0.0
fd_score -4.547473508864641e-05
rel_fd_err 4.547473508864641e-05
score_prime -1565.0047639874915
fd2 -1565.0047316739801
ll(1.9937851487) -1493.422385018096
ll(1.9946851487) -1493.421610030389
ll(1.9947851487) -1493.421602205054
ll(1.9948851487) -1493.421610029767
ll(1.9957851487) -1493.422384397024
n 2000 min 0.045721 max 5.399904 mean 1.0431360315000002
```

**Result:** Hessian finite-difference matches analytic ℓ''. Log-likelihood is higher at `lambda_hat` than at ±1e-4 and ±1e-3 neighbors. Train y range [0.045721, 5.399904].

**Invalidates:** nothing.

**Next:** synthetic recovery, then INSTALL.md clean-venv verification.

## 2026-09-06 17:52 — synthetic recovery of known λ=2

**Command:**
```powershell
.\.venv\Scripts\python.exe -c @"
import numpy as np
import sys
sys.path.insert(0, r'task\solution')
from estimate_lambda import mle_lambda, score

rng = np.random.default_rng(20260906)
true_lam = 2.0
m, n = 4, 2000
x = rng.exponential(1.0/true_lam, size=(n, m))
y = x.max(axis=1)
hat = mle_lambda(y)
print('synthetic_true', true_lam)
print('synthetic_hat', f'{hat:.16f}')
print('rel_err', abs(hat-true_lam)/true_lam)
print('score', score(hat, y))
"@
```

**Exit status:** 0

**Key output:**
```
synthetic_true 2.0
synthetic_hat 1.9610850005460938
rel_err 0.019457499726953098
score 0.0
```

**Result:** Estimator recovers a nearby value on a fresh n=2000 sample (error ~1.9%, consistent with SE ≈ 1/sqrt(1565) ≈ 0.025). This does not change the train-set MLE.

**Invalidates:** nothing.

**Next:** larger-n consistency check.

## 2026-09-06 17:53 — large-n consistency

**Command:**
```powershell
.\.venv\Scripts\python.exe -c @"
import numpy as np
import sys
sys.path.insert(0, r'task\solution')
from estimate_lambda import mle_lambda
rng = np.random.default_rng(7)
true_lam = 2.0
m, n = 4, 50000
x = rng.exponential(1.0/true_lam, size=(n, m))
y = x.max(axis=1)
hat = mle_lambda(y)
print('n', n, 'hat', f'{hat:.12f}', 'rel_err', abs(hat-true_lam)/true_lam)
"@
```

**Exit status:** 0

**Key output:**
```
n 50000 hat 2.002239754342 rel_err 0.0011198771710292377
```

**Result:** On n=50000 the same likelihood recovers λ=2 to 0.11% relative error. Likelihood implementation is consistent.

**Invalidates:** nothing.

**Next:** follow INSTALL.md in a fresh venv.

## 2026-09-06 17:53 — verify INSTALL.md in a clean venv

**Command:**
```powershell
python -m venv .venv_verify; .\.venv_verify\Scripts\python.exe -m pip install --upgrade pip; .\.venv_verify\Scripts\python.exe -m pip install numpy==2.5.3 scipy==1.18.1
```

**Exit status:** 0

**Key output:**
```
Successfully installed pip-26.2.1
Successfully installed numpy-2.5.3 scipy-1.18.1
```

**Result:** Fresh environment built from the pinned commands in INSTALL.md.

**Invalidates:** nothing.

**Next:** re-run the entry point inside `.venv_verify`.

## 2026-09-06 17:53 — reproduce estimate in verified venv

**Command:**
```powershell
.\.venv_verify\Scripts\python.exe task\solution\estimate_lambda.py; .\.venv_verify\Scripts\python.exe -m pip freeze
```

**Exit status:** 0

**Key output:**
```
n=2000
mean(y)=1.043136031500
lambda_mom=1.997182793444
lambda_hat=1.9947851486676018
score=0.000000e+00
loglik=-1493.421602205054
wrote C:\Users\micha\MAIStuff\randLab\geo&GenDL\gdlTests\grok_agent_workspace\task\results\estimate.json
numpy==2.5.3
scipy==1.18.1
```

**Result:** Clean-venv run reproduced `lambda_hat=1.9947851486676018` bit-for-bit. INSTALL.md last verified 2026-09-06 against `.venv_verify` (Python 3.12.5, numpy==2.5.3, scipy==1.18.1).

**Invalidates:** nothing — same artifact rewritten with the same number.

**Next:** done. Submission files are `task/solution/estimate_lambda.py` and `task/results/estimate.json`.
