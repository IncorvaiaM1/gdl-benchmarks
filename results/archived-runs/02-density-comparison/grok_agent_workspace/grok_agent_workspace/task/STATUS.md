## 2026-09-23 19:08 — initial state

**Command:**
```bash
python -V
python -c "import sys,platform; print(sys.executable); print(platform.platform()); print(platform.machine())"
git rev-parse --short HEAD
Get-FileHash -Algorithm SHA256 data\images.csv
```

**Exit status:** 0

**Key output:**
```
Python 3.12.5
C:\Python312\python.exe
Windows-11-10.0.26200-SP0
AMD64
git HEAD: 1ff2b53 (parent randLab repo; this workspace is not itself the git root)
sha256(data/images.csv): 98B5D6A6D90BC2BC1C22DF168E45E63DA97FA2CCE5D75001650F7F391A8B69C4
  matches DOWNLOAD.md 98b5d6a6d90bc2bc1c22df168e45e63da97fa2cce5d75001650f7f391a8b69c4
date: 2026-09-23 19:08:40 -04:00
```

**Result:** Workspace contains `data/images.csv` (1000 MNIST 8-bit images) and task spec. No solution yet. System Python 3.12.5 will be used only to create a local venv.

**Invalidates:** nothing

**Next:** create `.venv`, install numpy/scipy, implement the BPD comparison from first principles.

## 2026-09-23 19:09 — create venv and install numpy/scipy

**Command:**
```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy scipy
.\.venv\Scripts\python.exe -m pip freeze
.\.venv\Scripts\python.exe -V
```

**Exit status:** 0

**Key output:**
```
Successfully installed pip-26.2.1
Successfully installed numpy-2.5.3 scipy-1.18.1
numpy==2.5.3
scipy==1.18.1
Python 3.12.5
```

**Result:** Local `.venv` created from system Python 3.12.5. numpy is required for the histogram/closed-form Jacobian. scipy is used only to cross-check that closed form against `scipy.integrate.quad`.

**Invalidates:** nothing previously computed (no results yet)

**Next:** implement and run `task/solution/solve.py`.

## 2026-09-23 19:11 — first solver run (failed self-check)

**Command:**
```bash
.\.venv\Scripts\python.exe task\solution\solve.py
```

**Exit status:** 1

**Key output:**
```
RuntimeError: integral check failed, max abs err 8.461020947425624e-05
```

**Result:** Uniform Riemann sum of `-log s(1-s)` is a poor check at `lambda_min` for white pixels, where `s → 1` and the integrand has a log singularity. The closed form is the correct (integrable) expectation; the checker was too crude. Replaced with adaptive `quad`.

**Invalidates:** no `answer.json` was written

**Next:** re-run after switching the self-check to adaptive quadrature.

## 2026-09-23 19:12 — compute answer.json

**Command:**
```bash
.\.venv\Scripts\python.exe task\solution\solve.py
```

**Exit status:** 0

**Key output:**
```
{
  "bpd_A": 4.371180330825209,
  "bpd_B": 4.283593775537259,
  "better": "B",
  "lambda_min": 0.003875968992248062,
  "lambda_star": 0.053359990584068234,
  "bpd_C_own": 0.13201223176884364,
  "bpd_C_8bit": null
}
Q(lambda=0.05) = 2252.199577376331
wrote task/results/answer.json
```

**Result:** Jensen upper bounds: Model A 4.37118 bpd, Model B 4.28359 bpd so B is better. `lambda_min = 1/258`. `lambda_star ≈ 0.053360` equalizes the two flow bounds with the reported score held at −150. Model C's 93.7 nats is 0.132012 bpd on the 32×32 binary tensor it actually models; it does not bound 8-bit BPD.

**Invalidates:** nothing

**Next:** Monte-Carlo check of Model B's Jacobian against the closed form.

## 2026-09-23 19:13 — verify Jacobian and lambda_star

**Command:**
```bash
.\.venv\Scripts\python.exe -c "..."
```
(inline check: `bpd_B(lambda_star) - bpd_A`, 1-sample MC of `log |∂y'/∂u|` at λ=0.05 with seed 0)

**Exit status:** 0

**Key output:**
```
bpd_A 4.371180330825209
bpd_B(star) 4.371180330825224  diff 1.509903313490213e-14
lambda_min 0.003875968992248062 = 1/258
logJ MC    -2177.819806298047
logJ exact -2177.8221833713815
MC err 0.0023770733346282213
n pixels 784000.0  frac zero 0.8069604591836734  frac 255 0.0064885204081632656
unique used 256
```

**Result:** Root for `lambda_star` matches `bpd_A` to 1e-14. One MC noise draw per pixel agrees with the exact `E_u log|J|` to ~0.002 nats/image (far inside the 1e-3 bpd tolerance). Pixel histogram uses all 256 gray levels; ~80.7% zeros, as expected for MNIST.

**Invalidates:** nothing

**Next:** write INSTALL.md from the actual freeze and record that the venv path was followed end-to-end.

## 2026-09-23 19:14 — INSTALL.md locked to freeze

**Command:**
```bash
.\.venv\Scripts\python.exe -m pip freeze
```

**Exit status:** 0

**Key output:**
```
numpy==2.5.3
scipy==1.18.1
```

**Result:** `task/INSTALL.md` now lists the platform, the three create/install commands, and those two pins. The `.venv` was created from scratch earlier this session and `task/solution/solve.py` ran to completion in it, so the guide has been followed end-to-end once.

**Invalidates:** nothing

**Next:** none; answer is in `task/results/answer.json`.

