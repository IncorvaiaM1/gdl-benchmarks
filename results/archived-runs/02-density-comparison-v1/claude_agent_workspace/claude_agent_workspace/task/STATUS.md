# STATUS — append-only work log

Newest entries at the bottom. Never edit or delete an earlier entry; corrections are
appended as new entries.

---

## 2026-09-23 17:02 — initial state

**Machine:** `Windows-11-10.0.26200-SP0`, `AMD64`. Shell: PowerShell.
`Python 3.12.5` (`MSC v.1940 64 bit (AMD64)`), `pip 26.1.2`, `numpy==2.2.1`
(ambient system interpreter, no virtualenv created).

**Repo / commit:** the project folder contains no `.git` directory of its own, so
there is no project-local commit to record. Project root for every path in this file
is the directory that directly contains `data/` and `task/` — note the workspace has
a nested directory of the same name, and the inner one is the root.

**Command:**
```powershell
Get-ChildItem -Recurse -Force | Select-Object -ExpandProperty FullName
```

**Exit status:** 0

**Key output:**
```
data/DOWNLOAD.md
data/images.csv
task/INSTALL.md          (stub: "YOUR INSTALLATION GUIDE GOES HERE")
task/PROJECT_INSTRUCTION.md
task/STATUS.md           (stub: "THE AGENT LOG GOES HERE.")
task/TASK.md
task/results/.gitkeep
task/solution/.gitkeep
```

**Result:** Starting state is the task statement plus `data/images.csv`; `solution/`
and `results/` are empty. No prior artifacts to trust or invalidate.

**Next:** verify the data file against the published hash before computing anything.

---

## 2026-09-23 17:05 — verify environment and data integrity

**Command:**
```powershell
python -V; python -c "import numpy, sys; print(numpy.__version__)"; (Get-FileHash -Algorithm SHA256 data\images.csv).Hash
```

**Exit status:** 0

**Key output:**
```
Python 3.12.5
2.2.1
98B5D6A6D90BC2BC1C22DF168E45E63DA97FA2CCE5D75001650F7F391A8B69C4
```

**Command:**
```powershell
Get-Content data\images.csv -TotalCount 2 | ForEach-Object { $_.Substring(0, [Math]::Min(200, $_.Length)) }; (Get-Content data\images.csv | Measure-Object -Line).Lines
```

**Exit status:** 0

**Key output:**
```
header: p0,p1,p2,...
first data row begins with a long run of 0
1001            (1000 images + 1 header row)
```

**Result:** SHA-256 matches `data/DOWNLOAD.md`
(`98b5d6a6d90bc2bc1c22df168e45e63da97fa2cce5d75001650f7f391a8b69c4`) — this is the
intended file. Shape confirmed as 1000×784 integer pixels. Nothing written; no state
changed beyond confirming inputs.

**Next:** derive the two bpd expressions, then implement.

---

## 2026-09-23 17:12 — derivation fixed (no command; recorded because every later number depends on it)

The maps are `v = x+u`, `y = v/256`, `s = λ + (1−2λ)y`, `y' = logit(s)`, all
elementwise and increasing, so Jacobians are diagonal.

- **Model A:** `log p_V(v) = log p_A(y) − 8 D ln 2`, hence
  `bpd_A = 8 − ℓ_A/(D ln 2)`. **The data cancels** — constant scale factor — so
  `bpd_A` is a closed form in `ℓ_A` alone.
- **Model B:** `bpd_B = 8 − [ ℓ_B + D ln(1−2λ) − S ] / (D ln 2)` with
  `S = E_{x,u}[ Σ_d ( log s_d + log(1−s_d) ) ]`. `S` is the data-dependent part: the
  logit Jacobian is not constant.
- `S` is computed **exactly in `u`**, not sampled. For level `k`, `s = a_k + c·u`
  with `a_k = λ + (1−2λ)k/256`, `c = (1−2λ)/256`, and
  `∫_0^1 log(a+cu) du = log a + (1 + a/c)·log1p(c/a) − 1`. The `log(1−s)` term is the
  same with `a → 1−a_k`, `c → −c`. So `S` reduces to a 256-entry lookup dotted with
  the per-level pixel counts (`np.bincount`), divided by 1000.
- **Numerical note, load-bearing:** the raw antiderivative form
  `((a+c)log(a+c) − a log a)/c − 1` subtracts two nearly equal `O(a log a)` terms
  (`c ≈ 3.5e-3` vs `a ∈ [0.05, 0.947]`) and loses digits. The `log1p` form above is
  algebraically identical and cancellation-free; it is what `solve.py` uses.
- Because `λ = 0.05`, `s ∈ [0.05, 0.95]` for every level — no endpoint singularity,
  and `|c/a| ≤ 0.066` so `log1p` stays in its accurate range.

**Result:** Full derivation written to `task/solution/DERIVATION.md`.

**Next:** implement as `task/solution/solve.py` with independent cross-checks.

---

## 2026-09-23 17:20 — implement and run the solver (produces the final answer)

**Command:**
```powershell
python task\solution\solve.py
```

**Exit status:** 0

**Key output:**
```
sha256: 98b5d6a6d90bc2bc1c22df168e45e63da97fa2cce5d75001650f7f391a8b69c4
images: 1000   pixels/image: 784   zero-valued pixel fraction: 0.806960
cross-checks:
  closed form vs Gauss-Legendre(200): max abs diff 8.882e-16
  closed form vs Monte Carlo(4e5/level): max abs diff 3.838e-05
  bpd_B via sampled dequantization noise: 4.290817 (exact 4.290814, diff 3.34e-06)

S            = -2248.2758955354
D ln(1-2lam) =   -82.6026442757
D ln2        =   543.4273895590

bpd_A = 4.3711803308
bpd_B = 4.2908140260
better = B
```

**Result:** Wrote `task/solution/solve.py` and `task/results/answer.json`:
```json
{"bpd_A": 4.37118033082521, "bpd_B": 4.290814025963155, "better": "B"}
```
Model B is better by `0.0804` bpd. This **reverses** the naive reading of the raw
scores (`ℓ_A = +1972.0` vs `ℓ_B = −150.0`): B's density lives in a logit-stretched
space with a much larger volume element, and the `S` term is exactly the correction.

Three independent checks agree, so the closed form is not the only evidence for the
number: analytic vs 200-node Gauss–Legendre (`8.9e-16`), vs per-level Monte Carlo
(`3.8e-5`), and `bpd_B` recomputed end to end from actually sampled `u` over all
784,000 pixels × 20 repeats, seed `np.random.default_rng(0)` (`4.290817` vs
`4.290814`, diff `3.3e-6`). Task tolerance is `2e-3` bpd, so the margin is ~3 orders
of magnitude.

**Determinism:** `bpd_A`/`bpd_B` are deterministic (pure double arithmetic, no RNG on
the result path). Only the printed cross-check diffs depend on the RNG stream, hence
on the NumPy version; their asserts are loose (`1e-12` quadrature, `1e-4` per-level
MC, `5e-4` end-to-end bpd).

**Invalidates:** nothing — first result produced. If `data/images.csv` ever changes,
`bpd_B` and `better` must be recomputed (`bpd_A` would not change, since it is
data-independent).

**Next:** record the environment in `INSTALL.md` and time the fast path.

---

## 2026-09-23 17:26 — pin environment, time the fast path, rewrite answer.json

**Command:**
```powershell
python -c "import sys,platform;print(sys.version);print(platform.platform());print(platform.machine())"; python -m pip freeze | Select-String -Pattern "^(numpy|setuptools|pip)=" ; python -m pip --version
```

**Exit status:** 0

**Key output:**
```
3.12.5 (tags/v3.12.5:ff3bc82, Aug  6 2024, 20:45:27) [MSC v.1940 64 bit (AMD64)]
Windows-11-10.0.26200-SP0
AMD64
numpy==2.2.1
setuptools==80.9.0
pip 26.1.2
```

**Command:**
```powershell
Measure-Command { python task\solution\solve.py --no-checks --out task\results\answer.json } | Select-Object -ExpandProperty TotalSeconds; Get-Content task\results\answer.json
```

**Exit status:** 0

**Key output:**
```
0.3482113
{
  "bpd_A": 4.37118033082521,
  "bpd_B": 4.290814025963155,
  "better": "B"
}
```

**Result:** Wrote `task/INSTALL.md` with the platform block, the exact pins above,
and the venv creation commands. Rewrote `task/results/answer.json` via the
`--no-checks` path; byte-identical values to the 17:20 run, confirming the
cross-checks do not perturb the answer. Fast path is `0.35 s`; the full run is ~15 s
(all Monte Carlo).

**Correction to `INSTALL.md` as first drafted:** it estimated `--no-checks` at ~2 s
before measuring. The measured value is `0.35 s`, and the file now says so.

**INSTALL.md verification status — read this before trusting it:** the pins and
platform block were generated from `pip freeze` on this machine, not written from
memory, and the documented run command is the one exercised above. However,
`INSTALL.md` has **not** been followed from a clean virtualenv or container, so the
`python -m venv` + `pip install numpy==2.2.1` path is unverified. It is a two-line
install of a single dependency, so the risk is low, but it is a guess until someone
runs it. The `--data`/`--out` overrides and running from a non-root working directory
are likewise untested.

**Invalidates:** nothing.

**Next:** none — task complete. Remaining optional work is the clean-environment
`INSTALL.md` verification noted above.
