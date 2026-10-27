# STATUS — append-only work log

## 2026-09-23 18:55 — initial state

**Machine:** Windows 11 (`Windows-11-10.0.26200-SP0`, AMD64), PowerShell,
`Python 3.12.5` at `C:\Python312\python.exe`. No GPU used, no network needed.

**Workspace:** `.../agent_runs/claude_agent_workspace/claude_agent_workspace`
(the task tree is one level below the folder the session opened in). Contents at
start:

```
data/DOWNLOAD.md
data/images.csv
task/TASK.md
task/PROJECT_INSTRUCTION.md
task/STATUS.md        (placeholder: "THE AGENT LOG GOES HERE.")
task/INSTALL.md       (placeholder: "YOUR INSTALLATION GUIDE GOES HERE")
task/results/         (empty)
task/solution/        (empty)
```

Not a git repository at the workspace level, so there is no commit to record. No
`tests/` or `gt_code/` directory exists here; nothing outside the workspace was read.

**Task:** put the three reported model numbers (flow A: +1972.0 nats, logit-flow B:
−150.0 nats at λ=0.05, Bernoulli VAE C: 93.7 nats) on a common bits/dim scale and
answer the seven fields of `task/results/answer.json`.

**Next:** verify the data file, then derive the bounds.

---

## 2026-09-23 18:58 — verify input data integrity

**Command:**
```bash
python -c "import hashlib; h=hashlib.sha256(open('data/images.csv','rb').read()).hexdigest(); print(h); print(h=='98b5d6a6d90bc2bc1c22df168e45e63da97fa2cce5d75001650f7f391a8b69c4')"
```

**Exit status:** 0

**Key output:**
```
98b5d6a6d90bc2bc1c22df168e45e63da97fa2cce5d75001650f7f391a8b69c4
True
```

**Result:** `data/images.csv` matches the SHA-256 recorded in `data/DOWNLOAD.md`.
1001 lines = header + 1000 images, 784 columns. This hash is now also asserted at
every run of the entry point, so a swapped data file cannot go unnoticed.

---

## 2026-09-23 19:00 — record the environment

**Command:**
```bash
python -V
python -m pip freeze | Select-String -Pattern "^(numpy|scipy|mpmath)=="
python -c "import platform,sys; print(platform.platform()); print(platform.machine()); print(sys.version)"
```

**Exit status:** 0

**Key output:**
```
Python 3.12.5
mpmath==1.3.0
numpy==2.2.1
scipy==1.15.0
Windows-11-10.0.26200-SP0
AMD64
```

**Result:** Wrote `task/INSTALL.md`. **Nothing was installed** for this task — the
interpreter already had numpy/scipy/mpmath. The solution imports only `numpy`
(plus optional `mpmath` for the `--verify` cross-check); scipy is *not* imported,
the `lambda_star` root-find is a hand-written bisection to keep the dependency set
minimal. No PyTorch: the snippets in `TASK.md` are read as a specification of each
team's preprocessing, not executed.

---

## 2026-09-23 19:10 — derivation (no command; recorded because every later number depends on it)

Kept here so the numbers below can be checked without re-deriving them. Full
version lives in the docstring of `task/solution/solve.py`.

- **Jensen / dequantization bound.** If preprocessing maps image `x` to `y` drawn
  uniformly from a set `S_x` of volume `V`, then
  `P(x) = ∫_{S_x} p(y) dy = V·E_u[p(y)] ≥ V·exp(E_u[log p(y)])`, so
  `−log P(x) ≤ −E_u[log p(y)] + log(1/V)`. Valid as a bound on a probability
  **only if the `S_x` are disjoint** across images.
- **A:** `y = (x·255+u)/256 = (k+u)/256`. Cubes of side 1/256 tiling `[0,1)^784` —
  disjoint. `bpd_A = 8 − 1972.0/(784·ln2)`.
- **B:** `y = x + u/256 = k/255 + u/256`. *Not* `(k+u)/256`. Bin for value `k` is
  `[k/255, k/255+1/256)`, still width 1/256 but shifted; consecutive bins are
  separated by a gap `1/255 − 1/256 = 1/(255·256) > 0`, so they are **disjoint**
  (they just do not cover the space). Disjointness is what the bound needs ⇒ B's
  number *does* imply a valid upper bound, merely a loose one. Then
  `log p_Y(y) = log p_{Y'}(y') + Σ_i [log(1−2λ) − log s_i − log(1−s_i)]`
  since `dy'/dy = (1−2λ)/(s(1−s))`, and
  `bpd_B = (784·ln256 − (−150.0 + L(λ)))/(784·ln2)`, with `L(λ)` the mean log-det.
- **Exact noise average** (no sampling): with `a = λ+(1−2λ)k/255`, `b = (1−2λ)/256`,
  `s = a+bu`, and `G(z) = z log z − z` (`G(0)=0`),
  `E_u[−log s] = −(G(a+b)−G(a))/b`, `E_u[−log(1−s)] = (G(1−a−b)−G(1−a))/b`.
  Aggregated over a 256-bin pixel histogram of the 784 000 pixels.
- **λ_min:** need `0 < s < 1`; `s` is largest as `k=255, u→1`, i.e. `y → 1+1/256`
  (sup not attained because `u < 1`), so `λ + (1−2λ)(1+1/256) ≤ 1 ⇔ λ ≥ 1/258`.
  **λ_min = 1/258 = 0.003875968992248062.** (The minimum exists precisely because
  `torch.rand_like` is half-open `[0,1)`.)
- **λ\*:** setting `bpd_B(λ) = bpd_A` cancels the `784·ln256` term and reduces to
  `L(λ) = 1972.0 − (−150.0) = 2122.0` exactly. `L` is strictly decreasing on
  `[1/258, 1/2)`, so the root is unique; bisection.
- **C:** 93.7 nats is −ELBO for the padded binary image `b ∈ {0,1}^{32×32}` (BCE is
  summed over all 1024 pixels), so `bpd_C_own = 93.7/(1024·ln2)`.
  `bpd_C_8bit = null`: `b ~ Bernoulli(x)` is a stochastic, information-destroying,
  **many-to-one** map — different 8-bit images produce the same `b` with positive
  probability, so the "sets" overlap and no normalised probability over 8-bit
  images, hence no bound, follows from the reported number.

---

## 2026-09-23 19:20 — write and run the solver

**Command:**
```bash
python task/solution/solve.py
```

**Exit status:** 0

**Key output:**
```
sha256: 98b5d6a6d90bc2bc1c22df168e45e63da97fa2cce5d75001650f7f391a8b69c4  (expected match: True)
n_images: 1000
n_pixels: 784000
frac_zero_pixels: 0.8069604591836734
mean_logdet_B_at_lam_0.05: 2169.596933100584
mean_logdet_B_at_lambda_star: 2122.0000000000023
target_logdet_for_lambda_star: 2122.0
{
  "bpd_A": 4.371180330825209,
  "bpd_B": 4.28359377553728,
  "better": "B",
  "lambda_min": 0.003875968992248062,
  "lambda_star": 0.05335999058406782,
  "bpd_C_own": 0.13201223176884364,
  "bpd_C_8bit": null
}
```

**Result:** Wrote `task/solution/solve.py` (single entry point, numpy only) and
`task/results/answer.json`. Sanity: `L(0.05) = 2169.5969 > 2122`, and `L` decreases
in λ, so `λ* = 0.053360 > 0.05` as expected — B's advantage at λ=0.05 is exactly the
`(2169.597 − 2122)/(784·ln2) = 0.0876` bits/dim by which its bound beats A's.

**Invalidates:** nothing earlier (first result produced).

**Next:** cross-check the closed-form noise average two independent ways before
trusting it.

---

## 2026-09-23 19:24 — cross-check the closed-form expectation

**Command:**
```bash
python task/solution/solve.py --verify
```

**Exit status:** 0

**Key output:**
```
lam=0.0500000000  exact L=2169.596933  MC L=2169.597608  diff=-0.0007
lam=0.0533599906  exact L=2122.000000  MC L=2122.000612  diff=-0.0006
lam=0.0500000000  float64 L=2169.596933100584  mpmath L=2169.59693310058116
lam=0.0533599906  float64 L=2122.0000000000023  mpmath L=2121.99999999999798
```

**Result:** The closed-form float64 log-determinant agrees with 40-digit `mpmath`
numerical quadrature to ~1e-12 nats/image, and with a 40-replicate Monte-Carlo
average over the actual `u ~ U[0,1)` draws (`numpy.random.default_rng(0)`,
40 × 784 000 samples) to ~7e-4 nats/image — consistent with MC noise on a quantity of
size 2122. Propagated to bits/dim that MC gap is ~1.3e-6, far inside the 1e-3
tolerance. `answer.json` unchanged (rewritten identically).

`--verify` is informational only: the reported answer uses no randomness at all.

---

## 2026-09-23 19:30 — INSTALL.md verification in a clean environment

**Command:**
```bash
python -m venv .venv-verify
.\.venv-verify\Scripts\python.exe -m pip install --quiet --disable-pip-version-check "numpy==2.2.1"
.\.venv-verify\Scripts\python.exe -m pip freeze
Copy-Item task/results/answer.json answer_ref.tmp.json
.\.venv-verify\Scripts\python.exe task/solution/solve.py --verify
(Get-FileHash task/results/answer.json).Hash -eq (Get-FileHash answer_ref.tmp.json).Hash
```

**Exit status:** 0

**Key output:**
```
numpy==2.2.1
...
lam=0.0500000000  exact L=2169.596933  MC L=2169.597608  diff=-0.0007
mpmath not installed; skipped high-precision check
---- byte-identical:
True
```

**Result:** Followed `INSTALL.md` from a fresh venv containing **only**
`numpy==2.2.1` (no scipy, no mpmath, no torch) and reproduced
`task/results/answer.json` byte-for-byte. The optional mpmath check self-skipped as
designed, and the Monte-Carlo numbers were identical to the system-interpreter run,
confirming the seeding is deterministic across environments. `INSTALL.md` updated
with this verification record.

**Cleanup command:**
```bash
Remove-Item -Recurse -Force .venv-verify; Remove-Item -Force answer_ref.tmp.json
```
Workspace is back to `data/` + `task/` only. Re-running the clean-env check means
re-creating the venv (~50 s, the only step in this project that needs network).

---

## 2026-09-23 19:35 — final state

**Deliverables:**

- `task/solution/solve.py` — single entry point; `python task/solution/solve.py`
  reads `data/images.csv` and writes `task/results/answer.json`. `--verify` adds the
  cross-checks; `--images` / `--out` override paths.
- `task/results/answer.json` — final answer:

```json
{
  "bpd_A": 4.371180330825209,
  "bpd_B": 4.28359377553728,
  "better": "B",
  "lambda_min": 0.003875968992248062,
  "lambda_star": 0.05335999058406782,
  "bpd_C_own": 0.13201223176884364,
  "bpd_C_8bit": null
}
```

**Failures / dead ends to record:** none — no command in this project exited
non-zero. The two judgement calls that could have gone the other way, written down
so they are not silently re-litigated:

1. `bpd_B` is **not** `null`. The tempting reading is that B's `y = x + u/256`
   (with `x = k/255`, not `(k+u)/256`) breaks the bound. It does not: the bins stay
   disjoint because `1/256 < 1/255`; they only leave uncovered gaps, which makes the
   bound loose, not invalid. Had B's bins *overlapped*, `Σ_x P(x) > 1` and the
   answer would have been `null`.
2. `bpd_C_own` divides by **1024, not 784**. The BCE is summed over the zero-padded
   32×32 image and the decoder emits 1024 logits, so the variable whose −ELBO is
   93.7 nats is 1024-dimensional. `784` would give 0.172424 instead of 0.132012 —
   far outside the 1e-4 tolerance, so this choice is load-bearing.

**Next:** nothing outstanding.
