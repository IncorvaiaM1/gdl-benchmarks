# Solution: MLE of an Exponential rate from block maxima (`m = 4`)

**Answer:** `lambda_hat = 1.9947851486676023`, written to
`task/results/estimate.json`.

## 1. Derivation

Each observation is `y = max(x_1, ..., x_m)` with `x_j` i.i.d. `Exponential(λ)`
and `m = 4` known. The maximum of the block is below `y` exactly when all `m`
draws are, and the draws are independent, so the CDF of `Y` is the `m`-th power
of the Exponential CDF:

$$F(y;\lambda) \;=\; \Pr[Y \le y] \;=\; \Pr[X \le y]^m \;=\; \bigl(1-e^{-\lambda y}\bigr)^m, \qquad y>0 .$$

Differentiating gives the density of one observation (equivalently, the density
of the `m`-th order statistic of an Exponential sample):

$$f(y;\lambda) \;=\; m\,\lambda\, e^{-\lambda y}\bigl(1-e^{-\lambda y}\bigr)^{m-1}, \qquad y>0 .$$

This is the step that matters. `Y` is **not** Exponential, so `1/\bar y` is not
the MLE; on this dataset that naive estimator gives `0.9586`, off by 52%.

For an i.i.d. sample \(y_1,\dots,y_n\) the log-likelihood is

$$\ell(\lambda) \;=\; n\log m \;+\; n\log\lambda \;-\; \lambda\sum_{i=1}^n y_i \;+\; (m-1)\sum_{i=1}^n \log\bigl(1-e^{-\lambda y_i}\bigr).$$

The score, using \(\tfrac{d}{d\lambda}\log(1-e^{-\lambda y}) = y e^{-\lambda y}/(1-e^{-\lambda y})\), is

$$\ell'(\lambda) \;=\; \frac{n}{\lambda} \;-\; \sum_i y_i \;+\; (m-1)\sum_i \frac{y_i e^{-\lambda y_i}}{1-e^{-\lambda y_i}} \;=\; \frac{n}{\lambda} - \sum_i y_i + (m-1)\sum_i \frac{y_i}{e^{\lambda y_i}-1},$$

and the curvature is

$$\ell''(\lambda) \;=\; -\frac{n}{\lambda^2} \;-\; (m-1)\sum_i \frac{y_i^2 e^{-\lambda y_i}}{\bigl(1-e^{-\lambda y_i}\bigr)^2} \;<\; 0 \quad\text{for all } \lambda>0 .$$

Because \(\ell''<0\) everywhere, \(\ell\) is **strictly concave on \((0,\infty)\)**, so a
stationary point is the unique global maximum. Also \(\ell'(\lambda)\to+\infty\) as
\(\lambda\to0^+\) and \(\ell'(\lambda)\to-\sum_i y_i<0\) as \(\lambda\to\infty\), so the
score has exactly one sign change and any bracketing root-finder is guaranteed
to land on the MLE.

Setting \(\ell'(\lambda)=0\):

$$\frac{n}{\lambda} \;+\; (m-1)\sum_{i=1}^n \frac{y_i}{e^{\lambda y_i}-1} \;=\; \sum_{i=1}^n y_i .$$

For `m = 1` this collapses to \(\hat\lambda = 1/\bar y\). For `m > 1` the middle
term couples \(\lambda\) to every observation through a transcendental function, so
there is no closed form and the root is found numerically — as the task states.

**Starting value.** The maximum of `m` i.i.d. Exponentials is distributed as a
sum of independent Exponentials with rates \(m, m-1, \dots, 1\), hence
\(\mathbb{E}[Y] = H_m/\lambda\) with \(H_m=\sum_{k=1}^m 1/k\). The method-of-moments
estimator \(H_m/\bar y\) is consistent and, here, within `1.2e-3` of the MLE, which
makes it an ideal bracket centre. It is *not* the answer — the MLE is.

## 2. Numerical implementation

`blockmax_exp_mle.py` holds the model; `run_estimate.py` is the entry point.

Everything is evaluated in terms of `u = 1 - exp(-λy)` computed as
`-expm1(-λy)`, and `y/(exp(λy)-1)` is computed as `y*exp(-λy)/u`. The
`exp(-λy)` form underflows harmlessly to 0 for large `λy`, whereas the
algebraically identical `exp(λy)-1` form overflows once `λy > ~709`. Sums use
pairwise (`numpy`) or compensated (`math.fsum`) reduction.

The root is found three independent ways and the results are asserted to agree
to `1e-12` relative (they agree to `1.1e-16`, i.e. one ulp):

| Solver | Method | Result |
|---|---|---|
| `scipy.optimize.brentq` | Brent on the score, auto-expanded bracket | `1.9947851486676023` |
| stdlib only | bisection + Newton, different algebraic form of the score, `math.fsum` | `1.994785148667602` |
| `mpmath`, 50 digits | Newton, from the CSV's exact decimal strings | `1.99478514866760195042823544293` |

The reported float64 value is the correctly rounded double of the 50-digit
root, so all 17 of its digits are meaningful — far past the 8 significant
figures the task requires and the `3e-4` relative tolerance it is graded at.

## 3. Why the derivation itself is trusted

Three solvers agreeing only proves the arithmetic; it would agree just as
happily on a wrong likelihood. `verify_model.py` therefore tests the model from
outside the algebra:

- **Goodness of fit (KS on `data/train.csv`).** Against the fitted
  \((1-e^{-\hat\lambda y})^4\): `D = 0.014603`, `p = 0.78` — not rejected. Against
  the naive Exponential fit: `D = 0.237`, `p = 1.0e-99` — annihilated.
- **Simulation recovery.** 500 synthetic datasets of `n = 2000` drawn as
  `max` of 4 `Exponential(λ_0)` with `λ_0 = lambda_hat` (seed `20260906`),
  re-estimated: mean `1.9945499876`, bias `-2.35e-04` = **-0.21 Monte-Carlo
  standard errors**. The naive estimator on the same samples returns `0.957`,
  a bias of `-1.04`. A misderived likelihood shows up here as a bias many MC
  standard errors wide.
- **Standard error consistency.** The simulated spread of the MLE
  (`sd = 0.02486`) matches the observed-information standard error at the
  reported point (`0.02528`) to within 1.7%.

Reported estimate with uncertainty: \(\hat\lambda = 1.99479 \pm 0.02528\) (95% Wald
interval `[1.94524, 2.04433]`), which comfortably contains the round number
`2.0` that the data was most likely generated with.

## 4. Files

| File | Role |
|---|---|
| `blockmax_exp_mle.py` | model: log-likelihood, score, curvature, three solvers |
| `run_estimate.py` | **entry point** — verifies data hash, solves, writes `task/results/estimate.json` |
| `verify_model.py` | optional validation: KS goodness of fit + simulation recovery (uses seed `20260906`) |
| `requirements.txt` | pinned dependencies |

Run from the repo root:

```powershell
.\.venv\Scripts\python.exe task\solution\run_estimate.py    # writes the answer
.\.venv\Scripts\python.exe task\solution\verify_model.py    # optional checks
```

`run_estimate.py` is deterministic (no RNG). It exits non-zero if the
`data/train.csv` SHA-256 does not match the value published in
`data/DOWNLOAD.md`, if the solvers disagree, if the reported point is not a
strict local maximum, or if `estimate.json` fails to round-trip at full
precision.
