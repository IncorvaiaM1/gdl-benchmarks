# Derivation: MLE of λ from block maxima of m i.i.d. Exponential(λ)

## Generative model

Each row `y` of `data/train.csv` is

```
y = max(x_1, ..., x_m),   x_j i.i.d. ~ Exponential(λ),   m = 4 (known), λ > 0 (unknown)
```

## Distribution of the block maximum

For a single exponential draw, `P(X <= t) = 1 - exp(-λ t)` for `t >= 0`.
The maximum of `m` i.i.d. draws is below `t` iff *all* of them are, so

```
F_Y(t) = P(Y <= t) = [1 - exp(-λ t)]^m
```

Differentiating gives the density

```
f_Y(t; λ) = m λ exp(-λ t) [1 - exp(-λ t)]^(m-1),   t > 0
```

For `m = 1` this collapses to the ordinary exponential density; for `m > 1` the
extra factor `[1 - exp(-λ t)]^(m-1)` is exactly what the naive
"treat y as raw exponential data" approach throws away.

## Log-likelihood

With observations `y_1, ..., y_n`:

```
ℓ(λ) = n log m + n log λ − λ Σ y_i + (m − 1) Σ log(1 − exp(−λ y_i))
```

## Score (estimating) equation

```
ℓ'(λ) = n/λ − Σ y_i + (m − 1) Σ y_i exp(−λ y_i) / (1 − exp(−λ y_i))
      = n/λ − Σ y_i + (m − 1) Σ y_i / (exp(λ y_i) − 1)
```

Setting `ℓ'(λ) = 0` has no closed form in λ for `m > 1`, so it is solved
numerically. `ℓ'` is strictly decreasing on `(0, ∞)` with `ℓ'(λ) → +∞` as
`λ → 0+` and `ℓ'(λ) → −Σ y_i < 0` as `λ → ∞`, so the root exists, is unique,
and is the global maximiser of a strictly concave-at-the-root problem. We
bracket it and use Brent's method, then cross-check by directly minimising
`−ℓ(λ)`.

### Why the naive estimator is wrong (and how wrong)

Because `E[Y] = H_m / λ` with `H_m = Σ_{k=1..m} 1/k` (`H_4 = 25/12 ≈ 2.0833`),
applying the plain-exponential MLE `1/mean(y)` to block maxima underestimates λ
by a factor of about `H_m` — roughly 2.08× for `m = 4`. The method-of-moments
estimator `H_m / mean(y)` is in the right ballpark and is used only to build the
root-finding bracket, never as the reported answer.

## Files

| File | Purpose |
|---|---|
| `estimate_lambda.py` | Entry point. Reads the data, computes the MLE, writes `task/results/estimate.json`. |
| `selfcheck.py` | Validation harness: recovers known λ from simulated block maxima and checks the numerics against independent methods. |

## Usage

Run from anywhere; paths are resolved relative to the repository layout.

```bash
python task/solution/estimate_lambda.py
python task/solution/selfcheck.py
```
