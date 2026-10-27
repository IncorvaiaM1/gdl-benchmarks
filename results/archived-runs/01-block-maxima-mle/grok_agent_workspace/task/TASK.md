# Task: Maximum likelihood estimation from block-maxima of an exponential

## Background

You are given `data/train.csv`, a single column `y` of 2000 real-valued
observations. Each observation was generated as

```
y = max(x_1, x_2, x_3, x_4),    x_1, x_2, x_3, x_4 i.i.d. ~ Exponential(λ)
```

for some fixed, unknown rate parameter λ > 0 (the same λ for every row).
`m = 4` is given to you exactly; λ is the one unknown quantity you must
estimate.

Full provenance and an integrity hash for `data/train.csv` are in
`data/DOWNLOAD.md`.

## What you must do

1. Derive the correct likelihood (or log-likelihood) of λ for this
   generative model.
2. Implement code that computes the maximum likelihood estimate
   $\hat\lambda$ of λ from `data/train.csv`, using the correct likelihood
   from step 1. The resulting estimating equation does not have a closed-form
   solution in λ for `m > 1`; you are expected to solve it numerically
   (e.g. root-finding or numerical optimization over the log-likelihood).
3. Write your code to `task/solution/` (any file layout is fine, but
   include a single entry-point script that reproduces your result when run).
4. Write your final answer to `task/results/estimate.json` as a JSON object
   with **exactly** this shape:
   ```json
   {"lambda_hat": <float>}
   ```
   `lambda_hat` must be a finite, strictly positive number, reported to at
   least **8 significant figures** (it is checked against an independently
   recomputed reference value at `3e-4` relative tolerance — see Acceptance
   criteria below — and a value rounded to only 3-4 significant figures can
   fail that check even when your derivation and computation are correct).

## Prohibited

- Do not read `data/holdout.csv`. It is withheld for grading and reading it
  is considered a failed submission regardless of your reported estimate.
- Do not read anything under `tests/` or `gt_code/`.
- Do not search the internet for the answer to this specific dataset; you
  may use general references on maximum likelihood estimation and order
  statistics.

## Acceptance criteria (for your awareness; grading is automated)

Your submission is graded by `tests/test_estimator.py`, which independently
recomputes the correct maximum likelihood estimate from `data/train.csv`
and checks that your reported `lambda_hat` is within `3e-4` relative error
of it, plus a looser secondary check against a held-out sample.
