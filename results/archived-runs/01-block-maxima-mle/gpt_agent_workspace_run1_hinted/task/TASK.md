# Task: Maximum likelihood estimation from block-maxima of an exponential

## Background

You are given `data/train.csv`, a single column `y` of 2000 real-valued
observations. Each observation was generated as

```
y = max(x_1, x_2, x_3, x_4),    x_1, x_2, x_3, x_4 i.i.d. ~ Exponential(λ)
```

for some fixed, unknown rate parameter λ > 0 (the same λ for every row).
That is, `y` is **not** itself an exponential draw — it is the maximum of a
group of `m = 4` independent exponential draws. `m = 4` is given to you
exactly; λ is the one unknown quantity you must estimate.

Full provenance and an integrity hash for `data/train.csv` are in
`data/DOWNLOAD.md`.

## What you must do

1. Derive the correct likelihood (or log-likelihood) of λ under this
   block-maxima generative model — **not** the likelihood for a plain
   `Exponential(λ)` sample. (Hint: first work out the CDF/PDF of the
   maximum of `m` i.i.d. `Exponential(λ)` random variables.)
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
   `lambda_hat` must be a finite, strictly positive number.

## Prohibited

- Do not read `data/holdout.csv`. It is withheld for grading and reading it
  is considered a failed submission regardless of your reported estimate.
- Do not read anything under `tests/` or `gt_code/`.
- Do not search the internet for the answer to this specific dataset; you
  may use general references on maximum likelihood estimation and order
  statistics.

## Acceptance criteria (for your awareness; grading is automated)

Your submission is graded by `tests/test_estimator.py`, which independently
recomputes the correct MLE from an independent held-out sample and checks
that your reported `lambda_hat` is close to it (well within reach of the
correct derivation, well outside reach of naively applying the standard
`Exponential(λ)` MLE formula to `y` as if it were raw exponential data).
