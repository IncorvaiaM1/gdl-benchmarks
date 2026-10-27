# Task: Comparing flow likelihoods reported in different spaces

## Background

You are given `data/images.csv`: 1000 MNIST digits, one image per row, 784
columns (`p0` ... `p783`), each entry an integer pixel value
`x_d ∈ {0, 1, ..., 255}`. Let `D = 784`.

Two continuous normalizing-flow models were trained on 8-bit MNIST and then
evaluated on **exactly these 1000 images**. Both use uniform dequantization:
for each image `x`, draw `u ~ Uniform[0,1)^D` independently of everything
else and form

```
y = (x + u) / 256          (so y ∈ [0,1)^D)
```

The two models differ in the space their density is defined over.

- **Model A** is a flow whose density `p_A` is defined directly over `y`.
  Its reported score is
  `ℓ_A = E_{x,u}[ log p_A(y) ] = 1972.0` nats per image.

- **Model B** first applies the fixed elementwise preprocessing
  ```
  s  = λ + (1 − 2λ) · y,      λ = 0.05
  y' = logit(s) = log(s) − log(1 − s)
  ```
  and its flow density `p_B` is defined over `y'`. Its reported score is
  `ℓ_B = E_{x,u}[ log p_B(y') ] = −150.0` nats per image.

In both cases `E_{x,u}` averages over the 1000 images in `data/images.csv`
(uniform weight 1/1000 each) and, exactly, over the dequantization noise
`u ~ Uniform[0,1)^D`. All logarithms are natural logs. The numbers above are
exact; treat them as given.

## What you must do

For each model, let `p_V` denote the density that model implies over the
**unscaled** dequantized pixels `v = x + u ∈ [0,256)^D` (i.e. the model's
density re-expressed as a density on `v`, via change of variables through
whatever fixed maps connect `v` to the space the model's density lives in).
Define

```
bpd = − E_{x,u}[ log p_V(x + u) ] / (D · ln 2)
```

This is the standard dequantization upper bound on each model's discrete
negative log-likelihood of the 8-bit images, in bits per dimension, and it
is the quantity on which the two models can be compared fairly.

1. Derive `bpd` for Model A and for Model B in terms of the given scores and
   the data.
2. Compute both numbers for this dataset.
3. State which model has the lower (better) `bpd`.
4. Write your code to `task/solution/` (any file layout is fine, but
   include a single entry-point script that reproduces your result when run).
5. Write your final answer to `task/results/answer.json` as a JSON object
   with **exactly** this shape:
   ```json
   {"bpd_A": <float>, "bpd_B": <float>, "better": "A" or "B"}
   ```
   Report `bpd_A` and `bpd_B` to at least **6 significant figures**.

## Prohibited

- Do not read anything under `tests/` or `gt_code/`.
- Do not search the internet for the answer to this specific task; you
  may use general references on normalizing flows, change of variables,
  and dequantization.

## Acceptance criteria (for your awareness; grading is automated)

Your submission is graded by `tests/test_solution.py`, which independently
recomputes both `bpd` values from `data/images.csv` and checks each of
yours to an absolute tolerance of `2e-3` bits per dimension, and checks
that `better` is correct.
