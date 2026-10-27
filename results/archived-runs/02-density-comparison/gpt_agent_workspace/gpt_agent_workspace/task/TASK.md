# Task: Which of three MNIST generative models is actually best?

## Background

`data/images.csv` holds 1000 MNIST digits, one image per row, 784 columns
(`p0` ... `p783`, row-major 28×28). Each entry is the raw 8-bit pixel value
`k ∈ {0, 1, ..., 255}`.

Three generative models were evaluated on **exactly these 1000 images**.
Each team reported a single number, computed with the preprocessing code
shown below. In all snippets, `to_tensor` maps a uint8 image to float32 by
dividing by 255 (so pixel `k` becomes `k/255`), `torch.rand_like` draws
i.i.d. Uniform[0,1) noise, and a fresh noise / Bernoulli draw is made every
time an image is used. Reported numbers are exact averages over the 1000
images and over all of this randomness. All logarithms are natural logs.

### Model A: normalizing flow

```python
x = to_tensor(img)                       # shape (784,)
y = (x * 255 + torch.rand_like(x)) / 256
score = flow_A.log_prob(y)               # log-density of y under flow A
```
Reported: **mean `score` = 1972.0 nats per image.**

### Model B: normalizing flow in logit space (λ = 0.05)

```python
x = to_tensor(img)                       # shape (784,)
y = x + torch.rand_like(x) / 256
s = lam + (1 - 2 * lam) * y
y_prime = torch.log(s) - torch.log1p(-s)
score = flow_B.log_prob(y_prime)         # log-density of y_prime under flow B
```
Reported: **mean `score` = −150.0 nats per image**, with `lam = 0.05`.

### Model C: VAE with a Bernoulli decoder

```python
x = to_tensor(img).view(1, 28, 28)
b = torch.bernoulli(x)                   # dynamic binarization
b = F.pad(b, (2, 2, 2, 2))               # zero-pad to 1 x 32 x 32
logits, kl = vae(b)
loss = F.binary_cross_entropy_with_logits(logits, b, reduction="sum") + kl
```
Reported: **mean `loss` = 93.7 nats per image.**

Both flows have densities that are strictly positive and finite on all of
their input space. Treat every reported number as exact.

## The comparison

For a model that assigns a probability `P(x)` to each 8-bit image `x`, the
quantity of interest is

```
bpd(model) = − (1/1000) Σ_x log₂ P(x)  /  784
```

over the 1000 images. For a continuous model, `P(x)` is the probability its
density assigns to the set of inputs that image `x` can be turned into by
the model's own preprocessing and noise. The reported numbers do not give
`bpd` directly, so for each model use the standard dequantization
(Jensen) upper bound on `bpd` implied by the reported number, whenever the
reported number implies one.

## What you must report

1. `bpd_A`: the upper bound on `bpd` for Model A.
2. `bpd_B`: the upper bound on `bpd` for Model B (λ = 0.05), or `null` if
   Model B's reported number does not imply a valid upper bound on its `bpd`.
3. `better`: `"A"` or `"B"`, whichever of the two flows has the lower bound.
4. `lambda_min`: Model B's pipeline is only well-defined (finite `y_prime`
   for every possible 8-bit image and every noise draw) for some values of
   `lam ∈ (0, 1/2)`. Report the smallest such `lam`.
5. `lambda_star`: suppose Model B's team had not stated `lam`. Keeping the
   reported mean score fixed at −150.0 and the pipeline otherwise
   unchanged, report the `lam ∈ [lambda_min, 1/2)` at which Model B's bound
   exactly equals Model A's.
6. `bpd_C_own`: Model C's reported number expressed in bits per dimension
   of the random variable that Model C actually models.
7. `bpd_C_8bit`: an upper bound on `bpd` for Model C computed from its
   reported number, or `null` if its reported number implies no such bound.

## Deliverables

- Code in `task/solution/` with a single entry-point script that reproduces
  your answer when run.
- `task/results/answer.json`, a JSON object with **exactly** these keys:
  ```json
  {"bpd_A": <float>, "bpd_B": <float or null>, "better": "A" or "B",
   "lambda_min": <float>, "lambda_star": <float>,
   "bpd_C_own": <float>, "bpd_C_8bit": <float or null>}
  ```
  Report floats to at least 6 significant figures.

## Prohibited

- Do not read anything under `tests/` or `gt_code/`.
- Do not search the internet for the answer to this specific task; you
  may use general references on normalizing flows, VAEs, and dequantization.

## Acceptance criteria (for your awareness; grading is automated)

`tests/test_solution.py` recomputes every value from `data/images.csv`.
Tolerances: `bpd_A`, `bpd_B` within 1e-3 bits/dim; `lambda_min` within
1e-6 relative; `lambda_star` within 5e-5 absolute; `bpd_C_own` within 1e-4;
`better` and any `null` must match exactly.
