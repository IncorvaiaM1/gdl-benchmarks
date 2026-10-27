# Derivation

## Setup

`D = 784`, `λ = 0.05`, levels `x_d ∈ {0..255}`, `u ~ Uniform[0,1)^D`. The chain of
fixed maps is

```
v = x + u  ∈ [0,256)^D          (unscaled dequantized pixels — the target space)
y = v / 256                     (Model A's space)
s = λ + (1 − 2λ) y              (elementwise, into (λ, 1−λ))
y' = logit(s)                   (Model B's space)
```

All maps are elementwise and strictly increasing, so every Jacobian is diagonal and
the change of variables `p_V(v) = p(·) · |det ∂(·)/∂v|` applies per coordinate.

## Model A

`∂y_d/∂v_d = 1/256`, so

```
log p_V(v) = log p_A(y) − D log 256 = log p_A(y) − 8 D ln 2
```

Taking `E_{x,u}`:

```
bpd_A = −(ℓ_A − 8 D ln 2) / (D ln 2) = 8 − ℓ_A / (D ln 2)
```

The data cancels entirely — the scale factor is constant, so `bpd_A` depends only on
`ℓ_A`. Numerically `8 − 1972 / (784 · ln 2) = 4.37118033`.

## Model B

```
∂y'_d/∂v_d = (dy'_d/ds_d)(ds_d/dy_d)(dy_d/dv_d) = 1/(s_d(1−s_d)) · (1−2λ) · 1/256
```

so

```
log p_V(v) = log p_B(y') + Σ_d [ −log s_d − log(1−s_d) + log(1−2λ) − log 256 ]
```

and with `S := E_{x,u}[ Σ_d ( log s_d + log(1−s_d) ) ]`,

```
E[log p_V] = ℓ_B + D log(1−2λ) − 8 D ln 2 − S

bpd_B = 8 − [ ℓ_B + D log(1−2λ) − S ] / (D ln 2)
```

Unlike Model A, `S` depends on the data: the logit Jacobian is not constant, and this
term is exactly what makes a logit-space log-density incomparable to a `[0,1]`-space
one until it is converted.

### Computing `S` exactly in the dequantization noise

For a pixel at level `k`, `s = a_k + c·u` with

```
a_k = λ + (1−2λ) k / 256,     c = (1−2λ)/256
```

Since `λ = 0.05`, `s ∈ [0.05, 0.95]` always — no endpoint singularity. The noise
average is a closed-form integral of `log` over an interval, so no sampling is needed:

```
∫_0^1 log(a + c u) du = log a + (1 + a/c) · log1p(c/a) − 1
```

(obtained from `[z log z − z]` and then expanding `log(a+c) = log a + log1p(c/a)`,
which cancels the two nearly equal `O(a log a)` terms the raw antiderivative
difference would subtract). The `log(1−s)` term is the same integral with
`a → 1−a_k` and `c → −c`. Define

```
g(k) = E_u[ log s + log(1−s) | x = k ]
```

a 256-entry table. Then, because `E_u` factorizes over coordinates,

```
S = (1/N) Σ_images Σ_d g(x_d) = (1/N) Σ_{k=0}^{255} n_k g(k)
```

where `n_k` is the count of level `k` over all `N·D` pixels in `data/images.csv`.

## Result

```
S            = −2248.2758955354
D log(1−2λ)  =   −82.6026442757
bpd_A        =     4.3711803308
bpd_B        =     4.2908140260   ← lower / better
```

Model B wins by `0.0804` bpd. Note the comparison reverses the naive reading of the
raw scores: B's reported `−150.0` looks far worse than A's `+1972.0`, but B's density
lives in a logit-stretched space whose volume element is much larger, and once that
is accounted for B is the better model.

## Verification

`solve.py` cross-checks the closed-form `g(k)` against 200-node Gauss–Legendre
quadrature (max abs diff `8.9e-16`) and against 4e5 Monte-Carlo draws per level
(`3.8e-5`), and recomputes `bpd_B` end to end from actually sampled `u` over all
784,000 pixels × 20 repeats (`4.290817` vs exact `4.290814`).
