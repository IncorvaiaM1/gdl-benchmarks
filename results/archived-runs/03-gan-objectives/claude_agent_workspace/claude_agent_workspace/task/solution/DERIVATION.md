# Derivation

Notation: `p = U{0..3}`, `q_k = U{0..k-1}`. The two supports are nested, so
with `s = min(k, 4)` and `L = max(k, 4)` the smaller uniform sits inside the
larger. Set `r = s / L ∈ (0, 1]`. Write the smaller-support density as `1/s` and
the larger one as `1/L`.

## 1. JSD depends only on r, and it is strictly decreasing in r

JSD is symmetric, so it does not matter which of `p` and `q_k` is the
smaller one. On the `s` shared points, `m = (1/s + 1/L)/2 = (L+s)/(2sL)`. On the
`L - s` points covered only by the larger distribution, `m = 1/(2L)`.

- KL(small || m) = log((1/s)/m) = log(2/(1+r)).
- KL(large || m) = r·log((1/L)/m) + (1-r)·log 2 = r·log(2r/(1+r)) + (1-r)·log 2.

Therefore

    JSD = log 2 - (1+r)/2 · log(1+r) + (r/2) · log r.

Then `dJSD/dr = (1/2) log(r/(1+r)) < 0` for every r > 0, so JSD is **strictly
decreasing** in r (and hence injective). That gives
`JSD(p,q_j) = JSD(p,q_k)` **iff** `r(j) = r(k)`.

## 2. LSGAN criterion

For each x, minimizing `p(D-1)^2 + q(D+1)^2` gives `D_LS* = (p-q)/(p+q)`.
Plugging this in gives `V_LS = 1/2 Σ (p-q)^2/(p+q)`, which is half of the triangular
discrimination. For the nested uniforms:

- the shared points contribute `(1-r)^2/(1+r)`;
- the points covered by only one distribution contribute `1-r`.

So `V_LS = 1/2 · [ (1-r)^2/(1+r) + (1-r) ] = (1-r)/(1+r)`. This is also
strictly decreasing in r, so its ties are again **exactly** the pairs with
equal r. For `q_2`, `r = 1/2`, so `V_LS(q_2) = 1/3`.

## 3. Completeness of the tie lists

`r(k) = k/4` for `k ≤ 4`, and `r(k) = 4/k` for `k ≥ 4`. Within each branch, r is
injective in k. So a tie needs `j < 4 < k` with `j/4 = 4/k`, i.e. `jk = 16`.
Checking `j ∈ {1, 2, 3}`:

- `j = 1` gives `k = 16`;
- `j = 2` gives `k = 8`;
- `j = 3` gives `k = 16/3`, which is not an integer.

Hence both lists are exactly `[[1,16],[2,8]]`. `solve.py` checks this by
scanning `k ≤ 400`: it uses exact Fractions for `V_LS` and r, and also compares
direct float JSD values for every pair. This scan is only a sanity check. The
monotonicity argument above is the proof of completeness.

`k_hat = 8`.

## 4. Minimax quantities

The optimal discriminator is `D* = p/(p+q)`. A useful identity is
`V_star = 2·JSD - log 4`, so `V_star` must tie because the JSD values tie.

- **q_2:** `D* = 1/3` on {0,1} and `D* = 1` on {2,3}.
  - `V_star = (1/2)log(1/3) + log(2/3) = log 2 - (3/2)log 3 ≈ -0.9547712524`.
  - `L_MM = log(2/3) ≈ -0.4054651081`.
  - `L_NS = log 3 ≈ 1.0986122887`.
- **q_8:** `D* = 2/3` on {0..3} and `D* = 0` on {4..7}, where `p = 0`.
  - `V_star = log(2/3) + (1/2)log(1/3) = log 2 - (3/2)log 3`, the same as for q_2 (**tie**).
  - `L_MM = (1/2)log(1/3) + (1/2)log 1 = -(1/2)log 3 ≈ -0.5493061443`. This is
    lower than at q_2, so the generator **prefers q_8**.
  - `L_NS = (1/2)log(3/2) + (1/2)·(-log 0) = +inf`, because q_8 puts mass 1/2 on
    points where `D* = 0`. The generator **prefers q_2**.
