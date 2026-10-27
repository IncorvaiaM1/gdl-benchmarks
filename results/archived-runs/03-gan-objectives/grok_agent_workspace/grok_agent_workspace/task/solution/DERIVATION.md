# Derivation

All logarithms are natural. `p` is uniform on `{0,1,2,3}`, so `p(x) = 1/4` there and `0` elsewhere. For each positive integer `k`, `q_k` is uniform on `{0,1,...,k-1}`, so `q_k(x) = 1/k` there and `0` elsewhere. Write `n = 4` for the size of the data support. The two supports meet on `{0,...,min(k,4)-1}`.

## Optimal discriminators

The minimax integrand `p log D + q log(1-D)` is maximized pointwise at

```
D*(x) = p(x) / (p(x) + q(x))
```

on the support of `p+q` (values off that support do not appear in any criterion below).

The least-squares integrand `(1/2) p (D-1)^2 + (1/2) q (D+1)^2` is minimized pointwise at

```
D_LS*(x) = (p(x) - q(x)) / (p(x) + q(x)).
```

Explicitly, on the overlap `D*(x) = k/(k+4)` and `D_LS*(x) = (k-4)/(k+4)`. On data-only points (`k < 4` and `x ∈ {k,...,3}`) one has `D* = 1` and `D_LS* = 1`. On generator-only points (`k > 4` and `x ∈ {4,...,k-1}`) one has `D* = 0` and `D_LS* = -1`.

## Least-squares value

```
V_LS(q) = (1/2) E_p[D_LS*^2] + (1/2) E_q[D_LS*^2]
        = (1/2) sum_{p+q>0} (p-q)^2 / (p+q).
```

For `1 ≤ k ≤ 4` the overlap contributes `(4-k)^2 / (4(k+4))` and the `4-k` data-only points contribute `(4-k)/4`. The sum of those is `2(4-k)/(k+4)`, so

```
V_LS(q_k) = (4-k)/(k+4).
```

For `k ≥ 4` the overlap contributes `(k-4)^2 / (k(k+4))` and the `k-4` generator-only points contribute `(k-4)/k`. The sum is `2(k-4)/(k+4)`, so

```
V_LS(q_k) = (k-4)/(k+4).
```

The two expressions agree at `k = 4`, where `V_LS(q_4) = 0`. In one line,

```
V_LS(q_k) = |k-4| / (k+4).
```

Therefore `V_LS(q_2) = 2/6 = 1/3`.

### Why `ls_tie_pairs` is complete

On the real interval `[4, ∞)` the map `t ↦ (t-4)/(t+4)` has derivative `8/(t+4)^2 > 0`, so it is strictly increasing, and its integer samples are strictly increasing. On `{1,2,3,4}` the values `(4-k)/(k+4)` are strictly decreasing: if `1 ≤ j < k ≤ 4` and the values were equal, clearing denominators would give `j = k`. The same expansion shows that no two distinct integers `≥ 4` can agree. So a tie between distinct positive integers must put one index `j` in `{1,2,3}` and the other index `k > 4`.

Solving `(4-j)/(j+4) = (k-4)/(k+4)`:

```
(4-j)(k+4) = (k-4)(j+4)
k(4-j) - k(j+4) = -4(j+4) - 4(4-j)
-2jk = -32
jk = 16.
```

Thus `k = 16/j`. The only `j ∈ {1,2,3}` for which this is an integer are `j = 1` (`k = 16`) and `j = 2` (`k = 8`). For `j = 3`, `16/3` is not an integer.

Hence the only unordered pairs are `{1,16}` and `{2,8}`.

(The same bound makes the search finite: `V_LS(q_k) > V_LS(q_16) = V_LS(q_1)` for every `k > 16`, and `V_LS(q_1)` is the largest value on `k ≤ 4`. Every tie is therefore among `{1,...,16}`, and the rational values on that range can be compared exactly. The entry-point script does that check.)

## Jensen–Shannon divergence

Let `m = (p+q)/2`. Then `JSD(p,q) = (1/2) KL(p‖m) + (1/2) KL(q‖m)`, with the conventions `log 0 = -∞` and `0·(±∞) = 0`. Here `m` never vanishes on the support of `p` or of `q`, so every JSD below is finite.

For `1 ≤ k ≤ 4`,

```
KL(p‖m) = (k/4) ln(2k/(k+4)) + ((4-k)/4) ln 2,
KL(q‖m) = ln(8/(k+4)),
```

and therefore

```
JSD(p, q_k) = 2 ln 2 + (k/8) ln k - ((k+4)/8) ln(k+4).
```

For `k ≥ 4`,

```
KL(p‖m) = ln(2k/(k+4)),
KL(q‖m) = (4/k) ln(8/(k+4)) + ((k-4)/k) ln 2,
```

and therefore

```
JSD(p, q_k) = (1 + 4/k) ln 2 + (1/2) ln k - ((k+4)/(2k)) ln(k+4).
```

Call the second expression `φ(k)`. It is defined for every real `k > 0`, it agrees with the first expression at `k = 4` (both are `0`, since `q_4 = p`), and direct expansion gives the limit `φ(k) → ln 2` as `k → ∞`.

### The large-side formula is strictly increasing

Differentiate `φ`. With `a(k) = (k+4)/(2k)`,

```
φ(k) = (1 + 4/k) ln 2 + (1/2) ln k - a(k) ln(k+4),
φ'(k) = -4 ln 2 / k^2 + 1/(2k) + (2/k^2) ln(k+4) - 1/(2(k+4)) - 2/(k(k+4)).
```

The common denominator `2k^2(k+4)` is positive for `k > 0`. The numerator simplifies to

```
4(k+4) ln((k+4)/4),
```

because the rational terms `k(k+4) - k^2 - 4k` cancel, leaving

```
4(k+4) ln(k+4) - 8(k+4) ln 2 = 4(k+4) ln((k+4)/4).
```

Hence

```
φ'(k) = 2 ln((k+4)/4) / k^2,
```

which is positive for every `k > 0`. So `φ` is strictly increasing on `[4, ∞)`. In particular the integers `k ≥ 4` all have distinct Jensen–Shannon values, and each number in `(0, ln 2)` is attained at most once.

### Identity that produces the ties

For every real `j ∈ (0, 4]`, `φ(16/j)` equals the small-side formula at `j`. Substituting `k = 16/j`:

```
1 + 4/k = (j+4)/4,
(1/2) ln k = 2 ln 2 - (1/2) ln j,
(k+4)/(2k) = (j+4)/8,
ln(k+4) = 2 ln 2 + ln(j+4) - ln j.
```

The coefficient of `ln 2` coming from `(1+4/k) ln 2` is `(j+4)/4`, and the piece `((j+4)/8)·2 ln 2` of the last term cancels it. The coefficient of `ln j` that remains is

```
-1/2 + (j+4)/8 = j/8.
```

So

```
φ(16/j) = 2 ln 2 + (j/8) ln j - ((j+4)/8) ln(j+4),
```

which is exactly `JSD(p, q_j)` for every integer `j ∈ {1,2,3,4}`.

Therefore, whenever `16/j` is an integer, `JSD(p, q_j) = JSD(p, q_{16/j})`. The divisors of `16` that lie in `{1,2,3,4}` are `1`, `2`, and `4`. They give the partners `16`, `8`, and `4`. Dropping the non-distinct pair `{4,4}` leaves `{1,16}` and `{2,8}`.

### Why `jsd_tie_pairs` is complete

- If `4 ≤ j < k`, then `φ(j) < φ(k)`, so those Jensen–Shannon values differ.
- If `j ∈ {1,2,3}` and `k ≥ 4`, the equation `φ(k) = JSD(p, q_j)` has the unique real solution `k = 16/j`, because `φ` is strictly increasing and `JSD(p, q_j) = φ(16/j)`. That solution is an integer only for `j = 1` and `j = 2`. For `j = 3` one has `5 < 16/3 < 6`, so `JSD(p, q_3)` sits strictly between `JSD(p, q_5)` and `JSD(p, q_6)` and matches no positive integer `k ≠ 3`.
- If `1 ≤ j < k ≤ 4`, the four values are `φ(16)`, `φ(8)`, `φ(16/3)`, and `φ(4)`. These arguments are strictly ordered, so the four divergences are strictly ordered and there is no tie inside `{1,2,3,4}`.
- `JSD(p, q_4) = 0`, while `φ(k) > 0` for `k > 4` and `JSD(p, q_j) = φ(16/j) > 0` for `j < 4`. Nothing else matches `k = 4`.

Every positive integer falls in one of these cases. The only unordered pairs of distinct positive integers are `{1,16}` and `{2,8}`.

## The partner of `k = 2`

The unique positive integer `k ≠ 2` with `JSD(p, q_k) = JSD(p, q_2)` is `k = 8`. So `k_hat = 8`.

## Minimax, minimax-generator, and non-saturating losses

`V_star(q) = E_p[ln D*] + E_q[ln(1-D*)]`, `L_MM(q) = E_q[ln(1-D*)]`, and `L_NS(q) = E_q[-ln D*]`.

For `k = 2`, `D* = 1/3` on the support of `q_2` and `D* = 1` on `{2,3}`:

```
L_MM(q_2) = ln(2/3) = ln 2 - ln 3,
L_NS(q_2) = -ln(1/3) = ln 3,
V_star(q_2) = (1/2) ln(1/3) + ln(2/3) = ln 2 - (3/2) ln 3.
```

For `k = 8`, `D* = 2/3` on `{0,1,2,3}` and `D* = 0` on `{4,5,6,7}`:

```
L_MM(q_8) = (4/8) ln(1/3) + (4/8) ln 1 = -(1/2) ln 3,
V_star(q_8) = ln(2/3) + (1/2) ln(1/3) = ln 2 - (3/2) ln 3.
```

`L_NS(q_8) = +∞`. Indeed `q_8({4,5,6,7}) = 1/2 > 0` and `-ln D* = +∞` on that set. The problem's convention says an expectation that puts positive probability on an integrand equal to `+∞` is `+∞`. (The same points do not make `V_star` infinite: they contribute `ln(1-D*) = ln 1 = 0` to the generator term of `V`.)

As a check, `V_star = -ln 4 + 2 JSD`. Both indices have the same Jensen–Shannon divergence `(3/4) ln(4/3)`, and

```
-ln 4 + 2·(3/4) ln(4/3) = ln 2 - (3/2) ln 3,
```

which matches both calculations of `V_star`.

## Which distribution the generator prefers

The generator wants each scalar to be smaller.

- `V_star(q_2) = V_star(q_8)`, so `V_star` is a tie.
- `L_MM(q_8) - L_MM(q_2) = -(1/2) ln 3 - (ln 2 - ln 3) = (1/2) ln(3/4) < 0`, because `3 < 4`. So `L_MM(q_8) < L_MM(q_2)`, and the generator prefers `q_k_hat`.
- `L_NS(q_2) = ln 3 < +∞ = L_NS(q_8)`, so the generator prefers `q_2`.

## Reported decimals

`answer.json` records `1/3` and the three logarithms above, rounded half-even to 16 digits after the decimal point. Those expansions are produced by `solution/solve.py` from the same closed forms, after the script has checked them against the definitions on the supports.
