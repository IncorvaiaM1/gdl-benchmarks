# Task: Which generator does each GAN objective prefer?

## Setup

All logarithms are natural logs. All distributions live on the nonnegative
integers {0, 1, 2, ...}.

- The data distribution `p` is uniform on {0, 1, 2, 3}.
- For each positive integer `k`, the generator distribution `q_k` is uniform
  on {0, 1, ..., k-1}.

Conventions: `log 0 = -inf`, `0 * (±inf) = 0`, and an expectation that puts
positive probability on an integrand equal to `+inf` is `+inf`.

## Definitions

**Jensen-Shannon divergence.**
`JSD(p, q) = 1/2 KL(p || m) + 1/2 KL(q || m)`, with `m = (p + q)/2`.

**Minimax GAN.** For a discriminator `D` mapping integers to [0, 1], let

```
V(D, q) = E_{x~p}[ log D(x) ] + E_{x~q}[ log(1 - D(x)) ].
```

Let `D*_q` be the discriminator that maximizes `V(D, q)` (its values off the
support of `p + q` do not matter). Define, for a generator distribution `q`:

- `V_star(q) = V(D*_q, q)`
- `L_MM(q)   = E_{x~q}[ log(1 - D*_q(x)) ]`   (the minimax generator's own term)
- `L_NS(q)   = E_{x~q}[ -log D*_q(x) ]`       (the non-saturating generator loss)

**Least-squares GAN** with labels `a = -1` (generated), `b = 1` (real), and
generator target `c = 0`. The discriminator `D_LS*_q` minimizes

```
1/2 E_{x~p}[ (D(x) - 1)^2 ] + 1/2 E_{x~q}[ (D(x) + 1)^2 ]
```

over all real-valued `D`, and the generator criterion is

```
V_LS(q) = 1/2 E_{x~p}[ (D_LS*_q(x) - 0)^2 ] + 1/2 E_{x~q}[ (D_LS*_q(x) - 0)^2 ].
```

The generator wants each of `V_star`, `L_MM`, `L_NS`, and `V_LS` to be as
small as possible.

## What you must report

1. `jsd_tie_pairs`: every unordered pair {j, k} of **distinct** positive
   integers with `JSD(p, q_j) = JSD(p, q_k)` exactly.
2. `ls_tie_pairs`: every unordered pair {j, k} of distinct positive integers
   with `V_LS(q_j) = V_LS(q_k)` exactly.
3. `ls_value_q2`: the value of `V_LS(q_2)`.
4. `k_hat`: the integer `k_hat != 2` such that {2, k_hat} is in
   `jsd_tie_pairs`, or `null` if there is none.
5. If `k_hat` exists: the values of `V_star`, `L_MM`, and `L_NS` at `q_2`
   and at `q_{k_hat}`.
6. If `k_hat` exists: for each of `V_star`, `L_MM`, `L_NS`, which of `q_2`
   and `q_{k_hat}` the generator prefers (lower value), or `"tie"`.

## Deliverables

- Code in `task/solution/` with a single entry-point script that reproduces
  your answer when run.
- `task/solution/DERIVATION.md`: a short derivation. In particular, justify
  why your lists in items 1 and 2 are **complete** (no pair is missing).
- `task/results/answer.json`, a JSON object with **exactly** these keys:

  ```json
  {
    "jsd_tie_pairs": [[j, k], ...],
    "ls_tie_pairs":  [[j, k], ...],
    "ls_value_q2":   <float>,
    "k_hat":         <int or null>,
    "q2":       {"V_star": <float>, "L_MM": <float>, "L_NS": <float or "inf">},
    "q_k_hat":  {"V_star": <float>, "L_MM": <float>, "L_NS": <float or "inf">} or null,
    "prefers":  {"V_star": "q2" | "q_k_hat" | "tie",
                 "L_MM":   "q2" | "q_k_hat" | "tie",
                 "L_NS":   "q2" | "q_k_hat" | "tie"} or null
  }
  ```

  Write each pair as `[smaller, larger]` and sort the list by the first
  element. Write `+inf` as the string `"inf"` and `-inf` as `"-inf"`; every
  other value must be a finite number. Report floats to at least
  10 significant figures.

## Prohibited

- Do not read anything under `tests/` or `gt_code/`.
- Do not search the internet for the answer to this specific task. General
  references on GANs, f-divergences, and LSGAN are fine.

## Acceptance criteria (for your awareness; grading is automated)

`tests/test_solution.py` recomputes every value. Pair lists, `k_hat`, the
`"inf"` / `"-inf"` strings, and every `prefers` entry must match exactly.
Floats must be within 1e-8 absolute.
