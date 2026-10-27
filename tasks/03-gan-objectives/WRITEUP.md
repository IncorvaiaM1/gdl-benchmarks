# GDL-Bench Submission #3 (not part of the agent-facing repo; leave out of the zip)

## a. Challenge problem

See `task/TASK.md` (verbatim problem). No data files are needed.

Short version: `p` is uniform on {0,1,2,3} and `q_k` is uniform on
{0,...,k-1}. Find every pair of distinct `k` that tie under JSD, and every
pair that tie under the LSGAN (a=-1, b=1, c=0) generator criterion. Take the
JSD partner `k_hat` of `k = 2`. At the optimal discriminator, compare `q_2`
and `q_{k_hat}` under three generator objectives: the value of the minimax
game `V_star`, the minimax generator's own term `L_MM`, and the non-saturating
loss `L_NS`.

## b. Gold answer and solution

| Key | Gold |
|---|---|
| jsd_tie_pairs | [[1, 16], [2, 8]] |
| ls_tie_pairs | [[1, 16], [2, 8]] |
| ls_value_q2 | 1/3 |
| k_hat | 8 |
| q2: V_star, L_MM, L_NS | log 2 − (3/2) log 3 = −0.954771, log(2/3) = −0.405465, log 3 = 1.098612 |
| q8: V_star, L_MM, L_NS | log 2 − (3/2) log 3 = −0.954771, −(1/2) log 3 = −0.549306, +inf |
| prefers | V_star: tie; L_MM: q8; L_NS: q2 |

**Both divergences depend only on a support ratio.** `p` and `q_k` are
nested uniforms. Let the smaller one have support size `n` and the larger
`N`, with `r = N/n = max(k,4)/min(k,4) >= 1`. On the shared support, the
mixture `m` equals `(r+1)/(2N)`; off it, `m = 1/(2N)`. So

```
JSD(r) = log 2 + (1/2) log r − (1/2)(1 + 1/r) log(1 + r),
dJSD/dr = log(1 + r) / (2 r^2) > 0.
```

JSD is symmetric, so it doesn't matter which of `p`, `q_k` is the larger one.
It is a strictly increasing function of `r` alone, so `JSD(p,q_j) = JSD(p,q_k)`
iff `r_j = r_k`. For `j != k`, that requires one value below 4 and one above,
with `4/j = k/4`, i.e. `jk = 16`. The only integer solutions are {1,16} and
{2,8}. This argument covers all positive integers, so the list is complete.

**LSGAN.** Minimizing `(1/2)p(D−1)^2 + (1/2)q(D+1)^2` pointwise gives
`D_LS* = (p − q)/(p + q)`. Then
`V_LS = (1/2) Σ (p − q)^2/(p + q) = (r − 1)/(r + 1)`, which is also strictly
increasing in `r` alone. That gives the same pairs, and `V_LS(q_2) = 1/3`.

**Why they tie.** An f-divergence depends only on how the likelihood ratio
`p/q` is distributed. The pair (p, q_2) has the same ratio structure as
(q_8, p): a uniform against a uniform with half its support. So
`D_f(p || q_2) = D_g(p || q_8)` with the reversed generator
`g(u) = u f(1/u)`. Every symmetric f-divergence has to tie on this pair. Only
asymmetric ones can tell mode dropping (q_2) apart from over-spreading (q_8).

**GAN terms.** `D* = p/(p + q)`.

- q_2: on {0,1}, D* = 1/3; on {2,3}, D* = 1 (q = 0 there).
  V_star = (1/2) log(1/3) + log(2/3) = log 2 − (3/2) log 3;
  L_MM = log(2/3); L_NS = log 3.
- q_8: on {0..3}, D* = 2/3; on {4..7}, D* = 0.
  V_star = log(2/3) + (1/2) log(1/3) + (1/2) log 1 = log 2 − (3/2) log 3;
  L_MM = (1/2) log(1/3) = −(1/2) log 3;
  L_NS = (1/2) log(3/2) + (1/2)(+inf) = +inf.

`V_star = 2 JSD − log 4` ties, as expected. The generator's own minimax term
does **not** tie, because the `E_p[log D*]` term also depends on `q` through
`D*`, and it favors q_8. The non-saturating loss is infinite for any
generator that puts mass where the data has none, so it favors q_2, the
generator that drops modes. One game value gives three different verdicts.

Verification: `gt_code/reference_solution.py` computes every value directly
(exact Fractions for everything without a log), scans k = 1..512 for ties,
and asserts agreement with the closed form and with V_star = 2 JSD − log 4.
`tests/test_solution.py` passes 8/8 on the gold answer.

**Wrong answers the tests reject:**

| Mistake | Fails |
|---|---|
| Only report {2,8} (stop at the obvious pair, or not think to search below 4) | test_jsd_tie_pairs |
| Treat L_MM as the same as V_star (claim L_MM ties) | test_q_k_hat_terms, test_prefers |
| Assume non-saturating has the same optimum, so "tie" | test_prefers |
| Drop the 1/2 in V_LS (report 2/3) | test_ls_value_q2 |
| Finite L_NS at q_8 (ignore D* = 0 off the data support) | test_q_k_hat_terms |

## c. What does the problem test?

- **Topic:** GAN objectives and divergences (minimax / JSD, non-saturating
  loss, LSGAN / Pearson chi-square), mode collapse vs mode covering.
- **Problem type:** Derivation + conceptual reasoning + counterexample.
- **Core insight:** Symmetric f-divergences such as JSD and the LSGAN
  chi-square depend only on the support ratio, so they cannot distinguish
  dropping half the modes from spreading over twice the support. The
  involution k -> 16/k gives exactly the ties. A solver with only surface
  knowledge will repeat "GANs/JSD penalize mode collapse", and will treat the
  game value, the minimax generator term, and the non-saturating loss as
  interchangeable. In fact they give three different preferences on the
  same pair.

## d. AI evaluation

All three models were run as Cursor agents, each given only `task/` and the
fixed prompt in `agent_runs/PROMPT.md`. Each `answer.json` was graded with
`tests/test_solution.py` (output saved as `verifier_output.txt` in each
workspace folder under `agent_runs/`).

| Model / version | Result | Notes (1-3 sentences) |
|---|---|---|
| Claude Opus 5 (Cursor agent) | Correct (8/8) | Showed both criteria depend only on r = min(k,4)/max(k,4) and are monotone in it, which reduces ties to jk = 16, and noted V_star = 2 JSD - log 4. It cross-checked by scanning k <= 400 with exact Fractions. |
| GPT-5.6-sol (Cursor agent) | Correct (8/8) | Gave the same reduction to u = min(t, 1/t) with an explicit derivative, and the cleanest written derivation of the three. It ran its script in a fresh venv. |
| Grok 4.6 (Cursor agent) | Correct (8/8) | The most thorough: separate closed forms on each side of k = 4, an explicit identity phi(16/j) = JSD(p, q_j), and exact comparisons over k <= 16. It explicitly declined to read anything outside the workspace. |

All three models found the {1,16} pair and proved the tie lists complete. All
three also separated V_star, L_MM and L_NS correctly. None fell into the
anticipated traps (stopping at {2,8}, treating L_MM like V_star, assuming
non-saturating has the same optimum). As with submission #2, current agentic
models with code execution handle closed-form divergence calculations
reliably. The problem is best seen as a clean, verifiable check of the
symmetric-divergence insight and the three generator objectives, not as an
item that breaks models.
