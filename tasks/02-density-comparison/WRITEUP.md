# GDL-Bench Submission #2 (not part of the agent-facing repo; leave out of the zip)

## a. Challenge problem

See `task/TASK.md` (verbatim problem) with `data/images.csv` (1000 MNIST
test images, raw 8-bit pixels).

Short version: three MNIST models report one number each, computed with the
preprocessing code shown. A: a flow on `(255x + u)/256` scoring 1972.0 nats.
B: a logit-space flow whose pipeline is `x/255 + u/256` then
`logit(λ + (1−2λ)y)`, λ = 0.05, scoring −150.0 nats. C: a VAE on dynamically
binarized, zero-padded 32×32 images with mean loss 93.7 nats. Put A and B on
a common bits/dim bound (if valid), find the λ range where B's pipeline is
defined and the break-even λ, and say what C's number means.

The numbers and pipelines come from our Problem 4 runs (NICE and VAE in
`pytorch-generative`); Model B is a hypothetical variant with a common
`/255` preprocessing slip.

## b. Gold answer and solution

| Key | Gold |
|---|---|
| bpd_A | 4.371180 |
| bpd_B | 4.283594 |
| better | B |
| lambda_min | 1/258 = 0.00387597 |
| lambda_star | 0.053360 |
| bpd_C_own | 0.132012 |
| bpd_C_8bit | null |

**Common bound.** If a model's preprocessing sends image `x` to a set of
volume `V_x` in the space `y` where the (pre-logit) density lives, then
`P(x) = ∫ p_Y = V_x · E_u[p_Y(y)]`, and by Jensen
`−log P(x) ≤ −E_u log p_Y(y) − log V_x`. For A and B, `V_x = 256^{−784}`,
so `bpd = 8 − E[log p_Y] / (784 ln 2)`.

**A.** `y = (x+u)/256`; bins tile [0,1). `bpd_A = 8 − 1972/(784 ln 2) = 4.371180`.

**B, validity.** `y = k/255 + u/256`. Bins have width 1/256 but are spaced
1/255 apart, so they are disjoint with small gaps; the induced `Σ_x P(x) < 1`
(also true for A, since the flows put mass outside the bins). A deficient
model still gives a valid Jensen bound, and each bin still has volume
`256^{−784}`, so the bound is valid and uses `D log 256` (not `D log 255`).

**B, value.** `log p_Y = log p_B(y') + Σ_d [log(1−2λ) − log s_d − log(1−s_d)]`,
`s = λ + (1−2λ)(k/255 + u/256)`. The Jacobian depends on each pixel value and
on `u`. Per pixel value `k`, `s` is uniform on `[a, a+c]` with
`a = λ + (1−2λ)k/255`, `c = (1−2λ)/256`, so `E log s = [F(a+c) − F(a)]/c`,
`F(t) = t log t − t` (same for `1 − s`). Summing over the pixel-value
histogram gives `J = 2169.597` nats per image, so
`bpd_B = 8 − (−150 + 2169.597)/(784 ln 2) = 4.283594`. B is better.

**lambda_min.** Pixel 255 with `u → 1` gives `y → 1 + 1/256`. Finite logit
needs `λ + (1−2λ)(1 + 1/256) ≤ 1` (strict `s < 1` then holds since `u < 1`),
i.e. `λ ≥ 1/258`. The low end `s ≥ λ > 0` is never binding.

**lambda_star.** `bpd_B(λ)` is increasing on `[1/258, 1/2)` (both
`log(1−2λ)` and `−log s(1−s)` shrink as λ grows), from 1.39 to +∞ bits/dim, so
the root of `bpd_B(λ) = bpd_A` is unique: `λ* = 0.053360`. B only wins
because its λ is a bit below the break-even point.

**C.** The loss is `−ELBO ≥ −log p_C(b)` for the binary 32×32 variable `b`,
so in its own units `bpd_C_own = 93.7/(1024 ln 2) = 0.132012`. It is not a
statement about `P(x)` for the 8-bit image: `b` is a random binarization
(many 8-bit images map to the same `b`, and each image maps to many `b`),
so no bound on the 8-bit NLL follows. `bpd_C_8bit = null`. Comparing 0.13 to
4.3 bits/dim is meaningless.

Verification: `gt_code/reference_solution.py` (closed form). An independent
Monte Carlo solve (10 noise draws per pixel, bisection for λ*) matches every
value within tolerance and passes all 8 tests.

**Wrong answers the tests reject:**

| Mistake | Effect |
|---|---|
| Use the correct `/256` pipeline for B (miss the slip) | bpd_B = 4.290814, λ* = 0.053092 |
| Use `D log 255` because of the `/255` | bpd_B = 4.277947 |
| Declare B's bound invalid because bins don't tile | bpd_B = null |
| Divide C's loss by 784 (ignore padding) | bpd_C_own = 0.172415 |
| Report C's number as an 8-bit bpd | bpd_C_8bit ≠ null |
| Compare raw scores / constant logit Jacobian | wrong `better`, bpd_B far off |

## c. What does the problem test?

- **Topic:** Normalizing flows, change of variables and dequantization;
  VAEs and the ELBO; likelihood evaluation of generative models on discrete
  data.
- **Problem type:** Coding/debugging (reading preprocessing code) +
  derivation + numerical.
- **Core insight:** A reported likelihood is a statement about a specific
  random variable produced by a specific pipeline. The solver has to read
  each pipeline, push the logit map through change of variables with a
  data-dependent Jacobian, realize that a `/255` vs `/256` mismatch leaves
  the bound valid but changes both the Jacobian and the domain of λ, and
  see that a VAE's ELBO on binarized, padded images says nothing about 8-bit
  likelihood. A solver who only knows the formulas will treat all three
  numbers as interchangeable "NLLs", or will copy the textbook 256
  conversion without checking what the code actually does.

## d. AI evaluation

**Revision note.** A first version (v1, archived in
`gdlbench_submission_v1/` and `agent_runs_v1/`) asked only for bpd_A,
bpd_B and the verdict, with the pipelines stated in math rather than code.
Claude, GPT and Grok all solved v1 correctly. Its statement also gave away
most of the reasoning, so it was revised into this version before the
final evaluation below.

| Model / version | Result | Notes (1-3 sentences) |
|---|---|---|
| Claude Opus 5 (Cursor agent) | Correct | All 7 fields match gold. Explicitly argued that the /255 bins are disjoint but leave gaps (deficient model, bound still valid), derived lambda_min = 1/258, and verified its closed form against mpmath quadrature and Monte Carlo. |
| GPT-5.6-sol (Cursor agent) | Correct | All 7 fields match gold. Correct reasoning for C (many-to-one stochastic binarization gives no 8-bit bound), but gave only a very brief derivation. |
| Grok 4.6 (Cursor agent) | Correct | All 7 fields match gold. Correctly used k/255 + u/256 for B, 1024 dims for C, and the 1/258 finiteness condition. |

All three agents answered every field correctly (bpd values agree with gold to about 1e-13). None fell into the anticipated traps (/256 for B, D log 255, dividing C by 784, reporting an 8-bit bound for C). Current agentic models with code execution handle this kind of careful pipeline bookkeeping well; the problem is best viewed as a clean, verifiable check of that skill rather than a model-breaking item.
