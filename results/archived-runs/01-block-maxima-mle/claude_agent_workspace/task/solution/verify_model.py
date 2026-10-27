r"""Independent validation of the derivation, not just of the arithmetic.

``run_estimate.py`` already checks that three solvers find the same root of the
same score function. That would still agree with itself if the score function
were derived from the wrong model, so this script attacks the derivation from
outside the algebra:

A. **Goodness of fit.** Kolmogorov-Smirnov test of ``data/train.csv`` against
   the fitted CDF ``F(y) = (1 - exp(-lam_hat*y))**4``, plus the same test
   against the naive Exponential fit. If ``m = 4`` block maxima is the right
   model, the first should not be rejected and the second should be crushed.

B. **Simulation recovery.** Simulate many synthetic datasets of the same size
   from ``max of 4 i.i.d. Exponential(lam_0)`` at ``lam_0 = lam_hat``, re-run
   the estimator on each, and check that (i) the estimates centre on
   ``lam_0`` to within Monte-Carlo error, and (ii) their spread matches the
   asymptotic standard error that ``run_estimate.py`` reports from the observed
   information. An incorrectly derived likelihood fails (i) with a bias many
   Monte-Carlo standard errors wide -- for reference, the script also shows
   what the naive ``1/mean(y)`` estimator does on the very same samples.

This uses an RNG, so the seed is fixed and printed. Usage:

    python task/solution/verify_model.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blockmax_exp_mle import (  # noqa: E402
    load_column_y,
    mle_scipy_brentq,
    standard_error,
)

M = 4
SEED = 20260906
N_REPLICATES = 500

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TRAIN_CSV = REPO / "data" / "train.csv"
RESULTS_JSON = REPO / "task" / "results" / "estimate.json"


def main() -> int:
    from scipy import stats

    lam_hat = float(json.loads(RESULTS_JSON.read_text(encoding="utf-8"))["lambda_hat"])
    y = load_column_y(TRAIN_CSV)
    n = y.size
    print(f"lam_hat from estimate.json : {lam_hat!r}")
    print(f"n = {n}, m = {M}, seed = {SEED}, replicates = {N_REPLICATES}")

    # -- A. goodness of fit ------------------------------------------------ #
    fitted_cdf = lambda t: (1.0 - np.exp(-lam_hat * t)) ** M  # noqa: E731
    ks_block = stats.kstest(y, fitted_cdf)
    lam_naive = 1.0 / float(y.mean())
    ks_expon = stats.kstest(y, lambda t: 1.0 - np.exp(-lam_naive * t))

    print("\n[A] Kolmogorov-Smirnov goodness of fit on data/train.csv")
    print(f"    max-of-{M} Exponential(lam_hat) : D = {ks_block.statistic:.6f}, "
          f"p = {ks_block.pvalue:.6f}")
    print(f"    plain Exponential(1/ybar)     : D = {ks_expon.statistic:.6f}, "
          f"p = {ks_expon.pvalue:.3e}")
    if ks_block.pvalue < 0.01:
        raise RuntimeError("fitted block-maxima model is rejected at the 1% level")
    if ks_expon.pvalue > 1e-6:
        raise RuntimeError("plain Exponential is not clearly rejected -- check m")
    print("    block-maxima fit not rejected, plain Exponential decisively "
          "rejected: OK")

    # -- B. simulation recovery -------------------------------------------- #
    rng = np.random.default_rng(SEED)
    mle_draws = np.empty(N_REPLICATES)
    naive_draws = np.empty(N_REPLICATES)
    for b in range(N_REPLICATES):
        x = rng.exponential(scale=1.0 / lam_hat, size=(n, M))
        y_sim = x.max(axis=1)
        mle_draws[b] = mle_scipy_brentq(y_sim, M)
        naive_draws[b] = 1.0 / y_sim.mean()

    mle_mean = mle_draws.mean()
    mle_sd = mle_draws.std(ddof=1)
    mc_se = mle_sd / math.sqrt(N_REPLICATES)
    bias_in_mc_se = (mle_mean - lam_hat) / mc_se
    asymptotic_se = standard_error(lam_hat, y, M)

    print(f"\n[B] recovery of lam_0 = lam_hat = {lam_hat:.10f} from simulated data")
    print(f"    correct MLE   mean {mle_mean:.10f}  sd {mle_sd:.8f}  "
          f"bias {mle_mean - lam_hat:+.2e} = {bias_in_mc_se:+.2f} MC SE")
    print(f"    naive 1/ybar  mean {naive_draws.mean():.10f}  "
          f"bias {naive_draws.mean() - lam_hat:+.4f}  <-- WRONG MODEL")
    print(f"    simulated sd {mle_sd:.8f}  vs  observed-information SE "
          f"{asymptotic_se:.8f}  (ratio {mle_sd / asymptotic_se:.4f})")

    if abs(bias_in_mc_se) > 4.0:
        raise RuntimeError(
            f"estimator does not recover the generating lambda: bias is "
            f"{bias_in_mc_se:+.2f} Monte-Carlo standard errors"
        )
    if not 0.9 <= mle_sd / asymptotic_se <= 1.1:
        raise RuntimeError(
            "simulated spread disagrees with the observed-information SE by >10%"
        )
    print("    unbiased to within 4 MC SE and SE matches to within 10%: OK")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
