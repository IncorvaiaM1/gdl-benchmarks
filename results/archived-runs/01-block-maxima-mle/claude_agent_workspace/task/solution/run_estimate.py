r"""Entry point: estimate lambda from data/train.csv and write task/results/estimate.json.

Usage (from anywhere; paths are resolved relative to this file):

    python task/solution/run_estimate.py

What it does, in order:

1. Verifies the SHA-256 of ``data/train.csv`` against the value published in
   ``data/DOWNLOAD.md``. Refuses to continue on a mismatch, since every digit
   below is a property of that exact file.
2. Loads the ``y`` column and sanity-checks it (row count, strict positivity).
3. Maximises the block-maxima log-likelihood derived in ``blockmax_exp_mle.py``
   three independent ways -- scipy Brent root-find on the score, a stdlib-only
   bisection+Newton written against a different algebraic form of the score,
   and a 50-digit mpmath Newton solve straight from the file's decimal strings
   -- and asserts they agree to 1e-12 relative. The task's tolerance is 3e-4
   relative, so this leaves eight orders of magnitude of headroom.
4. Cross-checks that the maximiser really is a maximiser (score ~ 0, negative
   curvature, log-likelihood higher than at nearby lambdas) and that the naive
   "pretend the data are Exponential" estimator is *rejected*, since that is
   the failure mode this task is built around.
5. Writes ``task/results/estimate.json`` as ``{"lambda_hat": <float>}`` with
   full float64 precision (17 significant digits, well past the 8 required).

Deterministic: no RNG anywhere, so there is no seed to record.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blockmax_exp_mle import (  # noqa: E402
    harmonic_number,
    load_column_y,
    load_column_y_as_strings,
    log_likelihood,
    mle_mpmath,
    mle_scipy_brentq,
    mle_stdlib,
    moment_estimate,
    score,
    score_derivative,
    sha256_of_file,
    standard_error,
)

M = 4  # given exactly by the task statement
EXPECTED_ROWS = 2000  # per data/DOWNLOAD.md
AGREEMENT_RTOL = 1e-12  # cross-solver agreement we demand of ourselves
MP_DPS = 50  # digits for the high-precision re-solve

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TRAIN_CSV = REPO / "data" / "train.csv"
DOWNLOAD_MD = REPO / "data" / "DOWNLOAD.md"
RESULTS_JSON = REPO / "task" / "results" / "estimate.json"


def published_sha256() -> str:
    """Pull the expected train.csv hash out of data/DOWNLOAD.md."""
    text = DOWNLOAD_MD.read_text(encoding="utf-8")
    hashes = re.findall(r"\b[0-9a-f]{64}\b", text)
    if len(hashes) != 1:
        raise RuntimeError(
            f"expected exactly one sha256 in {DOWNLOAD_MD}, found {len(hashes)}"
        )
    return hashes[0]


def relative_gap(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def main() -> int:
    print("=" * 72)
    print("MLE of Exponential rate from block maxima of m = %d draws" % M)
    print("=" * 72)

    # -- 1. data integrity ------------------------------------------------- #
    expected = published_sha256()
    actual = sha256_of_file(TRAIN_CSV)
    print(f"\n[1] data/train.csv")
    print(f"    sha256 expected : {expected}")
    print(f"    sha256 actual   : {actual}")
    if actual != expected:
        print("    MISMATCH -- refusing to estimate from unexpected data.")
        return 1
    print("    match: OK")

    # -- 2. load and sanity-check ------------------------------------------ #
    y = load_column_y(TRAIN_CSV)
    n = y.size
    print(f"\n[2] sample")
    print(f"    n               : {n}")
    print(f"    min / max       : {y.min():.6f} / {y.max():.6f}")
    print(f"    mean            : {y.mean():.10f}")
    if n != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} rows, got {n}")
    if not np.all(np.isfinite(y)) or not np.all(y > 0.0):
        raise RuntimeError("y must be finite and strictly positive")

    # -- 3. three independent solves --------------------------------------- #
    lam_brentq = mle_scipy_brentq(y, M)
    lam_stdlib = mle_stdlib([float(v) for v in y], M)
    lam_mp = mle_mpmath(load_column_y_as_strings(TRAIN_CSV), M, dps=MP_DPS)
    lam_mp_float = float(lam_mp)

    print(f"\n[3] maximum likelihood estimates")
    print(f"    scipy brentq    : {lam_brentq!r}")
    print(f"    stdlib bis+newt : {lam_stdlib!r}")
    import mpmath as mp

    print(f"    mpmath ({MP_DPS} dps) : {mp.nstr(lam_mp, 30, strip_zeros=False)}")
    print(f"    |brentq - stdlib| / stdlib = {relative_gap(lam_brentq, lam_stdlib):.3e}")
    print(f"    |brentq - mpmath| / mpmath = {relative_gap(lam_brentq, lam_mp_float):.3e}")
    for name, value in (("stdlib", lam_stdlib), ("mpmath", lam_mp_float)):
        gap = relative_gap(lam_brentq, value)
        if not gap <= AGREEMENT_RTOL:
            raise RuntimeError(
                f"solver disagreement: brentq vs {name} relative gap {gap:.3e} "
                f"exceeds {AGREEMENT_RTOL:.0e}"
            )
    print(f"    all solvers agree to <= {AGREEMENT_RTOL:.0e} relative: OK")

    lam_hat = lam_brentq

    # -- 4. verify it is the maximum, and that the naive estimator is not -- #
    s = score(lam_hat, y, M)
    curvature = score_derivative(lam_hat, y, M)
    se = standard_error(lam_hat, y, M)
    ll = log_likelihood(lam_hat, y, M)
    print(f"\n[4] optimality and diagnostics")
    print(f"    score l'(lam)   : {s:+.6e}   (0 at the MLE)")
    print(f"    curvature l''   : {curvature:+.6e}   (< 0 => maximum)")
    print(f"    logL(lam_hat)   : {ll:.10f}")
    print(f"    asymptotic SE   : {se:.8f}")
    print(f"    95% Wald CI     : [{lam_hat - 1.959964 * se:.8f}, "
          f"{lam_hat + 1.959964 * se:.8f}]")
    if curvature >= 0.0:
        raise RuntimeError("curvature is not negative at the reported point")
    if abs(s) > 1e-6 * abs(curvature) * lam_hat:
        raise RuntimeError(f"score {s:.3e} is not acceptably close to zero")
    for eps in (1e-6, 1e-4, 1e-2):
        for signed in (-eps, eps):
            neighbour = lam_hat * (1.0 + signed)
            if log_likelihood(neighbour, y, M) >= ll:
                raise RuntimeError(
                    f"log-likelihood at lam_hat*(1{signed:+g}) is not lower"
                )
    print("    logL is strictly lower at lam_hat*(1 +/- 1e-6, 1e-4, 1e-2): OK")

    lam_mom = moment_estimate(y, M)
    lam_naive = 1.0 / float(y.mean())  # wrong model: y treated as Exponential
    print(f"\n[5] comparison estimators (context, not the answer)")
    print(f"    H_m = H_{M}         : {harmonic_number(M):.10f}")
    print(f"    moments H_m/ybar: {lam_mom!r}   "
          f"(rel. gap to MLE {relative_gap(lam_mom, lam_hat):.3e})")
    print(f"    naive 1/ybar    : {lam_naive!r}   "
          f"(rel. gap to MLE {relative_gap(lam_naive, lam_hat):.3e})  <-- WRONG MODEL")
    if relative_gap(lam_naive, lam_hat) < 0.1:
        raise RuntimeError(
            "naive 1/mean(y) is suspiciously close to the MLE; check m and the model"
        )

    # -- 5. write the answer ----------------------------------------------- #
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {"lambda_hat": lam_hat}
    RESULTS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    reread = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    if set(reread) != {"lambda_hat"}:
        raise RuntimeError(f"estimate.json must have exactly one key, got {set(reread)}")
    round_tripped = float(reread["lambda_hat"])
    if not math.isfinite(round_tripped) or round_tripped <= 0.0:
        raise RuntimeError("lambda_hat must be finite and strictly positive")
    if round_tripped != lam_hat:
        raise RuntimeError("estimate.json lost precision on round-trip")

    print(f"\n[6] wrote {RESULTS_JSON.relative_to(REPO).as_posix()}")
    print(f"    {json.dumps(payload)}")
    print(f"    lambda_hat to 12 s.f. : {lam_hat:.11e}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
