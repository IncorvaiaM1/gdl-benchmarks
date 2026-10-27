"""MLE of Exponential(λ) rate from block-maxima with known block size m=4.

Each observation y is max(X_1,...,X_m) with X_i i.i.d. Exponential(λ)
(rate parameterization): CDF F(x)=1-exp(-λx), PDF f(x)=λ exp(-λx).

The block-maximum then has
    G(y) = (1 - exp(-λy))^m
    g(y) = m λ exp(-λy) (1 - exp(-λy))^{m-1}

so the log-likelihood for n i.i.d. observations is
    ℓ(λ) = n log m + n log λ - λ Σ y_i + (m-1) Σ log(1 - exp(-λ y_i)).

The score has no closed root for m>1:
    s(λ) = n/λ - Σ y_i + (m-1) Σ [ y_i / (exp(λ y_i) - 1) ].
s'(λ) < 0 for all λ>0, so there is a unique MLE, found by root-finding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expm1, log1p


M = 4
WORKSPACE = Path(__file__).resolve().parents[2]
DATA_PATH = WORKSPACE / "data" / "train.csv"
OUT_PATH = WORKSPACE / "task" / "results" / "estimate.json"


def load_y(path: Path) -> np.ndarray:
    y = np.loadtxt(path, delimiter=",", skiprows=1)
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size == 0 or np.any(~np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("expected a positive finite column of observations")
    return y


def log_one_minus_exp_neg(z: np.ndarray) -> np.ndarray:
    """log(1 - exp(-z)) for z > 0, numerically stable."""
    return log1p(-np.exp(-z))


def log_likelihood(lam: float, y: np.ndarray, m: int = M) -> float:
    z = lam * y
    return float(
        y.size * np.log(m)
        + y.size * np.log(lam)
        - np.sum(z)
        + (m - 1.0) * np.sum(log_one_minus_exp_neg(z))
    )


def score(lam: float, y: np.ndarray, m: int = M) -> float:
    """dℓ/dλ. Uses y/(e^{λy}-1) = y e^{-λy} / (1 - e^{-λy}) for stability."""
    z = lam * y
    # expm1(z) = e^z - 1; for large z this overflows, fall back to e^{-z}/(1-e^{-z})
    term = np.empty_like(z)
    large = z > 40.0
    term[~large] = y[~large] / expm1(z[~large])
    ez = np.exp(-z[large])
    term[large] = y[large] * ez / (1.0 - ez)
    return float(y.size / lam - np.sum(y) + (m - 1.0) * np.sum(term))


def method_of_moments(y: np.ndarray, m: int = M) -> float:
    # E[max of m i.i.d. Exp(λ)] = H_m / λ
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    return harmonic / float(np.mean(y))


def mle_lambda(y: np.ndarray, m: int = M) -> float:
    # Bracket the unique root of the strictly decreasing score.
    lo = 1e-8
    hi = max(method_of_moments(y, m) * 20.0, 10.0)
    # Grow the upper bound if the score is still positive there.
    s_hi = score(hi, y, m)
    while s_hi > 0.0:
        hi *= 2.0
        s_hi = score(hi, y, m)
        if hi > 1e8:
            raise RuntimeError("failed to bracket the score root")
    return float(brentq(lambda lam: score(lam, y, m), lo, hi, xtol=1e-14))


def main() -> None:
    y = load_y(DATA_PATH)
    lam_hat = mle_lambda(y, M)
    if not np.isfinite(lam_hat) or lam_hat <= 0.0:
        raise RuntimeError(f"invalid MLE: {lam_hat!r}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"lambda_hat": lam_hat}
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lam_mom = method_of_moments(y, M)
    lam_naive = 1.0 / float(np.mean(y))
    print(f"n            = {y.size}")
    print(f"mean(y)      = {float(np.mean(y)):.10f}")
    print(f"naive Exp MLE= {lam_naive:.10f}   (treats y as raw Exp(lambda))")
    print(f"MoM (H_m/mean(y)) = {lam_mom:.10f}")
    print(f"block-max MLE= {lam_hat:.12f}")
    print(f"loglik       = {log_likelihood(lam_hat, y, M):.10f}")
    print(f"score        = {score(lam_hat, y, M):.3e}")
    print(f"wrote        {OUT_PATH}")


if __name__ == "__main__":
    main()
