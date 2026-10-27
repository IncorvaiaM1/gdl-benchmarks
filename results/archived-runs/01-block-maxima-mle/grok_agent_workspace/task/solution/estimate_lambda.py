"""MLE of Exponential rate λ from block maxima y = max(X_1..X_m).

Each train observation is Y = max(X_1, ..., X_m) with m = 4 and
X_j i.i.d. Exponential(λ) in the rate parameterization:

    f_X(x) = λ exp(-λ x),  F_X(x) = 1 - exp(-λ x),  x > 0.

The density of a single block maximum is therefore

    f_Y(y) = m [F_X(y)]^{m-1} f_X(y)
           = m (1 - exp(-λ y))^{m-1} λ exp(-λ y).

For n i.i.d. observations the log-likelihood is

    ℓ(λ) = n log m + n log λ
           + (m-1) Σ_i log(1 - exp(-λ y_i))
           - λ Σ_i y_i.

The score (used as the estimating equation) is

    ℓ'(λ) = n/λ + (m-1) Σ_i [ y_i / (exp(λ y_i) - 1) ] - Σ_i y_i.

This has no closed form for m > 1. ℓ''(λ) < 0 for all λ > 0, so the
root is unique and is the MLE. A method-of-moments start uses
E[Y] = H_m / λ, H_m = Σ_{k=1}^m 1/k.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, newton

M = 4
WORKSPACE = Path(__file__).resolve().parents[2]
TRAIN_PATH = WORKSPACE / "data" / "train.csv"
RESULT_PATH = WORKSPACE / "task" / "results" / "estimate.json"


def load_y(path: Path) -> np.ndarray:
    y = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"expected a single column of y, got shape {y.shape}")
    if np.any(y <= 0.0) or not np.all(np.isfinite(y)):
        raise ValueError("train y must be finite and strictly positive")
    return y


def score(lam: float, y: np.ndarray, m: int = M) -> float:
    """ℓ'(λ). Uses expm1 for stability when λ y is small."""
    lam = float(lam)
    return (
        y.size / lam
        + (m - 1) * np.sum(y / np.expm1(lam * y))
        - np.sum(y)
    )


def score_prime(lam: float, y: np.ndarray, m: int = M) -> float:
    """ℓ''(λ). Always negative: unique critical point."""
    lam = float(lam)
    e = np.exp(lam * y)
    return -y.size / lam**2 - (m - 1) * np.sum(y**2 * e / (e - 1.0) ** 2)


def log_likelihood(lam: float, y: np.ndarray, m: int = M) -> float:
    """ℓ(λ). log1p(-exp(-λy)) == log(1 - exp(-λy))."""
    lam = float(lam)
    return (
        y.size * np.log(m)
        + y.size * np.log(lam)
        + (m - 1) * np.sum(np.log1p(-np.exp(-lam * y)))
        - lam * np.sum(y)
    )


def moments_start(y: np.ndarray, m: int = M) -> float:
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    return harmonic / float(np.mean(y))


def mle_lambda(y: np.ndarray, m: int = M) -> float:
    lam0 = moments_start(y, m)
    # Bracket a sign change of the strictly decreasing score.
    lo, hi = lam0 / 20.0, lam0 * 20.0
    s_lo, s_hi = score(lo, y, m), score(hi, y, m)
    if s_lo == 0.0:
        return lo
    if s_hi == 0.0:
        return hi
    if s_lo * s_hi > 0.0:
        raise RuntimeError(
            f"score does not change sign on [{lo}, {hi}]: {s_lo}, {s_hi}"
        )
    lam_brent = brentq(score, lo, hi, args=(y, m), xtol=1e-16, rtol=2e-14)
    lam_newton = float(
        newton(score, lam0, fprime=score_prime, args=(y, m), tol=1e-16, maxiter=100)
    )
    if abs(lam_brent - lam_newton) / lam_brent > 1e-12:
        raise RuntimeError(
            f"root-finders disagree: brentq={lam_brent!r} newton={lam_newton!r}"
        )
    if not (np.isfinite(lam_brent) and lam_brent > 0.0):
        raise RuntimeError(f"non-positive or non-finite MLE: {lam_brent!r}")
    return float(lam_brent)


def main() -> None:
    y = load_y(TRAIN_PATH)
    lam_hat = mle_lambda(y)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"lambda_hat": lam_hat}
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"n={y.size}")
    print(f"mean(y)={np.mean(y):.12f}")
    print(f"lambda_mom={moments_start(y):.12f}")
    print(f"lambda_hat={lam_hat:.16f}")
    print(f"score={score(lam_hat, y):.6e}")
    print(f"loglik={log_likelihood(lam_hat, y):.12f}")
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
