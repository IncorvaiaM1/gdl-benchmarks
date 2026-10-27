r"""Maximum likelihood estimation of an Exponential rate observed through block maxima.

Model
-----
Each row of ``data/train.csv`` is

    y = max(x_1, ..., x_m),      x_1, ..., x_m  i.i.d. ~ Exponential(lambda),

with ``m = 4`` known and ``lambda > 0`` the single unknown parameter.

Likelihood
----------
An Exponential(lambda) variate has CDF ``G(x) = 1 - exp(-lambda x)`` on
``x > 0``. Because the m draws are independent, the maximum is below y exactly
when all m draws are, so the CDF of Y is the m-th power of G:

    F(y; lambda) = (1 - exp(-lambda y)) ** m,                     y > 0.

Differentiating gives the density of a single observation:

    f(y; lambda) = m * lambda * exp(-lambda y)
                     * (1 - exp(-lambda y)) ** (m - 1),           y > 0.

(This is the standard density of the m-th order statistic of an Exponential
sample; note it is *not* Exponential itself, and treating it as such -- i.e.
reporting ``1 / mean(y)`` -- is the main trap in this task.)

For an i.i.d. sample y_1, ..., y_n the log-likelihood is therefore

    l(lambda) = n log m + n log lambda - lambda * sum_i y_i
                  + (m - 1) * sum_i log(1 - exp(-lambda y_i)).

Score and curvature
-------------------
Differentiating with respect to lambda, and using
``d/dlambda log(1 - e^{-lambda y}) = y e^{-lambda y} / (1 - e^{-lambda y})``:

    l'(lambda) = n / lambda - sum_i y_i
                   + (m - 1) * sum_i  y_i e^{-lambda y_i}
                                      / (1 - e^{-lambda y_i})

               = n / lambda - sum_i y_i
                   + (m - 1) * sum_i  y_i / (exp(lambda y_i) - 1),

    l''(lambda) = -n / lambda**2
                   - (m - 1) * sum_i  y_i**2 e^{-lambda y_i}
                                      / (1 - e^{-lambda y_i})**2.

``l''`` is strictly negative for every ``lambda > 0`` (both terms are), so the
log-likelihood is strictly concave on ``(0, inf)``: the stationary point is the
unique global maximum. Moreover ``l'(lambda) -> +inf`` as ``lambda -> 0+``
(the ``n / lambda`` and ``(m-1) n / lambda`` terms dominate) and
``l'(lambda) -> -sum_i y_i < 0`` as ``lambda -> inf``. So the score has exactly
one sign change and any bracketing root-finder converges to the MLE.

For ``m = 1`` the score reduces to ``n / lambda - sum_i y_i``, recovering the
familiar closed form ``1 / mean(y)``. For ``m > 1`` the extra sum couples the
observations nonlinearly and no closed form in lambda exists, so the root is
found numerically.

Starting value
--------------
``E[Y] = H_m / lambda`` where ``H_m = sum_{k=1..m} 1/k`` is the m-th harmonic
number (the max of m i.i.d. Exponentials is a sum of independent Exponentials
with rates ``m, m-1, ..., 1``). The method-of-moments estimator
``H_m / mean(y)`` is consistent and lands very close to the MLE, which makes it
an excellent initial bracket centre.

Numerical care
--------------
All expressions above are evaluated through ``expm1``/``log1p`` in terms of
``u = 1 - exp(-lambda y)``:

    u = -expm1(-lambda y)          accurate when lambda y is small
    log(u)                         via log1p(-exp(-lambda y)) equivalently
    y / (exp(lambda y) - 1)   ==   y * exp(-lambda y) / u

The ``exp(-lambda y)`` form never overflows (it underflows harmlessly to 0 for
large ``lambda y``), whereas the algebraically identical ``exp(lambda y) - 1``
form overflows once ``lambda y > ~709``.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

__all__ = [
    "harmonic_number",
    "sha256_of_file",
    "load_column_y",
    "load_column_y_as_strings",
    "log_likelihood",
    "score",
    "score_derivative",
    "moment_estimate",
    "bracket_score",
    "mle_scipy_brentq",
    "mle_stdlib",
    "mle_mpmath",
    "standard_error",
]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def harmonic_number(m: int) -> float:
    """H_m = 1 + 1/2 + ... + 1/m."""
    return math.fsum(1.0 / k for k in range(1, m + 1))


def sha256_of_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's raw bytes, as lowercase hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_column_y(path: str | Path) -> np.ndarray:
    """Read the single ``y`` column of a one-column CSV with a header row."""
    y = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=1)
    if y.ndim != 1:
        raise ValueError(f"expected a single column in {path}, got shape {y.shape}")
    return y


def load_column_y_as_strings(path: str | Path) -> list[str]:
    """Read the ``y`` column as the raw decimal strings written in the file.

    Used by the mpmath cross-check so that the high-precision solve starts from
    the file's exact decimal values rather than from float64 round-trips.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip().lower() != "y":
        raise ValueError(f"expected a header line 'y' in {path}")
    return [line.strip() for line in lines[1:] if line.strip()]


# --------------------------------------------------------------------------- #
# likelihood, score, curvature (float64 / numpy)
# --------------------------------------------------------------------------- #
def log_likelihood(lam: float, y: np.ndarray, m: int) -> float:
    """l(lambda) for the block-maxima model, up to nothing (exact, incl. constants)."""
    if lam <= 0.0:
        return -np.inf
    n = y.size
    u = -np.expm1(-lam * y)  # 1 - exp(-lam*y), stable near 0
    return (
        n * math.log(m)
        + n * math.log(lam)
        - lam * float(np.sum(y))
        + (m - 1) * float(np.sum(np.log(u)))
    )


def score(lam: float, y: np.ndarray, m: int) -> float:
    """l'(lambda). The MLE is its unique positive root."""
    z = lam * y
    em = np.exp(-z)
    u = -np.expm1(-z)
    return (
        y.size / lam
        - float(np.sum(y))
        + (m - 1) * float(np.sum(y * em / u))
    )


def score_derivative(lam: float, y: np.ndarray, m: int) -> float:
    """l''(lambda) -- strictly negative, so also minus the observed information."""
    z = lam * y
    em = np.exp(-z)
    u = -np.expm1(-z)
    return -y.size / lam**2 - (m - 1) * float(np.sum(y * y * em / (u * u)))


def moment_estimate(y: np.ndarray, m: int) -> float:
    """Method-of-moments start: E[Y] = H_m / lambda  =>  lambda ~ H_m / mean(y)."""
    return harmonic_number(m) / float(np.mean(y))


def standard_error(lam: float, y: np.ndarray, m: int) -> float:
    """Asymptotic SE of the MLE from the observed information ``-l''(lambda_hat)``."""
    return 1.0 / math.sqrt(-score_derivative(lam, y, m))


# --------------------------------------------------------------------------- #
# solver 1: bracket + scipy.optimize.brentq
# --------------------------------------------------------------------------- #
def bracket_score(y: np.ndarray, m: int) -> tuple[float, float]:
    """Find ``(lo, hi)`` with ``score(lo) > 0 > score(hi)``, starting from MoM."""
    start = moment_estimate(y, m)
    lo = hi = start
    for _ in range(200):
        if score(lo, y, m) > 0.0:
            break
        lo /= 2.0
    else:  # pragma: no cover - unreachable given l' -> +inf as lambda -> 0+
        raise RuntimeError("failed to bracket the score from below")
    for _ in range(200):
        if score(hi, y, m) < 0.0:
            break
        hi *= 2.0
    else:  # pragma: no cover - unreachable given l' -> -sum(y) as lambda -> inf
        raise RuntimeError("failed to bracket the score from above")
    return lo, hi


def mle_scipy_brentq(y: np.ndarray, m: int) -> float:
    """MLE via Brent's method on the score equation (primary estimate)."""
    from scipy.optimize import brentq

    lo, hi = bracket_score(y, m)
    return float(
        brentq(score, lo, hi, args=(y, m), xtol=1e-15, rtol=8.9e-16, maxiter=200)
    )


# --------------------------------------------------------------------------- #
# solver 2: stdlib only (no numpy, no scipy) -- independent implementation
# --------------------------------------------------------------------------- #
def _score_pure(lam: float, y: list[float], m: int) -> float:
    total = math.fsum(y)
    extra = math.fsum(yi / math.expm1(lam * yi) for yi in y)
    return len(y) / lam - total + (m - 1) * extra


def _score_derivative_pure(lam: float, y: list[float], m: int) -> float:
    acc = math.fsum(
        yi * yi * math.exp(-lam * yi) / (-math.expm1(-lam * yi)) ** 2 for yi in y
    )
    return -len(y) / lam**2 - (m - 1) * acc


def mle_stdlib(y_list: list[float], m: int, tol: float = 1e-15) -> float:
    """MLE with only the standard library: bisection to a tight bracket, then Newton.

    Deliberately written against a different algebraic form of the score
    (``y / (exp(lam*y) - 1)`` rather than ``y * exp(-lam*y) / (1 - exp(-lam*y))``)
    and with Kahan-compensated ``math.fsum`` reductions, so that agreement with
    :func:`mle_scipy_brentq` is evidence about the derivation and not just about
    one shared code path.
    """
    n = len(y_list)
    lo = harmonic_number(m) / (math.fsum(y_list) / n)
    hi = lo
    while _score_pure(lo, y_list, m) <= 0.0:
        lo /= 2.0
    while _score_pure(hi, y_list, m) >= 0.0:
        hi *= 2.0

    for _ in range(200):  # bisection: robust, gets us within a few ulp
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if _score_pure(mid, y_list, m) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * hi:
            break

    lam = 0.5 * (lo + hi)
    for _ in range(60):  # Newton polish on the score
        s = _score_pure(lam, y_list, m)
        step = s / _score_derivative_pure(lam, y_list, m)
        lam_new = lam - step
        if not (lam_new > 0.0):  # pragma: no cover - concavity rules this out
            break
        if abs(lam_new - lam) <= 1e-16 * lam_new:
            lam = lam_new
            break
        lam = lam_new
    return lam


# --------------------------------------------------------------------------- #
# solver 3: mpmath at high precision -- independent of float64 entirely
# --------------------------------------------------------------------------- #
def mle_mpmath(y_strings: list[str], m: int, dps: int = 50, start: float | None = None):
    """MLE re-solved in ``dps``-digit arithmetic straight from the CSV's decimals.

    Returns an ``mpmath.mpf``. This exists to show that the float64 answer is
    correct far beyond the 8 significant figures the task requires, i.e. that
    the reported digits are a property of the data and not of rounding.
    """
    import mpmath as mp

    with mp.workdps(dps + 10):
        ys = [mp.mpf(s) for s in y_strings]
        n = mp.mpf(len(ys))
        total = mp.fsum(ys)
        mm1 = mp.mpf(m - 1)

        def score_mp(lam):
            extra = mp.fsum([yi / mp.expm1(lam * yi) for yi in ys])
            return n / lam - total + mm1 * extra

        def dscore_mp(lam):
            acc = mp.fsum(
                [
                    yi * yi * mp.exp(-lam * yi) / (-mp.expm1(-lam * yi)) ** 2
                    for yi in ys
                ]
            )
            return -n / lam**2 - mm1 * acc

        if start is None:
            start = mp.fsum([mp.mpf(1) / k for k in range(1, m + 1)]) * n / total
        lam = mp.mpf(start) if not isinstance(start, float) else mp.mpf(repr(start))

        # Newton on a strictly concave, smooth score: quadratic and monotone
        # from any positive start close to the root.
        target = mp.mpf(10) ** (-(dps + 5))
        for _ in range(200):
            step = score_mp(lam) / dscore_mp(lam)
            lam_new = lam - step
            converged = abs(lam_new - lam) <= target * abs(lam_new)
            lam = lam_new
            if converged:
                break
        else:  # pragma: no cover - Newton on this score converges in ~5 steps
            raise RuntimeError("high-precision Newton iteration did not converge")
        return +mp.mpf(lam)
