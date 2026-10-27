"""Maximum likelihood estimate of lambda from block maxima of m i.i.d. Exponential(lambda).

Model: y = max(x_1, ..., x_m) with x_j i.i.d. Exponential(lambda), m known.

    F_Y(t) = (1 - exp(-lambda t))^m
    f_Y(t) = m lambda exp(-lambda t) (1 - exp(-lambda t))^(m-1)

    loglik(lambda) = n log m + n log lambda - lambda * sum(y)
                     + (m - 1) * sum(log(1 - exp(-lambda y)))

    score(lambda)  = n / lambda - sum(y) + (m - 1) * sum(y / (exp(lambda y) - 1))

The score is strictly decreasing with score(0+) = +inf and score(inf) = -sum(y),
so its root is unique; it is found with Brent's method and cross-checked against
direct minimisation of the negative log-likelihood.

See README.md in this directory for the full derivation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

M_DEFAULT = 4

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data" / "train.csv"
DEFAULT_OUT = REPO_ROOT / "task" / "results" / "estimate.json"


def load_y(path: Path) -> np.ndarray:
    """Read the single-column csv (header 'y') into a float64 array."""
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    header = [c.strip().lower() for c in rows[0]]
    if header != ["y"]:
        raise ValueError(f"expected a single column named 'y', got {rows[0]!r}")
    y = np.array([float(r[0]) for r in rows[1:] if r and r[0].strip()], dtype=np.float64)
    if y.size == 0:
        raise ValueError(f"{path} contains no observations")
    if not np.all(np.isfinite(y)) or np.any(y <= 0.0):
        raise ValueError("observations must be finite and strictly positive")
    return y


def loglik(lam: float, y: np.ndarray, m: int) -> float:
    """Log-likelihood of lambda under the block-maxima model."""
    if lam <= 0.0:
        return -math.inf
    n = y.size
    # log1p(-exp(-u)) is evaluated stably: expm1 avoids cancellation for small u.
    u = lam * y
    log_cdf_term = np.log(-np.expm1(-u))
    return (
        n * math.log(m)
        + n * math.log(lam)
        - lam * float(y.sum())
        + (m - 1) * float(log_cdf_term.sum())
    )


def score(lam: float, y: np.ndarray, m: int) -> float:
    """d/dlambda of the log-likelihood."""
    n = y.size
    u = lam * y
    # y exp(-u) / (1 - exp(-u)); the negative-exponent form cannot overflow for
    # large u, and expm1 keeps the denominator accurate for small u.
    correction = float(np.sum(-y * np.exp(-u) / np.expm1(-u)))
    return n / lam - float(y.sum()) + (m - 1) * correction


def mle_lambda(y: np.ndarray, m: int = M_DEFAULT) -> dict[str, float]:
    """Solve score(lambda) = 0 for lambda, with an independent cross-check.

    Returns a dict with the estimate and diagnostics.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    mean_y = float(y.mean())

    if m == 1:
        lam_hat = 1.0 / mean_y  # closed form; nothing to solve
        return {
            "lambda_hat": lam_hat,
            "lambda_optimizer": lam_hat,
            "lambda_moment_start": lam_hat,
            "score_at_root": 0.0,
            "loglik": loglik(lam_hat, y, m),
        }

    # Method-of-moments start: E[Y] = H_m / lambda, H_m the m-th harmonic number.
    harmonic_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    lam0 = harmonic_m / mean_y

    # Expand a bracket around lam0 until the score changes sign. The score is
    # monotone decreasing, so this always terminates for a valid sample.
    lo, hi = lam0, lam0
    for _ in range(200):
        if score(lo, y, m) > 0.0:
            break
        lo *= 0.5
    else:
        raise RuntimeError("failed to bracket the root from below")
    for _ in range(200):
        if score(hi, y, m) < 0.0:
            break
        hi *= 2.0
    else:
        raise RuntimeError("failed to bracket the root from above")

    lam_hat = float(brentq(score, lo, hi, args=(y, m), xtol=1e-14, rtol=1e-15, maxiter=200))

    # Cross-check: maximise the log-likelihood directly, independently of the score.
    opt = minimize_scalar(
        lambda lam: -loglik(lam, y, m),
        bracket=(lo, lam_hat, hi),
        method="brent",
        options={"xtol": 1e-12},
    )
    if not opt.success:
        raise RuntimeError(f"log-likelihood optimisation failed: {opt.message}")
    lam_opt = float(opt.x)
    if not math.isclose(lam_hat, lam_opt, rel_tol=1e-6):
        raise RuntimeError(
            f"root-finding ({lam_hat!r}) and optimisation ({lam_opt!r}) disagree"
        )

    if not (math.isfinite(lam_hat) and lam_hat > 0.0):
        raise RuntimeError(f"non-finite or non-positive estimate: {lam_hat!r}")

    return {
        "lambda_hat": lam_hat,
        "lambda_optimizer": lam_opt,
        "lambda_moment_start": lam0,
        "score_at_root": score(lam_hat, y, m),
        "loglik": loglik(lam_hat, y, m),
    }


def observed_information(lam: float, y: np.ndarray, m: int) -> float:
    """-d2/dlambda2 of the log-likelihood, for a standard error."""
    n = y.size
    u = lam * y
    # d/dlam [ y exp(-u)/(1-exp(-u)) ] = -y^2 exp(-u) / (1 - exp(-u))^2
    term = float(np.sum((y**2) * np.exp(-u) / np.expm1(-u) ** 2))
    return n / lam**2 + (m - 1) * term


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="input csv (column 'y')")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output json path")
    ap.add_argument("--m", type=int, default=M_DEFAULT, help="block size (known, default 4)")
    args = ap.parse_args()

    y = load_y(args.data)
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()

    res = mle_lambda(y, args.m)
    lam_hat = res["lambda_hat"]
    se = 1.0 / math.sqrt(observed_information(lam_hat, y, args.m))
    naive = 1.0 / float(y.mean())  # wrong-model estimator, printed for contrast only

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"lambda_hat": lam_hat}) + "\n")

    print(f"data                 {args.data}")
    print(f"sha256               {digest}")
    print(f"n                    {y.size}")
    print(f"m                    {args.m}")
    print(f"mean(y)              {y.mean():.6f}")
    print(f"moment start         {res['lambda_moment_start']:.10f}")
    print(f"lambda_hat (MLE)     {lam_hat:.10f}")
    print(f"  cross-check (opt)  {res['lambda_optimizer']:.10f}")
    print(f"  score at root      {res['score_at_root']:.3e}")
    print(f"  loglik at root     {res['loglik']:.6f}")
    print(f"  approx std err     {se:.6f}  (95% CI {lam_hat - 1.96 * se:.4f}, {lam_hat + 1.96 * se:.4f})")
    print(f"naive 1/mean(y)      {naive:.10f}   <- wrong model, not reported")
    print(f"wrote                {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
