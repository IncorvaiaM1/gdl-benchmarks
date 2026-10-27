"""Convert two flow scores reported in different spaces into a common bits/dim.

Entry point: python task/solution/solve.py

See DERIVATION.md in this directory for the change-of-variables algebra.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- problem data

D = 784
LAM = 0.05
ELL_A = 1972.0
ELL_B = -150.0
N_LEVELS = 256
LN2 = math.log(2.0)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = REPO_ROOT / "data" / "images.csv"
ANSWER_JSON = REPO_ROOT / "task" / "results" / "answer.json"


# ------------------------------------------------------- exact noise integrals
#
# For a pixel level k the dequantized coordinate is v = k + u, u ~ U[0,1), so
#     y = v / 256,   s = lam + (1 - 2 lam) * y = a_k + c * u
# with
#     a_k = lam + (1 - 2 lam) * k / 256,   c = (1 - 2 lam) / 256.
# The two integrals below are therefore closed-form in k.


def _mean_log_affine(a: np.ndarray, c: float) -> np.ndarray:
    """Exact int_0^1 log(a + c*u) du, evaluated without catastrophic cancellation.

    The naive antiderivative difference ((a+c)log(a+c) - a log a)/c - 1 subtracts
    two nearly equal O(a log a) terms; expanding log(a+c) = log a + log1p(c/a)
    removes that cancellation.
    """
    r = c / a
    return np.log(a) + (1.0 + 1.0 / r) * np.log1p(r) - 1.0


def per_level_expected_log_jacobian_term() -> np.ndarray:
    """E_u[log s + log(1 - s)] for each pixel level k = 0..255.

    Exact in the dequantization noise (analytic integral, not sampled).
    """
    k = np.arange(N_LEVELS, dtype=np.float64)
    c = (1.0 - 2.0 * LAM) / N_LEVELS
    a = LAM + (1.0 - 2.0 * LAM) * k / N_LEVELS

    e_log_s = _mean_log_affine(a, c)
    # 1 - s runs from 1 - a down to 1 - a - c, i.e. the same integral with c -> -c.
    e_log_1ms = _mean_log_affine(1.0 - a, -c)
    return e_log_s + e_log_1ms


# ------------------------------------------------------------------------ data


def load_level_counts(csv_path: Path) -> tuple[np.ndarray, int]:
    """Return (counts over levels 0..255 summed across all pixels, n_images)."""
    x = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != D:
        raise ValueError(f"expected (n, {D}) integer array, got {x.shape}")
    if x.min() < 0 or x.max() > N_LEVELS - 1:
        raise ValueError(f"pixel values outside [0, {N_LEVELS - 1}]")
    return np.bincount(x.ravel(), minlength=N_LEVELS).astype(np.int64), x.shape[0]


# ------------------------------------------------------------------------ math


def bpd_model_a() -> float:
    """y = v / 256 only, so log p_V = log p_A(y) - D log 256 and the data drops out."""
    return 8.0 - ELL_A / (D * LN2)


def bpd_model_b(level_counts: np.ndarray, n_images: int) -> tuple[float, float]:
    """Return (bpd_B, S) where S = E_{x,u}[ sum_d (log s_d + log(1 - s_d)) ]."""
    g = per_level_expected_log_jacobian_term()
    s_total = float(np.dot(level_counts.astype(np.float64), g)) / n_images
    bpd = 8.0 - (ELL_B + D * math.log(1.0 - 2.0 * LAM) - s_total) / (D * LN2)
    return bpd, s_total


# ----------------------------------------------------------------- self-checks


def _check_integrals(rng: np.random.Generator) -> None:
    """Cross-check the closed-form noise integral two independent ways."""
    g = per_level_expected_log_jacobian_term()
    c = (1.0 - 2.0 * LAM) / N_LEVELS
    k = np.arange(N_LEVELS, dtype=np.float64)
    a = LAM + (1.0 - 2.0 * LAM) * k / N_LEVELS

    # (1) 200-node Gauss-Legendre quadrature on u in [0, 1].
    nodes, weights = np.polynomial.legendre.leggauss(200)
    u = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    s = a[:, None] + c * u[None, :]
    g_quad = (np.log(s) + np.log1p(-s)) @ w
    quad_err = float(np.max(np.abs(g_quad - g)))

    # (2) Monte Carlo over u, loose tolerance.
    u_mc = rng.random((N_LEVELS, 400_000))
    s_mc = a[:, None] + c * u_mc
    g_mc = (np.log(s_mc) + np.log1p(-s_mc)).mean(axis=1)
    mc_err = float(np.max(np.abs(g_mc - g)))

    print(f"  closed form vs Gauss-Legendre(200): max abs diff {quad_err:.3e}")
    print(f"  closed form vs Monte Carlo(4e5/level): max abs diff {mc_err:.3e}")
    assert quad_err < 1e-12, quad_err
    assert mc_err < 1e-4, mc_err


def _check_end_to_end(level_counts: np.ndarray, n_images: int, bpd_b: float,
                      rng: np.random.Generator) -> None:
    """Monte-Carlo the whole Model-B bpd from sampled u, as an independent path."""
    levels = np.repeat(np.arange(N_LEVELS), level_counts)  # all D*n_images pixels
    total = 0.0
    n_rep = 20
    for _ in range(n_rep):
        u = rng.random(levels.size)
        y = (levels + u) / N_LEVELS
        s = LAM + (1.0 - 2.0 * LAM) * y
        total += float(np.sum(np.log(s) + np.log1p(-s)))
    s_mc = total / (n_rep * n_images)
    bpd_mc = 8.0 - (ELL_B + D * math.log(1.0 - 2.0 * LAM) - s_mc) / (D * LN2)
    print(f"  bpd_B via sampled dequantization noise: {bpd_mc:.6f} "
          f"(exact {bpd_b:.6f}, diff {abs(bpd_mc - bpd_b):.2e})")
    assert abs(bpd_mc - bpd_b) < 5e-4, abs(bpd_mc - bpd_b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA_CSV)
    ap.add_argument("--out", type=Path, default=ANSWER_JSON)
    ap.add_argument("--no-checks", action="store_true",
                    help="skip the quadrature / Monte-Carlo cross-checks")
    args = ap.parse_args()

    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    print(f"data: {args.data}")
    print(f"sha256: {digest}")

    level_counts, n_images = load_level_counts(args.data)
    print(f"images: {n_images}   pixels/image: {D}   "
          f"zero-valued pixel fraction: {level_counts[0] / (n_images * D):.6f}")

    bpd_a = bpd_model_a()
    bpd_b, s_total = bpd_model_b(level_counts, n_images)

    if not args.no_checks:
        rng = np.random.default_rng(0)
        print("cross-checks:")
        _check_integrals(rng)
        _check_end_to_end(level_counts, n_images, bpd_b, rng)

    print()
    print("Model A:  bpd = 8 - ell_A / (D ln2)")
    print(f"          = 8 - {ELL_A} / {D * LN2:.10f} = {bpd_a:.10f}")
    print("Model B:  bpd = 8 - [ ell_B + D ln(1-2lam) - S ] / (D ln2)")
    print(f"          S = E[sum_d (log s_d + log(1-s_d))] = {s_total:.10f}")
    print(f"          D ln(1-2lam) = {D * math.log(1 - 2 * LAM):.10f}")
    print(f"          = {bpd_b:.10f}")

    better = "A" if bpd_a < bpd_b else "B"
    print(f"\nbpd_A = {bpd_a:.10f}\nbpd_B = {bpd_b:.10f}\nbetter = {better}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"bpd_A": bpd_a, "bpd_B": bpd_b, "better": better}, indent=2) + "\n"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
