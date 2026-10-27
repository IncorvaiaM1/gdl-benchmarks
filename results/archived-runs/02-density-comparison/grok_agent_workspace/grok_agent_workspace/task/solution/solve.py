#!/usr/bin/env python3
"""Compare three MNIST generative models via dequantization / ELBO bounds.

Entry point: from the workspace root,

    .venv/Scripts/python.exe task/solution/solve.py

writes task/results/answer.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad

D = 784  # 28 x 28
N_IMAGES = 1000
LEVELS = 256
LAM_B = 0.05
SCORE_A = 1972.0  # mean log-density of dequantized y, nats / image
SCORE_B = -150.0  # mean log-density of logit-space y', nats / image
LOSS_C = 93.7  # mean BCE+KL, nats / padded binary image
C_SHAPE = 32 * 32  # VAE models a 32x32 binary tensor after padding

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "images.csv"
OUT_PATH = ROOT / "task" / "results" / "answer.json"


def zlogz(z: np.ndarray) -> np.ndarray:
    """z * log(z), with the continuous extension 0 * log(0) = 0."""
    z = np.asarray(z, dtype=np.float64)
    out = np.zeros_like(z)
    pos = z > 0.0
    out[pos] = z[pos] * np.log(z[pos])
    return out


def expected_neg_log_s_one_minus_s(a: np.ndarray, b: float) -> np.ndarray:
    """E_u[-log s - log(1-s)] for s = a + b u, u ~ Unif[0, 1).

    Closed form from ∫ log t dt = t log t - t.
    """
    a = np.asarray(a, dtype=np.float64)
    a1 = a + b
    term_s = zlogz(a1) - zlogz(a) - b
    term_1s = zlogz(1.0 - a) - zlogz(1.0 - a1) - b
    return -(term_s + term_1s) / b


def pixel_logit_integral(k: np.ndarray, lam: float) -> np.ndarray:
    """Per-pixel E_u[-log s(1-s)] under Model B's map at this λ."""
    scale = 1.0 - 2.0 * lam
    y0 = k.astype(np.float64) / 255.0
    a = lam + scale * y0
    b = scale / LEVELS
    return expected_neg_log_s_one_minus_s(a, b)


def mean_image_Q(hist: np.ndarray, lam: float) -> float:
    """Average over images of sum_i E_u[-log s_i(1-s_i)].

    hist[k] = number of pixels equal to k across all N_IMAGES * D pixels.
    """
    ks = np.arange(256, dtype=np.float64)
    per_k = pixel_logit_integral(ks, lam)
    return float(np.dot(hist, per_k) / N_IMAGES)


def bpd_A() -> float:
    """Jensen upper bound from uniform dequantization y = (k + u) / 256.

    P(x) = ∫ p(y) dy over the unit-cube image in u, i.e.
         = E_u[ p(y(u)) * 256^{-D} ].
    log P(x) ≥ E log p(y) - D log 256, so
    bpd ≤ (D log 256 - mean_score) / (D log 2).
    """
    nats = D * math.log(LEVELS) - SCORE_A
    return nats / (D * math.log(2.0))


def bpd_B_from_Q(Q: float, lam: float) -> float:
    """Jensen upper bound for Model B, given the data-dependent Q(λ).

    y = k/255 + u/256, s = λ + (1-2λ) y, y' = logit(s).
    P(x) = E_u[ p(y') |∂y'/∂u| ], with
    |∂y'_i/∂u_i| = (1-2λ) / (s_i (1-s_i) * 256).
    """
    log_abs_det = Q + D * math.log(1.0 - 2.0 * lam) - D * math.log(LEVELS)
    nats = -SCORE_B - log_abs_det
    return nats / (D * math.log(2.0))


def lambda_min() -> float:
    """Smallest λ ∈ (0, 1/2) with s < 1 for every 8-bit pixel and u ∈ [0, 1).

    Max y approaches 1 + 1/256 (k = 255, u → 1). Then
    λ + (1-2λ)(1 + 1/256) ≤ 1  ⇔  λ ≥ 1/258.
    At λ = 1/258, s → 1 but never reaches it for u < 1, so y' stays finite.
    """
    return 1.0 / 258.0


def solve_lambda_star(hist: np.ndarray, target_bpd: float) -> float:
    """λ where Model B's bound equals Model A's, score held at -150."""
    lam_lo = lambda_min()
    lam_hi = 0.5 - 1e-12

    def gap(lam: float) -> float:
        return bpd_B_from_Q(mean_image_Q(hist, lam), lam) - target_bpd

    g_lo, g_hi = gap(lam_lo), gap(lam_hi)
    if g_lo == 0.0:
        return lam_lo
    if g_lo * g_hi > 0.0:
        raise RuntimeError(
            f"lambda_star not bracketed: gap({lam_lo})={g_lo}, gap({lam_hi})={g_hi}"
        )
    for _ in range(200):
        mid = 0.5 * (lam_lo + lam_hi)
        g_mid = gap(mid)
        if g_lo * g_mid <= 0.0:
            lam_hi = mid
            g_hi = g_mid
        else:
            lam_lo = mid
            g_lo = g_mid
        if abs(lam_hi - lam_lo) < 1e-15:
            break
    return 0.5 * (lam_lo + lam_hi)


def load_hist() -> np.ndarray:
    pixels = np.loadtxt(DATA_PATH, delimiter=",", skiprows=1, dtype=np.int64)
    if pixels.shape != (N_IMAGES, D):
        raise ValueError(f"expected images.csv shape {(N_IMAGES, D)}, got {pixels.shape}")
    if pixels.min() < 0 or pixels.max() > 255:
        raise ValueError("pixel values are not in {0,...,255}")
    hist = np.bincount(pixels.ravel(), minlength=256).astype(np.float64)
    return hist


def quadrature_check(lam: float) -> None:
    """Sanity-check the closed-form pixel integral against adaptive quadrature."""
    ks = np.array([0, 1, 127, 128, 254, 255], dtype=np.float64)
    closed = pixel_logit_integral(ks, lam)
    scale = 1.0 - 2.0 * lam
    b = scale / LEVELS
    for k, analytic in zip(ks, closed):
        a = lam + scale * (k / 255.0)

        def integrand(u: float, a=a, b=b) -> float:
            s = a + b * u
            # Clip off the endpoint singularity at λ = λ_min, k = 255.
            s = min(max(s, 1e-18), 1.0 - 1e-18)
            return -math.log(s) - math.log(1.0 - s)

        numeric, _ = quad(integrand, 0.0, 1.0, epsabs=1e-12, limit=200)
        if abs(analytic - numeric) > 1e-9:
            raise RuntimeError(
                f"integral check failed at lam={lam}, k={k}: "
                f"closed={analytic}, quad={numeric}"
            )


def main() -> None:
    hist = load_hist()
    quadrature_check(LAM_B)
    quadrature_check(lambda_min())

    bpd_a = bpd_A()
    Q_b = mean_image_Q(hist, LAM_B)
    bpd_b = bpd_B_from_Q(Q_b, LAM_B)
    better = "A" if bpd_a < bpd_b else "B"
    lam_min = lambda_min()
    lam_star = solve_lambda_star(hist, bpd_a)
    bpd_c_own = LOSS_C / (C_SHAPE * math.log(2.0))
    # The VAE defines a distribution on dynamically-binarized 32x32 images,
    # not on 8-bit 28x28 images, so the reported ELBO does not bound 8-bit bpd.
    bpd_c_8bit = None

    answer = {
        "bpd_A": bpd_a,
        "bpd_B": bpd_b,
        "better": better,
        "lambda_min": lam_min,
        "lambda_star": lam_star,
        "bpd_C_own": bpd_c_own,
        "bpd_C_8bit": bpd_c_8bit,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(answer, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(answer, indent=2))
    print(f"Q(lambda=0.05) = {Q_b:.12f}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
