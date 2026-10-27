"""Gold solution for the three-model likelihood comparison (v2).

Model A: y = (x + u)/256                     (x = raw pixel k in 0..255)
Model B: y = k/255 + u/256, s = lam + (1-2lam) y, y' = logit(s)   [the /255 pipeline]
Model C: VAE on b ~ Bernoulli(k/255), zero-padded to 32x32 (D_C = 1024).

For any model with density p_Y on its (pre-flow-preprocessing) input y and a
dequantization bin of volume 256^{-D} per image (true for A and B):
    -log P(x) <= -E_u log p_Y(y) + D log 256          (Jensen)
B's logit preprocessing adds log|dy'/dy| = log(1-2lam) - log s - log(1-s),
whose expectation over u is computed in closed form per pixel value.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PARAMS = json.loads((ROOT / "gt_code" / "true_params.json").read_text())
D_C = 32 * 32


def _mean_log(lo, hi):
    """(1/(hi-lo)) * integral_lo^hi log t dt, for 0 < lo < hi."""
    F = lambda t: t * math.log(t) - t
    return (F(hi) - F(lo)) / (hi - lo)


def mean_logit_jacobian(counts, lam, n_images):
    """E_{x,u} sum_d log|dy'_d/dy_d| per image, for B's /255 pipeline."""
    b = (1 - 2 * lam) / 256
    total = 0.0
    for k in np.nonzero(counts)[0]:
        a = lam + (1 - 2 * lam) * k / 255
        total += counts[k] * (math.log(1 - 2 * lam) - _mean_log(a, a + b) - _mean_log(1 - a - b, 1 - a))
    return total / n_images


def lambda_min():
    """Smallest lam with s < 1 for every k <= 255, u < 1: lam + (1-2lam)(1+1/256) <= 1."""
    return 1 / 258


def solve(X, ell_A, ell_B, lam, loss_C):
    N, D = X.shape
    counts = np.bincount(X.ravel(), minlength=256)
    f = D * math.log(2)
    bpd_A = 8 - ell_A / f
    bpd_B_of = lambda l: 8 - (ell_B + mean_logit_jacobian(counts, l, N)) / f
    bpd_B = bpd_B_of(lam)

    # lambda_star: bpd_B(l) = bpd_A on [lambda_min, 1/2); bpd_B increasing in l.
    lo, hi = lambda_min(), 0.5 - 1e-12
    g = lambda l: bpd_B_of(l) - bpd_A
    assert g(lo) < 0 < g(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
    lam_star = 0.5 * (lo + hi)

    return {
        "bpd_A": bpd_A,
        "bpd_B": bpd_B,
        "better": "A" if bpd_A < bpd_B else "B",
        "lambda_min": lambda_min(),
        "lambda_star": lam_star,
        "bpd_C_own": loss_C / (D_C * math.log(2)),
        "bpd_C_8bit": None,
    }


def load_images():
    return np.loadtxt(ROOT / "data" / "images.csv", delimiter=",", skiprows=1, dtype=np.int64)


def main():
    out = solve(load_images(), **PARAMS)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
