#!/usr/bin/env python3
"""Compare three MNIST generative models on a common bits/dim scale.

Entry point for the task in ``task/TASK.md``.  Run:

    python task/solution/solve.py

It reads ``data/images.csv`` and writes ``task/results/answer.json``.

--------------------------------------------------------------------------
Derivation
--------------------------------------------------------------------------
Notation: an image is 784 integer pixels k in {0,...,255}; D = 784;
``to_tensor`` gives x = k/255.  For a discrete image x the quantity of
interest is

    bpd = -(1/N) sum_x log2 P(x) / D .

For a continuous model, P(x) is the mass its density puts on the set S_x of
continuous inputs that image x can be mapped to by the model's own
preprocessing.  If the preprocessing draws y uniformly from S_x (volume V),

    P(x) = int_{S_x} p(y) dy = V * E_u[p(y)] >= V * exp(E_u[log p(y)])

by Jensen, so  -log P(x) <= -E_u[log p(y)] + log(1/V).  That is the standard
dequantization (ELBO) upper bound.  It is only a bound on a *probability*
assignment if the sets S_x for different images are disjoint; otherwise the
"P" being bounded sums to more than one and is not a distribution.

Model A -------------------------------------------------------------------
    y = (x*255 + u)/256 = (k + u)/256,   u ~ U[0,1)^D.
So y is uniform on the cube S_x = prod_i [k_i/256, (k_i+1)/256), volume
(1/256)^D.  These cubes tile [0,1)^D exactly: disjoint, valid bound.

    bpd_A = (D*log 256 - 1972.0) / (D*log 2) = 8 - 1972.0/(D*log 2).

Model B -------------------------------------------------------------------
    y = x + u/256 = k/255 + u/256,   s = lam + (1-2lam)*y,
    y' = logit(s) = log s - log(1-s).
Note y is *not* (k+u)/256: the bin for value k is [k/255, k/255 + 1/256),
still of width 1/256, but shifted.  Consecutive bins are separated by a gap
1/255 - 1/256 = 1/(255*256) > 0, so the bins are disjoint (they merely fail
to cover [0, 1+1/256) ).  Disjointness is what the bound needs, so the
reported number does imply a valid upper bound -- the bound is just a bit
loose because the model wastes mass in the gaps.

The flow reports log p_{Y'}(y').  Change of variables back to y, per pixel,

    dy'/dy = (1-2lam) / (s (1-s)),
    log p_Y(y) = log p_{Y'}(y') + sum_i [ log(1-2lam) - log s_i - log(1-s_i) ].

Hence, with L(lam) = mean over images and noise of that log|det| term,

    bpd_B = (D*log 256 - (-150.0 + L(lam))) / (D*log 2).

L(lam) is computed exactly (the u-average is an integral of a log, done in
closed form with the antiderivative G(z) = z log z - z):
for a pixel of value k, with a = lam + (1-2lam)k/255 and b = (1-2lam)/256,
s = a + b*u with u ~ U[0,1), so

    E_u[-log s]     = -(G(a+b) - G(a)) / b,
    E_u[-log(1-s)]  =  (G(1-a-b) - G(1-a)) / b.

lambda_min ----------------------------------------------------------------
y' is finite iff 0 < s < 1.  s is smallest at k=0,u=0 (s=lam>0, fine) and
approaches its supremum at k=255, u->1:  y -> 1 + 1/256, so we need
lam + (1-2lam)(1 + 1/256) <= 1  (the sup is not attained because u < 1).
This gives (1-2lam)/256 <= lam, i.e. lam >= 1/258.  So lambda_min = 1/258.

lambda_star ---------------------------------------------------------------
Setting bpd_B(lam) = bpd_A the D*log 256 terms cancel:

    -150.0 + L(lam) = 1972.0   <=>   L(lam) = 2122.0 .

L is strictly decreasing on [1/258, 1/2) (as lam grows the logit stretch
weakens, and log(1-2lam) -> -inf), so the root is unique; found by bisection.

Model C -------------------------------------------------------------------
The reported 93.7 nats is -ELBO for the *padded binary* image
b in {0,1}^{32x32}: the BCE is summed over all 1024 pixels and the decoder
emits 1024 logits, so the random variable the model actually models has
1024 dimensions.

    bpd_C_own = 93.7 / (1024 * log 2).

There is no bound on the 8-bit bpd.  The model never sees the 8-bit image:
b ~ Bernoulli(x) is a stochastic, non-invertible, information-destroying map
of x, not a dequantization of it.  A bound would require the modelled
variable to determine x (up to a known-volume set); many distinct 8-bit
images produce the same b with positive probability, and the marginal
sum_b p_binarize(b|x) P_theta(b) is a number <= 1 but is *not* a probability
assigned to x (summing it over all 8-bit images x gives ~256^784-ish mass,
i.e. it is wildly larger than 1).  So bpd_C_8bit = null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants
D8 = 784                      # dimension of the 8-bit image
D_C = 32 * 32                 # dimension of what Model C models (padded)
LN2 = math.log(2.0)

SCORE_A = 1972.0              # mean log p_A(y)      [nats/image]
SCORE_B = -150.0              # mean log p_B(y')     [nats/image]
LOSS_C = 93.7                 # mean -ELBO           [nats/image]
LAM_B = 0.05                  # lambda reported by team B

LAMBDA_MIN = 1.0 / 258.0      # see module docstring

EXPECTED_SHA256 = "98b5d6a6d90bc2bc1c22df168e45e63da97fa2cce5d75001650f7f391a8b69c4"


# ------------------------------------------------------------------- data
def load_pixel_histogram(csv_path: Path) -> tuple[int, np.ndarray]:
    """Return (n_images, counts) with counts[k] = number of pixels equal to k."""
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != D8:
        raise ValueError(f"expected (N, {D8}) pixels, got {arr.shape}")
    if arr.min() < 0 or arr.max() > 255:
        raise ValueError("pixel values outside {0,...,255}")
    return arr.shape[0], np.bincount(arr.ravel(), minlength=256).astype(np.float64)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------- model B
def _antideriv_log(z: np.ndarray) -> np.ndarray:
    """G(z) = z log z - z, the antiderivative of log, with G(0) = 0."""
    z = np.asarray(z, dtype=np.float64)
    out = np.zeros_like(z)
    pos = z > 0.0
    out[pos] = z[pos] * np.log(z[pos]) - z[pos]
    return out


def expected_logit_term_per_value(lam: float) -> np.ndarray:
    """E_u[ -log s - log(1-s) ] for every pixel value k = 0..255."""
    k = np.arange(256, dtype=np.float64)
    a = lam + (1.0 - 2.0 * lam) * (k / 255.0)
    b = (1.0 - 2.0 * lam) / 256.0
    lower = -(_antideriv_log(a + b) - _antideriv_log(a)) / b
    upper = (_antideriv_log(1.0 - a - b) - _antideriv_log(1.0 - a)) / b
    return lower + upper


def mean_logdet_B(lam: float, counts: np.ndarray, n_images: int) -> float:
    """Mean over images and noise of sum_i log|dy'_i/dy_i| for Model B."""
    per_value = expected_logit_term_per_value(lam)
    return D8 * math.log1p(-2.0 * lam) + float(counts @ per_value) / n_images


# ----------------------------------------------------------------- bounds
def dequant_bpd_bound(mean_log_density_y: float) -> float:
    """Jensen bound on bpd given the mean log-density in y-space (bins 1/256 wide)."""
    return (D8 * math.log(256.0) - mean_log_density_y) / (D8 * LN2)


def solve_lambda_star(counts: np.ndarray, n_images: int) -> float:
    """lam in [lambda_min, 1/2) with L(lam) = SCORE_A - SCORE_B."""
    target = SCORE_A - SCORE_B

    def f(lam: float) -> float:
        return mean_logdet_B(lam, counts, n_images) - target

    lo, hi = LAMBDA_MIN, 0.5 - 1e-12
    f_lo, f_hi = f(lo), f(hi)
    if f_lo < 0.0 or f_hi > 0.0:
        raise RuntimeError(f"root not bracketed: f({lo})={f_lo}, f({hi})={f_hi}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------- main
def compute(csv_path: Path) -> tuple[dict, dict]:
    n_images, counts = load_pixel_histogram(csv_path)

    bpd_A = dequant_bpd_bound(SCORE_A)

    logdet_B = mean_logdet_B(LAM_B, counts, n_images)
    bpd_B = dequant_bpd_bound(SCORE_B + logdet_B)

    lambda_star = solve_lambda_star(counts, n_images)

    answer = {
        "bpd_A": bpd_A,
        "bpd_B": bpd_B,
        "better": "A" if bpd_A <= bpd_B else "B",
        "lambda_min": LAMBDA_MIN,
        "lambda_star": lambda_star,
        "bpd_C_own": LOSS_C / (D_C * LN2),
        "bpd_C_8bit": None,
    }
    diagnostics = {
        "n_images": n_images,
        "n_pixels": int(counts.sum()),
        "frac_zero_pixels": float(counts[0] / counts.sum()),
        "mean_logdet_B_at_lam_0.05": logdet_B,
        "mean_logdet_B_at_lambda_star": mean_logdet_B(lambda_star, counts, n_images),
        "target_logdet_for_lambda_star": SCORE_A - SCORE_B,
    }
    return answer, diagnostics


def monte_carlo_check(csv_path: Path, lam: float, seed: int = 0) -> float:
    """Independent MC estimate of L(lam); only used by --verify."""
    rng = np.random.default_rng(seed)
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.int64)
    total, reps = 0.0, 40
    for _ in range(reps):
        u = rng.random(arr.shape)
        y = arr / 255.0 + u / 256.0
        s = lam + (1.0 - 2.0 * lam) * y
        total += float(np.sum(math.log1p(-2.0 * lam) - np.log(s) - np.log1p(-s)))
    return total / (reps * arr.shape[0])


def main() -> None:
    here = Path(__file__).resolve()
    root = here.parents[2]                      # workspace root
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", type=Path, default=root / "data" / "images.csv")
    ap.add_argument("--out", type=Path, default=root / "task" / "results" / "answer.json")
    ap.add_argument("--verify", action="store_true",
                    help="run Monte-Carlo and high-precision cross-checks")
    args = ap.parse_args()

    digest = sha256(args.images)
    print(f"images: {args.images}")
    print(f"sha256: {digest}  (expected match: {digest == EXPECTED_SHA256})")

    answer, diag = compute(args.images)

    print("\ndiagnostics:")
    for k, v in diag.items():
        print(f"  {k}: {v}")

    if args.verify:
        print("\nverification:")
        n_images, counts = load_pixel_histogram(args.images)
        for lam in (LAM_B, answer["lambda_star"]):
            exact = mean_logdet_B(lam, counts, n_images)
            mc = monte_carlo_check(args.images, lam)
            print(f"  lam={lam:.10f}  exact L={exact:.6f}  MC L={mc:.6f}  diff={exact-mc:+.4f}")
        try:
            import mpmath as mp
            mp.mp.dps = 40
            n_images, counts = load_pixel_histogram(args.images)
            for lam in (LAM_B, answer["lambda_star"]):
                L = mp.mpf(0)
                lam_m = mp.mpf(repr(lam))
                b = (1 - 2 * lam_m) / 256
                for k in range(256):
                    if counts[k] == 0:
                        continue
                    a = lam_m + (1 - 2 * lam_m) * mp.mpf(k) / 255
                    term = mp.quad(lambda u: -mp.log(a + b * u) - mp.log(1 - a - b * u),
                                   [0, 1])
                    L += mp.mpf(counts[k]) * term
                L = L / n_images + 784 * mp.log(1 - 2 * lam_m)
                exact = mean_logdet_B(lam, counts, n_images)
                print(f"  lam={lam:.10f}  float64 L={exact!r}  mpmath L={mp.nstr(L, 18)}")
        except ImportError:
            print("  mpmath not installed; skipped high-precision check")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(answer, fh, indent=2)
        fh.write("\n")

    print(f"\nwrote {args.out}")
    print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
