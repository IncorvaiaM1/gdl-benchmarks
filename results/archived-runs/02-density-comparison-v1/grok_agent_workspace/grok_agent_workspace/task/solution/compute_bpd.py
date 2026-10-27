"""Compare two dequantized flow scores in bits per dimension.

Model A reports E[log p_A(y)] on y = v / 256.
Model B reports E[log p_B(y')] on y' = logit(lambda + (1 - 2*lambda) * y).

Both scores are converted, by change of variables, into the density p_V of
the unscaled dequantized pixels v = x + u in [0, 256)^D. Bits per dimension
are

    bpd = -E[log p_V(v)] / (D * ln 2).

The Model A Jacobian is constant. The Model B Jacobian depends on v, and its
expectation over u ~ Uniform[0, 1) is computed in closed form for each
integer pixel value, then averaged over data/images.csv.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "images.csv"
ANSWER_PATH = ROOT / "task" / "results" / "answer.json"

D_EXPECTED = 784
N_EXPECTED = 1000
ELL_A = 1972.0
ELL_B = -150.0
LAM = 0.05
ALPHA = 1.0 - 2.0 * LAM  # 0.9
LEVELS = 256


def expected_log_s_one_minus_s(pixel: int) -> float:
    """Exact E_{u~Unif[0,1)}[log(s(1-s))] for one integer pixel.

    s(u) = lam + alpha * (pixel + u) / 256, which moves linearly from s0 to
    s1 = s0 + c over an interval of length c = alpha / 256. The integrals of
    log(s) and log(1-s) have elementary antiderivatives.
    """
    c = ALPHA / LEVELS
    s0 = LAM + c * pixel
    s1 = s0 + c
    log_s = (s1 * math.log(s1) - s0 * math.log(s0)) / c - 1.0
    log_one_minus_s = (
        (1.0 - s0) * math.log(1.0 - s0) - (1.0 - s1) * math.log(1.0 - s1)
    ) / c - 1.0
    return log_s + log_one_minus_s


def simpson_log_s_one_minus_s(pixel: int, intervals: int = 20000) -> float:
    """Composite Simpson check of the same one-dimensional expectation."""
    if intervals % 2 != 0:
        raise ValueError("Simpson's rule needs an even number of intervals")
    c = ALPHA / LEVELS
    h = 1.0 / intervals

    def integrand(u: float) -> float:
        s = LAM + c * (pixel + u)
        return math.log(s) + math.log(1.0 - s)

    total = integrand(0.0) + integrand(1.0)
    for i in range(1, intervals):
        weight = 4.0 if i % 2 else 2.0
        total += weight * integrand(i * h)
    return total * h / 3.0


def verify_closed_form(tol: float = 1e-10) -> float:
    """Return the max absolute gap between the closed form and Simpson."""
    worst = 0.0
    for pixel in range(LEVELS):
        gap = abs(
            expected_log_s_one_minus_s(pixel)
            - simpson_log_s_one_minus_s(pixel)
        )
        if gap > worst:
            worst = gap
    if worst > tol:
        raise RuntimeError(f"closed form disagrees with Simpson by {worst}")
    return worst


def load_pixel_counts(path: Path) -> list[int]:
    counts = [0] * LEVELS
    n_images = 0
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) != D_EXPECTED or header[0] != "p0" or header[-1] != "p783":
            raise RuntimeError(f"unexpected header shape: {len(header)} columns")
        for row in reader:
            if len(row) != D_EXPECTED:
                raise RuntimeError(f"row {n_images} has {len(row)} columns")
            n_images += 1
            for value in row:
                pixel = int(value)
                if pixel < 0 or pixel >= LEVELS:
                    raise RuntimeError(f"pixel {pixel} outside 0..255")
                counts[pixel] += 1
    if n_images != N_EXPECTED:
        raise RuntimeError(f"expected {N_EXPECTED} images, found {n_images}")
    if sum(counts) != n_images * D_EXPECTED:
        raise RuntimeError("pixel count does not equal N * D")
    return counts


def main() -> None:
    quadrature_gap = verify_closed_form()
    counts = load_pixel_counts(DATA_PATH)
    n_images = N_EXPECTED
    n_dims = D_EXPECTED

    # J = E_{x,u}[sum_d log(s_d (1 - s_d))]
    weighted = math.fsum(
        counts[pixel] * expected_log_s_one_minus_s(pixel) for pixel in range(LEVELS)
    )
    jacobian_expectation = weighted / n_images

    log_256 = math.log(LEVELS)
    log_2 = math.log(2.0)
    denom = n_dims * log_2

    # y = v / 256, so log p_V = log p_A - D log 256.
    bpd_a = -(ELL_A - n_dims * log_256) / denom

    # y' = logit(s(v)), so
    # log p_V = log p_B + D log(alpha) - D log 256 - sum log(s(1-s)).
    bpd_b = -(
        ELL_B + n_dims * math.log(ALPHA) - n_dims * log_256 - jacobian_expectation
    ) / denom

    better = "A" if bpd_a < bpd_b else "B"
    answer = {"bpd_A": bpd_a, "bpd_B": bpd_b, "better": better}

    ANSWER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANSWER_PATH.write_text(json.dumps(answer) + "\n", encoding="utf-8")

    print(f"quadrature_gap {quadrature_gap:.3e}")
    print(f"jacobian_expectation {jacobian_expectation:.12f}")
    print(f"bpd_A {bpd_a:.12f}")
    print(f"bpd_B {bpd_b:.12f}")
    print(f"better {better}")
    print(ANSWER_PATH.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
