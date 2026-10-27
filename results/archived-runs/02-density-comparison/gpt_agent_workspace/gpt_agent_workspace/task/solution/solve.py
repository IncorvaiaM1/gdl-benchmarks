"""Compute the comparable dequantization bounds requested in TASK.md."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


D = 28 * 28
N_IMAGES = 1000
FLOW_A_SCORE = 1972.0
FLOW_B_SCORE = -150.0
VAE_LOSS = 93.7


def pixel_histogram(csv_path: Path) -> list[int]:
    counts = [0] * 256
    rows = 0
    with csv_path.open(newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        header = next(reader)
        if len(header) != D:
            raise ValueError(f"expected {D} columns, found {len(header)}")
        for row in reader:
            if len(row) != D:
                raise ValueError(f"row {rows + 2} has {len(row)} columns")
            for value in row:
                pixel = int(value)
                if not 0 <= pixel <= 255:
                    raise ValueError(f"pixel outside uint8 range: {pixel}")
                counts[pixel] += 1
            rows += 1
    if rows != N_IMAGES:
        raise ValueError(f"expected {N_IMAGES} images, found {rows}")
    return counts


def xlogx_minus_x(x: float) -> float:
    """Antiderivative of log(x), continuously extended to x=0."""
    if x <= 0.0:
        if x > -1e-14:
            return 0.0
        raise ValueError(f"log argument became negative: {x}")
    return x * math.log(x) - x


def expected_logit_jacobian(pixel: int, lam: float) -> float:
    """E_u log|d logit(s)/du| for one pixel and u ~ Uniform[0,1)."""
    scale = 1.0 - 2.0 * lam
    a = lam + scale * pixel / 255.0
    b = scale / 256.0

    mean_log_s = (xlogx_minus_x(a + b) - xlogx_minus_x(a)) / b
    mean_log_one_minus_s = (
        xlogx_minus_x(1.0 - a) - xlogx_minus_x(1.0 - a - b)
    ) / b
    return math.log(scale / 256.0) - mean_log_s - mean_log_one_minus_s


def mean_image_log_jacobian(counts: list[int], lam: float) -> float:
    total = sum(
        count * expected_logit_jacobian(pixel, lam)
        for pixel, count in enumerate(counts)
    )
    return total / N_IMAGES


def flow_b_bpd(counts: list[int], lam: float) -> float:
    # Jensen: -log P(x) <= -E log q(z) - E log|dz/du|.
    nll_bound = -FLOW_B_SCORE - mean_image_log_jacobian(counts, lam)
    return nll_bound / (D * math.log(2.0))


def find_lambda_star(counts: list[int], lambda_min: float, target: float) -> float:
    def difference(lam: float) -> float:
        return flow_b_bpd(counts, lam) - target

    # Locate the first crossing without assuming monotonicity, then bisect it.
    left = lambda_min
    f_left = difference(left)
    right_limit = 0.5 - 1e-12
    bracket: tuple[float, float] | None = None
    for i in range(1, 20_001):
        right = lambda_min + (right_limit - lambda_min) * i / 20_000
        f_right = difference(right)
        if f_left == 0.0 or f_left * f_right <= 0.0:
            bracket = (left, right)
            break
        left, f_left = right, f_right
    if bracket is None:
        raise ValueError("no lambda_star crossing found")

    left, right = bracket
    f_left = difference(left)
    for _ in range(100):
        middle = (left + right) / 2.0
        f_middle = difference(middle)
        if f_left * f_middle <= 0.0:
            right = middle
        else:
            left, f_left = middle, f_middle
    return (left + right) / 2.0


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    counts = pixel_histogram(project_root / "data" / "images.csv")

    # A's cell has volume 256^-D in y space.
    bpd_a = (D * math.log(256.0) - FLOW_A_SCORE) / (D * math.log(2.0))

    # At this value the supremum for k=255 maps to s=1. Because u<1,
    # every realized input still has s<1; smaller lambdas admit s>1.
    lambda_min = 1.0 / 258.0
    bpd_b = flow_b_bpd(counts, 0.05)
    lambda_star = find_lambda_star(counts, lambda_min, bpd_a)

    answer = {
        "bpd_A": bpd_a,
        "bpd_B": bpd_b,
        "better": "A" if bpd_a < bpd_b else "B",
        "lambda_min": lambda_min,
        "lambda_star": lambda_star,
        # The VAE's BCE is summed over its padded 32x32 binary variable.
        "bpd_C_own": VAE_LOSS / (32 * 32 * math.log(2.0)),
        # Stochastic binarization is many-to-one, so this score does not
        # define probabilities (and hence no bound) for 8-bit images.
        "bpd_C_8bit": None,
    }

    output_path = project_root / "task" / "results" / "answer.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(answer, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
