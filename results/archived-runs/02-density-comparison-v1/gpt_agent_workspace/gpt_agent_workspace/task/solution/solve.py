"""Compute comparable bits/dimension scores for the two flow models.

For Model A, v -> y contributes -D*log(256) to log p_V.

For Model B, with a = 1 - 2*lambda and
    s = lambda + a*v/256,
the elementwise derivative is
    d logit(s) / dv = a / (256*s*(1-s)).
Thus
    E[log p_V] = ell_B + D*log(a/256)
                 - E[sum_d(log(s_d) + log(1-s_d))].

The final expectation is evaluated exactly over uniform dequantization noise
by integrating each pixel interval analytically.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


D = 784
ELL_A = 1972.0
ELL_B = -150.0
LAMBDA = 0.05
A = 1.0 - 2.0 * LAMBDA

TASK_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = TASK_DIR.parent / "data" / "images.csv"
ANSWER_FILE = TASK_DIR / "results" / "answer.json"


def antiderivative(s: float) -> float:
    """Antiderivative of log(s) + log(1-s) with respect to s."""
    return (
        s * math.log(s)
        - s
        - (1.0 - s) * math.log1p(-s)
        + (1.0 - s)
    )


def expected_log_s_terms() -> tuple[float, int]:
    """Return E[sum(log(s)+log(1-s))] and the number of images."""
    ds_du = A / 256.0
    total = 0.0
    count = 0

    with DATA_FILE.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) != D:
            raise ValueError(f"expected {D} columns, found {len(header)}")

        for row in reader:
            if len(row) != D:
                raise ValueError(
                    f"row {count + 2}: expected {D} values, found {len(row)}"
                )
            image_integral = 0.0
            for text in row:
                x = int(text)
                if not 0 <= x <= 255:
                    raise ValueError(f"pixel outside [0, 255]: {x}")
                s0 = LAMBDA + A * x / 256.0
                s1 = s0 + ds_du
                image_integral += (
                    antiderivative(s1) - antiderivative(s0)
                ) / ds_du
            total += image_integral
            count += 1

    if count == 0:
        raise ValueError("dataset is empty")
    return total / count, count


def main() -> None:
    expected_logs, image_count = expected_log_s_terms()

    expected_log_pv_a = ELL_A - D * math.log(256.0)
    expected_log_pv_b = (
        ELL_B
        + D * math.log(A / 256.0)
        - expected_logs
    )

    bpd_a = -expected_log_pv_a / (D * math.log(2.0))
    bpd_b = -expected_log_pv_b / (D * math.log(2.0))
    answer = {
        "bpd_A": bpd_a,
        "bpd_B": bpd_b,
        "better": "A" if bpd_a < bpd_b else "B",
    }

    ANSWER_FILE.parent.mkdir(parents=True, exist_ok=True)
    ANSWER_FILE.write_text(
        json.dumps(answer, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    print(f"images={image_count}, dimensions={D}")
    print(f"E[sum(log(s)+log(1-s))]={expected_logs:.15f}")
    print(f"E[log p_V,A]={expected_log_pv_a:.15f}")
    print(f"E[log p_V,B]={expected_log_pv_b:.15f}")
    print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
