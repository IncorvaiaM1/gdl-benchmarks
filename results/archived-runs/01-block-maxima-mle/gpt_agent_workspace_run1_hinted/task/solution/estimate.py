"""Estimate the exponential rate from maxima of fixed-size blocks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence


BLOCK_SIZE = 4
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "train.csv"
DEFAULT_OUTPUT = ROOT / "task" / "results" / "estimate.json"


def read_observations(path: Path) -> list[float]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["y"]:
            raise ValueError("input must have exactly one column named 'y'")
        values = [float(row["y"]) for row in reader]

    if not values:
        raise ValueError("input contains no observations")
    if any(not math.isfinite(y) or y <= 0.0 for y in values):
        raise ValueError("all observations must be finite and strictly positive")
    return values


def score(rate: float, values: Sequence[float], block_size: int) -> float:
    """Derivative of the block-maximum log-likelihood."""
    if rate <= 0.0:
        raise ValueError("rate must be strictly positive")

    correction = 0.0
    for y in values:
        z = rate * y
        # The summand tends to zero; avoiding expm1 overflow preserves that.
        if z < 700.0:
            correction += y / math.expm1(z)
    return len(values) / rate - math.fsum(values) + (block_size - 1) * correction


def maximum_likelihood_rate(
    values: Sequence[float],
    block_size: int = BLOCK_SIZE,
    relative_tolerance: float = 1e-13,
) -> float:
    """Find the unique positive score root by bracketed bisection."""
    if block_size < 1:
        raise ValueError("block_size must be at least one")

    total = math.fsum(values)
    lower = 0.0
    upper = block_size * len(values) / total
    while score(upper, values, block_size) > 0.0:
        upper *= 2.0

    for _ in range(200):
        midpoint = (lower + upper) / 2.0
        if score(midpoint, values, block_size) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= relative_tolerance * midpoint:
            break

    estimate = (lower + upper) / 2.0
    if not math.isfinite(estimate) or estimate <= 0.0:
        raise ArithmeticError("MLE solver did not produce a valid positive rate")
    return estimate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = read_observations(args.input)
    estimate = maximum_likelihood_rate(values, args.block_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"lambda_hat": estimate}) + "\n",
        encoding="utf-8",
    )
    print(f"lambda_hat={estimate:.15g}")


if __name__ == "__main__":
    main()
