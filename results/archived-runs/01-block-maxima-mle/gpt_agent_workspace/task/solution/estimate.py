"""Compute the MLE rate for maxima of iid exponential observations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = WORKSPACE_ROOT / "data" / "train.csv"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "task" / "results" / "estimate.json"


def read_observations(path: Path) -> list[float]:
    """Read and validate the positive observations in the CSV's y column."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["y"]:
            raise ValueError(f"expected exactly one column named 'y', got {reader.fieldnames}")
        observations = [float(row["y"]) for row in reader]

    if not observations:
        raise ValueError("input contains no observations")
    if any(not math.isfinite(y) or y <= 0.0 for y in observations):
        raise ValueError("all observations must be finite and strictly positive")
    return observations


def score(rate: float, observations: Sequence[float], block_size: int) -> float:
    """Derivative of the log-likelihood with respect to the rate."""
    reciprocal_terms = 0.0
    for y in observations:
        z = rate * y
        # y / expm1(z) tends rapidly to zero, while expm1 overflows near z=710.
        if z < 700.0:
            reciprocal_terms += y / math.expm1(z)
    return (
        len(observations) / rate
        - math.fsum(observations)
        + (block_size - 1) * reciprocal_terms
    )


def maximum_likelihood_rate(
    observations: Sequence[float],
    block_size: int = 4,
    relative_tolerance: float = 1e-13,
) -> float:
    """Find the unique positive score root by bracket expansion and bisection."""
    if block_size < 1:
        raise ValueError("block_size must be a positive integer")

    # The ordinary exponential MLE is below the block-maximum MLE and provides
    # a positive, data-scaled starting point for bracket construction.
    lower = len(observations) / math.fsum(observations)
    while score(lower, observations, block_size) <= 0.0:
        lower /= 2.0

    upper = lower * block_size
    while score(upper, observations, block_size) >= 0.0:
        upper *= 2.0

    while upper - lower > relative_tolerance * max(1.0, lower, upper):
        midpoint = (lower + upper) / 2.0
        if score(midpoint, observations, block_size) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--block-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observations = read_observations(args.input)
    estimate = maximum_likelihood_rate(observations, args.block_size)
    if not math.isfinite(estimate) or estimate <= 0.0:
        raise ArithmeticError("computed estimate is not finite and positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"lambda_hat": estimate}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"lambda_hat={estimate:.15g}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
