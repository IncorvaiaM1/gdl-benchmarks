"""Validation harness for estimate_lambda.py.

Nothing here touches data/holdout.csv. The checks are:

1. analytic score vs. a central finite difference of the log-likelihood;
2. m = 1 reduces to the closed-form exponential MLE 1/mean(y);
3. recovery of a known lambda from simulated block maxima (bias shrinks with n);
4. the naive 1/mean(y) estimator is biased low by the expected factor H_m;
5. split-half and bootstrap stability of the estimate on data/train.csv.

Exits non-zero if any check fails.
"""

from __future__ import annotations

import math

import numpy as np

from estimate_lambda import (
    DEFAULT_DATA,
    M_DEFAULT,
    load_y,
    loglik,
    mle_lambda,
    observed_information,
    score,
)

SEED = 20260906
failures: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


def simulate(lam: float, m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """n draws of max(m i.i.d. Exponential(lam))."""
    return rng.exponential(scale=1.0 / lam, size=(n, m)).max(axis=1)


def main() -> int:
    rng = np.random.default_rng(SEED)
    y = load_y(DEFAULT_DATA)
    m = M_DEFAULT
    harmonic_m = float(np.sum(1.0 / np.arange(1, m + 1)))

    # 1. score == numerical derivative of loglik
    worst = 0.0
    for lam in (0.3, 1.0, 2.0, 5.0):
        h = 1e-6 * lam
        fd = (loglik(lam + h, y, m) - loglik(lam - h, y, m)) / (2 * h)
        an = score(lam, y, m)
        worst = max(worst, abs(fd - an) / max(1.0, abs(an)))
    check("score matches finite difference", worst < 1e-5, f"max rel err {worst:.2e}")

    # 2. m = 1 collapses to the closed form
    y1 = simulate(1.7, 1, 5000, rng)
    got = mle_lambda(y1, 1)["lambda_hat"]
    want = 1.0 / float(y1.mean())
    check("m=1 equals 1/mean(y)", math.isclose(got, want, rel_tol=1e-12), f"{got:.10f} vs {want:.10f}")

    # 3. recover a known lambda; error should fall roughly as 1/sqrt(n)
    lam_true = 1.9
    for n in (2_000, 200_000):
        est = mle_lambda(simulate(lam_true, m, n, rng), m)["lambda_hat"]
        tol = 6.0 / math.sqrt(n)  # generous multiple of the sampling s.e.
        check(
            f"recovers lambda={lam_true} at n={n}",
            abs(est - lam_true) < tol,
            f"est {est:.6f}, |err| {abs(est - lam_true):.6f} < tol {tol:.6f}",
        )

    # 4. the naive estimator is biased low by ~H_m
    ysim = simulate(lam_true, m, 200_000, rng)
    ratio = lam_true / (1.0 / float(ysim.mean()))
    check(
        "naive 1/mean(y) low by factor H_m",
        abs(ratio - harmonic_m) < 0.02,
        f"lam_true / naive = {ratio:.4f}, H_{m} = {harmonic_m:.4f}",
    )

    # 5. stability on the real data: split-half and bootstrap
    res = mle_lambda(y, m)
    lam_hat = res["lambda_hat"]
    se = 1.0 / math.sqrt(observed_information(lam_hat, y, m))
    perm = rng.permutation(y.size)
    half_a = mle_lambda(y[perm[: y.size // 2]], m)["lambda_hat"]
    half_b = mle_lambda(y[perm[y.size // 2 :]], m)["lambda_hat"]
    check(
        "split-half estimates agree within 4 s.e.",
        abs(half_a - half_b) < 4 * se * math.sqrt(2) * math.sqrt(2),
        f"{half_a:.6f} vs {half_b:.6f} (full {lam_hat:.6f}, s.e. {se:.6f})",
    )

    boot = np.array(
        [mle_lambda(rng.choice(y, size=y.size, replace=True), m)["lambda_hat"] for _ in range(200)]
    )
    check(
        "bootstrap s.d. matches observed information",
        0.5 < boot.std(ddof=1) / se < 2.0,
        f"bootstrap sd {boot.std(ddof=1):.6f} vs analytic s.e. {se:.6f}",
    )
    print(f"\nlambda_hat on {DEFAULT_DATA.name}: {lam_hat:.10f} (s.e. {se:.6f})")
    print(f"bootstrap 95% percentile CI: ({np.quantile(boot, 0.025):.6f}, {np.quantile(boot, 0.975):.6f})")

    if failures:
        print(f"\n{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
