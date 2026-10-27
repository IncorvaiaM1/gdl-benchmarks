"""Reproduce the requested GAN objective values using closed-form formulas."""

from __future__ import annotations

import json
import math
from pathlib import Path


DATA_SUPPORT_SIZE = 4


def jsd(k: int) -> float:
    """JSD between uniforms on nested supports of sizes 4 and k."""
    n = DATA_SUPPORT_SIZE
    if k <= n:
        kl_p = (k / n) * math.log(2 * k / (k + n))
        kl_p += ((n - k) / n) * math.log(2)
        kl_q = math.log(2 * n / (k + n))
    else:
        kl_p = math.log(2 * k / (k + n))
        kl_q = (n / k) * math.log(2 * n / (k + n))
        kl_q += ((k - n) / k) * math.log(2)
    return 0.5 * (kl_p + kl_q)


def ls_value(k: int) -> float:
    """Optimized least-squares criterion."""
    return abs(k - DATA_SUPPORT_SIZE) / (k + DATA_SUPPORT_SIZE)


def exact_cross_side_ties() -> list[list[int]]:
    """Both criteria tie across k=4 exactly when the two indices multiply to 16."""
    n_squared = DATA_SUPPORT_SIZE**2
    return [
        [j, n_squared // j]
        for j in range(1, DATA_SUPPORT_SIZE)
        if n_squared % j == 0
    ]


def gan_values(k: int) -> dict[str, float | str]:
    n = DATA_SUPPORT_SIZE
    overlap = min(n, k)
    d_overlap = k / (k + n)

    e_p_log_d = (overlap / n) * math.log(d_overlap)
    e_q_log_one_minus_d = (overlap / k) * math.log(1 - d_overlap)
    v_star = e_p_log_d + e_q_log_one_minus_d
    l_mm = e_q_log_one_minus_d
    l_ns: float | str = (
        -math.log(d_overlap) if k <= n else "inf"
    )
    return {"V_star": v_star, "L_MM": l_mm, "L_NS": l_ns}


def main() -> None:
    tie_pairs = exact_cross_side_ties()
    k_hat = next(k for j, k in tie_pairs if j == 2)
    q2 = gan_values(2)
    q_k_hat = gan_values(k_hat)

    answer = {
        "jsd_tie_pairs": tie_pairs,
        "ls_tie_pairs": tie_pairs,
        "ls_value_q2": ls_value(2),
        "k_hat": k_hat,
        "q2": q2,
        "q_k_hat": q_k_hat,
        "prefers": {
            "V_star": "tie",
            "L_MM": "q_k_hat",
            "L_NS": "q2",
        },
    }

    assert math.isclose(jsd(2), jsd(k_hat), rel_tol=0, abs_tol=1e-15)
    output = Path(__file__).resolve().parents[1] / "results" / "answer.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(answer, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
