"""Entry point: python task/solution/solve.py  (writes task/results/answer.json).

Standard library only. Quantities are computed directly from the definitions
on the support of p + q_k, using exact Fractions for densities, D*, and V_LS.
Tie lists are found by an exhaustive scan over k <= K_MAX, which is complete
because both JSD and V_LS are strictly monotone functions of
r = min(k, 4) / max(k, 4) (see DERIVATION.md).
"""
import json
import math
from fractions import Fraction
from pathlib import Path

N_P = 4
K_MAX = 400


def p(x):
    return Fraction(1, N_P) if 0 <= x < N_P else Fraction(0)


def q(k, x):
    return Fraction(1, k) if 0 <= x < k else Fraction(0)


def support(k):
    return range(max(k, N_P))


def xlogy_ratio(a, b):
    """a * log(a / b) with 0 * log(...) = 0."""
    return 0.0 if a == 0 else float(a) * math.log(a / b)


def jsd(k):
    total = 0.0
    for x in support(k):
        px, qx = p(x), q(k, x)
        m = (px + qx) / 2
        total += 0.5 * xlogy_ratio(px, m) + 0.5 * xlogy_ratio(qx, m)
    return total


def v_ls(k):
    """Exact V_LS with D_LS* = (p - q) / (p + q) on supp(p + q)."""
    total = Fraction(0)
    for x in support(k):
        px, qx = p(x), q(k, x)
        d = (px - qx) / (px + qx)
        total += Fraction(1, 2) * (px + qx) * d * d
    return total


def safe_log(v):
    return -math.inf if v == 0 else math.log(v)


def expect(weights_and_vals):
    """Expectation with conventions 0 * (+-inf) = 0 and +inf if any mass on +inf."""
    total = 0.0
    for w, val in weights_and_vals:
        if w == 0:
            continue
        if math.isinf(val):
            return val
        total += float(w) * val
    return total


def minimax_terms(k):
    d = {x: p(x) / (p(x) + q(k, x)) for x in support(k)}
    e_p_logd = expect((p(x), safe_log(d[x])) for x in support(k))
    e_q_log1md = expect((q(k, x), safe_log(1 - d[x])) for x in support(k))
    e_q_neglogd = expect((q(k, x), -safe_log(d[x])) for x in support(k))
    return {"V_star": e_p_logd + e_q_log1md, "L_MM": e_q_log1md, "L_NS": e_q_neglogd}


def ratio(k):
    return Fraction(min(k, N_P), max(k, N_P))


def jsd_closed(r):
    r = float(r)
    return math.log(2) - (1 + r) / 2 * math.log(1 + r) + (r / 2 * math.log(r))


def vls_closed(r):
    return (1 - r) / (1 + r)


def tie_pairs(key):
    seen = {}
    for k in range(1, K_MAX + 1):
        seen.setdefault(key(k), []).append(k)
    pairs = []
    for ks in seen.values():
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                pairs.append([ks[i], ks[j]])
    return sorted(pairs)


def fmt(v):
    if v == math.inf:
        return "inf"
    if v == -math.inf:
        return "-inf"
    return float(v)


def main():
    for k in range(1, K_MAX + 1):
        assert abs(jsd(k) - jsd_closed(ratio(k))) < 1e-13, k
        assert v_ls(k) == vls_closed(ratio(k)), k

    # Both criteria are strictly decreasing in r, so ties <=> equal r (exact Fraction keys).
    jsd_pairs = tie_pairs(ratio)
    ls_pairs = tie_pairs(v_ls)
    # Cross-check the JSD ties against direct float evaluation.
    for j, k in jsd_pairs:
        assert abs(jsd(j) - jsd(k)) < 1e-14
    jsd_vals = {k: jsd(k) for k in range(1, K_MAX + 1)}
    near = [(j, k) for j in range(1, K_MAX + 1) for k in range(j + 1, K_MAX + 1)
            if abs(jsd_vals[j] - jsd_vals[k]) < 1e-9]
    assert sorted(map(list, near)) == jsd_pairs, near

    partners = [b if a == 2 else a for a, b in jsd_pairs if 2 in (a, b)]
    k_hat = partners[0] if partners else None

    answer = {
        "jsd_tie_pairs": jsd_pairs,
        "ls_tie_pairs": ls_pairs,
        "ls_value_q2": float(v_ls(2)),
        "k_hat": k_hat,
        "q2": None,
        "q_k_hat": None,
        "prefers": None,
    }
    t2 = minimax_terms(2)
    answer["q2"] = {name: fmt(v) for name, v in t2.items()}
    if k_hat is not None:
        tk = minimax_terms(k_hat)
        answer["q_k_hat"] = {name: fmt(v) for name, v in tk.items()}
        prefers = {}
        for name in ("V_star", "L_MM", "L_NS"):
            a, b = t2[name], tk[name]
            if a == b or (math.isfinite(a) and math.isfinite(b) and abs(a - b) < 1e-12):
                prefers[name] = "tie"
            else:
                prefers[name] = "q2" if a < b else "q_k_hat"
        answer["prefers"] = prefers

    out = Path(__file__).resolve().parents[1] / "results" / "answer.json"
    out.write_text(json.dumps(answer, indent=2) + "\n")
    print(json.dumps(answer, indent=2))
    print("JSD(q2) =", repr(jsd(2)), " JSD(q8) =", repr(jsd(8)), " JSD(q1)=", repr(jsd(1)), " JSD(q16)=", repr(jsd(16)))


if __name__ == "__main__":
    main()
