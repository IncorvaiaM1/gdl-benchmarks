"""INSTRUCTOR-ONLY gold solution for task/TASK.md. Not shipped to the solver.

Two independent routes to every answer:

1. Direct: build p and q_k as arrays, form the optimal discriminators
   D*(x) = p/(p+q) and D_LS*(x) = (p-q)/(p+q), and evaluate every quantity
   by summation, with exact Fractions wherever no log is involved.
2. Closed form: for two nested uniforms with support-size ratio
   r = max(k,4)/min(k,4) >= 1,
       JSD(r)  = log 2 + 1/2 log r - 1/2 (1 + 1/r) log(1 + r),
       V_LS(r) = (r - 1)/(r + 1),
   both strictly increasing in r (dJSD/dr = log(1+r)/(2 r^2) > 0), and
   symmetric in which distribution is the larger one. So two distinct k tie
   iff they have the same r, i.e. one is below 4, one above, and j*k = 16:
   the pairs are {1,16} and {2,8}.

The direct route scans k = 1..K_SCAN and must agree with the closed form.
"""
import math
from fractions import Fraction
from itertools import combinations

N_DATA = 4          # p = uniform on {0,...,N_DATA-1}
K_REF = 2           # the generator whose JSD partner defines k_hat
K_SCAN = 512        # direct-scan range for the tie search (closed form covers all k)
INF = float("inf")


def dists(k):
    n = max(N_DATA, k)
    p = [Fraction(1, N_DATA) if x < N_DATA else Fraction(0) for x in range(n)]
    q = [Fraction(1, k) if x < k else Fraction(0) for x in range(n)]
    return p, q


def _xlogy(a, b):
    """a * log(b) with the conventions 0*log(anything) = 0, log 0 = -inf."""
    if a == 0:
        return 0.0
    if b == 0:
        return -INF
    return float(a) * math.log(b)


def jsd(k):
    p, q = dists(k)
    m = [(a + b) / 2 for a, b in zip(p, q)]
    kl_pm = sum(_xlogy(a, Fraction(a, 1) / mm) if a else 0.0 for a, mm in zip(p, m))
    kl_qm = sum(_xlogy(b, Fraction(b, 1) / mm) if b else 0.0 for b, mm in zip(q, m))
    return 0.5 * kl_pm + 0.5 * kl_qm


def v_ls(k):
    """Exact Fraction. D_LS* = (b p + a q)/(p + q) = (p - q)/(p + q); c = 0."""
    p, q = dists(k)
    total = Fraction(0)
    for a, b in zip(p, q):
        if a + b == 0:
            continue
        d = (a - b) / (a + b)
        total += Fraction(1, 2) * (a + b) * d * d
    return total


def gan_terms(k):
    """V_star, L_MM, L_NS with D* = p/(p+q)."""
    p, q = dists(k)
    v_star = l_mm = l_ns = 0.0
    for a, b in zip(p, q):
        if a + b == 0:
            continue
        d = a / (a + b)
        v_star += _xlogy(a, d) + _xlogy(b, 1 - d)
        l_mm += _xlogy(b, 1 - d)
        l_ns += -_xlogy(b, d)
    return {"V_star": v_star, "L_MM": l_mm, "L_NS": l_ns}


# ---- closed forms -------------------------------------------------------
def _ratio(k):
    return max(k, N_DATA) / min(k, N_DATA)


def jsd_closed(k):
    r = _ratio(k)
    return math.log(2) + 0.5 * math.log(r) - 0.5 * (1 + 1 / r) * math.log(1 + r)


def v_ls_closed(k):
    r = Fraction(max(k, N_DATA), min(k, N_DATA))
    return (r - 1) / (r + 1)


def closed_form_pairs():
    """Same r <=> j*k = N_DATA^2 with j < N_DATA < k."""
    sq = N_DATA * N_DATA
    return [[j, sq // j] for j in range(1, N_DATA) if sq % j == 0]


def _scan_pairs(values, tol):
    ks = sorted(values)
    out = []
    for j, k in combinations(ks, 2):
        if abs(values[j] - values[k]) <= tol:
            out.append([j, k])
    return out


def _enc(x):
    if x == INF:
        return "inf"
    if x == -INF:
        return "-inf"
    return x


def _prefer(a, b, tol=1e-12):
    if a == b or (math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol):
        return "tie"
    return "q2" if a < b else "q_k_hat"


def solve(k_scan=K_SCAN):
    ks = range(1, k_scan + 1)

    # Tie search, direct route (exact for V_LS; floats for JSD).
    jsd_vals = {k: jsd(k) for k in ks}
    ls_vals = {k: v_ls(k) for k in ks}
    jsd_pairs = _scan_pairs(jsd_vals, tol=1e-12)
    ls_pairs = [[j, k] for j, k in combinations(sorted(ls_vals), 2) if ls_vals[j] == ls_vals[k]]

    # Cross-check against the closed form (which covers every k, not just the scan).
    cf = closed_form_pairs()
    assert jsd_pairs == cf, (jsd_pairs, cf)
    assert ls_pairs == cf, (ls_pairs, cf)
    for k in ks:
        assert abs(jsd_vals[k] - jsd_closed(k)) < 1e-12, k
        assert ls_vals[k] == v_ls_closed(k), k

    partners = [b if a == K_REF else a for a, b in jsd_pairs if K_REF in (a, b)]
    k_hat = partners[0] if len(partners) == 1 else None

    g2 = gan_terms(K_REF)
    out = {
        "jsd_tie_pairs": jsd_pairs,
        "ls_tie_pairs": ls_pairs,
        "ls_value_q2": float(ls_vals[K_REF]),
        "k_hat": k_hat,
        "q2": {key: _enc(v) for key, v in g2.items()},
        "q_k_hat": None,
        "prefers": None,
    }
    if k_hat is not None:
        gk = gan_terms(k_hat)
        out["q_k_hat"] = {key: _enc(v) for key, v in gk.items()}
        out["prefers"] = {key: _prefer(g2[key], gk[key]) for key in g2}
    return out


# Closed-form values for the write-up (independent of the code above).
GOLD_SYMBOLIC = {
    "jsd_tie_pairs": "[[1,16],[2,8]]",
    "ls_tie_pairs": "[[1,16],[2,8]]",
    "ls_value_q2": "1/3",
    "k_hat": "8",
    "q2.V_star": "log 2 - (3/2) log 3",
    "q2.L_MM": "log(2/3)",
    "q2.L_NS": "log 3",
    "q8.V_star": "log 2 - (3/2) log 3",
    "q8.L_MM": "-(1/2) log 3",
    "q8.L_NS": "+inf",
    "prefers": "V_star: tie, L_MM: q_k_hat (q8), L_NS: q2",
}


if __name__ == "__main__":
    import json

    ans = solve()
    print(json.dumps(ans, indent=2))
    # Symbolic sanity checks.
    L2, L3 = math.log(2), math.log(3)
    assert abs(ans["q2"]["V_star"] - (L2 - 1.5 * L3)) < 1e-12
    assert abs(ans["q_k_hat"]["V_star"] - (L2 - 1.5 * L3)) < 1e-12
    assert abs(ans["q2"]["L_MM"] - math.log(2 / 3)) < 1e-12
    assert abs(ans["q_k_hat"]["L_MM"] + 0.5 * L3) < 1e-12
    assert abs(ans["q2"]["L_NS"] - L3) < 1e-12
    assert ans["q_k_hat"]["L_NS"] == "inf"
    assert abs(ans["ls_value_q2"] - 1 / 3) < 1e-15
    # V_star(q) = 2 JSD(p,q) - log 4 (Goodfellow et al. 2014, Thm 1).
    for k in (1, 2, 3, 5, 8, 16):
        assert abs(gan_terms(k)["V_star"] - (2 * jsd(k) - math.log(4))) < 1e-12
    print("all symbolic checks passed")
