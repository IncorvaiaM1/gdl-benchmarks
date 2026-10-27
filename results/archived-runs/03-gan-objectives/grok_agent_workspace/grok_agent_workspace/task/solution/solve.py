"""Reproduce the GAN tie lists and loss values for this task.

Run from anywhere:

    python task/solution/solve.py

The script writes task/results/answer.json. It uses only the standard library.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_EVEN, getcontext
from fractions import Fraction
from pathlib import Path


N = 4  # p is uniform on {0, 1, ..., N - 1}
getcontext().prec = 80


class LogForm:
    """Value of a rational linear combination of logarithms of primes.

    Two forms are equal exactly when their prime coefficients agree, because
    a rational relation among ln(p) would make an integer equal to 1 with a
    nontrivial prime factorization.
    """

    def __init__(self) -> None:
        self.coeff: dict[int, Fraction] = defaultdict(Fraction)

    def add_log(self, weight: Fraction, value: Fraction) -> None:
        if weight == 0 or value == 1:
            return
        if value <= 0:
            raise ValueError("logarithm of a nonpositive rational")
        self._add_integer(weight, value.numerator)
        self._add_integer(-weight, value.denominator)

    def _add_integer(self, weight: Fraction, n: int) -> None:
        if n < 1:
            raise ValueError(n)
        x = n
        p = 2
        while p * p <= x:
            while x % p == 0:
                self.coeff[p] += weight
                x //= p
            p += 1 if p == 2 else 2
        if x > 1:
            self.coeff[x] += weight

    def signature(self) -> tuple[tuple[int, Fraction], ...]:
        return tuple(sorted((p, c) for p, c in self.coeff.items() if c != 0))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogForm):
            return NotImplemented
        return self.signature() == other.signature()

    def evaluate(self) -> Decimal:
        total = Decimal(0)
        for prime, weight in self.signature():
            total += Decimal(weight.numerator) / Decimal(weight.denominator) * Decimal(prime).ln()
        return total


def support_masses(k: int) -> dict[int, tuple[Fraction, Fraction]]:
    """Nonzero (p(x), q_k(x)) pairs. x runs over the support of p + q_k."""
    masses: dict[int, tuple[Fraction, Fraction]] = {}
    for x in range(max(N, k)):
        p = Fraction(1, N) if x < N else Fraction(0)
        q = Fraction(1, k) if x < k else Fraction(0)
        if p != 0 or q != 0:
            masses[x] = (p, q)
    return masses


def jsd_form(k: int) -> LogForm:
    form = LogForm()
    for p, q in support_masses(k).values():
        m = (p + q) / 2
        if p > 0:
            form.add_log(Fraction(1, 2) * p, p / m)
        if q > 0:
            form.add_log(Fraction(1, 2) * q, q / m)
    return form


def v_ls_fraction(k: int) -> Fraction:
    total = Fraction(0)
    for p, q in support_masses(k).values():
        d = (p - q) / (p + q)
        total += (p + q) * d * d
    return total / 2


def closed_form_v_ls(k: int) -> Fraction:
    return Fraction(abs(k - N), k + N)


def gan_terms(k: int) -> dict[str, Decimal | str]:
    """V_star, L_MM, and L_NS from D* = p / (p + q), summed on the support."""
    v_star = Decimal(0)
    l_mm = Decimal(0)
    l_ns = Decimal(0)
    ns_infinite = False
    for p, q in support_masses(k).values():
        d = p / (p + q)
        one_minus = 1 - d
        if p > 0:
            # D* = 1 only where q = 0, and ln 1 = 0.
            if d != 1:
                v_star += _frac_to_decimal(p) * _ln_frac(d)
        if q > 0:
            l_mm += _frac_to_decimal(q) * _ln_frac(one_minus)
            v_star += _frac_to_decimal(q) * _ln_frac(one_minus)
            if d == 0:
                ns_infinite = True
            else:
                l_ns += _frac_to_decimal(q) * (-_ln_frac(d))
    return {
        "V_star": v_star,
        "L_MM": l_mm,
        "L_NS": "inf" if ns_infinite else l_ns,
    }


def _frac_to_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _ln_frac(value: Fraction) -> Decimal:
    return Decimal(value.numerator).ln() - Decimal(value.denominator).ln()


def closed_form_losses(k: int) -> dict[str, LogForm | str]:
    """Closed forms derived in DERIVATION.md, as exact log-linear forms.

    L_NS is the string "inf" when q_k charges a point where D* = 0.
    """
    v_star = LogForm()
    l_mm = LogForm()
    if k <= N:
        # D* = k/(k+N) on the support of q, and D* = 1 off it.
        l_mm.add_log(Fraction(1), Fraction(N, k + N))
        l_ns: LogForm | str = LogForm()
        l_ns.add_log(Fraction(-1), Fraction(k, k + N))
        v_star.add_log(Fraction(k, N), Fraction(k, k + N))
        v_star.add_log(Fraction(1), Fraction(N, k + N))
    else:
        # D* = 0 on {N, ..., k-1}, which carries positive q-mass.
        l_mm.add_log(Fraction(N, k), Fraction(N, k + N))
        l_ns = "inf"
        v_star.add_log(Fraction(1), Fraction(k, k + N))
        v_star.add_log(Fraction(N, k), Fraction(N, k + N))
    return {"V_star": v_star, "L_MM": l_mm, "L_NS": l_ns}


def ls_tie_pairs(bound: int | None = None) -> list[list[int]]:
    """All distinct positive integers with equal V_LS.

    V_LS(q_k) = |k-4|/(k+4) is strictly decreasing on 1..4 and strictly
    increasing on 4,5,6,..., and V_LS(q_k) > V_LS(q_1) for every k > N*N.
    Comparing the exact rationals through k = N*N is therefore exhaustive.
    """
    limit = N * N if bound is None else bound
    values = {k: v_ls_fraction(k) for k in range(1, limit + 1)}
    pairs: list[list[int]] = []
    for j in range(1, limit + 1):
        for k in range(j + 1, limit + 1):
            if values[j] == values[k]:
                pairs.append([j, k])
    return pairs


def jsd_tie_pairs_through(limit: int) -> list[list[int]]:
    forms = {k: jsd_form(k) for k in range(1, limit + 1)}
    pairs: list[list[int]] = []
    for j in range(1, limit + 1):
        for k in range(j + 1, limit + 1):
            if forms[j] == forms[k]:
                pairs.append([j, k])
    return pairs


def analytic_tie_pairs() -> list[list[int]]:
    """Pairs {j, N*N/j} for j = 1, 2, 3 when N*N/j is an integer other than j."""
    pairs: list[list[int]] = []
    for j in range(1, N):
        if (N * N) % j == 0:
            k = (N * N) // j
            if k != j:
                pairs.append([j, k])
    pairs.sort()
    return pairs


def quantize(value: Decimal) -> Decimal:
    return value.quantize(Decimal("1e-16"), rounding=ROUND_HALF_EVEN)


def json_ready(value: LogForm | str) -> Decimal | str:
    if isinstance(value, str):
        return value
    return quantize(value.evaluate())


def prefer(left: LogForm | str, right: LogForm | str) -> str:
    """Which argument is smaller. left is q_2, right is q_{k_hat}.

    Equality is exact (identical prime coefficients, or both infinite).
    A strict inequality is decided from the evaluated closed form; the gaps
    in this problem are either infinite or larger than 1/10.
    """
    if left == "inf" and right == "inf":
        return "tie"
    if left == "inf":
        return "q_k_hat"
    if right == "inf":
        return "q2"
    assert isinstance(left, LogForm) and isinstance(right, LogForm)
    if left == right:
        return "tie"
    if left.evaluate() < right.evaluate():
        return "q2"
    if right.evaluate() < left.evaluate():
        return "q_k_hat"
    raise AssertionError("unequal log forms evaluated equal")


def assert_consistency() -> None:
    for k in range(1, 65):
        if v_ls_fraction(k) != closed_form_v_ls(k):
            raise AssertionError(f"V_LS mismatch at k={k}")

    # Finite exhaustive check. Completeness past 16 is the monotonicity
    # argument in DERIVATION.md: both criteria exceed their k=1 value after
    # k = 16 and never repeat a value on {4, 5, 6, ...}.
    expected = analytic_tie_pairs()
    if ls_tie_pairs() != expected:
        raise AssertionError(f"LS pairs {ls_tie_pairs()} != {expected}")
    if jsd_tie_pairs_through(N * N) != expected:
        raise AssertionError(f"JSD pairs {jsd_tie_pairs_through(N * N)} != {expected}")

    # Sanity check on a long initial segment. The proof that this continues
    # for every k > 4 is the derivative of the closed form in DERIVATION.md.
    previous = jsd_form(N).evaluate()
    for k in range(N + 1, 201):
        current = jsd_form(k).evaluate()
        if current <= previous:
            raise AssertionError(f"JSD failed to increase at k={k}")
        previous = current
    if not (jsd_form(N * N + 1).evaluate() > jsd_form(1).evaluate()):
        raise AssertionError("JSD should already exceed JSD(q_1) past k=16")

    for k in (2, 8):
        direct = gan_terms(k)
        closed = closed_form_losses(k)
        for key in ("V_star", "L_MM", "L_NS"):
            if direct[key] == "inf" or closed[key] == "inf":
                if direct[key] != closed[key]:
                    raise AssertionError(f"{key} infinity mismatch at k={k}")
            elif abs(direct[key] - closed[key].evaluate()) > Decimal("1e-60"):
                raise AssertionError(f"{key} mismatch at k={k}: {direct[key]} vs {closed[key].evaluate()}")

    if jsd_form(1) != jsd_form(16) or jsd_form(2) != jsd_form(8):
        raise AssertionError("predicted JSD ties are not exact")
    if jsd_form(3) == jsd_form(5) or jsd_form(3) == jsd_form(6):
        raise AssertionError("JSD(q_3) should lie strictly between k=5 and k=6")

    # V_star = -ln 4 + 2 JSD, checked on the two tied generators.
    for k in (2, 8):
        v_from_jsd = -_ln_frac(Fraction(N)) + 2 * jsd_form(k).evaluate()
        if abs(v_from_jsd - gan_terms(k)["V_star"]) > Decimal("1e-60"):
            raise AssertionError(f"V_star / JSD identity failed at k={k}")


def build_answer() -> dict:
    assert_consistency()
    pairs = analytic_tie_pairs()
    k_hat = next(k for j, k in pairs if j == 2)
    q2 = closed_form_losses(2)
    qk = closed_form_losses(k_hat)
    answer = {
        "jsd_tie_pairs": pairs,
        "ls_tie_pairs": ls_tie_pairs(),
        "ls_value_q2": quantize(_frac_to_decimal(v_ls_fraction(2))),
        "k_hat": k_hat,
        "q2": {key: json_ready(q2[key]) for key in ("V_star", "L_MM", "L_NS")},
        "q_k_hat": {key: json_ready(qk[key]) for key in ("V_star", "L_MM", "L_NS")},
        "prefers": {
            "V_star": prefer(q2["V_star"], qk["V_star"]),
            "L_MM": prefer(q2["L_MM"], qk["L_MM"]),
            "L_NS": prefer(q2["L_NS"], qk["L_NS"]),
        },
    }
    return answer


def write_answer(answer: dict, path: Path) -> None:
    # Pretty-print with the schema's key order. Decimal values are already quantized.
    text = json.dumps(_stringify_decimals(answer), indent=2)
    text = _restore_decimals(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _stringify_decimals(value):
    if isinstance(value, Decimal):
        return {"__decimal__": format(value, "f")}
    if isinstance(value, dict):
        return {key: _stringify_decimals(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_decimals(item) for item in value]
    return value


def _restore_decimals(text: str) -> str:
    """Turn tagged decimal objects back into JSON numbers."""
    return re.sub(r'\{\s*"__decimal__":\s*"([^"]*)"\s*\}', r"\1", text)


def main() -> None:
    answer = build_answer()
    destination = Path(__file__).resolve().parent.parent / "results" / "answer.json"
    write_answer(answer, destination)
    print(destination.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
