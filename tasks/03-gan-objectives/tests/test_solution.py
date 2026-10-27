"""Deterministic verifier for task/TASK.md.

Every gold value is recomputed by gt_code/reference_solution.py, which itself
cross-checks a direct scan over k against the closed form. Pair lists, k_hat,
"inf" strings, and preferences must match exactly; floats within 1e-8.
"""
import json
import math
from pathlib import Path

import pytest

from gt_code.reference_solution import solve

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "task" / "results" / "answer.json"
KEYS = {"jsd_tie_pairs", "ls_tie_pairs", "ls_value_q2", "k_hat", "q2", "q_k_hat", "prefers"}
TERMS = {"V_star", "L_MM", "L_NS"}
ABS_TOL = 1e-8


@pytest.fixture(scope="module")
def ref():
    return solve()


@pytest.fixture(scope="module")
def ans():
    assert RESULT.exists(), "task/results/answer.json is missing"
    return json.loads(RESULT.read_text())


def _match(got, want):
    if isinstance(want, str):
        return got == want
    if isinstance(got, bool) or not isinstance(got, (int, float)) or not math.isfinite(got):
        return False
    return abs(got - want) <= ABS_TOL


def test_result_shape(ans):
    """answer.json has exactly the seven required keys; q2 has the three GAN terms."""
    assert set(ans) == KEYS
    assert isinstance(ans["q2"], dict) and set(ans["q2"]) == TERMS


def test_jsd_tie_pairs(ans, ref):
    """All JSD ties are found and the list is complete: {1,16} and {2,8} (j*k = 16)."""
    assert ans["jsd_tie_pairs"] == ref["jsd_tie_pairs"]


def test_ls_tie_pairs(ans, ref):
    """LSGAN (a=-1, b=1, c=0) criterion is (r-1)/(r+1), so it ties on the same pairs."""
    assert ans["ls_tie_pairs"] == ref["ls_tie_pairs"]


def test_ls_value_q2(ans, ref):
    """V_LS(q_2) = 1/3 (half the symmetric chi-square sum (p-q)^2/(p+q))."""
    assert _match(ans["ls_value_q2"], ref["ls_value_q2"])


def test_k_hat(ans, ref):
    """The JSD partner of k = 2 is k_hat = 8."""
    assert ans["k_hat"] == ref["k_hat"]


def test_q2_terms(ans, ref):
    """At q_2: V_star = log2 - 1.5 log3, L_MM = log(2/3), L_NS = log 3."""
    for t in TERMS:
        assert _match(ans["q2"][t], ref["q2"][t]), t


def test_q_k_hat_terms(ans, ref):
    """At q_8: V_star = log2 - 1.5 log3, L_MM = -0.5 log 3, L_NS = +inf (D* = 0 off the data support)."""
    assert isinstance(ans["q_k_hat"], dict) and set(ans["q_k_hat"]) == TERMS
    for t in TERMS:
        assert _match(ans["q_k_hat"][t], ref["q_k_hat"][t]), t


def test_prefers(ans, ref):
    """V_star ties, the minimax generator term prefers over-spreading (q_8), non-saturating prefers mode dropping (q_2)."""
    assert ans["prefers"] == ref["prefers"]
