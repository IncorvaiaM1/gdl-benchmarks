import json
import math
from pathlib import Path

import pytest

from gt_code.reference_solution import PARAMS, load_images, solve

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "task" / "results" / "answer.json"
KEYS = {"bpd_A", "bpd_B", "better", "lambda_min", "lambda_star", "bpd_C_own", "bpd_C_8bit"}


@pytest.fixture(scope="module")
def ref():
    return solve(load_images(), **PARAMS)


@pytest.fixture(scope="module")
def ans():
    return json.loads(RESULT.read_text())


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def test_result_shape(ans):
    """answer.json has exactly the seven required keys with the right types."""
    assert set(ans) == KEYS
    assert ans["better"] in ("A", "B")
    for k in ("bpd_A", "lambda_min", "lambda_star", "bpd_C_own"):
        assert _num(ans[k])
    for k in ("bpd_B", "bpd_C_8bit"):
        assert ans[k] is None or _num(ans[k])


def test_bpd_A(ans, ref):
    """bpd_A accounts for the 1/256 bin width (D log 256)."""
    assert ans["bpd_A"] == pytest.approx(ref["bpd_A"], abs=1e-3)


def test_bpd_B(ans, ref):
    """B's /255 pipeline still gives a valid bound (bin width 1/256); logit Jacobian averaged over data and u at k/255."""
    assert _num(ans["bpd_B"])
    assert ans["bpd_B"] == pytest.approx(ref["bpd_B"], abs=1e-3)


def test_better(ans, ref):
    """Correct model (A or B) has the lower bound."""
    assert ans["better"] == ref["better"]


def test_lambda_min(ans, ref):
    """Smallest lambda keeping s < 1 under the /255 pipeline is 1/258."""
    assert ans["lambda_min"] == pytest.approx(ref["lambda_min"], rel=1e-6)


def test_lambda_star(ans, ref):
    """Break-even lambda at which B's bound equals A's."""
    assert ans["lambda_star"] == pytest.approx(ref["lambda_star"], abs=5e-5)


def test_bpd_C_own(ans, ref):
    """C's score in bits per dimension of the 32x32 binary variable it models (D = 1024)."""
    assert ans["bpd_C_own"] == pytest.approx(ref["bpd_C_own"], abs=1e-4)


def test_bpd_C_8bit(ans):
    """C's score gives no bound on the 8-bit NLL, so bpd_C_8bit must be null."""
    assert ans["bpd_C_8bit"] is None
