"""Deterministic verifier for task/TASK.md.

Two independent correctness checks, plus two design self checks that verify
the checks themselves discriminate the way they're supposed to:

1. test_estimate_matches_gold_mle_on_train (primary gate): recomputes the
   gold MLE independently on data/train.csv -- the *same* data the agent
   was given -- at a tight relative tolerance. This is deliberately tight
   enough that a solver who used the correct block-maxima likelihood passes
   easily (independent implementations agree to ~1e-8 relative in practice),
   but a solver who instead used the method-of-moments estimator
   lambda_hat = H_m / mean(y) -- which is numerically close to the MLE, but
   not it -- fails. A holdout-only check at a loose tolerance cannot make
   this distinction, because the gold MLE computed on train and on holdout
   legitimately differ by more than the gap between MoM and the true MLE on
   either sample; comparing against train at a tight tolerance is what
   actually verifies the agent solved the estimating equation.

2. test_estimate_generalizes_to_holdout (secondary check): recomputes the
   gold MLE on data/holdout.csv (which the agent was told not to read) and
   checks the reported lambda_hat is in the right ballpark. This catches
   gross errors (wrong data used, wrong m, a bug that only shows up out of
   sample) but is not the primary correctness gate.

3. test_naive_estimator_fails_the_train_check / (4)
   test_method_of_moments_fails_the_train_check: sanity checks on the test
   design itself (not on the agent), confirming that both the "treat y as
   raw Exponential(lambda) data" estimator and the method-of-moments
   estimator both fail check (1), so a submission using either of them is
   actually caught rather than accidentally passing.

Everything this file asserts on is declared in the constants block below.
"""
import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

M = 4                          # group size (given to the agent in TASK.md)
N_TRAIN_ROWS = 2000            # rows in data/train.csv
N_HOLDOUT_ROWS = 2000          # rows in data/holdout.csv

REL_TOL_TRAIN = 3e-4           # primary gate: relative error vs gold MLE on
                                # train.csv (the agent's own data). TASK.md
                                # tells the agent this value and requires
                                # >= 8 significant figures in lambda_hat, so
                                # the boundary is disclosed, not implicit.
                                # Calibration: an estimate correctly rounded
                                # to 4 significant figures has ~1.1e-4
                                # relative error (passes with ~2.8x margin);
                                # method-of-moments differs from the train
                                # MLE by ~1.2e-3 (fails with ~4x margin). See
                                # test_method_of_moments_fails_the_train_check.
REL_TOL_HOLDOUT = 0.10         # secondary, looser generalization check
NAIVE_MUST_BE_WRONG_BY = 0.25  # naive Exp(lambda) estimator's relative error
                                # vs the train MLE must exceed this (it is
                                # actually ~0.52 by construction -- see gt_code/)
MOM_MUST_BE_WRONG_BY = 6e-4    # method-of-moments estimator's relative error
                                # vs the train MLE must exceed this (it is
                                # actually ~1.2e-3 by construction -- see
                                # gt_code/); set to 2x REL_TOL_TRAIN so this
                                # self-check and the primary gate don't fight

REPO_ROOT = Path(__file__).resolve().parent.parent
ESTIMATE_PATH = REPO_ROOT / "task" / "results" / "estimate.json"
TRAIN_CSV = REPO_ROOT / "data" / "train.csv"
HOLDOUT_CSV = REPO_ROOT / "data" / "holdout.csv"

Submission = namedtuple("Submission", ["lambda_hat", "error"])


def _score(lam, y, m):
    n = len(y)
    e = np.exp(-lam * y)
    return n / lam - y.sum() + (m - 1) * np.sum(y * e / (1 - e))


def _gold_mle(y, m, lo=1e-6, hi=1e3):
    return brentq(lambda lam: _score(lam, y, m), lo, hi, xtol=1e-12)


def _naive_mle(y):
    """Treats y as raw i.i.d. Exponential(lambda) data (wrong model)."""
    return len(y) / y.sum()


def _method_of_moments(y, m):
    """E[max of m i.i.d. Exp(lambda)] = H_m / lambda, inverted. Uses the
    correct *moment* but not the correct *likelihood* -- numerically close
    to the MLE, but not equal to it, which is exactly why it needs its own
    check rather than relying on the naive-estimator check to catch it."""
    harmonic_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    return harmonic_m / float(np.mean(y))


@pytest.fixture(scope="module")
def submission():
    """Load the agent's reported estimate. A missing/malformed artefact is a
    failed submission, not a broken test, so it is returned as `error`."""
    if not ESTIMATE_PATH.exists():
        return Submission(None, "Cannot verify: task/results/estimate.json is missing at {}".format(ESTIMATE_PATH))
    try:
        obj = json.loads(ESTIMATE_PATH.read_text())
        lam = float(obj["lambda_hat"])
    except Exception as e:
        return Submission(None, "task/results/estimate.json is not valid ({}: {})".format(type(e).__name__, e))
    if not (lam > 0 and np.isfinite(lam)):
        return Submission(None, "lambda_hat = {} is not a finite positive number".format(lam))
    return Submission(lambda_hat=lam, error=None)


@pytest.fixture(scope="module")
def train_y():
    assert TRAIN_CSV.exists(), "data/train.csv is missing from the repository -- this is a setup error, not an agent error"
    table = np.loadtxt(TRAIN_CSV, delimiter=",", skiprows=1)
    assert table.shape[0] == N_TRAIN_ROWS, (
        "data/train.csv has {} rows; expected {} -- wrong file?".format(table.shape[0], N_TRAIN_ROWS))
    return table


@pytest.fixture(scope="module")
def holdout_y():
    assert HOLDOUT_CSV.exists(), "data/holdout.csv is missing from the repository -- this is a setup error, not an agent error"
    table = np.loadtxt(HOLDOUT_CSV, delimiter=",", skiprows=1)
    assert table.shape[0] == N_HOLDOUT_ROWS, (
        "data/holdout.csv has {} rows; expected {} -- wrong file?".format(table.shape[0], N_HOLDOUT_ROWS))
    return table


def test_estimate_is_well_formed(submission):
    """Checks that task/results/estimate.json exists and reports a finite,
    strictly positive lambda_hat, per the TASK.md contract."""
    assert submission.error is None, submission.error


def test_estimate_matches_gold_mle_on_train(submission, train_y):
    """Primary correctness gate. Recomputes the gold MLE independently on
    data/train.csv -- the same data the agent was given -- by solving the
    block-maxima score equation, and requires the agent's reported
    lambda_hat to be within 0.03% relative error of it. This is tight enough
    that only a submission that actually solved the correct likelihood's
    score equation (not an approximation of it, such as method-of-moments)
    passes; see test_method_of_moments_fails_the_train_check for the
    calibration check confirming this."""
    assert submission.error is None, submission.error
    gold = _gold_mle(train_y, M)
    rel_err = abs(submission.lambda_hat - gold) / gold
    assert rel_err <= REL_TOL_TRAIN, (
        "lambda_hat={:.6f} has relative error {:.6f} against the independently "
        "recomputed gold MLE {:.6f} on data/train.csv (the agent's own data); "
        "required <= {:.1e}. This tolerance is tight enough that even a "
        "numerically close but non-MLE estimator (e.g. method-of-moments, "
        "H_m/mean(y)) fails it -- if you are close but not passing, check "
        "that you solved the score equation itself rather than an "
        "approximation of it.".format(
            submission.lambda_hat, rel_err, gold, REL_TOL_TRAIN))


def test_estimate_generalizes_to_holdout(submission, holdout_y):
    """Secondary, looser sanity check: recomputes the gold MLE on the
    withheld data/holdout.csv (which the agent never saw) and checks the
    reported lambda_hat is within 10% of it. This is a generalization check
    (catches e.g. a bug that only manifests out of sample, or an estimate
    fit to the wrong file) and is intentionally looser than the primary
    train-based gate above, since the train and holdout gold MLEs
    legitimately differ by ~1% due to sampling noise."""
    assert submission.error is None, submission.error
    gold = _gold_mle(holdout_y, M)
    rel_err = abs(submission.lambda_hat - gold) / gold
    assert rel_err <= REL_TOL_HOLDOUT, (
        "lambda_hat={:.4f} has relative error {:.3f} against the independently "
        "recomputed gold MLE {:.4f} on data/holdout.csv; required <= {:.2f}.".format(
            submission.lambda_hat, rel_err, gold, REL_TOL_HOLDOUT))


def test_naive_estimator_fails_the_train_check(train_y):
    """Sanity check on the test design itself (not on the agent): confirms
    that the naive 'n / sum(y)' estimator -- the answer a solver gets by
    reflexively applying the standard Exponential(lambda) MLE formula
    without noticing y is a block maximum -- misses the train-data gold MLE
    by more than 25%."""
    gold = _gold_mle(train_y, M)
    naive = _naive_mle(train_y)
    rel_err = abs(naive - gold) / gold
    assert rel_err > NAIVE_MUST_BE_WRONG_BY, (
        "expected the naive estimator to clearly fail (rel. error > {:.2f}) "
        "on data/train.csv, but it only differed from the gold MLE by {:.3f}; "
        "the dataset parameters (m, n, true lambda) should be revisited.".format(
            NAIVE_MUST_BE_WRONG_BY, rel_err))


def test_method_of_moments_fails_the_train_check(train_y):
    """Sanity check on the test design itself (not on the agent): confirms
    that the method-of-moments estimator H_m/mean(y) -- which correctly
    identifies that y is a block maximum and uses the right moment
    E[Y]=H_m/lambda, but never derives or solves the actual likelihood --
    still fails the tight train-based gate above. Without this check, a
    solver could pass by recognizing the transformation but stopping short
    of the maximum-likelihood derivation the task actually asks for."""
    gold = _gold_mle(train_y, M)
    mom = _method_of_moments(train_y, M)
    rel_err = abs(mom - gold) / gold
    assert rel_err > MOM_MUST_BE_WRONG_BY, (
        "expected method-of-moments to fail the tight train check (rel. "
        "error > {:.1e}), but it only differed from the gold MLE by {:.2e}; "
        "REL_TOL_TRAIN may need to be tightened further, or the dataset "
        "parameters revisited.".format(MOM_MUST_BE_WRONG_BY, rel_err))
