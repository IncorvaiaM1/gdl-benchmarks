"""INSTRUCTOR ONLY reference solution. Not shipped to the solving agent.

Solves the corrected MLE score equation for lambda given observations that
are each the max of m i.i.d. Exponential(lambda) draws:

    n/lambda - sum(y) + (m-1) * sum( y * exp(-lambda*y) / (1 - exp(-lambda*y)) ) = 0

Existence: the score is +inf as lambda->0+ and -sum(y) < 0 as lambda->inf,
so by the intermediate value theorem a root in (0, inf) always exists, and
brentq (bracketed root-finding) finds it.

Uniqueness: this family is not an exponential family in lambda, so
concavity of the log-likelihood is not automatic and has to be checked
directly. Differentiating the score once more gives the observed
information (negative second derivative of the log-likelihood):

    ell''(lambda) = -n/lambda^2
                    - (m-1) * sum( y_i^2 * exp(lambda*y_i) / (exp(lambda*y_i) - 1)^2 )

Both terms are non-positive for every lambda > 0 (the first is manifestly
negative; the second is a sum of squares times a strictly positive factor,
so it is negative whenever m > 1), and the first term alone is already
strictly negative, so ell''(lambda) < 0 for all lambda > 0. Hence the
log-likelihood is strictly concave on (0, inf), so the root found above is
not merely *a* root of the score equation -- it is the unique global
maximizer of the likelihood.
"""
import numpy as np
from scipy.optimize import brentq


def score(lam, y, m):
    n = len(y)
    e = np.exp(-lam * y)
    return n / lam - y.sum() + (m - 1) * np.sum(y * e / (1 - e))


def mle_grouped_max(y, m, lo=1e-6, hi=1e3):
    y = np.asarray(y, dtype=float)
    return brentq(lambda lam: score(lam, y, m), lo, hi, xtol=1e-10)


def observed_information(lam, y, m):
    """-d^2/dlambda^2 of the log-likelihood. Strictly positive for all
    lambda > 0 (see module docstring), confirming the score's root is the
    unique global maximizer, not merely a critical point."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    e = np.exp(lam * y)
    return n / lam**2 + (m - 1) * np.sum((y ** 2) * e / (e - 1) ** 2)


def naive_mle_exponential(y):
    """The WRONG estimator: treats y as if it were raw i.i.d. Exp(lambda)
    data, ignoring that it is actually a max of m draws. Included only so
    tests can demonstrate how badly it fails."""
    y = np.asarray(y, dtype=float)
    return len(y) / y.sum()


if __name__ == "__main__":
    import pandas as pd

    m = 4
    y_train = pd.read_csv("data/train.csv")["y"].to_numpy()
    y_holdout = pd.read_csv("data/holdout.csv")["y"].to_numpy()

    lam_train = mle_grouped_max(y_train, m)
    lam_holdout = mle_grouped_max(y_holdout, m)
    naive_train = naive_mle_exponential(y_train)
    info_train = observed_information(lam_train, y_train, m)

    print(f"corrected MLE on train.csv:   {lam_train:.4f}")
    print(f"corrected MLE on holdout.csv: {lam_holdout:.4f}")
    print(f"observed information at MLE (train): {info_train:.4f}  (>0 confirms unique max)")
    print(f"naive (wrong) MLE on train.csv: {naive_train:.4f}")
