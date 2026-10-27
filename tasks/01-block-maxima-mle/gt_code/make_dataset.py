"""INSTRUCTOR-ONLY. Generates data/train.csv and data/holdout.csv.

This script is NOT shipped to the solving agent  it hardcodes the true
rate parameter LAMBDA_TRUE, which is exactly what the agent must estimate.
Re-running it reproduces data/train.csv and data/holdout.csv byte-for-byte
(fixed seeds), matching the hashes recorded in gt_code/true_params.json and
described (without the secret parameter value) in data/DOWNLOAD.md.

Generative model: each observation y is the MAXIMUM of M i.i.d. draws from
Exponential(LAMBDA_TRUE). This is the "twist" on the vanilla exponential
MLE exercise  see task/TASK.md.
"""
import hashlib
import json

import numpy as np

LAMBDA_TRUE = 2.0
M = 4
N_TRAIN, N_HOLDOUT = 2000, 2000
SEED_TRAIN, SEED_HOLDOUT = 20260904, 20260905


def gen(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.exponential(scale=1.0 / LAMBDA_TRUE, size=(n, M))
    return x.max(axis=1)


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


if __name__ == "__main__":
    y_train = gen(N_TRAIN, SEED_TRAIN)
    y_holdout = gen(N_HOLDOUT, SEED_HOLDOUT)

    np.savetxt("data/train.csv", y_train, delimiter=",", header="y", comments="", fmt="%.6f")
    np.savetxt("data/holdout.csv", y_holdout, delimiter=",", header="y", comments="", fmt="%.6f")

    meta = {
        "lambda_true": LAMBDA_TRUE,
        "m": M,
        "n_train": N_TRAIN,
        "n_holdout": N_HOLDOUT,
        "seed_train": SEED_TRAIN,
        "seed_holdout": SEED_HOLDOUT,
        "train_sha256": sha256("data/train.csv"),
        "holdout_sha256": sha256("data/holdout.csv"),
    }
    json.dump(meta, open("gt_code/true_params.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))
