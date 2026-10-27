"""Build data/images.csv: 1000 MNIST test images as raw 8-bit pixel values.

Source: the standard mnist.pkl.gz (deeplearning.net format), whose float
pixels are exactly k/256 for integer k in {0,...,255}. We recover k, sample
1000 test images with a fixed seed, and write them one image per row.
"""
import gzip
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MNIST = ROOT.parent / "mnist.pkl.gz"
SEED = 6257
N = 1000


def main():
    with gzip.open(MNIST, "rb") as f:
        _, _, (X_test, _) = pickle.load(f, encoding="latin1")
    K = np.rint(X_test * 256).astype(np.int64)
    assert np.allclose(K / 256, X_test) and K.min() >= 0 and K.max() <= 255
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(K), size=N, replace=False))
    X = K[idx].astype(np.uint8)
    header = ",".join(f"p{j}" for j in range(X.shape[1]))
    np.savetxt(ROOT / "data" / "images.csv", X, fmt="%d", delimiter=",", header=header, comments="")
    print("wrote", X.shape, "mean pixel", X.mean())


if __name__ == "__main__":
    main()
