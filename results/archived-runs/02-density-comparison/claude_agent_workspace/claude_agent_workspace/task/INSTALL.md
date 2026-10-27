# INSTALL — rebuilding the environment

This project is a closed-form / numerical analysis task. It trains nothing, needs
no GPU, no network access, and no dataset download: the only input is the
`data/images.csv` file already present in the workspace.

## Platform actually used

- **OS:** Windows 11, `Windows-11-10.0.26200-SP0`, `AMD64` (x86-64).
- **Shell:** PowerShell.
- **Python:** `Python 3.12.5` (`3.12.5 (tags/v3.12.5:ff3bc82, Aug  6 2024, 20:45:27)
  [MSC v.1940 64 bit (AMD64)]`), system interpreter at `C:\Python312\python.exe`.
- **GPU / CUDA:** none used. There is no PyTorch dependency — the PyTorch snippets
  in `task/TASK.md` are read as a *specification* of each team's preprocessing, not
  as code to run.

## Packages

Only the interpreter's pre-existing scientific stack was used; **nothing was
installed for this task**, so there is no install command to replay other than
creating an environment with these pins:

```bash
python -m pip install numpy==2.2.1 scipy==1.15.0 mpmath==1.3.0
```

Exact versions present when the results were produced (`python -m pip freeze`,
filtered to what the solution imports):

```
mpmath==1.3.0
numpy==2.2.1
scipy==1.15.0
```

Which pins are load-bearing:

- `numpy==2.2.1` — **incidental**. Used for CSV loading, a `bincount` histogram and
  vectorised float64 arithmetic. Any numpy >= 1.24 gives identical float64 results.
- `mpmath==1.3.0` — **optional, verification only**. Used by `--verify` to redo the
  noise-averaged log-determinant at 40 decimal digits as an independent check of the
  closed-form float64 path. If it is missing, `--verify` prints
  `mpmath not installed; skipped high-precision check` and everything else still runs.
- `scipy==1.15.0` — **not imported.** Listed only because it was present in the
  environment; the root-finding for `lambda_star` is a hand-written bisection
  precisely so that scipy is not a dependency.

The solution therefore runs on a bare `python + numpy` environment.

## Determinism

- The reported answer is fully deterministic: every expectation over the
  dequantization noise is evaluated **in closed form**, not sampled.
- The only randomness anywhere is in the optional `--verify` Monte-Carlo
  cross-check, which uses `numpy.random.default_rng(seed)` with `seed=0`. Its result
  is informational (agreement to ~1e-3 nats out of ~2122) and does not feed the
  answer.
- No thread-count or BLAS sensitivity: all arithmetic is elementwise plus one
  256-element dot product.

## Data

- `data/images.csv` — 1000 MNIST test images, 784 uint8 columns, ships with the
  workspace. Nothing is downloaded.
- Integrity is checked at every run; the entry point prints the SHA-256 and whether
  it matches the value recorded in `data/DOWNLOAD.md`:
  `98b5d6a6d90bc2bc1c22df168e45e63da97fa2cce5d75001650f7f391a8b69c4`.

## How to run

From the workspace root (the directory containing `data/` and `task/`):

```bash
python task/solution/solve.py            # writes task/results/answer.json
python task/solution/solve.py --verify   # same, plus Monte-Carlo + mpmath checks
```

Runtime: ~3 s without `--verify`, ~8 s with it (dominated by re-reading the CSV).

## Known-bad combinations

None encountered. The one Windows-specific caveat: run the command from the
workspace root as shown. The script resolves `data/images.csv` and
`task/results/answer.json` relative to its own location (`Path(__file__).parents[2]`),
so it also works from any other cwd; `--images` / `--out` override both paths.

## Verification record

Last followed end to end on 2026-09-23 against the environment above (see the
`INSTALL.md verification` entry in `STATUS.md`): a fresh `python -m venv .venv-verify`
with only `numpy==2.2.1` installed ran `python task/solution/solve.py --verify` and
reproduced `task/results/answer.json` byte-for-byte (SHA-256 equal to the copy made
beforehand). In that environment the mpmath cross-check correctly self-skipped and the
Monte-Carlo cross-check reproduced the same numbers, confirming the `numpy`-only claim
and the seeded determinism. The throwaway venv was deleted afterwards.
