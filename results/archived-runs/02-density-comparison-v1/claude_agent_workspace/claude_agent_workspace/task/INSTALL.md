# INSTALL

How to rebuild the environment for this project. The task is a closed-form
computation plus one pass over a 1000×784 CSV; there is no training, no GPU, and no
network access required.

## Platform

Recorded on the machine where the results in `STATUS.md` were produced:

- **OS:** `Windows-11-10.0.26200-SP0`, `AMD64` (x86-64)
- **Python:** `python -V` → `Python 3.12.5`
  (full banner: `3.12.5 (tags/v3.12.5:ff3bc82, Aug  6 2024, 20:45:27) [MSC v.1940 64 bit (AMD64)]`)
- **pip:** `pip 26.1.2`
- **GPU / CUDA:** not used.
- **Shell used for all recorded commands:** PowerShell.

Nothing here is platform-specific: any OS with CPython ≥ 3.9 and NumPy ≥ 1.20 will
reproduce the same numbers, because the computation is pure IEEE-754 double
arithmetic over `log`/`log1p`/`bincount`.

## Environment creation

The recorded run used the ambient system Python 3.12.5 with NumPy already present —
no virtualenv was created. To reproduce from clean, use a fresh venv:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy==2.2.1
```

## Packages (exact pins)

Only NumPy is imported by the solution. Pins below are from `python -m pip freeze`
on the recording machine:

```
numpy==2.2.1
setuptools==80.9.0
```

- `numpy==2.2.1` — **not load-bearing.** Used for `loadtxt`, `bincount`,
  `log`/`log1p`, `polynomial.legendre.leggauss` (cross-check only) and
  `random.default_rng` (cross-check only). Any NumPy ≥ 1.20 gives identical
  `bpd_A`/`bpd_B` to full double precision; only the Monte-Carlo cross-check
  printout depends on the version, via the PCG64 stream, and its tolerances are
  loose (`5e-4` bpd).
- `setuptools==80.9.0` — incidental, ambient in the interpreter; not imported.

## Non-pip steps

- **Data:** none. `data/images.csv` ships with the repo; no download is performed.
  Verify integrity before trusting any result:
  ```bash
  sha256sum data/images.csv
  # 98b5d6a6d90bc2bc1c22df168e45e63da97fa2cce5d75001650f7f391a8b69c4
  ```
  PowerShell equivalent:
  ```powershell
  (Get-FileHash -Algorithm SHA256 data\images.csv).Hash
  ```
- **Environment variables:** none required.
- **Caches:** none. No model or dataset caches are written.

## Run

From the repository root (the directory containing `data/` and `task/`):

```bash
python task/solution/solve.py
```

This prints the derivation terms and writes `task/results/answer.json`. Runtime is
~15 s, almost entirely the optional Monte-Carlo cross-checks; `--no-checks` reduces
it to ~0.35 s and does not change the answer. Flags: `--data PATH`, `--out PATH`,
`--no-checks`.

## Known-bad combinations

None encountered. Two things to be aware of rather than bugs hit:

- `solve.py` resolves `data/` and `task/results/` from its own file location
  (`Path(__file__).resolve().parents[2]`), not from `cwd`, so the working directory
  should not matter. Only the documented "from repository root" invocation was
  actually exercised; use `--data` / `--out` if you need to override.
- The workspace root contains a nested directory of the same name; the project root
  used throughout is the inner one, i.e. the directory that directly contains
  `data/images.csv` and `task/`.
