# Installation and reproduction

## Platform

- Windows 11, build 10.0.26200, AMD64
- Python 3.12.5 (`python -V` prints `Python 3.12.5`)
- CPU-only computation; CUDA and GPU drivers are not used.

## Dependencies

No third-party Python packages are required. The entry point uses only the
Python 3.12 standard library (`csv`, `json`, `math`, and `pathlib`), so there
is no package install command and no `pip freeze` lock list.

The input dataset is included at `data/images.csv`; no download is required.
Its expected SHA-256 is
`98b5d6a6d90bc2bc1c22df168e45e63da97fa2cce5d75001650f7f391a8b69c4`.
No environment variables or external caches are used.

## Reproduce

From the project root (the directory containing `task/` and `data/`), run:

```powershell
python -I task/solution/solve.py
```

Isolated mode (`-I`) excludes user site-packages and Python environment
variables. The command prints the intermediate expectations and writes
`task/results/answer.json`.

No known-bad dependency combinations were encountered.
