# Installation

## Platform

- Windows 11 (build 10.0.26200), AMD64
- PowerShell 5.1.26100.9444
- Python 3.12.5
- No GPU or CUDA is required.

## Dependencies

The solution uses only the Python 3.12 standard library. No packages are
installed, so there is no `pip freeze` output or package lockfile.

## Rebuild and run

From the project root:

```powershell
python .\task\solution\solve.py
```

This reads `data/images.csv` and writes `task/results/answer.json`.

## Verification

No separate environment creation or installation is needed. The command above
was verified in the stated base environment; the verification run is recorded
in `task/STATUS.md`.
