# Installation

## Platform

- OS: Windows 11, build 10.0.26200, AMD64
- Python: `Python 3.12.5` (`3.12.5 (tags/v3.12.5:ff3bc82, Aug  6 2024, 20:45:27) [MSC v.1940 64 bit (AMD64)]`)
- No GPU work, so no CUDA or driver pin

## Packages

No third-party packages. In a fresh virtualenv, `python -m pip freeze` prints no lines and exits 0.

`solution/solve.py` uses only the standard library (`json`, `re`, `collections`, `decimal`, `fractions`, `pathlib`). `Decimal.ln` has been in the standard library since Python 3.11; that is why the interpreter pin is 3.12.5 rather than an arbitrary 3.x. Verified on 3.12.5.

## Commands

From the directory that contains `task/` (this project root):

```bash
python -m venv .venv-clean
.\.venv-clean\Scripts\python.exe -m pip freeze
.\.venv-clean\Scripts\python.exe task/solution/solve.py
```

The third command writes `task/results/answer.json`.

Nothing else is required: no system packages, dataset downloads, environment variables, or cache paths.

## Known-bad combinations

None.

## Verification

The three commands above were run on 2026-10-09 with Python 3.12.5. `pip freeze` produced no output. `solve.py` exited 0. The virtualenv was then removed; recreate it with the commands above rather than expecting `.venv-clean` to be present.
