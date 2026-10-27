# Installation

## Platform

- OS: Windows 11, version 10.0.26200, AMD64
- Python: `Python 3.12.5` (`python -V`), interpreter `C:\Python312\python.exe`
- No GPU work. CUDA and driver versions are not used.

## Packages

The entry point uses only the Python standard library (`csv`, `json`, `math`, `pathlib`). It does not import third-party packages, so none are installed from an index.

`python -m venv .venv` on this interpreter bootstraps `pip 24.2`. That pin is incidental: the solution never calls pip. `python -m pip freeze` inside the fresh virtualenv prints no lines.

## Install commands

From the project root (the directory that contains `data/` and `task/`):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip freeze
```

No index URL and no `pip install`. An empty freeze is the expected lock.

## Run

```powershell
.\.venv\Scripts\python.exe task/solution/compute_bpd.py
```

The same script also runs with the base interpreter, because it has no dependencies:

```powershell
python task/solution/compute_bpd.py
```

Either command reads `data/images.csv` and writes `task/results/answer.json`.

## Data

`data/images.csv` is already in the workspace. No download is required. Expected SHA-256:

```
98b5d6a6d90bc2bc1c22df168e45e63da97fa2cce5d75001650f7f391a8b69c4
```

## Known-bad combinations

None.
