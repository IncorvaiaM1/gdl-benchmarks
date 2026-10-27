# INSTALL

## Platform
- OS: Windows 11 (`Windows-11-10.0.26200-SP0`), AMD64
- Python: `Python 3.12.5` (`C:\Python312\python.exe`)
- No GPU, CUDA, datasets, environment variables, or system packages needed.

## Packages
None. `solve.py` uses only the standard library (`json`, `math`, `fractions`,
`pathlib`). `pip freeze` in the clean venv below printed nothing (empty).

## Install / run (verbatim, PowerShell, from the folder containing `task/`)
```powershell
C:\Python312\python.exe -m venv $env:TEMP\gan_tie_venv
& $env:TEMP\gan_tie_venv\Scripts\python.exe task/solution/solve.py
```
Any Python >= 3.8 should work (no version-sensitive features). Only 3.12.5 was tested.

## Known-bad combinations
None encountered.
