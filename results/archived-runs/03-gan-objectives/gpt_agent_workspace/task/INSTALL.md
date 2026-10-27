 
# Installation

## Platform

- Windows 11, AMD64
- Python 3.12.5
- No GPU or CUDA is used.

The solution uses only the Python standard library. There are no third-party
packages to install or pin.

## Clean environment

From the workspace root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -V
.\.venv\Scripts\python.exe task\solution\solve.py
```

The second command must report `Python 3.12.5`. The final command prints the
answer and rewrites `task/results/answer.json`.

No system packages, downloads, environment variables, or external caches are
required. No known-bad dependency combinations were encountered.
