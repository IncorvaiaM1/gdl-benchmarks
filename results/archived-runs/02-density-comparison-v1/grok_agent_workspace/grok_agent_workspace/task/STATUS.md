# Status

## 2026-09-23 16:18 — initial state

**Command:**
```powershell
python -V
Get-FileHash -Algorithm SHA256 grok_agent_workspace\data\images.csv
```

**Exit status:** 0

**Key output:**
```
Python 3.12.5
C:\Python312\python.exe
Windows-11-10.0.26200-SP0
AMD64
SHA256 98B5D6A6D90BC2BC1C22DF168E45E63DA97FA2CCE5D75001650F7F391A8B69C4
```

**Result:** Workspace is the nested project directory containing `data/` and `task/`. No third-party packages are required. `data/images.csv` matches the SHA-256 in `data/DOWNLOAD.md`. Git metadata for this checkout lives outside the allowed workspace, so no commit hash was recorded.

**Invalidates:** nothing yet.

**Next:** derive both bits-per-dimension values and write `task/solution/compute_bpd.py`.

## 2026-09-23 16:20 — compute bits per dimension

**Command:**
```powershell
python task/solution/compute_bpd.py
```

Working directory: project root (contains `data/` and `task/`).

**Exit status:** 0

**Key output:**
```
quadrature_gap 4.707e-14
jacobian_expectation -2248.275895535439
bpd_A 4.371180330825
bpd_B 4.290814025963
better B
{"bpd_A": 4.371180330825209, "bpd_B": 4.290814025963129, "better": "B"}
```

**Result:** Wrote `task/results/answer.json`. Model A is data-independent: `bpd_A = 8 - 1972 / (784 * ln 2) = 4.371180330825209`. Model B's logit Jacobian expectation over exact uniform dequantization noise is `-2248.275895535439` nats per image, giving `bpd_B = 4.290814025963129`. Closed form versus composite Simpson (20000 intervals, all 256 pixel values) differs by at most `4.707e-14`. A separate one-draw Monte Carlo over every pixel (`random.Random(0)`) gave Jacobian `-2248.280566733253`, a difference of `0.0046711978143321176` nats per image (about `9e-6` bits/dim).

**Invalidates:** nothing downstream of `task/results/answer.json`. Re-run this command if `data/images.csv` or the script changes.

**Next:** confirm the script in a clean virtualenv.

## 2026-09-23 16:21 — verify clean virtualenv

**Command:**
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -V
.\.venv\Scripts\python.exe -m pip freeze
.\.venv\Scripts\python.exe -m pip --version
.\.venv\Scripts\python.exe task/solution/compute_bpd.py
Remove-Item -Recurse -Force .venv
```

**Exit status:** 0

**Key output:**
```
Python 3.12.5
pip 24.2
quadrature_gap 4.707e-14
jacobian_expectation -2248.275895535439
bpd_A 4.371180330825
bpd_B 4.290814025963
better B
{"bpd_A": 4.371180330825209, "bpd_B": 4.290814025963129, "better": "B"}
venv removed
```

`python -m pip freeze` printed no packages.

**Result:** `INSTALL.md` matches this environment. The clean virtualenv reproduced `task/results/answer.json` with the same floats. The virtualenv was removed after that check; recreate it with the commands in `INSTALL.md`.

**Invalidates:** nothing. The answer file was rewritten with identical contents.

**Next:** none. Both bits-per-dimension values and `better = B` are recorded in `task/results/answer.json`.
