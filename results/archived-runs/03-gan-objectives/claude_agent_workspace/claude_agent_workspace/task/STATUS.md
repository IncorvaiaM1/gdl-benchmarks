# STATUS (append-only log)

## 2026-10-09 10:16 — initial state

**Command:**
```powershell
python -V; python -c "import platform;print(platform.platform(), platform.machine())"; git rev-parse HEAD
```

**Exit status:** 0

**Key output:**
```
Python 3.12.5
Windows-11-10.0.26200-SP0 AMD64
1ff2b5353b57f9f440c4219ddd980f0cb047decb
```

**Result:** Workspace contained only `task/TASK.md`, `task/PROJECT_INSTRUCTION.md`,
the placeholder `STATUS.md`/`INSTALL.md`, and empty `solution/` and `results/`
folders (each holding a `.gitkeep`). The repo root is `C:/Users/micha/MAIStuff/randLab`,
at the commit above.

**Next:** derive closed forms (see `solution/DERIVATION.md`) and write `solution/solve.py`.

## 2026-10-09 10:20 — first run of solve.py (system Python)

**Command:**
```powershell
python task/solution/solve.py
```

**Exit status:** 0

**Key output:**
```
jsd_tie_pairs [[1,16],[2,8]]   ls_tie_pairs [[1,16],[2,8]]   ls_value_q2 0.3333333333333333   k_hat 8
q2:  V_star -0.9547712524422194  L_MM -0.40546510810816444  L_NS 1.0986122886681098
q8:  V_star -0.9547712524422194  L_MM -0.5493061443340549   L_NS "inf"
prefers: V_star tie, L_MM q_k_hat, L_NS q2
JSD(q2)=JSD(q8)=0.21576155433883565  JSD(q1)=JSD(q16)=0.3803956658485779
```

**Result:** Wrote `task/results/answer.json`. All closed-form checks against
direct evaluation passed for k <= 400. The run took 112.6 s because the O(K^2)
float cross-check recomputed JSD for every pair.

**Next:** cache JSD values in the cross-check (performance only, output unchanged).

## 2026-10-09 10:24 — clean-venv verification after caching JSD in the cross-check

**Command:**
```powershell
C:\Python312\python.exe -m venv $env:TEMP\gan_tie_venv; & $env:TEMP\gan_tie_venv\Scripts\python.exe -V; & $env:TEMP\gan_tie_venv\Scripts\python.exe -m pip freeze; & $env:TEMP\gan_tie_venv\Scripts\python.exe task/solution/solve.py | Select-Object -Last 1
```

**Exit status:** 0

**Key output:**
```
Python 3.12.5
(pip freeze: empty)
JSD(q2) = 0.21576155433883565  JSD(q8) = 0.21576155433883565  JSD(q1)= 0.3803956658485779  JSD(q16)= 0.3803956658485779
runtime ~8.3 s (including venv creation)
answer.json sha256 F5BE328B8333AACA434F46B6E97A7BD10F971A630C65664FD8A7C87E385A9DEB
```

**Result:** `INSTALL.md` verified end-to-end in a fresh venv. `answer.json`
was regenerated with the same contents as before.

**Invalidates:** nothing.

**Next:** done.
