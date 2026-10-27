# Project instructions

IMPORTANT: Do not cheat by using online solutions or hints specific to this task.
IMPORTANT: Write your solution in solution/ directory

Two files carry the reproducibility burden for this project: `STATUS.md` (what was
done) and `INSTALL.md` (what it was done with). Keep both current as you work.

**The bar for both:** someone with no memory of this project — a future session, a
collaborator, you in three months — should be able to rebuild the environment from
`INSTALL.md` and reproduce every result by reading `STATUS.md` top to bottom and
running the commands in order. If that would not work, the files are not done.

---

## 1. `STATUS.md` — append-only work log

**Write an entry after every command that changes state.** Installs, data
generation, training runs, calibration, test runs, anything that writes a file or
alters the environment. Not for edits to prose or comments.

**Append only. Never edit or delete an earlier entry.** If something you wrote
turns out to be wrong, add a new entry saying so. A log you have rewritten is a
summary, not evidence.

Use this format:

````markdown
## 2026-08-18 14:32 — regenerate task data

**Command:**
```bash
python make_data.py --out-dir ./data
```

**Exit status:** 0

**Key output:**
```
train.npz         (20000, 8)  0.4545 mean bit rate   sha256:dbdd420f9ee12caf
true entropy      0.59191 nats/dim
```

**Result:** Wrote `data/train.npz` and `data/ground_truth.npz`. Refactoring split
the RNG stream, so the samples changed even though the distribution parameters did
not.

**Invalidates:** `tests/calibration.json` and `TV_THRESHOLD` — must re-run
`calibrate.py` before any result is trusted.

**Next:** re-run calibration, then re-verify the oracle.
````

Rules:

- **Record commands verbatim**, with every flag, exactly as run. A paraphrased
  command is not reproducible. Copy-pasteable or it does not count.
- **Record failures too**, with the actual error text. What did not work and why is
  the highest-value content in this file, and it is the first thing lost otherwise.
  A failed approach you do not write down will be attempted again.
- **Quote the numbers you will need later** — metrics, hashes, thresholds,
  runtimes, seeds. Do not summarize them away. "Training went well" is useless;
  `nll/dim 0.59125, TV 0.02578, 8.1s` is the record.
- **Record anything affecting determinism**: seeds, library versions that changed,
  hardware, thread counts.
- **State what a change invalidates.** If an artifact changed, list the downstream
  steps that must be re-run before their results can be trusted. This is not
  hypothetical: a refactor that changes an RNG stream will silently invalidate a
  calibrated threshold, and nothing will error.
- Start the file with an entry recording the initial state: repo, commit, machine,
  date.

---

## 2. `INSTALL.md` — how to rebuild the environment

**Create it at the first install**, as a stub declaring the base environment
(OS, Python version, how the environment is created). **Then keep it current** as
dependencies are added — it is a live lockfile, not a one-time document. You cannot
write the full version list before installing anything, so do not try.

Must contain:

- **Platform:** OS and architecture, `python -V` output, and for GPU work the CUDA
  and driver versions.
- **Every package with an exact pinned version** — `numpy==2.1.3`, never `numpy`.
  Generate this from `pip freeze` (or `uv pip freeze` / `conda env export`). Do not
  write it from memory; you will get it wrong.
- **The install commands verbatim, in order**, including any index URLs
  (e.g. a CPU-only PyTorch wheel index) and flags.
- **Anything that is not pip:** system packages, dataset or model downloads,
  environment variables, cache locations.
- **Known-bad combinations** you hit, and how you resolved them.

**Verify it before calling it done.** Follow your own `INSTALL.md` from a clean
environment — a fresh virtualenv or a fresh container — and confirm the project
runs end to end. An install guide that has never been followed is a guess. Record
in `STATUS.md` when you last verified it and against what.

**When a version turns out to matter, say why inline.** If a pin works around a
bug, or an upgrade broke something, write the reason next to the pin. A future
reader needs to know which pins are load-bearing and which are incidental,
otherwise they will either upgrade the wrong one or be afraid to touch any of them.