# Standard prompt (paste verbatim, same for all three models, no extra hints)

You are working in this workspace. Read `task/TASK.md` and
`task/PROJECT_INSTRUCTION.md`, then complete the task exactly as specified.

Rules:
- Work only with the files in this workspace. Do not open, search for, or
  read any files outside this folder.
- Do not search the internet for this specific problem or its answer, and do
  not use any existing solution to it. General references on normalizing
  flows, VAEs, change of variables, and dequantization are fine.
- Do not guess or reverse-engineer the grader. There are no tests or
  reference solutions in this workspace; solve the problem from first
  principles.
- Put your code in `task/solution/` with a single entry-point script, and
  write your final answer to `task/results/answer.json` in the exact format
  TASK.md requires.
- Keep `task/STATUS.md` and `task/INSTALL.md` up to date as
  PROJECT_INSTRUCTION.md describes.

When you are done, reply with a short summary of your derivation and your
final numbers.

---

## Run checklist (for you, not the model)

1. Unzip each workspace **outside** the randLab repo (e.g. Desktop), so the
   agent can't find `gt_code/`, `tests/`, or `WRITEUP.md` by searching.
2. Open each unzipped folder as its **own** Cursor window, one model per
   window, fresh chat, web search off if Cursor allows it.
3. Paste the prompt above, unchanged. Don't answer clarifying questions with
   hints; if asked, reply "Use your best judgment from TASK.md."
4. When it finishes, copy the whole workspace back into
   `Thing2/agent_runs/` and grade it:
   ```
   cp <workspace>/task/results/answer.json gdlbench_submission/task/results/
   cd gdlbench_submission && python -m pytest -q tests
   ```
   (move answer.json back out afterwards)
5. Record model name/version, Correct / Partial / Incorrect, and the failure
   mode in WRITEUP.md part d.
