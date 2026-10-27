# Standard prompt (paste verbatim, same for every model, no extra hints)

## Agent version (Cursor / Claude Code / Codex: workspace with task/ only)

You are working in this workspace. Read `task/TASK.md` and
`task/PROJECT_INSTRUCTION.md`, then complete the task exactly as specified.

Rules:
- Work only with the files in this workspace. Do not open, search for, or
  read any files outside this folder.
- Do not search the internet for this specific problem or its answer, and do
  not use any existing solution to it. General references on GANs,
  f-divergences, and LSGAN are fine.
- Do not guess or reverse-engineer the grader. There are no tests or
  reference solutions in this workspace; solve the problem from first
  principles.
- Put your code in `task/solution/` with a single entry-point script, put
  your derivation in `task/solution/DERIVATION.md`, and write your final
  answer to `task/results/answer.json` in the exact format TASK.md requires.
- Keep `task/STATUS.md` and `task/INSTALL.md` up to date as
  PROJECT_INSTRUCTION.md describes.

When you are done, reply with a short summary of your derivation and your
final answer.

## Chat version (plain chat models, no workspace)

Paste the following line, then the contents of `task/TASK.md` from
"# Task:" through the end of "## What you must report" (i.e. stop before
"## Deliverables"), then the JSON block from "## Deliverables" and its
formatting rules:

> Solve the following problem from first principles. Do not search the web.
> Show your derivation, and justify why your tie lists are complete. End
> your reply with a single JSON object in exactly the format given.

---

## Run checklist (for you, not the model)

1. Unzip `claude_agent_workspace.zip`, `gpt_agent_workspace.zip`, and `grok_agent_workspace.zip` (in this folder) **outside** the randLab repo
   (e.g. Desktop), so the agent can't find `gt_code/`,
   `tests/`, or `WRITEUP.md` by searching.
2. Open each copy as its **own** project window, one model per window, fresh
   chat, web search off where possible.
3. Paste the prompt above unchanged. Don't answer clarifying questions with
   hints; if asked, reply "Use your best judgment from TASK.md."
4. Grade it (chat models: save the final JSON as answer.json first):
   ```
   cp <workspace>/task/results/answer.json gdlbench_submission/task/results/
   cd gdlbench_submission && python -m pytest -q tests
   ```
   Move answer.json back out afterwards. `tests/statistics.json` shows which
   checks failed; the docstrings say what each check means.
5. Record model name/version, Correct / Partial / Incorrect, and the failure
   mode in WRITEUP.md part d. Suggested rule: Correct = 8/8; Partial = got
   the pairs right but missed part of the GAN-term comparison (or vice versa);
   Incorrect = wrong pairs and wrong preferences.
