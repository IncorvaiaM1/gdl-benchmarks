# Maximum likelihood from exponential block maxima
Read `task/TASK.md` for the challenge. `gt_code` and `tests` are evaluator-only material; they must not be visible to an agent under evaluation. Existing `WRITEUP.md`, where present, contains spoilers.

Run the original grader from this directory after an agent writes the required JSON into `task/results`:

```sh
python -m pytest tests -q
```

Run each task in a separate process: the original graders reuse the Python package name `tests`.
Use the repository export utility to prepare an agent-only workspace. Reference code is not an agent submission and does not necessarily write the required answer file.
