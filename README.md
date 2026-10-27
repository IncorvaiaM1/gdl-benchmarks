# GDL Benchmarks
Coursework benchmark challenges by Michael Incorvaia for ECE 6257, Generative and Geometric Deep Learning, Georgia Tech, Fall 2026.

| Task | Focus |
|---|---|
| [Block-maxima MLE](tasks/01-block-maxima-mle) | Likelihood estimation from maxima of exponential samples |
| [Density comparison](tasks/02-density-comparison) | MNIST preprocessing, Jacobians, units, and score comparability |
| [GAN objectives](tasks/03-gan-objectives) | Divergence ties and generator-loss preferences |

## Setup
```sh
python -m pip install -r requirements.txt
python scripts/export_agent_workspace.py 01-block-maxima-mle
```
Run an agent only against the exported workspace in `exports`. Copy its result JSON into the corresponding task's `task/results`, then run `python -m pytest tests -q` from that task directory. Grade each task in a separate Python process because the original graders reuse package names. Reference implementations and grading code are evaluator-only; this public repository contains spoilers and is not itself an isolated agent workspace.

## Reproducibility
`python scripts/check_references.py` writes reference-generated answers into temporary task result files and runs all three original graders in separate processes. It restores any existing result files afterward. This checks packaging and reference/grader agreement, not model performance.

Task 02 uses the current three-model submission, not the earlier two-model version. See [archived agent evidence](results) for historical runs and limitations. Original `data/DOWNLOAD.md` files record dataset provenance. Task 01 includes synthetic train/holdout data; task 02 includes MNIST-derived image data; task 03 needs no data.

Related [coursework repository](https://github.com/IncorvaiaM1/generative-geometric-deep-learning). See [RIGHTS.md](RIGHTS.md) for authorship and reuse status and [publication notes](docs/PUBLICATION.md).
