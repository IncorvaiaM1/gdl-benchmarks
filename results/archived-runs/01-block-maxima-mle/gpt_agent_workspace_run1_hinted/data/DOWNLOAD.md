# Data provenance

Synthetic data, generated once with a fixed random seed by a private
instructor script (not included in this directory). The generative model is:

> Each observation `y` is the **maximum of `m = 4` independent draws** from
> an Exponential(λ) distribution, for some fixed but unknown rate λ > 0.
> Equivalently: `y = max(x_1, x_2, x_3, x_4)` where `x_1,...,x_4` are i.i.d.
> `Exponential(λ)`.

The same true λ was used to generate both files below. `m = 4` is given;
λ is exactly the quantity task/TASK.md asks you to estimate.

| File | Rows | Columns | Given to |
|---|---|---|---|
| `data/train.csv` | 2000 | 1 (`y`) | the solving agent |
| `data/holdout.csv` | 2000 | 1 (`y`) | the verifier only — **do not read this file** |

## Integrity check

A reader (or agent) can confirm they have the exact intended `train.csv` by
recomputing its SHA-256 and comparing:

```
sha256sum data/train.csv
# 7fc1dade9370cab98f675808ed0c7b9cea2b17bf974ec4d1860435db26c74587
```

(`data/holdout.csv`'s hash is intentionally not published here, since
publishing it alongside the file would defeat the point of withholding it.)
