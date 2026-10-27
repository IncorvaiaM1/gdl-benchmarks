# Data provenance

`data/images.csv` holds 1000 images drawn without replacement (fixed random
seed) from the 10,000-image MNIST test split. Each row is one 28×28 image
flattened in row-major order to 784 columns `p0` ... `p783`. Each entry is
the raw 8-bit pixel value, an integer in `{0, ..., 255}`.

| File | Rows | Columns | Given to |
|---|---|---|---|
| `data/images.csv` | 1000 (+ header) | 784 | the solving agent |

## Integrity check

A reader (or agent) can confirm they have the exact intended file by
recomputing its SHA-256 and comparing:

```
sha256sum data/images.csv
# 98b5d6a6d90bc2bc1c22df168e45e63da97fa2cce5d75001650f7f391a8b69c4
```
