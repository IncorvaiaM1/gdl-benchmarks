# Block-maximum exponential MLE

For one exponential draw with rate \(\lambda\),
\(F_X(y)=1-e^{-\lambda y}\) for \(y\geq0\). Therefore, for
\(Y=\max(X_1,\ldots,X_m)\),

\[
F_Y(y)=(1-e^{-\lambda y})^m,\qquad
f_Y(y)=m\lambda e^{-\lambda y}(1-e^{-\lambda y})^{m-1}.
\]

For observations \(y_1,\ldots,y_n\), the log-likelihood is

\[
\ell(\lambda)
=n\log m+n\log\lambda-\lambda\sum_i y_i
 (m-1)\sum_i\log(1-e^{-\lambda y_i}).
\]

Its score is

\[
\ell'(\lambda)
=\frac n\lambda-\sum_i y_i
+(m-1)\sum_i\frac{y_i}{e^{\lambda y_i}-1}.
\]

The second derivative is strictly negative for \(\lambda>0\), so the score is
strictly decreasing and its positive root is the unique MLE. `estimate.py`
brackets that root and solves it by bisection, evaluating the correction term
with `expm1` for numerical stability.

Run from the project root:

```powershell
python task\solution\estimate.py
```

This reads `data/train.csv` and writes `task/results/estimate.json`.
