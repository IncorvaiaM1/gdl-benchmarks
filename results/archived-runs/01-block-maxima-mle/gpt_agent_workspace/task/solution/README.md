# Exponential block-maximum MLE

For one exponential draw with rate \(\lambda\),

\[
F_X(y)=1-e^{-\lambda y}, \qquad y\geq 0.
\]

The maximum \(Y\) of \(m\) independent draws therefore has

\[
F_Y(y)=(1-e^{-\lambda y})^m
\]

and density

\[
f_Y(y;\lambda)
=m\lambda e^{-\lambda y}(1-e^{-\lambda y})^{m-1}.
\]

For independent observations \(y_1,\ldots,y_n\), the log-likelihood is

\[
\ell(\lambda)
=n\log m+n\log\lambda-\lambda\sum_i y_i
 +(m-1)\sum_i\log(1-e^{-\lambda y_i}).
\]

Its score is

\[
\ell'(\lambda)
=\frac{n}{\lambda}-\sum_i y_i
 +(m-1)\sum_i\frac{y_i}{e^{\lambda y_i}-1}.
\]

The score is strictly decreasing because

\[
\ell''(\lambda)
=-\frac{n}{\lambda^2}
 -(m-1)\sum_i
 \frac{y_i^2e^{\lambda y_i}}{(e^{\lambda y_i}-1)^2}<0.
\]

Thus its unique positive root is the maximum-likelihood estimate. The entry
point brackets that root and solves it by bisection:

```powershell
python task/solution/estimate.py
```

The script uses only the Python standard library, reads `data/train.csv`, and
writes `task/results/estimate.json`.
