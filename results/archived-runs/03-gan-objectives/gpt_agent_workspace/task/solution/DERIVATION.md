# Derivation

Let \(n=4\), let \(k\) be the size of the generator support, and put
\(t=k/n\). The supports are nested and their overlap has size
\(\min(n,k)\).

## Jensen--Shannon ties

For \(k\leq n\), direct substitution into the two KL terms gives

\[
\begin{aligned}
\operatorname{JSD}(p,q_k)
=\frac12\bigg[
&\frac{k}{n}\log\frac{2k}{k+n}
+\frac{n-k}{n}\log 2\\
&+\log\frac{2n}{k+n}
\bigg].
\end{aligned}
\]

After simplification, this is

\[
F(t)=\log 2+\frac12\left[t\log t-(1+t)\log(1+t)\right],
\qquad 0<t\leq1.
\]

JSD is symmetric in its two arguments, so for \(k\geq n\) the value is
\(F(1/t)\). Thus every value is \(F(u)\), where
\(u=\min(t,1/t)\). Moreover,

\[
F'(u)=\frac12\log\frac{u}{1+u}<0.
\]

Consequently there are no same-side ties: the values are strictly decreasing
for integer \(k<4\) and strictly increasing for integer \(k>4\). A cross-side
pair \(j<4<k\) ties exactly when
\(j/4=4/k\), or \(jk=16\). The only distinct positive-integer pairs are
\((1,16)\) and \((2,8)\). This proves the JSD list is complete and gives
\(\widehat k=8\).

## Least-squares ties

Pointwise minimization of the discriminator objective gives

\[
D^*_{\rm LS}(x)=\frac{p(x)-q(x)}{p(x)+q(x)}
\]

where \(p(x)+q(x)>0\). Therefore

\[
V_{\rm LS}(q)=\frac12\sum_x
\frac{(p(x)-q(x))^2}{p(x)+q(x)}.
\]

Separating overlap and exclusive-support points and simplifying yields

\[
V_{\rm LS}(q_k)=\frac{|k-4|}{k+4}
=\frac{1-u}{1+u},\qquad
u=\min(k/4,4/k).
\]

This is strictly decreasing in \(u\), so exactly the same completeness
argument applies: there are no same-side ties, and cross-side ties require
\(jk=16\). Hence the complete LS tie list is also
\((1,16),(2,8)\), and

\[
V_{\rm LS}(q_2)=\frac13.
\]

## GAN losses for the JSD partner of \(q_2\)

The optimal minimax discriminator is \(D^*(x)=p(x)/(p(x)+q(x))\).
For \(q_2\), it equals \(1/3\) on the overlap and \(1\) on the two
data-only points. For \(q_8\), it equals \(2/3\) on the overlap and \(0\)
on the four generator-only points. It follows that

\[
\begin{array}{c|ccc}
&V_\star&L_{\rm MM}&L_{\rm NS}\\
\hline
q_2&\log2-\tfrac32\log3&\log(2/3)&\log3\\
q_8&\log2-\tfrac32\log3&-\tfrac12\log3&+\infty
\end{array}
\]

Thus \(V_\star\) ties, \(L_{\rm MM}\) prefers \(q_8\), and \(L_{\rm NS}\)
prefers \(q_2\).
