# Iteration 17: RLS as the ML2S base statistic

## Construction

Two-stage ML first estimates saturated Gaussian-FIML moments
\(\widehat\eta_{\mathrm{EM}}\), then fits the SEM to those moments by ordinary
normal-theory ML. This makes fitted-model RLS a direct Stage-2 residual
statistic rather than an approximation constructed from a direct-FIML
displacement. Define the Stage-1-to-model residual

\[
r_b=
\begin{pmatrix}
\widehat\mu_{\mathrm{EM},b}-\widehat\mu_b\\
\operatorname{vech}(\widehat S_{\mathrm{EM},b}-\widehat\Sigma_b)
\end{pmatrix}.
\]

With the working-normal moment covariance evaluated at the fitted model,

\[
\Omega_{\mathrm{NT}}(\widehat\eta_b)=
\begin{pmatrix}
\widehat\Sigma_b & 0\\
0 & \Gamma_{\mathrm{NT}}(\widehat\Sigma_b)
\end{pmatrix},
\]

the full residual statistic is

\[
T_{\mathrm{2S,RLS}}
=
\sum_b n_b r_b^\top\Omega_{\mathrm{NT}}(\widehat\eta_b)^{-1}r_b
\]

\[
=\sum_b n_b\left[
(\widehat\mu_{\mathrm{EM},b}-\widehat\mu_b)^\top
\widehat\Sigma_b^{-1}
(\widehat\mu_{\mathrm{EM},b}-\widehat\mu_b)
+\frac12\operatorname{tr}\left\{
\left[\widehat\Sigma_b^{-1}
(\widehat S_{\mathrm{EM},b}-\widehat\Sigma_b)\right]^2
\right\}
\right].
\]

The six-indicator CFA used here has freely estimated indicator intercepts, so
there is no residual mean component. The implemented frontier primitive returns
the mean term, covariance term, and their sum separately; it treats means as
saturated when means were not part of the fitted moment structure. The older
`rls_chi2()` remains covariance-only because that is its lavaan
`browne.residual.nt.model` parity contract.

Under the pseudo-null, Stage-2 ML and RLS are asymptotically equivalent. The
same two-stage \(U\Gamma\) spectrum is therefore used to calibrate both base
statistics. This is a statistic-only crossing: saturated EM moments, fitted
SEM, degrees of freedom, and spectrum are held fixed.

## Smoke design

The run used the six-indicator one-factor model, 200 replications per cell,
\(n\in\{120,500\}\), normal/VM2/IG2 generators, and complete/30% MCAR/30% MAR
observation mechanisms. Nonnormal-MAR cells remain estimator-level alternatives
and are excluded from size summaries. All 3,600 ML2S fits and RLS evaluations
were usable.

## Results

At \(n=120\), RLS was generally slightly smaller than Stage-2 ML. The median
RLS/ML statistic ratios under normal data were 0.988 for complete data, 0.952
under MCAR, and 0.979 under MAR. At \(n=500\), the corresponding ratios were
0.993, 0.987, and 0.996.

The normal-data rejection rates show where the finite-sample gain occurs:

| \(n\) | mechanism | correction | Stage-2 ML | Stage-2 RLS |
|---:|---|---|---:|---:|
| 120 | complete | SS | 0.020 | 0.020 |
| 120 | MCAR | SS | 0.085 | 0.055 |
| 120 | MAR | SS | 0.045 | 0.040 |
| 120 | complete | pEBA(4) | 0.025 | 0.025 |
| 120 | MCAR | pEBA(4) | 0.105 | 0.080 |
| 120 | MAR | pEBA(4) | 0.050 | 0.045 |
| 500 | complete | SS | 0.035 | 0.035 |
| 500 | MCAR | SS | 0.050 | 0.050 |
| 500 | MAR | SS | 0.040 | 0.030 |

Mean absolute size error across the three normal mechanisms was:

| \(n\) | correction | Stage-2 ML | Stage-2 RLS |
|---:|---|---:|---:|
| 120 | SB | 0.030 | 0.022 |
| 120 | SS | 0.023 | 0.015 |
| 120 | pEBA(4) | 0.027 | 0.020 |
| 120 | full mixture | 0.023 | 0.017 |
| 500 | SB | 0.007 | 0.010 |
| 500 | SS | 0.008 | 0.012 |
| 500 | pEBA(4) | 0.007 | 0.010 |
| 500 | full mixture | 0.010 | 0.012 |

Across all seven pseudo-null cells, including nonnormal complete and MCAR data,
the same modest small-sample advantage remained:

| \(n\) | correction | Stage-2 ML | Stage-2 RLS |
|---:|---|---:|---:|
| 120 | SB | 0.106 | 0.088 |
| 120 | SS | 0.056 | 0.045 |
| 120 | pEBA(4) | 0.089 | 0.077 |
| 120 | full mixture | 0.054 | 0.041 |
| 500 | SB | 0.031 | 0.033 |
| 500 | SS | 0.013 | 0.013 |
| 500 | pEBA(4) | 0.024 | 0.027 |
| 500 | full mixture | 0.016 | 0.014 |

## Interpretation

Unlike the direct-FIML displacement quadratics, two-stage RLS is a natural
statistic: Stage 1 gives an estimated moment vector and Stage 2 explicitly fits
that vector. Evaluating normal-theory curvature at the fitted covariance gives
the conventional RLS stabilization. It improves the difficult \(n=120\)
missing-data cells without changing the estimator or spectrum, while becoming
nearly indistinguishable from ML by \(n=500\).

The gain is real but not large enough to reverse the broader pilot conclusion
that ML2S is a secondary story. If ML2S is retained, RLS paired with SS or the
full mixture is the more attractive finite-sample formulation to carry into a
larger validation.

As an implementation smoke for restricted means, the five-wave linear growth
model was run for 20 normal replications in each of complete, 30% MCAR, and 30%
MAR conditions at \(n=120\). All 60 fits were usable. The mean component was
nonzero in every replication and its cell means were 2.59, 4.47, and 3.31,
respectively; the corresponding covariance-component means were 7.85, 11.08,
and 10.93. In every replication the reported total equalled the two components
to numerical tolerance.

Reproduce with:

```sh
Rscript experiments/research/banked/44-fiml-global-tests/lanes/global/smoke_ml2s_rls.R --reps 200

# Restricted-mean implementation smoke
Rscript experiments/research/banked/44-fiml-global-tests/lanes/global/smoke_ml2s_rls.R \
  --reps 20 --n 120 --model linear_growth_5 --distributions normal
```
