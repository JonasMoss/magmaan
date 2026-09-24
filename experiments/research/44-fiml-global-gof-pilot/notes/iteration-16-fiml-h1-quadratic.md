# Iteration 16: one-stage FIML residual-curvature quadratics

## Question

Let

\[
e=\widehat\eta_1-\eta(\widehat\theta_0)
\]

be the saturated-FIML moment displacement from the restricted fitted model.
This diagnostic compares the FIML LR statistic with three one-stage residual
quadratics:

\[
T_{1O}=e^\mathsf{T}\widehat H_{\eta,O}(\widehat\eta_1)e,
\qquad
T_{0O}=e^\mathsf{T}\widehat H_{\eta,O}
  \{\eta(\widehat\theta_0)\}e,
\qquad
T_{0E}=e^\mathsf{T}\widehat H_{\eta,E}
  \{\eta(\widehat\theta_0)\}e.
\]

The subscripts identify H1 versus H0 evaluation and observed versus expected
curvature. Every matrix is the ambient saturated-coordinate information, not
the restricted-model Hessian in \(\theta\)-coordinates. Therefore these
statistics do not contain the additional second-derivative term created by
the nonlinear composition \(\eta(\theta)\). The observed H0 matrix does contain
realized-residual terms from differentiating the saturated likelihood at a
nonstationary ambient point; that is a distinct sense in which it differs from
working-normal expected information.

Magmaan reports saturated information in summed log-likelihood units, so none
of the displayed quadratics receives another factor of \(n\). They remain
direct one-stage FIML diagnostics: saturated EM supplies the unrestricted
likelihood point, while the SEM is fitted directly to the incomplete raw data.

## Design and implementation checks

The smoke uses the six-indicator one-factor model, 200 replications per cell,
\(n\in\{120,500\}\), normal/VM2/IG2 generators, and complete/30% MCAR/30% MAR
observation mechanisms. All three displacement quadratics are calibrated by
the same estimated FIML LR/FM G spectrum. The native expected-Fisher projected
score statistic is also recorded with its own spectrum. Nonnormal-MAR cells
are estimator-level alternatives and are excluded from pseudo-null summaries.

The experiment reconstructs the observed saturated FIML Hessian directly from
the observed-pattern likelihood. At H1 it agrees with magmaan's analytic
information to a maximum absolute discrepancy below \(1.1\times10^{-10}\).
Under complete data, \(T_{0E}\) equals conventional fitted-covariance RLS; the
mean absolute numerical discrepancy was below \(2\times10^{-6}\) in every
cell. These checks identify both the moment ordering and likelihood scaling.

The run attempted 3,600 fits and retained 3,596. One VM2/MCAR fit failed to
converge, two fits failed the FMG spectrum rank diagnostic, and one additional
VM2/MCAR row had an unstable score tangent rank.

## Result

| pseudo-null pEBA(4) rejection | \(n=120\) | \(n=500\) |
|---|---:|---:|
| LR | 0.107 | 0.062 |
| H1 observed quadratic, \(T_{1O}\) | 0.305 | 0.114 |
| H0 observed quadratic, \(T_{0O}\) | 0.184 | 0.083 |
| H0 expected quadratic, \(T_{0E}\) | 0.160 | 0.076 |
| native expected-Fisher score | 0.042 | 0.039 |

Moving the residual quadratic from H1 curvature to H0 curvature helps
substantially, and expected H0 curvature has the best pooled quadratic result.
Neither H0 variant matches the native score statistic. Merely sharing its
expected-Fisher metric does not reproduce the score construction: the score
uses the exact gradient at H0 and an explicit nuisance projection, whereas
\(e\) is a finite displacement connected to that gradient only through a local
Taylor approximation.

The observed H0 ambient Hessian was indefinite in 6.2% of the pooled
pseudo-null replications at \(n=120\), including 24.9% of VM2/MCAR and 9.0% of
IG2/MCAR replications. It was positive definite in every pseudo-null
replication at \(n=500\). Thus evaluation at H0 removes the H1 endpoint problem
but introduces a finite-sample definiteness problem when realized-residual
terms are retained.

The complete-data means show the exact RLS reduction and the remaining effect
of observed H0 curvature:

| \(n\) | generator | LR | \(T_{1O}\) | \(T_{0O}\) | \(T_{0E}=\) RLS |
|---:|---|---:|---:|---:|---:|
| 120 | normal | 8.719 | 9.975 | 8.880 | 8.554 |
| 120 | VM2 | 27.733 | 40.935 | 33.447 | 26.983 |
| 120 | IG2 | 10.066 | 11.450 | 11.330 | 10.056 |
| 500 | normal | 8.573 | 8.899 | 8.543 | 8.514 |
| 500 | VM2 | 30.828 | 33.911 | 33.412 | 30.765 |
| 500 | IG2 | 9.067 | 9.251 | 9.401 | 9.092 |

Under MCAR, expected H0 curvature remains imperfect at small N. For VM2 at
\(n=120\), pEBA(4) rejection was 0.386 for \(T_{0E}\), 0.371 for \(T_{0O}\),
and 0.244 for LR, while the native score gave 0.041. At \(n=500\), these rates
were 0.175, 0.180, 0.135, and 0.030. The expected metric is therefore not, by
itself, the reason for the score statistic's good calibration.

## Decision

All three formulas are correctly scaled local quadratic approximations, and
their agreement with LR improves with sample size. The H0 versions are better
than the H1 version, but neither is competitive with the native expected-score
pairing in this smoke. Do not add a displacement quadratic to the planned
paper battery. The result is still theoretically useful: it separates
curvature evaluation from the use of scores and shows why “use expected Fisher”
does not identify a statistic without also specifying the vector being made
quadratic and its nuisance projection.

Reproduce with:

```sh
Rscript experiments/research/44-fiml-global-gof-pilot/smoke_fiml_h1_quadratic.R --reps 200
```
