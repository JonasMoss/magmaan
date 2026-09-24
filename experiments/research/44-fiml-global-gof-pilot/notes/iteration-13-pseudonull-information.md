# Iteration 13: expected, observed-H0, and observed-H1 information at a pseudo-null

## Question

How badly can the pattern-count expected Fisher construction fail when the
Gaussian-FIML pseudo-null is genuinely true? Can the realized saturated H1
information provide a stable alternative to the realized H0 information?

This check isolates information geometry from the pseudo-target problem. Every
condition satisfies the estimator-level null

\[
\eta_1^\star \in \mathcal M_0.
\]

## Exact pseudo-null construction

The four variables are mutually independent standardized exponentials. The
fitted model freely estimates every mean and variance and every covariance
within the `x2:x4` block. It fixes only the three covariances between the
always-observed driver `x1` and `x2:x4` to zero, giving three degrees of
freedom.

Variables `x2:x4` are jointly missing with probability

\[
\Pr(R_i=0\mid x_{i1})=
\operatorname{logit}^{-1}(a_\beta+\beta x_{i1}),
\]

where the intercept gives 50% marginal missingness. This is MAR with strict
positivity. Because `x1` is independent of `x2:x4`, selection on `x1` cannot
change the latter variables' marginal moments or create a covariance with
them. The saturated Gaussian-FIML population score is therefore zero at the
block-independent H0 moments for every value of \(\beta\).

The construction nevertheless breaks the expected-information geometry. The
realized sensitivity for an `x1`--outcome covariance depends on the selected
moments of `x1`. The expected Fisher approximation substitutes the unconditional
mean zero and variance one. More importantly, it sets the covariance--outcome-
mean sensitivity to zero, whereas the actual term depends on
\(E(x_1\mid R=1)\), which is nonzero under the tail rule. The empirical meat
cannot repair a nuisance projection built from the wrong sensitivity.

The deterministic population calculation is:

| MAR slope | `x1` mean when `x2:x4` observed | `x1` variance when observed | Fisher/actual variance ratio |
|---:|---:|---:|---:|
| 0.0 | 0.000 | 1.000 | 1.00 |
| 1.5 | -0.444 | 0.254 | 3.94 |
| 3.0 | -0.572 | 0.128 | 7.79 |
| 5.0 | -0.635 | 0.080 | 12.52 |

As a numerical pseudo-null gate, one 200,000-case draw at \(\beta=5\) had
50.07% missingness, an LR statistic of 1.080 on 3 df, per-case discrepancy
\(5.40\times10^{-6}\), and maximum absolute tested saturated-H1 covariance
0.0077. The exact null argument does not depend on this finite simulation; the
large draw checks its implementation.

## Information geometries

Five global-score constructions use the same fitted H0 score:

| Label | Nuisance sensitivity | Quadratic/spectrum metric |
|---|---|---|
| expected-H0 | expected Fisher at H0 | expected Fisher at H0 |
| observed-H0/expected-metric | realized Hessian at H0 | expected Fisher at H0 |
| observed-H1/expected-metric | realized Hessian at saturated H1 | expected Fisher at H0 |
| observed-H0 | realized Hessian at H0 | realized Hessian at H0 |
| observed-H1 | realized Hessian at saturated H1 | realized Hessian at saturated H1 |

The `observed-h1` option is now exposed by
`global_score_flip_test()`{.literal-code}. It changes
only the geometry; the saturated scores and observed statistic remain evaluated
at H0. H0 and H1 observed information share a probability limit under this
pseudo-null, but they are different finite-sample plug-ins.

## Simulation

The main grid used 1,000 replications per cell at \(n=200,500,2000\). A second
run used 1,000 replications at \(n=10{,}000\). All 80,000 reported score
constructions were usable. The table reports pEBA(4) rejection at 5% for the
three sensitivity choices while retaining the stable expected metric.

| MAR slope | n | Expected H0 | Observed H0 | Observed H1 |
|---:|---:|---:|---:|---:|
| 0.0 | 200 | .052 | .074 | .000 |
| 0.0 | 500 | .050 | .066 | .008 |
| 0.0 | 2,000 | .038 | .045 | .018 |
| 0.0 | 10,000 | .051 | .051 | .046 |
| 1.5 | 200 | .004 | .040 | .000 |
| 1.5 | 500 | .009 | .047 | .000 |
| 1.5 | 2,000 | .004 | .061 | .002 |
| 1.5 | 10,000 | .003 | .048 | .024 |
| 3.0 | 200 | .000 | .044 | .000 |
| 3.0 | 500 | .000 | .055 | .000 |
| 3.0 | 2,000 | .000 | .056 | .000 |
| 3.0 | 10,000 | .000 | .056 | .003 |
| 5.0 | 200 | .000 | .048 | .000 |
| 5.0 | 500 | .000 | .059 | .000 |
| 5.0 | 2,000 | .000 | .040 | .000 |
| 5.0 | 10,000 | .000 | .055 | .000 |

SB and the full plug-in mixture give the same qualitative verdict. At
\(\beta=3\) and 5 they also reject zero times with expected-H0 sensitivity,
while the observed-H0/expected-metric construction remains close to nominal.
The failure is therefore in the estimated geometry, not peculiar to pEBA(4).

Using observed H0 information as both sensitivity and metric is somewhat
liberal at the smaller sample sizes but approaches the hybrid result: at
\(\beta=5\), pEBA(4) rejection is .066, .071, .043, and .056 as n increases.
Unlike the previous broad latent-model audit, this targeted three-direction
model had no non-positive-definite H0 calls. That does not remove the earlier
PD warning for general SEMs.

## What happened to observed H1?

Observed H1 information is positive definite and had no failures, but it is a
poor finite-sample substitute in the adverse MAR cells. At \(\beta=5\), using
H1 sensitivity with the expected metric rejected zero times even at
\(n=10{,}000\); making H1 information the quadratic metric did not help.

This is slow convergence, not a different population target. In one
200,000-case \(\beta=5\) draw, the three eigenvalues using observed-H0 versus
observed-H1 sensitivity with the expected metric were respectively

\[
(0.0564,0.0576,0.0591)
\quad\text{and}\quad
(0.0568,0.0577,0.0593).
\]

Strong selection makes the saturated H1 curvature highly sensitive to its
noisy estimated cross-covariances. H0 fixes those covariances to their null
values, so its realized sensitivity is dramatically more stable in exactly
the directions being tested.

## Decision

1. Expected Fisher is not a generally valid nuisance sensitivity under
   nonnormal MAR, even at an exact Gaussian-FIML pseudo-null. Its size can be
   effectively zero rather than 5%, and the error persists with increasing n.
2. The best current construction is observed-H0 sensitivity with the stable
   expected metric. The metric need not equal the sensitivity because its
   scaling is carried by the robust spectrum; the nuisance projection does
   need the correct sensitivity.
3. Full observed-H0 geometry remains diagnostic-only because its positive-
   definiteness problem in broader SEMs is real and unnecessary here.
4. Observed-H1 geometry is asymptotically defensible and numerically PD, but its
   convergence is far too slow under strong selection to recommend it.
5. This block-independence model is a deliberately sharp information-matrix
   diagnostic, not a candidate for the paper's five substantive SEMs. The next
   gate is to construct pseudo-null versions of the source-based latent models
   and determine whether the same sensitivity failure is practically large.

## Reproduction

```sh
Rscript experiments/research/44-fiml-global-gof-pilot/investigate_pseudonull_information.R \
  --reps 1000 --n 200,500,2000 --beta 0,1.5,3,5 --cores 4

Rscript experiments/research/44-fiml-global-gof-pilot/investigate_pseudonull_information.R \
  --reps 1000 --n 10000 --beta 0,1.5,3,5 --cores 4 \
  --results-dir experiments/research/44-fiml-global-gof-pilot/results/pseudonull-information-n10000
```

The result CSVs are local ignored artifacts. The design, summary, and decision
are tracked here.
