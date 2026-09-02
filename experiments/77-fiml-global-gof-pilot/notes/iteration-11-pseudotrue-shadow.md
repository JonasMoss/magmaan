# Iteration 11: pseudo-true targets and a Gaussian shadow null

## Purpose

The nonnormal-MAR cells satisfy the generating full-data covariance model but
need not satisfy the Gaussian observed-data likelihood model used by FIML. This
check approximates both relevant Gaussian-FIML pseudo-targets:

- the restricted one-factor H0 target, represented by its fitted SEM
  parameters and implied moments; and
- the saturated H1 target, represented by the saturated-EM mean and covariance.

It then performs a deliberately artificial check: generate Gaussian data at
the numerical H0 pseudo-true moments, impose the same MAR mechanism, and rerun
the LRT/FMG tests. This “Gaussian shadow” makes the working-model null true but
does not preserve the original nonnormal distribution.

## Design

`estimate_pseudotrue_shadow.R` used the six-indicator one-factor model, VM2 and
IG2 generators, and the pilot 30% MAR rule. Each pseudo-target was approximated
with one 200,000-case draw. The shadow check used 1,000 replications at
`n = 500`. The original and Gaussian-shadow arms used the same model and MAR
rule; only the complete-data distribution and its moments differed.

## Estimated pseudo-target drift

| Quantity | VM2 MAR | IG2 MAR |
|---|---:|---:|
| Restricted-vs-saturated LR / case | .006533 | .006551 |
| `n * discrepancy` at n = 500 | 3.267 | 3.276 |
| Maximum absolute H0 parameter drift | .1490 | .1285 |
| RMS H0 parameter drift | .0731 | .0601 |
| Maximum absolute H1 moment drift from generating moments | .1882 | .1458 |
| Maximum absolute H0-H1 pseudo-moment gap | .0308 | .0271 |

The largest parameter movements are residual variances. For VM2, the fitted
pseudo-true residual variances for `x3`--`x6` are .410, .502, .360, and .451,
versus generating values .550, .650, .500, and .600. For IG2 they are .421,
.544, .379, and .483. The factor variance moves from 1 to about 1.042 for VM2
and 1.037 for IG2, while most loadings move downward.

The H0 pseudo-true parameter vector therefore exists and is estimable, but it
does not make the global-fit null true for the original distribution. The
saturated Gaussian pseudo-target remains outside the one-factor model, as
shown by the nonzero H0-H1 moment gap and positive per-case LR discrepancy.

## Gaussian shadow experiment

| Generator label | Arm | Mean LRT | SB reject | pEBA(4) reject | ALL reject |
|---|---|---:|---:|---:|---:|
| VM2 | Original nonnormal MAR | 36.86 | .187 | .159 | .107 |
| VM2 | Gaussian H0 shadow MAR | 9.46 | .074 | .072 | .068 |
| IG2 | Original nonnormal MAR | 15.08 | .234 | .206 | .150 |
| IG2 | Gaussian H0 shadow MAR | 9.10 | .058 | .056 | .053 |

All 4,000 fits were usable and the target-column missing rates averaged .300--
.301. The shadow results are compatible with ordinary finite-sample error at
`n = 500`, especially given that the earlier normal-MAR pEBA(4) cell was .050
in 500 replications and .067 in a separate 1,000-replication run.

## Interpretation

The shadow is useful for exposition: the same numerical H0 parameters and MAR
rule behave normally once the Gaussian working model is true. It is not a
repair or a valid recentering of the original nonnormal-MAR global test. Merely
plugging the H0 pseudo-true parameters into the original likelihood cannot
remove the saturated-vs-restricted population discrepancy. A parameter test of
the H0 pseudo-true vector would be a different inferential question from global
goodness of fit.

The scientifically honest presentation is therefore:

1. show the generating parameters;
2. show the H0 and H1 Gaussian-FIML pseudo-targets under nonnormal MAR;
3. show that their discrepancy is positive;
4. optionally show the Gaussian shadow as an implementation illustration; and
5. label the original nonnormal-MAR rejection as pseudo-model/estimand stress,
   not Type-I error.
