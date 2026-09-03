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

## Restrictions that are zero under H0

The restricted pseudo-parameters above are not enough to demonstrate model
failure: by construction they always lie in H0. For this one-factor model,
literal cross-loadings do not exist. The corresponding zero-under-H0
quantities are off-diagonal residual covariances. Holding the H0 pseudo-true
factor component fixed, define the descriptive H1 shadow residual as

`Theta_shadow,ij = Sigma_H1*,ij - Sigma_H0*,ij`, for `i != j`.

The largest values are:

| Generator | Residual pair | Shadow covariance | Standardized shadow value |
|---|---|---:|---:|
| VM2 | x3 ~~ x2 | .0287 | .0237 |
| VM2 | x6 ~~ x2 | .0271 | .0229 |
| VM2 | x5 ~~ x2 | .0308 | .0226 |
| VM2 | x2 ~~ x1 | -.0258 | -.0192 |
| IG2 | x3 ~~ x2 | .0245 | .0200 |
| IG2 | x2 ~~ x1 | -.0266 | -.0196 |
| IG2 | x6 ~~ x2 | .0239 | .0196 |
| IG2 | x5 ~~ x2 | .0271 | .0195 |

These are descriptive coordinates because a saturated covariance has no unique
factor/residual decomposition. A parameterization-free demonstration uses the
one-factor model's vanishing tetrads. At the generating correlation matrix the
tetrads are zero to numerical precision. At the saturated FIML pseudo-target,
the maximum absolute tetrad is .0340 for VM2 and .0313 for IG2. The largest
violations repeatedly involve the always-observed `x1,x2` pair together with
two variables selected for missingness. Thus H0 is explicitly false at the
observed-data H1 pseudo-target, not merely represented by shifted parameters
inside the best-fitting H0.

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

## Prospective decision

From the next design onward, the Gaussian-FIML pseudo-null is the only null.
Writing the fitted SEM as \(\mathcal M_0\) and the saturated Gaussian-FIML
pseudo-target as \(\eta_1^\star\), a DGP enters a calibration table only if

\[
\eta_1^\star \in \mathcal M_0.
\]

A full-data covariance generated inside \(\mathcal M_0\) is not itself a null
certificate under MAR. The original VM2/IG2 MAR arms in this note are therefore
pseudo-model alternatives. The Gaussian shadow satisfies the pseudo-null but
does not preserve the nonnormal-MAR problem, so a future information-matrix
comparison still needs a deliberately constructed nonnormal-MAR pseudo-null.

Operationally, every proposed MAR null must be screened before simulation with
a very-large-sample saturated H1 fit and restricted H0 fit. Admission requires
a restricted--saturated per-case discrepancy indistinguishable from zero at
the stated numerical/Monte Carlo tolerance and negligible zero-under-H0 shadow
directions. Power perturbations are then defined relative to this same
pseudo-null target.
