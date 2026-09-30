# Iteration 02: talk-oriented DGP stress grid

## Why this iteration

Iteration 01 compared only Gaussian and moderately skewed log-normal margins,
with complete data or MCAR. That was enough to find a problem, but not enough
to say whether it was an ordinary finite-moment robustness failure, a
small-sample tail problem, or a deliberate violation of the statistic's
asymptotic contract. This iteration expands the mechanisms and labels those
interpretive regions before comparing methods.

## Design

The affine bridge is unchanged: the null is a five-variable zero-covariance
model and the alternative frees all ten covariances and is saturated. Thus the
nested score is a genuine 10-df global GOF score in this special model. All
cells use `n = 120`, 500 replications, and 199 multiplier draws. Matched power
cells introduce a latent Gaussian/copula correlation of .35 between `x1` and
`x2`.

Seven outcome laws are crossed with five observation mechanisms:

- Gaussian, `t(8)`, and `t(5)`;
- standardized log-normal margins with `g = 0.8` and `g = 1.2`;
- 5% shared row-scale contamination with scale multiplier 6;
- `t(3)`, which has no fourth moment;
- complete data, 30% MCAR, nonmonotone MAR driven by always-observed `x1`,
  monotone MAR dropout driven by `x1`, and self-masked MNAR.

This gives 35 matched mechanisms and 70 null/power cells. The primary region
contains the four Gaussian mechanisms other than MNAR plus the ten
finite-fourth-moment non-Gaussian complete/MCAR mechanisms. Nonnormal MAR is a
separate common-use pseudo-ML stress region. `t(3)` and MNAR are deliberate
moment and ignorability violations, not evidence against a promised property.

The run took 230.5 seconds on four workers. All 35,000 restricted and saturated
fits converged, and the LR, score, and Wald computations reported no failures.
MLR itself returned 168 non-finite null and 167 non-finite power p-values, almost
all in the severe-tail cells.

## Primary null calibration

The effective multiplier score was the clear calibration winner across the 14
primary scenarios. Every cell was between .032 and .068, its mean absolute
size error was .009, and its pooled rejection rates were .0460 in the Gaussian
reference region and .0426 in the finite-moment nonnormal region.

The analytic score corrections were usable but conservative. Score SB ranged
from .016 to .060 (mean absolute error .0134), while score pEBA4 ranged from
.004 to .056 (mean absolute error .0220). The effective multiplier remained
well calibrated even in the diagnostic regions: .0308 pooled under nonnormal
MAR, .0470 at the `t(3)` moment boundary, and .0577 under MNAR. The latter two
are empirical sensitivity results, not general validity claims.

The H1-based alternatives were much less even. LRT pEBA4 ranged from .038 to
.394 in the primary cells (mean absolute error .0620), and robust Wald ranged
from .060 to .192. The exact LRT mixture had a smaller mean error than LRT
pEBA4, but its .014--.184 range makes it an unreliable headline alternative.

Yuan--Bentler/Mplus MLR was liberal even in the primary region. Gaussian
reference rejection pooled to .0780. Finite-moment nonnormal rejection pooled
to .3166, including roughly .16--.32 under log-normal margins and .67--.83
under 5% row-scale contamination. Across the 14 primary cells it ranged from
.068 to .829 and had mean absolute size error .1988. This is the clean talk
contrast: the failure does not depend on the `t(3)` or MNAR contract-violation
cells.

## Detection

Raw power is not comparable when size differs. Within each mechanism, using
the empirical fifth percentile of the method's null p-values as a diagnostic
cutoff gave primary-region mean size-adjusted power .423 for the effective
multiplier, .405 for LRT pEBA4, .404 for LRT SB, .391 for score SB, and .390
for score pEBA4. MLR reached only .343 after matching its null size. Across all
35 mechanisms the effective multiplier again ranked first at .314, narrowly
ahead of the exact LRT mixture (.308) and LRT pEBA4/SS (.307).

The log-normal transforms reduce the observed signal: average pre-missing
`r12` is about .289 for `g=.8` and .234 for `g=1.2`, versus .348 for Gaussian
data. The report therefore shows realized pre- and post-missing correlations;
cross-DGP raw-power differences should not be read as pure method effects.

## MLR oracle

The expanded same-data oracle fit magmaan and lavaan MLR to 875 null datasets
(25 per mechanism). Five p-values were non-finite in both implementations.
There were zero finite/non-finite, .05-decision, or df mismatches. Among finite
p-values the maximum absolute difference was .000281; mean absolute difference
was `2.89e-6`. The largest scaled-statistic difference was .432 only because
the two statistics were approximately 1899.2 and 1899.6; the relative
difference was below .00023 and both p-values were zero. The oracle therefore
confirms that the simulation's MLR behavior is lavaan behavior, not an
experiment-local reconstruction.

## Decision

For a talk, retain:

- Yuan--Bentler/Mplus MLR as the familiar target;
- the effective multiplier score as the leading alternative;
- score SB as the simplest analytic H0-only alternative;
- score pEBA4 as the spectrum-aware analytic alternative;
- LRT pEBA4 and robust Wald as H1-based comparators, not recommendations.

The next credibility checks are not more exotic failure mechanisms. First,
repeat a reduced primary grid at larger `n` to distinguish slow convergence
from persistent calibration failure. Second, implement the general
saturated-moment H0 score projection and repeat the comparison in an ordinary
latent SEM.
Until then, the defensible claim is about this covariance-model bridge at
`n=120`, not Yuan--Bentler tests in all SEMs.
