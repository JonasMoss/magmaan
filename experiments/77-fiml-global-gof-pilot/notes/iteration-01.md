# Iteration 01: affine global GOF bridge

## Design

The restricted model is a five-variable independence covariance model. Its
less-restricted counterpart frees all ten observed covariances and is saturated,
so the existing affine nested score/flip machinery is a genuine global GOF test
in this special case. The eight cells cross normal versus standardized
log-normal margins, complete versus 30% MCAR data, and null versus one omitted
Gaussian-copula correlation of .35. `x1` is always observed. Each cell has
`n = 120`, 10 df, 500 replications, and 199 multiplier draws.

The persisted run completed in 25.9 wall-clock seconds on four workers (28.3
seconds including R startup and shutdown). All 4,000 restricted and
saturated fits converged; the LRT spectrum, score, flip, and Wald computations
had no reported failures. The MLR p-value was non-finite in three extremely
small-p power replications, while every other recorded method remained finite.

## Null calibration

The effective multiplier score was the cleanest result: its four cellwise
rejection rates were .048, .040, .046, and .050 (pooled .046). Ordinary score
chi-square ranged from .042 to .070 (pooled .0565). Score pEBA4 was mildly
conservative (.030--.048, pooled .0365), while the direct sandwich and exact
score mixture were more conservative still (pooled .0335 and .0205).

LRT pEBA4 was usable but less even (.042--.072, pooled .0565); the exact mixture
was conservative overall (.041). MLR was clearly broken by skewness: rejection
rose from .060--.070 in the normal cells to .130 without missingness and .206
with 30% MCAR (pooled .1165). The robust Wald was also liberal, including .186
under normal data with 30% MCAR (pooled .113). Model-based Wald was better but
reached .100 in the skewed-missing cell.

## Detection

Raw pooled power was .583 for LRT pEBA4, .547 for effective flips, .487 for
score pEBA4, .345 for the score sandwich, and .540 for robust Wald. Because the
methods have different size, the diagnostic matched-null adjustment is more
informative. Mean size-adjusted power over the four distribution/missingness
cells was .587 for LRT SB, .583 for LRT pEBA4, .573 for effective flips, .555
for score SB, and .548 for score pEBA4. Thus the effective flip's excellent
calibration did not require a material detection sacrifice in this pilot.

## Decision

Keep three representatives for the next iteration:

- LRT pEBA4 as the strongest existing H1-based comparator;
- global score pEBA4 as the deterministic H0-only spectrum method;
- effective multiplier score as the best-calibrated method here.

Retain MLR and robust Wald only as familiar negative controls. Do not spend the
next iteration on the exact score mixture, direct sandwich score, or a
permutation Wald: they were conservative or unstable here, and the global
permutation construction remains special to this independence model.

The next useful step is the general saturated-moment score projection for an
ordinary latent SEM. That removes this pilot's affine-independence shortcut and
directly tests whether H0-only score methods avoid the saturated-H1 collapse in
the current FIML-FMG paper cells.
