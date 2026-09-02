# Iteration 10: statistic-by-spectrum crossing

## Question

When FIML LRT/FMG and projected-score pEBA(4) have different null rejection
rates, is the difference mostly due to the base statistic or to the estimated
weighted-chi-square spectrum?

The diagnostic crosses the two statistics and two spectra within every fitted
sample:

| Statistic | FMG spectrum |
|---|---|
| LRT | LRT |
| LRT | projected score |
| projected score | LRT |
| projected score | projected score |

This is a diagnostic decomposition, not four proposed tests. A statistic and
its spectrum are derived as a pair, so the two cross-paired combinations do not
in general have a justified reference distribution.

## Pilot design

The check used `check_lrt_score_crossing.R` with 1,000 replications per cell,
`n = 120`, the six-indicator one-factor model, normal/VM2/IG2 generators, and
complete versus 30% MAR data. The reported calibration is pEBA(4). There were
6,000 attempted fits for each score-sensitivity choice.

The expected-sensitivity score is the exact construction used by the small
`eigen.R` demonstration. All 6,000 replications were usable. The
observed-sensitivity score retains the expected-Fisher metric but substitutes
the observed sensitivity in the projected score construction; 63 of its 6,000
replications failed because the tangent information was not positive definite,
all in nonnormal MAR cells.

## Expected-sensitivity score

| Generator | Missingness | LRT/LRT | LRT/score | Score/LRT | Score/score |
|---|---:|---:|---:|---:|---:|
| Normal | Complete | .063 | .053 | .056 | .046 |
| Normal | MAR | .087 | .067 | .063 | .054 |
| VM2 | Complete | .144 | .048 | .115 | .040 |
| VM2 | MAR | .241 | .182 | .118 | .049 |
| IG2 | Complete | .097 | .054 | .092 | .055 |
| IG2 | MAR | .162 | .187 | .068 | .065 |

Across these six cells, mean absolute deviation of rejection from .05 was:

| Statistic/spectrum | Mean absolute size error |
|---|---:|
| LRT/LRT | .0823 |
| LRT/score | .0492 |
| Score/LRT | .0353 |
| Score/score | .0065 |

The score statistic accounts for slightly more of the aggregate improvement
than the score spectrum, but the decomposition is not uniform. The spectrum is
the larger change in the complete VM2 cell; the statistic is the larger change
under IG2 MAR; both are needed under VM2 MAR. The native score pairing is much
better calibrated than either one-component substitution.

## Observed-sensitivity score

| Generator | Missingness | LRT/LRT | LRT/score | Score/LRT | Score/score |
|---|---:|---:|---:|---:|---:|
| Normal | Complete | .063 | .088 | .007 | .014 |
| Normal | MAR | .087 | .121 | .002 | .026 |
| VM2 | Complete | .144 | .336 | .003 | .057 |
| VM2 | MAR | .236 | .551 | .002 | .096 |
| IG2 | Complete | .097 | .164 | .015 | .064 |
| IG2 | MAR | .153 | .373 | .002 | .066 |

The nonnormal-MAR denominators were 956 for VM2 and 981 for IG2 because of the
63 tangent-information failures. Mean absolute size errors over the six cells
were .0800, .2222, .0448, and .0238 for LRT/LRT, LRT/score, Score/LRT, and
Score/score respectively.

Here the observed-sensitivity score statistic is much smaller than the LRT,
while its mean spectrum is also smaller. Using its spectrum with the LRT is
severely liberal; using its statistic with the LRT spectrum is severely
conservative. The native pairing partly balances these changes. This is a
reason to analyze the statistic and spectrum jointly rather than attributing
the apparent performance of this score construction to either component in
isolation.

## Provisional conclusion

For the expected-Fisher score in the motivating example, the answer is “both,”
with a modest aggregate advantage for the base-statistic change and important
cell-specific reversals. For the observed-sensitivity score, cross-pairing
reveals much stronger co-movement: the statistic and spectrum should not be
interpreted as separable corrections. These results are a one-model pilot and
must be repeated after the paper's model and MAR panels are frozen.
