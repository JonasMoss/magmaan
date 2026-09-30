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

## LRT implementation and increasing-N audit

A same-data oracle rerun compared magmaan with lavaan over 105 FIML/MLR fits
spanning complete, MCAR, MAR, and MNAR data and seven outcome laws. The maximum
absolute difference was `3.62e-6` for the unscaled LRT, `5.60e-3` for the
scaled MLR statistic, and `7.03e-5` for its p-value. There were no degrees-of-
freedom or 5% decision mismatches. This directly validates the FIML LRT base
statistic and scalar MLR path; lavaan does not provide an oracle for the FMG
full spectrum.

The expected-score crossing was then repeated at larger sample sizes. The
table gives native LRT/FMG pEBA(4) rejection. The `n = 120` and `n = 500`
columns use 1,000 and 500 replications per cell; `n = 2000` uses 250.

| Generator | Missingness | n = 120 | n = 500 | n = 2000 |
|---|---:|---:|---:|---:|
| Normal | Complete | .063 | .046 | .052 |
| Normal | MAR | .087 | .050 | .076 |
| VM2 | Complete | .144 | .084 | .068 |
| VM2 | MAR | .241 | .176 | .236 |
| IG2 | Complete | .097 | .052 | .044 |
| IG2 | MAR | .162 | .194 | .592 |

The complete-data and normal-MAR conditions move toward nominal calibration.
The nonnormal-MAR conditions do not. This is the signature of Gaussian-FIML
pseudo-true target drift rather than a generic observed-versus-expected
information error. Under nonnormal MAR, different missingness patterns can
have different selected conditional moments, so the common Gaussian mean and
covariance model fitted by FIML need not have the generating SEM moments as its
pseudo-true target. A robust spectrum changes the null fluctuation scale but
cannot remove a positive population discrepancy between the restricted and
saturated observed-data likelihoods.

The projected score is not immune to this drift at larger N. Its native
pEBA(4) rejection under VM2 MAR rose from .049 to .086 to .176, and under IG2
MAR from .065 to .176 to .608. Its attractive `n = 120` behavior in these
stress cells therefore must not be called Type-I calibration. Both methods are
eventually detecting observed-data pseudo-model failure.
