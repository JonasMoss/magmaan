# Iteration 1: broad direct-FIML null calibration

## Decision

The experiment moved from the 36-cell probe in experiment 66 to a null-only
stress atlas before doing more power work. The screen crosses four generators,
four sample sizes, five missingness conditions, and restriction ranks 1/3/5.
It uses one configural fit per generated/masked sample and reuses that fit for
all selected ranks. An independent 11-cell confirmation then spends the larger
2,000-replication/999-flip budget only on predefined difficult corners.

This is not a replication of the FIML--FMG paper. It borrows its two-group,
six-indicator population and severe VM/IG/PL targets because those DGPs are
published, familiar, and already supported. It deliberately omits Student-t
and the unbiased-Gamma variants.

## Design details

- Group-1 n is 50, 100, 200, or 400; group 2 is 70% as large.
- H1 is configural. H0 equates the first 1, 3, or 5 non-marker loadings.
- Missingness is complete, 15%/30% Savalei--Bentler MCAR, calibrated 30%
  Savalei--Bentler MAR, or stronger item-specific logistic MAR.
- The stronger MAR leaves x1/x2 observed and calibrates four different logistic
  selection surfaces for x3:x6 to 30% missingness among eligible entries.
- Normal, VM, IG, and PL generators share the null population covariance and
  means. VM/IG/PL target marginal skewness 3 and excess kurtosis 21.
- Broad screen: 240 cells x 500 attempts, 199 signs, seed 20260714.
- Confirmation: 11 cells x 2,000 attempts, 999 signs, independent seed
  20261714.
- Rejection is `p <= .05`; strict `< .05` is retained because the flip p-values
  are discrete.

## Main results

The broad screen produced 120,000 rank-specific test attempts from 40,000 base
replications. Equal-cell mean rejection rates were:

| method | rejection |
|---|---:|
| basic flip | 0.0128 |
| nuisance-effective flip | 0.0693 |
| standardized flip | 0.0678 |
| score SB | 0.0526 |
| score pEBA4 | 0.0496 |
| score exact mixture | 0.0416 |
| direct score sandwich | 0.0491 |
| nested LR SB | 0.1135 |
| nested LR pEBA4 | 0.1102 |
| nested LR exact mixture | 0.1004 |

Score pEBA4 placed 97.9% of its 240 cells in the 0.025--0.075 band. Effective
and standardized flips placed 66.7% and 70.4% there. The flip problem is
localized rather than universal: normal means were 0.0558/0.0549, versus
0.0749/0.0730 under VM and 0.0760/0.0744 under PL. Small n and higher rank are
the clearest effect modifiers. Strong MAR was not uniformly the worst
calibration condition; it often reduced rejection relative to complete or
paper-MAR data.

The independent hard-cell confirmation strengthened rather than erased the
screen result. Across its 11 cells, equal-cell rejection was 0.0808 effective,
0.0778 standardized, 0.0473 score pEBA4, 0.0542 score SB, and 0.1146 nested LR
pEBA4. The VM and PL n1=50, 30%-MCAR effective flips rejected 0.123 and 0.115;
standardization reduced these only to 0.116 and 0.109. The VM n1=50 score
pEBA4 cell was mildly liberal at 0.0667, but the score approximation was much
more stable than either flip.

## When standardization matters

The effect is in the hypothesized direction but is too small at ranks 1--5.
Across 119,386 paired screen results, effective and standardized decisions
differed 0.145% of the time. By rank this was 0.040%, 0.178%, and 0.216% for
ranks 1, 3, and 5. The independent hard-cell confirmation raised disagreement
to 0.301% overall and at most 0.719% in a cell. Median flip-covariance
displacement was 0.0050 in the screen and 0.0091 in the confirmation.

The working hypothesis survives in a narrower form: standardization becomes
more relevant with more tested directions and more heterogeneous per-case
information, but the relevant signal is visible covariance displacement or
leverage, not missingness or rank by itself. At df <= 5, it is a useful
diagnostic rather than a default.

## Timing

The 12-worker broad screen took 856.9 seconds (14m17s), max RSS about 654 MiB.
Reusing H1 avoided 80,000 configural fits. Median cellwise phase times were
18.0 ms for the two fits needed by one rank, 1.8 ms for the effective sign
sums, 3.8 ms for flip-specific standardization, and 5.0 ms for the nested FMG
battery. Median base-replication time increased from 0.112 s at n1=50 to 0.167
s at n1=400 and from 0.088 s complete to 0.178 s under strong MAR.

The 12-worker confirmation took 647.8 seconds (10m48s), max RSS about 219 MiB.
At 999 signs, the median effective sign-sum phase was 12.1 ms but the
standardization phase was 94.8 ms, versus 32.0 ms for two fits at one rank.
Thus high-resolution standardization becomes the dominant cost while changing
very few decisions.

## Availability

In the screen, flip/effective/standardized p-values were available for 99.49%
of attempts and nested FMG p-values for 98.85%. The 1,281 flagged rows contain
593 fit failures, 21 post-fit flip failures, and 685 post-fit nested failures
(some post-fit failures overlap). Losses concentrate at n1=50 with 30% MCAR or
strong MAR; complete data were fully available. In the confirmation, flip and
nested availability were 98.17% and 96.10%.

These are method outcomes, not rows to discard silently. Calibration is
conditional on a finite p-value and must always be read beside availability.

## Why score SB looks suspiciously good

The good score-SB size does not validate a scalar chi-square law. At rank one
there is only one eigenvalue, so SB and the spectral alternatives coincide. In
the screen, the SB and pEBA4 p-values differed by 0.0048 on average at rank 3
and 0.0063 at rank 5, changing 0.35% and 0.54% of decisions. Their mean score
spectrum coefficients of variation were 0.57 and 0.72. In the hard-cell
confirmation, rank-5 eigenvalue CV averaged 0.88 and SB versus pEBA4 decision
disagreement rose to 0.80%.

The defensible interpretation is limited sensitivity plus error cancellation:
the trace scaling is often close enough around the 5% boundary in this DGP,
while averaging over rank-1 cells helps its aggregate. It remains a misspecified
scalar reference whenever the eigenvalues differ. Score pEBA4 is the stronger
empirical choice here because it retains the spectrum and remains calibrated
across the factor slices.

## Next probe

Do not expand the null grid again immediately. The next useful simulation is a
matched-null power comparison in a much smaller set of calibrated cells,
retaining score pEBA4, effective flip, standardized flip as a diagnostic, and
possibly score SB as the familiar scalar comparator. Power must use
method/cell-specific null thresholds; raw nominal power would reward the
liberal flips and nested tests.

For method development, the unresolved question is a small-sample correction
to the nuisance-effective flip itself under severe nonnormality. More
flip-specific covariance standardization is not the answer in this rank range.
Near-singular nuisance information remains a separate core-hardening target.
