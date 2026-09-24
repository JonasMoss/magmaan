# Iteration 01: fixed-rank calibration frontier

## Why this experiment exists

Experiment 63 made two things unusually clear. Score pEBA4 looked good but is a
penalty choice; the nuisance-effective flip was nearly as well calibrated
without that choice. Flip-specific standardization, however, almost never
changed a decision at n=400/group, even when weak-invariance rank reached 133,
and became very expensive at that rank. More restrictions alone are therefore
not the next useful axis.

The present design fixes `G = 8` and `df = 28`. Both p=5 and p=20 test only the
28 equality restrictions on `x2:x5`; the p=20 model leaves `x6:x20` free by
group. This isolates added nuisance estimation from added restriction rank.

## What would make standardization useful

The working mechanism is flip-specific conditional-variance displacement after
the tested score has already been orthogonalized against estimated nuisance
directions. It should be largest when individual observations or small groups
have high leverage on that nuisance fit. The predicted ordering is therefore:

- n=50 more than n=100, and both more than n=200/400;
- p=20 more than p=5 at fixed df=28;
- unequal allocation more than balanced allocation;
- alternating factor/residual information more than homogeneous information;
- severe nonnormality more than normality.

Evidence must be paired. A small difference in marginal rejection can conceal
offsetting decision changes. The runner saves direction-specific decision
changes, absolute p-value shifts, variance displacement, conditioning, and the
correlation between displacement and p-value movement.

This design cannot say whether standardization is more valuable for few versus
many restrictions because rank does not vary. If the nuisance hypothesis
survives, a later small rank probe can cross tested-item counts while holding
the surrounding model fixed.

## Comparator interpretation

Score SB is retained as a diagnostic, not treated as a correct reference law.
Its scalar mean correction replaces the estimated weighted chi-square spectrum
with equal eigenvalues. Good aggregate size can arise from shrinkage plus error
cancellation and is not evidence that the score statistic is truly scaled
chi-square. Score MV/SS, pEBA2/4/6, PALL/ALL, the direct sandwich, nested ML
variants, and two RLS variants expose the relevant spectrum/shrinkage choices.

Power is not ranked at the nominal threshold alone. Each method also receives a
matched empirical-null critical value in the same design corner. The primary
question is power after achieved-size differences are removed.

## Computation decisions

- Rejection uses `p <= .05`; `p < .05` is saved as sensitivity. At 199 random
  flips plus identity, the two conventions correspond to 10/200 and 9/200 grid
  points.
- Generator transformations are calibrated outside replication timing.
  Identical group covariances are deduplicated within a population, and the
  calibrated population is cached across n/allocation cells.
- Null and power pairs share underlying random-number seeds. Chunks are stable
  under interruption and resume only under an identical manifest.
- Modal has zero retries, a local dry-run default, and a timeout-envelope guard
  below $10. The account-side $10 budget remains the actual hard stop.

The homogeneous p=20 PL smoke is a useful scale check: eight redundant group
calibrations originally took about 133 seconds; deduplication reduced setup to
about 15 seconds, while the fit/flip/FMG replication took about 0.6 seconds.

## Gate after the run

Carry standardized flips forward only if they reduce cellwise calibration RMSE
or hard-corner liberalism, retain matched-null power, and cost a modest share of
the full replication. If their benefit is negligible or isolated to one
generator, retain the effective flip as the default research candidate and the
standardized version as a displacement diagnostic.

