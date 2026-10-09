# Criteria: ordinary fit indices (lane fit_indices)

**Status: proposed 2026-10-09 by the planner for the user's review. It becomes
binding when the user approves it, before the production run; any later change
is dated and marked as written before or after seeing results.**

Design: [fit-index study](../../../../project/design/fit-index-evaluation-study.md)
under the [simulation study principles](../../../../project/design/simulation-study-principles.md).

## Decision

For the ordinary `magmaan` package, separately for each index (RMSEA, CFI,
TLI, SRMR) and estimator (ML on complete data; FIML under MAR):

1. **Point estimator.** Keep the misspecification-robust correction (the
   default adopted 2026-10-06) or switch to the naive or the lavaan-robust
   correction.
2. **Interval.** Keep reporting no interval (the current default) or report one
   90% interval method.

DWLS and ordinal data are not covered; they get their own lane.

## Problems

- **Models:** the twelve tier-1 basis models
  (`experiments/_support/basis/sem_score`), each fitted as its `h1`. The data
  come from `h1` plus the natural misspecification in
  `misspecification.csv` (single parameter or block, with its sign) at
  standardized size delta in {0, .1, .2, .4}; delta 0 is correct
  specification.
- **Sample sizes:** N in {200, 500, 1000, 4000}.
- **Generators:** normal; Vale-Maurelli with skewness 1 and excess kurtosis 2;
  the independent generator with the same marginal moments; discretized scores
  (five equal-probability categories, integer scores, treated as continuous).
- **Estimator and missingness:** ML on complete data; FIML with single-cause
  MAR (cause: the first observed variable; targets: the last half of the
  others) at 15% and 30%.
- **Cells:** 12 x 4 x 4 x 4 x 3 = 2,304 (768 ML, 1,536 FIML).
  **Replications:** R = 1000 per cell. **Seed base:** 2026101001, distinct
  from the smoke and pilot seed bases. Pricing (2026-10-09, seconds per draw at
  N = 4000): ML 0.5-6, FIML 1.2-99 (cfa_long30); about 90 CPU-hours for the ML
  cells and about 1,950 for the FIML cells at R = 1000. The ML cells may run
  first; the FIML cells run after a performance repair, under these same rules.

All models are held-out test problems; the pricing pilot's draws (other seed
bases) are development data and are not reported as evidence.

## Targets

- **ML:** exact population values from the generating moments: F0, the
  independence-baseline F0b, RMSEA0 = sqrt(G F0 / df), CFI0, TLI0 (nominal df)
  and SRMR0, as defined in `project/design/fit-indices.md`. Discretized cells
  use the integer scores' own moments.
- **FIML under MAR:** the estimator's own target, the missingness-dependent
  population discrepancy, from one fixed N = 200000 draw fitted with tightened
  stopping controls (recorded) and required to pass `fit$converged`. The
  complete-data targets are reported beside it, descriptively.

## Arms

- **Points:** naive, lavaan-robust and misspecification-robust (default).
- **Intervals (90%):** lavaan's naive and scaled noncentral chi-square
  intervals, and the misspecification-robust Wald intervals on the discrepancy
  scale (`wald_f`) and the square-root scale (`wald_sqrt`).

Every draw fits `h1` with the ordinary package's defaults. Unavailable arms keep
their reason and are never dropped.

## Judge and rules

Draws whose fit fails `fit$converged` stay in the attempted denominators:
availability is available over attempted draws; point and coverage summaries
use converged draws with an available value. A family is one model, estimator
setting and index; results are reported per cell and per family, never pooled.

**Points**, per index and estimator setting. In a cell, a candidate beats the
default if its RMSE against the cell's own target is at most 0.9 times the
default's, and loses if the default's RMSE is at most 0.9 times its own;
otherwise they tie. A candidate is eligible if, in every family, its losses do
not exceed its wins, and over all cells its wins exceed its losses by at least
10% of the cells. If several are eligible, the one with the most wins becomes
the default for that index and estimator setting; if none is, the default
stays. Every loss is listed.

**Intervals**, per index and estimator setting. A cell is flagged if coverage
lies outside 87.5-92.5%. An arm is eligible if (i) its availability is at least
95% in every cell, (ii) at most 5% of its cells are flagged, and (iii) no cell
has coverage below 80%. Among eligible arms the one with the smallest median
width becomes the reported interval; if none is eligible, the ordinary package
keeps reporting no interval for that index and estimator setting. Boundary
conventions: the RMSEA lower endpoint is clipped at 0, and a population RMSEA of
0 counts as covered when the lower endpoint is 0; CFI is clipped to [0, 1] and a
population CFI of 1 counts as covered when the upper endpoint is 1; TLI follows
`project/design/fit-indices.md`.

## Outcome

The report opens with the register (each covered default, its value, status and
evidence). Changes the rules imply go to implementation cards for the ordinary
package; adoption follows the user's review of the report.
