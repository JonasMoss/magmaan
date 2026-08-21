# Iteration 07: preliminary story and matched-power continuation

Date: 2026-08-21

## Preliminary null story

The paired `n=200`, 500-replication panel contains 75 cells per estimator. Its
primary calibration region is now restricted to 55 cells: all complete and
MCAR cells plus normal-data MAR. The 20 nonnormal-MAR cells remain an estimand
stress because Gaussian FIML can target pseudo-true moments outside the
generating SEM even when the generating covariance satisfies the fitted SEM.

Within the 55-cell FIML region, score pEBA(4) has mean absolute size error
.0127 and range .014--.074. The completed panel's expected-sensitivity
Rademacher multiplier has error .0139 and range .024--.094, and score SB has
error .0145 and range .034--.092. The pEBA result
does not dominate cell by cell; its advantage is containment of the larger
errors.

The score/LR interaction is the main methodological result. Score SB and
pEBA(4) are competitive, whereas score SS and `all` are conservative. The LR
ordering reverses: SS and `all` are the viable variants, while LR SB and
pEBA(4) are liberal. ML2S does not improve the useful methods and is retained as
a secondary negative result rather than expanded into the power phase.

## Fair updated geometry

The completed score-FMG panel uses the same expected-sensitivity projected
score as the legacy Rademacher multiplier. Current magmaan also exposes the
realized observed-data sensitivity needed for a genuinely defined pseudo-true
FIML nuisance projection. A 100-rep normal-complete audit showed that replacing
the projection by its realized version can be extremely conservative at
`n=200`, especially for the 42- and 87-df models. It therefore does not replace
the expected-sensitivity score rows in the primary identified-null comparison.

The continuation instead records a three-corner multiplier ablation:
expected-sensitivity Rademacher, expected-sensitivity Mammen, and
observed-sensitivity Mammen. Score SB, SS, pEBA(4), and `all` are saved under
both expected and observed projection geometry. This isolates the multiplier
weight law from the sensitivity change. It also does not restore a
population-SEM null under nonnormal MAR; those cells remain estimand stress.

## Power design

Each model receives two prespecified covariance misspecifications:

- `sparse`: one omitted residual covariance, chosen so it cannot be absorbed by
  the fitted loading pattern; and
- `diffuse`: an omitted method factor on a small set of indicators or waves
  spanning the model's substantive blocks.

For each model and alternative, a deterministic population-moment fit tunes the
perturbation to 50% asymptotic power for the ordinary normal-theory LR test at
`n=200`. If `df` is the model degrees of freedom, the target noncentrality
solves

`P(chi-square_df(lambda) > chi-square_df,.95) = .50`,

and the population ML discrepancy is set to `F0=lambda/(n-1)`. This reference
does not privilege any robust method. The calibrated residual-correlation or
method-factor variance shifts range from about .068 to .239 across the five
models and produce the requested normal-theory power to numerical tolerance.
A 100-rep normal-complete audit put ordinary-LR rejection between .44 and .60
for every model/alternative combination, centered near the intended .50.

Null, sparse, and diffuse cells share random-number seeds. Primary power uses
method-, model-, generator-, and missingness-specific empirical-null fifth
percentiles from the matched null replications. Raw nominal-threshold power is
retained but is not the ranking criterion.

## Run profiles

- `focus`: FIML only, 55 identified-null mechanisms times null/sparse/diffuse,
  1,000 replications, and 999 multiplier draws: 165 Modal shards.
- `stress`: FIML only, 20 nonnormal-MAR mechanisms times the three truths,
  500 replications and 999 draws: 60 separately labeled shards.
- ML2S power is deferred. The existing paired null panel is enough for the
  secondary two-stage conclusion unless a separate two-stage question emerges.

The runner is resumable, writes corrected and legacy score/multiplier fields,
uses `p <= .05` as the nominal convention while retaining `p < .05`, and emits
matched-null power summaries. A local three-shard null/sparse/diffuse smoke and
the Modal combiner completed without failure.
