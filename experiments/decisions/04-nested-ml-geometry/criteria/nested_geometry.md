# Nested ML geometry registration (2026-10-02)

Decision: retain or reconsider the adopted observed nested ML geometry. Baseline:
ordinary `magmaan()` defaults, including its convergence verdict; alternatives:
expected and observed geometry for score and LR, each calibrated by SB and PEBA4.
No library default is changed by this runner.

- Model: two-group CFA, 2 factors x 4 indicators, metric invariance test: H1 = configural (loadings free across groups), H0 = loadings equal (group.equal = 'loadings', latent means/intercepts as magmaan_model defaults). Restriction df = 6.
- Larger model correct: population = H1 model with equal loadings in both groups. Larger model misspecified: both groups share the same population with one omitted residual covariance (x1 ~~ x5, correlation 0.3) not in either model, so the pseudo-true loadings are equal across groups and H0 holds for the misspecified H1.
- Power cells: same, but one group's loading x3 is 0.15 lower (standardized).
- Distributions: normal; skewed via Vale-Maurelli (skew 2, excess kurtosis 7) applied to all indicators.
- N per group: 100 and 300. Roles: null, power. 2 x 2 x 2 x 2 = 16 cells. Reps: 2000 (null), 1000 (power).
- Per replicate: fit H1 and H0 with magmaanlab ML (as magmaan() would), then prepare_inference() for each fit, prepare_hypothesis(null, alternative), and for geometry in expected/observed and test in score/lr: inference_quadratic(hyp, test, geometry) -> calibrate_quadratic(q, c('sb','peba4')). Record statistic, df, p-values, convergence, failures, timing. Also record anova()'s policy p-values to confirm the policy equals the observed arm.
- Outcomes: null rejection at 5% per arm and cell with Wilson intervals; size-adjusted power (critical value from the matching null cell); failures.

Retain the observed geometry unless, in some misspecified-larger-model null cell, its PEBA4 or SB size error exceeds the expected geometry's by more than 1 percentage point with a paired 95% interval excluding zero. Report correct-model cells as controls; there both geometries are consistent. Size-adjusted power is reported, not gating.

Population details fixed before pilot: standardized loadings .7, .8, .75, .65
on each factor; factor variances 1, correlation .4, residual variances 1-loading²,
zero means. The omitted covariance is .3 times the two residual standard
deviations. Power changes x3 in group b to .60 and adjusts its residual variance
to keep variance 1. Lavaan's Vale-Maurelli generator preserves the population
covariance and imposes the requested marginal moments.

Smoke, pilot and production seed bases are 526100021, 626100031 and 726100041;
seed = base + 10000 * stable cell ID + replicate. Smoke/pilot are development
checks, never confirming evidence. Production requires the task-43 compute
decision. Correct-model cells are controls; misspecified null cells are held out.

Wilson intervals condition on available arms; failures are reported separately
against all draws. Paired comparisons use complete pairs and a paired percentile
bootstrap (10000 resamples, fixed independent summary seeds), of the difference
in absolute size errors. Report every cell and arm without pooling. Size-adjusted
power compares calibrated p-values with their matching null cell's empirical
5% quantile (type 1); the LR statistic is common to geometries, so p-value
thresholds retain the geometry-specific calibration. Insufficient pairs or null
values yield unavailable summaries. Pilot decision status always remains open.

Frozen pilot summary CSVs provide compute pricing and construction diagnostics;
raw replicate rows are ignored. Metadata records criteria/source/native hashes,
package versions and Git commit. Any amendment after results must be dated.

## Amendment 2026-10-03, after the pilot and before production

The larger-model factor gains a third level, strong. Both groups share an
omitted cross-loading of x5 on f1 of .4 (standardized population; adjust x5's
residual variance to keep its variance 1) in addition to the mild level's
residual correlation. Groups stay identical, so the loading-equality null
holds for the pseudo-true parameters. The decision rule applies to every
misspecified (mild or strong) null cell. 24 cells; production 2000 null / 1000
power replicates per cell. The original misspecified level is now named mild;
its population, stable cell IDs 9–16 and seeds are unchanged. Strong cells
receive IDs 17–24. The fresh pilot uses the registered pilot seed base.

Population diagnostics fit configural H1 to exact covariance matrices in both
groups with 10,000,000 observations per group and no covariance rescaling.
Population RMSEA uses the limiting ML discrepancy (without the finite-N df
subtraction); the largest absolute covariance residual is standardized by the
population marginal standard deviations. Geometry gaps use complete paired
replicates: median absolute PEBA4 p-value difference for each test, and median
absolute score-statistic difference divided by the expected score statistic.

## Amendment 2026-10-03 (user decision), before production

magmaan's inference policy is misspecification-robust by default (see the
interface vision, Inference policy). The observed nested geometry is therefore
fixed by principle, not selected by this study: expected information is not
consistent when the larger model is misspecified. This run is calibration
evidence for the adopted observed geometry, with the expected geometry as a
reported comparator. The registered rule is kept as a reporting criterion. If
observed fails it in some cell, the consequence is finite-sample correction
work on the observed recipe, not a return to expected information.

