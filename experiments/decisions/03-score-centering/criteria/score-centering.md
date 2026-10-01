# Likelihood-score covariance centering

Preregistered 2026-10-01, before the study's smoke, pilot or confirmation draws.

## Decision and scope

Can the ordinary policy retain uncentered likelihood-score second moments for
parameter covariance and analytic score calibration, or does a specific sampling
regime require empirical centering? ML is the current-policy lane. Direct FIML
is a prospective lane: the ordinary package does not yet supply its inference.
Neither lane changes optimizer, information geometry, observed test numerator,
normalization, or estimation between covariance arms.

Established moment Gamma/NACOV, WLS/DWLS/DLS, LR reference construction,
noniterative delta maps and multiplier promotion remain banked. This study does
not validate penalties, PSD boundary inference, ordinal inference, arbitrary
nested misspecification or nonstationary terminal points. Those parts of the
active task cannot be closed by this grid.

## Baseline and candidates

Fits use `magmaan()` with the library's solver/start defaults, selecting only the
model, estimator, identification, means and grouping required by the population.
The judge is `isTRUE(as_lab_fit(fit)$converged)`; no study convergence tolerance.
The ML baseline is the ordinary package's actual policy. Its score statistic,
SB/PEBA4 p-values and parameter covariance must match the independently composed
native primitives before candidate comparisons are interpreted.

- `raw`: cross-products of the original likelihood-score rows.
- `global`: subtract their empirical column means before cross-products.
- `group`: subtract means within each fixed sampling group; evaluated only in
  the grouped family. Missingness patterns are never treated as sampling groups.

Parameter covariance uses the same observed-information bread for all arms.
ML tests use the policy's expected sensitivity and metric. FIML reports separate
expected-sensitivity (actual primitive default) and observed-sensitivity
(prospective policy) strata, both with the expected metric. No comparisons pool
those geometries. All score projection, covariance solves and calibration use
native magmaan primitives. Study-local row centering defines the candidates.

## Problems and independent checks

Held-out confirmation families, independently specified in this study:

1. One-factor six-indicator global ML null/alternative: normal and standardized
   chi-square(3) innovations; N=80/300, alternative residual correlation .18.
2. Loading-equality nested ML null/alternative: the same innovation families;
   loadings .75/.75 under the null and .75/.90 under the alternative.
3. Fixed 1:3 groups, two observed independent unit-variance variables, shared
   means, and fixed covariance identity. Group X means are target+.6 and
   target-.2; the shared approximation mean is target (0/.15). Y mean is .3.
   Test the shared X mean against zero while Y remains a nuisance. The larger
   model is misspecified even under the restriction null. The true variance of
   its shared X estimate is exactly 1/N, independent of the between-group means.
4. Normal global and loading-equality FIML controls, MCAR and MAR, N=80/300.
   X1 always observed; X2/X5 missing probabilities .25/.30 under MCAR, or
   logistic(-.9+.8*X1)/logistic(-1+.6*X1) under MAR.

The CFA loadings are (.65,.75,.75,.60,.80,.70), latent variance 1, and independent
errors have variance 1-loading^2. Under the global alternative only errors 1/4
are correlated; loading coverage there has no asserted pseudo-true target.
The nested alternative changes loading 3 to .90, keeping the larger model exact.
Coverage is evaluated at the larger-model fit for nested families.

Development checks use distinct draws and are not gating: row sums versus the
observed score, raw minus centered Gram = N times the score-mean outer product,
PSD Gram checks, projected primitive versus ML policy agreement, stationary
parameter-score equivalence, and the analytic grouped variance counterexample.
Scores of fitted parameters and released test directions are assessed separately.

## Runs and provenance

- Smoke: one draw per cell, seed base 202610001; mechanics only.
- Pilot: default 100 draws per cell, seed base 302610011; estimate runtime,
  failures and effect sizes. Pilot rates cannot select a default.
- Confirmation: at least 2,000 independent draws per cell, seed base 402610031;
  no reused smoke/pilot draws. Label runs `confirmation` explicitly.

Replicate seeds are seed_base + 10000*stable_cell_id + replicate_id; identifiers
are assigned before any cell filter. Criteria, runner and installed native-binary
hashes and package paths/versions are recorded. Confirmation populations and
seeds are held out from candidate development. A defect found on a draw makes
it development data; a corrected confirmation requires a new recorded seed base.
All errors/nonconvergence/unavailable inference survive in raw rows and summary
counts. Reports give each family/N/geometry separately, never pooled rates.

## Acceptance rules and implied outcomes

These rules apply only to confirmation runs; pilot decisions read `open`.

1. At least 1,800 usable pairs per cell; no increase in unavailable inference
   versus raw. Any failed algebra/policy identity prevents a decision.
2. For each null family/N/geometry and each SB/PEBA4 method, the candidate's
   95% Wilson interval for rejection must lie within [.03,.07]. The candidate
   must not worsen absolute distance from .05 by over .01, assessed by a paired
   bootstrap 95% interval for the difference in absolute errors.
3. For every model with a known parameter target, the candidate's 95% Wilson
   coverage interval must lie within [.92,.98]. In the grouped family at N=300,
   its mean reported variance divided by the exact 1/N must be in [.98,1.02].
4. Compare power at matched null size using each arm's empirical null fifth
   percentile of p-values, never nominal power alone. Resample paired null and
   alternative draws together in 400 bootstrap iterations, recalculating the
   thresholds, to report uncertainty. No matched-power loss over .02 is allowed
   (lower 95% bound above -.02) in any family/N/geometry/method.
5. Changing a default needs an identifiable benefit: a demonstrated sampling-
   covariance defect, or a paired 95% interval showing at least .01 reduction
   in absolute null-size error in one cell, without violating the other rules.
   Otherwise retain raw provisionally; inconclusive precision stays open.

The grouped counterexample can refute a universal never-center claim by its
first-principles covariance identity even before Monte Carlo confirmation.
It cannot select centering for single-group CFA or FIML score references.
Eligibility is reported separately for parameter covariance and each test lane;
no winner across methods is inferred by pooling. Failure of both arms' calibration
leaves the inference policy open rather than selecting the less bad arm.
Any adopted implementation requires separate authorized core/binding work and
regression gates; running this experiment does not change package defaults.
