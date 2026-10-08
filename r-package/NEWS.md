# magmaanlab (development version)

- Identification reports classify scale, location and rotation freedoms, enforce
  fixed entries and linear equalities, and expose per-direction suggested fixes.
  Remaining null dimensions are reported as information deficits.

- `augment_model_spec(model, categories)` augments ordinal specifications from
  declared category counts with missing threshold starts. Ordinal and mixed
  augmentation now builds rows column by column; prepared models use this helper.

- The `lavaan-0.7.2` FIML preset now uses lavaan's scale-dependent H1
  covariance ridge and retains stalled EM moments. `fit$fitting$h1` records
  H1 convergence and update/repair counts. Converged H1 moments retain strict
  oracle agreement; stalled H1 endpoints are outside the compatibility
  contract. Native FIML H1 defaults are unchanged.

- `structural_identification()` checks a specification or reuses a prepared
  model's data-free report. Prepared fits reuse construction-time identification
  and express cached null directions at the deciding probe point. Lab fits
  remain available for unidentified models, with a failed verdict.

- The `lavaan-0.7.2` preset now adapts weak markers for complete-data ML.
  `marker = "default"` keeps the requested identification. Fits retain the
  actual fitted specification and a switch/revert table; nested inference
  explicitly refuses switched marker coordinates. FIML and DWLS adaptation
  follows separately; their preset currently reports `marker = "default"`.

- Fits using `convergence = "lavaan-0.7.2"` (including the preset) report
  lavaan's post-estimation check separately from convergence. The six flags
  are available in `fitting$post_check` on the lab fit; ordinary fits expose
  them through `as_lab_fit()`. This adds no warning or inference restriction.

- Continuous ULS, DWLS and WLS/ADF fits now warn when within-group observed
  variances differ by more than a factor of 1000, with advice to rescale.
  The C++ diagnostic also covers ML2S Stage-2 and continuous mixed-LS moments;
  ML, GLS, FIML, NT/DLS weights and ordinal-only fits do not emit this advice.
  Lab fits expose `observed_variance_ratio` and `numerical_scaling_message`.
- A structurally unidentified model is never `converged = TRUE`, under any
  rule: a data-free rank check of the moment Jacobian
  (`fit$diagnostics$identification`) fails the verdict and names the null
  directions. A local Newton pass no longer certifies a scale ridge.

# magmaanlab 0.2.0

## Lab API and migration

- U-factor builders require explicit bread; `score_quadratic()` requires an
  explicit meat matrix. Observed bread supports pseudo-true covariance and
  nested restrictions; complete-data global GOF can use expected geometry.
- `frontier_profile_lrt_parameter_gmm()` and
  `frontier_profile_lrt_ci_parameter_gmm()` remove `ij_weight` and `dls_a`:
  the weight recipe comes from the fit.
- Estimated-weight inference follows recorded moment and Stage-2 weight
  recipes, including continuous DWLS and DLS mixing weights. Supplied-W and
  ordinal NT/DLS fits without a supported recipe have typed refusals;
  explicit fixed-weight inference remains available.
- WLS procedures use recorded `fit$W`; optional `weight` must match it or a
  positive multiple. ML2S RBM and case influence likewise use recorded Stage-2
  weights and DLS mixing weights, and validate explicit overrides.

## Inference policy and evidence

- `policy_inference()` and `policy_nested()` compose ML/FIML empirical-score,
  observed-bread inference, with observed score sensitivity and an expected
  metric. Nested LR calibration handles restricted, misspecified means.
  Components retain separate typed unavailable reasons.
- All-ordinal and complete mixed DWLS policy composers use exact empirical
  first-stage influence, estimated-weight IJ covariance and global/nested
  fit-function tests. Nested geometry uses the observed Hessian and supports
  parameter restrictions and moment-nested Wu–Estabrook threshold invariance.
  Nested DWLS score tests remain unavailable.
- Policy defaults are PEBA4 for ML/FIML and All for DWLS global/nested tests.
  SB/PEBA4 DWLS comparators remain available. Exact-first-stage and reference-law
  studies support the DWLS choice across ordinal, mixed, threshold-invariance
  and textbook designs; finite-sample size varies across settings.
- `robust_ordinal_ij(..., first_stage = "exact")` exposes exact empirical
  sampling rows and Gamma. Its default remains `"opg"`; the ordinary policy
  selects exact influence, without changing fitting weights.

## Test references

- `calibrate_quadratic()` accepts std/sb/ss/mv/scaled_f/all/pall and eba<k>/peba<k>.
  Block counts at or above df give singleton blocks. Policy ML/FIML results
  label `reference = "peba4"` and retain `p_sb`/`sb_scale` comparators.

## Modification indices

- `policy_modification_indices()` composes robust one-df indices and affine
  equality releases for ML/FIML and all-ordinal/mixed DWLS, including absent
  regressions among variables already in equations. `policy_mi_refit()` refits
  a candidate from its embedded null using the nested policy.
- ML/FIML robust MI and releases use observed nuisance sensitivity with an
  expected-information metric and EPCs. Negative observed release curvature
  alone no longer removes candidates; score tables retain `reason` and `detail`
  for numerical failures.
- Robust LS indices and releases support observed sensitivity and estimated
  weights for continuous ULS/GLS/DWLS/WLS/DLS and ordinal ULS/DWLS/WLS.
  Complete mixed DWLS/WLS adds exact first-stage estimated-weight influence.
  Expected sensitivity and fixed weights remain explicit comparators.
- Validated caller `gamma` or per-group NACOV blocks support complete ML,
  continuous LS and categorical LS indices/releases. Supplied Gamma requires
  explicit fixed-weight inference; it cannot provide casewise weight influence.
  Continuous-LS covariance and ML/LS profile-LRT adapters also accept Gamma.
- ML2S indices and score tests support every Stage-2 weight. `mi` is the naive
  Stage-2 statistic (`mi_type = "naive_stage2"`); `mi.scaled` uses Stage-1
  covariance and, when requested, Stage-2 weight influence.

## Standardized estimates

- `standardized_rows(fit, vcov)` provides std.lv/std.all estimates, delta-method
  SEs and endogenous R-squared, including fixed markers, groups and ordinal
  parameters. Missing covariance preserves estimates with typed missing SEs;
  defined standardized scales are unsupported.

## Fit measures

- `policy_fit_measures()` provides robust point indices and per-index reasons
  with discrepancy, trace and df details for ML/FIML, continuous ULS, ML2S-NT
  and ordinal/mixed DWLS. It fits a matching independence baseline and corrects
  residual points for sampling bias. Policy intervals are not yet provided.
- Ordinal/mixed misspecification fit-measure wrappers accept exact first-stage
  influence; OPG remains their default and generalized-df TLI is retained.
- Lavaan-compatible fit measures include standard, scaled and robust families,
  intervals and close-fit p-values. FIML ML/MLR uses tighter H1 EM precision
  and is validated against lavaan.

## Lavaan compatibility

- The lavaan 0.7.2 fitting preset supports continuous ML/FIML and all-ordinal
  or complete mixed DWLS, including delta/theta and grouped equal loadings,
  unrestricted covariance and supported linear equalities. Prepared `estimate()`
  accepts DWLS fitting options through the same engine as `fit_model()`.
- `convention_inference()` and `convention_nested()` compose named bundles on
  retained fits with component-level unavailable reasons. Fitting presets and
  reporting bundles remain separate choices. Complete-data MLR uses the
  H1-minus-H0 trace; ordinal reporting uses per-group n minus one.
- Complete mixed DWLS supports WLSMV covariance, Wald intervals, scaled-shifted
  global and Satorra-2000 nested tests. Missing mixed observations, nonlinear
  constraints and finite bounds remain outside the mixed fitting preset.

## Lab building blocks

- Association-ML covariance, global/nested tests, modification indices and
  linear releases use exact empirical Stage-1 influence and observed sensitivity.
  Tests expose All/SB/PEBA4; MI uses robust one-df laws. Ordinary association
  inference remains gated by policy calibration. Missing data, mixed indicators,
  nonlinear constraints, covariance boundaries and penalties are unsupported;
  moment nesting is refused.
- ML robust SE and score/MI meats default to exact casewise likelihood
  projections, including restricted means and group score means.
  `moments = "auto"` selects this recipe; structured/unstructured comparators
  remain explicit. Global GOF primitives retain their named conventions.
- `refit_from_null()` returns a new larger-model fit from a verified embedded
  restricted estimate, with native convergence diagnostics.
- Robust procedures with estimated-weight channels default to estimated-weight
  inference; MI selects that default by estimator so ML/FIML need no explicit
  FALSE. Explicit FALSE retains the fixed-weight comparator.
- Scalar profile tests and intervals default to the misspecification-scaled
  reference. Empirical laws require raw data; ordinary and robust-scaled
  comparators remain selectable.
- Global score flips use observed sensitivity for MAR FIML and expected
  sensitivity for complete-data correct-model GOF.
- Reliability with NULL Gamma derives empirical Gamma from raw data and errors
  without them. `parameter_covariance()` derives empirical meat from retained
  observations; `meat = "model"` requests inverse information.
- `robust_nested_lrt()` composes ML/DWLS policy tests and refuses unsupported
  routes. Explicit methods and Satorra aliases retain compatibility conventions.
- `vcov(regime = "sandwich_ij")` provides all-ordinal estimated-weight
  covariance; the default remains fixed-weight sandwich covariance.
- SAM, case-influence and FMG/semTests routes retain diagnostic/comparator
  conventions. `infer_continuous_ls_robust()` requires explicit
  `fixed_weight = TRUE` for recorded data-estimated weights.

## Mplus frontend

- `mplus_model()` imports whole continuous single/grouped inputs, growth models,
  MODEL CONSTRAINT with NEW/equalities/DO loops, and MODEL INDIRECT. Specs retain
  source notes, numeric group order, labels and rebuildable model metadata.
  SQRT, PHI/pnorm and LOG10 extend expressions; inequalities remain unsupported.
- Binary/ordinal outcomes support threshold ranges/labels, DELTA scales, THETA
  residuals and grouped CONFIGURAL/SCALAR models. All-ordinal DWLS uses Mplus
  defaults; mixed, conditional and categorical ML routes report scope refusals.
- Complete observed-X moment mentions select joint random-X models. Partial
  mentions give the required completion statement. Group-specific role changes
  are refused, and unlisted group codes require filtering.
- `mplus_data()` reads free/fixed individual and summary data, missing flags and
  FILE/NGROUPS groups. It reports observation counts and dropped grouping codes;
  summary N-1 covariances are rescaled to N for ML. Terminal DOS Ctrl-Z markers
  are accepted; embedded markers give a DA01 diagnostic.
- Unsupported Mplus inputs report specific scope reasons and required edits.
  Auxiliary NEW coordinates are supported in barrier gradients and moment
  curvature; noniterative CFA and profiled SNLLS refuse them explicitly.

## Fixes

- Equality-reduced ordinal parameter counts now give lavaan-compatible df for
  grouped threshold/loading and label constraints.
- Unequal-group all-ordinal WLSMV covariance now preserves group-weighted
  sandwich geometry and the reporting denominator.
- Exact ordinal/mixed first-stage influence and Gamma-diagonal movement use
  affected score blocks, category counts and sparse derivatives, reducing cost.
- Mixed MI now uses the equality-release criterion scale, removing an extra
  factor two without changing EPCs or fitting weights.

# magmaanlab 0.1.0

First versioned dependency for the magmaan 0.1.0 simulation prerelease. The lab
retains its methods-development API, including the EQS constructor.

Development is unfinished: remaining bugs and incomplete coverage of the main
estimators and inferential procedures remain to be resolved and validated.
Version 0.1.0 provides a versioned simulation snapshot, not a finished release.

High-level fit reconstruction retains defined parameter estimates using the
shared C++ evaluator, independently of covariance inference.

- Generic lab covariance and nested score sensitivity now default to observed
  information with empirical meat; categorical and GLS/WLS covariance uses
  estimated-weight IJ. Explicit expected/fixed-weight choices remain comparators.
- FIML fit measures request robust inference by default. Noniterative NT
  defaults remain documented comparators in the postponed development area.
