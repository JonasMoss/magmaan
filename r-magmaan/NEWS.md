# magmaan (development version)

- The `lavaan-0.7.2` marker component now also covers FIML and ordinal/mixed
  DWLS, using each route's saturated H1 covariance. Switched fits replay the
  selected fitting attempts and revert when rejected; `marker = "default"`
  keeps the requested markers.

- Fully fixed models with latent regressions or higher-order factors and fixed
  means now return their implied moments without crashing in the stationarity audit.

- Identification refusals now suggest specific scale, location and rotation
  restrictions. Information deficits name affected parameters without suggesting
  an automatic fix; no identifying constraints are added.

- The `lavaan-0.7.2` FIML preset now uses lavaan's scale-dependent H1
  covariance ridge and retains stalled EM moments. `fit$fitting$h1` records
  H1 convergence and update/repair counts. Converged H1 moments retain strict
  oracle agreement; stalled H1 endpoints are outside the compatibility
  contract. Native FIML H1 defaults are unchanged.

- `magmaan_model()` now caches and prints structural identification.
  `magmaan()` refuses unidentified models before fitting with
  `magmaan_identification_error`, naming the free parameter directions.
  Unchecked models fit normally; deliberate ridge fits remain in magmaanlab.

- The `lavaan-0.7.2` preset now adapts weak markers for complete-data ML.
  `marker = "default"` keeps the requested identification. Fits retain the
  actual fitted specification and a switch/revert table; nested inference
  explicitly refuses switched marker coordinates. FIML and DWLS adaptation
  follows separately; their preset currently reports `marker = "default"`.

- Fits using `convergence = "lavaan-0.7.2"` (including the preset) report
  lavaan's post-estimation check separately from convergence. The six flags
  are available in `fitting$post_check` on the lab fit; ordinary fits expose
  them through `as_lab_fit()`. This adds no warning or inference restriction.

- ULS fits and the continuous part of mixed DWLS/WLS fits now warn when
  within-group observed variances differ by more than a factor of 1000,
  with advice to rescale. ML, GLS, FIML and ordinal-only fits do not emit
  this advice. The diagnostic is computed by the shared C++ core.

# magmaan 0.2.0

## Ordinary API and migration

Calls written for 0.1.0 need changes. Removed arguments raise errors naming
their replacements.

- `magmaan_model()` constructs a reusable model with a frozen data schema.
  `prototype` declares groups and ordinal categories; factor levels allow a
  zero-row prototype. Schema violations raise `magmaan_schema_error`.
- `magmaan(model, data, estimator, covariance, inference, options)` fits that
  model. Syntax strings and lab specifications construct it from `data`;
  undeclared ordered factors are rejected. `inference = FALSE` estimates only,
  and `infer(fit)` adds inference later.
- Every model has a mean structure. Saturated intercepts add entries to
  `coef()` and `vcov()` without changing other estimates or inference.
- Observed covariates are random: `fixed.x` is removed. ML structural estimates
  stay unchanged, while GLS/ULS estimates can change in overidentified models.
  Lab specifications with observed covariates and `fixed_x = TRUE` are rejected.
- `covariance = "unrestricted"`, `"psd"`, `"barrier"` or `barrier(lambda)`
  replaces `psd = TRUE`. Barrier fitting is experimental and has no inference
  (`penalized`); `barrier(0)` gives the unrestricted fit.
- `options$start` replaces top-level `start` and `options$starts`. It accepts
  `"default"`, `"fabin3"`, `"lavaan-0.7.2"`, a previous fit or a parameter
  table, and overrides preset starts.
- `missing`, `cluster` and `meanstructure` are removed. Estimators other than
  FIML and ML2S delete incomplete rows listwise; pairwise deletion and
  two-level fitting remain lab features.
- Group order follows prototype factor levels, setting the latent-mean
  reference group.
- Mplus specifications from `magmaanlab::mplus_model()` retain source, groups
  and category schema through fitting and serialization. Unsupported inputs
  raise `magmaan_mplus_error` with all required edits; model printing shows
  fittability. Plain strings continue to mean lavaan syntax.

## Inference policy and evidence

- ML and FIML provide observed-bread, empirical-score sandwich covariance and
  global/nested score and LR tests. Score tests use observed sensitivity with
  an expected-information metric; nested LR calibration uses empirical scores
  at the larger fit, including restricted, misspecified means. Components
  retain separate typed unavailable reasons.
- All-ordinal and complete mixed DWLS use exact empirical first-stage influence
  and estimated-weight sandwich covariance. Global tests use the fit-function
  statistic; nested tests use its difference with observed-Hessian geometry.
  OPG fitting weights and estimates are unchanged. DWLS has no LR test and
  no nested score test; ordinal ULS/WLS have no ordinary policy inference.
- DWLS nested comparisons include moment-nested Wu–Estabrook threshold
  invariance. Three-category threshold steps report `equivalent_models`;
  released-scale thresholds-plus-loadings versus loadings-only is `not_nested`.
- DWLS global and nested tests use the exact weighted chi-square All reference.
  Exact-first-stage and reference-law studies reconfirmed the policy across
  ordinal, mixed and threshold-invariance designs and eight textbook models.
  These studies support the default without guaranteeing nominal size in
  every finite-sample setting.
- `anova()` retries a worse-fitting larger model from the embedded restricted
  estimate. A better endpoint is used only with a passing native verdict;
  input fits stay unchanged. Recovery is recorded in `refit`; internal warnings
  are retained in `attr(result, "reseed")$warnings` and printed as notes.

## Test tables and references

- `summary()` and `anova()` use columns `test`, `statistic`, `df`, `reference`,
  `pvalue`, `recommended` and `reason`. Migrate `p.sb`/`p.peba4` to `pvalue`
  selected by `reference == "sb"`/`"peba4"`; select tests by their stable codes
  `score`, `lr`, `fit_function` and `fit_function_difference`.
- ML/FIML default to PEBA4 only; DWLS global and nested tests default to All.
  `references = c("sb", "peba4")` requests the former comparator p-values.
  Other supported reference laws reuse the same statistic and spectrum.
  `recommended` marks only the primary policy test with its default reference.
  LR rows remain visible with a small-sample over-rejection note.
- Unavailable tests retain typed rows. Compatibility tables add unscaled
  statistics, scales and shifts; `references` cannot accompany `lavaan_compat`.

## Modification indices

- `modindices()` combines robust one-df score indices and affine equality
  releases for ML, FIML and all-ordinal/mixed DWLS. Candidates include absent
  structural regressions among variables already in equations and reverse
  paths. Candidate order and typed unavailable rows are preserved.
- ML/FIML use observed nuisance sensitivity with expected-information metrics
  and EPCs, so negative observed release curvature alone no longer removes a
  candidate. DWLS uses exact first-stage rows and estimated-weight influence.
- `test = "lr"` refits candidates from embedded null starts and applies the
  nested policy (LR for ML/FIML, fit-function differences for DWLS).

## Standardized estimates

- `summary(fit, standardized = TRUE)` adds `std.lv`/`std.all` estimates and
  delta-method SEs using the selected covariance, plus endogenous-variable
  R-squared and SEs. Fixed markers, groups and ordinal parameters are included.
  Without covariance, estimates remain and SEs have typed missing reasons;
  defined parameters have no declared standardized scale.

## Fit measures

- `fit_measures(fit)` provides misspecification-robust policy point indices
  with a matching independence baseline and typed unavailable reasons.
  `summary(fit, fit_measures = TRUE)` attaches and prints them on request.
  Policy intervals and close-fit p-values await evaluation.
- `fit_measures(fit, lavaan_compat = ...)` provides lavaan's standard, scaled
  and robust families, intervals and close-fit p-values for supported bundles.
  FIML ML/MLR measures are validated against lavaan and use tighter saturated
  H1 EM precision.

## Lavaan compatibility

- `options$preset = "lavaan-0.7.2"` selects lavaan starts, search and acceptance
  rules for continuous ML/FIML and all-ordinal or complete mixed DWLS, with
  unrestricted covariance and supported linear equalities. Explicit options
  override the preset. Prepared DWLS fits accept these options; ML2S retains
  its fresh-fit fallback.
- `vcov()`, `confint()`, `summary()` and `anova()` accept named bundles through
  `lavaan_compat`; `infer()` can cache an additional bundle. Complete-data ML
  supports ML/MLM/MLR; FIML supports ML/MLR. All-ordinal reporting supports
  DWLS/WLSMV, ULS/ULSMV and WLS, including default nested difference tests.
- Complete mixed DWLS supports WLSMV covariance, Wald intervals, scaled-shifted
  global tests and Satorra-2000 nested tests, including grouped delta/theta
  fitting. Missing mixed observations and other mixed bundles remain unsupported.

## Fixes

- Grouped all-ordinal WLSMV covariance now preserves group-weighted sandwich
  geometry with the reporting denominator, correcting unequal-group SEs.
- Exact ordinal/mixed first-stage influence uses affected score blocks, pair
  counts and sparse Gamma-diagonal derivatives, reducing post-fit cost.
- Mixed modification indices now share the equality-release criterion scale,
  removing an extra factor two while preserving EPCs and fitting weights.

# magmaan 0.1.0

First simulation prerelease, paired with magmaanlab 0.1.0. The ordinary API
accepts lavaan syntax and lavaan-backed model specifications. EQS is lab-only.

Development is unfinished: remaining bugs and incomplete coverage of the main
estimators and inferential procedures remain to be resolved and validated.
Version 0.1.0 provides a versioned simulation snapshot, not a finished release.

- Saved specifications supply ordered variables, parameterization and grouping;
  conflicting call options error before fitting. Row accounting uses the
  resolved grouping.
- Defined parameter estimates are retained independently of inference.
- Confidence levels and parameter selections are validated. Unchecked
  convergence is distinguished from a failed fit.
- Help pages and the README document extraction for simulation consumers.

Automatic policy inference covers single-level complete-data ML. Other
estimators retain their estimates and report unsupported inference components.
This prerelease does not expand inference coverage or change its defaults.
