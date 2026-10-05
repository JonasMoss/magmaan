# magmaan 0.2.0 (in development)

- Mixed DWLS estimated-weight inference computes diagonal Gamma influence
  from sparse case blocks, reducing post-fit cost without changing inference.

## Ordinary API and migrations

Calls written for 0.1.0 need changes. Each removed argument raises an error
that names its replacement.

- `anova()` retries a larger model that fits worse than the restricted model
  from the embedded restricted estimate. A better endpoint with a passing
  native verdict is used for the comparison and reported in a printed note
  and `refit` attribute; the supplied fit objects are unchanged. Internal refit
  warnings are retained in `attr(result, "reseed")$warnings` and printed as
  notes instead of being emitted. Explicit lab refits still emit fit warnings.

- `magmaan_model(model, prototype, ordered, group, group.equal, group.partial,
  identification, parameterization)` constructs a model once, with a frozen
  data schema, for any number of fits. `prototype` declares groups and ordinal
  categories; factor levels declare them completely, so a zero-row data frame
  suffices. Data that break the schema raise a `magmaan_schema_error`.

- `magmaan(model, data, estimator, covariance, inference, options)` keeps only
  the choices that define the estimate. A syntax string or lab specification
  is a shortcut that constructs the model from `data`; it rejects undeclared
  ordered factors.

- Every model has a mean structure. Saturated intercepts change no other
  estimate, standard error or test, but `coef()` and `vcov()` gain one entry
  per observed variable.

- Observed covariates are random: `fixed.x` is gone. ML structural estimates
  are unchanged; GLS and ULS estimates of overidentified models change.
  Regressions on observed covariates now get the ML policy's inference.
  Lab specifications built with `fixed_x = TRUE` and observed covariates are
  rejected, not converted.

- `covariance = "unrestricted" | "psd" | "barrier" | barrier(lambda)` replaces
  `psd = TRUE`. Barrier fitting is experimental: a once-per-session message,
  printed status and inference unavailable with reason `"penalized"`.
  `barrier(0)` is the unrestricted fit.

- `options$start` replaces both the top-level `start` and `options$starts`:
  `"default"`, `"fabin3"`, `"lavaan-0.7.2"`, a previous fit or a parameter
  table. An explicit start overrides a preset's.

- `missing`, `cluster` and `meanstructure` are removed. Estimators other than
  FIML and ML2S delete incomplete rows listwise; pairwise deletion and
  two-level fitting remain in magmaanlab.

- Group order follows the prototype's factor levels (lavaan uses first
  appearance), which also sets the reference group for latent means.

## Lavaan-compatible fitting preset

- `options$preset = "lavaan-0.7.2"` selects lavaan 0.7.2 starts, search and
  acceptance rules for continuous ML/FIML and all-ordinal DWLS with unrestricted
  covariance and supported linear equality constraints. Explicit fitting options
  override the preset. Fitting and `lavaan_compat` reporting are separate choices.

- Ordinal DWLS fitting options now use reusable prepared handles; ML2S
  retains its fresh-fit fallback.

- Reporting methods `vcov`, `confint`, `summary` and `anova` accept named
  lavaan bundles through `lavaan_compat`, defaulting to `NULL`. ML/MLM/MLR
  and all-ordinal DWLS/WLSMV, ULS/ULSMV and WLS covariance/global recipes run
  on retained fits; complete-data ML also has lavaan's default difference
  tests. `infer(fit, lavaan_compat)` caches an additional bundle. FIML fits
  support `"ML"` and `"MLR"` (observed Hessian or Huber-White covariance,
  Yuan-Bentler Mplus global test, standard or SB2001 nested difference), and
  all-ordinal `anova()` reports lavaan's default nested test (scaled-shifted
  Satorra-2000 for WLSMV/ULSMV). Incompatible choices error.

## Ordinary-policy inference: ML and FIML

- Complete-data ML nested LR calibration now uses exact casewise likelihood
  scores when the larger model has a restricted, misspecified mean structure.
  Saturated-mean comparisons retain their calibration.

- FIML fits now receive the ordinary inference policy: observed-bread
  casewise-score sandwich covariance, global and nested score and LR tests,
  each calibrated with SB and PEBA4. Score sensitivity is observed with an
  expected metric; nested LR uses empirical scores at the larger fit.
  Components fail separately with typed reasons.

- The score test is the primary test. `anova()` now lists it first and the
  likelihood-ratio (or fit-function difference) row second; select rows by
  `test`, not position. When a likelihood-ratio test is shown, `summary()` and
  `anova()` print a note that it tends to over-reject when N is small relative
  to its df.

- `anova()` for complete-data ML uses observed information in its nested
  tests, which stay valid when the larger model is misspecified. The score
  statistic and both tests' reference distributions change; the
  likelihood-ratio statistic does not.

## Ordinary-policy inference: DWLS

- All-ordinal DWLS ordinary policy now uses exact empirical first-stage
  influence. Policy SEs and global/nested p-values change; OPG fitting weights
  and estimates, `lavaan_compat`, and all lab defaults are unchanged. Global
  All and nested SB/PEBA4 references remain. Exact-first-stage calibration
  reconfirmation is pending (TASK-79).
  The global test retains n F and uses every positive robust-ordinal eigenvalue.
  The nested law has one term per restriction, using the observed Hessian at
  the larger fit and the exact restriction map. Earlier OPG-first-stage
  fresh-draw checks gave 2.9–6.8% global and 4.3–7.8% nested rejection at nominal
  5%; these results do not validate the new exact first stage. The separate-point
  profile law remains an explicitly named lab comparator.

- Complete mixed continuous/ordinal DWLS now composes the exact first-stage IJ
  covariance, global fit-function All test, and observed-Hessian parameter-space
  nested difference test with SB/PEBA4. Fitting keeps OPG NACOV weights; the
  global spectrum uses exact sampling rows. Validation is limited and calibration
  is pending. Nested score and mixed lavaan compatibility remain unavailable.

- DWLS nested tests now support moment-nested Wu–Estabrook threshold equality
  through an implied-moment embedding and null tangent in the larger model.
  Three-category threshold steps report `equivalent_models`; released-scale
  thresholds+loadings versus loadings-only stays `not_nested`. Validation is
  limited to numerical, interface and a 100-replicate mean/variance check;
  calibration is pending.

- All-ordinal DWLS fits get magmaan's inference policy: standard errors from
  the estimated-weight sandwich, which accounts for the data-dependent DWLS
  weight, and one global test, the fit-function statistic with the All reference
  p-value. DWLS has no likelihood, so no likelihood-ratio test is reported;
  `summary()` says so in a note. `anova()` compares two DWLS fits with the
  fit-function difference, whose reference distribution accounts for the
  estimated weight; no nested score test is available for DWLS yet. Ordinal
  ULS and WLS remain without policy inference.

## Mplus frontend

- Mplus fitting refusals list all needed input edits in order, with vector
  `reason` and named vector `edit` condition fields. Printing a Mplus model
  shows fittability and every needed edit.

- `magmaan_model()` accepts explicit `magmaanlab::mplus_model()` specs, preserving
  source, groups and category schema through prepared fits and serialization.
  Conditional X, NOMEANSTRUCTURE and summary-without-MEANS fits raise
  `magmaan_mplus_error` with a reason and input edit. Joint inputs use ordinary
  estimator choices and inference; plain strings remain lavaan syntax.

## Fixes

- Exact ordinal and mixed first-stage sampling influence now differentiates
  only affected mean-score blocks. Ordinal pairs reuse category-cell counts and
  bivariate-normal corner grids, reducing inference time while preserving
  sampling rows, Gamma and policy outputs within 1e-7.

- Mixed ordinal component modification indices now use the same criterion
  scale as equality releases, removing an extra factor two from ordinary and
  fixed-weight robust MI. EPC and fitting weights are unchanged; estimated
  mixed-weight inference remains unsupported.

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
