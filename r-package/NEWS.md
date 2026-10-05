# magmaanlab 0.2.0 (in development)

- All-ordinal DWLS ordinary policy now uses exact empirical first-stage
  influence. Policy SEs and global/nested p-values change; OPG fitting weights
  and estimates, `lavaan_compat`, and all lab defaults are unchanged. Global
  All and nested SB/PEBA4 references remain. Exact-first-stage calibration
  reconfirmation is pending (TASK-79).

- `robust_ordinal_ij(..., first_stage = "exact")` provides a complete-data
  all-ordinal empirical-Jacobian comparator, with sampling influence rows and
  Gamma. The default remains `"opg"`; fitting weights are unchanged. The ordinary policy now selects exact influence.

- Complete mixed continuous/ordinal DWLS now composes the exact first-stage IJ
  covariance, global fit-function All test, and observed-Hessian parameter-space
  nested difference test with SB/PEBA4. Fitting keeps OPG NACOV weights; the
  global spectrum uses exact sampling rows. Validation is limited and calibration
  is pending. Nested score and mixed lavaan compatibility remain unavailable.

- Lab ML robust SEs and score/MI meats now default to exact casewise likelihood
  projections, including restricted-mean shifts and joint-sampling group score
  means. `moments="auto"` chooses this empirical recipe; `"structured"` and
  `"unstructured"` remain explicit weight comparators. Raw, centered Zc and
  caller Gamma routes agree. Global GOF moment primitives retain their named
  conventions; grouped centered LS meats retain their fixed-allocation scope.

- `mplus_model()` accepts complete observed-X moment mentions as joint random-X
  models; partial mentions name the exact completion statement. Mplus 9.1
  probes show all X variance mentions suffice without means or WITH.

- DWLS nested tests now support moment-nested Wu–Estabrook threshold equality
  through an implied-moment embedding and null tangent in the larger model.
  Three-category threshold steps report `equivalent_models`; released-scale
  thresholds+loadings versus loadings-only stays `not_nested`. Validation is
  limited to numerical, interface and a 100-replicate mean/variance check;
  calibration is pending.


- `refit_from_null(fit_H1, fit_H0)` explicitly refits a larger model from the
  verified embedded restricted estimate with its own estimator and options,
  returning a new fit with native convergence diagnostics.

- Complete-data ML nested LR calibration now uses exact casewise likelihood
  scores when the larger model has a restricted, misspecified mean structure.
  Saturated-mean comparisons retain their calibration.
- Prepared `estimate()` now accepts all-ordinal DWLS fitting options, including
  the lavaan preset, through the same configured engine as `fit_model()`.
- Mixed ordinal component modification indices now use the same criterion
  scale as equality releases, removing an extra factor two from ordinary and
  fixed-weight robust MI. EPC and fitting weights are unchanged; estimated
  mixed-weight inference remains unsupported.
- Mplus stability closeout replaces generic increment-era rejections with
  rule-specific Mplus behavior, scope reasons and remedies. A single inventory
  coverage matrix and expanded `mplus_model()`/`mplus_data()` help document the
  accepted subset. Optional sanitizer corpus sweeps and seven-family portable,
  prepared and fresh-process refit gates protect the lab adapter.

- Robust LS modification indices and equality releases now support observed
  sensitivity with estimated-weight influence for continuous ULS/GLS/DWLS/WLS/DLS
  and all-ordinal ULS/DWLS/WLS. Lab defaults use this route with the expected
  quadratic metric; expected sensitivity and fixed weights remain comparators.
- `fit_measures()` now counts equality-reduced ordinal parameters against the
  categorical moment layout. Grouped threshold/loading and label constraints
  now give lavaan-compatible df; dependent indices retain native n F reporting.
- Mplus growth, MODEL CONSTRAINT (NEW, equality equations, DO loops) and
  MODEL INDIRECT now use the shared model/constraint/defined-parameter core.
  SQRT, PHI/pnorm and LOG10 extend the shared expression language. Inequalities
  retain the deliberate boundary.
  Auxiliary NEW coordinates are safe in barrier gradients and moment curvature;
  noniterative CFA and profiled SNLLS reject these coordinates explicitly.

- DWLS global policy now reports the exact spectrum (All) weighted chi-square
  p-value on unchanged n F and robust-ordinal eigenvalues (decision study 05).
  Confirmed on fresh draws (2.9-6.8% at nominal 5%).
- Robust MI and equality-release tests accept validated caller `gamma` matrices
  or per-group NACOV blocks for complete ML, continuous LS and categorical LS.
  Fitting weights are preserved; supplied Gamma requires explicit fixed-weight
  inference and cannot supply casewise weight influence. Continuous-LS robust
  covariance and ML/LS profile-LRT adapters also accept caller Gamma.

- Nested all-ordinal DWLS now calibrates the unchanged fit-function difference
  with the df-length parameter-space estimated-weight IJ law: observed Hessian
  at the larger fit and exact restriction map, with SB and PEBA4, confirmed on
  fresh draws (4.3-7.8% at nominal 5%). The separate-point profile
  law remains an explicitly named lab comparator.
- `mplus_model()` imports binary/ordinal outcomes, threshold ranges and labels,
  DELTA scales (including fixed and equal scales), THETA residuals, and grouped
  CONFIGURAL/SCALAR models. All-ordinal DWLS uses explicit Mplus defaults;
  mixed, conditional and categorical ML routes report their boundaries.

- `mplus_data()` reads free/fixed individual and summary data through a typed
  C++ data plan, including missing flags and FILE/NGROUPS groups. It reports
  NOBSERVATIONS and dropped GROUPING codes; LISTWISE remains a fitting choice.
  Summary input covariances use N-1 and are rescaled to N for ML, matching Mplus.

- FIML fits now receive the ordinary inference policy: observed-bread
  casewise-score sandwich covariance, global and nested score and LR tests,
  each calibrated with SB and PEBA4. Score sensitivity is observed with an
  expected metric; nested LR uses empirical scores at the larger fit.
  Components fail separately with typed reasons.
- mplus_model() imports continuous GROUPING models, cumulative group sections,
  cross-group defaults and one CONFIGURAL/METRIC/SCALAR shortcut. Specs preserve
  numeric group order and source labels; unlisted data codes require filtering
  in R. Group-specific variable-role changes remain explicit rejections.
- Scalar profile tests and intervals now default to the misspecification-scaled
  reference. Empirical laws require raw data; callers can explicitly select
  ordinary or robust-scaled comparators.
- U-factor builders require an explicit bread. Observed is needed for
  pseudo-true covariance/nested restrictions; complete-data global GOF may
  use expected geometry.
- Global score flips select observed sensitivity for MAR FIML and expected
  sensitivity for complete-data correct-model GOF.
- Reliability with NULL Gamma derives empirical Gamma from supplied raw data
  and errors if no raw data are available.
- Structured/saturated moments remain unchanged pending a joint derivation
  recorded in the backlog.
- SAM and case-influence SE/type defaults are retained as diagnostic conventions.
- parameter_covariance() derives empirical meat from its retained observations;
  meat = "model" explicitly requests inverse information.
- score_quadratic() requires an explicit meat matrix; its metric is a free choice.
- Generic robust_nested_lrt() composes the ML/DWLS nested policy and raises
  typed unsupported reasons elsewhere. Explicit methods and Satorra aliases
  remain compatibility routes.
- FMG/semTests replication routes retain their comparator conventions.
  infer_continuous_ls_robust() documents its caller-fixed-weight law and
  rejects recorded data-estimated weights unless fixed_weight = TRUE is explicit.
- `mplus_model()` constructs rebuildable single-group continuous SEM specs from whole Mplus inputs, retaining reported input items in `$mplus_notes`.

- All lab `estimated_weight` switches now default to `TRUE`, the
  misspecification-robust choice: robust modification indices and score tests,
  residuals, GMM and ML2S profile tests/intervals, ordinal/mixed
  misspecification fit measures, and `frontier_rbm()`. Explicit `FALSE` remains
  the fixed-weight comparator, including for supplied-W fits without an
  estimated-weight recipe. Ordinary inference policy defaults are unchanged.

- `convention_inference()` and `convention_nested()` expose C++ composers for
  named lavaan inference bundles on retained estimates, with component-level
  unavailable reasons. Complete-data MLR uses the H1-minus-H0 trace; ordinal
  compatibility reporting uses per-group n minus one. Existing lab primitives
  and the ordinary package's automatic policy retain their recipes.

- Estimated-weight inference follows the weight recipe a fit records
  (`fit$moment_weight`, `fit$stage2_dls_a`), not its computational estimator
  label. Continuous DWLS and DLS fits now receive their own weight influence,
  with the fitted DLS mixing weight; previously they received the ADF influence
  and DLS assumed a = 0.5. Fits with a supplied W, and ordinal NT/DLS fits,
  fail with an `UnsupportedInference` error for the estimated-weight channel;
  fixed-weight inference is unchanged.
- WLS-computed fits use their recorded `fit$W`, so `weight =` is optional in
  modification indices, score tests, robust sandwiches, profile tests, RBM,
  residuals and case influence. An explicit `weight` must equal the recorded
  weight or a positive multiple of it.
- `frontier_profile_lrt_parameter_gmm()` and
  `frontier_profile_lrt_ci_parameter_gmm()` drop `ij_weight` and `dls_a`; the
  recipe comes from the fit.
- Two-stage (ML2S) fits support `modification_indices()`, `score_tests()` and
  their `_robust` variants for every Stage-2 weight. `mi` is the naive Stage-2
  statistic (attribute `mi_type = "naive_stage2"`); `mi.scaled` uses the
  Stage-1 moment covariance, plus the Stage-2 weight influence with
  `estimated_weight = TRUE`.
- `frontier_rbm()` and estimated-weight case influence use an ML2S fit's
  recorded Stage-2 weight and DLS mixing weight; `stage2_weight` and `dls_a`
  default to the record and must agree with it.
- Modification indices and score tests refuse ordinal association-ML fits in
  every interface.
- `policy_inference()` covers all-ordinal DWLS fits: the estimated-weight
  (IJ) covariance and one global test, the fit-function statistic labelled
  `"fit_function"`, with the likelihood-ratio component `"inapplicable"`.
- `policy_nested()` compares two all-ordinal DWLS fits with the fit-function
  difference and its parameter-space estimated-weight IJ reference (label
  `"fit_function_difference"`); the nested score is unavailable.
- `vcov()` gains `regime = "sandwich_ij"` for all-ordinal fits, the
  estimated-weight sandwich; the default stays the fixed-weight sandwich.
- `policy_nested()` uses observed nested geometry (see the interface vision);
  `inference_quadratic(geometry = "observed")` reproduces it, and the default
  expected geometry keeps lavaan's Satorra-2000 and `lavTestScore()`.

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
