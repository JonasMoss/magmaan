### R bindings and public namespace transition

- Fitting and post-fit Rcpp glue compiles in topical `fit_*.cpp` files
  (estimation, ordinal, missing data, inference, robust, measures, frontier,
  profile tests, noniterative and prepared interfaces). Shared helper
  declarations/types live in private `glue_internal.h`, with definitions in
  `glue_util.cpp`; helpers used by one topic remain local. The split preserves
  the exported R interface and function bodies.
  Uncached Clang 21 `-O1 -g0 -march=native` glue compilation on the
  workstation took 33.76 s for the original `fit.cpp`, versus 6.25–11.37 s
  per topical file (shared utility: 7.44 s; serial total: 95.83 s). The split
  reduces recompilation after a topic-only edit, while increasing cold-build
  work because each translation unit parses the shared headers.

- **Prepared R interface:** `prepare_model`, `prepare_data`, `prepare_weight`
  and `estimate` retain separate, immutable process-local native handles for
  continuous ML/ULS/GLS/WLS/DWLS, FIML, ordinal ULS/DWLS/WLS and mixed DWLS/WLS.
  Ordinal schema augmentation and native matrix representation are prepared
  once; each dataset refreshes moments/patterns/starts. Weights are dataset-bound,
  and full categorical Gamma is optional. New FIML estimation does not compute
  H1 eagerly unless versioned fitting options require it. Prepared fitting
  matches `fit_model()`'s ordinary starts and optimizer options, including
  versioned ML/FIML presets, fit-local keyed start tables and replayable
  `fit_model()` routes. Continuous and all-ordinal covariance policies and
  mixed PSD fitting use the existing reference estimator compositions. Mixed
  barriers remain unsupported in both paths; prepared ML2S raises the typed
  `magmaan_unsupported_estimator` error and must use `fit_model()`.
  Ordinal loading equalities (single and grouped), theta loading/threshold
  invariance and nested DWLS policy pairs match fresh fits. Default ordinal LS
  fits retain the moment/Gamma/weight layout required by policy and named
  lavaan reporting bundles, including ULS; mixed ULS preserves the explicit
  unsupported-data diagnostic. Explicit `full = FALSE` weights remain
  estimation-only inputs.
  An internal structural-preparation counter gates repeated handle reuse.
  Existing fit lists and numerical audits are preserved. Legacy
  entry points remain supported for compatibility and specialized families;
  see [the rollout status](../../design/r-model-preparation.md) and
  [R usage](../../../r-package/README.md#reusable-model-data-and-weights).

- Exploratory R bindings cover lavaanify, fitting, sample-stat bundles, robust
  inference, fit measures, model implied moments, LS estimators, SNLLS, Ceres
  paths when enabled, and data-frame-to-model sample statistics. Frontier
  parameter profile-LR wrappers expose robust scaling for ML and continuous
  GMM, including opt-in estimated-weight GLS/WLS scaling for caller-fixed GMM
  weights.
- The `magmaan_core` data/robust surface includes explicit observed-missing
  mixed continuous/ordinal builders:
  `data_mixed_ordinal_stats_observed_from_{raw,df}` for fully pairwise
  observed stats and `data_mixed_ordinal_stats_hybrid_fiml_from_{raw,df}` for
  the continuous-FIML hybrid. `robust_mixed_ordinal_ij` exposes the
  misspecification-robust IJ covariance for mixed ULS/DWLS/WLS fits using the
  same explicit `magmaan_mixed_ordinal_data` object as fitting and profile
  RMSEA/LRT.
- **Case-level influence diagnostics** (exact leave-one-out engine; semfindr
  parity) landed 2026-06-23 as pure-R `r-package/R/case_influence.R`:
  `case_rerun()` (drop one case, down-date the sample statistics, warm-started
  refit; continuous ML/ULS/GLS, single- or multiple-group — the latter by
  passing the original `data` frame, with group-label-suffixed columns), then
  `est_change_raw()` (raw or std.all DFBETA), `est_change(se = )` (DFTHETAS +
  generalized Cook's distance `gcd`, standardized by the leave-one-out
  covariance under a `standard` / `robust.sem` / `robust.huber.white` regime),
  `fit_measures_change()`, and `mahalanobis_rerun()`. It reuses existing primitives only (`fit_ml`/`fit_uls`/
  `fit_gls`, `df_to_data`, `inference_information_expected`/`inference_vcov`,
  `fit_measures`), so no core C++ was touched. Output format mirrors
  `semfindr` (Cheung & Lai, 2026; Pek & MacCallum, 2011), validated live by
  `r-package/examples/case_influence_semfindr.R` and frozen against semfindr
  fixtures (`cpp/tests/fixtures/case_influence/`, regen
  `cpp/tests/tools/regen_semfindr_fixtures.R`, pin `semfindr_version.txt`).
  Parity is to ~1e-5 for the estimate/gcd changes and machine precision for
  fit-measure changes and Mahalanobis distance. The **approximate one-step
  (no-refit) engine** also landed: `est_change_raw_approx()` /
  `est_change_approx()` via `θ̂ − θ̂₍ᵢ₎ ≈ (N/(N−1))·V·s_i`, built on a new core
  accessor `inference::casewise_scores` (the N×n_free per-case ML score matrix
  `Z_c·WΔ` that `information_cross_products` is now the Gram of; bound as
  `infer_casewise_scores_fit`, verified equal to `lavaan::lavScores`).
  `est_change_raw_approx` matches semfindr to machine precision;
  `est_change_approx` uses the *correct* finite-sample scaling and so diverges
  from semfindr 0.2.0 by two documented constant factors (semfindr applies
  `N/(N-1)` twice to DFTHETAS and once too few inside gCD — a recorded oracle
  defect, gated transitively up to those factors). `fit_measures_change_approx`
  is intentionally not provided (the exact `fit_measures_change` covers that leg
  cheaply since refits are reused from `case_rerun`; the no-refit version would
  be lossier). The multi-group and robust-regime exact variants also landed
  2026-06-23 (see above), as did the **misspecification-robust ("complete-sandwich")
  one-step** — `est_change_{raw_,}approx(fit, type = "estimated.weight")`, the
  casewise dual of the estimated-weight SE, via the new core accessor
  `estimate::continuous_ls_casewise_influence_ij` (see the estimated-weight stream
  above and `papers/estimated-weight-se`). Still in the backlog: DWLS/categorical
  and multigroup for that one-step path.
- **`fit_measures()` baseline is now `fixed.x`-aware** (2026-06-23). It calls
  `infer_baseline_fit(fit)` → `measures::baseline_chi2(pt, samp)`, which frees
  the exogenous (co)variances in the independence/baseline model exactly as
  lavaan does under `fixed.x` (baseline `df` drops by `px(px-1)/2`). Previously
  `fit_measures()` used the partable-unaware `infer_baseline(ss)`, so CFI/TLI
  diverged from lavaan for models with observed exogenous predictors; they now
  match to machine precision. A no-op for fits without exogenous variables.
- **Multiple-group order follows lavaan** (2026-06-23). The R group helpers
  (`fit_model()`, `df_to_data`, `df_to_fiml_data`, the ordinal/mixed data builders)
  derive group labels by data-appearance order (`unique(as.character(g))`),
  matching lavaan, instead of factor `levels()`. Previously a factor group
  column whose level order differed from its appearance order made magmaan and
  lavaan index groups differently. Explicit ordering remains available via the
  model spec's `group_labels`.
- **Packaging is portable** (2026-06-17). `r-package/` is self-contained: the
  C++ core plus `cpp/third_party/{port,quadpack}` is vendored into
  `r-package/src/{core,magmaan,third_party}/` by `r-package/tools/vendor-cpp.sh`
  (`just vendor`), so `R CMD INSTALL` / `remotes::install_github` builds it with
  no CMake and no prebuilt library. NLopt comes from a system install
  (pkg-config) or the `nloptr` CRAN package (bundled/self-built; `LinkingTo`/
  `Imports: nloptr`), so no system NLopt module is required (the Saga fix). The
  fast dev loop stays `just r-dev` (glue-only
  compile + prebuilt `libmagmaan.a` link via the `r-package/build-rdev/` mirror with
  `r-package/tools/r-makevars-dev`); the portable build is `just r-install`; `just
  vendor-check` guards vendor drift in `just check`. Cluster install (Saga /
  Sigma2): `r-package/tools/saga/`. This required one GCC-portability fix (R compiles with
  GCC, the canonical build uses clang): `ordinal.cpp` local lambdas renamed to
  the file's `*_fn` convention; the vendored f2c C is pinned to `-std=gnu11`.
- Composite (`<~`) model specifications are visible at the R boundary as
  folded `<~` partable rows while the hidden expanded Henseler-Ogasawara
  partable is retained as internal metadata for fitting and post-fit helpers.
  The R `composite_weights(fit, vcov)` accessor exposes recovered composite
  weights and delta-method SEs from the C++ post-fit primitive. The C++
  `BuildOptions` default is `CompositeMode::None`: core callers that accept
  `<~` must explicitly select either the historical Henseler-Ogasawara
  expansion or the native FC-SEM path. The R lavaanify boundary selects the
  historical expansion, while `api::frontier::model_from_lavaan_fcsem()` and
  the R FC-SEM helpers select native FC-SEM and require at least one `<~` row.
  R-side ordinary SEM helpers and native FC-SEM helpers reject each other's
  model/data classes and native FC-SEM partable data.frames rather than
  reinterpret a prebuilt object under the other composite semantics.
- A parallel native FC-SEM composite spec path is scaffolded behind
  `spec::BuildOptions::composite_mode = CompositeMode::FcSem`. In that mode
  `<~` rows are preserved, the first composite weight is marker-fixed by the
  ordinary auto-fix-first rule, composite indicator T-block rows and composite
  self-variance rows are stamped as fixed/derived placeholders, and both verbal
  `LatentNames::composites` and name-free
  `LatentStructure::composite_blocks` carry the composite contract. MatrixRep
  intentionally rejects native `<~` rows because native FC-SEM is evaluated by
  a separate W/T evaluator rather than the ordinary LISREL matrix path.
- The first native FC-SEM covariance evaluator exists as
  `model::FcSemEvaluator`. It assembles W and sample-backed T blocks, derives
  composite loadings, solves the derived composite disturbance variances through
  the structural system, and returns implied covariance matrices. The evaluator
  is covariance-only and separate from the ordinary LISREL `ModelEvaluator`.
  `estimate::ml_objective(FcSemEvaluator, SampleStats)` wraps it in the
  complete-data normal-theory ML discrepancy and supplies a central
  finite-difference gradient for the first native composite optimization
  tranche. `estimate::simple_fcsem_start_values` and
  `estimate::fit_ml_fcsem` provide the first low-level complete-data ML fitting
  path; the pure-composite, composite-plus-factor, and composite-structural HS
  fixtures fit from native starts and match lavaan's native `<~`
  objective/implied covariance plus the lavaan-reported weights, loadings, and
  regressions. `FcSemEvaluator::dsigma_dtheta` supplies a central
  finite-difference covariance Jacobian, and
  `inference::information_expected_fcsem` uses it for native expected
  information/vcov; the same three fixtures now match lavaan SEs for reported
  weights, loadings, and regressions. Native
  `measures::standardize::standardize_lv_fcsem` and
  `standardize_all_fcsem` use the evaluator's sample-backed W/T and total
  construct covariance semantics; the same fixture trio matches lavaan
  `std.lv`/`std.all` values and delta-method SEs for reported free rows.
  `standardized_rows_fcsem` lifts that into a lavaan-like row surface for
  native `<~`, `=~`, and `~` rows, including fixed marker rows with
  delta-method standardized SEs. Native FC-SEM df subtracts the composite
  indicator T-block moments from the user model, while the independence
  baseline remains the ordinary observed-variable baseline; `fit_extras_fcsem`
  and the existing fit-measure helper match the same lavaan fixture trio for
  chi-square, df, CFI/TLI/RMSEA, SRMR, loglik, AIC, BIC, and sample-size
  adjusted BIC. `api::frontier` now exposes a native FC-SEM model builder,
  complete-data ML fitting, expected SEs, fit measures, and standardized row
  reporting. The R frontier mirrors that slice with `fcsem_model_spec()`,
  `df_to_fcsem_data()`, `fit_ml_fcsem()` / `magmaan_fcsem()`,
  `fcsem_standard_errors()`, `fcsem_fit_measures()`, and
  `fcsem_standardized_rows()`, plus `magmaan_core$frontier_*` aliases for
  method-development workflows.
- Single-group native ML FC-SEM lavaan parity is fixture-gated for the
  pure-composite, composite-plus-factor, and composite-structural HS cases under
  `cpp/tests/fixtures/composite/`. The public-surface golden
  (`cpp/tests/golden/composite_golden_test.cpp`) fits through `fit_ml_fcsem()` and
  checks objective/chi-square, implied covariance in lavaan observed-variable
  order, df/npar, fit measures, reported raw rows, SEs, and `std.lv`/`std.all`
  rows against lavaan's native `<~` output. Native W/T matrix construction
  remains covered as evaluator/unit-test machinery rather than as a public
  fixture contract; the R bridge is still a methods-developer frontier surface,
  not a lavaan replacement interface.
- `fit_model(model, data, estimator, groups)` (named `magmaan()` before the
  2026-09-25 package split) is the high-level estimate-only
  convenience of the compiled R package `magmaanlab`. It composes `model_spec()`, data-frame sample-stat/raw-data
  construction, and the matching point-estimation wrapper for complete-data
  ML/ULS/GLS/WLS, FIML, ML2S, and ordinal/mixed DWLS/WLS where the lower-level
  inputs are available. `model_spec(meanstructure = "default")` enables means
  for grouped models; ordered declarations and explicit intercept syntax imply
  means as in lavaan. Saved specs retain both the resolved option and the
  requested default, so adding groups or ordered variables later re-evaluates
  defaults without losing explicit choices. For FIML and ML2S syntax calls it
  auto-enables a mean structure (and rebuilds syntax-backed no-mean specs) because the raw-data
  missing-data paths are mean-based; explicit `meanstructure = FALSE` errors
  early. Lavaan-style `se = "none"` and `test = "none"` are accepted as explicit
  point-estimate-only shortcuts; other values error and point users to explicit
  post-fit inference calls. It returns a `magmaan_fit` list: the raw primitive
  fit fields plus the source `model_spec`, syntax, estimator options,
  ordered-variable metadata, parameterization, and grouping metadata. Its print
  method reports only point-fit status and directs users to explicit post-fit
  primitives. `psd = TRUE` dispatches every estimator branch except two-level
  to its PSD-constrained frontier fitter and records `options$psd`.
- Lab `vcov()` exposes formula-named regimes: expected/observed inverse
  information (ML/FIML), expected/observed-bread empirical sandwiches,
  normal-theory/empirical delta covariances (non-iterative CFA), and stored SAM
  covariance. Unsupported combinations error. Omission and legacy aliases
  preserve the existing estimator-specific defaults. Raw-data methods use
  retained observations when available; FIML requires its own retained data.
  These wrappers compose existing C++ inference primitives and do not change
  the ordinary-user inference policy.
- Named downstream inference conventions (2026-10-02): ordinary `vcov()`,
  `confint()`, `summary()` and `anova()` accept `convention`, defaulting to
  `"magmaan"`. `api::lavaan_inference_ml`, `lavaan_inference_ordinal` and
  `lavaan_nested_ml` compose retained-fit compatibility bundles; lab adapters
  are `convention_inference()` and `convention_nested()`. The same retained
  estimates support several bundles, optionally cached with ordinary `infer()`.
  Complete-data ML/MLM/MLR and ordinal DWLS/WLSMV, ULS/ULSMV and WLS have
  covariance/global routes; complete-data ML has default nested difference
  tests. FIML and ordinal nested compatibility remain typed unavailable.
  This is an ordinary reporting-interface limit: C++ and lab FIML covariance,
  global/nested engines and ordinal Satorra-2000 adapters already exist. The
  ordinal golden gate covers the delta mean-scaled result; the default shifted
  reporting bundle still needs composition and whole-bundle gates. The
  [reporting gap audit](../../validation/capabilities.md#reporting-gap-audit-2026-10-02)
  records remaining wiring separately from validation and policy composition.
  ML/MLM/MLR tests use standard/SB/YB-Mplus, with SB2001 for robust nested
  comparisons. Ordinal reporting uses per-group `n_g - 1` and re-evaluates the
  criterion at retained theta without optimizing. Live lavaan integration tests
  gate whole bundles, and C++ tests gate covariance algebra and unsupported
  states. The [capability inventory](../../validation/capabilities.md) records
  evidence limits. This changes the reporting contract and supersedes the
  no-convention decision; the automatic inference policy is unchanged.
- Ordinary API adopted 2026-10-01 and implemented 2026-10-02 (0.2.0 in
  development): [`magmaan_model()` then `magmaan(model, data, estimator,
  covariance, inference, options)`](../../design/r-interface-vision.md#ordinary-api).
  `magmaan_model()` builds the lab `model_spec()` once with
  `meanstructure = TRUE` and `fixed_x = FALSE` and freezes the data schema:
  observed variables, group labels and order (prototype factor levels, else
  first appearance) and ordinal categories (factor levels, else sorted values).
  A zero-row prototype suffices. Each fit checks its data against the schema,
  raising `magmaan_schema_error` with reason `undeclared_group`,
  `undeclared_category`, `changed_levels`, `empty_group` or `empty_category`;
  ordered columns are passed as factors with the declared levels and grouped
  rows are stably reordered into the model's group order, so every lab route
  sees one order. A syntax string or lab spec is a shortcut that constructs the
  model with `data` as prototype and rejects undeclared ordered factors. Lab
  specs with `fixed_x = TRUE` and exogenous observed rows are rejected.
  `covariance = barrier(lambda)` maps to the lab joint barrier with weight
  lambda; `barrier(0)` fits the unrestricted model and keeps its inference.
  `options$start` (`"default"`, `"fabin3"`, `"lavaan-0.7.2"`, a fit or a
  table) replaces `start` and `options$starts`; with engine options an explicit
  start overrides the preset's. Removed arguments (`psd`, `start`, `fixed.x`,
  `meanstructure`, `missing`, `cluster` and the structural arguments now on
  `magmaan_model()`) raise errors naming their replacement. Fits use a lazily built native
  prepared handle cached by reference on
  the model. Repeated fits reuse structure; serialization and process changes
  rebuild the handle from portable fields. ML2S and ordinal DWLS with fitting
  options retain the `fit_model()` route pending prepared support.
- Fresh-fit parity, structural reuse, save/reload and PSOCK reconstruction are
  covered by the ordinary prepared-handle tests. Task-25.3 supplies ordinal
  equality and reporting metadata parity used by this composition.
- The ordinary-user R package `magmaan` (`r-magmaan/`, pure R, imports
  `magmaanlab`) is a scaffold of the two-package design
  ([r-interface-vision.md](../../design/r-interface-vision.md)). `magmaan()`
  rejects estimator-plus-correction names such as MLR and WLSMV, reports rows
  used and deleted listwise, and fits through prepared handles with the
  documented fallbacks. `infer()` runs the
  inference policy (next entry) and records each component (covariance,
  global score, global LR) as available or with a typed reason; for an
  unavailable component `vcov()` and `confint()` raise a
  `magmaan_inference_unavailable` condition carrying that reason.
  Its tests check parameter rows and free estimates against lavaan for ML,
  `std.lv`, multi-group `group.equal`, syntax intercepts and ordinal DWLS.
  The 0.1.0 simulation prerelease (2026-10-01) accepts lavaan syntax and
  lavaan-backed saved specs, inheriting ordered variables, parameterization
  and grouping before validation; conflicting explicit options error. EQS
  and partable-only specs stay lab-only. Per-group row counts use the resolved
  grouping. Defined estimates are evaluated through the C++ evaluator during
  lab fit reconstruction even without inference; uncertainty remains absent.
  Confidence arguments are checked and unchecked convergence stays distinct
  from failure. Help pages and the ordinary README specify simulation
  extraction and version-pinning contracts; the matched lab dependency is
  versioned 0.1.0 too.
  Release verification: 189 ordinary-package assertions and 82 focused lab
  assertions pass; the ordinary source archive has a clean `R CMD check`.
  The compiled source archive installs without CMake or a prebuilt core,
  exercising the bundled `nloptr` fallback. Its native sources match the frozen
  release projection. Both packages are distributed as source archives for
  the local `v0.1.0` simulation prerelease; broader inference is not a gate.
  This completes the simulation packaging milestone only. Development remains
  unfinished: remaining bugs need fixing, and coverage of the main estimators
  and inferential procedures needs completing and validating before a finished
  release; the active backlog tracks that work.
- The banked [score-centering decision study](../../../experiments/decisions/03-score-centering/report.qmd)
  (2026-10-01) records a fixed-allocation grouped-mean counterexample to universal
  raw likelihood-score meat. Independent confirmation (8,000 draws) and
  closed-form replay find N=300 parameter-variance ratios 1.118 raw/global versus
  0.994 within groups, against the exact 1/N target. Within-group parameter
  covariance meets the registered criteria in this scope; centered score
  calibration fails the N=80 acceptance rule (6.2% rejection, 95% Wilson interval
  5.2–7.3%). Package defaults are unchanged. The
  [sampling-law covariance formula](../../scope.md#group-allocation-and-likelihood-score-covariance)
  is settled: fixed allocation uses within-group covariance; joint sampling
  retains between-group score-mean variation. The ordinary fixed-allocation
  extension is [banked](../../backlog/speculative.md#fixed-design-inference-under-mean-misspecification),
  with no queued experiment to choose that formula. Promotion needs a named
  consumer and native contract/regression gates. Regular complete-data ML
  confirmation (32,000 datasets) retains the raw reference under the registered
  rule: centering changes null rejection by at most 0.75 percentage points and
  empirical matched-null power is identical, with no qualifying size-error
  benefit. All nominal-size Wilson intervals are inside 3–7%, but two global
  N=80 SB paired noninferiority bounds exceed the one-point margin. Parameter
  covariances agree numerically; skewed nested N=80 Wald null coverage is 90.3%
  (95% Wilson interval 88.9–91.5%), a separate active validation issue. Prospective
  FIML confirmation adds 32,000 fresh MCAR/MAR datasets with no qualifying
  centering benefit in either sensitivity stratum and matched-null power changes
  at most 0.10 points. Observed-sensitivity N=80 global PEBA4 rejection is
  0.8–1.8%; nested MCAR rejection rises from 6.5% raw to 7.25% centered. Both
  strata retain raw as a comparator, without selecting an ordinary FIML default.
  Stationary covariances agree, but nested MCAR N=80 null Wald coverage is 92.35%
  (Wilson interval 91.10–93.44%), failing the registered coverage condition.
  The [likelihood-score bank](../../backlog/speculative.md#likelihood-score-centering-alternatives)
  owns reopening; no further generic centering comparison is queued. Absolute
  FIML calibration/interval validation and excluded boundary/penalty cases remain
  open under their existing backlog owners. Random missingness
  patterns must not inherit the fixed-group centering rule.
- The inference policy for single-level complete-data ML is
  `api::policy_inference_ml()` (`api/policy.hpp`), exposed as
  `magmaanlab::policy_inference(fit)` and run by `magmaan::infer()`. The
  parameter covariance is `robust::frontier::ntml_score_sandwich()`: the
  observed-information sandwich of exact casewise likelihood scores at the
  fitted point, so a misspecified structured mean does not bias the meat (the
  centered-moment `ntml_covariance(fit, true)` and `casewise_scores()` use the
  sample mean). The global score and likelihood-ratio tests come from the
  shared expected-information NTML geometry, each calibrated with SB and PEBA4.
  Components carry typed reasons (`not_converged`, `saturated`,
  `unsupported_model`, `numeric_failure`, `not_nested`, `penalized`) instead
  of substitute results. A barrier estimate with positive weight
  (`PolicyFitState::penalized`, set by `policy_fit_state(PenalizedFit)` and by
  the lab's `penalty_inference` marker) gets `penalized` for every component
  and nested test, for every estimator and before the convergence gate
  (2026-10-02). A PSD estimate on the cone boundary gets every component, flagged
  `psd_boundary`: the inference assumes an interior population (since
  2026-09-26; before, it was refused). Nested tests are
  `api::policy_nested_ml()`, exposed as `magmaanlab::policy_nested()` and
  `magmaan::anova()`: the likelihood-ratio difference and the efficient score
  at the restricted fit, each with SB and PEBA4, for a restricted model that
  drops, fixes or constrains alternative paths. Since 2026-10-02 both use
  observed geometry (`ntml_quadratic(hypothesis, score, Information::Observed)`):
  the score projects with the larger model's observed information at the
  restricted fit and keeps the expected metric on the projected directions (the
  FIML recipe, so its statistic changes too), and the Satorra-2000 LR spectrum
  reduces through the observed information at the larger model. The expected
  geometry, lavaan's, remains the default of `ntml_quadratic` and the lab
  (`inference_quadratic(geometry =)`); the calibration run is open.
  The decisions/04 amended 24-cell development pilot covers correct, mild
  residual-correlation and strong cross-loading misspecification, with frozen
  population misfit and paired geometry-gap diagnostics; 480 draws have no
  failed arms and zero ordinary-policy gap. This is not confirming size evidence.
  All-ordinal DWLS has its own policy since 2026-10-02
  (`api::policy_inference_dwls`, routed by `magmaanlab::policy_inference()`):
  the estimated-weight IJ covariance (`robust_ordinal_ij`, observed bread,
  Stage-1 threshold and polychoric influence and the diagonal weight's
  influence) and one global test, the fit-function statistic n F with the
  fixed-weight UGamma spectrum and SB/PEBA4, reported in `score` with label
  `fit_function`; `lr` carries the typed reason `inapplicable`. Only plain
  DWLS qualifies. The lab `vcov()` names the policy covariance as
  `regime = "sandwich_ij"`. Gates: exact-fit reduction to the fixed-weight
  sandwich and a stratified delete-one jackknife (delta single and two
  groups, theta) in `policy_dwls_test.cpp`; calibration remains open.
  Nested DWLS comparisons (`api::policy_nested_dwls`, routed by
  `policy_nested()` and `magmaan::anova()`) report the fit-function
  difference with the estimated-weight profile law
  (`ordinal_dwls_profile_lrt`): positive spectrum with numerical zeros
  removed, SB as full trace over the restriction df, PEBA4 on the whole
  spectrum, nesting verified by `embed_nested_null`, nested score typed
  unavailable. Gates: global-statistic difference, delta/theta invariance,
  two groups and convergence to Satorra-2000 under a true null. The shared inference-side
  `robust::embed_nested_null` matches formula keys including group and level,
  lifts the null's affine constraints and estimate into H1, and verifies its
  implied covariance and mean moments to relative tolerance 1e-10. Omission
  defaults to zero only for loadings, regressions and off-diagonal covariances;
  intercepts, means, variances, thresholds and scales need resolved rows.
  The model triple (`LatentStructure`, `LatentNames`, `Starts`) and its
  partable projection are unchanged. Score inference uses H1 derivatives at
  the embedded null point, with the lifted null nuisance tangent. ML/FIML
  moment reparameterizations use a damped analytic moment fit of H1 to H0's
  fitted moments and both Jacobians at that solution; tangent inclusion and
  interior rank/covariance checks are required. Other estimators support the
  key embedding and return typed `unsupported_nesting` for unavailable
  correspondences. Singular tangent/factor-covariance nestings return
  `boundary_nesting`. The shared map also feeds complete-data/FIML/ML2S exact
  Satorra-2000, ordinal/mixed-ordinal, pairwise-composite, continuous weighted
  inference and the SB2010 null-point injection. Mixed-point delta remains
  the lavaan compatibility option. In the expected geometry SB matches
  lavaan's `satorra.2000` with the exact restriction map. A model without free
  parameters is evaluated at its fixed values instead of optimized
  (`evaluate_fixed` in `estimate/fit.cpp`), passes the verdict vacuously, and
  gets an empty covariance. Its global score and LR tests use the full moment
  space: the zero-direction expected-information projector is identity before
  normal-theory whitening, and fixed means retain their moment rows even with
  no parameter columns. The ML objective also includes the fixed-mean
  discrepancy when its mean Jacobian has zero columns. Independent
  saturated-normal formulas and empirical
  spectrum calculations gate covariance-only/mean-structure fits, single and
  unequal-sized multiple groups, and casewise/tiled storage; saturated models
  still have no global test. Ordinary R direct/deferred inference is covered,
  including LR discrepancy parity against lavaan's fitted ML objective (lavaan
  suppresses global tests for zero-free-parameter models). `fitted()` in
  `magmaan` returns model-implied moments, so such a model gives a population's
  moments. C++ unit tests check the
  covariance against finite-difference casewise scores, including two-group
  scalar invariance with misfitting means; the R tests match lavaan's MLR
  standard errors (single- and multi-group with structured means), its
  Satorra-Bentler statistic, scaling factor and p-value, and its delta-method
  standard errors for defined parameters.
- `compute_defined(model, fit, vcov)` exposes C++ defined-parameter evaluation
  for `:=` rows through R. It keeps covariance selection explicit, supports
  chained definitions, and resolves `.pN.` plabel references using the fitted
  lavaanified model.
- Friendly R post-fit wrappers expose routine inspection without hiding
  statistical choices: `standardized(fit, vcov, type)` requires an explicit
  covariance matrix, `stats::residuals(fit, standardized)` / `lav_residuals(fit)`
  wrap raw or standardized residuals including the lavaan-style continuous
  residual z-statistics and the per-block `$summary` table (the
  `lavResiduals(fit)$summary` analogue: cor.bentler SRMR/USRMR with SE,
  exact-fit and close-fit z-tests, and a close-fit CI for cov/mean/total,
  the raw-metric residual ACOV congruence-scaled into the correlation metric),
  with an `estimated_weight = TRUE` frontier route
  (`measures::frontier::standardized_residuals_estimated_weight`) that swaps the
  NT residual ACOV for the Hall-Inoue infinitesimal-jackknife sandwich so the
  residual SE/z and `$summary` reflect an estimated continuous-LS second-stage
  weight (beyond lavaan; bootstrap-calibrated under non-normality where the NT
  residual SE is anti-conservative),
  `factor_scores(fit, data, method)` requires complete raw data
  and dispatches continuous regression/Bartlett vs ordinal/mixed EBM/ML/EAP
  according to the fitted data type, and `modification_indices(fit, data,
  candidates)` / `score_tests(fit, data)`
  forward to the explicit scaffold primitives. Ordinal and mixed-ordinal fit
  objects retain the categorical stats object used for fitting, so ordinary
  categorical MI/score calls work directly; callers may still pass the stats
  object explicitly.
- R post-fit inference helpers now separate primitive-shaped entry points from
  fit-list adapters for the first audited slice: vcov, z tests, Wald tests,
  RLS chi-square, U-factor construction, and robust SE helpers can be called
  with explicit `partable` / `sample_stats` / `theta` pieces, while existing
  fit-list calls remain available with explicit `*_fit` aliases.
- The R sample-moment path accepts `list(S = , nobs = , mean = )`, reorders
  named covariance matrices to model observed-variable order, and rejects
  malformed group counts, non-square or wrong-sized covariance matrices,
  non-finite moments, nonpositive `nobs`, and wrong-length means before calling
  C++ fitters.
- Future public API direction is a staged, explicit workflow rather than
  lavaan-style compound estimator strings. Users should build a model, build
  data or sample moments, fit point estimates, and then explicitly request
  standard errors, robust corrections, test statistics, fit measures, defined
  parameters, or summaries. High-level objects may retain model/data/fit
  context for ergonomic chaining, but each statistical choice must remain
  inspectable.
- The planned C++ facade sits above the existing primitive namespaces as value
  objects such as `Model`, `Data`, `Fit`, `Analysis`, and `Summary`. It may
  support fluent usage, while the primitive namespaces remain the
  methods-developer surface. Convenience recipes can be added later only if
  they expand to visible choices such as estimator, moment builder, standard
  error method, test statistic, and fit-measure inputs.
- Bindings should keep the same split between a small friendly surface and an
  inspectable primitive layer. Python can expose a friendly top-level API plus a
  `magmaan.core` submodule for C++-shaped primitives. R should avoid exporting
  every primitive into the package namespace; instead, the friendly staged API
  should be exported directly and low-level functions should live behind a
  single `magmaan_core` object.
- The staged C++ facade and R `magmaan_core` surface expose the experimental
  robust ordinal moment builders explicitly: all-ordinal h-weighted,
  all-ordinal DPD, mixed continuous/ordinal fixed-marginal polyserial DPD, and
  mixed continuous/ordinal Huberized residual stats. The staged C++ facade also
  exposes all-ordinal Huberized residual stats. Default ordinal and mixed data
  builders remain lavaan-compatible ML paths.
- The R `magmaan_core` surface also exposes mixed continuous/ordinal covariance
  shrinkage as an explicit `data::frontier` transformation, rebuilding mixed
  moments, `NACOV`, and DWLS/WLS weights before fitting rather than hiding
  shrinkage inside estimator strings.
- The R package is intended as a methods-developer interface over the C++
  library, not a second SEM implementation.
- Standard errors, information matrices, Wald/z tests, robust corrections,
  fit measures, defined parameters, and nested tests remain explicit post-fit
  calls outside `fit_model()`.
- Primary public declarations and internal implementation now live in the
  target namespaces: `parse`, `spec`, `model`, `data`, `estimate`,
  `inference`, `robust`, `measures`, `sim`, `optim`, and `compat::lavaan`. Repository
  code and R binding internals use those namespaces directly.
- The `optim` namespace owns optimizer interfaces/backends, terminal audit
  helpers, and the equality-constraint reparameterization transforms that map
  theta-space scalar/GMM problems into constraint-reduced alpha coordinates.
  Constraint construction still lives in `estimate` until the broader
  constraint-header retiering lands; the old `estimate/reparameterize.hpp` path
  is a forwarding shim.
- Old `fit/*` and `partable/*` compatibility headers have been removed; use the
  target namespace headers directly.
