### Continuous FIML

## Lavaan compatibility reporting

`api::lavaan_inference_fiml` and `lavaan_nested_fiml` compose ML/MLR
compatibility bundles at retained estimates, exposed by `convention_inference`
and `convention_nested` and ordinary `vcov`, `confint`, `summary`, `anova`.
They do not change the ordinary policy below. MLM is inapplicable.

Pinned installed lavaan 0.7.2 settings are `missing="ml"`, `fixed.x=FALSE`,
`meanstructure=TRUE`, observed information, Hessian observed-information form,
and structured H1 information. ML uses standard covariance and the standard
likelihood-ratio test. MLR uses the observed-Hessian Huber–White sandwich and
`yuan.bentler.mplus`; that test internally overrides H1 to unstructured EM
moments. Its scale is `[tr(A1^-1 B1) - tr(A0^-1 B0)] / df`, composed with
`fiml_robust_mlr`. Yuan–Bentler global tests retain their own label.
Default `lavTestLRT` uses standard differences for ML and
`satorra.bentler.2001` for MLR, with the YB-Mplus single-model scales;
the saturated alternative's weighted scale contributes zero. This uses the
scalar reducer with explicit scales, rather than the residual-projector
single-model scaling of the data-level SB engines.

Installed-lavaan gates in `test_lavaan_compat.R` cover HS MCAR/MAR CFA,
school loading/intercept invariance, both argument orders, saturated
alternatives and a random covariate with missing values. Fits use the pinned
`lavaan-0.7.2` optimization preset. Cached, deferred and serialized reporting
preserve estimates and policy; nonnested, penalized, nonconverged and
mismatched-observation pairs refuse explicitly. These are compatibility gates,
not calibration evidence.

## Ordinary policy

`api::policy_inference_fiml` and `api::policy_nested_fiml` compose single-level
random-x continuous FIML with affine equality constraints, one or more groups.
The R inference context retains the raw observations and `FIMLPack`.
`api::FimlPolicyFit` owns an immutable evaluation-point snapshot and lazily
retains `fiml_score_meat_bread` (including failures) across covariance and
larger-fit nested LR calls. Global policy results are retained too. Existing
signatures remain fresh-computation wrappers; pair-specific embedded-null score
geometry keeps its original calculation. A fit-owned R cache invalidates on
changed portable inputs, cleared pointers or PID changes. `inference_reuse`
reports the ingredient build count. Ordinary `vcov`, `summary` and `anova`
consume these same policy results without adding another fitting route.
Covariance uses observed deviance Hessian bread and empirical casewise score
meat (`fiml_score_meat_bread`), the lavaan-gated MLR sandwich. Global score
uses `global_score_components` with observed-H0 sensitivity and the
pattern-conditional expected metric, direct saturated-moment casewise scores
at model-implied moments, and uncentered empirical score cross-products.
Nested score uses `nested_score_components` with observed sensitivity and
expected metric; its casewise parameter scores are evaluated at the embedded
null fit. Missingness patterns are random, not fixed sampling strata.
Global LR uses `fiml_ugamma_spectrum` with saturated observed-H1 information.
Nested LR uses the larger-fit observed bread and direct casewise likelihood
score meat, reduced by the exact restriction map through
`compute_satorra2000_from_sandwich`. The lab/compatibility FIML Satorra driver
remains unchanged: its empirical Magmaan route transports saturated influence
into the larger fit, which can differ under larger-model misspecification.
Global and nested score/LR tests default to PEBA4 (TASK-97; decision
2026-10-06), aligned with complete-data ML. SB p-values and scales remain
comparator fields, available through explicit ordinary `references = "sb"`.
Only score/PEBA4 is recommended; typed unavailability reasons stay independent.
Penalty and convergence gates precede computation; saturation suppresses only
global tests. PSD boundary fits retain the policy's interior-population caveat.

Reuse gates in `policy_fiml_test.cpp` compare cached and fresh covariance,
statistics, spectra and calibration probabilities with exact equality for
MCAR/MAR, one/two groups and nested pairs. Both R suites gate repeated calls,
serialized restoration and fork/PID rebuilding, with construction counters.

Single-thread timing on HS-sized data (301 rows, six indicators, three nested
models, two successive `anova` calls; median of three batches of 30 sequences,
fit time excluded): FIML fresh contexts 3.167 ms versus retained 2.633 ms
(16.8% saving, 1.20x); theta DWLS 5.233 ms versus 2.133 ms (59.2%, 2.45x).
Fresh calls remove only the fit's cache, reproducing the preceding per-call
ingredient construction; retained calls start warm. These timings measure
reuse, not statistical calibration, and preserve bit-identical results.

Validation is limited: `policy_fiml_test.cpp` gates complete-data covariance
and LR-statistic reductions, exact nested observed LR-spectrum reduction with
free means, frozen MCAR/MAR sandwich agreement with the lavaan-gated primitive
and a direct frozen lavaan MLR SE comparison,
grouped components, and first-principles finite-difference casewise scores at a
misspecified larger fit. Correct-specification large-N comparison with the
transported spectrum is asymptotic (5% at N=20000), not a finite-sample identity.
`policy_score_contracts_test.cpp` additionally gates free-direction stationarity,
raw score-meat covariance, observed bread by score differentiation, direct
embedded-null projected scores, nuisance orthogonality, sparse MAR patterns,
constraint-spelling equivalence at a common point and actual PSD-boundary
flagging. The two-group complete-data nested observed score statistic and
spectrum reduce to ML at relative 1e-8; covariance reduces at absolute 1e-9.
Global score and LR-spectrum reductions to ML are asymptotic.
Calibration evidence is research/44 for global geometry and decisions/03 for
nested observed score, with no MAR cells in the latent sensitivity panel,
one df-1 normal nested family, and growing conservatism with df. No new
confirmatory calibration is claimed. FIML lavaan compatibility is composed separately above.


- The versioned `lavaan-0.7.2` fitting adapter (2026-10-02) composes the
  observed-pattern objective with EM H1 starts, the ordered ML equality QR
  coordinates, pinned PORT controls and standardized/simple retries. The H1
  initializer uses marginal variances with zero covariances and SQUAREM
  acceleration ([Du and Varadhan, Table 1](https://arxiv.org/abs/1810.11163));
  native H1 keeps its pairwise initializer and plain EM. Residual-variance
  starts and retry scales use available-case variances followed by the
  whole-group `(N-1)/N` rescaling.
  Search minimizes half the observed-pattern deviance relative to H1, while
  native FIML likelihood/derivative primitives retain their full kernel scale.
  Both R packages expose this through `options` for unrestricted continuous
  FIML. Frozen MCAR/MAR and installed-lavaan gates cover single/grouped CFA,
  shared labels, `group.equal` loadings/intercepts, invalid starts, retry scales
  and identical-point gradients. The x100 rescaled fixture retains endpoint,
  objective, retry and verdict parity, but compares gradients at identical
  points because PORT's final gradient vector follows its floating-point path
  (opt maximum `1.03649e-5`, oracle `2.44914e-4`, threshold `1e-3`).
  Nonzero affine RHS, nonlinear constraints, PSD/barrier, ordinal and
  multilevel preset routes remain explicit errors;
  homogeneous standardized retries retain the shared constraint-surface guard.
  Native diagnostics remain available alongside the selected preset verdict.

- Direct observed-pattern ML over raw continuous data with missingness masks.
- Rows are compressed into observed-value patterns; the observed-pattern
  objective and analytic gradient reuse `ModelEvaluator` Jacobians.
- The explicit frontier `fit_fiml_psd()` path evaluates that same cached
  objective on Cholesky-lifted primitive covariance blocks and exposes the thin
  R wrapper `frontier_fit_fiml_psd()`. It preserves ordinary FIML's fixed-x
  missingness rules and returned partable parameterization; boundary inference
  is deliberately not claimed.
- `fit_fiml()` defaults to NLopt L-BFGS with an SLSQP retry when the L-BFGS
  run fails or returns a non-clean optimizer status; explicit `nlopt-lbfgs` and
  `nlopt-slsqp` remain available for diagnostics and parity checks.
- Cross-call precomputation is value-based, with no mutable cache state:
  `FIMLPack` (immutable pattern cache + pairwise-complete start statistics,
  built by `fiml_pack`) and `FIMLH1` (per-block saturated EM moments plus the
  converged H1 objective value, built by one EM run in `fiml_h1_moments`).
  The H1 EM fails closed by default when a missing-data block reaches its
  iteration cap, returning `FitError::OptimizerNonConvergence` rather than a
  last-iterate H1 value. The default relative EM tolerance is `1e-6`.
  `FIMLH1Options` exposes `max_iter`, tolerance,
  covariance floor/warning thresholds, and an explicit
  `error_on_nonconvergence = false` diagnostic mode that restores the old
  last-iterate-with-warning behavior. Near-singular EM covariance updates are
  still diagonally floored before iteration continues, with
  `FIMLH1::warnings` recording any repairs or low eigenvalues.
  Every post-fit helper (`fiml_extras`, `fiml_observed_information`,
  `fiml_robust_mlr`, `saturated_em_moments`, `fiml_eta_jacobian`,
  `fiml_ugamma_spectrum`, `fiml_baseline_chi2`), the FIML score/MI helpers
  (ordinary plus MLR robust), the FIML Satorra-2000 nested-test helper, and
  `fit_fiml` carry pack overloads next to the raw-only signatures, which now
  build the pack/H1 once and delegate. `api::sem` FIML fits build both eagerly
  at `fit()` time and expose them via `Fit::fiml_pack()` / `Fit::fiml_h1()`.
  The R FIML fit list mirrors this with opaque `fiml_pack` / `fiml_h1` external
  pointers: `fit_fiml_impl()` uses the retained pack for optimization, and the
  Rcpp MLR, FMG, and score/MI wrappers consume the retained pack/H1 when
  present, falling back to raw-data rebuilding only for old/minimal fit lists.
  A fit → test → fit_measures → SE/MLR session therefore runs the saturated EM
  exactly once (the H1 value and H1 moments also share that single EM run, where
  the raw-only `fiml_extras` previously ran two).
- Current checked-in fixtures cover single- and multi-group CFA, three-factor
  CFA, labeled equality CFA, latent structural models, observed-variable path
  models under random-x and complete fixed.x policies, equality-constrained
  structural regressions, dense non-monotone missingness, complete observed-row
  equivalence, multi-group fixed.x with complete exogenous variables and
  missing outcomes, explicit mean structures, and a full 25-item five-factor
  bfi real-data FIML parity case.
- Post-fit FIML extras include observed-data normal constants, saturated/H1
  likelihood, baseline/independence likelihood accounting, chi-square,
  information criteria, and fit-index inputs for the current fixture tranche.
- `SaturatedMoments` carries propagated H1 warnings plus warnings from the
  saturated information repair. The saturated information matrix is symmetrized
  and diagonally floored only when needed before forming `H^-1 J H^-1`; routine
  well-conditioned cases are unchanged, while high-dimensional sparse-missing
  cases now return a diagnostic Stage-1 object instead of aborting on a singular
  495x495 H1 information matrix. The advisory stress harness is
  `cpp/tests/checks/fiml_h1_edge`.
- Two-stage EM ML (`ML2S`) is a packaged missing-data estimator path alongside
  direct FIML. Stage 1 fits the saturated EM mean/covariance model and exposes
  `(H, J, ACOV)` plus casewise saturated-moment influence rows through
  `saturated_em_moment_influence`; Stage 2 runs complete-data ML on those
  saturated moments.
  `estimate::fiml::two_stage_em_ml_inference` converts the Stage-1 ACOV to the
  moment Gamma scales expected by the shared robust SE and U-Gamma reducers,
  returning Savalei-Bentler-style sandwich SEs, ML chi-square, df, the corrected
  U-Gamma spectrum, scaling factor, and scaled chi-square.
  `TwoStageBread::Observed` is available on the inference path for the
  misspecification-robust Stage-2 bread regime; complete-data tests reduce it
  to unstructured observed-bread robust SEM for the NT weight and observed-bread
  `robust_continuous_ls` for the ADF weight. The default remains the
  lavaan-parity expected-bread convention.
  The separate single-group/mean-structure frontier diagnostics
  `estimate::fiml::frontier::{two_stage,fiml}_information_choices` enumerate
  the historical matrix-estimation axes without changing either estimator's
  default. The ML2S audit crosses four saturated/structured,
  observed/expected Stage-1 breads, two score-meat evaluation points, and four
  complete-data Stage-2 residual metrics (32 rows). The direct-FIML audit
  crosses the six Equation-37 residual metrics catalogued by Savalei and
  Rosseel (2022) with four sandwich breads and two meat points (48 rows).
  Experiment replications/09 applies all 80 choices to the same generated samples to identify
  the Savalei--Falk/EQS configurations. Its 2 x 1,000 targeted run identifies
  the two-stage source candidate (11.5% rejection versus the published 10.0%):
  saturated observed Stage-1 bread, saturated score meat, and structured
  observed Stage-2 information. Direct FIML remains unresolved. The literal
  all-structured observed Equation-(5/6) row was usable in only 935 samples and
  rejected 36.5%, whereas the closest row used saturated expected residual
  information and a saturated expected/saturated-score sandwich (61.9% versus
  63.3%). This conflict is now an explicit matrix-oracle gate before any
  paper-era direct-FIML route is added.
  `estimate::fiml::two_stage_fit_measures` adds the matching TS global-index
  layer: baseline scaling (`cB`), scaled CFI/TLI/RMSEA, and robust CFI/TLI/RMSEA
  with RMSEA confidence intervals and p-values. Complete-data multi-group tests
  anchor the corrected SE/spectrum path against the ordinary complete-data
  robust.sem machinery; missing-data golden tests gate the TS scaling and global
  indices against lavaan `missing = "robust.two.stage"`.
  - *Frontier:* the Stage-2 weight is selectable
    (`estimate::fiml::TwoStageWeight` ∈ {`Nt`, `Uls`, `Dwls`, `Adf`, `Dls`},
    built by `two_stage_stage2_weight`). `Nt` is the lavaan
    `robust.two.stage` default and is unchanged; the non-NT members are
    weighted-LS Stage-2 estimators (`Uls` = identity, `Dwls` =
    `diag(Γ_FIML)⁻¹`, `Adf` = `Γ_FIML⁻¹`, `Dls` = Browne
    `Γ_NT`/`Γ_FIML` mix) that route the single-model GOF/SE path
    through `robust_continuous_ls` and the ML2S Satorra-2000/2001 difference
    tests through the same eigen-cores (a `(V, Γ)` swap, no new test machinery).
    R: `fit_model(estimator = "ML2S", stage2_weight =, dls_a =)`. Opt-in
    Stage-1 covariance conditioning is also exposed as
    `stage1_regularization = TRUE` / `list(...)`: it regularizes the saturated
    EM covariance before Stage 2 with a diagonal/scaled-identity/identity target,
    chooses the smallest intensity needed to meet the requested condition/eigen
    cap when no fixed intensity is supplied, preserves `$stage1_raw`, and
    propagates the same moment map through `$stage1$acov` by delta method.
    Regularized ML2S is frontier/non-lavaan-parity; the default remains raw
    robust.two.stage. Rationale and the open heavy-MAR efficiency/calibration
    sim: `two_stage_weighting.tex`.
- Robust FIML MLR post-fit reporting computes observed-pattern casewise
  sandwich SEs and Yuan-Bentler Mplus scaled-test traces for fixture-backed
  non-saturated single- and multi-group cases. The R surface also exposes
  `fiml_observed_vcov()` / `inference_fiml_observed_vcov()` and routes
  `vcov(fit, regime = "model" | "robust")` for FIML to the inverse observed
  information or the MLR sandwich respectively; saturated (`df = 0`) FIML fits
  still return robust `vcov`/`se`, with scaled-test scalars set to `NaN`. The
  observed FIML information
  (`fiml_observed_information`, the MLR sandwich bread, and the `api` FIML
  information path) is analytic: the per-pattern moment-space Hessian chained
  through the model Jacobian plus the pattern-aggregated moment gradient
  contracted with the closed-form LISREL second derivatives (shared with the
  complete-data analytic observed information via `detail_second_order.hpp`);
  `diagnostic::fiml_observed_information_fd` retains the central-difference
  route as a regression comparator.
  The comparison primitive
  `inference_fiml_information_vcov()` additionally exposes the structured
  expected Fisher, observed-H1, and full observed-Hessian matrices at one
  retained fit. Each information matrix is returned with its inverse and with
  the same observed-pattern score-cross-product sandwich, preserving equality-
  constraint projection through the model covariance. Experiment research/39 validates
  these six SE conventions against explicit lavaan settings and studies their
  calibration without changing the high-level FIML defaults.
- The public fixed.x policy rejects missing observed exogenous variables rather
  than approximating lavaan's conditional likelihood behavior.
- The R boundary exposes `df_to_fiml_data()`, estimate-only `fit_fiml()` with
  retained FIML pack/H1 state, and packaged `fit_ml2s()` /
  `estimate_two_stage_em(..., kind = "ml")`. `fit_measures()` dispatches FIML
  fits to the FIML likelihood and independence-baseline path rather than the
  complete-data `2N*fmin` baseline helper; `robust = TRUE` / `"MLR"` adds the
  corrected `XX3`/baseline scaling and robust CFI/TLI/RMSEA fields from
  `estimate::fiml::fiml_corrected_fit_measures`, using lavaan's FIML-C(V3)
  trace and reusing a fit's retained Stage-1 saturated moments when present.
  ML2S attaches its corrected
  `vcov`, `se`, `chisq`, `df`, `chisq_scaled`, and `scaling_factor` fields
  directly because those corrections are part of the named two-stage estimator
  rather than optional post-fit reporting; the C++ post-fit surface also exposes
  the lavaan `robust.two.stage` scaled/robust CFI/TLI/RMSEA family. On the R
  surface, `fit$ml2s` carries the same baseline/scaled scalar list so
  `fit_measures(fit, robust = fit$ml2s)` reports the lavaan two-stage robust
  global-index fields.
- FMG single-model goodness-of-fit p-values are first-class complete-data ML
  and continuous ULS/GLS/WLS post-fit diagnostics on the R surface.
  `fmg_tests()` returns the p-value,
  df, ML/RLS source statistic, method/parameter, UG flag, chi-square-equivalent
  diagnostic, truncation count, and UGamma/lambda spectra; `fit_measures(...,
  fmg = ...)` attaches the same table to the ordinary fit-measure path, while
  the legacy `fmg_pvalues()` remains a named-vector compatibility view. Fits
  built from `fit_model(..., data.frame, estimator = "ML")` or
  `fit_ml(model, df_to_data(...))` retain listwise-complete raw blocks in
  `fit$raw_data`, so FMG no longer requires a separate raw-data argument in the
  normal fit workflow. Continuous LS composes the existing
  `estimate::robust_continuous_ls` primitive and exposes empirical or
  normal-theory Gamma explicitly; continuous WLS also takes the caller's
  fitting weight because fit lists do not retain it. The public
  `df_to_data(scaling = "n" | "n-1")` choice makes covariance scaling explicit
  for cross-implementation work. The fused C++ spectra primitive supports
  complete-data single- and multi-group ML, including mean structures.
  FIML/missing-data fits
  are also supported (`fmg_tests()` accepts a `fit_fiml()` /
  `fit_model(..., estimator = "FIML")` fit, single- or multi-group): the
  missing-data UGamma spectrum is built first-principles by
  `estimate::fiml::fiml_ugamma_spectrum` from the saturated-moment ACOV
  `Gamma_mis = H^-1 J H^-1` (`saturated_em_moments`, with analytic observed-row
  Hessians for saturated H1 information and a C++-only finite-difference
  diagnostic comparator) plus the saturated observed H1 information `V = H` as
  the projector metric (PD by second-order optimality — the FMG-spectrum
  convention; a selectable structured-at-θ̂ variant was removed 2026-06-24, see
  the backlog). The route uses
  `U = V - V Delta (Delta' V Delta)^-1 Delta' V` and the df eigenvalues of
  `U Gamma_mis`, with the FIML LRT as the base statistic. Equality constraints
  are honored by projecting `Delta` into the local free-coordinate space:
  affine linear constraints use their `K` reparameterization, while nonlinear
  equality constraints use the tangent-space null basis of `[A_eq ; dh/dtheta]`
  at `theta_hat`.
  The saturated H1 is the `h1.information = "unstructured"` convention
  natural to FIML's EM saturated model: on complete data it reproduces lavaan's
  unstructured UGamma spectrum element-for-element (~1e-7), validated in
  `examples/fmg.R`. semTests'
  rescale-the-mis-normalized-lavInspect-UGamma FIML hack is unsound and is
  deliberately NOT matched. Under FIML only the biased Gamma-hat and the ML base
  are defined; `_ug` and `_rls` are rejected.
  Two-stage ML (`ML2S`) fits are supported the same way: `fmg_tests()` accepts a
  `fit_ml2s()` / `fit_model(..., estimator = "ML2S")` fit and applies the
  eigenvalue-tail transforms to the df-dimensional two-stage UGamma spectrum and
  Stage-2 ML base chi-square already attached as `fit$ml2s` (`eigvals`, `chisq`,
  `df`) by `two_stage_em_ml_inference`. The two-stage spectrum uses the
  saturated-moment EM sandwich ACOV as its meat and an expected normal-theory
  Satorra-Bentler weight built from the *unstructured* (sample/saturated h1)
  moments as the U-metric; as under FIML, `_ug` is rejected. Unsuffixed ML2S
  FMG requests use the Stage-2 ML discrepancy. An explicit `_rls` suffix uses
  `inference::rls_chi2()` with the same ML2S UGamma spectrum — the same helper
  the FIML and nested paths use, so `ml` and `rls` are now commensurable bases
  (both model-projected) under one spectrum. This previously routed to a
  separate `frontier::rls_mean_cov_chi2()` because `rls_chi2()` was
  covariance-only; that split is gone (see the RLS entry below). ML2S must be
  dispatched before FIML
  because a two-stage fit also carries a `magmaan_fiml_data` raw object. The
  two-stage scaling and SEs match lavaan's `missing = "robust.two.stage"`
  convention (Huber-White sandwich Stage-1 ACOV) to machine precision - base,
  `pvalue.scaled`, and SEs agree to ≲`1e-4` across the exp-research/05 grid (typically
  ~`1e-7`), the residual being EM/optimizer convergence tolerance, not a
  convention difference. It is *not* lavaan's plain `missing = "two.stage"`, which uses a
  normal-theory ACOV that collapses toward the naive test under non-normality (its
  implied reference-law mean diverges sharply from magmaan's while
  robust.two.stage's tracks it); the *base* matches both lavaan conventions
  (identical point estimates). `trace(UGamma) = E[T]` (normal-data `ncp_hat =
  mean(base) - mean(trace) ~ 0`) is an independent first-principles check.
  (The U-metric weight was previously built from the *structured* model-implied
  moments, leaving a 1-3% trace gap to robust.two.stage that grew with
  non-normality; the unstructured weight - the convention lavaan two-stage forces
  and FIML FMG already used - closed it exactly.) The historical comparison runner is
  `experiments/research/banked/44-fiml-global-tests/lanes/two-stage`; only a ten-rep
  local smoke survives there. Substantive SEM calibration and its target limits
  are summarized in that study's parent report. A separate literature reconstruction in
  `experiments/replications/08-savalei-falk-2014-test-conventions` separates
  modern oracle targets from Savalei and Falk's (2014) written conventions.
  The article describes analytic observed information (`SE=EXACT`) and
  structured-model residual projections, with an observed-information
  two-stage correction. Numerical rejection fingerprints instead favor
  expected-information-like FIML choices; original EQS 6.1 identity is still
  unresolved. The current MLR trace-difference, FIML saturated-H1 FMG metric
  and ML2S unstructured-H1 metric retain their existing oracle meanings.
  Any paper-era route needs independent same-data matrix/trace validation;
  numerical similarity must not relabel a current route as an exact replication.
  Nested/model-pair FIML FMG is
  available through the existing `robust_nested_lrt()` / `nestedTest()`
  `method = "restriction_map"` route when both fits are FIML and carry
  compatible `raw_data`: for empirical Gamma it builds the H1-anchored split
  sandwich `A1 = K1' I_obs(theta_H1) K1`,
  `B1 = (V Delta K1)' Gamma_mis (V Delta K1)`. The default
  `convention = "magmaan"` uses the saturated-EM eta-space metric
  (`V = SaturatedMoments::H`, `Gamma_mis = SaturatedMoments::acov`). The
  `convention = "lavaan"` compatibility mode instead mirrors
  `lavTestLRT(method = "satorra.2000")`: parameter bread is the per-observation
  observed FIML information, `WLS.V` is the per-group expected information
  over observed missingness patterns at the fitted larger model covariance,
  and `Gamma_g = n_g * SaturatedMoments::acov_g` is the saturated EM sandwich
  covariance. Its bread is the saturated observed Hessian and its meat is the
  saturated casewise score crossproduct. This matches lavaan MLR's default
  observed/structured information settings. The correction on 2026-09-25
  replaces the former masked raw-residual Gamma and saturated-covariance weight;
  native defaults and the ML2S compatibility convention remain unchanged. The R
  wrapper defaults `A.method` to `"delta"` when
  `convention = "lavaan"` is requested. Both conventions report the existing
  unscaled/scaled/mean-variance/scaled-shifted/exact-mixture nested result
  shape. `GammaSource::NT` intentionally keeps the saturated eta-space bread so
  every difference eigenvalue collapses to one. Nonlinear equality constraints
  are supported by local tangent bases for both models; when `"exact"` is
  requested for such a pair, the route warns and uses the local tangent
  restriction because no global affine exact map exists.
  Frontier reference regularization is opt-in through
  `h1_reference_regularization` on `robust_nested_lrt()` / `nestedTest()` for
  direct-FIML restriction-map tests. Defaults remain raw/lavaan-parity. In the
  lavaan convention, the option transforms the saturated covariance and
  propagates that transformation into its ACOV by the delta method; the weight
  remains evaluated at the fitted larger model. In the native eta-space
  convention it
  floors saturated H1 information blocks before inversion and recomputes the
  coherent H1 reference Gamma. The returned nested-test list includes
  `$h1_reference_regularization` diagnostics when the option is enabled. ML2S
  does not consume this option; its separate Stage-1 input regularization stays
  fit-time only.
  Two-stage ML (`ML2S`) pairs are routed the same way
  (`robust_nested_lrt()` / `nestedTest()` dispatch ML2S before the FIML check,
  since an ML2S fit's `raw_data` is a `magmaan_fiml_data`; `computation =
  "ml2s_eta"`). The ML2S difference reduction uses the selected Stage-2
  two-stage weight (`Nt` by default; `Uls`/`Dwls`/`Adf`/`Dls` for weighted
  Stage-2 fits, not the saturated `V = H` of the FIML route) with the EM-ACOV
  meat `two_stage_gamma_from_acov(sm)` and the Stage-2 chi-square difference as
  the base. Under `convention = "lavaan"`, ML2S keeps the selected Stage-2
  `WLS.V` but swaps the meat to lavaan's model-based raw-moment Gamma, which
  matches `missing = "robust.two.stage"` nested differences in the paper parity
  gate. A `GammaSource::NT` collapse (every difference eigenvalue exactly 1)
  gates the NT convention. The eigenvalue-tail battery is applied to any of these
  difference spectra (continuous ML / FIML / ML2S) through `fmg_nested(fit_H1,
  fit_H0, ...)`, the model-pair analogue of `fmg_nested_ordinal()`: it harvests
  `(T_diff, df_diff, eigenvalues)` from the restriction-map result and runs the
  `pEBA`/`pOLS`/`all`/`penalized-all` transforms, returning a `magmaan_fmg_tests`
  table. Complete-data ML pairs additionally accept an explicit biased `_rls`
  base: `T_RLS,H0 - T_RLS,H1` is evaluated at the two ML estimates and fed
  through the same restriction-map spectrum. The `_ug` suffix selects the
  grouped, mean-structure-aware Browne/Du-Bentler finite-sample Gamma
  correction; unsuffixed nested names remain ML for compatibility, while
  FIML/ML2S reject both `_ug` and `_rls`. Continuous ULS/GLS/WLS pairs use the
  estimator's quadratic-form difference and the continuous-moment sandwich
  restriction map; ULS defaults to normal-theory Gamma, GLS/WLS to empirical
  Gamma, and WLS requires its caller-supplied fitting weight. Their FMG rows
  use the `_ls` base suffix. The R workflow checks source statistics directly
  and all biased/unbiased ML plus continuous GLS/ULS transformed tails against
  `semTests::pvalues_nested(method = "2000")`. This is an asymptotically
  equivalent RLS comparator, not an RLS estimator or casewise RLS score-flip
  method.
  FIML convention note (2026-06-30): lavaan's public
  `lavTestLRT(method = "satorra.2000")` default is delta/scaled-shifted with
  the lavaan `WLS.V`/Gamma convention above. `convention = "lavaan"` matches
  that oracle for complete, MCAR, and MAR FIML/ML2S paper parity cells; the
  independent saturated-EM convention remains the magmaan default. The
  strict-invariance `"exact"` row space can still differ from lavaan's internal
  exact helper because lavaan carries earlier invariance rows into that exact
  body before projection; use `A.method = "delta"` for the lavaan-default oracle.
  See `project/validation/satorra2000_parity.md`.
- **Method-2001 difference spectrum (`U_D = U0 - U1`) for FIML/ML2S.** Alongside
  the Satorra-2000 restriction map ("method 2000", `U_D` from the H1 fit),
  `robust_nested_lrt()` / `nestedTest(ud_method = "2001")` builds the
  Satorra-Bentler (2001) difference of the two single-model residual projectors
  (`compute_diff_spectrum_2001`, the top `df0 - df1` eigenvalues of `(U0-U1)·Γ`
  in a common saturated meat space; FIML uses observed model-information bread
  with `V = sm.H, Γ = sm.acov` unless the same direct-FIML
  `h1_reference_regularization` option is enabled, while ML2S uses
  `V = ml2s NT weight, Γ = two_stage_gamma`). It feeds the same scaled/mixture
  readouts and the FMG/pEBA transforms, and unlike method 2000 accepts non-`==`
  nesting (different parameter counts) but can carry negative eigenvalues
  (flagged for fallback). This is the second `U_D` estimator the FMG-nested paper
  and `semTests::ugamma_nested(., "2001")` cross against; validated to ~5e-5 vs
  semTests on complete data (single + multi-group). The scalar two-constant
  baselines also exist for FIML/ML2S: `nestedTest(method =
  "satorra.bentler.2001"/"2010")` (the trace-based SB2001 and the M10 positivity
  fix; 2010 embeds the restricted point into the alternative slots). NOTE:
  under missing data,
  `lavInspect("UGamma")` uses a non-official normalization; magmaan matches
  lavaan's *official* scaling factor / `trace.UGamma` (1e-7), so validate the
  spectrum against complete-data semTests, not the missing-data `lavInspect`.
  The helper computes biased and optional Browne/Du-Bentler unbiased U-Gamma
  spectra through C++, keeping the U-factor, tiled casewise-contribution
  projection, grouped reduced matrices, and eigensolves out of R list
  roundtrips. The unbiased correction mirrors lavaan's complete-data rule: the
  covariance block uses `Gamma_NT(S)` at the *sample* covariance plus Browne's
  finite-sample coefficients, and mean-structure blocks keep `S` for means with
  the mean-covariance third-order block scaled by `n/(n-2)`. Biased-only
  single-block spectra can still eigensolve in row space when `N < df`; grouped
  and unbiased paths take the full reduced route so per-block denominators and
  the non-identity sample NT term are honored.
  FIML FMG is validated for the full multi-group measurement-invariance workflow
  (configural -> metric -> scalar: cross-group loading/intercept equality plus
  mean structure), for both the GOF spectrum and the nested restriction map, by
  C++ algebra cases in `cpp/tests/unit/fiml_test.cpp` and by
  `experiments/research/active/06-fiml-invariance-tests/lanes/oracle`, whose `--lavaan-parity` run
  reproduces lavaan's FIML LRT chi-square (~1e-7) and, on complete data, the full
  unstructured UGamma eigenvalue spectrum (~1e-5) across all three invariance
  levels and normal / heavy-tailed / MCAR cells. That audit also found and fixed
  a complete-data robust bug: the `build_u_factor` Expected-info projector had
  dropped the per-group weight `n_b/N`, biasing the UΓ spectrum (SB scaling, FMG
  p-values, robust difference test) for models with unequal group sizes plus a
  cross-group equality constraint; it was masked by equal-group designs where the
  weight is a global scalar (regression note in
  [project/validation/test_ledger.md](../../validation/test_ledger.md)).
  Saturated-moment reuse now covers the FIML nested path too: the
  `lr_test_satorra2000/2001_fiml_from_data` core functions take an optional
  precomputed `SaturatedMoments* sm_precomputed` (mirroring the ML2S overloads),
  and the `infer_fiml_lr_test_satorra2000` glue reconstructs it from a fit's
  `$stage1` (`saturated_from_stage1`) before falling back to a rebuild -- so a
  caller that stamps `$stage1` onto a FIML fit (or an ML2S fit that already
  carries it) skips the from-scratch saturated EM + observed-information build in
  the difference-spectrum driver. `fit_ml2s(stage1 = ...)` likewise skips the
  rung-independent Stage-1 EM when handed a precomputed object. Bit-identical to
  the rebuild (the EM is deterministic; asserted in `fiml_test.cpp` and at the R
  surface). One residual remains by design: each `fit_fiml` still computes its own
  `fiml_h1_moments` (the cheap mu/Sigma-only EM, no H/J), so the structured H1 is
  rebuilt per rung; injecting it would need a `fit_fiml` h1 argument plus a
  FIMLH1-from-R reconstructor, deferred as the structured optimization dominates.
- Ordinal and mixed-ordinal (polychoric/polyserial least-squares) fits are FMG
  supported via `fmg_tests_ordinal()` / `fmg_tests_mixed_ordinal()`. These reuse
  the UGamma spectrum, base LS chi-square `N*F_min`, and df that
  `robust_ordinal()` / `robust_mixed_ordinal()` already build from the polychoric
  NACOV sandwich (validated against lavaan ordinal robust internals in the
  ordinal goldens), then apply the estimator-agnostic `robust::frontier::fmg_test`
  eigenvalue-tail transforms - no new spectrum machinery. An ordinal LS fit has a
  single base statistic (no ML/RLS split) and the polychoric NACOV is already the
  asymptotic Gamma, so `_ml` and `_ug` test suffixes are rejected. The categorical
  sample statistics are passed explicitly, mirroring `robust_ordinal(fit, stats,
  weight)`; single- and multi-group fits are supported through the same
  per-block `n_b/N` robust ordinal sandwich. Anchored by
  `cpp/tests/unit/ordinal_test.cpp` ("Ordinal FMG transforms consume the
  robust_ordinal UGamma spectrum"; the FMG SB reproduces the stored
  Satorra-Bentler scaling to 1e-12) and `r-package/examples/fmg_ordinal.R`,
  which checks single-group and two-group all-ordinal/mixed SB parity. This is
  the gate for the polychoric-FMG paper track; the pEBA/pOLS/PALL transforms
  remain magmaan-original with no external oracle, as on the complete-data and
  FIML paths.
- Ordinal nested Satorra-2000 tests are wired for all-ordinal and mixed
  continuous/ordinal DWLS/WLS fits via `nestedTest()` / `robust_nested_lrt()`
  when the caller supplies the same `magmaan_ordinal_data` or
  `magmaan_mixed_ordinal_data` statistics object used for fitting. The C++
  workers build the H1 ordinal/mixed moment Jacobian, project through the
  equality reparameterization, pool the `n_b/N` weighted `{A1, B1}` sandwich over
  polychoric/polyserial NACOV blocks, derive either exact parameter restrictions
  or the lavaan-style delta tangent map, and reuse the generic Satorra-2000
  reduced eigenproblem. Mixed pairwise missing remains unsupported because the
  mixed pairwise moment/NACOV constructor does not exist yet. Validation lives
  in `r-package/examples/nested_test_ordinal.R`:
  a single-group loading equality and a two-group configural-vs-metric ordinal
  WLSMV comparison match lavaan's `lavTestLRT(..., method = "satorra.2000",
  A.method = "exact", scaled.shifted = FALSE)` row under lavaan's ordinal WLSMV
  convention, where the displayed difference statistic/`Df diff` are the
  spectrum-derived mean/variance-adjusted `T_adjusted`/`d0` for `m > 1`;
  `r-package/examples/mplus_wlsmv_invariance.R` pins the mixed listwise bridge
  on a small deterministic mixed CFA.
  Friendly FMG composition is available for both branches:
  `fmg_nested_ordinal()` and `fmg_nested_mixed_ordinal()` consume the existing
  Satorra-2000 difference triple and keep both `A.method = "delta"` and
  `"exact"` selectable.
- All-ordinal pairwise-deletion sample statistics are implemented for the
  categorical LS path (`data::ordinal_stats_from_observed_integer_data`,
  surfaced in R as `missing = "pairwise"` with `pd_gamma = "overlap" |
  "nominal"`). Thresholds use itemwise observed cases and polychorics use
  observed pairs; both variants feed the existing `OrdinalStats` / DWLS / WLSMV
  machinery. The `overlap` Gamma is the default and applies the support-overlap
  finite-sample scaling from `ordinal_pd_gamma.tex`; the
  `nominal` variant keeps the same PD point estimates but suppresses the
  overlap reweighting for replication of the conventional wrong implementation.
  Experimental second-stage all-ordinal weights can now be built from the same
  precomputed `OrdinalStats` without rerunning threshold/polychoric/Gamma
  estimation: `estimate::frontier::ordinal_stage2_weight_blocks()` supports
  ULS, DWLS, observed full WLS/ADF, a GLS-like NT association weight, and DLS
  interpolation, with R helpers `ordinal_stage2_weight_blocks()` and
  `fit_ordinal_stage2()`. The NT endpoint is intentionally a research
  construction, not a lavaan oracle: threshold uncertainty remains the observed
  ordinal Gamma while the association block uses the MVN normal-theory
  correlation Gamma; robust reporting keeps the observed pairwise Gamma as the
  meat.
  Mixed continuous/ordinal pairwise missingness remains unsupported. Regression
  coverage lives in `cpp/tests/unit/ordinal_test.cpp`. The redundant construction
  probe was removed; its smoke outputs were not calibration evidence.
