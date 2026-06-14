# magmaan roadmap

This document summarizes the current implementation state and architectural
contracts for magmaan. It is not the active backlog. Remaining work lives in
[docs/backlog/todo.md](../backlog/todo.md).

Out of scope for this track: Bayesian SEM, multilevel SEM, latent
interactions/mixtures, EFA, inequality constraints (and active-bound
inference), and end-user lavaan
replacement ergonomics.

## Current State

magmaan is a C++23 library for methods developers working on linear SEM. It is
built under `-fno-exceptions -fno-rtti`, Eigen runs under
`EIGEN_NO_EXCEPTIONS`, and fallible APIs return `std::expected<T, Error>` —
the single C++23 feature the library depends on. Extension points are free
function templates over structural (duck-typed) interfaces rather than virtual
hot-path interfaces or `concept` constraints.

Lavaan remains the oracle. Parser output, lavaanified partables, point
estimates, standard errors, and chi-square statistics are compared against
checked-in lavaan fixtures where support is claimed. Fixture regeneration is
done with `tests/tools/regen_oracle.R`; CI does not invoke R.

The lavaanified model contract is the triple:

- `LatentStructure`: the estimable model, name-free except where estimator or
  identification choices require structure.
- `LatentNames`: the verbal model, including variable names, labels, groups,
  group levels, and `.pN.` plabels.
- `Starts`: free-parameter start hints.

`to_lavaan_partable()` and `from_lavaan_partable()` project this model to and
from `LavaanParTable`, which is the compatibility format used by R bindings and
golden `parTable()` fixtures.

## Implemented Capabilities

### Parser, lavaanify, and matrix representation

- Lavaan-style syntax parser with normative grammar in `docs/grammar/`,
  including fixed numeric intercept shorthand (`x ~ 0`) and parenthesized
  modifier labels (`(label)*x`), signed numeric modifiers/starts, chained
  modifiers, multi-LHS regressions, tolerated repeated `+` separators, and
  RHS continuations after the operator newline.
- Checked-in lexer, parser, partable, matrix, and fit oracle fixtures.
- Single- and multi-group LISREL matrix representation.
- Reduced LISREL lowers measurement rows whose RHS observed variable is already
  promoted into `lv_ext_order` into the structural `Beta` matrix, while keeping
  the mechanically inserted `Lambda[observed, phantom] = 1` identity. This
  keeps latent-change-score and single-indicator state chains on the same
  state vector as their autoregressions.
- Fixed.x resolution, mean structures, marker/std.lv/effect-coding
  identification, lavaan-style single-indicator residual fixing, start hints,
  and linear equality constraints.
- Linear equality constraints through affine reparameterization (θ = θ₀ + K·α)
  for the ML, GMM/GLS, and bounded ordinal LS paths; per-θ box bounds fold onto
  the reduced α for the pure-merge case.
- Nonlinear equality constraints (`a == b*c`, `b1 == (b2+b3)^2`) for the ML
  and complete-data LS paths: compiled to name-free expression trees by
  `resolve_lin_constraints`, enforced by NLopt SLSQP or the optional IPOPT
  interior-point backend, with the constrained vcov / df projected through the
  constraint Jacobian H(θ̂). They may be combined with linear equality
  constraints in the same model — the constrained optimizer then runs in the
  linear-constraint-reduced α-space. FIML has the same NLopt SLSQP / IPOPT
  nonlinear constraint support; ordinal and the separable (SNLLS) path reject
  them.
- The expression sub-language shared by `:=` defined parameters and `==`
  constraints supports `+ - * / ^`, unary `+ -`, and the unary functions
  `exp` / `log`; both the defined-parameter evaluator and the
  nonlinear-constraint evaluator evaluate them with forward-mode AD.
- Effect coding for loadings.

### Complete-data ML and inference

- Normal-theory ML fitting defaults to NLopt L-BFGS. Heuristic start values
  sign each free loading by its indicator's covariance with the factor's
  marker, and terminal audit records projected-gradient stationarity at the
  returned iterate so soft optimizer failures can be classified by geometry.
- A frontier complete-data ML covariance-continuation path fits
  `S_alpha = (1 - alpha) S + alpha T(S)` with `T(S)` either diagonal or
  identity-like (`mean(diag(S)) I` or raw `I`), warm-starting each stage and
  ending at `alpha = 0` by default. The C++ primitive is
  `estimate::frontier::fit_ml_ridge_continuation()` and the R research surface
  is `magmaan_core$frontier_fit_ml_ridge_continuation()`.
- Experimental complete-data ML IRLS paths fit the same ML objective through
  outer Fisher reweighting and inner GLS solves:
  `estimate::fit_ml_irls()` uses the full parameter block, while
  `estimate::fit_ml_irls_snlls()` uses Golub-Pereyra profiling for separable
  models. Mean-structure IRLS adjusts each frozen inner covariance target by
  the current mean residual outer product so the inner score matches the ML
  score up to scale. Linear equality constraints are handled in reduced
  coordinates; nonlinear constraints are rejected, and the SNLLS variant also
  rejects box bounds.
- Experimental local Fisher scoring for complete-data ML is exposed as
  `estimate::fit_ml_fisher()` and `magmaan_core$fit_ml_fisher()`. It computes
  the analytic ML gradient and expected-information Hessian approximation on
  the true `F_ML` scale, solves a damped local Fisher equation, and accepts
  steps by Armijo backtracking on the ML objective. This is separate from the
  IRLS paths above, which reoptimize a frozen GLS subproblem at each outer
  iterate. A companion `estimate::fit_ml_fisher_snlls()` /
  `magmaan_core$fit_ml_fisher_snlls()` path solves the same local Fisher
  equation through a Schur complement over the SNLLS β/α split; this is local
  block elimination, not Golub-Pereyra objective profiling.
- Objective-scale convention (unified 2026-06-09): `est.fmin = ½·F` for EVERY
  estimator (ML/FIML, ULS/GLS/WLS, ordinal, mixed) — the optimiser's minimum,
  half the discrepancy, and the quantity whose Hessian is the Fisher
  information. The goodness-of-fit statistic is `T = 2N·fmin = N·F` uniformly
  (`inference::chi2_stat`), matching lavaan's stored `fmin` element-for-element.
  The `½` lives only in the optimiser adapters; the math kernels stay full-`F`
  so the information/SE and score paths are untouched. Deliberate exceptions
  (ULS-standard Browne, FIML-standard LRT, the test-side `(N−G)/N` lavaan
  offset) are documented at `chi2_stat` and in
  [docs/design/numerical-conventions.md](../design/numerical-conventions.md).
- Expected information, finite-difference observed information, and analytic
  observed information for covariance and mean-structure models.
- Vcov/SE, Wald/z tests, chi-square/df helpers, LR/Satorra-2000 and
  Satorra-Bentler 2001/2010 compatibility nested tests, robust U-Gamma
  machinery, Satorra-Bentler-family statistics, robust SEs,
  FMG eigenvalue p-value tests (explicit method/options API, no parser),
  Browne residual NT/ADF, fixed-parameter modification indices,
  equality-release score tests, fit measures including RMSEA close-fit
  p-values and lavaan's saturated-user-model `TLI = 1` convention,
  structural-aware standardization, and C++ defined-parameter evaluation.
- `inference::frontier` robust (generalized / Satorra-Bentler-scaled)
  modification indices and equality-release score tests: each candidate carries
  the ordinary `mi` and a `mi_scaled = mi / c` with the per-direction scaling
  `c = gᵀB1g / gᵀA1g`, where A1/B1 are the parameter-space sandwich bread/meat
  surfaced by `robust::param_space_sandwich` (the same Δ'WΔ / Δ'WΓ̂WΔ that
  `robust_se` uses) and g is the efficient-score direction. Goes beyond lavaan,
  which falls back to the ordinary statistic when `se != "standard"`. Covers
  complete-data ML, both breads (`Information::Expected` ≈ robust.sem/MLM;
  `Information::Observed` ≈ robust.huber.white/MLR), single or multi-group (the
  per-block `n_b/N`-weighted sandwich pools across groups); reduces to the
  ordinary statistic exactly under the model-implied Γ_NT meat (Expected bread).
  Friendly entries under `api::frontier::{modification_indices,score_tests}_robust`.
  The fixed-parameter robust MI sweep batches eligible candidate rows into one
  augmented null-point evaluation, so score/info and the parameter-space sandwich
  are built once per sweep rather than once per candidate; duplicate matrix-cell
  edge cases fall back to the conservative one-candidate path. The sandwich
  helper also accepts precomputed casewise contributions (`Zc`) for callers that
  reuse the same raw-data meat.
  Validated four ways (lavaan implements no robust score test to diff against):
  exact reduction-to-NT, independent A1/B1 re-assembly, an R-internals oracle
  built from lavaan's delta/wls.v/gamma/ceq.JAC (`regen_robust_score.R`,
  convention-free θ-space scaling), and an advisory calibration + Wald/LRT-trinity
  simulation (`tests/checks/robust_score/`).
- The same scaling in the moment metric for the LS estimator tiers (2026-06).
  Continuous ULS/GLS/WLS/DWLS: `inference::frontier`
  `{modification_indices,score_tests}_robust` overloads taking the
  `estimate::gmm::Weight`, with A1 = Σ_b (n_b/N)·Δ'WΔ and
  B1 = Σ_b (n_b/N)·Δ'WΓ̂WΔ built by
  `estimate::continuous_ls_param_space_sandwich` (expected bread only; Γ̂
  empirical from raw, caller-supplied per-block, or model-implied Γ_NT per
  `WeightMoments` — the GLS weight with the Γ_NT(S) meat collapses to c ≡ 1
  exactly). All-ordinal and mixed DWLS/WLS (plus all-ordinal ULS):
  `estimate::frontier`
  `{modification_indices,score_tests}_{ordinal,mixed_ordinal}_robust` over the
  [thresholds ; associations] moment metric, reusing the `robust_ordinal`
  block assembly (W = estimation weight, Γ̂ = `stats.NACOV`) through the shared
  `estimate::weighted_param_space_sandwich`; full WLS (W = NACOV⁻¹) reduces to
  the ordinary statistic exactly, and `mi`/`mi_scaled` keep the lavaan-matched
  row-type moment-scale convention (c carries no moment-scale factor). The
  per-direction worker is shared as
  `inference::frontier::score_for_direction_robust` (c is not W-scale
  invariant: A1/B1 must be on the same weight scale as score/info). The
  continuous ML/LS and ordinal/mixed-ordinal tiers are single- or multi-group
  (see the multi-group bullets). api/R wrappers deferred until a concrete
  consumer appears.
  Oracles:
  continuous DWLS (`se = "robust.sem"`, so lavaan's wls.v/gamma are the
  ADF NACOV — gaps ~1e-9) and all-ordinal WLSMV (polychoric NACOV — c gap
  ~6e-10) release-score fixtures 0007/0008, plus the two-group ordinal WLSMV
  release-score fixture 0012, in `regen_robust_score.R`; exact WLS/GLS
  reductions and a primitives re-assembly live in
  `tests/unit/score_robust_test.cpp`.
- FIML (missing-data) robust MI and equality-release score tests, the MLR corner
  (2026-06): `inference::frontier::{modification_indices,score_tests}_fiml_robust`
  build the bread A1 = (N/2)·H (the analytic observed FIML information) and the
  meat B1 = ¼·scoresᵀscores from the casewise observed-pattern deviance
  gradients, reusing the FIML-FMG machinery via the shared
  `estimate::fiml::fiml_score_meat_bread` (also feeds `fiml_robust_mlr`). Because
  the FIML score is `-½·Σ_i scores_i`, the unscaled `mi` equals the non-robust
  FIML MI and the per-direction `c = gᵀB1g/gᵀA1g` is the Huber-White correction,
  → 1 under a correct normal model. Observed bread only (no expected-info FIML
  analogue), no H1 EM needed (two data passes per candidate, no EM), single- or
  multi-group via the same block-stacked Hessian / casewise-score layout used by
  `fiml_robust_mlr`. FIML has no batch augmentation, so this uses the one-by-one
  robust sweep.
  Oracle: FIML/MLR release-score fixture 0009 (`information.observed` /
  `lavScores`, θ-space assembly, c ≈ 2.16 on heavy-tailed + MCAR data) plus a
  non-robust-`mi` match and a c → 1 normal-data anchor in
  `tests/unit/score_robust_test.cpp`.
- Multi-group robust MI / score tests for the continuous ML and LS tiers
  (2026-06-13): the single-group guards in `inference::frontier` are removed; the
  per-block `n_b/N`-weighted sandwich (`robust::param_space_sandwich` /
  `estimate::weighted_param_space_sandwich`), per-group candidate enumeration
  (each absent statement is one candidate per group), and the full-θ-space
  nuisance projection all carry over unchanged, so no new statistics were needed.
  Validated by a two-group Γ_NT reduction (c = 1 exactly across unequal groups),
  a heavy-tailed empirical case, a GLS multi-group reduction, and the cross-group
  loading-invariance golden 0010 — the latter pins the `n_b/N` weighting that
  within-group reductions cannot (`A1 = lavInspect(fit,"information")` equals
  `Σ_b (n_b/N)·Δ_b'V_bΔ_b` exactly, c ≈ 1.24).
- Multi-group robust MI / score tests for the ordinal and mixed-ordinal tiers
  (2026-06-13): the `require_single_group_ordinal` guard in
  `estimate::frontier` is removed; the ordinal sandwich already loops over
  `stats.R.size()` and pools `A1/B1` through the shared `n_b/N`-weighted
  `estimate::weighted_param_space_sandwich`. The per-block threshold and
  association moment Jacobians write into the correct full-θ columns, so the
  continuous-tier nuisance projection carries over. Validated by exact
  two-group full-WLS reductions for all-ordinal MI/score and mixed-ordinal
  MI/score, a finite non-trivial two-group DWLS ordinal scaling case, and the
  two-group WLSMV golden 0012 (`c ≈ 0.856`) assembled from lavaan's
  per-group `delta` / `wls.v` / `gamma` lists.
- df>1 total release (2026-06-13): `inference::frontier::score_tests_robust_joint`
  releases all active equality constraints at once. The NT joint statistic is the
  multivariate score (Lagrange-multiplier) form `T = uᵀV⁻¹u` over the
  df-dimensional efficient-score subspace `G` (release normals made
  info-orthogonal to the nuisance subspace; u = Gᵀs, V = GᵀIG), which reproduces
  lavaan's `lavTestScore(fit)$test$X2`. The robust report mean-scales by
  `c̄ = tr((GᵀA1G)⁻¹(GᵀB1G))/df = Σλ/df` and also gives the exact eigenvalue-mixture
  p-value `Pr(Σλⱼχ²₁ > T)` via the QUADPACK `imhof_upper`, with λ the generalized
  eigenvalues of (GᵀB1G, GᵀA1G) (`JointScoreTestResult`; the shared worker
  `score_for_subspace_robust`). At df=1 it reduces to the per-row `c` bit-for-bit.
  Complete-data ML (raw or caller Γ̂); single- or multi-group. Oracle: df=2 joint
  fixture 0011 (mi = lavaan total, c̄ ≈ 2.03, p_mixture vs `CompQuadForm::imhof`)
  plus a df=1-reduces-to-per-row unit check.
- Observed-bread robust SEs and observed-Hessian U-factors use total-N scaling
  and work on block-stacked multi-block covariance and mean-structure models.
- Browne's unbiased reduced gamma has a single-block reduced-matrix shorthand
  and a casewise multi-block primitive.
- The weighted-sum-of-chi-squares tail behind the FMG/pEBA/pOLS p-values
  (`robust::weighted_chisq::imhof_upper`) uses Ruben's positive-weight
  central-χ² series first, avoiding oscillatory-cancellation failures in deep
  tails. If the series does not converge it falls back to Imhof's characteristic
  function inversion through vendored QUADPACK `qagi`
  (`third_party/quadpack/`, f2c-translated, public domain;
  cmake/QuadpackVendor.cmake — the second vendored static library after PORT),
  with a dense-Simpson fallback for weakly damped small-df tails where qagi's
  extrapolation breaks. A self-contained C++ golden pins fixed-spectrum FMG
  p-values for each method against constants generated from R `stats` and
  `CompQuadForm::imhof`; unit tests also pin deep equal-weight tails to exact
  χ² references.

### Continuous FIML

- Direct observed-pattern ML over raw continuous data with missingness masks.
- Rows are compressed into observed-value patterns; the observed-pattern
  objective and analytic gradient reuse `ModelEvaluator` Jacobians.
- `fit_fiml()` currently optimizes directly with NLopt L-BFGS.
- Cross-call precomputation is value-based, with no mutable cache state:
  `FIMLPack` (immutable pattern cache + pairwise-complete start statistics,
  built by `fiml_pack`) and `FIMLH1` (per-block saturated EM moments plus the
  converged H1 objective value, built by one EM run in `fiml_h1_moments`).
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
- Two-stage EM ML (`ML2S`) is a packaged missing-data estimator path alongside
  direct FIML. Stage 1 fits the saturated EM mean/covariance model and exposes
  `(H, J, ACOV)`; Stage 2 runs complete-data ML on those saturated moments.
  `estimate::fiml::two_stage_em_ml_inference` converts the Stage-1 ACOV to the
  moment Gamma scales expected by the shared robust SE and U-Gamma reducers,
  returning Savalei-Bentler-style sandwich SEs, ML chi-square, df, the corrected
  U-Gamma spectrum, scaling factor, and scaled chi-square. Complete-data
  multi-group tests anchor the corrected SE/spectrum path against the ordinary
  complete-data robust.sem machinery; missing-data tests check finite corrected
  output under MAR patterns.
- Robust FIML MLR post-fit reporting computes observed-pattern casewise
  sandwich SEs and Yuan-Bentler Mplus scaled-test traces for fixture-backed
  non-saturated single- and multi-group cases. The observed FIML information
  (`fiml_observed_information`, the MLR sandwich bread, and the `api` FIML
  information path) is analytic: the per-pattern moment-space Hessian chained
  through the model Jacobian plus the pattern-aggregated moment gradient
  contracted with the closed-form LISREL second derivatives (shared with the
  complete-data analytic observed information via `detail_second_order.hpp`);
  `diagnostic::fiml_observed_information_fd` retains the central-difference
  route as a regression comparator.
- The public fixed.x policy rejects missing observed exogenous variables rather
  than approximating lavaan's conditional likelihood behavior.
- The R boundary exposes `df_to_fiml_data()`, estimate-only `fit_fiml()` with
  retained FIML pack/H1 state, and packaged `fit_ml2s()` /
  `estimate_two_stage_em(..., kind = "ml")`. ML2S attaches its corrected
  `vcov`, `se`, `chisq`, `df`, `chisq_scaled`, and `scaling_factor` fields
  directly because those corrections are part of the named two-stage estimator
  rather than optional post-fit reporting.
- FMG single-model goodness-of-fit p-values are first-class complete-data ML
  post-fit diagnostics on the R surface. `fmg_tests()` returns the p-value,
  df, ML/RLS source statistic, method/parameter, UG flag, chi-square-equivalent
  diagnostic, truncation count, and UGamma/lambda spectra; `fit_measures(...,
  fmg = ...)` attaches the same table to the ordinary fit-measure path, while
  the legacy `fmg_pvalues()` remains a named-vector compatibility view. Fits
  built from `magmaan(..., data.frame, estimator = "ML")` or
  `fit_ml(model, df_to_data(...))` retain listwise-complete raw blocks in
  `fit$raw_data`, so FMG no longer requires a separate raw-data argument in the
  normal fit workflow. The fused C++ spectra primitive supports complete-data
  single- and multi-group ML, including mean structures. FIML/missing-data fits
  are also supported (`fmg_tests()` accepts a `fit_fiml()` /
  `magmaan(..., estimator = "FIML")` fit, single- or multi-group): the
  missing-data UGamma spectrum is built first-principles by
  `estimate::fiml::fiml_ugamma_spectrum` from the saturated-model EM information
  `V = H` and the saturated-moment ACOV `Gamma_mis = H^-1 J H^-1`
  (`saturated_em_moments`, with analytic observed-row Hessians for saturated
  H1 information and a C++-only finite-difference diagnostic comparator), via
  `U = V - V Delta (Delta' V Delta)^-1 Delta' V` and the df eigenvalues of
  `U Gamma_mis`, with the FIML LRT as the base statistic. Equality constraints
  are honored by projecting `Delta` into the local free-coordinate space:
  affine linear constraints use their `K` reparameterization, while nonlinear
  equality constraints use the tangent-space null basis of `[A_eq ; dh/dtheta]`
  at `theta_hat`.
  This is the `h1.information = "unstructured"` convention natural to FIML's EM
  saturated model: on complete data it reproduces lavaan's unstructured UGamma
  spectrum element-for-element (~1e-7), validated in `examples/fmg.R`. semTests'
  rescale-the-mis-normalized-lavInspect-UGamma FIML hack is unsound and is
  deliberately NOT matched. Under FIML only the biased Gamma-hat and the ML base
  are defined; `_ug` and `_rls` are rejected. Nested/model-pair FIML FMG is
  available through the existing `robust_nested_lrt()` / `nestedTest()`
  `method = "restriction_map"` route when both fits are FIML and carry
  compatible `raw_data`: it builds the H1-anchored saturated eta-space
  `Delta1/P/A/C/S` reduction from `V = H` and `Gamma_mis`, supports exact
  equality-constraint maps and delta tangent maps, reports the existing
  unscaled/scaled/mean-variance/scaled-shifted/exact-mixture nested result
  shape, and rejects complete-data-only nested compatibility methods for FIML
  pairs. Nonlinear equality constraints are supported by local tangent bases for
  both models; when `"exact"` is requested for such a pair, the route warns and
  uses the local tangent restriction because no global affine exact map exists.
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
  C++ algebra cases in `tests/unit/fiml_test.cpp` and by
  `experiments/21-fiml-measurement-invariance-fmg`, whose `--lavaan-parity` run
  reproduces lavaan's FIML LRT chi-square (~1e-7) and, on complete data, the full
  unstructured UGamma eigenvalue spectrum (~1e-5) across all three invariance
  levels and normal / heavy-tailed / MCAR cells. That audit also found and fixed
  a complete-data robust bug: the `build_u_factor` Expected-info projector had
  dropped the per-group weight `n_b/N`, biasing the UΓ spectrum (SB scaling, FMG
  p-values, robust difference test) for models with unequal group sizes plus a
  cross-group equality constraint; it was masked by equal-group designs where the
  weight is a global scalar (regression note in
  [docs/validation/test_ledger.md](../validation/test_ledger.md)).
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
  `tests/unit/ordinal_test.cpp` ("Ordinal FMG transforms consume the
  robust_ordinal UGamma spectrum"; the FMG SB reproduces the stored
  Satorra-Bentler scaling to 1e-12) and `r-package/examples/fmg_ordinal.R`,
  which checks single-group and two-group all-ordinal/mixed SB parity. This is
  the gate for the polychoric-FMG paper track; the pEBA/pOLS/PALL transforms
  remain magmaan-original with no external oracle, as on the complete-data and
  FIML paths.
- Ordinal nested Satorra-2000 tests are wired for all-ordinal DWLS/WLS fits via
  `nestedTest()` / `robust_nested_lrt()` when the caller supplies the same
  `magmaan_ordinal_data` statistics object used for fitting. The C++ worker
  builds the H1 ordinal moment Jacobian, projects through the equality
  reparameterization, pools the `n_b/N` weighted `{A1, B1}` sandwich over
  polychoric NACOV blocks, derives either exact parameter restrictions or the
  lavaan-style delta tangent map, and reuses the generic Satorra-2000 reduced
  eigenproblem. Mixed continuous/ordinal nested tests remain explicitly
  unsupported. Validation lives in `r-package/examples/nested_test_ordinal.R`:
  a single-group loading equality and a two-group configural-vs-metric ordinal
  WLSMV comparison match lavaan's `lavTestLRT(..., method = "satorra.2000",
  A.method = "exact", scaled.shifted = FALSE)` row under lavaan's ordinal WLSMV
  convention, where the displayed difference statistic/`Df diff` are the
  spectrum-derived mean/variance-adjusted `T_adjusted`/`d0` for `m > 1`.

### Staged C++ facade

- `magmaan::api` provides a tested staged facade over the currently
  implemented core primitives. Model construction, data construction, fitting,
  standard SEs, robust reporting, tests, modification indices, score tests,
  Wald/z tests, standardization, defined parameters, fit measures, and nested
  tests remain explicit calls rather than a lavaan-style one-shot summary.
- The facade supports complete-data ML, raw continuous FIML, continuous
  ULS/GLS/explicit-weight WLS, and ordinal/mixed DWLS/WLS through the same
  underlying fit and post-fit primitives described above. Unsupported
  combinations fail with `api::ErrorStage::UnsupportedCombination`.

### Simulation primitives

- A first C++ simulation namespace, `magmaan::sim`, provides NORTA data
  generation as an explicit primitive rather than a fitting shortcut.
  `calibrate_norta()` maps a target observed correlation matrix to a Gaussian
  copula correlation matrix by deterministic Gauss-Hermite quadrature and
  pairwise bisection, then validates the calibrated latent correlation by
  Cholesky factorization. `simulate_norta_matrix()` samples a complete
  `Eigen::MatrixXd`; `simulate_norta_raw()` wraps the same matrix in
  `data::RawData` for downstream sample-stat builders. Both draw functions also
  accept `NortaCalibration` directly, so repeated draws can reuse the
  deterministic latent-correlation calibration.
- The same marginal-transform surface is reused by independent generators:
  `simulate_independent_matrix()` and `simulate_independent_raw()` draw
  independent latent standard normals, transform each column through its
  marginal, and skip NORTA correlation calibration entirely.
- Foldnes-Olsson style independent-generator simulation is available through
  `calibrate_ig()`, `simulate_ig_matrix()`, and `simulate_ig_raw()`.
  `calibrate_ig()` chooses a square root `A` of the target covariance
  (`IgRootKind::Cholesky` by default, or `IgRootKind::Symmetric` via
  eigendecomposition), solves the linear `A^3` and `A^4` systems for generator
  skewness and excess kurtosis, then moment-matches each independent generator
  marginal. The lower-level overload accepts an already chosen root and fitted
  generator marginals for experiments that want to inspect or cache the
  calibration. The exploratory R package exposes the same mechanism through
  `magmaan_core$sim_ig_batch()` and the reusable
  `sim_ig_calibrate()` / `sim_ig_draw()` split. Pearson IG draws use direct
  Pearson random generators instead of inverse-CDF transforms where closed-form
  RNGs are available; the R wrapper additionally batches reusable Type VI
  Pearson IG draws through R's gamma RNG before splitting the returned samples.
- Vale-Maurelli / Fleishman polynomial simulation is available through
  `fit_fleishman_coefficients()`, `calibrate_vale_maurelli()`,
  `simulate_vale_maurelli_matrix()`, and `simulate_vale_maurelli_raw()`.
  The first slice implements the classic third-order Fleishman transform
  `a + bZ + cZ^2 + dZ^3`: coefficients are solved from target skewness and
  excess kurtosis by exact polynomial moments, pairwise intermediate normal
  correlations are solved from the Vale-Maurelli covariance cubic, and the
  assembled intermediate correlation matrix is Cholesky-validated before
  sampling. The R package exposes the two-stage split as
  `sim_vm_calibrate()` / `sim_vm_draw()` plus `sim_vm_batch()` (calibrate from a
  target correlation + per-margin skewness/excess kurtosis; draw consumes the
  calibration's Fleishman coefficients + intermediate correlation), validated by
  `r-package/examples/sim_vm.R`.
- PLSIM piecewise-linear simulation is available through
  `fit_plsim_marginal()`, `diagnose_plsim()`, `calibrate_plsim()`,
  `simulate_plsim_matrix()`, and `simulate_plsim_raw()`. The first slice uses
  regular normal-quantile breakpoints, fits continuous PL slopes to target
  marginal skewness and excess kurtosis, exposes Hermite coefficients, and
  calibrates pairwise intermediate normal correlations with selectable
  covariance evaluators:
  `PlsimCovarianceMethod::Hermite`, `Quadrature`, `Rectangle`,
  `HermiteThenQuadrature`, or `HermiteThenRectangle`. The rectangle path
  evaluates the Foldnes-Grønneberg segment decomposition by reducing bivariate
  normal rectangle probability/first/cross moments to conditional-normal
  one-dimensional adaptive integration. Gauss-Hermite quadrature remains useful
  as a smooth deterministic comparison path, but the advisory PLSIM bench shows
  it can differ from the rectangle/Hermite paths around kinked transforms.
  Hermite is the default calibration path; `diagnose_plsim()` reports fitted
  marginals, feasible pairwise target ranges, per-pair calibration failures,
  achieved correlations, and the minimum intermediate-correlation eigenvalue
  before the stricter `calibrate_plsim()` wrapper returns a success value.
  `PlsimCalibration` is draw-ready in C++, and the R package exposes both
  `sim_plsim_batch()` and the reusable
  `sim_plsim_calibrate()` / `sim_plsim_draw()` split.
- Model-implied simulation is available as a population-construction bridge:
  `sim::lower_model_implied()` lowers a lavaanified `LatentStructure` plus
  `MatrixRep` and `theta` through `ModelEvaluator::sigma()` into per-group
  `MixedPopulation` blocks, shared observed-variable names/kinds, and
  projection thresholds. Empty implied means become zero population means;
  explicit mean structures become the continuous population mean. Threshold
  rows are read from the partable, sorted per group/observed variable, and
  projected on the raw latent-response scale, so Delta/Theta ordinal
  parameterizations require no simulation-side standardization option.
  `sim::simulate_model_implied_group()` and `sim::simulate_model_implied()`
  dispatch that lowered population through the existing normal, Student-t,
  finite scale-mixture, contaminated-normal, and slash generators. The R
  package exposes the same split as
  `sim_model_calibrate()` / `sim_model_draw()` plus `sim_model_batch()`,
  accepting either a fitted magmaan object or a lavaan-shaped population
  partable with an explicit `theta` vector. Copula/NORTA/vine/IG/VM/PLSIM
  bridges remain separate because the SEM supplies moments and thresholds, not
  marginal distribution specifications.
- Elliptical generator diagnostics report the radial scale second/fourth
  moments, the covariance-normalization multiplier used by the draw path, the
  kurtosis inflation factor, and implied marginal excess kurtosis for
  Student-t, finite scale mixtures, contaminated normal, and slash generators.
  Infinite fourth moments are explicit for the covariance-finite but
  fourth-moment-infinite Student-t/slash cases (`df`/`q` in `(2, 4]`).
  `tests/unit/elliptical_test.cpp` pins the deterministic formulas and fixed-seed
  covariance smokes.
- Ordinal/mixed observed-correlation calibration inverts a target observed
  correlation matrix plus per-variable marginals (ordinal category proportions
  or continuous) into a latent Gaussian correlation matrix + thresholds.
  `sim::calibrate_ordinal_correlation()` solves each off-diagonal independently
  (NORTA-style) under one of three `ObservedCorrelationMetric` options:
  `Polychoric` (ordinal x ordinal latent rho == target, closed form),
  `PearsonCodes` (monotone bisection so the integer-code Pearson r matches the
  target, using `data::ordinal_bvn_rect_prob` as the forward map), and
  `Polyserial` (ordinal x continuous closed form `rho*sum phi(tau)/sd`;
  continuous x continuous is identity). The assembled latent matrix is repaired
  through the shared `repair_correlation_matrix_if_requested`
  (None/Error/Ridge/Shrinkage), and the calibration object records per-pair
  diagnostics plus the achieved-correlation matrix at the shipped latent.
  `sim::ordinal_correlation_population()` lowers it to a `MixedPopulation` that
  feeds the existing `simulate_mixed_population_*` generators, while
  `sim::calibrate_ordinal_correlation_multigroup()` composes one such
  calibration per group with shared variable shape, group labels, and
  group-keyed achieved category proportions; its draw path threads one RNG
  sequentially across per-group `MixedPopulation` blocks just like
  model-implied simulation. `simulate_ordinal_correlation_normal()` is the
  single-group one-shot convenience. The R
  package exposes the two-stage split as `sim_ordcorr_calibrate()` /
  `sim_ordcorr_draw()` plus `sim_ordcorr_batch()`, and the multi-group split as
  `sim_ordcorr_mg_calibrate()` / `sim_ordcorr_mg_draw()` plus
  `sim_ordcorr_mg_batch()`. `MixedProjectionResult` now also carries achieved
  `category_proportions`; `sim::raw_data_from_mixed_projection()` wraps a
  projected block as a `data::RawData` with optional variable names and ordinal
  level labels, and `sim::raw_data_from_mixed_projections()` composes
  per-group projected blocks into multi-block `RawData` with populated
  `group_labels`. For workflows that already have estimated categorical
  summaries, `sim::calibrate_ordinal_correlation_summary()` and
  `sim::calibrate_ordinal_correlation_summary_multigroup()` accept observed
  kinds, latent-response thresholds, and a pre-estimated latent Gaussian
  correlation matrix directly (polychoric/polyserial summary `R`); they skip
  pairwise inversion, apply the same matrix-repair policies, and record
  `max_abs_error` as the largest off-diagonal repair delta. The R mirror is
  `sim_ordcorr_summary_calibrate()` and
  `sim_ordcorr_mg_summary_calibrate()`, whose outputs feed the existing
  `sim_ordcorr*_draw()` functions.
- Calibrated simulation generators are standardized around a two-stage
  contract: deterministic `calibrate_*()` calls return reusable state objects
  with fitted marginals, latent/intermediate matrices, achieved diagnostics, and
  iteration metadata; stochastic `simulate_*()`/draw calls accept that state
  plus only `n`, RNG/seed, and draw-time options. One-shot batch helpers remain
  as convenience wrappers. This path is now in place for IG, NORTA, PLSIM,
  pairwise Archimedean copulas, generic fixed-order C-vines, specialized
  three-variable C-vine root/family-selection policies, and single-/multi-group
  ordinal/mixed observed-correlation calibration at the R boundary and for IG,
  PLSIM, NORTA, copula/vine, and ordinal-correlation generators at the C++
  boundary.
  Long-running experiments persist those calibration objects explicitly through
  the `experiments/_support` cache helpers, which key ignored `results/cache/`
  entries by population, generator/options, magmaan package version, and git
  reference. Core and the R package do not perform hidden global disk caching.
- Initial NORTA marginals are standard normal, standardized lognormal, Tukey
  g-and-h, Pearson-system distributions, and Johnson-system SU/SB
  distributions. Fleishman polynomial transforms can also be passed through the
  same marginal-transform machinery, but they are generator transforms rather
  than guaranteed quantile functions. The Tukey path is a pragmatic
  skew/tail stress family with numerically standardized moments and
  `0 <= h < 0.25` so fourth moments are finite. Pearson marginals use the
  PearsonDS parameter convention for Types 0, I, II, III, IV, V, VI, and VII;
  Type IV quantiles are computed by finite-interval integration of the
  normalized theta-space density and safeguarded bisection, avoiding a GSL or
  PearsonDS runtime dependency. Johnson marginals use monotone transforms of a
  standard normal variate, fit SU/SB shape parameters to target skewness and
  excess kurtosis by deterministic quadrature, and plug into the same
  NORTA/independent-generator transform surface.
- Fixed-parameter t-copula simulation is available through `TCopulaSpec`,
  `simulate_t_copula_matrix()`, and `simulate_t_copula_raw()`. It draws a
  multivariate Student-t copula from a supplied correlation matrix and degrees
  of freedom, maps the resulting uniforms through the existing marginal
  quantile machinery, and rejects non-quantile generator transforms such as
  Fleishman. This is deliberately still a fixed-parameter generator; VITA/covsim
  calibration from requested observed moments/correlations to generator
  parameters lives above these copula draws. `vinecopulib`/`rvinecopulib` are
  treated as oracle/reference implementations for copula conventions, not as
  core runtime dependencies. `simulate_mixed_population_t_copula()` composes the
  same generator with the observed projection layer.
- Fixed-parameter bivariate Archimedean copula simulation is available through
  `BivariateCopulaSpec`, `simulate_bivariate_copula_matrix()`, and
  `simulate_bivariate_copula_raw()`. The first local families are independence,
  Clayton, Gumbel, Frank, and Joe. Sampling uses conditional inversion of
  `dC(u,v)/du`, then feeds the resulting uniforms through the same marginal
  quantile machinery as the t-copula path. The conditional CDF and inverse
  conditional CDF helpers are fixture-checked against rvinecopulib `hbicop()`.
  `bivariate_copula_tau()` and `bivariate_copula_from_tau()` provide the
  Kendall-tau parameter scale used for rank-based copula setup.
  `simulate_mixed_population_bivariate_copula()` composes the same generator
  with the observed projection layer. The first pairwise VITA/covsim-style
  calibration helper is also present: `bivariate_copula_observed_corr()`
  evaluates the quadrature-implied observed Pearson correlation after marginal
  transforms, and `calibrate_bivariate_copula_correlation()` bisects on Kendall
  tau for one bivariate family. `calibrate_bivariate_copula_correlation_matrix()`
  extends this to every off-diagonal entry of a target correlation matrix and
  returns pairwise parameters, achieved correlations, feasible bounds, and
  iteration counts. It also reports maximum absolute error and achieved-matrix
  eigenvalue diagnostics, with opt-in error/ridge/shrinkage repair toward a
  requested minimum eigenvalue. This is a matrix diagnostic/calibration layer
  rather than an automatic matrix-to-vine fitter. The first explicit joint vine
  sampler is available through `CVine3CopulaSpec` and
  `cvine3_copula_inverse_rosenblatt()` / `simulate_cvine3_copula_*()`: a fixed
  three-variable C-vine with variable 0 as root, bivariate pair copulas for
  `0-1` and `0-2`, and a conditional pair copula for `1-2|0`. The inverse
  Rosenblatt helper is fixture-checked against rvinecopulib
  `inverse_rosenblatt()` using the equivalent `cvine_structure(c(3,2,1))`.
  `cvine3_copula_observed_corr()` evaluates its implied observed correlation
  matrix by deterministic quadrature, and
  `calibrate_cvine3_copula_correlation()` fits the two root pair copulas plus
  the conditional copula to a 3x3 observed-correlation target.
  `calibrate_cvine3_copula_correlation_select_root()` tries all three possible
  roots by permutation and returns the best achieved matrix in the original
  variable order. `CVine3FamilySpec` allows different families on `0-1`, `0-2`,
  and `1-2|0`, and `calibrate_cvine3_copula_correlation_select_families()`
  searches a caller-provided family set for the fixed root-0 C-vine.
  `calibrate_cvine3_copula_correlation_select_structure()` combines the
  three-root search with per-edge family search for the three-variable case.
  The `simulate_cvine3_copula_*()` overloads that take a
  `CVine3CorrelationCalibration` simulate in the selected vine order and return
  columns in the caller's original variable order.
  `simulate_mixed_population_cvine3_copula()` composes explicit or calibrated
  three-variable C-vine draws with the observed projection layer.
  Unit validation now covers the full 3-variable path from target observed
  correlation through structure/family calibration to deterministic-seed
  simulated correlations. Generic fixed-order C-vine sampling is available
  through `CVineCopulaSpec`, `cvine_copula_inverse_rosenblatt()`, and
  `simulate_cvine_copula_*()` for arbitrary dimension, with inverse
  Rosenblatt validation against both the 3-variable specialization and a
  four-variable rvinecopulib fixture. `simulate_mixed_population_cvine_copula()`
  composes generic fixed-order C-vine draws with the observed projection layer.
  `cvine_copula_observed_corr()` and `calibrate_cvine_copula_correlation()`
  provide the first higher-dimensional fixed-order VITA slice: deterministic
  observed-correlation evaluation and sequential Kendall-tau calibration for
  one caller-supplied family/order. Broader structure/family policies and
  ordinal/polyserial/polychoric calibration are still future work.
- Scalar special-function helpers needed by Pearson quantiles and FMG F tails
  (regularized beta/gamma and their inverses, Student-t CDF/quantile, F upper
  tail) are centralized in the private `src/detail_distribution_math.hpp` header,
  and that is the settled long-term policy: hand-rolled local kernels with
  dedicated goldens vs base R (`tests/tools/regen_distribution_math_fixtures.R` ->
  `tests/fixtures/distribution_math.json`,
  `tests/unit/distribution_math_test.cpp`), no Boost.Math or other third-party
  special-functions dependency (Boost.Math throws under `-fno-exceptions`). Every
  new kernel ships a golden; `src/inference/inference.cpp` reuses the shared
  header rather than keeping its own copy of the incomplete-gamma helpers.
- Moment-matching now has an explicit simulation API:
  `fit_marginal_to_moments()` takes a `MomentMatchSpec` with target skewness
  and excess kurtosis, returning a fitted `MarginalSpec` plus achieved moment
  diagnostics. `MomentMatchFamily::TukeyGH` is implemented with a reusable
  two-moment finite-difference solve over Tukey `g,h`.
  `MomentMatchFamily::Pearson` implements the PearsonDS `pearsonFitM()`
  moment-classification formulas, converting magmaan's target excess kurtosis
  to PearsonDS raw kurtosis internally. Checked fixtures generated by
  `tests/tools/regen_pearson_sim_fixtures.R` compare fitted parameters and
  quantiles, including Pearson Type IV, to PearsonDS 1.3.2.
  `MomentMatchFamily::Johnson` fits Johnson SU and SB marginals numerically
  against the same two shape moments; SL (log-normal) is intentionally excluded
  from moment-matching (its skew and kurtosis lie on a 1-parameter curve, so it
  cannot 2-moment-fit) and remains reachable only via the direct
  `MarginalSpec::johnson(1, ...)` constructor. Checked fixtures generated by
  `tests/tools/regen_johnson_sim_fixtures.R` compare the fitted shape pair
  (gamma, delta), the SU/SB type, and quantiles to SuppDists 1.1.9.9
  `JohnsonFit`/`qJohnson`; because SuppDists's own moment fit is only
  loosely accurate, the fixture targets the realized moments of its returned
  shape (computed by independent high-accuracy quadrature) so both
  implementations describe the identical distribution and agree to ~1e-7.
  `MomentMatchFamily::Fleishman` reuses the Vale-Maurelli coefficient solver
  and exposes the resulting cubic as a regular transform marginal, while
  `marginal_quantile()` rejects it because the cubic need not be monotone.

### Optimizer backends

User-facing the optimizer is selected by a single kebab-case `optimizer = "..."`
string threaded through `magmaan()` and the underlying `fit_ml/fit_uls/fit_gls/
fit_wls/fit_*_snlls/fit_*_ordinal` entries; the table lives in
`include/magmaan/estimate/backend_strings.hpp` and parses into the C++
`Backend` enum. Accepted strings: `"ceres"`, `"ceres-bfgs"`,
`"nlopt-slsqp"`, `"nlopt-bobyqa"`, `"nlopt-tnewton"`, `"nlopt-var2"`,
`"nlopt-lbfgs"`, `"ipopt"`, `"port"`, `"port-nls"`. The R side passes the same
string through to a single Rcpp shim per fit family — no per-Backend wrapper
explosion. Solver tuning rides on a generic `control = list(max_iter, ftol,
gtol, history)` argument.

The R complete-data ML and LS helpers also expose box constraints through
`bounds = list(lower, upper)` or the named bound builders
`bounds_variance()` / `bounds_pos_var()`, `bounds_standard()`,
`bounds_wide()`, and `bounds_loading()`. The high-level `magmaan()` ML, ULS,
GLS, WLS, and ordinal LS paths thread the same bounds object into the C++
fit layer; FIML remains unbounded at the R surface.

Optimizer outputs carry function/gradient evaluation counts plus a refined
success status and final stationarity diagnostic. The C++ `OptimResult` and
`estimate::Estimates` report `OptimStatus` (`Converged`,
`LineSearchSalvaged`, `SingularConvergence`, or `Unknown`) and a final
(projected, when bounded) gradient infinity norm when the backend can compute
one. R fit lists expose these as `optimizer_status` and `grad_norm`; the
legacy `converged` boolean is now true only for a clean stationary optimizer
stop rather than any usable non-error return.


- `Backend::NloptLbfgs` is the current default for scalar discrepancies. NLopt
  is a required dependency in the ordinary build, so the default is always
  available.
- `Backend::Port` is the trust-region cross-check: vendored PORT (Bell Labs)
  `drmngb_` (TOMS 611 Dennis-Gay-Welsch model-Hessian trust region; the
  algorithm behind R's `nlminb`), supports bounds natively. Vendored at
  `third_party/port/` from AMPL/ASL + Fermi-LAT (both BSD-3, manifest in
  `third_party/port/README.md`). Replaces the previous CppNumericalSolvers
  `Backend::TrustRegion`.
- `Backend::PortNls` is the least-squares-shape counterpart: PORT `drn2gb_`
  (TOMS 573 NL2SOL adaptive trust region — the algorithm behind R's `nls`).
  Drives the multi-residual `GmmProblem` directly through reverse
  communication (alternating R/J requests), so NL2SOL sees the true
  Gauss-Newton-plus-secant model Hessian instead of the scalarised
  ½‖r‖² collapse. On non-convex SNLLS problems with multiple local
  optima (e.g., Bollen's democracy SEM under GLS) it may converge to a
  different basin than the gradient backends; the cross-check tests
  document this rather than enforce agreement.
- `Backend::Ceres` / `Backend::CeresBfgs` cover Ceres Levenberg-Marquardt and
  dense line-search BFGS on the least-squares path (build with
  `MAGMAAN_WITH_CERES=ON`).
- `Backend::Ipopt` is the optional system-IPOPT interior-point backend (build
  with `MAGMAAN_WITH_IPOPT=ON`). It is available as a general scalar optimizer
  and accepts nonlinear equality constraints. v1 uses IPOPT's limited-memory
  Hessian approximation, so magmaan supplies objective gradients and
  constraint Jacobians but not exact Lagrangian Hessians.
- `Backend::NloptSlsqp` exposes NLopt's SLSQP for ML, FIML, and LS scalar
  paths (gradient SQP, Kraft 1988). It accepts simple bounds and nonlinear
  equality constraints via analytic constraint Jacobians.
- `Backend::NloptBobyqa`, `Backend::NloptTnewton`, `Backend::NloptVar2`,
  `Backend::NloptLbfgs` round out the NLopt roster with distinct algorithm
  ideas: Powell 2009 derivative-free quadratic-model trust region (BOBYQA,
  finite bounds required); Nash 1985 preconditioned truncated Newton with
  CG inner solve (TNEWTON); Shanno-Phua 1980 full (dense) variable-metric
  BFGS (VAR2); and NLopt's own L-BFGS. All five share the one
  `NloptOptimizer` adapter parameterised over an opaque
  `NloptAlgorithm` enum.
- The local `ceres` and `ipopt` presets are optional optimizer comparison
  builds. Ceres is FetchContent-managed; IPOPT is deliberately a system
  dependency because its BLAS/LAPACK and sparse-linear-solver stack is a
  toolchain choice. `just r-install-ceres` and `just r-install-ipopt` mirror
  the relevant compile definition into the R shared object so the accepted
  optimizer strings are executable from the R dev surface in one install.

### Least-squares estimators

- ULS, GLS, and explicit-weight WLS discrepancies with scalar
  value/gradient and residual/Jacobian interfaces.
- Bounded LS fitting through scalar NLopt L-BFGS and optional Ceres.
- Optional Ceres dense line-search BFGS is exposed only for unbounded SNLLS LS
  research comparisons. It consumes the already-profiled nonlinear block, so
  within-block linear equalities remain compatible through the SNLLS
  reparameterization. General-linear equality constraints use a
  component-wise null-space basis, so independent loading-only and
  intercept-only constraints stay block-separable instead of being rotated
  together by one global SVD. It is not a general bounded or constrained
  backend.
- Automatic nonnegative variance bounds.
- Linear equality constraints on the LS path via the affine α-reparameterization
  (θ = θ₀ + K·α), shared with the ML path — no quadratic penalty.
- Fixed-parameter modification indices and equality-release score tests reuse
  the estimator-specific LS residual/Jacobian weighting for ULS/GLS/WLS.
- Representative modification-index and equality-release score-test fixtures
  now pin fixed rows plus generated absent covariance rows against lavaan across
  complete ML, observed-information FIML, continuous ULS, ordinal DWLS, and
  mixed ordinal DWLS. `ModificationIndexOptions` exposes absent cross-loadings
  and covariances for those paths; generated structural-regression rows remain
  deliberately out of scope.
- Continuous LS fixtures cover point estimates, degrees of freedom, and
  estimator-specific chi-square reporting for representative CFA,
  multi-group, labeled-equality, mean-structure, and observed-exogenous
  fixed.x cases. Mean-structure LS fixtures also cross-check SNLLS against
  the full LS path, and Ceres against NLopt L-BFGS when the Ceres backend is
  enabled.
- The Geiser textbook GLS corpus is checked in as a parity-tier golden
  fixture generated from `corpus/textbook-corpus/raw/geiser` plus installed
  lavaan. The test
  exercises all curated Geiser GLS cases and compares full/PORT and
  SNLLS/PORT-NLS implied moments against lavaan where the current model
  surface has strict parity. Manifest fixed.x path cases and the known
  alternate-basin `latent_ar_cross_lagged` case remain smoke-checked but not
  used as strict implied-moment parity oracles.
- A local ignored Mplus SEM source corpus can be built from
  `corpus/textbook-corpus/raw/MPLUS/*.zip` into
  `corpus/textbook-corpus/raw/mplus_sem` with
  `tests/tools/build_mplus_sem_corpus.R`. The tracked oracle layer
  (`tests/tools/regen_mplus_sem_fixtures.R`,
  `tests/fixtures/mplus_sem/`, and
  `tests/golden/mplus_sem_golden_test.cpp`) freezes derived lavaan sample
  statistics and estimates only. The current strict tranche gates six
  continuous growth examples across ML/ULS/GLS/WLS (24 optimizer fits) and
  records retained observed-path, ordinal, and mixed examples for corpus
  classification without committing Mplus raw data.
- Local ignored Little and Newsom textbook corpora can be built from
  `corpus/textbook-corpus/raw/little/*.rar` and
  `corpus/textbook-corpus/raw/newsom/*.zip` with
  `tests/tools/build_little_corpus.R` and
  `tests/tools/build_newsom_corpus.R`. The shared fixture regen script
  (`tests/tools/regen_little_newsom_fixtures.R`) writes grouped
  continuous/ordinal/mixed/observed manifests and lavaan oracles under
  `tests/fixtures/little/` and `tests/fixtures/newsom/`. The current strict
  C++ golden gates 17 auto-converted Little LISREL measurement models and 21
  curated Newsom lavaan examples; the rest remain catalogued as extracted
  source/data/model formulations with explicit unsupported or not-yet-gated
  status. A consolidated `magmaan_textbook_corpus_v1` index under
  `tests/fixtures/textbook_corpus/manifest.json` summarizes Geiser, Mplus SEM,
  Little, and Newsom as one textbook corpus without duplicating their heavy
  oracle payloads. The same directory also carries an advisory generated
  overlap graph (`overlap.json`) plus an empty curation hook
  (`overlap_overrides.json`) so future paper work can find same-data,
  same-syntax, same-shape, and same-oracle-structure examples while preserving
  every source case as its own fixture record. The local raw-corpus
  parser/lavaanify smoke sweep currently accepts all 38 Little catalogue
  models and all 142 Newsom lavaan models. Kline Guo measurement-invariance
  parity is now represented by a compact checked-in export
  (`case_exports.json`, regenerated by
  `tests/tools/regen_textbook_case_fixtures.R`) plus an optional
  submodule-backed cross-check in `textbook_corpus_golden_test.cpp`; both
  re-fit the four Guo invariance rungs from per-group summary statistics and
  assert df-exact + chisq parity. The tests document that lavaan's
  `guo_mi_strong` reference is under-converged (magmaan reaches a strictly lower
  chisq at identical df, confirmed by three independent optimizers), so that
  rung is gated by df-exact + no-worse-than-oracle.
- Paper-corpus curation now lives in the ignored nested Git repository
  `external/paper_corpus`. That repository owns raw downloads, minimal derived
  lavaan-ready data/models, raw-to-derived validation, and magmaan-facing JSON
  exports. magmaan keeps copied snapshots under `tests/fixtures/paper_corpus/`;
  the C++ tests never read `external/paper_corpus` directly. The compact
  `zxqvn` empirical mediation/SEM project is the first promoted case: its
  export is a core complete-data ML point-estimate fixture, while the source
  script's clustered-SE option is catalogued outside the current core parity
  surface. The next promotion candidate is the richer `hwkem`
  measurement-invariance tutorial.
- Continuous ULS/GLS/WLS robust adapters reuse the shared weighted-moment
  sandwich/U-Gamma primitive with either supplied Gamma blocks or raw-data
  Gamma construction. ULS `robust.sem` SEs and Satorra-Bentler-family
  statistics are lavaan-backed for the non-fixed.x continuous LS fixtures;
  GLS/WLS robust paths have shape and scaling coverage because lavaan does not
  expose matching robust scaled-test targets for those estimators.
- GLS/WLS reporting follows lavaan's `2 * N * fmin` convention; ULS
  standard chi-square is pinned to lavaan's Browne residual NT statistic,
  including lavaan's `fixed.x` convention of zeroing the fixed exogenous
  rows/columns in the NT inverse weight. ULS robust scaled-test reporting
  follows lavaan's robust LS `2 * N * fmin` base statistic.
- Continuous ULS/GLS/WLS objectives can be evaluated explicitly at any
  supplied parameter vector via `estimate::evaluate_ls_objective()`, keeping
  discrepancy evaluation separate from chi-square reporting scale. RLS exposes
  a matching theta-based `nt::infer::rls_chi2()` overload that builds implied
  moments from the model structure before applying Browne's model-based
  residual quadratic.
- Distributionally weighted least squares (DLS) is available as an explicit
  weight-matrix builder over the existing moment-quadratic LS surface. The
  fixed-scalar builder mixes covariance-moment Gamma matrices between
  normal-theory GLS and empirical ADF/WLS endpoints, while
  `empirical_bayes_dls_mixing_scalar()` estimates a global reliability-style
  scalar from the observed `Gamma_ADF - Gamma_NT` departure and casewise
  fourth-moment product noise, following the empirical-Bayes weight-selection
  direction of Du and Wu (2024). `empirical_bayes_dls_weight()` then delegates
  to the same DLS builder, so modification-index and weighted-moment sandwich
  paths consume the result exactly like any supplied WLS weight. Local
  simulation checks live under `tests/checks/dls/`. The mixed/structured Gamma
  is inverted through a strict eigen-gated SPD inverse
  (`detail::symmetric_inverse_pd_gated`, `tol = 1e-10·max(1,λmax)`): a
  numerically rank-deficient fourth-moment Gamma returns an explicit
  `FitError::NumericIssue` with dim / numerical rank / rcond / smallest
  eigenvalue rather than a barely-PD inverse that would later trip the terminal
  stationarity audit.
- Model-implied fourth-order / structured-ADF weights are available as an
  explicit `estimate::frontier::structured_gamma_weight()` builder for
  complete-data, covariance-only pure CFA. The builder estimates independent
  factor/uniqueness fourth cumulants from raw data, builds the structured
  covariance-moment Gamma, inverts it, and returns an ordinary `gmm::Weight`
  for the existing WLS path. The raw structured Gamma matrix is also exposed so
  paper-local R code can inspect or regularize it before inversion. It is a
  paper-facing frontier helper, not a new estimator or default.
- Separable nonlinear least squares profiling exists for LS estimators where
  conditionally linear parameters can be profiled out.

### Ordinal and mixed categorical LS

- Threshold (`|`) and response-scale (`~*~`) parser/partable projection.
- Integer all-ordinal complete/listwise sample statistics.
- Pairwise polychoric correlations.
- Public enum-backed polychoric h-score API under `data::eval_polychoric_h_score()`
  for ML, WMA hard cap, smooth cap, and exponential cap experiments, returning
  score values, derivatives, and objective contributions while leaving the
  default lavaan-compatible ordinal sample-stat path unchanged.
- Experimental fixed-threshold all-ordinal bivariate h-weighted rho fitting
  under `data::fit_ordinal_pair_rho_h_weighted()`. It keeps thresholds fixed,
  accepts the predefined h-score options, returns rho/objective/score,
  convergence and bound diagnostics, adjusted counts, fitted probabilities,
  expected/residual/Pearson tables, and per-cell robust weights. ML and
  `WmaHardCap(k = Inf)` delegate to the existing ML rho path, preserving the
  lavaan-compatible limit; finite caps remain diagnostics/prototype machinery,
  not the default ordinal moment builder.
- Experimental pair-local all-ordinal bivariate joint h-weighted fitting under
  `data::fit_ordinal_pair_joint_h_weighted()`. It estimates both pair-local
  thresholds and rho by minimizing the h-score/minimum-disparity objective,
  reusing ordered-threshold and bounded-rho transforms. It returns pair-local
  thresholds, rho, objective/gradient diagnostics, adjusted counts, fitted
  probabilities, expected/residual/Pearson tables, and per-cell robust weights.
  ML and `WmaHardCap(k = Inf)` delegate to the existing joint ML kernel; finite
  caps are validated as experimental bivariate diagnostics, not SEM moment
  construction or calibrated robust inference.
- Experimental pair-local all-ordinal h-weighted influence diagnostics under
  `data::ordinal_pair_h_weighted_influence()`. Given integer bivariate counts,
  pair-local thresholds, and rho, it expands casewise estimating-function rows,
  returns h-score ratios/values/derivatives/diagnostic weights, analytic
  probability-Hessian bread, centered `h'(t)`-weighted meat rows, influence
  rows, score Gamma, and sandwich Gamma with `S'S / n` scaling. The WMA
  hard-cap derivative convention is pinned as `dh(t) = 1[t < k]`, so the exact
  kink uses derivative zero. This remains the robust bivariate Gamma primitive
  used for pair-level validation.
- Experimental pair-local all-ordinal density power divergence fitting under
  `data::fit_ordinal_pair_joint_dpd()`. It estimates pair-local thresholds and
  rho with DPD tuning `alpha`, delegates `alpha = 0` to joint ML, and returns
  probabilities, expected/residual/Pearson tables, and `p^alpha` attenuation
  weights. This is the main non-h-score bivariate comparator and is not yet
  wired into SEM moment/Gamma construction.
- Experimental all-ordinal SEM integration under
  `data::pairwise_ordinal_stats_h_weighted_from_integer_data()` now uses a
  shared-threshold composite h-score estimator: one threshold block per ordinal
  variable plus one polychoric correlation per pair are optimized jointly under
  the h-weighted objective. It rebuilds casewise moment influence/Gamma from
  the shared robust estimating equations and refreshes DWLS/WLS weights.
  Optional robust-R handling records the raw minimum eigenvalue and can either
  fail explicitly or repair low-eigen correlation matrices by ridge/shrinkage
  toward the identity, with the same transformation applied to the correlation
  influence columns before Gamma and weights are rebuilt.
- Experimental all-ordinal shared-threshold DPD stats are available under
  `data::pairwise_ordinal_stats_dpd_from_integer_data()`. The parameter and
  output contract matches the h-weighted SEM path, but the composite objective
  is density power divergence with tuning `alpha`; `alpha = 0` delegates to
  the default ML/lavaan-compatible stats path.
- Experimental all-ordinal shared-threshold Huberized Pearson-residual stats
  are available under
  `data::pairwise_ordinal_stats_huber_residual_from_integer_data()`. The
  clipped residual is the cell Pearson residual; hard Huber, pseudo-Huber,
  Tukey biweight, and no-clip options share the same moment/Gamma contract as
  h-weighted and DPD stats, with no-clip delegating to the default
  ML/lavaan-compatible stats path. Robust-R ridge/shrinkage repair applies the
  same correlation-column influence transformation before Gamma and weights
  are rebuilt.
- Experimental mixed continuous/ordinal Huberized Pearson-residual stats are
  available under `data::mixed_ordinal_stats_huber_residual_from_data()`.
  Hard Huber, pseudo-Huber, Tukey biweight, and no-clip residual options are
  exposed for ordinal-containing threshold/correlation/polyserial rows while
  continuous-only means, variances, and covariances remain ordinary moments.
  The path rebuilds casewise moment influence, Gamma/NACOV, and DWLS/WLS
  weights, including the single-ordinal case where threshold uncertainty enters
  only through polyserial links. Unit-level simulation checks cover the no-clip
  ML limit, contaminated polyserial tails, sparse ordinal margins, positive
  DWLS diagonals, finite Gamma conditioning, and DPD comparator stability. This
  remains a methods comparator rather than a lavaan-backed compatibility claim.
- Robust ordinal moment builders are experimental methods-developer surfaces,
  not changes to the default lavaan-compatible ordinal builders. SEM-facing
  robust builders use shared ordinal thresholds and rebuild moment influence,
  Gamma/NACOV, and DWLS/WLS weights under the robust equations. Pair-local
  threshold estimators remain diagnostics/prototypes rather than SEM moment
  constructors. Mixed robust builders currently robustify ordinal-only and
  continuous-ordinal links only; continuous marginal moments and
  continuous-continuous covariances remain ordinary unless a separate design
  reopens them. R exposes predefined robust methods only; arbitrary C++
  h-functions remain internal.
- Public all-ordinal pairwise ML kernel and `PairwiseOrdinalStats` wrapper for
  complete/listwise ordinal data, exposing pair labels, count/adjustment
  diagnostics, fitted fixed-threshold rho diagnostics, fitted expected/residual
  tables, missingness/repair diagnostics, casewise moment influence, Gamma,
  and minimum eigenvalue diagnostics while preserving the existing
  `OrdinalStats` moment/weight path.
- Bivariate ordinal observed-pair table kernel for explicit pairwise
  observed-data composite-likelihood work. `NaN` skips the pair and increments
  missing-pair diagnostics; finite observed values must be positive integer
  categories inside the declared level ranges. Observed-pair wrappers feed
  those counts into both fixed-threshold rho ML and pair-local joint
  threshold/rho ML while preserving `n_obs` and `n_missing`.
- Experimental joint bivariate ordinal ML kernel for one complete pairwise
  table, estimating pair-local nuisance thresholds and rho. This kernel backs
  the pair-local complete/listwise and observed-pair composite objective
  prototypes, but it is not wired into the lavaan-compatible SEM moment
  construction path.
- Public SEM-facing all-ordinal pairwise composite objective API under
  `estimate::pairwise_ordinal_composite_objective()`. It consumes
  `PairwiseOrdinalStats` plus SEM-implied shared thresholds and correlations,
  maps each pair to its bivariate threshold margins, exposes per-pair boundary
  diagnostics plus score/Gamma rows for the bivariate threshold/rho margin,
  and makes scaling/weighting explicit. The current reporting contract
  deliberately does not report chi-square or degrees of freedom until a
  calibrated composite-likelihood test is implemented.
- Complete/listwise all-ordinal pair-local joint composite prototype under
  `estimate::pairwise_ordinal_joint_composite_objective()`. It consumes the
  same `PairwiseOrdinalStats` diagnostics and options surface, refits every
  pair with the complete bivariate joint threshold/rho ML kernel, and returns
  pair-local thresholds, rho, adjusted counts, fitted counts, residuals,
  score contributions, score Gamma, boundary flags, and objective scaling.
  This is a saturated/reference composite target for future SEM fitting, not a
  lavaan-compatible DWLS moment builder and not a calibrated global chi-square
  test.
- Observed-pair all-ordinal joint composite prototype under
  `estimate::pairwise_ordinal_observed_joint_composite_objective()`. It takes
  ordinal data blocks with `NaN` missingness plus declared level counts, fits
  each bivariate observed-pair table independently with the joint threshold/rho
  ML kernel, preserves per-pair `n_obs`/`n_missing`, and rejects all-missing
  pairs or empty marginal categories. It exposes the same bivariate
  threshold/rho score contributions and score Gamma as the complete/listwise
  path. This is pairwise observed-data likelihood, not multivariate MAR
  ordinal FIML.
- Checked-in pairwise diagnostic fixture coverage under
  `tests/fixtures/pairwise/`: complete all-ordinal polychoric diagnostics,
  mixed pair labels and primitive ML helpers, complete/listwise joint
  composite diagnostics, and observed-pair composite missingness/count
  semantics.
- The SEM-facing ordinal moment path remains the lavaan-compatible
  shared-threshold path. Pair-local joint threshold/rho ML is reserved for
  diagnostics, robust pair experiments, and future composite likelihood rather
  than for constructing the current `OrdinalStats` moment vector.
- Shared-threshold multivariate missing ordinal modeling remains out of scope:
  the implemented missing-data ordinal path is pairwise observed-data
  composite likelihood, not multivariate MAR ordinal FIML.
- Muthen-style all-ordinal NACOV construction for thresholds plus
  polychorics.
- A first internal ordinal workspace split is in place: `OrdinalMoments` /
  `MixedOrdinalMoments` carry observed statistics and metadata only,
  `OrdinalGammaCache` records which Gamma/weight pieces are materialized, and
  `OrdinalWeightPlan` encodes the fit-only ULS/DWLS/WLS Gamma cost rules. The
  legacy `OrdinalStats` and `MixedOrdinalStats` structs remain the
  compatibility adapters that materialize full NACOV plus DWLS/WLS weights.
- The all-ordinal fit-only path can now consume `OrdinalMoments` plus an
  `OrdinalWeightPlan`: ULS builds identity weights and does not touch Gamma,
  while DWLS consumes only diagonal Gamma entries from `OrdinalGammaCache`.
  The cache-aware ULS/DWLS/WLS paths profile thresholds out of the optimizer
  through a joint affine threshold design `tau_b = c_b + H_b gamma` built from
  the prepared partable: free thresholds, fixed rows (kept as threshold
  residuals in the profiled full-moment objective), equality-label merges
  within and across groups (cross-group threshold invariance), and
  threshold-only linear equality constraints folded through a null-space
  basis. The threshold normal equations are joint across blocks with `n_b/N`
  sample weights, so multi-group profiled fitting is supported and one gamma
  coordinate may span groups; block `b`'s profiled thresholds then consume the
  stacked correlation residual over all blocks. Non-threshold equalities and
  constraints mixing threshold with non-threshold columns remain outside this
  cache-aware fit-only path. The cache-aware WLS path uses the full
  inverse-weight threshold/correlation blocks, applies the equivalent profiled
  affine threshold/correlation residual transform, and reconstructs thresholds
  with the profiled cross-block formula.
  Fit-plus-inference and inference-only all-ordinal plans can also reuse a
  full `OrdinalGammaCache` for robust DWLS/WLS reporting: DWLS materializes
  diagonal weights from the full Gamma when needed, WLS materializes the full
  inverse weight, and the robust result matches the legacy materialized
  `OrdinalStats` path. The default all-ordinal SNLLS entry point reuses the
  same joint threshold-design profiled objective for delta ULS/DWLS/WLS:
  thresholds are fixed out before the generic Golub-Pereyra classifier sees the
  problem (absorbed threshold-only linear rows are stripped from the reduced
  partable), conditionally linear covariance parameters are profiled, WLS uses
  the same affine full-weight transform, and the returned vector is
  reconstructed in the ordinary prepared ordinal partable coordinate. For theta, cache-aware
  bounded fitting uses the full standardized threshold/correlation moment
  objective with the requested ULS/DWLS/WLS cache materialization, and SNLLS
  profiles only threshold free parameters because the standardized covariance
  moments make the remaining covariance block nonlinear. A second full-threshold
  ordinal SNLLS entry point keeps the full threshold+correlation moment stack
  and marks threshold free parameters as Golub-Pereyra linear coordinates, so
  linear threshold constraints remain compatible without using the ordinal
  threshold-profiling map. `experiments/_archive/12-ordinal-threshold-constraints`
  validated the original split (free/shared-label cases through both paths,
  general linear threshold constraints only through the full bounded and
  full-threshold SNLLS paths); the threshold-profiled paths have since gained
  general threshold-only linear constraints and multi-group invariance, gated
  by unit parity tests against the full-threshold and legacy bounded fits and
  by lavaan oracle fixtures (0013 cross-group threshold invariance via explicit
  shared labels, 0014 threshold-only linear constraint) driven through both the
  legacy bounded and profiled SNLLS golden paths. The ordinal golden chisq
  gates now apply the lavaan `Σ(n_g−1)F̂_g` convention rescale at 5e-3 (see
  numerical-conventions exception 4 and the test ledger). `experiments/_archive/13-ordinal-construction-boundary`
  now compares the legacy eager constructor with
  `ordinal_workspace_from_integer_data()`: fit-only ULS returns
  `OrdinalMoments` without Gamma, fit-only DWLS returns `OrdinalMoments` plus
  the Gamma diagonal, and WLS/fit-plus-inference still fall back to full
  `OrdinalStats`/Gamma materialization. `experiments/_archive/11-ordinal-snlls-speed`
  now includes delta/theta timing rows plus construction-aware raw-to-SNLLS
  rows: the legacy row rebuilds `OrdinalStats`/moments/starts/cache inside the
  timed operation, while the lazy ULS/DWLS row rebuilds `OrdinalWorkspace`,
  starts, and the profiled SNLLS fit. Theta rows use the cache-aware bounded
  comparator and threshold-only SNLLS profiling, so the report keeps them
  separate from delta's threshold-plus-covariance profiling split. The same
  benchmark/report now includes mixed continuous/ordinal delta rows comparing
  materialized full bounded DWLS/WLS with materialized full-threshold SNLLS,
  plus raw-to-fit mixed bounded/SNLLS construction timings. Fit-only mixed
  DWLS can also build a lazy `MixedOrdinalWorkspace` with
  `MixedOrdinalMoments` plus the exact Gamma diagonal, so the speed pilot
  carries legacy-versus-lazy mixed DWLS raw-to-fit rows; mixed WLS still
  materializes the full Gamma/inverse-weight path.
- DWLS diagonal weights, full WLS weights, bounded ordinal LS fitting, and
  thin R wrappers for ordinal stats plus DWLS/WLS fits.
- The R ordinal data boundary exposes consolidated dispatchers:
  `data_ordinal_stats_from_raw(robust = ...)` for all-ordinal data and
  `data_mixed_ordinal_stats_from_raw(polyserial = ..., ordered_mask = ...)`
  for mixed continuous/ordinal data. Method-specific Rcpp names remain
  callable compatibility aliases rather than entries in the displayed
  `magmaan_core` data group.
- Current ordinal fixtures validate thresholds, polychoric `R`, `NACOV`,
  `WLS.V`, `WLS.VD`, free sets, point estimates, degrees of freedom, and
  chi-square statistics across representative single-group, multi-group,
  skewed, sparse, near-empty, equality-constrained, and multi-group loading
  equality cases.
- The implemented ordinal LS boundary supports both lavaan delta and theta
  parameterizations for all-ordinal and mixed continuous/ordinal DWLS/WLS point
  estimates, and the all-ordinal cache-aware/SNLLS path now covers theta for
  ULS/DWLS/WLS point estimation. Theta post-fit support uses
  parameterization-aware threshold and association Jacobians for robust ordinal
  reporting, modification indices, score tests, and standardized-solution
  reporting.
- Explicit post-fit robust ordinal reporting returns sandwich SEs plus
  Satorra-Bentler, mean/variance-adjusted, and scaled/shifted statistics from
  the threshold-plus-polychoric moment vector. The implementation now uses a
  shared weighted-moment sandwich/U-Gamma primitive that can be reused by other
  LS moment stacks with arbitrary block weights and NACOV matrices.
- `standardize_lv`/`standardize_all` and `compute_defined` accept
  ordinal/mixed-ordinal fits at both the C++ api and the Rcpp bindings. These
  parameterization-agnostic transforms operate over the
  *prepared* ordinal partable: `fit_ordinal_bounded` fixes the latent-response
  residual variances the delta constraint determines and compacts the free set,
  so the stored estimates/vcov live in that reduced space while `Model` carries
  the un-prepared structure. The api functions reconstruct the prepared
  structure on demand (an internal `prepared_structure` helper replaying
  `prepare_ordinal_partable`); the Rcpp bindings get it for free because
  `ctx_from_fit` parses the prepared partable. (Before this bridge the api-level
  guard-removal was dead: it fed the reduced theta into the un-prepared
  evaluator and aborted on the dimension mismatch.) `standardize_all` takes an
  `ordinal_delta_unit` flag: under the
  delta parameterization a categorical indicator's latent response is
  unit-variance, so its loading is standardized by the latent SD only (σ_rr = 1)
  rather than the assembled `λ²ψ + 1`. This is applied in both the plain-CFA
  `Lambda` slot and the all-y RAM `Beta` slot, so a mixed SEM's endogenous-factor
  loadings and structural paths standardize to lavaan `std.all`. The bindings
  read the parameterization from the partable attribute and the api from the
  fit's `EstimatorSpec`. Mixed-ordinal stats construction also no longer aborts
  DWLS when the full-WLS NACOV is singular (common at small N with many
  indicators): the inverse is non-fatal, `W_wls` is left empty, and DWLS / the
  robust sandwich proceed on the diagonal weight / NACOV.
- Ordinal/mixed-ordinal factor scores are a separate categorical estimator, not
  a guard flip over the continuous regression/Bartlett predictor. The measures
  layer exposes `factor_scores_ordinal()` and `factor_scores_mixed_ordinal()`;
  `api::factor_scores()` dispatches ordinal and mixed fits there; and the R
  `factor_scores()` wrapper defaults categorical fits to EBM while accepting
  `method = "EBM"`, `"ML"`, or `"EAP"`. EBM/ML score each unique complete
  response pattern by damped Newton with analytic ordinal interval-probability
  gradient/Hessian terms; EAP is supported for one-factor models through the
  vendored QUADPACK infinite-interval integrator. The same EAP quadrature also
  exposes one-factor posterior variance/SE, sample-moment PRMSE, and the
  direct concrete ordinal reliability through `factor_score_precision_*` /
  `api::factor_score_precision()` and the R `factor_score_precision()` helper.
  The current categorical scope is diagonal residual `Theta`; multi-factor EAP
  and correlated-residual orthant
  probabilities remain deferred. Checked-in lavaan parity is gated by
  `tests/golden/ordinal_golden_test.cpp` ("ordinal/mixed factor scores (EBM/ML)
  match lavaan", 5e-4) over the `fits.DWLS.fscores` oracle: single-group EBM
  (all-ordinal and mixed) and mixed ML. All-ordinal ML (unbounded mode on
  extreme patterns) and EAP (no categorical `lavPredict()` oracle) are not
  lavaan-gated (EAP stays self-checked). The EAP precision surface additionally
  carries a Monte-Carlo ground-truth gate (`tests/unit/api_sem_test.cpp`,
  "ordinal EAP factor-score precision tracks Monte-Carlo PRMSE"): on a
  five-indicator three-category one-factor model simulated with retained latent
  `Z` and fit under `std.lv`, the reported `pooled_prmse` matches the realized
  `corr(Z, E[Z|Y])²`, the mean posterior variance matches the realized EAP MSE,
  and the concrete reliability reduces exactly to `1 - mean Var(Z|Y)` under unit
  latent variance (gaps ~1e-3 at n=8000). Multi-group categorical EBM is correct
  and is validated transitively: lavaan's own multi-group categorical
  `lavPredict()` returns a non-stationary point for non-reference groups (it is
  not a usable oracle there), so the same golden instead checks that each
  group's multi-group EBM equals an independent single-group fit on that group's
  data (~3e-8) for the unconstrained two-group fixture, and single-group EBM is
  lavaan-gated.
- All-ordinal DWLS/WLS fit measures are exposed through
  `estimate::fit_measures_ordinal()` and `api::fit_measures()`: CFI/TLI/RMSEA
  use the categorical independence baseline over the polychoric moment stack,
  with WLS minimizing threshold nuisance residuals under the full weight
  matrix, and ordinal SRMR uses the lavaan correlation-metric denominator that
  includes zero diagonal residuals. The bfi ordinal parity fixture gates DWLS
  and WLS CFI/TLI/RMSEA/SRMR against lavaan.
- Weighted-χ² reducer formulas are shared across eigenvalue and trace-summary
  callers: Satorra-Bentler, mean/variance-adjusted, and scaled/shifted tests
  can consume either the UΓ spectrum or `(Σλ, Σλ²)` when a low-rank trick has
  already computed the traces.
- A first mixed continuous/ordinal path builds lavaan-ordered thresholds,
  continuous means/variances, polychoric/polyserial/covariance moments,
  NACOV/DWLS/WLS weights, and DWLS/WLS delta/theta fits. Mixed delta SNLLS now
  has a materialized-stats full-threshold entry point,
  `estimate::fit_mixed_ordinal_snlls_full_thresholds()`, that profiles the
  conditionally linear threshold, mean, variance, and covariance parameters
  through the generic Golub-Pereyra split and matches bounded mixed DWLS/WLS on
  a focused unit test. The fit-only mixed DWLS workspace path now avoids full
  Gamma/WLS materialization by carrying `MixedOrdinalMoments` plus the Gamma
  diagonal into bounded and full-threshold SNLLS fits. Mixed full-Gamma cache
  reuse also covers robust DWLS/WLS reporting through a mixed-moments
  overload. The mixed Gamma construction mirrors lavaan's muthen1984
  estimating-equation sandwich exactly (stage-1 mu/var ML scores with
  per-variable bread blocks, pair-ML scores including mu/var coupling channels
  for polyserial and continuous-continuous pairs, and the delta-rule
  correlation-to-covariance transform applied to post-sandwich variance
  influence; see the design doc's "Mixed Gamma Construction"), so the mixed
  goldens gate NACOV/weights at 1e-6, point estimates at the all-ordinal
  theta 1e-5 / chisq 5e-3 contract, and the robust scaled-test fields at
  all-ordinal tightness — at lavaan's theta-hat and at magmaan's own. The
  same construction backs the lazy fit-only DWLS diagonal and the
  Huber-residual single-ordinal rebuild (no-clip reproduces the ML Gamma
  exactly). Mixed theta SNLLS runs through the same full-threshold stack:
  under theta only thresholds stay Golub-Pereyra linear (the standardized
  covariance moments make the rest nonlinear), gated against the bounded
  theta fit on a well-identified three-category design — binary-indicator
  theta models carry a near-flat lambda/psi ridge where optimizer endpoints
  are arbitrary, so theta parity is only meaningful on identified designs.
  Mixed WLS and fit-plus-inference workspaces are lazy about weights: the
  builder carries moments plus full Gamma into the cache and defers the
  O(m³) WLS inverse (and DWLS weight extraction) to the
  `ordinal_gamma_cache_ensure_*` helpers at first use. Threshold-profiled
  mixed objectives remain a later slice; reduced-Gamma robust products sit in
  the speculative backlog. The lavaan-backed fixtures include a
  complete/listwise sparse 4-category boundary case.
- Covariance shrinkage is available under `data::frontier` for both continuous
  `SampleStats` and mixed continuous/ordinal `MixedOrdinalStats`. Mixed
  shrinkage leaves thresholds and continuous means in place, transforms the
  lower-triangle association/covariance block, propagates the moment
  transformation through `NACOV`, and rebuilds DWLS/WLS weights so C++ and R
  consume the same shrunk moment stack. The old `magmaan/data/shrinkage.hpp`
  include path remains a forwarding shim.
- Public complete-data polyserial pair kernel for mixed continuous/ordinal
  work, exposing fixed-threshold rho ML, likelihood, casewise threshold/rho
  scores, and pairwise score Gamma. The mixed sample-stat builder now reuses
  this kernel for polyserial associations.
- Experimental fixed-marginal polyserial DPD is available under
  `data::fit_polyserial_pair_rho_dpd()` and
  `data::polyserial_pair_dpd_scores()`. It keeps the shared ordinal thresholds
  and standardized continuous marginal fixed, estimates only the polyserial
  association, delegates `alpha = 0` to the ML kernel, and returns DPD
  attenuation weights plus score/Gamma/bread diagnostics.
- Experimental SEM-facing mixed polyserial DPD stats are available under
  `data::mixed_ordinal_stats_polyserial_dpd_from_data()`. The builder preserves
  the existing mixed moment order, shared ordinal thresholds, continuous
  means/variances, ordinal-ordinal polychorics, and continuous-continuous
  covariance moments, while replacing only continuous-ordinal association
  equations with fixed-marginal DPD and rebuilding NACOV/DWLS/WLS weights from
  the mixed casewise influence rows.
- Experimental pair-local full DPD polyserial fitting is available under
  `data::fit_polyserial_pair_joint_dpd()`. It jointly estimates continuous
  mean/scale, ordinal thresholds, and rho with DPD tuning `alpha`, and returns
  probabilities, joint densities, and `f(x, y)^alpha` attenuation weights. DPD
  here means density power divergence and is not part of robcat parity.
- Pair-local full polyserial DPD remains a bivariate diagnostic only and is not
  used to construct `MixedOrdinalStats`; SEM-facing robust mixed moments use
  the shared-marginal fixed-threshold contract instead.
- Public complete-data mixed pair helpers also expose continuous-continuous
  normal pair likelihood/diagnostics, casewise mean/variance/covariance scores,
  score Gamma, and labels for the exact threshold, negative-mean, variance,
  and lower-triangle pair order used by `MixedOrdinalStats`.
- The continuous normal pair likelihood is currently a mixed-pair primitive
  and benchmark against complete-data ML/FIML, not a supported standalone
  normal-data pairwise SEM estimator.
- Ordinal and mixed delta DWLS/WLS expose fixed-parameter modification indices
  and equality-release score tests over the same threshold/correlation moment
  vectors and weights used by fitting.
- Mixed continuous/ordinal DWLS/WLS fit-measures are exposed through the same
  `api::fit_measures()` surface as all-ordinal fits. The mixed independence
  baseline profiles the marginal threshold/mean/variance block under the fitted
  DWLS/WLS weight before testing the zero-association model, and SRMR is
  computed from standardized mixed association residuals; both are fixture-gated
  against lavaan's mixed ordinal CFI/TLI/RMSEA/SRMR fields.
- Ordinal and mixed categorical entry points validate block counts, threshold
  metadata, ordered masks, moment/weight/NACOV dimensions, finite values,
  positive `n_obs`, and positive NACOV diagonals before fitting or robust
  reporting.

### R bindings and public namespace transition

- Exploratory R bindings cover lavaanify, fitting, sample-stat bundles, robust
  inference, fit measures, model implied moments, LS estimators, SNLLS, Ceres
  paths when enabled, and data-frame-to-model sample statistics.
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
  `tests/fixtures/composite/`. The public-surface golden
  (`tests/golden/composite_golden_test.cpp`) fits through `fit_ml_fcsem()` and
  checks objective/chi-square, implied covariance in lavaan observed-variable
  order, df/npar, fit measures, reported raw rows, SEs, and `std.lv`/`std.all`
  rows against lavaan's native `<~` output. Native W/T matrix construction
  remains covered as evaluator/unit-test machinery rather than as a public
  fixture contract; the R bridge is still a methods-developer frontier surface,
  not a lavaan replacement interface.
- `magmaan(model, data, estimator, groups)` is the high-level estimate-only
  R convenience. It composes `model_spec()`, data-frame sample-stat/raw-data
  construction, and the matching point-estimation wrapper for complete-data
  ML/ULS/GLS/WLS, FIML, ML2S, and ordinal/mixed DWLS/WLS where the lower-level
  inputs are available. For FIML and ML2S syntax calls it auto-enables a mean
  structure (and rebuilds syntax-backed no-mean specs) because the raw-data
  missing-data paths are mean-based; explicit `meanstructure = FALSE` errors
  early. Lavaan-style `se = "none"` and `test = "none"` are accepted as explicit
  point-estimate-only shortcuts; other values error and point users to explicit
  post-fit inference calls. It returns a `magmaan_fit` list: the raw primitive
  fit fields plus the source `model_spec`, syntax, estimator options,
  ordered-variable metadata, parameterization, and grouping metadata. Its print
  method reports only point-fit status and directs users to explicit post-fit
  primitives.
- `compute_defined(model, fit, vcov)` exposes C++ defined-parameter evaluation
  for `:=` rows through R. It keeps covariance selection explicit, supports
  chained definitions, and resolves `.pN.` plabel references using the fitted
  lavaanified model.
- Friendly R post-fit wrappers expose routine inspection without hiding
  statistical choices: `standardized(fit, vcov, type)` requires an explicit
  covariance matrix, `stats::residuals(fit, standardized)` wraps raw or
  standardized residuals including the lavaan-style continuous residual
  z-statistics, `factor_scores(fit, data, method)` requires complete raw data
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
  calls outside `magmaan()`.
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

### Local build workflow

- The normal C++ edit loop is the `fast` preset and `just test-fast` / `just
  test`: Debug, no sanitizers, Ceres off, with ccache and mold enabled.
- The `dev` preset is the sanitizer validation loop: Debug plus ASan/UBSan.
  Use `just test-dev` before handing back risky core changes, and prefer
  targeted `ctest --preset dev -R ...` while narrowing failures.
- The C++ doctest suite is split into labeled executables (`smoke`, `spec`,
  `estimate`, `inference`, `ordinal`, `api`, `parity`, and `robcat`) behind
  the aggregate `magmaan_tests` target. This keeps
  `cmake --build --preset <p> --target magmaan_tests` working while allowing
  narrower relinks and `ctest -L`.
- R bindings are intentionally outside the default C++ loop. `just r-install`
  links the optimized non-Ceres core, `just r-install-fast` is for interactive
  wrapper work, and `just r-install-ceres` is the explicit Ceres-enabled R
  path.

#### Build-loop timings

Fast-loop snapshot refreshed on 2026-05-19 (commit `97dd197`), on a 13th-gen
i7-1355U (12 threads), clang 19.1.7, with ccache and mold enabled.
Wall-clock; approximate orientation, not a benchmark. Rows marked
"not remeasured" are older orientation values retained until the corresponding
loop is refreshed.

| Loop step (`just` recipe)                   | Time   | Notes |
|---------------------------------------------|--------|-------|
| no-op `fast` build (`just fast`)            | 0.03 s | nothing changed |
| touched core TU, `fast`                     | 1.2 s  | not remeasured; mtime only, ccache hit, relink `libmagmaan` + test exes |
| edited core TU, `fast`                      | 7.5 s  | not remeasured; content change, cold clang compile of one core TU |
| touched test TU, `fast`                     | 0.5 s  | not remeasured; mtime only, ccache hit, relink one test exe |
| edited test TU, `fast`                      | 5.6 s  | not remeasured; content change, cold compile of one test TU |
| C++ suite (`just test-fast`)                | 81.5 s | 458 tests; no-op build + `ctest` |
| C++ suite minus parity (`just test-quick`)  | 37.4 s | 454 tests; excludes the 4 parity tests |
| sanitizer suite (`just test-dev`)           | 287 s  | not remeasured; old ASan/UBSan orientation value |
| R install, opt (`just r-install`)           | 12 s   | not remeasured; warm core; rebuilds the 5 R-glue TUs + link |
| R install, fast (`just r-install-fast`)     | 15 s   | not remeasured; warm core |
| R install, Ceres (`just r-install-ceres`)   | 123 s  | not remeasured; included a one-time post-refactor rebuild of the Ceres core; ~15 s once warm |

The everyday inner loop (edit a core file, `just test-quick`) is comfortably
under a minute on the measured fast tree; the sanitizer suite and the Ceres R
path are minutes-scale and run deliberately rather than on every change.

### Argument-minimality sweep

One core architecture rule is that functions should take exactly the data they
need, and no more. The current C++ core mostly follows this:

- Fit results are plain `Estimates` and carry no back-pointer to the model.
- Continuous fitters take the estimable structure, matrix representation,
  sample/raw data, starts, discrepancy, and optimizer. They do not take
  parameter names.
- Information, covariance, SE, chi-square, degrees of freedom, Wald/z tests,
  robust U-Gamma reducers, both casewise-Zc and caller-supplied-Gamma robust
  test reductions, both-bread robust test trace-moment reducers, the
  materialized empirical-Gamma reference reducer, and fit measures are exposed
  as separate primitives rather than as methods on one fitted-object bundle.
- Satorra-Bentler-family reducers take scalar test statistics, degrees of
  freedom, and eigenvalues or trace summaries rather than fit objects.
- The Satorra-2000 C++ helper takes the model, reparameterization matrices,
  raw-data pieces, chi-square values, degrees of freedom, and an explicit
  `A.method` option (`exact` default, `delta` for lavaan-style
  moment-Jacobian column-space restrictions). Its empirical-Gamma computation
  can run in the default streaming casewise-reduced mode or in a materialized
  full-Gamma reference mode for diagnostics and benchmarks.

The remaining pressure point is the R boundary. Several exported R wrappers
currently accept the transparent fit list for convenience and then unpack the
partable, sample statistics, and estimates internally. That is acceptable as an
interactive convenience while the R package is exploratory, but it is not the
final architectural ideal. Thin R wrappers should gradually mirror the C++
primitive signatures; fit-list helpers can remain as explicit convenience
adapters layered on top.

### Robust-test naming and compatibility

The core robust-test API should use statistical names for the object being
computed, not historical SEM estimator labels. The preferred core surface is:

- weighted chi-square mixture tests from explicit eigenvalues or trace
  summaries;
- moment reductions of that same mixture: mean-scaled, mean/variance-adjusted,
  and scaled/shifted;
- likelihood-ratio / nested-model helpers named for the statistical contract
  they implement, for example exact restriction-map LRT rather than a default
  public name centred on "Satorra-2000";
- robust SEs as named sandwich constructions.

Historical labels remain important for lavaan/Mplus parity, but they belong in
`compat::lavaan` or similarly explicit compatibility wrappers. This includes
estimator shortcuts such as `MLM`, `MLMV`, `MLMVS`, and `MLR`, lavaan test
labels such as `satorra.bentler`, `scaled.shifted`,
`mean.var.adjusted`, `yuan.bentler`, and `yuan.bentler.mplus`, and legacy
nested-test formulas such as Satorra-Bentler 2001/2010. The compatibility
wrappers should bind each label to the exact lavaan/Mplus bundle it denotes;
they should not expose a combinatorial menu that lets callers freely pair a
test label with unrelated bread, gamma, vcov, or scaling choices.

`yuan.bentler.mplus` is especially a compatibility target rather than a core
weighted-mixture reducer. Lavaan's MLR default computes a scalar trace
difference, H1 minus H0, relying on Mplus-style approximations such as
`A0 ~= Delta' A1 Delta` and the analogous first-order matrix. The plain
Yuan-Bentler formula instead constructs the H1 residual-space trace directly.
Both should be documented when exposed, but neither should force historical
names into the core naming scheme.

The R boundary now stages this policy with `robust_nested_lrt()` as the
friendly statistical name for nested robust likelihood-ratio work. Its default
`method = "restriction_map"` names the exact restriction-map contract, while
`"lavaan_sb2001"` and `"lavaan_sb2010"` are explicit compatibility methods.
The restriction-map route also exposes `computation = "streaming"` versus
`"materialized"` so paper-local benchmarks can compare the algebra without
using lavaan as the timing denominator. When both fits are FIML,
`robust_nested_lrt()` dispatches only to `method = "restriction_map"` and uses
the retained FIML `raw_data` rather than a caller-supplied complete-data
argument; mixed FIML/complete-data pairs and the lavaan SB2001/SB2010
compatibility methods are rejected for this boundary.
The older `nestedTest()` spelling and lavaan labels (`"satorra.2000"`,
`"satorra.bentler.2001"`, `"satorra.bentler.2010"`) remain as compatibility
aliases during exploration. Low-level `magmaan_core` exposes both
`robust_nested_lrt_restriction_map` and `compat_lavaan_nested_lrt_*` aliases
over the same C++ primitives so methods scripts can choose the intended
surface directly.

### Testing and validation

Validation has three deliberately separate surfaces:

- **Corpus golden tests** — breadth. 26 small synthetic models in
  `tests/fixtures/corpus.json` exercise every parser, lavaanify, matrix, fit,
  and inference stage against checked-in lavaan fixtures. Oracle:
  `tests/tools/regen_oracle.R`.
- **Parity golden tests** — depth. `tests/golden/lavaan_parity_golden_test.cpp`
  gates magmaan against lavaan on the real-data benchmark cases
  (HolzingerSwineford1939, PoliticalDemocracy, Demo.growth, bfi, Mplus ex5.1):
  real sample sizes, real conditioning, real SEs and fit measures, and the
  raw-data ingestion path. The FIML tranche includes both a quick two-factor
  bfi missing-data sentinel with robust MLR post-fit reporting and a full
  25-item five-factor bfi missing-data convergence/global-fit gate.
  Self-contained fixtures live in
  `tests/fixtures/parity/`; oracle: `tests/tools/regen_parity_fixtures.R`.
  The same parity executable also includes the Mplus SEM corpus golden,
  generated from local `corpus/textbook-corpus/raw/mplus_sem` into
  `tests/fixtures/mplus_sem/`.
- **Benchmarks** — `benchmarks/` fits *live lavaan* on every run and gates
  timing, not CI correctness. Active advisory cases include complete-data ML,
  controlled-missingness FIML, and continuous ULS/GLS smoke paths. The harness
  is R-dependent and advisory; the parity layer is the bridge that freezes its
  correctness checks into the gated C++ suite.

The suite builds as eight doctest executables — `magmaan_test_{smoke, spec,
estimate, inference, ordinal, api, parity, robcat}` — so areas build and run
independently (`ctest -L parity`). CI never invokes R; fixture regeneration is
a manual developer step. Property and boundary tests are expected to catch
structural mistakes early, before they surface as hard-to-debug parity
failures.

## Current Boundaries

- Complete-data and FIML support are fixture-backed only for the documented
  estimator/model slices.
- The complete-data ML parity layer includes the `lavaan::growth()` defaults
  for linear latent growth models: observed-variable intercepts are fixed to
  zero, latent growth-factor means are freely estimated, and the relevant
  growth-factor covariance is auto-added. The R boundary exposes this through
  `model_spec(..., model_type = "growth")`.
- FIML with missing observed exogenous variables under `fixed.x = TRUE`
  remains unsupported.
- FIML score tests and modification indices use observed-pattern gradients
  plus a finite-difference observed information matrix; fixture parity is
  against lavaan's observed-information score output for the covered fixed-row
  and equality-release cases.
- Delta and theta parameterizations are supported for all-ordinal and mixed
  continuous/ordinal DWLS/WLS point estimates. Ordinal robust reporting,
  modification indices, score tests, and standardized reporting use the fitted
  parameterization; remaining lavaan parity is fixture-backed for the covered
  ordinal slices and smoke-tested where lavaan fixture coverage is not yet
  available.
- WLS ordinal point estimates and standard chi-square are lavaan-backed.
  Robust WLS scaled-test reporting remains shape-only because lavaan rejects
  Satorra-Bentler-family `test=` requests with `estimator = "WLS"` for the
  current categorical fixtures; DWLS robust reporting remains lavaan-backed.
- Continuous LS robust ULS reporting is lavaan-backed for non-fixed.x cases;
  fixed.x robust LS still follows lavaan's conditional exogenous bookkeeping
  and is not part of the generic continuous adapter coverage.
- Mixed categorical NACOV/weight, fit, fit-measure, and robust-reporting parity
  remains looser than the all-ordinal path, but now includes a sparse listwise
  4-category boundary fixture. Pairwise polyserial ML, fixed-marginal
  polyserial DPD, pair-local full DPD polyserial diagnostics, and
  continuous-normal score/Gamma primitives are exposed for follow-on inference
  work. The SEM-facing mixed polyserial DPD path robustifies only
  continuous-ordinal associations; robust continuous marginals and
  continuous-continuous covariances remain outside that contract.
- Observed-pair ordinal table kernels are pair-level primitives only. Missing
  ordinal SEM estimation is limited to the explicit pairwise observed-data
  composite prototype; shared-threshold multivariate missing ordinal modeling
  remains unsupported and requires a separate design note before
  implementation.
- Browne residual ADF is complete-data only.
- Score-test EPC is raw, unstandardized EPC only; standardized EPC and absent
  row-generation helpers are not yet part of the public contract.
- Nonlinear *equality* constraints are supported for ML, complete-data LS, and
  FIML through NLopt SLSQP or the IPOPT backend (Jacobian-projected vcov/df),
  including in combination with linear equality constraints in the same model
  — but not for ordinal or the separable SNLLS path; those combinations fail
  explicitly.
- Inequality constraints (`<` / `>`) and active-bound inference remain
  unsupported: inequality-constrained estimation needs boundary
  (chi-bar-squared) asymptotics magmaan does not implement. They fail with an
  explicit early error rather than silently reporting ordinary χ²/SE theory.

## Design Invariants

- No global mutable state.
- No in-place mutation of shared public structures; entry points operate on
  values or local copies and return values/errors.
- No groups == one group. Single-group models use the same block/group shape
  as multi-group models.
- Unsupported statistical combinations fail with an explicit error rather than
  silently approximating nearby lavaan-compatible behavior.
- Lavaan-shaped partables are boundary formats for oracle comparison,
  interchange, and compatibility projection. Core work should prefer the model
  triple plus explicit derived structures.
- Extension points remain free function templates over structural interfaces,
  not virtual hot-path interfaces or `concept` constraints.
- Parser behavior follows `docs/grammar/grammar.ebnf`; if parser code and the
  EBNF disagree, the parser is wrong.
- Function signatures should be argument-minimal. A computation should receive
  the smallest structure or primitive values it needs: no names for numeric
  fitters, no fitted object for degrees of freedom, no full fit list where a
  parameter vector or sample-size vector is enough. Convenience adapters may
  unpack larger objects at the boundary, but core APIs should not depend on
  those bundles.

## Planning Documents

Use this file to understand the current state and contracts before structural
changes. Remaining work lives in the backlog:

- [docs/backlog/todo.md](../backlog/todo.md) — the active SEM/parser/estimation
  backlog (open work only; completed items are folded into this roadmap and the
  test ledger).
- [docs/backlog/simulation.md](../backlog/simulation.md) — the `magmaan::sim`
  work queue and decision log.
- [docs/backlog/speculative.md](../backlog/speculative.md) — may-never-build
  ideas kept findable, with the cheaper alternative and the build-if trigger.
- [docs/backlog/newsom-corpus-failures.md](../backlog/newsom-corpus-failures.md)
  — open optimizer/evaluator-accuracy cases surfaced by the Newsom corpus.

Validation and design context:

- [docs/validation/test_ledger.md](../validation/test_ledger.md) — the
  test-protection map by subsystem plus the regression notes for fixed
  cross-subsystem bugs.
- [docs/validation/lavaan_tutorial_parity.md](../validation/lavaan_tutorial_parity.md)
  — section-by-section audit of magmaan against the lavaan tutorial.
- [docs/design/documentation_proposal.md](../design/documentation_proposal.md)
  — the proposed Quarto manual split, documentation vocabulary, and first
  documentation milestones.
