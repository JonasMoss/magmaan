# magmaan — roadmap

## Architectural cleanup — magmaan identity and compositional API

The library is now named **magmaan**. The rename is intentionally breaking:
public includes are under `include/magmaan/`, the C++ namespace is `magmaan`,
the CMake target is `magmaan::magmaan`, and the exploratory R package loads
`magmaan`.

The public API doctrine is:

- **Explicit composition over fitted mega-objects.** Core functions take the
  smallest value inputs they need: model structure, names, starts, matrix
  representation, sample stats, estimates, information matrices, or raw data.
  A fitted R list may exist as a convenience container, but it is not the
  conceptual API.
- **Immutable values first.** Shared model/data objects are not mutated in
  place. When fixed.x or derived quantities must be resolved, functions work on
  local copies and return values/errors.
- **No hot-path OOP.** Extension points remain concepts and free-function
  templates (`Discrepancy`, `Optimizer`, SE/test/index primitives), not virtual
  inheritance.
- **Lavaan-shaped partables are a boundary format.** The lavaan data-frame
  shape is kept for oracle comparison, R interchange, and compatibility
  projection. Core fitting/inference should operate on the model triple
  (`LatentStructure`, `LatentNames`, `Starts`) plus explicit derived structs
  such as `MatrixRep` and `SampleStats`.
- **R wrappers mirror namespaces, not package branding.** R exports use names
  such as `fit_fit`, `infer_df_stat`, `model_matrix_rep`,
  `lavaan_lavaanify`, and `data_sample_stats_from_raw`; they do not use
  `magmaan_` prefixes.

Current cleanup status:

- [x] Full package identity rename to `magmaan` for C++
      namespace, includes, CMake target/options, R package metadata, tests,
      fixtures, and docs.
- [x] R exported functions renamed away from package prefixes.
- [x] The R `infer_chi2_stat`, `infer_df_stat`, and `infer_baseline` wrappers
      now take explicit primitive inputs instead of requiring a fit object.
- [x] Add transition headers for the target namespace map. The old public
      includes still compile; new code can start including `spec/`,
      `lavaan/`, `estimate/`, `optim/`, `nt/`, `gls/`, and `data/`.
- [x] Convert C++ tests and R binding internals to the target public headers
      and namespaces, with one focused compatibility test keeping old
      `fit/` and `partable/` includes alive for the transition window.
- [ ] Mechanically move implementation files and primary definitions to the
      target namespaces. The transition headers are currently aliases only.
- [ ] Continue replacing R fit-object-only wrappers with primitive signatures,
      keeping fit-list extraction helpers only as boundary conveniences.

Target namespace map:

- `magmaan::spec` owns model specification and construction:
  `LatentStructure`, `LatentNames`, `Starts`, `LavaanifyOptions`,
  `lavaanify`, equality-group computation, and linear-constraint resolution.
- `magmaan::lavaan` owns lavaan-shaped boundary projections:
  `LavaanParTable`, `ParsedLavaanParTable`, `to_lavaan_partable`, and
  `from_lavaan_partable`. Lavaan-shaped tables are for oracle comparison, R
  interchange, and user-facing compatibility views; they are not the core
  carrier.
- `magmaan::model` stays responsible for `MatrixRep`, `ModelEvaluator`,
  implied moments, LISREL matrix ids, and structural cell layout.
- `magmaan::estimate` owns estimation orchestration: `Estimates`, `fit`,
  `fit_bounded`, starts, fixed.x resolution, equality-reparameterization, and
  parameter bounds.
- `magmaan::optim` owns optimizer concepts and implementations:
  `Optimizer`, `BoundedOptimizer`, `LbfgsOptimizer`, `LbfgsBOptimizer`, and
  optional Ceres adapters.
- `magmaan::nt` owns complete-data normal-theory reporting and inference:
  `nt::ml` for ML discrepancy/stat helpers, `nt::infer` for information,
  vcov, SE, df, z/Wald/LR primitives, `nt::robust` for UΓ/SB/YB/scaled tests,
  robust SE, Satorra 2000, Browne residual tests, `nt::measures` for fit
  indices and information criteria, and `nt::standardize` for standardized
  solutions.
- `magmaan::gls` owns GLS-family discrepancies in the statistical sense:
  ULS now, GLS/WLS/DWLS-adjacent pieces later. Optimizers stay in `optim`.
- `magmaan::data` owns `SampleStats`, `RawData`, and raw-data/sample-stat
  construction.

R compatibility roadmap:

- Add lavaan partable comparison helpers. The R layer should be able to return
  a lavaan-shaped table on demand from `LatentStructure + LatentNames + Starts`
  plus optional `Estimates`, then compare against `lavaan::parTable(fit)` by
  `(lhs, op, rhs, group, free/plabel)` rather than raw row position. The helper
  should support including/excluding mean-structure rows and fixed rows.
- Add a lavaan-like model/data preparation helper. It should accept model
  syntax, a data frame, optional `group`, lavaanify options, and selected
  observed variables, then return the parsed model triple/projection,
  `SampleStats`, lavaan-compatible group labels, `MatrixRep`, and raw
  per-group matrices for robust paths. It should prepare canonical inputs for
  methods work without becoming a `cfa()` clone. C++ continues to consume
  explicit matrices, raw blocks, names, and sample stats rather than owning a
  data-frame abstraction.

---

# Full-data normal-theory ML feature roadmap

Status: P0–P8 done (parser → partable → matrix rep → evaluator → ML/LBFGS → expected/observed SEs,
χ², df) plus post-P8 extensions (multi-group — configural *and* measurement invariance via shared
labels / `c(...)` per-group modifiers / cross-group equality constraints, with the grouping variable
+ labels stored in the partable; robust SEs / UΓ / Satorra-Bentler family — multi-group for the
cov-only `Expected`-bread path; mean structure `~1`; `:=` defined parameters; equality constraints —
simple (`a == b`, shared labels) *and* general linear (`a == 2*b+c`, `b2+b3 == 1.5`) via an affine
reparam `θ = θ_0 + Kα`; identification conventions `std.lv` / `effect.coding` (`LavaanifyOptions::std_lv`
/ `::effect_coding`; marker stays default); standardized `std.lv`/`std.all`; CFI/TLI/RMSEA; LR/Wald/z
tests; Browne residual tests). Golden-tested against
lavaan (`tests/golden/`). This file tracks what's left to make **full-data (complete-data)
normal-theory ML estimation feature-complete vs. lavaan's `cfa()`/`sem()`/`lavaan()` with
`estimator = "ML"`.** Out of scope (other tracks): FIML/missing data, ordinal/DWLS/polychoric,
bootstrap SE, Bayesian, multilevel. Design invariants (no mutable global state, no in-place mutation
of shared structures, no-groups == one-group) are spelled out below.

(The original architecture/phase plan is `~/.claude/plans/hello-i-want-to-eager-dream.md`; the
detailed P8 plan is `docs/p8_inference.md`. "v0.5: constrained optimizer" there ≙ **G1/P9** below.)

## Tier 1 — core estimation

- **G1 / P9 — constraint enforcement in `fit()`** — ✅ **done** (phase 1: linear/simple equality —
  shared labels, explicit `a == b`, cross-group equality for measurement invariance; see the P9
  section below). `fit()` reparameterizes `θ = K·α` (`include/magmaan/fit/constraints.hpp` +
  `src/fit/constraints.cpp`); `src/fit/inference.cpp` projects the info through `K` and adjusts `df`;
  `src/fit/robust.cpp`'s `build_u_factor` / `robust_se` likewise use `Δ·K` (so the metric-invariance
  multi-group SB / robust SE run). Tested (`tests/unit/constraints_test.cpp`,
  `tests/unit/robust_test.cpp`); wired through to R. Demos:
  `r-package/examples/constraints_and_satorra_bentler.R`, `holzinger_invariance.R`,
  `holzinger_2group_satorra_bentler.R`. Remaining: phase 2 (general linear `a == 2*b + c`) and phase 3
  (nonlinear / inequality) — they currently error with a clear message.
- **G2 — `c(...)` per-group modifiers + group identity in the partable** — ✅ **done**.
  `select_group_atom()` applies the g-th `c(v1,…,vk)*x` atom for group g (a `c(...)` arity ≠
  `n_groups`, `n_groups < 1`, or `group_labels` arity ≠ `n_groups` errors with
  `PartableError::Kind::BadGroupSpec`). `ParTable` carries `group_var` (the grouping-variable name)
  + `group_labels` (per-group level names) + `n_groups()`; `LavaanifyOptions` has matching
  `group_var` / `group_labels`. On the R side they ride as data.frame attributes (`magmaan.group_var` /
  `magmaan.group_labels`), `lavaan_lavaanify(…, group_var=, group_labels=)`, and the fit object carries
  `$group_var` / `$group_labels`. Tested (`tests/unit/lavaanify_test.cpp`).
- **G3 — robust SE / UΓ / SB-χ² for multi-group** — ✅ **done** (cov-only AND mean-structure).
  `reduced_gamma_sample` now takes a per-block divisor vector (`UFactor::Block` stores `n_obs`); the
  multi-group reduced meat is `Σ_g BᵍᵀΓ̂ᵍBᵍ` with each `Γ̂ᵍ` divided by *that group's* `n_g` (sum,
  not pooled-average) — reproduces `lavaan::cfa(…, group=, estimator="MLM")` `chisq.scaling.factor` /
  `chisq.scaled` to ~1e-4 on 2-group HS (configural, `group.equal="loadings"`, scalar invariance,
  partial invariance). `robust_se` is block-stacked for the `Expected` bread (≙ `se = "robust.sem"` /
  MLM SE; per-block weighting `A1 = Σ_g (n_g/N)Δ_gᵀW_gΔ_g`, `B1 = Σ_g (n_g/N)Δ_gᵀW_gΓ̂_gW_gΔ_g`,
  `vcov = (1/N)A1⁻¹B1A1⁻¹`, expanded through `K`). Residuals (smaller, deferred):
  - **G3b — mean-structure (`~1`) in `build_u_factor` / `robust_se`** — ✅ **done** (Tranche C). Each
    block's stacked slice is `[μ_b; vech(Σ_b)]` (μ-rows on top, σ-rows below); per-block Γ_NT
    block-diagonalises as `[M_b 0; 0 Γ_NT_cov(M_b)]` so the per-block triangular solves apply
    independently to the two segments (anonymous-namespace `apply_L_inv_block` /
    `apply_L_invT_block` / `apply_gamma_nt_block` helpers in `src/fit/robust.cpp`). Lavaan-golden
    parity on 0021 (1F multi-group), 0022 (3F configural), 0023 (3F scalar invariance — λ AND ν
    shared via labels + K_con reparam), 0024 (3F configural with unequal n via `group="sex"`),
    0025 (3F partial invariance — `c(l3a,l3b)*x3` per-group free, others shared), 0026 (single-
    group 2F + meanstructure) — robust SEs match lavaan within 1e-4 on all 5 multi-group fixtures
    plus the single-group case. Empirical-Γ̂ MVN convergence on the meanstructure path verified in
    `tests/unit/robust_test.cpp` (eigenvalues → 1 at n=20000, robust_se vs Expected SE under MVN
    within 5%).
  - **G3c — `Observed`-bread multi-block (MLR / `se = "robust.huber.white"`)**. `build_u_factor` /
    `robust_se` with `Information::Observed` are single-block only (the `H_obs` units are clean only
    single-block; needs a per-block-scaled `A` stack `A_g = √(n_g/N)·L_Γ,g⁻¹·Δ_g` to match the
    pooled per-unit `H_obs = info/N_total`).
  - **G3d — multi-block `reduced_gamma_unbiased`** (Browne's distribution-free correction). The
    reduced `M_sample`/`M_nt` it takes are already block-summed; generalizing needs either per-block
    reduced M's or a recompute-from-`uf` redesign.
- **G4 — analytic observed-info SE for mean-structure models**. `AnalyticObservedInfoSE::compute`
  returns `PostError::Kind::NumericIssue` for mean-structure models (the comment lists the missing
  `∂²μ/∂θ²` cases — Λ×α, Λ×Β, α×Β, Β×Β + η/∂²μ cross-terms). `FdObservedInfoSE` is a correct fallback.
- **G5 — parameter bounds / Heywood detection** — ✅ **partial** (the warnings half). `Inference`
  now carries a `warnings` vector; `src/fit/inference.cpp`'s `heywood_warnings()` flags free
  diagonal Ψ/Θ parameters that came out negative (Heywood cases), wired into all three
  `*InfoSE::compute`. Tested (`tests/unit/inference_test.cpp`). Remaining: actual **box constraints**
  (the `+∞`-when-non-PD objective is a soft barrier, not a real bound) — see **G5b**.
- **G5b — optimizer robustness / vendored optimizers / Ceres backend**.
  ✅ **No longer a crash** (Tranche A): `src/fit/lbfgs_optimizer.cpp` (`-fexceptions` TU) wraps
  `solver.minimize` in a `try/catch (const std::exception&)` and returns
  `FitError::Kind::LineSearchFailed` as a value — the throw site is outside any objective-callback
  frame, so unwinding never crosses a `-fno-exceptions` frame. Tested
  (`tests/unit/lbfgs_optimizer_test.cpp` — a "lying gradient" objective forces the line-search throw
  and asserts the value, not a terminate).
  - **Remaining LBFGS limitations.** LBFGS++ still can't *fit* the single-group 3F+meanstructure
    Holzinger model (the 2-group version converges; lavaan's nlminb converges on both). ULS landing
    (see "Other complete-data estimators" below) exposed two more: (a) shallow-landscape conditioning
    — saturated 6-param 1F ULS needs `max_iter=5000, ftol=1e-14, gtol=1e-9` to converge; (b)
    Heywood wandering — ULS has no PD barrier, so LBFGS drifts into negative residual variance
    without box constraints.
  - **Ceres backend** — 🔍 **active investigation**. Optional via `MAGMAAN_WITH_CERES=ON` CMake flag
    (default OFF; first build ~5–15 min while FetchContent compiles Ceres). Two adapters in
    `include/magmaan/fit/ceres_optimizer.hpp`: `CeresOptimizer` wraps Ceres'
    `GradientProblemSolver` (drop-in for `LbfgsOptimizer`, no bounds) and `CeresBoundedOptimizer`
    wraps Ceres' `Problem` API with native `SetParameterLowerBound` / `SetParameterUpperBound`.
    Two new concepts `Optimizer<O>` and `BoundedOptimizer<O>` in
    `include/magmaan/fit/concepts.hpp` formalize the extension point. A parallel
    `fit_bounded<D, BoundedOptimizer O>()` template in `fit.hpp` takes a `Bounds` struct
    auto-derived from the partable via `bounds_from_partable` (variance diagonals get
    `lower=0`). Benchmark targets: (i) Holzinger 3F-means LBFGS-fail; (ii) ULS shallow landscape
    with default Ceres options; (iii) Heywood-prone ULS where bounds must hold.
  - **Known limitations of this tranche.** `fit_bounded` errors if equality constraints are
    active (multi-group invariance models stay on `fit<>` for now); the bounds path uses Ceres'
    single-residual `r₀ = √(2F+ε)` trick to flow non-LS objectives through the Problem API.
  - **Single-residual sqrt-trick is broken on shallow LS objectives.** Confirmed in
    `tests/unit/ceres_integration_test.cpp`: `fit_bounded<ULS>` on a saturated 1F-feasible cov
    fit stalls — F drops from ~4.0 to ~1.0 in 500 iters, then to ~0.26 in 5000 iters, never
    converging. Root cause: a 1×n Jacobian gives a rank-1 `JᵀJ`, so Levenberg-Marquardt's
    damping factor dominates and the trust radius shrinks to ~1e-6. The bounds infrastructure
    works (`CeresBoundedOptimizer — lower bound is enforced` in `ceres_optimizer_test.cpp`
    confirms enforcement on a well-conditioned quadratic) — the cost-function *shape* is wrong
    for LS estimators. The proper fix is **per-discrepancy multi-residual `ceres::CostFunction`**
    — for ULS, one residual per vech entry (`r_a = (S−Σ)_a`); `JᵀJ` becomes full rank and LM
    converges cleanly. ULS/GLS/WLS all have this LS structure; ML does not (and stays on
    `CeresOptimizer` / LBFGS++). **This is the next Tranche after Ceres lands.**
    `tests/fixtures/fit_std/` still omits a single-group 3F+meanstructure fixture for the
    LBFGS-default path; once the multi-residual Ceres path lands, that fixture can come back.

## Tier 2 — standard reporting

- **G6 — SRMR + AIC + BIC in `fit_measures`** — ✅ **done**. `src/fit/fit_measures.cpp` adds
  `fit_extras(pt, rep, samp, est) → post_expected<FitExtras>` carrying `logl` /
  `unrestricted_logl` / `aic` / `bic` / `bic2` (SABIC) / `srmr` / `npar` / `ntotal`; `fit_measures()`
  still owns CFI/TLI/RMSEA (now with the lavaan `√G` multi-group RMSEA correction). `logl` is the
  full normal-theory log-likelihood (`−½ Σ_b n_b[p_b log 2π + log|Σ̂_b| + tr(S_b Σ̂_b⁻¹) + mahal_b]`),
  *conditional* on `fixed.x` observed exogenous variables (lavaan's convention); `srmr` is the
  Bentler/correlation-residual type, sample-size-pooled over groups. Golden-tested
  (`tests/golden/fit_measures_golden_test.cpp` vs `fitMeasures()`) + unit tests
  (`tests/unit/fit_measures_test.cpp`); wired through R (`measures_fit`). G6 follow-up — **RMSEA
  confidence interval** (`rmsea.ci.lower/upper`) — ✅ **done**: `FitMeasures` gains `rmsea_ci_lower` /
  `rmsea_ci_upper`; `src/fit/fit_measures.cpp` inverts the noncentral-χ²(df_u, λ) CDF for the ncp λ
  at the 95% / 5% percentiles (bisection) and reports `√(λ/(df_u·N))·√G`, matching lavaan's
  `lav_fit_rmsea_ci` (c.hat = 1, incl. the zero-bound conventions and the `N.RMSEA = max(N, 4·X²)`
  upper bracket). New `magmaan::fit::noncentral_chisq_cdf(x, df, ncp)` in `src/fit/inference.cpp` —
  Poisson(ncp/2)-weighted mixture of central χ²(df+2j) CDFs summed outward from the Poisson mode
  (log-space weights → no underflow for large ncp); reuses the existing incomplete-gamma routines.
  Golden-tested (`rmsea.ci.lower/upper` vs `fitMeasures()`) + unit tests for `noncentral_chisq_cdf`
  (vs R's `pchisq`) and the CI edge cases; wired through R (`measures_fit()` now returns
  `rmsea.ci.lower` / `rmsea.ci.upper`). The `rmsea.pvalue` (close-fit test, H₀: RMSEA ≤ 0.05) is
  still not computed.
- **G7 — standardized Ψ off-diagonals** — ✅ **done**. `standardize_lv()` now rescales
  `ψ_jk → ψ_jk / √(ψ_jj ψ_kk)` with the delta-method Jacobian (mirrors `standardize_all`); unit +
  golden coverage (see the test-coverage section).
- **G8 — `:=` chained references + `plabel`/fixed-param references** in `compute_defined()` — ✅ **done**
  (`src/fit/effects.cpp`). `:=` rows now resolve `Param`s against (a) user labels, (b) `.pN.` plabels
  — free *and* fixed/exo rows (a fixed ref carries value, zero gradient), (c) other `:=` names.
  `:=` rows are evaluated in dependency order (Kahn's algorithm, mirroring lavaan's
  `lav_graph_order_adj_mat`); a cycle (incl. a self-reference) is `PostError::NumericIssue`. The
  accumulated θ-gradient flows through chained refs, so a composite `:=` gets the correct
  delta-method SE. Entries returned in source order. Unit-tested (`tests/unit/effects_test.cpp` —
  chained, plabel free+fixed, cyclic, self-cyclic, unknown-id). Not yet exposed in the R bindings
  (`compute_defined` is C++-only today — wiring it through is a separate small task).

## Tier 3 — model exploration (lavaan-parity, arguably beyond "estimation")

- **G-Nested — Satorra (2000) nested-model χ² difference test** — ✅ **done**.
  `compute_satorra2000` (`src/fit/satorra2000.cpp`) runs the streaming low-rank `m × m`
  reduction (m = df_H0 − df_H1) so the `p* × p*` `UΓ` matrix is never formed and Γ̂ stays
  implicit. `lr_test_satorra2000_from_data` (`src/fit/lr_test_satorra.cpp`) orchestrates the
  full pipeline: derives the restriction `A_α` from the two fits' `K` matrices via
  `restriction_alpha_from_K` (`src/fit/restriction.cpp`), evaluates `Π_g` and `Σ̂_g` at
  θ̂_H1, runs the streaming accumulator, and returns scaled/adjusted/exact-mixture p-values.
  Exact-mixture uses `imhof_upper` (`src/fit/weighted_chisq.cpp`). Eigenvalues match the
  inline form-the-full-`UΓ` reference to ≤ 1e-10 on the synthetic tests
  (`tests/unit/satorra2000_test.cpp`). See `docs/nested_tests.md` for the math + API.
- **G9 — modification indices / univariate score tests / EPC** (lavaan `modindices()`). Score test
  for each currently-fixed / equality-constrained parameter at θ̂ + expected parameter change. Needs
  the gradient of F w.r.t. each fixed parameter and the full (including-fixed-params) information
  matrix. Not present at all. Decide whether it's "estimation feature-complete" or a separate track.

---

## Test-coverage gaps (supported but under-tested)

Most of the estimation surface above is golden-tested vs lavaan, but several "supposedly supported"
code paths have *no* test or only a smoke test. Closing these is as much "feature completeness" as
the G-items.

- [x] **Robust SE / Satorra–Bentler / mean-var-adjusted / scaled-shifted χ²** — `robust_se`
      `{Expected,…}`/`{Observed,…}` golden vs `se_robust_sem`/`se_robust_huberwhite` and the
      SB / mean.var.adjusted / scaled.shifted χ² + scaling-factor / shift / Satterthwaite-df
      *outputs* now checked in `tests/golden/inference_golden_test.cpp` (1e-4 / 1e-3) plus the
      multi-group robust-SE parity block in `tests/golden/multigroup_inference_golden_test.cpp`
      (Tranche C: G3b). Note: lavaan's `scaled.shifted` `scaling.factor` is the *reciprocal* of our
      `ScaledShiftedResult::scale_a` (divisor `T/c+b` vs multiplier `T·a+b`); the shift `b` is
      identical. The empirical-Γ̂ UΓ eigenvalue path is exercised on synthetic MVN data
      (`tests/unit/robust_test.cpp` — both the cov-only sister test and the meanstructure analog
      for G3b). Still open: the `MLMV`/`MLMVS` *SE* variants (only the χ² family is checked).
- [x] **`browne_residual_nt` / `rls_chi2`** — golden vs `test="browne.residual.nt"` /
      `"browne.residual.nt.model"` in `tests/golden/inference_golden_test.cpp` (1e-3), in addition to
      the existing hard-coded-number unit tests.
- [x] **`wald_test`** — unit test in `tests/unit/inference_test.cpp` (single restriction → `(θ̂_k/SE_k)²`,
      rank-deficient → error) **and** golden vs `lavTestWald(fit, "<plabel> == 0")` on the first free
      loading of 3F HS in `tests/golden/test_stats_golden_test.cpp`.
- [x] **`lr_test` / `z_test` / `chi2_pvalue`** — `z_test` golden vs `parameterEstimates(fit)` z/pvalue
      and `chi2_pvalue` via the Wald p-value, in `tests/golden/test_stats_golden_test.cpp`; unit
      coverage for all three in `tests/unit/inference_test.cpp`. (`lr_test` = subtraction of two
      already-golden `Inference`s — no separate golden.)
- [x] **Standardized solution `std.lv` / `std.all`** — golden vs `standardizedSolution(fit, type=...)`
      in `tests/golden/standardized_golden_test.cpp` (value 1e-4, SE 1e-3); fixtures under
      `tests/fixtures/fit_std/` — `0001_three_factor_hs` (cov-only, the std.lv Ψ-off-diagonal surface)
      and `0002_three_factor_hs_2group` (configural 2-group + meanstructure, ν rescaling under std.all).
      Open: a single-group + meanstructure variant (blocked on **G5b** — LBFGS crashes on the natural
      3F+means model); free latent-mean (`α`) rescaling under std.all (needs a scalar-invariance fit).
- [x] **`std.lv` Ψ off-diagonal rescaling** (= **G7**) — done. `standardize_lv` now rescales
      `ψ_jk → ψ_jk/√(ψ_jj ψ_kk)` with the delta-method Jacobian (mirrors `standardize_all`); unit +
      golden coverage.
- [x] **Path analysis (`0019_path_hs`) and CFA+structural (`0020_cfa_plus_structural_hs`)** — both
      iterated end-to-end by `fit_theta_golden_test.cpp`, `fit_implied_golden_test.cpp`,
      `fit_measures_golden_test.cpp`, `inference_golden_test.cpp`, and
      `observed_inference_golden_test.cpp`. The Reduced-Β U-factor projector identity is covered by
      `tests/unit/robust_test.cpp` (`U·Γ_NT eigenvalues all 1 on path model`). `AnalyticObservedInfoSE`
      still errors on means models — see G4 — but works on cov-only Β-path fixtures.
- [x] **Multi-group depth** — Tranche C added `0023_scalar_invariance_3f_hs` (λ AND ν shared across
      groups via labels), `0024_multigroup_unequal_n_hs` (3F across HS `sex`, unequal n = 146/155),
      and `0025_partial_invariance_3f_hs` (λ_x3 free per group via `c(l3a, l3b)*x3`, others shared).
      All five multi-group fixtures (0021/0022/0023/0024/0025) now check θ̂/SE/χ²/df/robust_se vs
      lavaan within 1e-4 in `tests/golden/multigroup_inference_golden_test.cpp`; the regen script
      emits per-block `gamma_hat` as a JSON array of `{block, matrix}` objects (`tools/regen_oracle.R`),
      with `tests/oracle.hpp::read_gamma_hat_blockdiag` doing the global-frame block-diagonal lift.
      Known follow-up: lavaan's `browne.residual.nt` differs slightly (~8e0 / 8%) under partial
      invariance with distinct per-group labels; θ̂/SE/χ²/df match exactly, so the Browne check is
      skipped for `0025`. (Tracked as a Tranche C follow-up — likely a per-group-free K_con
      treatment in `browne_residual_nt`.)
- [x] **`:=` chained references / `.pN.` plabel references** in `compute_defined()`
      (`src/fit/effects.cpp`) — done (= **G8**); chained refs evaluated in dependency order
      (cycle → `NumericIssue`), `.pN.` plabels resolve to free *and* fixed rows. Unit-tested.
      Remaining (not G8): wire `compute_defined` through to the R bindings + a golden vs
      `parameterEstimates(fit)` on a model with a chained `:=`.
- [x] **`FdObservedInfoSE` for mean-structure (single-group)** — covered by
      `0026_two_factor_meanstructure_hs` in `tests/golden/observed_inference_golden_test.cpp`
      (Tranche C). The `kSkipAnalyticOnly` set routes 0026 to FD only since `AnalyticObservedInfoSE`
      still errors on means (= G4, deferred). FD matches lavaan's `se_observed` within 1e-4.
- [ ] **`AnalyticObservedInfoSE` for mean-structure / reduced-form Β** — currently
      `PostError::NumericIssue` (= **G4**).
- [x] **Heywood / negative-variance warnings** (= **G5**, warnings half) — `Inference::warnings` +
      `heywood_warnings()` + unit tests done; box constraints / the LBFGS crash remain (**G5b**).
- [ ] **`fit_extras` conditional logl for `fixed.x` observed exogenous** — handled for the
      single-block case via `pt.exo` / `pt.ov_pos`; verify multi-group fixed.x once such a fixture
      exists, and the meanstructure × fixed.x interaction.

## Other complete-data estimators (GLS / ULS / WLS / ADF) — readiness

Verdict: **architecturally ready.** `fit<D = ML, O = LbfgsOptimizer>()`
(`include/magmaan/fit/fit.hpp`) is already templated on the discrepancy and only ever calls
`discrepancy.value(SampleStats, ImpliedMoments)` and `discrepancy.gradient(SampleStats,
ImpliedMoments, J, Jmu)` — where `J = ∂vech(Σ)/∂θ` and `Jmu = ∂μ/∂θ` come from `ModelEvaluator`, so
the model Jacobian is a reusable building block independent of ML. `SampleStats` carries
`S` / `mean` / `n_obs` per block; `gamma_nt(Σ)` and `empirical_gamma(X)` (`src/fit/raw_data.cpp`)
already build the fourth-moment matrix a WLS weight needs; the SE machinery (`src/fit/inference.cpp`
info-trace, `src/fit/robust.cpp`'s `A1 = ΔᵀWΔ`, `B1 = ΔᵀWΓWΔ` sandwich with `W = Γ_NT(M)⁻¹`) is
already Δ-generic — ML already *uses* the GLS-form weight, so ML and GLS share that path. No `fit()`
API change, no new optimizer, no partable/lavaanify change. (The "MLM/MLR/MLMV/MLMVS" *scaled
tests/SEs* are already done, cov-only — they aren't separate estimators, just post-fit corrections on
the ML estimates.)

### ULS — ✅ **done**

- [x] **`ULS`** (`include/magmaan/fit/uls.hpp` + `src/fit/uls.cpp`, ~210 lines total).
      `F = Σ_b (n_b/N) · ½·[(m̄−μ)ᵀ(m̄−μ) + vech(S−Σ)ᵀvech(S−Σ)]`. Gradient stacks
      `w_b = (n_b/N)·vech(Σ_b−S_b)` and `u_b = (n_b/N)·(μ_b−m̄_b)` directly — **no
      vech-doubling factor** like ML's gradient because vech entries appear linearly. Tests in
      `tests/unit/uls_test.cpp` (8 cases, 43 assertions): F=0 at saturated, hand-formula match,
      FD-gradient parity (1F cov, 1F+mean, 3F at lavaan-θ̂), multi-group `(n_b/N)` weighting,
      and `fit<ULS>()` integration recovering ground-truth Σ̂ AND ν̂ on in-manifold S. Two
      operational findings now baked into the tests + tracked in **G5b**: (a) shallow-landscape
      conditioning requires `LbfgsOptions{ max_iter=5000, ftol=1e-14, gtol=1e-9 }`; (b)
      Heywood wandering without bounds (motivates the Ceres tranche).

Prep partly done as a result of the ULS landing:
- [x] `Discrepancy` concept (`include/magmaan/fit/concepts.hpp`) — formalized, applied to
      `fit<D, O>`, guarded by `static_assert(Discrepancy<ML>)`.
- [x] `vech` / `vech_unpack` / `vech_lower` / `vech_index` / `vech_len` — all live in shared
      `src/fit/detail_vech.hpp`; per-TU duplicates in `ml.cpp` / `inference.cpp` removed.
- [ ] **W-generalization of `robust.cpp`'s `A1 = ΔᵀWΔ` / `B1 = ΔᵀWΓWΔ`** to take an arbitrary `W`
      so GLS/ULS/WLS "expected"/sandwich SEs reuse it (ULS: `W = I`; GLS: `W = Γ_NT(Σ̂)⁻¹` = the
      ML path; WLS: `W = Γ⁻¹`). **Deferred until the WLS landing** — driving the abstraction
      with two non-Γ_NT consumers in hand avoids designing the wrong interface.
- [ ] Decide how the WLS weight enters: carry it on the discrepancy (`struct WLS { Eigen::MatrixXd
      gamma_inv; ... };`, built by the caller from `gamma_nt`/`empirical_gamma`) rather than as a new
      `fit()` argument — keeps the `fit()` signature stable.
- [ ] Multi-group: discrepancies and inference already loop over blocks; GLS/ULS/WLS just need
      per-block weight application (~5–10 lines each) — not a blocker for a single-block first cut.

Remaining estimators (each modeled on `include/magmaan/fit/ml.hpp` + `src/fit/ml.cpp` or
`include/magmaan/fit/uls.hpp` + `src/fit/uls.cpp`), with golden parity vs `lavaan::cfa(...,
estimator=...)`. Currently **paused** while the Ceres/bounds tranche lands; after that, GLS is
the natural next consumer (shares ML's `W = Γ_NT(Σ̂)⁻¹` path) and WLS drives the
W-generalization.
- [ ] `GLS` (`gls.hpp`/`gls.cpp`, ~120 lines): `F = ½·tr((Σ⁻¹(S−Σ))²)`, `∇ = Δᵀ·vech-doubled(Σ⁻¹(S−Σ)Σ⁻¹)`.
- [ ] `WLS`/ADF (`wls.hpp`/`wls.cpp`, ~160 lines): `F = ½·vech(S−Σ)ᵀΓ⁻¹vech(S−Σ)`,
      `∇ = ΔᵀΓ⁻¹vech(S−Σ)`; constructor PD-checks `Γ⁻¹`.
- [ ] **ULS golden vs lavaan**: regen-script entry for `cfa(..., estimator="ULS")`; would
      reveal lavaan's exact multi-group weighting convention if it differs from `(n_b/N)`.

Smaller not-on-roadmap items noticed along the way:
- [ ] Sample-moments-only input path (fit from `S` + `n` with no raw data) — `SampleStats` already
      models exactly this; needs an R/API entry point + a test.
- [ ] `se = "none"` / `test = "none"` skip-inference shortcuts (trivial).

---

## Design invariants

- **No global / static mutable state.** The only `static` is factory methods (`Parser::parse`,
  `ModelEvaluator::build`) and the stateless `*SE{}` functor structs — never a mutable singleton,
  cache, or registry. Tools are pure functions: outputs (`Estimates`, `Inference`, `StartValues`,
  `UFactor`, `RobustSeResult`, …) are returned, never written back into their inputs.
- **No in-place mutation of shared structures.** Public entry points (`fit`, `*InfoSE::compute`,
  `build_u_factor`, `robust_se`, `lavaanify`, …) take `ParTable` **by value** and own their copy; the
  only mutation is `resolve_fixed_x_from_sample` filling `ustart` on that local copy. `lavaanify`
  builds a fresh `ParTable` from `PendingRow`s — it never touches its `FlatPartable` input.
- **No groups == one group.** `ParTable::n_groups()` ≥ 1 always; the single-group case is just
  `n_groups() == 1` with `group_var == ""` — not a different shape. Every estimation / inference /
  robust function loops uniformly over `samp.S.size()` blocks; the R wrappers take and return
  per-group lists (length ≥ 1) with no matrix-vs-list / scalar-vs-vector special-casing.

---

## P9 — constraint enforcement (phases 1–2 done; phase 3 out of scope)

**Phase 1 ✅: simple equality, `θ_i = θ_j`.** Covers shared-label equality (`f =~ x1 + a*x2 +
a*x3`), measurement invariance (shared loading labels across groups → cross-group `==` rows), equal
residuals, etc. — the overwhelming majority of real usage. Approach: a **union-find over the free
parameter indices**. Scan ParTable rows with `op == EqConstraint`; each side is a single identifier
(a `.pN.` plabel for the auto-synthesized rows, or a row `label` for explicit `a == b`); resolve to a
free θ index; `union(i, j)`. The resulting groups partition `{1..npar}` into `n_alpha` "reduced"
parameters. Reparameterize `θ_full = K α` where `K` (npar × n_alpha) is the 0/1 group-membership
matrix (`θ[k] = α[group[k]]`); `fit()` optimizes `F_ML(Kα)` over `α` (gradient `∇_α = Kᵀ ∇_θ`,
start `α_0[g] = mean of simple_start_values over group g`), returns the full `θ̂ = Kα̂`. Inference:
compute the unconstrained npar×npar info `I(θ̂)` as today, then `vcov(θ̂) = K (KᵀIK)⁻¹ Kᵀ`,
`se = √diag`, and `df` uses `n_alpha` instead of `npar` (i.e. `df += rank = npar − n_alpha` — lavaan:
`df = #moments − npar + #equality.constraints`). `χ² = N·F_ML(θ̂)` unchanged. The unconstrained case
is the identity reparameterization (`group[k] = k`, `n_alpha = npar`, `rank = 0`) — one code path.

New: `include/magmaan/fit/constraints.hpp` + `src/fit/constraints.cpp` (`struct EqConstraints` +
`build_eq_constraints(const ParTable&) → post_expected<EqConstraints>`). Touches: `fit.hpp` (the
reparam wrapper around the objective + map θ̂ back), `src/fit/inference.cpp` (the K-projection + df in
all three `*InfoSE::compute`), `CMakeLists.txt` (new TU). `build_eq_constraints` errors clearly on:
`<`/`>` rows present (inequalities → phase 3); an `==` side that isn't a bare identifier (arbitrary
linear expressions like `a == 2*b` → phase 2); an `==` side referencing a fixed (free==0) param.

**Phase 2 ✅: general linear equality** (`a == 2*b + c`, `b2 + b3 == 1.5`, `d == 0`). Solver-free —
still a reparameterization, just affine: `θ = θ_0 + Kα` where `K` (npar × n_alpha) is an orthonormal
basis of `ker(R_full)` and `θ_0` a min-norm solution of `R_full · θ = d` (`R_full` stacks the
`eq_groups` "merge" rows `θ_i − θ_j = 0` above the general-linear rows). Inference stays standard
(`df += rank R_full`, `vcov = K(KᵀIK)⁻¹Kᵀ` — basis-independent so it matches lavaan's). Implemented
*structurally* (no AD): `src/partable/lin_constraints.cpp` `analyze_linear()` reduces each `==`-row
side to an affine `Σ coef·θ + cst` or `nullopt` (genuinely nonlinear ⇒ stays flagged → fit() errors);
`resolve_lin_constraints()` (called by `lavaanify` and `from_lavaan_partable`, re-parsing the rows'
canonical text) fills `LatentStructure.lin_constraint_R`/`_d` and clears `has_unenforced_constraints`
if only linear rows remained. `build_eq_constraints` then does one `JacobiSVD(R_full)` for `K`/`θ_0`/rank
and the feasibility check; the no-general-linear path stays byte-identical to phase 1's 0/1-`K` code.
`fit.hpp` projects start values onto the surface and short-circuits the fully-pinned (`n_alpha == 0`)
case; `inference.cpp`/`robust.cpp`/`fit_measures.cpp` unchanged (already generic in `K`). New TUs in
`CMakeLists.txt`. `LavaanifyOptions::effect_coding` rides on top: skips auto.fix.first/std.lv (all
loadings + LV var free) and synthesizes a `.p…+.p… == #indicators` row per (latent, group) — mutually
exclusive with `std_lv`. Golden: `tests/golden/lin_constraint_golden_test.cpp` + `tests/fixtures/fit_lincon/`
(`cfa("visual =~ x1 + b2*x2 + b3*x3 ; b2 + b3 == 1.5", HS)`: θ̂ ≤ 5e-6, se ≤ 1e-4, χ²/df exact,
`df == unconstrained_df + 1`). Unit: `tests/unit/constraints_test.cpp` (general/infeasible/nonlinear/effect-coding),
`tests/unit/lin_constraints_test.cpp` (`analyze_linear`). R: `effect_coding` param on `lavaan_lavaanify`;
`r-package/examples/linear_equality_constraint.R`. Not built: mean-structure effect coding (`Σν == 0`).

**Phase 3 — nonlinear equality + inequality `<`/`>` (`a == b*c`, `(...)^2`, `exp(...)`, `a > 0`, `L1 > L2`):
out of scope.** Two blockers: (1) can't reparameterize away — needs an augmented-Lagrangian/penalty
outer loop around LBFGS or a switch to a constrained optimizer (the deferred "v0.5 constrained
optimizer"); (2) **inference** — an active inequality at θ̂ puts the truth on a boundary, so the LRT
→ chi-bar-squared (a χ² mixture with active-cone-geometry weights, often needing Monte Carlo) and Wald
SEs are non-standard. Shipping `fit()` under inequalities with ordinary-theory SEs/χ²/df would be
silently wrong exactly where it matters, and doing the inference right is a separate research-grade
module. Keep the current clear error. (Nonlinear *equality* alone is asymptotically fine, but travels
with phase 3 computationally.) If Heywood cases ever bite, the honest fix is the detector in **G5**
(flag a parameter that hit its bound, mark its SE untrustworthy) — not constraint enforcement.

### Verification (P9 phase 1)

Add `tests/golden/` (or unit) coverage: `f =~ x1 + a*x2 + a*x3` (within-factor equality);
`f1 =~ x1 + a*x2; f2 =~ x4 + a*x5` (cross-factor equal loading); a 2-group metric-invariance model
(shared loading labels across groups ≙ `lavaan::cfa(..., group="g", group.equal="loadings")`). Check
`θ̂` ≤ 1e-6, `se` ≤ 1e-4, `df` exact vs lavaan; the constrained loadings come out equal to machine
precision; `df` increases by `rank` relative to the unconstrained fit. Fixtures via
`tools/regen_oracle.R`. Build: `cmake --preset asan && cmake --build --preset asan && ctest --preset asan`.
