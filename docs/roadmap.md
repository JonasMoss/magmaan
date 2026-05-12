# latva — roadmap (completing full-data normal-theory ML)

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
  section below). `fit()` reparameterizes `θ = K·α` (`include/latva/fit/constraints.hpp` +
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
  `group_var` / `group_labels`. On the R side they ride as data.frame attributes (`latva.group_var` /
  `latva.group_labels`), `latva_lavaanify(…, group_var=, group_labels=)`, and the fit object carries
  `$group_var` / `$group_labels`. Tested (`tests/unit/lavaanify_test.cpp`).
- **G3 — robust SE / UΓ / SB-χ² for multi-group** — ✅ **done** (cov-only). `reduced_gamma_sample`
  now takes a per-block divisor vector (`UFactor::Block` stores `n_obs`); the multi-group reduced meat
  is `Σ_g BᵍᵀΓ̂ᵍBᵍ` with each `Γ̂ᵍ` divided by *that group's* `n_g` (sum, not pooled-average) —
  reproduces `lavaan::cfa(…, group=, estimator="MLM")` `chisq.scaling.factor` / `chisq.scaled` to
  ~1e-4 on 2-group HS (configural and `group.equal="loadings"`). `robust_se` is block-stacked for the
  `Expected` bread (≙ `se = "robust.sem"` / MLM SE; per-block weighting `A1 = Σ_g (n_g/N)Δ_gᵀW_gΔ_g`,
  `B1 = Σ_g (n_g/N)Δ_gᵀW_gΓ̂_gW_gΔ_g`, `vcov = (1/N)A1⁻¹B1A1⁻¹`, expanded through `K`). Residuals
  (smaller, deferred):
  - **G3b — mean-structure (`~1`) in `build_u_factor` / `robust_se`**. Both gate off `~1` (the
    `dmu_dtheta`-non-empty check). Prepend the per-block μ-rows to Δ + the `[Σ̂_b, 0; 0, Γ_NT(Σ̂_b)]`
    block layout — `browne_residual_nt` already does exactly this; use it as the template.
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
- **G5 — parameter bounds / Heywood detection**. No bounds anywhere; LBFGS can land negative
  variances (the `+∞`-when-non-PD objective is a soft barrier, not a real bound). Minimal: an
  `Inference::warnings` vector + a negative-variance check at θ̂. (Box constraints later.)

## Tier 2 — standard reporting

- **G6 — SRMR + AIC + BIC in `fit_measures`** — ✅ **done**. `src/fit/fit_measures.cpp` adds
  `fit_extras(pt, rep, samp, est) → post_expected<FitExtras>` carrying `logl` /
  `unrestricted_logl` / `aic` / `bic` / `bic2` (SABIC) / `srmr` / `npar` / `ntotal`; `fit_measures()`
  still owns CFI/TLI/RMSEA (now with the lavaan `√G` multi-group RMSEA correction). `logl` is the
  full normal-theory log-likelihood (`−½ Σ_b n_b[p_b log 2π + log|Σ̂_b| + tr(S_b Σ̂_b⁻¹) + mahal_b]`),
  *conditional* on `fixed.x` observed exogenous variables (lavaan's convention); `srmr` is the
  Bentler/correlation-residual type, sample-size-pooled over groups. Golden-tested
  (`tests/golden/fit_measures_golden_test.cpp` vs `fitMeasures()`) + unit tests
  (`tests/unit/fit_measures_test.cpp`); wired through R (`latva_fit_measures`). Remaining G6
  follow-up: **RMSEA confidence interval** (`rmsea.ci.lower/upper` — needs a noncentral-χ² quantile
  root-find on top of the existing `chi2_pvalue` machinery).
- **G7 — standardized Ψ off-diagonals**. `standardize_lv()` passes Ψ off-diagonals through unscaled;
  should rescale `ψ_jk → ψ_jk / √(ψ_jj ψ_kk)` with the delta-method Jacobian.
- **G8 — `:=` chained references + `plabel`/fixed-param references** in `compute_defined()`.

## Tier 3 — model exploration (lavaan-parity, arguably beyond "estimation")

- **G9 — modification indices / univariate score tests / EPC** (lavaan `modindices()`). Score test
  for each currently-fixed / equality-constrained parameter at θ̂ + expected parameter change. Needs
  the gradient of F w.r.t. each fixed parameter and the full (including-fixed-params) information
  matrix. Not present at all. Decide whether it's "estimation feature-complete" or a separate track.

---

## Test-coverage gaps (supported but under-tested)

Most of the estimation surface above is golden-tested vs lavaan, but several "supposedly supported"
code paths have *no* test or only a smoke test. Closing these is as much "feature completeness" as
the G-items.

- [ ] **Robust SE / Satorra–Bentler / mean-var-adjusted / scaled-shifted χ²** — golden parity vs
      lavaan `cfa(..., estimator="MLM"/"MLMV"/"MLMVS"/"MLR")`. Today: UΓ-eigenvalue *sanity* smoke
      checks in `tests/unit/robust_test.cpp`; the `.fit.json` fixtures carry `sb_chi2` / `sb_scale` /
      `mean_var_chi2` / `scaled_shifted_*` but no test compares our `robust_se` / wrapper output to
      lavaan's SEs and stats.
- [ ] **`browne_residual_nt` / `rls_chi2`** (`include/latva/fit/inference.hpp`) — no test calls
      these; fixtures have `browne_residual_nt` / `rls_chi2`. Add golden vs lavaan
      `test="browne.residual.nt"` / `"browne.residual.nt.model"`.
- [ ] **`wald_test`** — no test; unit test (fix a loading → known χ²(1)) + golden vs `lavTestWald`.
- [ ] **`lr_test` / `z_test` / `chi2_pvalue`** — no direct test; `chi2_pvalue` vs
      `pchisq(x, df, lower.tail=FALSE)`, `z_test` vs lavaan's `parameterEstimates` z/pvalue columns.
- [ ] **Standardized solution `std.lv` / `std.all`** — only formula smoke tests; golden vs lavaan
      `parameterEstimates(fit, standardized=TRUE)` (single + multi-group, with/without `~1`).
- [ ] **`std.lv` Ψ off-diagonal rescaling** — currently passed through unscaled
      (`src/fit/standardized.cpp`); = **G7**. Fix to `ψ_jk/√(ψ_jj ψ_kk)` with the delta-method
      Jacobian (`std.all` already does this) + test.
- [ ] **Path analysis (`0019_path_hs`) and CFA+structural (`0020_cfa_plus_structural_hs`)** — fitted
      by `fit_theta_golden_test.cpp` but no `implied`/`fit_measures` golden for the reduced-form
      Β path beyond θ̂. Extend coverage (and note where `AnalyticObservedInfoSE` still errors on Β).
- [ ] **Multi-group depth** — only `0021`/`0022`. Add scalar invariance
      (`group.equal="loadings,intercepts"`), partial invariance via `c(...)`, unequal group sizes.
- [ ] **`:=` chained references / `.pN.` plabel references** in `compute_defined()`
      (`src/fit/effects.cpp`) — header says unsupported; = remainder of **G8**. Implement + test, or
      document the limitation and test the error path.
- [ ] **`AnalyticObservedInfoSE` for mean-structure / reduced-form Β** — currently
      `PostError::NumericIssue` (= **G4**); meanwhile test that `FdObservedInfoSE` matches lavaan for
      those cases (it's only golden-tested on cov-only CFA today).
- [ ] **Heywood / negative-variance warnings** — no `warnings` vector on `Inference` (= **G5**); add
      one + a post-fit negative-variance check, with a known-Heywood test model.
- [ ] **`fit_extras` conditional logl for `fixed.x` observed exogenous** — handled for the
      single-block case via `pt.exo` / `pt.ov_pos`; verify multi-group fixed.x once such a fixture
      exists, and the meanstructure × fixed.x interaction.

## Other complete-data estimators (GLS / ULS / WLS / ADF) — readiness

Verdict: **architecturally ready.** `fit<D = ML, O = LbfgsOptimizer>()`
(`include/latva/fit/fit.hpp`) is already templated on the discrepancy and only ever calls
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

Prep before adding GLS/ULS/WLS:
- [ ] Formalize the `Discrepancy` concept (`include/latva/fit/concepts.hpp`): require
      `value(SampleStats, ImpliedMoments) -> fit_expected<double>` and
      `gradient(SampleStats, ImpliedMoments, MatrixXd J, MatrixXd Jmu) -> fit_expected<VectorXd>`;
      constrain the `fit<D,O>` template on it. (~30 lines; AGENTS.md already promises this.)
- [ ] Factor the `vech` / `vech_unpack` / duplication helpers out of `src/fit/robust.cpp` into a
      shared `src/fit/detail_vech.hpp` so GLS/ULS/WLS reuse them.
- [ ] Decide how the WLS weight enters: carry it on the discrepancy (`struct WLS { Eigen::MatrixXd
      gamma_inv; ... };`, built by the caller from `gamma_nt`/`empirical_gamma`) rather than as a new
      `fit()` argument — keeps the `fit()` signature stable.
- [ ] Generalize `robust.cpp`'s `A1 = ΔᵀWΔ` / `B1 = ΔᵀWΓWΔ` builder to take an arbitrary `W` so
      GLS/ULS/WLS "expected"/sandwich SEs reuse it (ULS: `W = I`; GLS: `W = Γ_NT(Σ̂)⁻¹` = the ML
      path; WLS: `W = Γ⁻¹`).
- [ ] Multi-group: discrepancies and inference already loop over blocks; GLS/ULS/WLS just need
      per-block weight application (~5–10 lines each) — not a blocker for a single-block first cut.

Then the estimators (each modeled on `include/latva/fit/ml.hpp` + `src/fit/ml.cpp`), with golden
parity vs `lavaan::cfa(..., estimator=...)`:
- [ ] `ULS` (`uls.hpp`/`uls.cpp`, ~80 lines): `F = ½·vech(S−Σ)ᵀvech(S−Σ)`, `∇ = Δᵀvech(S−Σ)`.
- [ ] `GLS` (`gls.hpp`/`gls.cpp`, ~120 lines): `F = ½·tr((Σ⁻¹(S−Σ))²)`, `∇ = Δᵀ·vech-doubled(Σ⁻¹(S−Σ)Σ⁻¹)`.
- [ ] `WLS`/ADF (`wls.hpp`/`wls.cpp`, ~160 lines): `F = ½·vech(S−Σ)ᵀΓ⁻¹vech(S−Σ)`,
      `∇ = ΔᵀΓ⁻¹vech(S−Σ)`; constructor PD-checks `Γ⁻¹`.

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

New: `include/latva/fit/constraints.hpp` + `src/fit/constraints.cpp` (`struct EqConstraints` +
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
`tests/unit/lin_constraints_test.cpp` (`analyze_linear`). R: `effect_coding` param on `latva_lavaanify`;
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
