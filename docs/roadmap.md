# latva — roadmap (completing full-data normal-theory ML)

Status: P0–P8 done (parser → partable → matrix rep → evaluator → ML/LBFGS → expected/observed SEs,
χ², df) plus post-P8 extensions (multi-group — configural *and* measurement invariance via shared
labels / `c(...)` per-group modifiers / cross-group equality constraints, with the grouping variable
+ labels stored in the partable; robust SEs / UΓ / Satorra-Bentler family — multi-group for the
cov-only `Expected`-bread path; mean structure `~1`; `:=` defined parameters; standardized
`std.lv`/`std.all`; CFI/TLI/RMSEA; LR/Wald/z tests; Browne residual tests). Golden-tested against
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

- **G6 — SRMR + AIC + BIC in `fit_measures`**. `src/fit/fit_measures.cpp` has CFI/TLI/RMSEA; SRMR
  needs the standardized residuals of `S − Σ̂` (+ a mean-residual term with meanstructure); AIC/BIC
  need the log-likelihood (`ll = −½ N (F_ML + log|S| + p + p·log 2π)`).
- **G7 — standardized Ψ off-diagonals**. `standardize_lv()` passes Ψ off-diagonals through unscaled;
  should rescale `ψ_jk → ψ_jk / √(ψ_jj ψ_kk)` with the delta-method Jacobian.
- **G8 — `:=` chained references + `plabel`/fixed-param references** in `compute_defined()`.

## Tier 3 — model exploration (lavaan-parity, arguably beyond "estimation")

- **G9 — modification indices / univariate score tests / EPC** (lavaan `modindices()`). Score test
  for each currently-fixed / equality-constrained parameter at θ̂ + expected parameter change. Needs
  the gradient of F w.r.t. each fixed parameter and the full (including-fixed-params) information
  matrix. Not present at all. Decide whether it's "estimation feature-complete" or a separate track.

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

## P9 — constraint enforcement (phase 1 done; phases 2–3 pending)

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

**Phase 2 (later): general linear equality** (`a == 2*b + c`, `a == 0`). Needs the constraint Expr
ASTs (from the `FlatPartable` — same pattern as `compute_defined()` which takes both `flat` and `pt`)
+ the forward-mode AD in `src/fit/effects.cpp`'s `eval()` to get each `==` row's gradient (= a row of
`R`) and check linearity. General null-space reparam `θ = θ_0 + Kα` (K = orthonormal basis of
`ker(R)`, θ_0 a particular solution). `fit()` and the SE methods would take the `FlatPartable` too.

**Phase 3 (later): nonlinear equality + inequality `<`/`>`** (`a == b*c`, `a > 0`). Augmented-Lagrangian
(or penalty) outer loop around the existing LBFGS, evaluating constraint functions + Jacobians each
outer iteration (again via `effects.cpp`'s `eval()`). lavaan's `nlminb`-with-`eval_g` path.

### Verification (P9 phase 1)

Add `tests/golden/` (or unit) coverage: `f =~ x1 + a*x2 + a*x3` (within-factor equality);
`f1 =~ x1 + a*x2; f2 =~ x4 + a*x5` (cross-factor equal loading); a 2-group metric-invariance model
(shared loading labels across groups ≙ `lavaan::cfa(..., group="g", group.equal="loadings")`). Check
`θ̂` ≤ 1e-6, `se` ≤ 1e-4, `df` exact vs lavaan; the constrained loadings come out equal to machine
precision; `df` increases by `rank` relative to the unconstrained fit. Fixtures via
`tools/regen_oracle.R`. Build: `cmake --preset asan && cmake --build --preset asan && ctest --preset asan`.
