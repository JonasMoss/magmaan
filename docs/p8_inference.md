# P8 — Inference (Expected-info SEs + χ²)

Short execution plan for the next session. Completes the v0 statistical surface: standard errors, χ², and df. The non-inference pipeline (parser → lavaanify → matrix_rep → ModelEvaluator → ML → LBFGS) is done; 91/91 tests green; we have lavaan-equivalent θ̂ on CFA, path analysis, and CFA+structural.

## Goal

`compute_inference(pt, rep, samp, estimates) → Inference` that returns SEs, χ², df, the information matrix, and the parameter-covariance matrix. Goldens against `lavaan::parTable(fit)$se`, `fitMeasures(fit)[c("chisq","df")]`.

## Math

For ML, the per-block expected information about θ is:

$$ I_b = \tfrac{n_b - 1}{2} \cdot \mathrm{tr}\!\left(\Sigma_b^{-1}\,\tfrac{\partial \Sigma_b}{\partial\theta_a}\,\Sigma_b^{-1}\,\tfrac{\partial \Sigma_b}{\partial\theta_b}\right) $$

stacked into a `(n_free × n_free)` matrix `I`. Then:

- `vcov = I⁻¹` (use LLT; fall back to LDLT if not PD)
- `se[k] = sqrt(vcov[k, k])`
- `chi2 = (n - 1) · fmin`
- `df = sum_b p_b(p_b + 1)/2 − n_free`

Computational trick: for each free parameter k, materialize `M_k = ∂Σ/∂θ_k` (un-vech the Jacobian column to a symmetric `p × p` matrix). Compute `T_k = Σ⁻¹ M_k` once. Then `I[a, b] = ((n-1)/2) · trace(T_a · T_b)`, evaluated efficiently with `(T_a.transpose().array() * T_b.array()).sum()`. **O(n_free · p³)** for the `T_k`s plus **O(n_free² · p²)** for the pairwise traces — cheap at v0 sizes.

## API

New types in `include/latva/fit/inference.hpp`:

```cpp
struct Inference {
  Eigen::MatrixXd info;     // (n_free × n_free)  — Fisher information
  Eigen::MatrixXd vcov;     // info⁻¹             — parameter covariance
  Eigen::VectorXd se;       // n_free             — sqrt(diag(vcov))
  double          chi2 = 0;
  int             df   = 0;
};

struct ExpectedInfoSE {
  static constexpr std::string_view name = "expected";
  post_expected<Inference>
  compute(const partable::ParTable&  pt,
          const model::MatrixRep&    rep,
          const SampleStats&         samp,
          const Estimates&           est) const;
};
```

`PostError` already exists in `error.hpp` (kinds: `InfoMatrixSingular`, `BootstrapFailed`). Wire `post_expected<T>` via `using post_expected = std::expected<T, PostError>;` in `expected.hpp` (it's there — just check).

Add an overload to `fit()` (optional) that runs ML + LBFGS + ExpectedInfoSE in one call, returning a combined `FittedModel { Estimates est; Inference inf; }`. Or keep them separate — caller composes.

## Files

| File | Purpose |
|---|---|
| `include/latva/fit/inference.hpp` | `Inference`, `ExpectedInfoSE` |
| `src/fit/inference.cpp` | `compute()` implementation |
| `tests/unit/inference_test.cpp` | Shape, SE positivity, I≈Iᵀ, df formula |
| `tests/golden/inference_golden_test.cpp` | SE / χ² / df vs lavaan |

Wire `src/fit/inference.cpp` into `CMakeLists.txt`.

## Tests

Unit tests:
1. **Shape**: `inf.info.rows() == n_free`, `inf.se.size() == n_free`, `inf.df > 0` for identified models.
2. **PSD**: For 1F CFA at θ̂_lavaan, `info` is symmetric positive-definite.
3. **df formula**: For Holzinger 3F (`p=9, n_free=21`), `df = 9·10/2 − 21 = 24`.
4. **SE positivity**: All `inf.se(k) > 0` for properly identified models.
5. **Information equals expected Hessian**: For 1F CFA at random θ near identity, compare `info` against the analytic Hessian of `(n-1)·F_ML/2` numerically (finite-diff Jacobian of the gradient).

Goldens (extend `tools/regen_oracle.R`):
```R
# Add to the fit-fixture payload:
se   <- parTable(fit)$se[parTable(fit)$free > 0]
chi2 <- as.numeric(fitMeasures(fit)["chisq"])
df_v <- as.integer(fitMeasures(fit)["df"])
# include `se`, `chi2`, `df` fields in the JSON
```

Then `inference_golden_test.cpp` iterates the corpus, fits with our pipeline, computes inference, asserts:
- `max |se - se_lavaan| ≤ 1e-4`
- `|chi2 - chi2_lavaan| ≤ 1e-3`
- `df == df_lavaan` exactly

Reuse the existing skip list for under-identified models (`0010`, `0013`, `0015`, `0018`).

## Implementation order

1. Add `post_expected<T>` alias in `include/latva/expected.hpp` if missing.
2. Write `inference.hpp` + `inference.cpp`. Use existing `ModelEvaluator::sigma` and `dsigma_dtheta` — DON'T duplicate the Σ/J machinery; `compute()` should take a `ModelEvaluator&` or call `build(pt, rep)` internally.
3. Unit tests; iterate until shape + positivity + df work.
4. Extend R script; regenerate fit fixtures.
5. Golden test; iterate until lavaan parity holds.
6. Build + ctest under both presets.

## Where you left off

- **All current state**: 91/91 tests green under asan + default. 18 lexer fixtures, 18 parser fixtures, 15 ptable fixtures (+1 tolerated divergence), 14 matrix_rep fixtures (+1 deferred for mean structure), 13 implied-Σ fixtures, 9 fit-θ̂ fixtures (+4 under-identified skips).
- **Reduced LISREL form** (P5.2/P6.2) just landed. `MatrixRep::form` is `PureCFA` or `Reduced`; `ModelEvaluator` uses `Σ = Λ (I−B)⁻¹ Ψ (I−B)⁻ᵀ Λᵀ + Θ` always. Phantom-Λ identity columns are inserted as `StructuralCell`s for ov.y/ov.x in Reduced form.
- **`resolve_fixed_x_from_sample`** (`include/latva/fit/resolve_fixed_x.hpp`) fills `pt.ustart` for fixed.x rows from sample S. The `fit()` template calls it automatically; the golden test for implied Σ also calls it.
- **LBFGS-on-GCC gotcha**: `src/fit/lbfgs_optimizer.cpp` is built with `-fexceptions` (set in CMakeLists.txt via `set_source_files_properties`) because LBFGS++ has a `throw` in its param check that GCC refuses to compile under `-fno-exceptions`. The throw path is unreachable with our default params.
- **Test convention**: `must_build(model_str)` helpers in unit tests use `static thread_local` slots to keep partable/matrix_rep alive across `ModelEvaluator::build` calls — the build borrows references. Production code owns them at the call site.
- **Tolerance lesson** (P7 epilogue): On 21-param 3F Holzinger, LBFGS and lavaan's nlminb land within 1.03e-6 of each other — they converge to different points on a flat section of the ML surface. The fit-θ̂ golden uses 2e-6 tolerance (with comment); tightening LBFGS past 1e-7 trips line-search aborts under `-fno-exceptions`. **For SEs/χ² this matters less** because both quantities are smooth at the optimum — expect cleaner parity.

## Status — what's done

- **`ExpectedInfoSE`** ✓ — closed-form, `(N/2) tr(Σ⁻¹ ∂Σ/∂θ_a Σ⁻¹ ∂Σ/∂θ_b)`. Matches lavaan SE / χ² / df on the full single-group corpus.
- **`FdObservedInfoSE`** ✓ — central-difference Hessian of the analytic ML gradient. Works on every fittable model.
- **`AnalyticObservedInfoSE`** ✓ — closed-form `H1 + H2` for the general LISREL form. All six non-zero (mat_a, mat_b) cases of `∂²Σ` derived and implemented. Cross-checks `FdObservedInfoSE` on Pure CFA + Reduced (path, CFA + structural).
- **Per-block evaluator surface** ✓ — `ModelEvaluator::assembled()` + `param_locations()` expose Λ, Ψ, Θ, B, A, LamA, Mid and the free-parameter location table that analytic post-fit math needs.

## Next up

- **Observed-info goldens vs lavaan** ✓ — `regen_oracle.R` dumps `se_observed` (lavaan with `information = "observed"`); `tests/golden/observed_inference_golden_test.cpp` runs both `FdObservedInfoSE` and `AnalyticObservedInfoSE` against this oracle. **FD 9/9, analytic 9/9** within `1e-4` absolute SE on the full fittable corpus.

- **Multi-group inference math** ✓ — all three SE methods accumulate `(n_b/2)·H_b` per block:
  - `ExpectedInfoSE`: already per-block (no change).
  - `FdObservedInfoSE`: now uses `ML::gradient_block(b)` (new) for per-block central-difference Hessian.
  - `AnalyticObservedInfoSE`: per-block precomputes (GLam, GLamA, GLamMid, K, P, MKA, M_k_b, T_k_b, ZM_k_b) and per-block H1+H2 accumulation. The (a, b) loop only contributes H2_b when both params live in block b (no shared-param support yet).
  - Verified by `Inference: multi-block infrastructure (synthetic 2-block 1F CFA)` — duplicates the corpus's 1F CFA into 2 identical independent blocks and checks block-diagonal structure + FD ≈ analytic + diagonal-blocks-match-single-block.

## Multi-group — remaining work to expose to users

The inference math is ready. The remaining pieces are upstream:

- **`lavaanify` multi-group parsing.** `group = "var"` argument; `c(val_1, val_2)` modifier interpreted per-group. Produces a ParTable with rows split across blocks.
- **`build_matrix_rep` multi-block dims.** Per-block `ov_names` / `lv_names` / dims; cell.block set correctly.
- **ML weighting decision.** Currently `F = Σ_b F_b` (uniform). Lavaan's `likelihood = "normal"` uses `F = Σ_b (n_b/N)·F_b`. For matching lavaan θ̂ on unbalanced multi-group, switch to the weighted form. Single-group is unchanged either way.
- **Corpus / R script.** Add multi-group fixtures (e.g. HolzingerSwineford × `school`); have `regen_oracle.R` fit and dump per-block S / n_obs / Σ̂ / SEs.
- **Multi-group goldens.** A `multigroup_inference_golden_test.cpp` mirroring the single-group one, iterating multi-group corpus entries.

A `SharedParam` extension (multi-group with cross-group label equality / equal-loadings constraints) is a separate refinement: `ModelEvaluator::param_locations()` would return a list-of-locations per free index, and analytic H2 would need to accumulate across all blocks a shared param affects rather than just one.

## Normal-theory feature roadmap

Priority order, from "finish in-flight" to "smaller deltas after":

1. **Multi-group end-to-end** (in flight):
   - **Inference math** ✓ — `ExpectedInfoSE`, `FdObservedInfoSE`, `AnalyticObservedInfoSE` all accumulate per-block with `(n_b/2)` weighting. Synthetic 2-block test verifies block-diagonal structure + FD ≈ analytic on multi-block.
   - **ML weighting** ✓ — `ML::value` and `ML::gradient` use lavaan's `likelihood = "normal"` weighting `F_ML = Σ_b (n_b/N)·F_b`. `ML::gradient_block(b)` stays un-weighted (per-block ∂F_b/∂θ) so `FdObservedInfoSE`'s per-block accumulation is consistent. Single-group unchanged; multi-group `χ² = N · fmin` matches `Σ_b n_b · F_b`.
   - **`resolve_fixed_x_from_sample`** ✓ — already block-aware (each cell carries its block index, looks up the right `samp.S[block]`).
   - **`lavaanify` multi-group** ✓ (configural)
     - Drop the `n_groups != 1` error. Refactor steps 1-6 (formula rows + auto-* + fixed.x) into `build_group_template(flat, opts, v)` and call it `n_groups` times, setting `block`/`group` per replica.
     - Plabels assigned globally so equality constraints can reference any row across groups; auto-equality scan picks up cross-group label matches automatically (when the user supplies a shared label, both groups' rows get that label → cross-group `==` row).
     - **`c(v0, v1)` modifier** ✓ — `select_group_atom(modifier, group_idx, n_groups, context)` picks the right atom by index; wrong-arity is rejected with a clear message. Unlocks per-group fixed values (`f =~ c(1, NA)*x1`) and breaking cross-group equality via per-group labels (`f =~ c("a1", "a2")*x1`).
   - **`build_matrix_rep` multi-block** ✓
     - Counts `n_blocks` from `max(pt.block)` (lavaan-style 1-based).
     - Resizes `dims` / `ov_names` / `lv_names` to `n_blocks`. In v0 each block has the same variable set (same model template), so `classify_vars` runs once and broadcasts.
     - Per-cell `block` field set from `pt.block[i] − 1` (1-based → 0-based).
     - Structural phantom-Λ cells per block for Reduced form.
   - End-to-end multi-group fit verified: 1F CFA × 2 groups with same S → joint fit recovers each group's θ̂ matching the single-group fit; expected and FD-observed info matrices are 12×12 and block-diagonal (configural → no cross-group info coupling).
   - **Multi-group + mean structure golden** ✓ — `f =~ x1 + x2 + x3` with `n_groups = 2` + `meanstructure = true`, fit on lavaan's per-block `sampstat` from HolzingerSwineford × school. θ̂ matches lavaan within `1e-5`, ExpectedInfoSE SEs within `1e-4`, df = 0 (saturated), chi² ≈ 0.
   - **Corpus + R-script pipeline for multi-group fixtures** ✓ — `corpus.json` entries can now carry optional `n_groups`, `group_var`, `meanstructure` fields. `regen_oracle.R`:
     - Builds the `cfa()` arg list dynamically, threading `group =` and `meanstructure = TRUE` through when set.
     - Normalizes single- and multi-group `lavInspect(fit, "sampstat")` returns into the same per-block-array shape; writes `sample_cov`, `sample_mean` (when present), `n_obs_per_block`, and per-block `implied_sigma`.
     - Re-runs the `information = "observed"` refit through the same code path so `se_observed` works for multi-group too.
   - Corpus entry `0021_multigroup_1f_hs` (1F CFA × school + meanstructure) drives the new `tests/golden/multigroup_inference_golden_test.cpp`. The existing single-group goldens (`fit_theta_golden_test`, `fit_implied_golden_test`, `inference_golden_test`, `observed_inference_golden_test`) all skip `n_groups > 1` entries — multi-group lives in its own golden. Golden verifies θ̂ (≤1e-5), `ExpectedInfoSE` SE (≤1e-4), `FdObservedInfoSE` SE against `se_observed` (≤1e-4), chi² (≤1e-3), and df exactly.
   - **df fix for mean structure**: `compute_chi2_df` now adds `p_b` mean moments per block when `samp.mean[b]` is populated. Caught while writing the multi-group golden — the in-line chi²/df logic inside `ExpectedInfoSE::compute` was an out-of-date copy; refactored to call the shared `compute_chi2_df` so all three SE methods stay in sync.

2. **Mean structure** (`~1` operator):
   - **Phase 1 — types + evaluator** ✓
     - `MatId::Nu` (p × 1 indicator intercepts), `MatId::Alpha` (m × 1 latent means).
     - `BlockMatrices` and `BlockState` expose Nu/Alpha; `has_means` flag per block.
     - `build_matrix_rep` emits Nu cell for `ov ~ 1` and Alpha cell for `lv ~ 1`.
     - `decide_form` no longer triggers Reduced for `~1`-only models — pure CFA + intercepts now stays in PureCFA form.
     - `ModelEvaluator::sigma()` computes `μ = Nu + Λ·A·α` per block when `has_means`; `ImpliedMoments::mu[b]` is populated (empty otherwise, so downstream consumers detect "covariance-only" via `mu[b].size() == 0`).
     - `assembled()` exposes Nu/Alpha; `dsigma_dtheta` handles Nu/Alpha as zero columns (Σ doesn't depend on mean params).
     - `start_values` defaults Nu_i to `samp.mean[b](i)` (falls back to 0) and Alpha_j to 0.
   - **Phase 2 — ML** ✓
     - `ModelEvaluator::dmu_dtheta(theta)` — closed-form `(Σ_b p_b) × n_free` Jacobian. Per param: `∂μ/∂ν_i = e_i`, `∂μ/∂α_j = LamA[:, j]`, `∂μ/∂Λ_{r,c} = e_r · (A·α)_c`, `∂μ/∂B_{r,c} = LamA[:, r] · (A·α)_c`, Ψ/Θ → 0. Returns empty matrix sentinel when no block has mean structure.
     - `ML::value` adds `(m̄_b − μ_b)' Σ_b⁻¹ (m̄_b − μ_b)` per block (weighted by `n_b/N`).
     - `ML::gradient` and `ML::gradient_block` take an optional `Jmu` argument (default empty). The mean-term contribution decomposes as a rank-1 correction `−scale·zz'` to the existing `vech-doubled(G)` (where `z = Σ⁻¹d`) plus a separate `−2·scale·z` vector multiplied by `Jmu.transpose()`. Single-group + no-mean-structure callers don't change behavior; mean-structure paths flow through the same fill helper with the rank-1 add-on.
     - `fit()` and `FdObservedInfoSE` updated to compute `Jmu` and thread it through. For covariance-only models `Jmu` is empty and the mean-term branch is skipped automatically.
     - Bug found and fixed along the way: `lavaanify`'s `auto.var` was double-adding `xi ~~ xi` rows when `xi` appeared in both `ov_ind` (RHS of `=~`) and `ov_y` (LHS of `~1`) — the deduplication checked only `user == 1` rows. Now checks for any matching row.
     - Tests: F value matches hand calculation; analytic gradient matches central-difference FD to `1e-5`; `fit()` recovers `ν̂_i ≈ m̄_i` exactly on a saturated mean-structure CFA.
   - **Phase 3 — inference** ✓ (partial)
     - **`ExpectedInfoSE`** ✓ — extended with the closed-form mean-term contribution `(n_b/2) · 2 · ν_a' Σ_b⁻¹ ν_b` per block. The factor of 2 inside the `(n/2)` scale is the standard SEM expected-info formula: `I[a,b] = (N/2)·tr(WM_aWM_b) + N·ν_a'·W·ν_b`. Pre-computes `η_{k,b} = Σ_b⁻¹·ν_{k,b}` once per (k, b). Verified: ν SEs match the closed-form `√(Σ̂_ii / n)` on a saturated mean-structure CFA.
     - **`FdObservedInfoSE`** ✓ — already correct because `gradient_block` propagates the mean term. Verified: FD ≈ Expected at saturated mean-structure fit (rel SE diff < 1e-4).
     - **`AnalyticObservedInfoSE`** — errors out cleanly with `PostError::NumericIssue` on mean-structure models for now. Full closed-form needs Part A's extra η/zz' terms in `∂G*/∂θ` plus Part B's `−2·(∂²μ)'·z + 2·ν_a'·W·M_b·z + 2·ν_a'·η_b` derivations plus per-pair `∂²μ/∂θ_a∂θ_b` cases (non-zero for (Λ,α), (Λ,B), (α,B), (B,B)). FD path works in the meantime.
   - **Phase 4 — lavaanify auto-add defaults** ✓ — `LavaanifyOptions::meanstructure` (default false). When true, auto-adds `xi ~ 1` (free) for every observed variable that doesn't already have a `~1` row, and `lv ~ 1` (fixed at 0) for every user latent. Matches lavaan's `meanstructure = TRUE` convention.
   - **Phase 5 — fixtures + goldens**: `regen_oracle.R` dumps per-block sample means and lavaan's intercept/mean estimates; goldens for SE / chi² / df under mean structure.

3. **Equality constraints** (`==`, plus label-based equality propagation):
   - Lavaanify already auto-generates `==` rows from shared labels; need partable handling + optimizer projection or Lagrangian.

4. **Defined parameters** (`:=`) ✓ — indirect / total effects:
   - `fit::compute_defined(flat, pt, est, vcov) → DefinedParams` evaluates every `:=` constraint via forward-mode AD over the existing `parse::Expr` AST. Per-node value/gradient rules: `Num`, `Param` (via user label → θ_k lookup or fixed-value lookup), `+ − * / ^`, unary `+ −`.
   - SE via delta method: `se² = ∇' · vcov · ∇`.
   - Resolves labels on partable rows; unknown identifiers surface as `PostError::NumericIssue`.
   - Verified: `a² := a^2` gives `(θ̂_a², |2·a|·se(a))`; `prod := a·b` matches `b²·var(a) + 2ab·cov(a,b) + a²·var(b)`.

5. **Standardized solutions** ✓ (std.lv and std.all done; β-rescaling pending)
   - `fit::standardize_lv(pt, rep, est, vcov) → StandardizedSolution`: factor variances → 1, Λ → λ·√ψ_jj, others identity. Delta-method SEs.
   - `fit::standardize_all(pt, rep, est, vcov)`: on top of std.lv, observed variables are rescaled by √σ_ii — `Λ → λ·√ψ_cc/√σ_rr`, `Θ_rr → θ_rr/σ_rr`, `ν_r → ν_r/√σ_rr`, `α_j → α_j/√ψ_jj`, `ψ_jk → ψ_jk/√(ψ_jj·ψ_kk)`. The Jacobian threads `∂σ_rr/∂θ` from `dsigma_dtheta` for the indicator-scale terms. β coefficients and Θ off-diagonals still pass through as identity (β rescaling needs ψ_kk_orig/ψ_jj_orig accounting).

6. **Inequality constraints** (`<`, `>`), **modification indices** — slot in later.
7. **LR test** ✓ — `lr_test(restricted, unrestricted) → {chi2_diff, df_diff}`.
8. **Wald test** ✓ — `wald_test(R, q, est, vcov) → {chi2, df}`. `(Rθ̂−q)' (R·vcov·Rᵀ)⁻¹ (Rθ̂−q)`. LLT-only; rank-deficient restrictions surface as `PostError::InfoMatrixSingular`.
9. **χ² p-values** ✓ — `chi2_pvalue(chi2, df)` via hand-rolled regularized upper incomplete gamma (series for `x < a+1`, continued fraction otherwise). Spot-checked against R's `pchisq(..., lower.tail=FALSE)`.
10. **Per-parameter z-test** ✓ — `z_test(est, inf) → {z, p_value}`. `z_k = θ̂_k / SE_k`; `p_k = χ²(1) p-value of z_k²`. NaN for SE = 0.
11. **RLS / Browne residual chi²** ✓ — `rls_chi2(samp, implied) → double`. Per block `F_RLS_b = ½·tr((Σ̂_b⁻¹·(S_b − Σ̂_b))²)`, total `T_RLS = Σ_b n_b · F_RLS_b`. Matches lavaan's `test = "browne.residual.nt.model"` (RLS / model-based variant) to 1e-3 on 3F Holzinger (81.3677 vs ML's 85.3055).
12. **Browne residual NT (projected)** ✓ — `browne_residual_nt(pt, rep, samp, est) → double`. Full residual-based stat: `T = N_total · res' U res` where `U = Γ⁻¹ − Γ⁻¹Δ(Δ'Γ⁻¹Δ)⁻¹Δ'Γ⁻¹`, `Γ = Γ_NT(S) = 2D⁺(S⊗S)D⁺ᵀ`, `res = vech(S−Σ̂)`. Matches lavaan's `test = "browne.residual.nt"` to 1e-3 on:
    - 3F Holzinger single-group: 77.9034
    - 3F Holzinger × school configural (multi-group + meanstructure): 98.4422
    - 1F Holzinger × school saturated: ≈ 0

    Implementation never forms the (p* × p*) Γ matrix: uses the identity that `Γ_NT⁻¹·vech(M) = vech(WMW)` with diagonal halved (for symmetric M, W = S⁻¹). Multi-group accumulation uses the joint formulation with per-block weights `n_b/N_total` inside Γ⁻¹ — collapses to lavaan's per-group sum for configural multi-group and correctly threads cross-group equality labels through the joint `Δ'Γ⁻¹Δ` solve. Mean structure: stacked `(m̄_b − μ̂_b ; vech(S_b − Σ̂_b))` per block, μ-part of Γ_NT is `Σ_b`. This sets up the U-matrix scaffolding the robustness phase will plug `Γ̂_empirical` into (sandwich SE, Satorra-Bentler scaling).

## Convenience (any time)

- **Fit measures** ✓ (CFI, TLI, RMSEA done; SRMR + AIC/BIC pending)
  - `fit::baseline_chi2(samp) → BaselineFit` — closed-form independence-model fit: `F_b = log|diag(S_b)| − log|S_b|` per block, so `T_baseline = Σ_b n_b·F_b` and `df_baseline = Σ_b p_b(p_b−1)/2`. No optimizer call needed.
  - `fit::fit_measures(inf, baseline, samp) → {cfi, tli, rmsea}` — standard formulas. Verified against lavaan's reported values for 3F Holzinger (CFI ≈ 0.9305, TLI ≈ 0.8959, RMSEA ≈ 0.0921) to 1e-3.
  - SRMR (needs Σ̂ vs S residual matrix) and AIC/BIC (need log-likelihood) still pending.
- **Warnings vector on `Inference`** — lavaan emits warnings for negative variances etc.

## Robustness + missing data

**Status summary (2026-05): the classical UΓ test-statistic family AND robust SEs are complete.**
We have the information-matrix vocabulary (`Information` / `WeightMoments` /
`ScoreCovariance`, mirroring lavaan's grid minus `first.order`), the U-factor
reduction for both the `Expected` (projector) and `Observed` (non-idempotent,
MLR) breads, all three Γ "meat" flavors (sample / normal-theory /
Browne-unbiased), the UΓ eigenvalue spectrum, the standard robust χ² wrappers
(Satorra-Bentler / mean-and-variance-adjusted / scaled-shifted), and the
sandwich SE `robust_se` (`{Expected,…,Empirical}` ≡ lavaan `se = "robust.sem"`;
`{Observed,…,Empirical}` ≡ `se = "robust.huber.white"` / MLR) — all
golden-checked against lavaan to 1e-3 / 1e-4. The remaining gaps:
- **Browne residual ADF test** (`test = "browne.residual.adf"`) — the
  residual quadratic form `N·res'·U·res` but with empirical `Γ̂` instead of
  `Γ_NT(S)`. `browne_residual_nt` already has the projection machinery; this
  is a one-argument swap of the weight matrix. **Not done.**
- **Mean-structure / multi-group in `build_u_factor` and `robust_se`** —
  both gate `~1` models / multi-group off (Observed-bread is single-block);
  `browne_residual_nt` already handles both, so the stacking convention is
  known. **Not done.**
- **FIML** (missing data) and **bootstrap SEs** — separate tracks, not
  UΓ-related. **Not done.**

The items below need a **raw-data binding** — `SampleStats` (S, m̄, n) isn't
enough. NT estimation and inference, by contrast, work entirely from sample
moments.

### Foundations ✓
- **`RawData`** ✓ — per-block raw observations type in `include/latva/fit/raw_data.hpp`. Carries `X[b]` per-block `(n_b × p_b)` matrices plus a reserved (v0-unused) `mask` for the FIML phase. NT consumers (`fit`, `ExpectedInfoSE`, `browne_residual_nt`) keep consuming `SampleStats` directly and ignore `RawData`.
- **`sample_stats_from_raw(raw) → SampleStats`** ✓ — derives per-block `m̄_b = (1/n_b)·Σ x_i`, `S_b = (1/n_b)·Σ (x_i−m̄_b)(x_i−m̄_b)ᵀ` (lavaan's `likelihood = "normal"` N-divisor convention).
- **`empirical_gamma(X) → MatrixXd`** ✓ — `Γ̂ = (1/n)·Σ (d_i − vech(S))(d_i − vech(S))ᵀ` with `d_i = vech((x_i − m̄)(x_i − m̄)ᵀ)`. The empirical 4th-moment ACOV of vech(S); the building block that swaps in for `Γ_NT` to unlock robust-SE / SB-scaling.
- **`gamma_nt(Sigma) → MatrixXd`** ✓ — analytic `Γ_NT[ij, kl] = σ_ik·σ_jl + σ_il·σ_jk`. Built directly in (p* × p*) without forming the duplication-matrix pseudoinverse. Convergence test: `empirical_gamma → gamma_nt` on `n = 20 000` MVN draws within 5% relative error.

### The information-matrix vocabulary (`Information` / `WeightMoments` / `ScoreCovariance`) ✓
Three orthogonal knobs in `include/latva/fit/robust.hpp`, shared by the SE path (`robust_se`, plus the existing naive-SE classes) and the UΓ test path (`build_u_factor` + `reduced_gamma_*`). Mirrors lavaan's `information` / `h1.information` **minus** `first.order` — for ML the gradient-outer-product `K` is not a distinct quantity, it equals `ΔᵀWΓ̂WΔ` (the empirical meat below), so it's exposed only as a meat choice, never inverted-as-a-bread.
- **`Information { Expected, Observed }`** — the q×q "bread" that gets inverted. `Expected` = `J = ΔᵀWΔ` (the GLS/Fisher form; for the test keeps `U` a rank-`df` projector — also ≈ lavaan `observed.information = "h1"` for the ML discrepancy). `Observed` = `H = H1+H2` (the actual ML Hessian — the `H2` term makes it ≠ `J`; for the test `U` is non-idempotent; the MLR convention).
- **`WeightMoments { Structured, Unstructured }`** — which moments build `W = Γ_NT(M)⁻¹` (= lavaan `h1.information`). `Structured` = Σ̂ (model-implied, default); `Unstructured` = S (sample, the weight `browne_residual_nt` uses).
- **`ScoreCovariance { ModelImplied, Empirical, BrowneUnbiased }`** — the "meat" ACOV of vech(S). `ModelImplied` = `Γ_NT` (SE collapses to bread⁻¹ = the naive SE; test `U` is a projector, no χ² scaling); `Empirical` = `Γ̂` 4th-moment (needs raw data; SE = full sandwich; test = the SB-family χ² scaling); `BrowneUnbiased` = the distribution-free correction.
- Lavaan estimator-shorthand mapping: naive `se = "standard"` ≡ `ExpectedInfoSE` (= `{Expected, Structured, ModelImplied}`); `information = "observed"` naive ≡ `Fd`/`AnalyticObservedInfoSE` (= `{Observed, Structured, ModelImplied}`); `estimator = "MLM"` ≡ `{Expected, Structured, Empirical}` (robust.sem SE + SB χ²); `estimator = "MLR"` ≡ `{Observed, Structured, Empirical}` (robust.huber.white SE + YB χ²).

### Eigenvalues + robust statistics + robust SEs ✓
`include/latva/fit/robust.hpp` exposes the reduced eigenvalue path for `UΓ`, three lavaan-compatible robust χ² wrappers, and the sandwich SE. For the `Expected` bread the core trick (via GPT-5.5, verified) is `U = B·Bᵀ` with `B = L·N`, `L = Cholesky(Γ_NT⁻¹)`, `N = orthonormal basis of ker((LᵀΔ)ᵀ)` ⇒ `eigvals(UΓ) = eigvals(BᵀΓB)` (`df × df` symmetric eigenproblem). For the `Observed` bread `U = L_Γ⁻ᵀ·(I − A·H_obs⁻¹·Aᵀ)·L_Γ⁻¹` (non-idempotent) ⇒ `eigvals(UΓ) = eigvals(R̃ᵀ·(I − A·H_obs⁻¹·Aᵀ)·R̃)` where `R̃·R̃ᵀ = L_Γ⁻¹·Γ·L_Γ⁻ᵀ` (`p* × p*`; the spectrum is genuinely indefinite — negative eigenvalues — when the model is misspecified). Never forms `UΓ`; never forms `Γ` when structure permits.

- **`build_u_factor(pt, rep, samp, est, spec={}) → UFactor`** ✓ — `spec.bread == Expected` ⇒ `kind = ProjectionExpected` (LLT(`Γ_NT(M)`) + `A = L_Γ⁻¹Δ` + QR → `N` + `B = L_Γ⁻ᵀN`); `spec.bread == Observed` ⇒ `kind = ObservedHessian` (single-block; `A`, `H_obs⁻¹` via `AnalyticObservedInfoSE` with FD fallback). Caches per-block `L_Γ`, `Σ̂_b`, `S_b`. The `Expected` projection path stacks per block (multi-block); `Observed` is single-block in v1. Mean structure gated off in v0.
- **Three Γ flavors** ✓ — `reduced_gamma_{sample,nt,unbiased}(uf, …)` produce the M-matrix `ugamma_eigenvalues` then eigensolves. For `ProjectionExpected`: `M = BᵀΓB` (`df × df`; `reduced_gamma_nt` is operator-only `vech(M) ↦ vech(2·Σ̂·M_½·Σ̂)`, eigenvalues exactly `(1,…,1)`). For `ObservedHessian`: `M = R̃ᵀ·(I − A·H_obs⁻¹·Aᵀ)·R̃` (`p* × p*`; `reduced_gamma_nt` short-circuits since `M̃ = I`). `reduced_gamma_unbiased` is `ProjectionExpected`-only; `reduced_gamma_sample_streaming` likewise.
- **`casewise_contributions(raw, samp)`** ✓ — builds `Z_c` from `RawData` (row `i` of block `b`: `vech((x_i − m̄_b)(x_i − m̄_b)ᵀ) − vech(S_b)`, stacked block-major).
- **`ugamma_eigenvalues(M)`** ✓ — `SelfAdjointEigenSolver(M, EigenvaluesOnly)`, symmetrises first.
- **`satorra_bentler` / `mean_var_adjusted` / `scaled_shifted`** ✓ — `T_SB = T_ML/(Σλ/df)`; Satterthwaite `df_adj = (Σλ)²/Σλ²`, `T = T_ML·Σλ/Σλ²`; SB-2010 `a = √(df/Σλ²)`, `b = df − a·Σλ`, `T = T_ML·a + b`. Match lavaan `satorra.bentler` / `mean.var.adjusted` / `scaled.shifted` to 1e-3.
- **`robust_se(pt, rep, samp, est, {gamma_hat | raw}, spec={Expected,Structured,Empirical}) → RobustSeResult`** ✓ — sandwich `vcov = (1/N)·bread⁻¹·meat·bread⁻¹`, `bread = J = ΔᵀWΔ` (Expected) or `H_obs` (Observed), `meat = ΔᵀWΓ̂WΔ = (Z_c·WΔ)ᵀ(Z_c·WΔ)/N` (RawData overload — never forms the p* × p* matrix) or `(WΔ)ᵀΓ̂(WΔ)` (gamma_hat overload). Collapses to the naive expected `vcov` when `Γ̂ → Γ_NT`. Matches lavaan `se = "robust.sem"` (Expected bread) and `se = "robust.huber.white"` / MLR (Observed bread) to 1e-4 on 3F Holzinger (using lavaan's stored NACOV as Γ̂). Single-block in v1; `BrowneUnbiased` meat not yet wired.

### Still pending
- **Browne residual ADF test** (`test = "browne.residual.adf"`) — same `N·res'·U·res` form `browne_residual_nt` already computes, but with empirical `Γ̂` (from `casewise_contributions`) as the weight matrix instead of `Γ_NT(S)`. A one-argument generalization.
- **Mean-structure + multi-group U-factor** — the eigenvalue pipeline currently gates these off (`build_u_factor` errors on `~1` models / multi-block). Adding them is the same stacking convention `browne_residual_nt` uses: per block, the layout is `(μ_b ; vech(Σ_b))` and the μ-block of `Γ_NT` is just `Σ̂_b` (no diag halving).
- **FIML** (full-information ML with missing data) — per-case likelihood `Σ_i log f(x_i | Σ_{i}, μ_{i})` where `Σ_i, μ_i` are the rows/cols of `Σ, μ` for observation i's observed pattern. Needs raw data + missingness mask (`RawData::mask` is reserved for it). With complete data FIML reduces to standard ML. **Separate track — not UΓ-related.**
- **Bootstrap SE** — resampling cases, by definition raw-data only. `PostError::BootstrapFailed` is already in place. **Separate track — not UΓ-related.**

## Verification

After P8 lands:
```sh
cmake --build --preset asan
ctest --preset asan
```
Should report **~94 tests passing** (91 current + unit tests + golden). The golden message should read something like `inference goldens: 9 / 9 pass` for SE/χ² matching lavaan within tolerances.
