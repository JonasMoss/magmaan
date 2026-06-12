# Test Ledger

This ledger maps magmaan's current validation surface by subsystem. It is a
maintainer aid, not a second backlog: use it to find the tests, fixtures, and
reports that protect an area before editing it, then keep remaining-work details
in [`docs/backlog/todo.md`](../backlog/todo.md) or the simulation-specific
backlog.

## How to Use This Ledger

- Start with the CTest label that owns the area: `spec`, `estimate`,
  `inference`, `ordinal`, `api`, `sim`, `parity`, or `robcat`.
- Prefer the narrow loop while developing, for example
  `just test-area estimate FIML` or `just test-area inference score`.
- Run `just test-quick` before handing back ordinary C++ changes; add
  `just test-area parity` or `just r-check` when the edited behavior crosses
  real-data parity or R boundary code.
- Treat advisory checks under `tests/checks/` and experiments as evidence for
  research claims, not as default CI gates.

## Regression Notes

When a fixed bug gets a guard test, preserve the bug shape and the protecting
test so future edits know why the assertion exists. Use this compact format:

```text
Regression: <short symptom and root cause>.
Guard: <test, fixture, example, or report that fails if it comes back>.
Scope: <optional remaining gap or intentionally uncovered cases>.
```

Put the note near the focused test when the guard is local and easy to find.
Put it in this ledger when the bug crosses subsystems, depends on an external
oracle, or needs a maintainer to run a non-obvious report. Keep notes to the
reason the test exists; unresolved work still belongs in the backlog.

The notes below are cross-subsystem, oracle-dependent fixes. Full root-cause
write-ups live in the commits that introduced each guard.

**Mixed Γ ignored the continuous stage-1 estimating equations.**
Regression: the mixed continuous/ordinal NACOV put continuous means/variances
and continuous-continuous covariances into Γ̂ as raw moment residuals and
patched the polyserial variance channel additively, instead of routing them
through the muthen1984 stage-1/stage-2 sandwich. Block-partition diffs vs
lavaan reached 0.65 (variance rows) while pure-ordinal blocks matched at 1e-7;
the mixed goldens hid it behind NACOV ≤ 1.2 / robust ≤ 4.5e-1 gates. Root
cause: missing mu/var stage-1 scores in `A11`, missing pair-score mu/var
coupling channels in `A21`, zero scores for continuous-continuous pairs, and
the delta-rule `H` applied to raw residuals instead of post-sandwich variance
influence.
Guard: `tests/golden/ordinal_golden_test.cpp` mixed stats gates (NACOV/W
1.2 → 1e-6), mixed fit gates (θ 2e-2 → 1e-5, χ² 3e-1 → 5e-3 with the
convention rescale), mixed robust gates at all-ordinal tightness (SE 3e-4,
eig 2e-4, scales 2e-4, χ²-scale 5e-3), plus an own-θ̂ robust check;
`tests/unit/ordinal_test.cpp` lazy-workspace diagonal parity (1e-8) and the
Huber no-clip ≡ ML Γ identity.
Scope: polyserial-DPD Γ stays a research surface (ML-identity mu/var
channels); the ≥2-ordinal Huber branch keeps its per-pair scalar-bread
polyserial influence.

**Ordinal golden chisq gate absorbed the lavaan `(N−G)` convention.**
Regression: the ordinal goldens compared `2N·fmin` directly against lavaan's
categorical-LS statistic `Σ_g (n_g−1)·F̂_g` with an 8e-2 absolute gate, which
silently absorbed the entire convention gap for fixtures 0001–0012 (e.g. 0001:
diff 8.7e-3, 0004 WLS: 5.6e-2); the larger-χ² threshold-invariance fixture 0013
finally exceeded the slack and exposed it. Same shape as the continuous GLS/WLS
audit-item template. The lavaan source factor is per-group
(`lav_model_objective.R`: `group.fx = 0.5·(nobs−1)/nobs·group.fx`), which also
makes the *estimator* differ once equality constraints couple groups (θ̂ shift
`O(1/n_g)`; see `docs/design/numerical-conventions.md` exception 4).
Guard: `tests/golden/ordinal_golden_test.cpp` (`to_lavaan_ls_chisq` rescale,
χ² gates 8e-2 → 5e-3, robust χ²-scale gates 8e-2 → 5e-3; documented 1.5e-4 θ /
1.5e-2 scaled-shifted exceptions for fixture 0013 only).
Scope: mixed bounded θ/χ² gates remain loose (2e-2/3e-1) pending the mixed
robust-parity root-cause in the backlog; the SS shift is N-free and compared
unrescaled.

**Robust UΓ projector per-group weight.**
Regression: the complete-data `build_u_factor` ProjectionExpected path built the
kernel basis from `A_b = L_Γ,b⁻¹·Δ_b` without the per-group weight `w_b = n_b/N`,
so the reduced UΓ spectrum (SB scaling, FMG p-values, robust difference test) was
off by up to ~1% for models with both unequal group sizes and a cross-group
equality constraint. Masked for single-group, equal-group, and configural cases.
Guard: `tests/unit/fiml_test.cpp` (metric-invariance Unstructured degeneracy
~1e-6); `experiments/21-fiml-measurement-invariance-fmg --lavaan-parity`.
Scope: Expected bread only (the FMG path); the Observed-bread spectrum tail is
left as-is. `robust_se` uses its own w_b-weighted bread and was unaffected.

**Ordinal/mixed standardized delta-unit.**
Regression: the generic `λ·√Var(η)/√σ_rr` formula divided by `σ_rr ≈ λ²ψ+1`, but
a delta-ordinal `y*` is unit-variance, so a true .6 loading came back ~.52; the
path had been guarded to refuse ordinal fits. `standardize_all` now takes
`ordinal_delta_unit` and standardizes ordinal-indicator loadings (the `Lambda`
and all-y `Beta` slots) by the latent SD only (σ_rr = 1).
Guard: `r-package/examples/ordinal_dwls_wls.R`; `experiments/16-li-2021-mixed`
and `19-li-2016-ordinal` `--lavaan-parity` (≤~1e-6 vs `standardizedSolution`).
Scope: no checked C++ golden for ordinal-SEM standardized rows yet;
`compute_defined`/`factor_scores` stay `require_not_ordinal`-guarded (in backlog).

**Mixed-ordinal NACOV eager inversion for DWLS.**
Regression: `mixed_ordinal_stats_from_data` eagerly inverted the full NACOV to
build the WLS weight and errored if it was not PD — wasting an O(m³)
factorization for DWLS (which needs only diag Γ) and failing every mixed DWLS fit
at small N with many indicators (Li-2021 20-var SEM, N=200). `full_wls_weight`
now skips the inverse for DWLS-only callers, and even when requested the inverse
is non-fatal (a singular NACOV leaves `W_wls` empty; DWLS/robust proceed). Ported
to the all-ordinal path too.
Guard: `tests/golden/ordinal_golden_test.cpp` keeps the eager W_wls-vs-lavaan
contract; `experiments/19-li-2016-ordinal`.
Scope: an explicit full-WLS fit on an empty weight reports it via
`validate_stats` / `weight_factors`.

**Continuous ULS standard-vs-robust base — not a bug (do not re-chase).**
Recorded so it is not re-investigated: ULS uses different base statistics for the
standard (`continuous_ls_chisq` Browne residual NT) vs robust
(`robust_continuous_ls`, scaling off `2N·fmin`) paths. This looks inconsistent
but faithfully mirrors lavaan's default-ULS Browne test vs its `se="robust.sem"`
unscaled base.
Guard: `tests/golden/lavaan_parity_golden_test.cpp` ULS
`chisq_standard`/`satorra_bentler`/`scaled_shifted` against the lavaan
`hs_3factor_ls*` fixtures; a "uniform Browne base" rewrite breaks them.
Scope: see memory `uls-robust-test-base-bug`.

**GLS/WLS standard χ² multiplier (N vs N−G) — convention, not a bug.**
Continuous GLS/WLS standard χ² uses `2N·fmin = N·F`, whereas lavaan reports
`(N−G)·F`; the ratio is exactly `(N−G)/N`. Decision: keep magmaan's `N`
multiplier. ULS already carries `N−G` via `browne_residual_nt`, so it matches
lavaan directly.
Guard: `lavaan_parity_golden_test.cpp` and `ls_golden_test.cpp` pin
`magmaan·(N−G)/N == lavaan` to 5e-3 (previously a loose gate absorbed the gap).
Scope: documented in `docs/design/numerical-conventions.md`; memory
`magmaan-gls-chi-convention`.

**½-everywhere objective unification (2026-06-09) — convention.**
`est.fmin` was overloaded (continuous LS stored ½F; ML/FIML/ordinal stored full
F, ordinal via an explicit `2·fmin` doubling), with the χ² multiplier flipping
between N and 2N to compensate. Unified so every estimator stores `fmin = ½F` and
`T = 2N·fmin = N·F`; the ½ lives only in the optimiser adapters, the math kernels
stay full-F, so information/SE/score paths are untouched and χ² values are
unchanged.
Guard: the single contract is documented at `inference::chi2_stat`; deliberate
exceptions (ULS Browne, FIML LRT, test-side `(N−G)/N`) recorded there and in
`docs/design/numerical-conventions.md`.

**Kline/Guo duplicated lavaanify term.**
Regression: duplicated formula terms (`NA*LM1 + c(a1,a2)*LM1`) produced two
partable rows for one matrix cell, leaving a phantom free parameter that moved
the analytic gradient but not the model moments. `lavaanify` now merges repeated
`lhs op rhs` within a block (`build_group_template`, `src/spec/build.cpp`).
Guard: root-cause cases in `tests/unit/lavaanify_test.cpp`; end-to-end Guo
invariance rungs in `tests/golden/textbook_corpus_golden_test.cpp`
(`case_exports.json`).
Scope: lavaan's `guo_mi_strong` reference is under-converged (magmaan reaches a
strictly lower chisq at equal df, confirmed by three optimizers), so that rung is
gated df-exact + no-worse-than-oracle with two-sided chisq parity skipped.
Per-parameter θ̂/SE parity deferred (needs a lavaan→magmaan param map — backlog).

**ADF/WLS Γ̂ eigen-gated inverse.**
Regression: a barely-PD but numerically rank-deficient empirical Browne NACOV
(reproducer `muthen_2017_ch2_ex2_1__adf`, rcond≈5e-18) passed a bare
`Eigen::LLT`; the fit reached the correct saturated θ̂ but a ~1e16 weight
eigendirection amplified residuals into `grad_inf≈3.86`, tripping the terminal
audit. The continuous ADF/WLS builders (`dls_weight`, `structured_gamma_weight`)
now invert through `detail::symmetric_inverse_pd_gated` (`tol=1e-10·max(1,λmax)`);
a rank-deficient Γ̂ returns `FitError::NumericIssue` with dim/rank/rcond/λmin.
Guard: `tests/unit/detail_linalg_test.cpp` (RNG-free muthen-spectrum pin);
rank-deficient rejection in `dls_weight_test.cpp`.
Scope: an optional `spectral_truncate` parity-restore policy is not built
(backlog); `experiments/00-lavaan-parity` inverts the raw NACOV in R and is not
routed through the gate.

**FMG unbiased-Gamma NT-term.**
Regression: the fused FMG unbiased path special-cased the NT term as the identity
(`B'Γ_NT(Σ̂)B = I`) and applied Browne's correction as a `-bI` shift. Wrong: the
Du-Bentler unbiased Gamma is distribution-free with NT term `Γ_NT(S)` at the
*sample* covariance, not the model-implied Σ̂. This made every `_ug` test (incl.
the defaults `SB_UG_RLS`, `pEBA2_UG_RLS`) silently model-dependent and broke
semTests parity by up to ~4e-4. Fixed by `reduced_gamma_nt_sample()`
(`B'Γ_NT(S)B`); parity back to ~1e-8.
Guard: `examples/fmg.R` value-for-value vs `semTests::pvalues` (<1e-6); unit pin
`reduced_gamma_nt_sample` in `tests/unit/robust_test.cpp`.
Scope: the row-space unbiased optimization and rank-one secular update that
relied on the same wrong identity are deferred (see `docs/backlog/speculative.md`).

**Satorra-2000 nested-test parity oracle (not a bug).**
Regression: an apparent ~13% scale gap vs lavaan's
`lavTestLRT(method = "satorra.2000")` came from lavaan's `A.method` *default*
(`"delta"`, a moment-Jacobian column-space construction for covariance-nested
models), not from the Satorra-2000 scaling formula. magmaan's
`robust_nested_lrt(method = "restriction_map")` uses the exact
parameter-restriction matrix and reports the mean-scaled `T/c`.
Guard / oracle: regenerate fixtures and diff against
`lavTestLRT(fit_h1, fit_h0, method = "satorra.2000", A.method = "exact",
scaled.shifted = FALSE)` — NOT the bare default; `tests/unit/satorra2000_test.cpp`,
`r-package/examples/nested_test_satorra2000.R`.
Scope: lavaan's bare default (`A.method = "delta"`, `scaled.shifted = TRUE`) is a
documented compatibility alternative for covariance-nested checks, not the
magmaan oracle. Full investigation: `docs/validation/satorra2000_parity.md`.

**Ordinal api standardize/compute_defined fed the un-prepared structure.**
Regression: `api::standardize_lv`/`standardize_all`/`compute_defined` dropped
their `require_not_ordinal` guard but never worked for ordinal/mixed fits — the
api `Fit` stores the un-prepared `LatentStructure` (latent-response residual
variances still free) while `fit_ordinal_bounded` fits over the *prepared*
partable (those fixed, free set compacted), so `estimates().theta` and the
robust vcov are reduced. Standardize fed the reduced theta into the un-prepared
evaluator and aborted (`theta has size N; ModelEvaluator expects N+p_ord`);
`compute_defined` would have indexed the wrong free slots. Only the Rcpp path
worked, because `ctx_from_fit` parses the prepared partable. Fix: the api
reconstructs the prepared structure on demand (`prepared_structure` in
`src/api/sem.cpp`, replaying `prepare_ordinal_partable`).
Guard: `tests/unit/api_sem_test.cpp` ordinal case (standardize_lv/all +
compute_defined succeed, factor_scores stays guarded); live lavaan value-parity
in `r-package/examples/ordinal_dwls_wls.R` (ordinal `:=` to 5e-3, mixed std.all
to 1e-3).
Scope: the C++ *golden* (stored lavaan oracle) for ordinal standardized rows is
still pending a lavaan-pin realignment — see `docs/backlog/todo.md`.

## Validation Areas

| Area | Oracle | Protection | Important files/tests | Known gaps |
|---|---|---|---|---|
| Parser and lexer | `docs/grammar/grammar.ebnf`, checked parser fixtures | Unit plus golden tests under `spec` | `tests/unit/lexer_test.cpp`, `tests/unit/parser_test.cpp`, `tests/golden/lexer_golden_test.cpp`, `tests/golden/parser_golden_test.cpp` | Grammar-coverage walk remains manual; grammar changes must edit EBNF first. |
| Lavaanify, spec, and partable projection | lavaan `parTable()` fixtures and corpus exports | Unit, golden, and corpus parity under `spec` and `parity` | `tests/unit/lavaanify_test.cpp`, `tests/golden/lavaanify_golden_test.cpp`, `tests/golden/textbook_corpus_golden_test.cpp` | Little/Newsom and Mplus corpus promotion remains ongoing. |
| Matrix representation and model evaluation | lavaan implied moments plus algebraic invariants | Unit plus golden tests under `spec` | `tests/unit/matrix_rep_test.cpp`, `tests/unit/model_evaluator_test.cpp`, `tests/golden/matrix_rep_golden_test.cpp`, `tests/golden/fit_implied_golden_test.cpp` | Some observed fixed.x path-model implied-moment comparisons remain documented parity exceptions. |
| Complete-data ML and LS estimation | lavaan JSON fixtures and real-data parity fixtures | Unit, golden, and parity tests under `estimate` and `parity` | `tests/unit/ml_test.cpp`, `tests/unit/ls_path_test.cpp`, `tests/golden/ls_golden_test.cpp`, `tests/golden/lavaan_parity_golden_test.cpp` | Tolerance audit is still open; Geiser GLS exceptions need tighter documentation or fixes. |
| FIML and missing data | lavaan FIML fixtures, saturated EM invariants, FIML FMG diagnostics | Unit, golden, R examples, and advisory checks | `tests/unit/fiml_test.cpp`, `tests/golden/fiml_golden_test.cpp`, `r-package/examples/fiml.R`, `tests/checks/fiml_fmg_trace/` | High-level `magmaan(estimator = "FIML")` mean-structure defaults and multi-group starts need care. |
| Ordinal and mixed moments | lavaan ordinal fixtures, robcat fixtures, internal moment invariants | Unit, golden, robcat, R examples, and experiments | `tests/unit/ordinal_test.cpp`, `tests/golden/ordinal_golden_test.cpp`, `tests/golden/robcat_parity_golden_test.cpp`, `r-package/examples/ordinal_dwls_wls.R` | Mixed robust scaled-test parity is still loose; lazy mixed WLS and mixed theta SNLLS remain open. |
| Inference, standardization, and fit measures | lavaan SE, score, standardized, and fit-measure fixtures | Unit plus golden tests under `inference` | `tests/unit/inference_test.cpp`, `tests/unit/score_test.cpp`, `tests/unit/standardized_test.cpp`, `tests/golden/inference_golden_test.cpp`, `tests/golden/standardized_golden_test.cpp` | Ordinal defined-parameter validity and additional ordinal-SEM standardized goldens remain follow-ups. |
| Robust tests, FMG, and nested restrictions | lavaan/semTests parity, R-internals fixtures, weighted-chi-square oracles | Unit, golden, R examples, and advisory checks | `tests/unit/robust_test.cpp`, `tests/unit/fmg_test.cpp`, `tests/golden/score_robust_golden_test.cpp`, `r-package/examples/fmg.R` | Robust MI is still complete-data ML single-group only; self-contained FMG p-value transform goldens are wanted. |
| Optimizers and terminal audits | Recomputed objectives, projected gradients, cross-backend agreement | Unit tests and benchmark/report tracks under `estimate` | `tests/unit/terminal_audit_test.cpp`, `tests/unit/optimizer_crosscheck_test.cpp`, `tests/unit/fit_diagnostics_test.cpp`, `docs/design/terminal-audit.md` | Ultimate verifier and stationarity tolerance calibration remain research work. |
| Simulation | Distribution goldens, deterministic calibration fixtures, stochastic smokes | Unit tests under `sim` plus advisory checks | `tests/unit/norta_test.cpp`, `tests/unit/plsim_test.cpp`, `tests/unit/vale_maurelli_test.cpp`, `tests/checks/plsim/` | Model-implied simulation, ordinal/mixed observed-correlation calibration, and persistent caches remain open. |
| R boundary and examples | lavaan parity through examples and R-shaped wrapper checks | `just r-check` examples plus C++ API tests | `tests/unit/api_sem_test.cpp`, `r-package/examples/*.R`, `r-package/examples/tutorial/run_all.R` | Examples are smoke tests, not exhaustive wrapper coverage; R reconstruction is sensitive around means and groups. |
| Composite frontier | lavaan native composite fixtures and FC-SEM evaluator invariants | Unit, golden, and R frontier example tests | `tests/unit/fcsem_evaluator_test.cpp`, `tests/unit/fcsem_ml_test.cpp`, `tests/golden/composite_golden_test.cpp`, `r-package/examples/fcsem_frontier.R` | Post-fit native W-matrix parity validation is still skipped while fixture ownership is settled. |
| Corpus parity | lavaan-generated real-data and textbook fixtures | Heavy `parity` target and corpus-specific goldens | `tests/golden/geiser_golden_test.cpp`, `tests/golden/mplus_sem_golden_test.cpp`, `tests/golden/paper_corpus_golden_test.cpp`, `tests/golden/textbook_corpus_golden_test.cpp` | Corpus breadth is intentionally staged; some cases document alternate optima or unsupported syntax. |

## High-Risk Map

### FIML and Missing Data

Protected by:

- `tests/unit/fiml_test.cpp`
- `tests/golden/fiml_golden_test.cpp`
- `r-package/examples/fiml.R`
- `tests/checks/fiml_fmg_trace/` and `tests/checks/fiml_fmg_nested/`

Known weak spots: high-level mean-structure defaults, multi-group starts, and
nonlinear equality tangent-space support remain backlog items.

### Ordinal and Mixed Moments

Protected by:

- `tests/unit/ordinal_test.cpp`
- `tests/golden/ordinal_golden_test.cpp`
- `tests/golden/pairwise_golden_test.cpp`
- `tests/golden/robcat_parity_golden_test.cpp`
- `r-package/examples/ordinal_dwls_wls.R`
- `experiments/_archive/10-ordinal-inference-cache-probe` and
  `experiments/_archive/11-ordinal-snlls-speed`

Known weak spots: mixed robust scaled-test parity has loose guards, and lazy
mixed WLS construction plus mixed theta SNLLS are still open.

### Robust U-Gamma and FMG Reductions

Protected by:

- `tests/unit/robust_test.cpp`
- `tests/unit/fmg_test.cpp`
- `tests/unit/weighted_chisq_test.cpp`
- `tests/golden/score_robust_golden_test.cpp`
- `r-package/examples/fmg.R`
- `tests/checks/robust_score/`

Known weak spots: robust MI has deferred estimator tiers and no df-greater-than
one joint-release path yet; FMG p-value transforms still need a self-contained
C++ golden independent of R examples.

### Optimizer Terminal Audit

Protected by:

- `tests/unit/terminal_audit_test.cpp`
- `tests/unit/fit_diagnostics_test.cpp`
- `tests/unit/optimizer_crosscheck_test.cpp`
- `tests/golden/lavaan_parity_golden_test.cpp`
- `docs/design/terminal-audit.md`

Known weak spots: the absolute stationarity tolerance is provisional, and SNLLS
still needs a post-hoc full-theta audit path for apples-to-apples diagnostics.

### Parser and Lavaanify

Protected by:

- `docs/grammar/grammar.ebnf`
- `tests/unit/parser_test.cpp`
- `tests/unit/lavaanify_test.cpp`
- `tests/golden/parser_golden_test.cpp`
- `tests/golden/lavaanify_golden_test.cpp`
- `tests/golden/textbook_corpus_golden_test.cpp`

Known weak spots: grammar coverage is not yet mechanically reported, and
external corpus promotion is still staged case by case.

### R Boundary Reconstruction

Protected by:

- `tests/unit/api_sem_test.cpp`
- `r-package/examples/high_level_magmaan.R`
- `r-package/examples/model_spec_df_to_data.R`
- `r-package/examples/lavaan_partable_comparison.R`
- `r-package/examples/fit_measures.R`
- `r-package/examples/fmg.R`

Known weak spots: R examples catch workflow regressions but are not exhaustive;
mean-structure, group, ordinal, and post-fit reconstruction paths remain the
places to validate deliberately after R glue edits.
