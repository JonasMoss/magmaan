# Test Ledger

This ledger maps magmaan's current validation surface by subsystem. It is a
maintainer aid, not a second backlog: use it to find the tests, fixtures, and
reports that protect an area before editing it, then keep remaining-work details
in [`project/backlog/todo.md`](../backlog/todo.md) or the simulation-specific
backlog.

## How to Use This Ledger

- Start with the CTest label that owns the area: `spec`, `estimate`,
  `inference`, `ordinal`, `api`, `sim`, `parity`, or `robcat`.
- Prefer the narrow loop while developing, for example
  `just test-area estimate FIML` or `just test-area inference score`.
- Run `just test-quick` before handing back ordinary C++ changes; add
  `just test-area parity` or `just r-check` when the edited behavior crosses
  real-data parity or R boundary code.
- Treat advisory checks under `cpp/tests/checks/` and experiments as evidence for
  research claims, not as default CI gates.
- This ledger records magmaan bugs we fixed. For the opposite — the rare cases
  where the *oracle* (lavaan, etc.) is provably wrong and magmaan is right — see
  [`oracle-defects.md`](oracle-defects.md).

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

**`group.equal = "intercepts"` kept the latent means fixed.**
Regression: `spec::build` tied indicator intercepts across groups but left the
auto-added latent means fixed at 0 in every group, whereas lavaan frees them in
groups 2+ unless `means` is also equal. Keyword scalar invariance therefore
fitted a more restricted model than lavaan (HS 1939 by school: fmin .163
against lavaan's .106). The R invariance helpers wrote `f ~ c(0, NA)*1`
explicitly and were unaffected; the ordinary-user package's lavaan-MLR
standard-error gate exposed it.
Guard: `cpp/tests/unit/group_equal_means_test.cpp` (release, `means`, user-fixed
means, `group.partial`, three-group marker/std.lv release, per-group fixed/free
means and growth identification, read from lavaan 0.7.2);
`r-package/tests/testthat/test-scalar-invariance.R` gates ML/FIML parameter-row
identity, free/fixed status, estimates, df and chi-square; three-group CFA and
growth fits; and ML/FIML scalar nested tests with the delta map. FIML coverage
includes marker/std.lv identification and overlapping missingness patterns.
`r-magmaan/tests/testthat/test-magmaan.R`
matches lavaan's MLR standard errors under `group.equal = c("loadings",
"intercepts")`.
Scope: the closed-form (Guttman) grouped path still rejects free latent means;
its intercept-equality test now requests `means` explicitly.

**FIML lavaan-convention nested test used the wrong moment reference.**
Regression: with HS school metric/scalar fits and every seventh `x2` missing,
`convention="lavaan"` returned scaled difference 22.13388 versus lavaan MLR's
22.81168. Estimates and the unscaled difference agreed; substituting lavaan's
estimates did not remove the discrepancy. Independent matrix reconstruction
from public lavaan outputs isolated two ingredients: its expected moment weight
uses the fitted larger-model covariance, while Gamma is the saturated EM
sandwich (observed saturated Hessian and saturated casewise scores). The former
implementation instead used the saturated covariance in the weight and zeroed
missing raw-moment residuals in Gamma. The compatibility path now uses
`Gamma_g = n_g * acov_g`, retaining observed parameter bread and the delta
restriction map. Native defaults and ML2S are unchanged.
Guard: `cpp/tests/golden/fiml_golden_test.cpp` reads the frozen
`fiml/scalar_invariance_hs_nested.json` fixture at oracle estimates and compares
streaming/materialized/dense results (scale 1e-5, statistic 1e-4 absolute).
Regenerate through `regen_oracle.R` or its `regen_oracle_fiml_nested.R` slice.
The R scalar-invariance suite also compares fitted marker/std.lv pairs, df,
unscaled/scaled differences, scale and p-value; the former skip is removed.
Scope: direct-FIML empirical-Gamma Satorra-2000 compatibility with lavaan MLR's
default observed/structured information settings. The opt-in covariance
regularizer now affects the saturated ACOV through its existing delta-method
transformation; its calibration remains separate backlog work.

**Convergence was returned without covariance admissibility.**
Regression: complete-data ML/LS checked the final implied observed `Sigma`
only, FIML did not run the shared L2 finalizer, and R exposed no structured
answer to whether the assembled reduced-LISREL `Theta` and `Psi` were
covariance matrices. A converged Heywood solution could therefore have PD
observed moments while carrying a negative residual variance, with no
high-level warning; positive variances and valid pairwise correlations also
could hide a higher-dimensional joint PSD failure.
Guard: `cpp/tests/unit/fit_diagnostics_test.cpp` separates observed-`Sigma` PD from
component PSD with deterministic negative-residual and three-factor joint-PSD
cases; `cpp/tests/unit/fiml_test.cpp` requires fitted FIML estimates to carry the
same audit; `r-package/tests/testthat/test-admissibility.R` checks the R schema,
single warning, unchanged convergence flag, source partable rows, FIML path,
and printed status.
Scope: fitting remains unconstrained for lavaan compatibility. A future
frontier covariance-domain optimizer needs boundary-aware inference; ordinal
and native FC-SEM fits do not yet run this ordinary MatrixRep finalizer.

**FIML fit measures used complete-data ML plumbing.**
Regression: the R `fit_measures()` wrapper treated FIML fits like complete-data
ML by deriving `chisq = 2N*fmin` and calling the sample-statistics baseline
helper, so complete-data FIML could report impossible ordinary CFI/RMSEA values;
automatic robust FIML CFI/RMSEA also lacked the baseline scaled fields even
though the core `fiml_corrected_fit_measures()` reduction already computed them.
Guard: `r-package/tests/testthat/test-fit-measures.R` mocks the FIML branch so
the complete-data helpers fail if reached, then fits the same complete dataset
under FIML and ML and requires ordinary `chisq`/baseline/CFI/RMSEA parity while
checking finite FIML robust baseline, CFI, and RMSEA fields.
Scope: complete-data MLM/MLR and ordinal WLSMV automatic robust fit-measure
dispatch remain backlog items.

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
Guard: `cpp/tests/golden/ordinal_golden_test.cpp` mixed stats gates (NACOV/W
1.2 → 1e-6), mixed fit gates (θ 2e-2 → 1e-5, χ² 3e-1 → 5e-3 with the
convention rescale), mixed robust gates at all-ordinal tightness (SE 3e-4,
eig 2e-4, scales 2e-4, χ²-scale 5e-3), plus an own-θ̂ robust check;
`cpp/tests/unit/ordinal_test.cpp` lazy-workspace diagonal parity (1e-8) and the
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
`O(1/n_g)`; see `project/design/numerical-conventions.md` exception 4).
Guard: `cpp/tests/golden/ordinal_golden_test.cpp` (`to_lavaan_ls_chisq` rescale,
χ² gates 8e-2 → 5e-3, robust χ²-scale gates 8e-2 → 5e-3; documented 1.5e-4 θ /
1.5e-2 scaled-shifted exceptions for fixture 0013 only) and the real-data bfi
ordinal block in `cpp/tests/golden/lavaan_parity_golden_test.cpp` (stale relative
χ² gates replaced by the same lavaan-scale 5e-3 checks).
Scope: the scaled-shifted shift is N-free and compared unrescaled; fixture 0013
keeps its documented cross-group weighting exception only.

**Saturated-model TLI was hidden by a finite-oracle soft skip.**
Regression: fit-measure goldens skipped comparisons when magmaan returned a
non-finite value even if the lavaan fixture was finite. Tightening that audit
check exposed saturated user models (`df_user = 0`) where lavaan reports
`tli = 1` but magmaan returned `NaN`.
Guard: `measures::fit_measures()` now follows lavaan's saturated-model
convention; `cpp/tests/unit/fit_measures_test.cpp` pins the edge case and
`cpp/tests/golden/fit_measures_golden_test.cpp` fails finite lavaan oracle fields
when magmaan is non-finite.
Scope: non-finite lavaan fixture fields remain skipped because there is no
numeric oracle to compare.

**Golden soft skip sets are count-pinned.**
Regression: `MESSAGE`-only tolerated/skipped/deferred/no-oracle buckets could
absorb new fixture drift without failing the suite.
Guard: `lavaanify_golden_test.cpp`, `matrix_rep_golden_test.cpp`,
`fit_implied_golden_test.cpp`, `fit_measures_golden_test.cpp`,
`fit_theta_golden_test.cpp`, `inference_golden_test.cpp`,
`test_stats_golden_test.cpp`, and `observed_inference_golden_test.cpp` assert
their soft-bucket counts; the old `test_stats` `pe_z` pre-regen path is now
recorded as `NO-Z` and pinned empty.
Scope: observed-inference no-oracle fixtures are currently pinned at zero under
the regenerated lavaan-backed corpus; future structural no-oracle cases need an
explicit count and rationale.

**Robust UΓ projector per-group weight.**
Regression: the complete-data `build_u_factor` ProjectionExpected path built the
kernel basis from `A_b = L_Γ,b⁻¹·Δ_b` without the per-group weight `w_b = n_b/N`,
so the reduced UΓ spectrum (SB scaling, FMG p-values, robust difference test) was
off by up to ~1% for models with both unequal group sizes and a cross-group
equality constraint. Masked for single-group, equal-group, and configural cases.
Guard: `cpp/tests/unit/fiml_test.cpp` (metric-invariance Unstructured degeneracy
~1e-6); the consolidated invariance runner with `--lane oracle --lavaan-parity`
(`experiments/research/active/06-fiml-invariance-tests/run_experiment.R`).
Scope: Expected bread only (the FMG path); the Observed-bread spectrum tail is
left as-is. `robust_se` uses its own w_b-weighted bread and was unaffected.

**ML2S weighted nested-test dispatch.**
Regression: R `robust_nested_lrt()` recognized only the exact estimator label
`"ML2S"` as two-stage. Weighted two-stage fits such as `"ML2S_DWLS"` still carry
`magmaan_fiml_data`, so the wrapper misrouted them into the FIML nested-test
binding and paid the saturated FIML difference-spectrum cost while using the
wrong metric. The ML2S branch now matches the `ML2S` prefix, runs before the
FIML raw-data check, and passes the fit's Stage-2 weight (`nt`, `uls`, `dwls`,
`adf`, or `dls`) into the ML2S Satorra-2000/2001 cores.
Guard: `r-package/tests/testthat/test-nested-test.R` stubs the low-level ML2S
and FIML bindings and asserts that `ML2S_DWLS`/`ML2S_ULS` go through the ML2S
branch with the selected weight; `cpp/tests/unit/fiml_test.cpp` pins ULS as the
identity two-stage moment weight.
Scope: scalar SB2001/SB2010 compatibility approximations remain baseline
comparison paths; the FMG-able restriction-map and U0-U1 difference spectra are
the weighted ML2S routes.

**Ordinal/mixed standardized delta-unit.**
Regression: the generic `λ·√Var(η)/√σ_rr` formula divided by `σ_rr ≈ λ²ψ+1`, but
a delta-ordinal `y*` is unit-variance, so a true .6 loading came back ~.52; the
path had been guarded to refuse ordinal fits. `standardize_all` now takes
`ordinal_delta_unit` and standardizes ordinal-indicator loadings (the `Lambda`
and all-y `Beta` slots) by the latent SD only (σ_rr = 1).
Guard: `r-package/examples/ordinal_dwls_wls.R`; `experiments/replications/03-li-2021-mixed`
and `04-li-2016-ordinal` `--lavaan-parity` (≤~1e-6 vs `standardizedSolution`).
Scope: checked standardized/defined-parameter goldens and a diagonal-Theta
ordinal/mixed factor-score scorer have since landed; multi-factor categorical
EAP and correlated residual-Theta scoring remain backlog/speculative work.

**Mixed-ordinal NACOV eager inversion for DWLS.**
Regression: `mixed_ordinal_stats_from_data` eagerly inverted the full NACOV to
build the WLS weight and errored if it was not PD — wasting an O(m³)
factorization for DWLS (which needs only diag Γ) and failing every mixed DWLS fit
at small N with many indicators (Li-2021 20-var SEM, N=200). `full_wls_weight`
now skips the inverse for DWLS-only callers, and even when requested the inverse
is non-fatal (a singular NACOV leaves `W_wls` empty; DWLS/robust proceed). Ported
to the all-ordinal path too.
Guard: `cpp/tests/golden/ordinal_golden_test.cpp` keeps the eager W_wls-vs-lavaan
contract; `experiments/replications/04-li-2016-ordinal`.
Scope: an explicit full-WLS fit on an empty weight reports it via
`validate_stats` / `weight_factors`.

**Continuous ULS standard-vs-robust base — not a bug (do not re-chase).**
Recorded so it is not re-investigated: ULS uses different base statistics for the
standard (`continuous_ls_chisq` Browne residual NT) vs robust
(`robust_continuous_ls`, scaling off `2N·fmin`) paths. This looks inconsistent
but faithfully mirrors lavaan's default-ULS Browne test vs its `se="robust.sem"`
unscaled base.
Guard: `cpp/tests/golden/lavaan_parity_golden_test.cpp` ULS
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
Scope: documented in `project/design/numerical-conventions.md`; memory
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
`project/design/numerical-conventions.md`.

**Kline/Guo duplicated lavaanify term.**
Regression: duplicated formula terms (`NA*LM1 + c(a1,a2)*LM1`) produced two
partable rows for one matrix cell, leaving a phantom free parameter that moved
the analytic gradient but not the model moments. `lavaanify` now merges repeated
`lhs op rhs` within a block (`build_group_template`, `cpp/src/spec/build.cpp`).
Guard: root-cause cases in `cpp/tests/unit/lavaanify_test.cpp`; end-to-end Guo
invariance rungs in `cpp/tests/golden/textbook_corpus_golden_test.cpp`
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
Guard: `cpp/tests/unit/detail_linalg_test.cpp` (RNG-free muthen-spectrum pin);
rank-deficient rejection in `dls_weight_test.cpp`.
Scope: an optional `spectral_truncate` parity-restore policy is not built
(backlog); `experiments/showcases/01-lavaan-parity` inverts the raw NACOV in R and is not
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
`reduced_gamma_nt_sample` in `cpp/tests/unit/robust_test.cpp`.
Scope: the row-space unbiased optimization and rank-one secular update that
relied on the same wrong identity are deferred (see `project/backlog/speculative.md`).

**Satorra-2000 nested-test parity oracle (not a bug).**
Regression: an apparent ~13% scale gap vs lavaan's
`lavTestLRT(method = "satorra.2000")` came from lavaan's `A.method` *default*
(`"delta"`, a moment-Jacobian column-space construction for covariance-nested
models), not from the Satorra-2000 scaling formula. magmaan's
`robust_nested_lrt(method = "restriction_map")` uses the exact
parameter-restriction matrix and reports the mean-scaled `T/c`.
Guard / oracle: regenerate fixtures and diff against
`lavTestLRT(fit_h1, fit_h0, method = "satorra.2000", A.method = "exact",
scaled.shifted = FALSE)` — NOT the bare default; `cpp/tests/unit/satorra2000_test.cpp`,
`r-package/examples/nested_test_satorra2000.R`.
Scope: lavaan's bare default (`A.method = "delta"`, `scaled.shifted = TRUE`) is a
documented compatibility alternative for covariance-nested checks, not the
magmaan oracle. Full investigation: `project/validation/satorra2000_parity.md`.

**Satorra-2000 degenerate-pencil fail-closed guards.**
Regression: near-Heywood missing-data invariance fits could feed the FIML
Satorra-2000 lavaan/exact path singular or non-finite `WLS.V`/model-Gamma/
restriction pencils. The low-level path then reached unguarded solves and
eigensolvers; on AVX-512 optimized builds this surfaced as a flaky segfault or
hang rather than a value-level `PostError`. Relatedly, the saturated FIML H1 EM
returned a last iterate on an iteration-cap hit by default, letting bad Stage-1
objects propagate into nested tests.
Guard: `cpp/tests/unit/satorra2000_test.cpp` pins sandwich degenerate-pencil
rejection, indefinite-but-nonsingular observed bread acceptance, singular pooled
info, and rank-deficient restrictions; `cpp/tests/unit/saturated_em_moments_test.cpp`
pins the default H1 cap as `FitError::OptimizerNonConvergence` plus the explicit
diagnostic opt-out. Existing FIML nested tests cover finite nonsingular observed
bread that is not SPD.
Scope: no exact Saga C++ file:line was recovered from the flaky AVX-512 crash;
the shipped fix is input validation and fail-closed numerical preconditions, not
an ISA-specific workaround.

**Ordinal WLSMV nested Satorra-2000 row convention.**
Finding: lavaan's ordinal WLSMV `lavTestLRT(..., method = "satorra.2000",
A.method = "exact", scaled.shifted = FALSE)` table reports the
mean/variance-adjusted Satorra-2000 difference statistic for `m > 1`: the
displayed `Df diff` is the spectrum-derived `d0 = (sum lambda)^2 / sum
lambda^2`, and the table scale is `sum(lambda) / d0`, not the integer-rank
mean scale `sum(lambda) / m`. magmaan exposes both rows; the ordinal nested
oracle compares lavaan's row to `T_adjusted`, `adjust_d0`, and the
`trace/d0` scale while keeping the integer restriction rank in `df_diff`.
Guard: `r-package/examples/nested_test_ordinal.R` covers a single-group ordinal
loading equality and a two-group configural-vs-metric WLSMV comparison; the
C++ helper `compute_satorra2000_from_sandwich` is pinned against the
moment-space wrapper in `cpp/tests/unit/satorra2000_test.cpp`.

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
`cpp/src/api/sem.cpp`, replaying `prepare_ordinal_partable`).
Guard: `cpp/tests/unit/api_sem_test.cpp` ordinal case (standardize_lv/all +
compute_defined succeed, categorical factor scores reject continuous methods
and expose EBM/EAP); live lavaan value-parity in
`r-package/examples/ordinal_dwls_wls.R` (ordinal `:=` to 5e-3, mixed std.all to
1e-3, categorical EBM factor scores to 5e-4); the C++ golden
`cpp/tests/golden/ordinal_golden_test.cpp`
"ordinal/mixed standardized + := rows match lavaan" now gates the `=~` loading
std.lv/std.all rows and the `lprod := L2*L3` value+SE against the stored lavaan
oracle (fixtures `ordinal/0015_defined_param_3cat_cfa` plus the per-fit
`fits.DWLS.standardized` block).

**lavaan 0.7 `lavTest()` returns a nested per-test list.**
Regression: under the oracle pin `0.7-1.2691`, `lavTest(fit, "satorra.bentler")`
(and the mean.var/scaled.shifted/browne variants) returns
`list(standard = ..., satorra.bentler = ...)` rather than the flat single-test
list 0.6-22 returned, so `$stat`/`$scaling.factor`/`$shift.parameter`/
`$scaled.test.stat`/`$df` read `NULL`. The fixture-gen helpers serialized those
as empty arrays and the consuming goldens crashed (`nlohmann::json get<double>`
abort on `null`) — robust ULS, inference, multi-group, and ordinal/mixed.
Fix: `lav_test1()` shim in `benchmarks/r/fixture_json.R` extracts `res[[what]]`
(falling back to the flat shape), routed through all six call sites
(`fixture_json.R` ls/ordinal robust helpers + `regen_oracle.R` fit layer).
Guard: regenerating under `0.7-1.2691` and the robust goldens (`ls`,
`inference`, multi-group, `ordinal`/`mixed` robust blocks); the empty-array
shape reappears if the shim is dropped.
Scope: only the lavaan-version-dependent `regen_oracle.R` streams were
realigned; the real-data `parity` stream and the little/newsom/mplus/paper
corpora stay at `0.6-22.2560` (their regen needs data packages, e.g.
`psychTools`, absent in this environment) and pass within tolerance. The
little/newsom/mplus_sem/geiser/textbook_corpus fixtures were regenerated
under 0.7-2 on 2026-09-25 with the source-verified corpus (see
[textbook-translation-audit.md](textbook-translation-audit.md)).

**Textbook categorical (WLSMV) lane.** Added 2026-09-25.
`textbook_ordinal_golden_test.cpp` fits 21 all-ordinal, covariate-free corpus
cases (Mplus User's Guide and Newsom 2015 and 2024) from derived ordinal
moments with magmaan's DWLS estimator. The cases were verified in the corpus
against the book's own output. For each case it compares the objective, df,
every estimate and the implied latent-response correlations with lavaan.
- **Passing (19):** delta and theta CFAs, longitudinal invariance models,
  threshold-constrained models, two-group twin models, freed or zero-fixed
  theta residual variances, delta scale factors and latent means in the
  implied thresholds, all to about 1e-6. ex5.19 is a reflected-loading
  equivalent solution. The six partable-semantics gaps closed with the
  ordinal partable fix of 2026-09-29.
- **Known gaps (2), reported without failing.**
  - Newsom 2015 ex9.2: lavaan's free-delta optimum implies negative
    residual variances in magmaan's free-theta translation, which the default
    variance bounds exclude (latent change model; free-delta bound relaxation
    is consumer-gated).
  - Newsom 2024 ex1.3c: from lavaan's starts L-BFGS stops on a flat ridge
    of this saturated theta model (fmin 5.8e-9), within the Newton accuracy
    budget but with different raw parameters.
- **Diagnosis.** Failing cases are refit from lavaan's θ, which separates
  optimizer issues from model semantics.

**Multi-group ordinal robust score tests used to be blocked by a stale guard.**
Regression: the ordinal/mixed-ordinal robust MI and score-test implementations
already assembled the per-block `n_b/N` sandwich over categorical moments, but
`estimate::frontier` rejected `stats.R.size() != 1` before that code could run.
This left the continuous ML/LS robust tiers multi-group while the ordinal tier
was artificially single-group. Fix: remove `require_single_group_ordinal` in
`cpp/src/estimate/ordinal.cpp`. Guard: `cpp/tests/golden/score_robust_golden_test.cpp`
fixture `0012_robust_release_mg_ordinal` checks a two-group WLSMV
cross-loading release against lavaan-internals `delta`/`wls.v`/`gamma` assembly;
`cpp/tests/unit/score_robust_test.cpp` adds exact two-group WLS reductions for
all-ordinal MI/score and mixed-ordinal MI/score, plus a non-trivial DWLS
finite-scaling case.

**FIML reporting test crashed on a failed precondition; opt L-BFGS stall — two
findings.**
Regression (real bug): tests build with
`DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS`, under which a failed
`REQUIRE` cannot throw to abort the case (`throwException()` is a no-op), so
`REQUIRE_OK(fit)` logged the failure and execution continued into
`fit->fiml_pack()` — `operator->` on an error-state `std::expected`, i.e. UB /
segfault. It only fired when the FIML fit actually failed (opt preset), masking
the real failure behind a crash; `dev`, where the fit succeeds, never tripped it.
Convention (not a bug): that fit failed only under `-O3 -march=native` (AVX2/FMA
on this CPU), and only because the test's near-perfect 1-factor data drove
residual variances to a near-Heywood wall (θ≈0.003) where L-BFGS finds no Wolfe
step (‖∇f‖≈59), while the SSE / `-O0` paths thread to the interior optimum
(f=−4.2424). The gradient is correct there (finite-difference-verified); magmaan
reports a clean `LineSearchFailed` error. No solver "rescue" was added — an
SLSQP fallback would prejudge the optimizer-comparison tracks, and the
sensitivity is `-march=native`/ISA-specific, not a magmaan correctness bug.
Guard: `REQUIRE_OK` / `REQUIRE_OK_OR` (`cpp/tests/unit/api_sem_test.cpp`) now
`return` on failure, so a failed precondition reports cleanly instead of
dereferencing a valueless expected. The FIML reporting test uses
`fiml_interior_raw()` (substantial idiosyncratic variance → interior optimum),
so it converges with pure L-BFGS on every ISA and exercises the reporting
surface rather than a degenerate optimization geometry.
Scope: the near-Heywood + `-march=native` L-BFGS sensitivity is an accepted
optimizer-path limitation, documented here, not guarded by a test.

**Geiser GLS/ULS implied-moment parity: fixed.x resolution, multi-start basin,
and the manifest cross-lagged observed-ordering fix — three findings.**
Regression (real bug): the Geiser goldens rebuilt the implied-moment evaluator
from the caller's `pt`, but `fit_*` takes `pt` by value and resolves fixed.x on
its own copy, so the manifest fixed.x path models' exogenous observed-moment
block stayed NaN-filled and was excluded from the parity gate. Fix: call
`estimate::resolve_fixed_x_from_sample` on the test's `pt` before the implied
check (mirrors `fit_implied_golden_test`), so `manifest_regression`,
`manifest_path`, and `manifest_path_non_saturated` now gate Σ/μ against lavaan.
Recipe (not a bug): the *latent* AR cross-lagged family has a shallow
moment-objective surface where `simple_start_values` settles in a higher,
non-lavaan basin (e.g. GLS 0.326 vs 0.272). A plain ML fit lands in lavaan's
basin; running each GLS/ULS arm from both the simple start and the ML warm start
and keeping the lower-objective optimum (`best_start`) reaches lavaan's Σ/μ to
~1e-7 for all three latent cases without disturbing the well-behaved models
(where a warm-started full fit can otherwise stop short of the GLS optimum). The
objective cross-check was relaxed from an absolute `2e-4` to
`max(2e-4, 2.5e-3·|fx|)` to absorb the known ~1/N GLS mean-structure scale
convention, which only bites the high-|fx| latent AR case (Σ/μ already match to
1e-7); the looser ULS callers `max(5e-3, 5e-3·|fx|)` reproduce the prior
`5e-3·max(1,|fx|)` gate exactly.
Regression (harness bug, since fixed): the two *manifest* fixed.x cross-lagged
path models (`manifest_ar_cross_lagged`, `…_extended`) appeared to converge to a
worse-than-lavaan optimum (GLS 0.131 vs 0.126; 0.051 vs 0.002, implied covariance
off ~0.03-0.06) and were gated for self-consistency only. The earlier diagnosis —
"magmaan-side fixed.x propagation difference" — was wrong. Root cause: an
observed-order mismatch in the golden harness. magmaan orders observed variables
`[ov.y, ov.x]` (classify), and `data::SampleStats` is name-free / positionally
aligned to that `ov_order`; but the fixtures carry `sample_cov`/`sample_mean` and
lavaan's Σ/μ in their own data column order. These two are the only Geiser cases
where the exogenous variables (`d11`, `c11`) are not already last in the data, so
`resolve_fixed_x_from_sample` read exogenous moments from the wrong sample
positions (e.g. `d11~1` resolved to 2.088, the c12 mean, instead of 0.307) and
the objective compared mis-ordered Σ vs S. Fix: the harness builds the matrix rep
with `LatentNames` (so `rep.ov_names` carries the real order) and reconciles by
variable name (`perm_to_magmaan`), permuting the sample and the lavaan moments
into magmaan's `ov_order` before fit/compare (identity for the other Geiser
cases). Both ids now gate Σ/μ against lavaan to ~1e-8 and `fmin` to within the
existing `max(2e-4, 2.5e-3·|fx|)` GLS gate. Note (not fixed here, out of scope):
the C++ `api::data_from_sample_stats` likewise assumes `ov_order`-aligned summary
stats; a named-summary-stat reconciliation in the api/R layer would let callers
pass `sample.cov` in any column order.
Guard: `cpp/tests/golden/geiser_golden_test.cpp` (the two `Geiser {GLS,ULS} goldens`
cases) — all Geiser ids now gate Σ/μ against lavaan; the self-consistency
exception (`has_known_nonparity_fit`) is removed.
Scope: the latent family's residual objective-value gap is the documented GLS/ULS
scale convention, with the implied moments as the authoritative parity.

**Latent-scale convention parity.**
Regression: multi-group std.lv metric invariance fixed every latent variance,
adding restrictions beyond its marker equivalent. Group-2+ latent variances
are now released when loadings, but not latent variances, are equated.
Guard: `constraints_test.cpp` checks the coordinate-change invariant;
`fit_stdlv/0002_three_factor_hs_2group_loadings` directly gates estimates, SE,
chi-square and df against lavaan. Scale-free discrepancy bounds are 3e-5 for
estimates, 1e-4 for SE and 1e-8 for chi-square; n_free/df are exact.
`lavaanify_test.cpp` separately pins per-level std.lv, single-indicator fixing,
user-fixed growth loadings and latent-variance modifiers: `start(2)*` becomes
a fixed value of 2 under std.lv, while `NA*` keeps the row free. Marker
loadings stay fixed across groups where the group-1 loading is fixed.

**Whitened expected-information arithmetic.**
Regression: the original expected-information assembly stored mostly-zero
parameter-by-group matrices and contracted them pairwise. The factored
Jacobian assembly removes that cost while preserving the trace identity.
Guard: `expected_info_whitened_test.cpp` compares independent explicit traces
across 1–4 groups, with and without means. `score_robust_test.cpp` checks
identity-versus-flip covariance agreement at 1e-12; exact-zero relative shift
was an arithmetic-path accident (the former implementation also gave
1.8e-16–2.7e-16 shifts on other seeds), not an invariant.

**Reduced-bias and estimated-weight inference assembly.**
Regression protection: `rbm_fd_test.cpp` independently reconstructs the
`-0.5 trace(j^-1 e)` penalty derivative and reduced-space correction over ML,
FIML, continuous/ordinal/mixed LS and ML2S. Every family gets the explicit
one-step check; cheap implicit families also get explicit/implicit order and
adjusted-objective stationarity checks. Weighted ordinal/mixed/ML2S families
are not claimed to have those more expensive implicit checks.
`weighted_inference_test.cpp`, `ordinal_test.cpp` and `fiml_test.cpp` retain
fixed-weight reductions, case-weight FD gates for Gamma influence, analytic
versus FD bread, complete-versus-observed-data reductions and live
estimated-weight channels. The default NT ML2S robust-score path remains
separate from the moment-quadratic GLS IJ adapter. Advisory calibration
harnesses under `cpp/tests/checks/ordinal_{dwls_profile,rmsea_inference,
crmr_inference,cfi_inference}/` assess prototype finite-sample behavior;
passing deterministic reductions alone does not establish calibration.

### Mplus continuous MODEL lowering (TASK-51.2)

Guard: `cpp/tests/unit/mplus_parser_test.cpp` independently checks marker and
last-mention rules, physical-line labels, label lists/equality partitions,
explicit starts, ordered ranges, paired/crossed relations and DF01–DF13 roles.
`cpp/tests/golden/mplus_model_golden_test.cpp` consumes all 13 frozen
`mplus/golden.json` cases: exact parameter-key sets, fixed/free status,
fixed values, explicit start hints, equality partitions, ML estimates,
implied means/covariances, expected-information SEs, npar, df and chi-square.
Numeric differences must be below 1e-5. Eligible Demo `probes.json` variants
also gate TECH1 distinct free counts, cell status and equality partitions.
The reserved `.eq1.` equality-label spelling is accepted by pinned lavaan
0.7.2; named labels cannot collide with it.

Local corpus sweep on 2026-10-03: `extract_mplus_corpus.sh` read case
`original.inp` files, raw inputs and `.inp` zip entries from the ignored
textbook corpus. The manifest and files live under
`~/.cache/magmaan-mplus-corpus/`; each extraction replaces `files/` while
preserving the previous cache directory. A non-zip mirror,
`raw/mplususerguid/ex11.8imp.zip`, was reported and skipped. With
`MAGMAAN_MPLUS_CORPUS` set to the extracted `files/`, the opt spec tests
parsed 2,438 inputs: **189 accepted, 2,249 rejected; no crash or hang**.
Each rejected file counts once per rule, so the following counts overlap:

| Rule | Files | Rule | Files | Rule | Files |
| --- | ---: | --- | ---: | --- | ---: |
| CL03 | 170 | CL04 | 88 | CL06 | 66 |
| CL10 | 452 | CL11 | 146 | CL13 | 139 |
| CL14 | 168 | CL15 | 531 | CL16 | 436 |
| CL17 | 879 | CL18 | 253 | CL20 | 3 |
| CL21 | 2 | CL22 | 86 | CL23 | 55 |
| CL24 | 44 | CL26 | 135 | CL27 | 599 |
| CL29 | 628 | CL31 | 108 | CL32 | 20 |
| GR01 | 47 | LX01 | 647 | LX02 | 1 |
| LX03 | 28 | MS01 | 2 | MS08 | 4 |
| MS09 | 11 | MS10 | 57 | MS11 | 16 |
| NM03 | 345 | | | | |

Scope: this corpus gate checks parser stability and classified boundaries,
not estimation or parity of every accepted corpus model. Frozen golden
models provide the numeric gate; later language families and API/R exposure
remain separate increments. Focused opt validation: 84 test cases and
7,994 assertions passed (Mplus, EQS, parser, lavaanify and spec goldens).

### Mplus API/lab end-to-end gate (TASK-51.3)

`api::model_from_mplus()` retains source and classified notes. Lab tests compare
four independently written lavaan models on rows, fixed/free status, fixed
values, starts, ML estimates, expected-information SEs, chi-square and df
(1e-5), and exercise fresh/prepared fits, source rebuilds and specs/fits reloaded
in a fresh R process. Generated equality plabels depend on row order; tests
compare constraint counts and the resulting constrained estimates. Fixed-x
sample moments have no fitted estimate in the lab partable.

Local `cpp/tests/tools/check_mplus_corpus.R` on 2026-10-03 fitted original
inputs against `book.json` and available original output: **68 eligible,
15 accepted and matched, 53 rejected**, with first-rule counts CL10 14,
CL11 13, CL13 3, CL15 1, CL16 1, CL27 7, GR01 7 and NM03 7.
The report is `~/.cache/magmaan-logs/mplus-corpus-gate.csv`; it includes
per-comparison deviations, recorded decimal counts and tolerances. All 15
cases match N, df and chi-square; available npar (11), H0 log-likelihood (10)
and printed parameter estimates (220 comparisons) also match. MLR chi-square
is excluded because its printed statistic is scaled.

Of the stated 79 cases, 11 lack `data/raw.csv` (none lack original input or
book expectations): Muthen `ch2_ex2_17`; Brown `tab4_1_neuroticism_extraversion`,
`tab6_3_mtmm_correlated_uniqueness`, `tab7_17_mimic_phobia`,
`tab7_2_tau_equivalent`, `tab7_5_parallel`, `tab7_8_longitudinal_invariance`,
`tab7_9_effects_coding`, `tab8_12_sem_formative_stress`,
`tab8_2_higher_order_coping` and `tab8_8_reliability_ptsd`.

The planner authorized printed-precision allowances in board comment #8:
for chi-square, log-likelihood and estimates use the maximum of the original
quantity-specific tolerance and `0.5*10^(-d) + 1e-5*abs(expected)`, with `d`
counted from the recorded JSON/output token. Brown tab8_6 prints 167.63;
167.634078594 matches within its resulting 0.0066763 tolerance. Exact count
checks retain zero tolerance. No frontend rule or numeric gate was relaxed.
Focused opt gates: 24 Mplus cases/4,636 assertions and one API case/nine
assertions pass. Full lab testthat passes (5,018 assertions) with two existing two-level
admissibility warnings and two documented skips.

### Mplus multiple groups (TASK-52)

The continuous GROUPING frontend uses ascending integer code order and source
label metadata. P-MG8–P-MG12 settled asymmetric rows, repeated sections,
last-mention overrides, multi-label rejection and integer coding (including
negative/integral-decimal codes). P-MG13 adds group-specific role probes:
Mplus rejects an overall-absent group regression as ignored and accepts an
extra indicator. Both are rejected with a rewrite instruction where the common
variable-role contract cannot preserve the group-specific roles.

`regen_oracle_mplus.R` now extends `mplus/golden.json` with 14 independently
specified grouped cases (the separate helper is `regen_oracle_mplus_groups.R`).
Default/three-group CFA, named/numeric ties, first-group mean release, asymmetric
residual covariance, reversed declaration/raw-row order, structural regressions,
second-order CFA and all six shortcut/identification combinations have proper
lavaan fits (nonnegative variances, chi-square p >= .001). The second-order
fixture explicitly pins first-order latent intercepts, avoiding redundant
higher-/first-order latent mean identification. Demo TECH1 checks every
fixed/free cell and equality partition; counts/df are exact. Every printed
Demo estimate and chi-square matches the existing printed-precision allowance.
C++ gates compare independently written rows, ML estimates, expected SEs,
implied means/covariances and chi-square against pinned lavaan at 1e-5.
Both shortcut identifications give 38/34/30 parameters and 16/20/24 df.

The end-to-end corpus gate increases from 15 to **22 accepted/matched among
68 eligible**, with zero failures; 46 remain rejected (CL10 15, CL27 10, GR01
7, NM03 7, CL13 3, CL15/CL16/IV04/MG03 one each). Four raw CSVs store source
labels rather than integer codes; the harness restores their declared coding
without dropping rows. The adapter still rejects undeclared codes with row
counts and an explicit filtering instruction.

`check_mplus_input_corpus.R` sweeps 2,440 original files including archive
members; same-manifest pre-increment/current acceptance is **253/291 reader**
and **189/217 MODEL**. All rejections are classified, with no crashes or hangs.
One existing archive (`ex11.8imp.zip`) is unreadable and reported separately.
The inventory records the first-rule tallies; raw results stay in the log cache.
The older reader-only tally used a different extraction/uniqueness convention.

The lab gates compare live independent lavaan rows, fits and expected SEs,
check group order regardless of row appearance, source rebuilds and serialized
specs, and reject conflicting grouping and undeclared data codes. Projection
checks cover per-group starts and labelled intercepts. Generated default/zero
row provenance is applied in the C++ compatibility layer before projection.
Focused opt gates pass 30 Mplus cases / 11,976 assertions and three API
cases / 20 assertions. The full magmaanlab suite passes 166 tests / 5,083
expectations (5,079 passes, two skips and two warnings), with no failures or
errors; the
full suite retains its optional-test skips and existing two-level admissibility
warnings. No tolerance was widened and no oracle exemption was added.

### Mplus data files (TASK-55)

`MplusDataPlan` carries ordered FILE references, expanded FORMAT operations,
TYPE, NOBSERVATIONS/NGROUPS/LISTWISE and MISSING flags through the parser and
API. Separate-file and summary groups use source labels and `.mplus_group`;
GROUPING continues to use integer codes. The reader drops unlisted codes with
counts but leaves LISTWISE and estimation-sample rules unapplied.

`regen_mplus_probes.R` adds P-DA3/P-DA4/P-DA5 (12 variants; 50 probes/122 variants
in total), preserving every previous fixture entry. Mplus 9.1 omits NU/ALPHA
for summary inputs without MEANS, rejects explicit intercept/factor mean
mentions without MEANS, and accepts correlation without SD as unit covariance.

`check_mplus_data.R` generates synthetic data locally and gates 25 Demo data
cases / 28 group comparisons: wrapping/extra fields/NOBSERVATIONS; fixed repeats,
skips/tabs/record breaks/explicit and implied decimals; global symbols, numeric
and range missing flags; triangular/full covariance/correlation, means/SD;
NGROUPS, FILE groups and dropped GROUPING codes. Derived evidence is frozen in
`mplus/data_summary.json`, consumed by C++ tests. N is exact; means/covariances
agree within half the Demo's three-decimal printed unit (0.00050001). Synthetic
all-missing rows are explicitly excluded by the comparison harness to reproduce
DA05; the reader retains them. Raw covariances use divisor N. Summary input matrices use divisor N-1;
the reader converts each group by (N-1)/N to the lab divisor-N convention.
Ten saturated summary group comparisons gate Demo ML estimates and live lavaan using its
default sample.cov.rescale = TRUE, including means and NGROUPS.

Independent R frames/matrices gate reader behavior and shape/file errors.
The free-format raw.csv round trips for User's Guide ex5.1 (single group) and
ex5.14 (multiple groups) reproduce direct-data fit estimates at 1e-8. The existing
end-to-end corpus gate remains 22 accepted/matched of 68 cases, zero failures.
The original-input sweep classifies all 2,440 files: 389 reader and 310 MODEL
acceptances; the previously unreadable ex11.8imp.zip remains reported.

Focused opt C++ checks pass 33 Mplus cases / 12,297 assertions and four API
cases / 27 assertions. Full magmaanlab testthat passes 5,499 assertions with
two existing admissibility warnings and two documented skips. Structural
tracked-file and dependency-layering checks pass. No oracle exemptions or
numeric-tolerance changes were introduced.

### Explicit ordinal preparation prerequisite (TASK-53, partial)

The lab DWLS/ULS/WLS, PSD and stage-2 glue parsed `LatentNames::row_user`
but omitted it from preparation, silently fixing explicit residual/intercept
rows. Passing provenance preserves the explicit group-2 THETA residuals.
With `auto_var = FALSE`, a response-scale row previously had no residual sibling
for the preparation's scale-to-residual translation. Construction now emits a
generated fixed unit sibling, which preparation releases when requested.

Guards: the ordinal C++ test with explicit response scales and automatic
variances disabled checks six generated residuals and the requested release;
the live-lavaan grouped THETA R provenance test uses an independent explicit
model and the versioned fitting preset. Existing expectations are unchanged.
The full ordinal C++ suite passes 145 cases / 7,058 assertions.

The independent P-IV2 diagnostic now uses `convention_inference(fit, "WLSMV")`
for default-lavaan robust SEs and scaled/shifted tests. Both native and versioned
fitting routes pass (maximum common SE difference 1.04e-6, test difference
2.07e-6, df 24). Released DELTA scales project `1/sqrt(Sigma*_ii)` rather
than unit values; they remain fixed projection rows (`free = 0`), without a
delta-method SE. Common estimates including scales pass existing tolerances.
Native expected-information reporting remains an observation, not this gate:
its test is about 27.535189, with common SE ratios about 0.999 versus default
lavaan. No inference default or tolerance changed.

`check_mplus_categorical_releases.R` adds an independent single-group reference
with two fixed thresholds identifying an explicitly freed first response scale
(DELTA) or residual (THETA). Both converge with df 9; maximum estimate difference
4.44e-7, SE difference 9.25e-8, test difference 3.43e-7. Full ordinal C++ tests
pass unchanged (145 cases / 7,058 assertions); full magmaanlab passes 5,503
assertions, two documented skips and two existing two-level warnings.

The same diagnostic with `--scale-equality` exposes a new prerequisite defect.
Two response scales labelled `shared` are equal in the independent lavaan DELTA
model (about 0.825), with df 11. Preparation translates the two free scale rows
into independent residual dimensions, losing their original scale equality;
magmaan reports about 0.771 and 0.880, df 10. Its WLSMV bundle nevertheless
claims covariance/test availability: maximum common estimate error 0.154098,
SE error 0.009725 and scaled-test difference -1.235463. The corresponding THETA
residual equality passes. This is a lost model restriction, not a reporting
convention discrepancy. The user superseded the temporary rejection from
`ae94fd9b` with TASK-53.1: DELTA now retains live response-scale coordinates.
The same equality diagnostic reports both scales 0.8246215, df 11, maximum
estimate error 1.58e-7, scale/common SE error 4.38e-8 and scaled-test error
1.65e-7. THETA's residual-equality counterpart remains unchanged and passes.

`regen_ordinal_delta_scales.R` pins lavaan 0.7-2 and seed 531072, writing only
synthetic derived summaries to `ordinal/delta_scales.json`. Seven models cover
fixed 0.8, two shared scales, a single release, DELTA threshold/loading
invariance, the P-IV2 label-equality SCALAR model, equality over time and a fixed
non-unit scale plus a linear scale constraint. The C++ golden gates free flags,
labels, all estimates (including derived residuals), DWLS and WLSMV SEs
(including live scales), unscaled/scaled statistics, scaling, shift and df at
existing 1e-5 magnitude-scaled tolerances. Compact bounded and generic SNLLS
routes additionally match frozen ULS estimates, and the observed Hessian is
checked against a finite-difference gradient away from the optimum. The live R
ordinal provenance file independently regenerates the synthetic data and fits
all seven models against installed lavaan. It also gates standardized loading
values/SEs at 1e-5 and fixed/equal/released-scale EBM predictions at the existing
5e-4 factor-score tolerance.

Changed expectations are coordinate corrections: the explicit-release unit
guard now checks a free `~*~` rather than a free `~~`; the temporary rejection
guard now checks retained restrictions; the provenance regression restricts a
response scale rather than its derived residual; the old SNLLS rejection test
now checks full-moment fitting, with frozen ULS estimates providing independent
numerical evidence. The THETA API regression explicitly selects THETA during
preparation, retaining all its existing numerical assertions. Lavaan confirms the explicit
group-2 fixed scale remains fixed under threshold invariance. Augmentation no
longer duplicates a source scale row; prepared handles use the selected
parameterization and row provenance. Existing THETA numeric expectations and
native inference defaults are unchanged. The grouped diagnostic counts DELTA
scale rows and includes their SEs, rather than counting translated residuals.

Final opt C++ verification: ordinal 147 cases / 9,474 assertions; spec 211 /
14,676; API 28 / 788; inference 376 / 482,841. Estimate passes 564 / 16,320 and robcat 1 / 1. All previously fixed-unit DELTA and THETA numerical gates
retain their existing tolerances. The full installed magmaanlab suite passes
5,643 expectations (two existing skips and two existing two-level warnings);
the ordinary R suite passes 1,736 expectations without failures or skips.

### TASK-85 output meaning gate

The local `cpp/tests/tools/check_mplus_outputs.R` reads disk outputs and ZIP
members in place, retains relative corpus paths and Mplus versions, and stores
only hashes and derived counts in `mplus/out_meaning_summary.json`. It scans
1,933 outputs, 1,337 Mplus outputs, seven Mplus-error outputs skipped, and
683 distinct echoed inputs; 130 accepted / 553 rejected with rule IDs. Every
accepted input matches all three printed dimensions: groups 130, npar 130, df 130.
No mismatch or counting case remains unresolved, and no version exemption
was needed. Sixteen Mplus 5.1/5.2 inputs indent the printed free-parameter
heading; trimming headings fixes that gate artifact and restores their direct
free-count comparisons.

Equality restrictions reduce the count by the rank of an independently
constructed symbolic Jacobian, checked at three deterministic generic points.
Parameter labels and plabels map to the same independent coordinates;
redundant label equalities have zero residual derivatives. Free NEW coordinates
count as parameters; NEW declaration metadata and derived quantities do not.
Six independently specified counting examples gate redundancy, nonlinear
restrictions, derived quantities and auxiliary coordinates. Threshold counts
come from unstandardized MODEL RESULTS per variable/group, with a fallback to
positive-count categories in the printed categorical proportions section for
three inputs. Group matching uses Mplus group labels rather than projected
numeric codes. Growth thresholds share indices across each time-score statement
(GR05); means and conditioned x follow the lowered model. No data files are read.

Eight Little inputs exercise the fixed bracket-group defect: six previously
could not resolve later intercept labels in constraints, and two Mplus 7
strong-invariance outputs printed npar 18 / df 9 while the frontend gave 20 / 7.
The latter files are `raw/little/source/CH6_mplus/CH5/`
`ch5.fig.5.3.factor.strong.out` (hash `9bf71b28ab3693ad`) and
`ch5.fig.5.3.marker.strong.out` (`776ae6ac019c541b`). They now match 18 / 9.
The parser consumes all consecutive bracket groups and retains each segment's
labels and modifiers. The same defect affected threshold and DELTA scale groups;
multiline variance segments already retained their labels and modifiers.
P-LB7 provides nine independent Mplus 9.1 Demo variants, including shared labels,
equality numbers, fixed modifiers and same-line rejections. Independent unit rows
and the TECH1 partition gate check the canonical fix. The summary records these
eight resolved cases as `fixed_magmaan_bug` and the 16 legacy heading cases as
`gate_artifact`; counting corrections carry no frontend statistical-policy change.

Input normalization preserves physical line boundaries because LB02/LB03 make
lines semantically relevant. This yields 683 distinct inputs rather than the
draft's 682. Canonical parser and vendored copy are synchronized. Validation:
opt build, all 1,550 C++ tests, 18 focused Mplus MODEL cases / 12,163 assertions,
full magmaanlab testthat suite (7,327 passed, two two-level admissibility
warnings and two unavailable/opt-in oracle skips), six counting examples,
the output-only gate,
and the unchanged numerical corpus gate pass. The latter remains 50/68 accepted,
44 matched, six unsupported fitting routes and zero failures. Tracking, layering
and whitespace checks pass.

## Validation Areas

| Area | Oracle | Protection | Important files/tests | Known gaps |
|---|---|---|---|---|
| Parser and lexer | `project/grammar/grammar.ebnf`, checked parser fixtures | Unit plus golden tests under `spec` | `cpp/tests/unit/lexer_test.cpp`, `cpp/tests/unit/parser_test.cpp`, `cpp/tests/golden/lexer_golden_test.cpp`, `cpp/tests/golden/parser_golden_test.cpp` | Grammar-coverage walk remains manual; grammar changes must edit EBNF first. |
| Lavaanify, spec, and partable projection | lavaan `parTable()` fixtures and corpus exports | Unit, golden, and corpus parity under `spec` and `parity` | `cpp/tests/unit/lavaanify_test.cpp`, `cpp/tests/golden/lavaanify_golden_test.cpp`, `cpp/tests/golden/textbook_corpus_golden_test.cpp` | Little/Newsom and Mplus corpus promotion remains ongoing. |
| Matrix representation and model evaluation | lavaan implied moments plus algebraic invariants | Unit plus golden tests under `spec` | `cpp/tests/unit/matrix_rep_test.cpp`, `cpp/tests/unit/model_evaluator_test.cpp`, `cpp/tests/golden/matrix_rep_golden_test.cpp`, `cpp/tests/golden/fit_implied_golden_test.cpp` | fixed.x exogenous moments are sample-resolved before comparison; summary-stat goldens must reconcile their data column order with magmaan's `[ov.y, ov.x]` `ov_order` by name (see `geiser_golden_test.cpp`). All Geiser implied-moment cases now gate against lavaan. |
| Complete-data ML and LS estimation | lavaan JSON fixtures, real-data parity fixtures, and covariance-domain invariants | Unit, golden, and parity tests under `estimate` and `parity` | `cpp/tests/unit/ml_test.cpp`, `cpp/tests/unit/ls_path_test.cpp`, `cpp/tests/unit/psd_ml_test.cpp`, `cpp/tests/golden/ls_golden_test.cpp`, `cpp/tests/golden/lavaan_parity_golden_test.cpp`, `r-package/tests/testthat/test-admissibility.R` | Covariance-honest fixed-weight GMM has derivative, interior-equivalence, repair, and R-contract gates; broader estimator-specific boundary/corpus/timing validation remains deliberately open. |
| FIML and missing data | lavaan FIML fixtures, saturated EM invariants, FIML FMG diagnostics | Unit, golden, R examples, and advisory checks | `cpp/tests/unit/fiml_test.cpp`, `cpp/tests/golden/fiml_golden_test.cpp`, `r-package/examples/fiml.R`, `cpp/tests/checks/fiml_fmg_trace/` | High-level `fit_model(estimator = "FIML")` mean-structure defaults and multi-group starts need care. |
| Ordinal and mixed moments | lavaan ordinal fixtures, robcat fixtures, internal moment invariants | Unit, golden, robcat, R examples, and experiments | `cpp/tests/unit/ordinal_test.cpp`, `cpp/tests/golden/ordinal_golden_test.cpp`, `cpp/tests/golden/robcat_parity_golden_test.cpp`, `r-package/examples/ordinal_dwls_wls.R` | Lazy mixed WLS construction, mixed theta SNLLS, and research robust-association paths remain open. |
| Inference, standardization, and fit measures | lavaan SE, score, standardized, and fit-measure fixtures | Unit plus golden tests under `inference` | `cpp/tests/unit/inference_test.cpp`, `cpp/tests/unit/score_test.cpp`, `cpp/tests/unit/standardized_test.cpp`, `cpp/tests/golden/inference_golden_test.cpp`, `cpp/tests/golden/standardized_golden_test.cpp` | Ordinal defined-parameter validity and additional ordinal-SEM standardized goldens remain follow-ups. |
| Robust tests, FMG, and nested restrictions | lavaan/semTests parity, R-internals fixtures, weighted-chi-square oracles | Unit, golden, R examples, and advisory checks | `cpp/tests/unit/robust_test.cpp`, `cpp/tests/unit/fmg_test.cpp`, `cpp/tests/unit/weighted_chisq_test.cpp`, `cpp/tests/golden/fmg_pvalue_golden_test.cpp`, `cpp/tests/golden/score_robust_golden_test.cpp`, `r-package/examples/fmg.R`, `r-package/examples/nested_test_ordinal.R` | FIML robust score tests remain single-group v1; mixed-sign weighted-χ² tails remain outside magmaan's non-negative spectrum contract. |
| Optimizers and terminal audits | Recomputed objectives, projected gradients, cross-backend agreement | Unit tests and benchmark/report tracks under `estimate` | `cpp/tests/unit/terminal_audit_test.cpp`, `cpp/tests/unit/optimizer_crosscheck_test.cpp`, `cpp/tests/unit/fit_diagnostics_test.cpp`, `project/design/terminal-audit.md` | Ultimate verifier and stationarity tolerance calibration remain research work. |
| Simulation | Distribution goldens, deterministic calibration fixtures, stochastic smokes, covsim reference cross-check | Unit tests under `sim` plus advisory checks | `cpp/tests/unit/norta_test.cpp`, `cpp/tests/unit/plsim_test.cpp`, `cpp/tests/unit/vale_maurelli_test.cpp`, `cpp/tests/checks/plsim/` (incl. `plsim_vs_covsim.R`, which validates PLSIM's covariance integral + root-find against covsim's `get_cov` to ~1e-8 via the headline get_cov-on-magmaan-marginals cross-check) | Model-implied simulation, ordinal/mixed observed-correlation calibration, and persistent caches remain open; covsim cross-checks for NORTA/Vale-Maurelli not yet added. |
| R boundary and examples | lavaan parity through examples and R-shaped wrapper checks | `just r-check` examples plus C++ API tests | `cpp/tests/unit/api_sem_test.cpp`, `r-package/examples/*.R`, `r-package/examples/tutorial/run_all.R` | Examples are smoke tests, not exhaustive wrapper coverage; R reconstruction is sensitive around means and groups. |
| Composite frontier | lavaan native composite fixtures and FC-SEM evaluator invariants | Unit, golden, and R frontier example tests | `cpp/tests/unit/fcsem_evaluator_test.cpp`, `cpp/tests/unit/fcsem_ml_test.cpp`, `cpp/tests/golden/composite_golden_test.cpp`, `r-package/examples/fcsem_frontier.R` | Native W/T matrix internals are intentionally unit-test diagnostics rather than a public fixture contract; multi-group and non-ML composites remain deferred. |
| Corpus parity | lavaan-generated real-data and textbook fixtures | Heavy `parity` target and corpus-specific goldens | `cpp/tests/golden/geiser_golden_test.cpp`, `cpp/tests/golden/mplus_sem_golden_test.cpp`, `cpp/tests/golden/paper_corpus_golden_test.cpp`, `cpp/tests/golden/textbook_corpus_golden_test.cpp`, `cpp/tests/golden/textbook_ordinal_golden_test.cpp` | Corpus breadth is intentionally staged; some cases document alternate optima or unsupported syntax. The textbook ordinal lane runs 2 known-gap cases (a free-delta Heywood bound and a flat ridge where the optimizer stops early; see the all-ordinal sections of `project/backlog/todo.md`) as reported, non-failing checks. |

## High-Risk Map

### FIML and Missing Data

Protected by:

- `cpp/tests/unit/fiml_test.cpp`
- `cpp/tests/golden/fiml_golden_test.cpp`
- `r-package/examples/fiml.R`
- `cpp/tests/checks/fiml_fmg_trace/` and `cpp/tests/checks/fiml_fmg_nested/`

Known weak spots: high-level mean-structure defaults and multi-group starts need
care; pairwise-data FMG remains deferred.

### Ordinal and Mixed Moments

Protected by:

- `cpp/tests/unit/ordinal_test.cpp`
- `cpp/tests/golden/ordinal_golden_test.cpp`
- `cpp/tests/golden/pairwise_golden_test.cpp`
- `cpp/tests/golden/robcat_parity_golden_test.cpp`
- `r-package/examples/ordinal_dwls_wls.R`
- `experiments/_archive/ordinal-inference-cache-probe` and
  `experiments/_archive/ordinal-snlls-speed`

Mixed WLSMV compatibility (TASK-90) has strict 1e-5 relative reporting gates
at identical parameter points in `test_convention_mixed.R`, whole-bundle
ordinary gates in `test_lavaan_compat.R`, and a checked-in mixed golden composer
gate. Stage-1 moments/NACOV agree within 1.78e-8 / 5.34e-8 absolute. Grouped
theta endpoints differ by up to 7.59e-4: the equal-loadings canonical objective
is 0.037259343480420655 at magmaan versus 0.037259344025859446 at lavaan, with
tangent gradient norms 7.50e-9 versus 6.74e-6. Common-Stage-1 lavaan and bundle
reporting objectives agree within 1.4e-17 at both endpoints. Unequal-group
n_g-1 reporting weights differ from fitted n_g weights, so constrained endpoint
differences also include that convention. Single-group delta retains endpoint
gates; grouped/theta endpoint reporting has limited validation. Full evidence
is in [the convention observation](oracle-defects.md#mixed-wlsmv-endpoint-and-reporting-weight-conventions-task-90);
no oracle exemption or wider tolerance is used.

Known weak spots: older mixed robust primitive gates retain loose guards, and
lazy mixed WLS construction plus mixed theta SNLLS are still open.

### Robust U-Gamma and FMG Reductions

Protected by:

- `cpp/tests/unit/robust_test.cpp`
- `cpp/tests/unit/fmg_test.cpp`
- `cpp/tests/unit/weighted_chisq_test.cpp`
- `cpp/tests/golden/score_robust_golden_test.cpp`
- `r-package/examples/fmg.R`
- `r-package/examples/nested_test_ordinal.R`
- `cpp/tests/checks/robust_score/`

Known weak spots: robust MI has deferred estimator tiers and no df-greater-than
one joint-release path yet; FMG p-value transforms still need a self-contained
C++ golden independent of R examples.

### Optimizer Terminal Audit

Protected by:

- `cpp/tests/unit/terminal_audit_test.cpp`
- `cpp/tests/unit/fit_diagnostics_test.cpp`
- `cpp/tests/unit/optimizer_crosscheck_test.cpp`
- `cpp/tests/golden/lavaan_parity_golden_test.cpp`
- `project/design/terminal-audit.md`

Known weak spots: the absolute stationarity tolerance is provisional, and SNLLS
still needs a post-hoc full-theta audit path for apples-to-apples diagnostics.

### Parser and Lavaanify

Protected by:

- `project/grammar/grammar.ebnf`
- `cpp/tests/unit/parser_test.cpp`
- `cpp/tests/unit/mplus_input_test.cpp`
- `cpp/tests/unit/mplus_parser_test.cpp`
- `cpp/tests/golden/mplus_model_golden_test.cpp`
- `cpp/tests/unit/lavaanify_test.cpp`
- `cpp/tests/golden/parser_golden_test.cpp`
- `cpp/tests/golden/lavaanify_golden_test.cpp`
- `cpp/tests/golden/textbook_corpus_golden_test.cpp`

Known weak spots: grammar coverage is not yet mechanically reported, and
external corpus promotion is still staged case by case.

### R Boundary Reconstruction

Protected by:

- `cpp/tests/unit/api_sem_test.cpp`
- `r-package/examples/high_level_magmaan.R`
- `r-package/examples/model_spec_df_to_data.R`
- `r-package/examples/lavaan_partable_comparison.R`
- `r-package/examples/fit_measures.R`
- `r-package/examples/fmg.R`

Known weak spots: R examples catch workflow regressions but are not exhaustive;
mean-structure, group, ordinal, and post-fit reconstruction paths remain the
places to validate deliberately after R glue edits.

### Mplus categorical prerequisite probes (TASK-53; initial failures)

The initial native failures below are superseded by the explicit ordinal
preparation prerequisite section above; its equality-scale blocker is superseded by TASK-53.1.

The separate `mplus/probes_categorical.json` preserves the existing 972,086-byte
probe fixture unchanged and adds P-IV2's 12 variants. Mplus 9.1 rejects binary
and ordinal METRIC under DELTA and THETA. CONFIGURAL and SCALAR meaning is
recorded through generated commands and TECH1. Ordinal SCALAR fits converge
properly with tight Demo convergence controls; binary SCALAR does not converge
on this fixed sample, so those two variants supply no numerical golden.

The independent `check_mplus_categorical_conventions.R` specifies ordinal
SCALAR rows directly in pinned lavaan 0.7.2. Default lavaan WLSMV gives scaled
chi-square 27.482813436125 (delta) / 27.482814143160 (theta), whereas
`mimic="Mplus"` gives 27.535187965637 / 27.535188674204. The Demo prints
27.535 for both with tight convergence controls. Parameters (30), df (24),
and raw lavaan discrepancy agree between lavaan conventions; the robust scaling
factor changes. The default-convention difference exceeds the corpus's
printed precision allowance (approximately 0.000775 at this value).

The planner resolved the reference-convention decision in TASK-53 comment 5:
Mplus-mimic checks meaning and printed values; default lavaan checks native
numbers. This is not an oracle defect exemption or a reporting preset.

`check_mplus_categorical_conventions.R --native` additionally checks the same
independently written SCALAR model through `model_spec()` / native DWLS and
the `lavaan-0.7.2` preset. The diagnostic uses the existing simulation parity
allowance `1e-5 * (1 + max(abs(actual), abs(reference)))`. A fresh opt install
reproduces native common-estimate errors 0.128535813260 (delta) and
0.195224660091 (theta). All six group-2 scale/residual rows become fixed,
although the explicit model frees them. The preset instead returns
"lavaan QR coordinates require ordered affine rows from model resolution"
before optimization in both parameterizations. The native delta fit also
reports a covariance-admissibility warning. The diagnostic fails deliberately;
these results cannot be classified as a robust convention mismatch.

No frontend acceptance, preparation behavior, inference defaults, tolerances,
or corpus coverage is changed. The prerequisite numerical failure is resolved by TASK-53.1; the full
frontend increment is accepted by the completed gates below. Previously
completed opt Mplus checks pass: 34 cases / 12,362 assertions.

### TASK-53 categorical frontend validation and model-meaning gate

Frontend checkpoint `e28746a0` and DELTA repair `e6ec7a30` are followed by
completed threshold/scale lowering and ten independent default-lavaan
categorical golden cases (binary/ordinal single-group, grouped CONFIGURAL,
SCALAR and default under both parameterizations). Opt gates pass: spec
211 / 14,674 assertions, API 28 / 788, ordinal 148 / 11,872; the categorical
golden accounts for 2,398 assertions. Full installed magmaanlab passes 5,721
expectations, with two existing skips and two existing two-level warnings.
Live frontend tests cover grouped prepared/rebuilt fits and fixed/equal scales.
The original `probes.json` remains unchanged; P-IV2 adds 12 variants separately.

Corpus: 68 eligible, 31 accepted, 25 matched and six unsupported fit routes,
zero failures; two matched cases have independently verified robust convention
differences. Increment 2 accepted/matched 22. The unchanged 2,440-input manifest
(1,117 distinct inputs) has reader 291→456 and model 217→359, every rejection
classified. Logs are under `~/.cache/magmaan-logs/task-53-frontend-*`.

The optional `regen_oracle_mplus_categorical.R --demo` meaning gate asserts
free-parameter count, df, estimates and scaled tests at unchanged printed
precision against explicit lavaan `mimic="Mplus"` references. Group equalities
are specified by the independent syntax, with `group.equal="none"` preventing
mimic from adding its default grouped restrictions. Demo SEs remain frozen
observations, not model-meaning assertions (TASK-53 decision #24); magmaan SEs
remain gated against default lavaan through retained-estimate WLSMV bundles.

For seed 533072's single-group ordinal DELTA fit, Mplus 9.1 prints U3 loading
SE 0.083 versus lavaan 0.7-2 mimic 0.08355350 (ratio 0.9933755); U6 0.073
versus 0.07357317 (0.9922095); U1 first-threshold and U6 second-threshold
0.053 versus 0.05356258 (0.9894970). `gamma.vcov.mplus=FALSE` multiplies mimic
SEs by sqrt(599/600) = 0.9991663, but U6 and both thresholds still exceed
printed precision. The remaining component is unexplained. This is an observed
Mplus/lavaan convention difference, not an oracle exemption; native default-
lavaan SEs and tests pass their existing 1e-5 gates. Reproduce with
`Rscript cpp/tests/tools/regen_oracle_mplus_categorical.R --demo`; ignored
inputs/data/output remain under `~/.cache/magmaan-logs/mplus-categorical-golden/`.

Final decision-24 regeneration passes all ten Demo model-meaning cases. All
ten default-lavaan reference payloads remain exactly unchanged; only Demo
observations were added. Rechecked categorical C++ goldens: 2,398 assertions;
reader/lowering: 68 assertions; tracked-file, layering and diff checks pass.


### TASK-54 growth, constraint and indirect frontend validation

Fourteen continuous and four categorical independent Demo/lavaan goldens
protect polynomial/free-time/piecewise/group growth, DELTA/THETA growth,
NEW free/derived coordinates, affine and nonlinear equalities, DO, shared
sqrt/pnorm/log10 and total/specific/VIA/factor indirect effects. Demo gates
model meaning at existing printed precision; frozen default-lavaan gates
estimates, SEs, definitions, df and tests at 1e-5. Sources are the two
`regen_oracle_mplus_growth*.R` tools; fixtures are the two
`cpp/tests/fixtures/mplus/golden_growth*.json` files. C++ consumers are the
spec growth golden and ordinal categorical golden tests. Live lab tests in
`test-mplus-growth.R` cover independent references, saveRDS/rebuild and active
LS/FIML restrictions. Inequalities are deliberately refused with the CN01
PSD/barrier redirect; ordinal nonlinear equalities explicitly name TASK-54.2.

Regression: auxiliary NEW coordinates have no matrix cell. Initializing their
location to block -1 and excluding them from FIML analytic second derivatives
prevents invalid block access while preserving restriction derivatives.
Guard: the new-free ML frozen/live fit and ULS/missing-FIML restriction gates.
Regression: nonlinear fit extras and the lab df adapter previously rejected or
omitted fitted equality rank; both now use the fitted coordinates.
Guard: explicit/implicit nonlinear ML df/test parity and LS/FIML residual checks.

Opt validation: spec 215 cases / 16,935 assertions; API 28 / 788;
constraint checks 42 / 534; defined/effect checks 7 / 71; categorical golden
1 / 3,144. Full installed lab: 6,367 expectations, two existing skips and
two existing two-level warnings. Live R growth data use independently
specified interior population moments; the frozen C++ continuous cases retain
their original sample and oracle admissibility observations.
Corpus: 50/68 accepted, 44 matched, six explicit unsupported fits, zero
failures and five verified categorical convention differences. The input
sweep accepts 658 reader / 614 MODEL inputs from 2,440 files, all rejections
classified. Logs: `~/.cache/magmaan-logs/task-54-*.log`.

Auxiliary NEW route regressions in `test_mplus_auxiliary_routes.R` compare
linear and nonlinear auxiliary-coordinate fits with equivalent models without
NEW across ML, GLS, ULS, FIML, PSD and barrier, plus linear ordinal DWLS.
They check covariance and standardized estimates/SE inputs, policy calls,
existing typed nonlinear robust-inference refusals, and typed noniterative CFA
refusals. The C++ multi-information penalty test independently checks zero
auxiliary gradient and Hessian rows, unchanged penalty values after changing
the auxiliary coordinate, and zero continuous moment curvature.

### TASK-56 Mplus stability closeout

The current language boundary is recorded once in the
[coverage matrix](../grammar/mplus.md#coverage-matrix): 121 primary CL/LX/NM/MS/LB/
DF/MG/IV/CT/GR/CN/DA IDs, plus settled LX02a and NM01a aliases. Lab help and
inventory classifications match it. The normative EBNF commentary now describes
implemented increments 1–5 and DELTA restrictions; parser utilities cite their
productions. The 66 rejection-class contracts in `detail_mplus_diagnostic.hpp`
were reviewed together with the reader/MODEL call sites and categorical
preparation/API/lab boundaries. Each diagnostic identifies the construct and
source span, describes Mplus behavior, gives the magmaan reason and a specific
remedy. DEFINE/selection direct users to R, family exclusions cite Not planned,
and inequalities retain the scope/PSD/barrier explanation. Only the exact
ordinal nonlinear equality boundary names pending TASK-54.2.

`cpp/tests/tools/check_mplus_sanitizers.py` builds the standalone reader/parser
against its required parse sources and `mplus_model_sweep.cpp` against the
ASan/UBSan library target. Neither driver is in CI. The script header records
commands; every input gets a ten-second subprocess deadline. The 2,440-file
manifest includes original case inputs, raw files and ZIP members. The existing
unreadable `ex11.8imp.zip` is reported separately and supplies no extractable
inputs. Results: reader 658 accepted / 1,782 rejected; MODEL parser and API
lowering each 614 accepted / 1,826 rejected; **zero crashes, ASan/UBSan reports,
hangs, unclassified rejections or incomplete parser diagnostics**. Both runs
apply the existing expansion bounds. ASan memory-access and UBSan checks are
active; LeakSanitizer is disabled because the ptraced sandbox cannot run it.
No leak-check claim is made. The initial nonsanitized sweep linked a stale warm
archive and aborted; rebuilding it restores the same 658/614 counts as the
independently compiled sanitized reader/parser, with no source crash found.

Every firing rule ID is counted below. Counts are files containing an ID,
once per file per stage, across **all** aggregated diagnostics; they are not
mutually exclusive first-rule counts. API counts equal MODEL counts. Example
inputs are paths in the corpus or extracted manifest; `archive_N/` denotes the
Nth ZIP in the R sweep's sorted archive list. Full diagnostic/example records
remain in `~/.cache/magmaan-logs/task-56-sweep-dev/summary.json` and `results.tsv`;
original third-party inputs remain untracked.

| Rule | Reader files | MODEL/API files | One example input |
| --- | ---: | ---: | --- |
| CL02 | 43 | 43 | `cases/muthen_2017/muthen_2017_ch8_ex8_29_2/source/original.inp` |
| CL04 | 88 | 88 | `raw/mplusbook/ex10.12.inp` |
| CL06 | 66 | 66 | `cases/mplus_users_guide_v8/mplus_users_guide_v8_ch11_ex11_4/source/original.inp` |
| CL10 | 109 | 109 | `raw/little/source/CH12_mplus/LCA/LCA_T2_covid5items_1CLASS_STARTS_00100_SATURATED.inp` |
| CL13 | 139 | 139 | `cases/muthen_2017/muthen_2017_ch2_ex2_12/source/original.inp` |
| CL14 | 168 | 168 | `raw/mplusbook/ex5.22.inp` |
| CL15 | 533 | 533 | `cases/mplus_users_guide_v8/mplus_users_guide_v8_ch11_ex11_1/source/original.inp` |
| CL16 | 436 | 436 | `cases/muthen_2017/muthen_2017_ch1_ex1_19_art2/source/original.inp` |
| CL17 | 881 | 881 | `raw/brown_archive/tab11.4.inp` |
| CL18 | 329 | 329 | `raw/brown_archive/tab11.2.inp` |
| CL21 | 2 | 2 | `raw/mplususerguid/ex5.33.inp` |
| CL22 | 32 | 32 | `raw/little/source/CH12_mplus/LCA/LCA_T2_covid5items_1CLASS_STARTS_00100_SATURATED.inp` |
| CL23 | 55 | 55 | `raw/mplusbook/ex5.11_Part1.inp` |
| CL24 | 44 | 44 | `raw/mplususerguid/ex12.10.inp` |
| CL26 | 5 | 5 | `cases/mplus_users_guide_v8/mplus_users_guide_v8_ch6_ex6_18/source/original.inp` |
| CL29 | 628 | 628 | `raw/brown_archive/tab10.2.inp` |
| CL31 | 108 | 108 | `raw/mplususerguid/ex12.12.inp` |
| CL32 | 20 | 20 | `raw/brown_archive/tab11.2.inp` |
| CN01 | 0 | 3 | `cases/mplus_users_guide_v8/mplus_users_guide_v8_ch5_ex5_20/source/original.inp` |
| CN02 | 0 | 2 | `archive_45/ex8-3b.inp` |
| CT05 | 0 | 3 | `cases/mplus_users_guide_v8/mplus_users_guide_v8_ch3_ex3_12/source/original.inp` |
| CT07 | 0 | 3 | `cases/mplus_users_guide_v8/mplus_users_guide_v8_ch5_ex5_22/source/original.inp` |
| IV04 | 3 | 3 | `cases/brown_2015/brown_2015_tab7_16_invariance_mdd_metric/source/original.inp` |
| LX01 | 647 | 647 | `raw/brown_archive/tab10.2.inp` |
| LX02 | 1 | 1 | `raw/mplusbook/ex10.1_part2.inp` |
| LX03 | 28 | 28 | `raw/geiser_companion/extracted/Chapter 3 (SEM)/1_Simple_Manifest_Linear_Regression/2_simple_regression_with_centering.inp` |
| MG03 | 11 | 11 | `cases/mplus_users_guide_v8/mplus_users_guide_v8_ch6_ex6_18/source/original.inp` |
| MS01 | 0 | 5 | `raw/little/source/CH4_mplus/CH3/CH3.Fig.3.6.1indicatorB.inp` |
| MS08 | 0 | 7 | `cases/brown_2015/brown_2015_tab8_12_sem_formative_stress/source/original.inp` |
| MS09 | 0 | 21 | `raw/brown_archive/tab5.9.inp` |
| MS10 | 57 | 57 | `raw/mplususerguid/ex10.10.inp` |
| MS11 | 16 | 16 | `archive_45/ex5-3a.inp` |
| NM03 | 345 | 345 | `cases/mplus_users_guide_v8/mplus_users_guide_v8_ch11_ex11_4/source/original.inp` |

The end-to-end corpus gate remains **50 of 68 accepted, 44 matched, six explicit
unsupported fits, five independently verified categorical test-convention
differences and zero failures**. Accepted inputs with output files gate printed
free-parameter count and df directly (also cross-checking the derived book
metadata). The six unsupported conditional/mixed categorical routes additionally
gate both dimensions using independently counted conditional/mixed moments,
shared label coordinates and data-driven threshold/default-release coordinates;
they still have no fitting/inference claim. The existing fitted routes gate N,
convergence, dimensions, chi-square, H0 log-likelihood, printed parameter estimates,
NEW quantities and indirect effects where conventions agree. Tolerances and
convention classes are unchanged.

`r-package/tests/testthat/test-mplus-roundtrip.R` gates seven kinds: single group,
multiple groups, categorical, growth, NEW/constraint, indirect and an input-relative
data file. Partable fits retain the expression projection for defined reporting
and compare keyed rows after category completion (row order can differ).
Original-source rebuilds preserve identical tables, and fresh/prepared/refitted
estimates agree at 1e-5. A fresh Rscript process reloads portable specs and fits
via saveRDS/readRDS, rereads the data file, rebuilds and refits each case from both
spec and fitted-model metadata. All 85 assertions pass. The pending ordinal
nonlinear equality gate additionally checks its source coordinates and four
message parts.

Validation: opt build and all 1,525 C++ ctest cases pass; standalone ASan/UBSan
sweeps pass; rebuilt nonsanitized input sweep and end-to-end corpus gate pass;
lane-b opt lab installation and full magmaanlab suite pass (6,708 passing
expectations; two existing
intentional two-level skips, two existing covariance-admissibility warnings).
Vendor refresh, tracked-file and dependency-layering checks pass. Logs use
`~/.cache/magmaan-logs/task-56-*.log`. No ordinary-package code is changed;
TASK-57 retains that integration work.

### Ordinal nonlinear equalities (TASK-54.2)

Native all-ordinal ULS/DWLS/WLS bounded and compact fits now enforce nonlinear
rows through constrained LS; configured DWLS uses SLSQP with recorded preset
substitution. DELTA free-set compaction remaps nonlinear expression leaves.
Expected covariance, global tests, df, fit measures, nested and score inference
use stacked affine/nonlinear fitted Jacobians. Scalar expected profile scaling
has a regression for the formerly omitted nonlinear tangent. Ordinary policy,
observed/IJ and misspecification profile inference explicitly refuse these
models pending Lagrangian curvature. Categorical Mplus nonlinear equalities
are accepted; the Demo npar=11/df=3 meaning gate has a lavaan numeric gate.

Six frozen synthetic cases exercise five estimators, binary/three-category,
THETA/DELTA, product, combined and cross-group restrictions. The frozen golden
passes 2422 assertions. Default-start nonlinear nested/combined release-score
oracle values are replaced only by the proved fitted-start transitive reference
in `oracle-defects.md`; no tolerance was widened. Full required C++ ctest areas
pass 1358 tests; ordinary full testthat passes 1775 assertions (one PSOCK skip).
Full magmaanlab testthat passes 7025 assertions (two known warnings and two
skips). The null calibration rejects 26/500 for both implementations; see the oracle
entry and reproducible test-tool runner for scope and uncertainty.

### TASK-57 Mplus ordinary integration

P-MS08b adds eight isolated Mplus 9.1 Demo variants to
`cpp/tests/fixtures/mplus/probes_joint_x.json`; regenerate with
`Rscript cpp/tests/tools/regen_mplus_probes.R --joint-x`. All use the same
synthetic seed 58052. Every x variance mention (`x1 x2;`) brings both variables
into the joint model: TECH1 frees both means, both variances and their covariance,
without explicit WITH or mean statements. Complete and missing-X variants each
use 500 cases, 16 parameters and 4 df; printed ML/FIML chi-squares are 1.974 and
1.596. Independent explicit lavaan fixed.x=FALSE models agree on printed
estimates within 0.001 and chi-square within 0.002. The fixture also freezes
synthetic rows and lavaan estimates; the C++ TECH1 consumer checks free/fixed
cells, counts and equality partitions, including the earlier WITH-only probe.

`r-magmaan/tests/testthat/test_mplus.R` independently reconstructs the synthetic
observations and gates ordinary ML/FIML against live lavaan (estimates and
chi-square within 2e-5), the Demo's printed chi-square and N. It checks classed
refusals and exact completion text, applies input edits (including raw data via
`mplus_data()`), and verifies their fitted estimates. Continuous, grouped and
categorical inputs preserve source/schema through prepared fitting, reconstruction
and saveRDS/readRDS in a fresh R process (1e-8 estimate tolerance). Plain strings
remain lavaan; existing lab fixed-X rejection and ordinary defaults are unchanged.

Validation: opt build; all 1,534 C++ CTest cases passed; full magmaanlab testthat
7,147 assertions passed (two skips: unavailable lavaan multi-group two-level
reference and opt-in simulation; two existing two-level admissibility warnings);
full ordinary testthat 1,956 assertions passed (one existing PSOCK socket skip).
New fresh-process serialization gates ran without skips. Corpus unchanged:
68 inputs, 50 accepted, 44 matched, six unsupported fits, five convention
differences and zero failures. Vendor regeneration and tracked/layering guards
passed. No tolerance or ordinary inference default changed.
