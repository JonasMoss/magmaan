# magmaan TODO

Remaining-work backlog. Current state, architecture, and contracts live in
[docs/architecture/roadmap.md](../architecture/roadmap.md); this file only tracks unfinished work.

Effort tags: **S** bounded docs/fixtures/wrapper cleanup · **M** focused
implementation or test slice · **L** new estimator plumbing or cross-module
semantics · **XL** statistical design/research track before implementation.

## Correctness bugs

- **Open (minor, non-production).** **build_u_factor's Unstructured-weight
  reduced UΓ spectrum disagrees with the saturated/FIML spectrum for multi-group
  + cross-group equality.** With `InferenceSpec{moments = Unstructured}`,
  `build_u_factor` (`src/robust/robust.cpp`) produces a reduced spectrum that
  differs ~0.6% from the first-principles FIML saturated spectrum
  (`fiml_ugamma_spectrum`) for a model with **both** multiple groups **and** an
  across-group equality constraint (metric/scalar invariance), at the same θ̂
  (`tests/unit/fiml_test.cpp`, the metric/scalar cases). Single-group (±within-
  group equality) and multi-group *configural* agree to ~1e-6. The FIML spectrum
  is the correct one — it matches lavaan's unstructured `lavInspect("UGamma")`
  to ~1e-9 across configural/metric/scalar
  (`experiments/21-fiml-measurement-invariance-fmg`, `--lavaan-parity`). **This
  does not affect production FMG:** `infer_fmg_ugamma_spectra` /
  `fmg_tests()` use the **Structured** weight, and that path matches lavaan's
  structured UGamma to ~3e-7 for the identical metric/scalar models (exp 21).
  So the defect is confined to the non-default Unstructured weight + cross-group-K
  combination. Likely site: the reduced meat (`reduced_gamma_sample*`) or the
  per-block weighting when the global kernel basis `N` couples blocks under the
  S-based (vs Σ̂-based) weight; the bread/`A` and global QR of `A` appear correct.
  (Earlier triage mis-scoped this to the Structured/production path; that was a
  model-spec mismatch in the ad-hoc check — magmaan labels vs lavaan
  `group.equal` produced different χ² — not a build_u_factor defect. With
  identical specs the structured spectra match.)

- **Fixed (properly, ordinal-aware).** **Standardized solution for
  ordinal / mixed-ordinal fits.** Previously the generic standardize formula
  `λ·√Var(η)/√σ_rr` (`src/measures/standardized.cpp`) divided by the assembled
  `σ_rr ≈ λ²ψ + 1`, but a delta-ordinal `y*` is unit-variance, so a true `.6`
  loading came back as ~`.52`; `standardize_all`/`standardize_lv` were therefore
  guarded to *refuse* ordinal fits (a stop-gap). `measures::standardize::standardize_all`
  now takes an `ordinal_delta_unit` flag: for ordinal indicators under the delta
  parameterization it standardizes measurement loadings (the `Lambda` slot) by
  the latent SD only (σ_rr = 1), matching lavaan `std.all`. `Beta` is the m × m
  latent regression matrix and `Mid` the m × m latent covariance, so structural
  paths (latent → latent) always standardize as `b·SD(pred)/SD(out)` regardless
  of parameterization — there are no observed rows in `Beta` to special-case.
  The `require_not_ordinal` guard is removed from `standardize_lv`/`standardize_all`
  (C++ api + Rcpp bindings); `factor_scores` and `compute_defined` stay guarded.
  Validated against lavaan `standardizedSolution` for mixed CFA/SEM and the
  all-ordinal SEM (≤ ~1e-6 on loadings/paths); regression assertions in
  `r-package/examples/ordinal_dwls_wls.R` and the `--lavaan-parity` checks of
  `experiments/16-li-2021-mixed` / `experiments/19-li-2016-ordinal`. Surfaced
  while building exp 16; a follow-up bug (the original entry assumed an all-y
  RAM `Beta` with observed nodes offset after the latents and standardized
  structural paths by the predictor SD only, which mis-fired for *all-ordinal*
  SEMs and inflated structural paths past 1.0) was fixed while building exp 19.
  **Follow-ups:** (1) add a checked C++ golden fixture for ordinal-SEM
  standardized rows (currently only the R examples + the experiments'
  `--lavaan-parity` cover it); (2) `compute_defined` is the remaining unguarded
  `require_not_ordinal` R path — decide whether delta-method defined parameters
  are valid for ordinal fits.

- **Fixed.** **Mixed-ordinal stats inverted the full NACOV even for DWLS.**
  `mixed_ordinal_stats_from_data` (`src/data/ordinal.cpp`) eagerly inverted the
  full NACOV to build the WLS weight and returned an error if it was not positive
  definite — which both wasted an O(m³) factorization (DWLS needs only the
  diagonal `W_dwls = 1/diag(Γ)`; the robust sandwich uses `NACOV` itself) and
  blocked DWLS when the inverse failed. At small `N` with many indicators (the
  20-variable Li 2021 SEM at `N = 200`) this made every mixed DWLS fit fail,
  while lavaan WLSMV (diagonal weight) converged. Two changes: (1)
  `mixed_ordinal_stats_from_data` / `data_mixed_ordinal_stats_from_{raw,df}` take
  `full_wls_weight = true`; a DWLS-only caller passes `false` to skip the inverse
  entirely (W_wls left empty). (2) Even with `full_wls_weight = true` the inverse
  is non-fatal — a singular NACOV leaves W_wls empty and DWLS / robust proceed.
  An explicit full-WLS fit on a skipped/empty weight reports it clearly
  (`validate_stats` / `weight_factors`, `src/estimate/ordinal.cpp`). The eager
  default preserves the existing W_wls-vs-lavaan golden/unit contracts. The same
  `full_wls_weight` flag and non-fatal inverse were ported to the **all-ordinal**
  path (`ordinal_stats_from_integer_data` / `pairwise_ordinal_stats_from_integer_data`,
  `data_ordinal_stats_from_{raw,df}`) while building `experiments/19-li-2016-ordinal`,
  where the all-ordinal 20-indicator SEM at `N = 200` hit the same singular NACOV.

- **Investigated, NOT a bug (recorded so we don't re-chase it).** The continuous
  ULS test-statistic builders use *different* base statistics for the standard
  vs robust paths: `continuous_ls_chisq` (`src/robust/weighted_inference.cpp`)
  reports the Browne residual-based NT statistic for ULS (`weight.empty()`),
  while the robust path `robust_continuous_ls` → `robust_weighted_moments` scales
  off `2N·fmin`. This looks like an inconsistency but **faithfully mirrors
  lavaan**: lavaan's default ULS test is the Browne residual NT statistic, but
  `estimator="ULS", se="robust.sem", test="satorra.bentler"` reports a *different*
  unscaled base (the naive `N·F_ULS`) that the scaling then corrects. Verified by
  the lavaan-generated parity fixtures (`tests/fixtures/parity/hs_3factor_ls*`,
  regenerated by `tests/tools/regen_parity_fixtures.R` with both the default and
  the `robust.sem` ULS fits) and by `lavaan_parity_golden_test.cpp` checking
  ULS `chisq_standard` / `satorra_bentler` / `scaled_shifted` against those
  lavaan values; they pass on the current code and a "uniform Browne base"
  rewrite breaks them. Leave the split as is.

- **Resolved (convention chosen + tests made honest); doc still owed.**
  Continuous GLS/WLS standard χ² uses multiplier `N` (`2N·fmin = N·F`), whereas
  lavaan reports `(N−G)·F` (Wishart/unbiased). Characterized exactly against the
  lavaan-generated fixtures: on `hs_3factor_ls` (N=301) magmaan/lavaan =
  77.7290/77.4707 = 83.5963/83.3186 = 301/300 to 6 digits, and the relation is
  exactly `(N−G)/N` across single- and multi-group cases. ULS already carries
  `N−G` via `browne_residual_nt`'s `n_used`, so it matches lavaan directly.
  **Decision:** keep magmaan's `N` multiplier (we use `N` always; `N−1` only
  where it is genuinely the unbiased quantity, e.g. the sample covariance, which
  IS already the `N−1` divisor). Not a bug. Instead of changing the estimator we
  made the golden tests honest: `lavaan_parity_golden_test.cpp` and
  `ls_golden_test.cpp` now pin `magmaan_chisq·(N−G)/N == lavaan` to 5e-3 (was a
  loose `max(5e-2, 2·N·fmin·2e-3)` ≈ 1.2 gate that silently absorbed the gap).
  **Still owed:** the user-facing documentation of this convention (see the docs
  item below) and the `magmaan-gls-chi-convention` note updated to say `(N−G)/N`,
  not just "half".

- **M, organization.** Revisit how the continuous/ordinal test-statistic and
  robust-inference builders are organized. Today the standard `test()` path uses
  `continuous_ls_chisq` (Browne residual for ULS, `2N·fmin` for GLS/WLS) while
  the robust path uses `robust_ls_standard_chisq` (`2N·fmin` uniformly), and the
  continuous vs ordinal robust paths reach `robust_weighted_moments` with
  different `fmin` scaling conventions (`src/estimate/ordinal.cpp:2509,2652`
  passes `est.fmin`, the continuous path passes `2·fmin`). The per-estimator base
  statistic is correct (it matches lavaan; see the two entries above), but the
  rules are spread across several call sites. Audit for a single, documented
  contract on what each path's base χ² means per estimator (continuous
  ULS/GLS/WLS standard vs robust, ordinal DWLS/WLS, mixed) and where the
  estimator-kind dispatch lives, so the deliberate ULS standard-vs-robust split
  is obvious by construction rather than implicit.

- **S/M.** Add Kline-corpus parity coverage for the Guo measurement-invariance
  models (`guo_mi_weak`, `guo_mi_strong`, `guo_mi_partial_strong`). The
  underlying bug — duplicated formula terms (`NA*LM1 + c(a1,a2)*LM1`) becoming
  two partable rows for one matrix cell, leaving a phantom parameter that moved
  the analytic gradient but not the model moments — is fixed: `lavaanify` now
  merges repeated `lhs op rhs` terms within a block (`src/spec/build.cpp`,
  `build_group_template`), and `guo_mi_weak` fits to the lavaan reference
  (`chisq = 25.5159`, `df = 11`) under both L-BFGS and PORT. Root-cause
  regression tests live in `lavaanify_test.cpp`; what remains is wiring the
  finalized `corpus/textbook-corpus/raw/kline` corpus into an end-to-end
  parity test.

- **Low priority, S/M.** **ADF/WLS Γ̂ inversion policy and conditioned KKT
  diagnostics** — At small N with a binary covariate, the empirical Browne
  NACOV can be positive-definite to working precision but numerically rank
  deficient. Reproducer: `muthen_2017_ch2_ex2_1__adf` (Hayes PROTEST
  mediation; N=129; binary treatment 41/88 split; saturated path model; Γ̂ has
  8 well-conditioned eigenvalues 10.4 → 0.09 plus one at 7.5×10⁻¹⁷). The
  strict `chol2inv(chol(Γ̂))` audit amplifies machine residuals at a saturated
  fit and reports a non-stationary projected gradient, while the
  `experiments/00-lavaan-parity` conditioned diagnostic trims Γ̂ to rank 8/9
  and gives `conditioned_grad_inf ≈ 9e-15`. Do not silently replace the ADF/WLS
  objective. If this moves into C++, make it an explicit rank-revealing helper
  or policy (`strict` vs `spectral_truncate`) that returns diagnostics such as
  rank, tolerance, `rcond`, dropped rank, retained weighted residual, and
  conditioned projected-gradient norm. The experiment report now treats this
  as diagnostic telemetry only; core fitting/inference should stay strict until
  a downstream need justifies an explicit API.

## Robust score / modification-index tests (frontier)

`inference::frontier::{modification_indices,score_tests}_robust` landed for
complete-data ML, both breads, single group (see roadmap). It reuses
`robust::param_space_sandwich` (extracted from `robust_se`'s setup) for the
bread/meat and rescales each candidate by `c = gᵀB1g / gᵀA1g`. Validation in
place (four independent angles):

- exact reduction-to-NT under Γ_NT and the independent-assembly check of A1/B1 vs
  explicit `Δ'WΔ` / `Δ'WΓ̂WΔ` (`tests/unit/score_robust_test.cpp`); plus the
  robust reduction anchor on the lavaan ML fixtures (`score_golden_test.cpp`).
- R-internals oracle: `tests/golden/score_robust_golden_test.cpp` vs
  `tests/fixtures/score/0006_robust_release_mlm.score_robust.json`, assembled by
  `tests/tools/regen_robust_score.R` from lavaan's `delta`/`wls.v`/`gamma`/`ceq.JAC`
  (lavaan has no robust `lavTestScore`). The scaling `c` is convention-free in
  θ-space, so magmaan's own delta/weight/Γ̂ reproduce it; matched to 5e-3 with
  `c ≈ 1.34`. NOTE: the fixture is assembled at the *installed* lavaan (0.6.21,
  recorded in `_meta`); delta/gamma are version-stable so the number is not
  sensitive to the pin — refresh on a maintainer regen if desired.
- advisory simulation `tests/checks/robust_score/`: NT MI over-rejects (~21% at
  df=5, α=0.05) while robust MI ≈ α; robust score ≈ robust Wald z² ≈ SB scaled
  LRT (mean ratios within ~1%). Outside ctest.

Remaining work:

- **Deferred estimator tiers.** FIML robust MI (needs casewise score
  contributions per missingness pattern; `FimlEvaluator` builds info by finite
  difference, no Γ̂ analogue), continuous-LS (GLS/WLS) robust MI (`J'WΓ̂WJ` in the
  moment metric), and ordinal-DWLS robust MI (polychoric NACOV meat). Multi-group
  (per-block `n_b/N` denom weighting; currently single-group guarded).
- **df>1 total release.** Mean-scaled `T_total/c_total` plus an optional Imhof
  eigenvalue-mixture p-value for joint releases.
- **Perf.** v1 rebuilds the sandwich per augmented MI candidate; `Zc` and the
  per-block `L_Γ` are candidate-invariant, so cache them and rank-1-update the
  extra Δ-column (`// TODO(perf)` site in the robust evaluator).

## Local hardening and validation tooling

Local-first safety tooling for an AI-assisted repo. Design note:
[docs/validation/local_hardening.md](../validation/local_hardening.md).

- **S.** Calibrate the landed local LLVM coverage lane after its first real
  run. Initial tooling now exists as a `coverage` CMake preset plus
  `just coverage` and `just coverage-html`, all writing under ignored
  `build/coverage/`. Once exercised, adjust the ignore regex, object list, and
  domain-level interpretation notes as needed.
- **S.** Calibrate the landed local report/health recipes after first use.
  Initial tooling now exists as `just test-report`, `just test-quick-report`,
  and `just health`. Keep the command boring and local; sanitizer,
  parity-heavy, and optional optimizer lanes can be opt-in or later extensions.
- **S.** Add `docs/validation/test_ledger.md`: one table per validation area
  mapping subsystem, oracle, test kind, important files/tests, and known gaps.
  This should orient maintainers and agents; it is not a second backlog.
- **S/M.** Add a risk-map section to the test ledger for scary areas such as
  FIML, ordinal/mixed moments, robust test reductions, optimizer terminal
  audit, parser/lavaanify, and R boundary reconstruction. Each entry should
  say "protected by" and name the concrete test files or reports to run.
- **S.** Adopt a lightweight regression-note convention: when a fixed bug gets
  a guard test, record the bug shape and the protecting test either near the
  test or in the test ledger.
- **M, audit.** Audit test tolerances so the suite is not "faked" by loose
  gates that pass a known divergence instead of pinning the right answer.
  First offender already fixed as the template: continuous GLS/WLS chisq was
  gated at `max(5e-2, 2·N·fmin·2e-3)` (≈1.2) in both `lavaan_parity_golden_test`
  and `ls_golden_test`, absorbing the whole `(N−G)/N` multiplier gap; both now
  pin `magmaan·(N−G)/N == lavaan` to 5e-3. Sweep the remaining golden/parity
  tests for similar slack tolerances, `MESSAGE`-only soft checks, and fixtures
  regenerated from magmaan's own output rather than an external oracle; for each,
  either tighten to the oracle or write down explicitly why the looseness is
  principled. Goal: a tolerance should encode a real numerical bound, never paper
  over an unexplained discrepancy.

- **M, docs.** Write the first numerical-conventions doc (we have none yet) and
  decide how the API should surface these choices. Must cover: the optimizer's
  `½·rᵀWr` objective scale (so `fmin = ½F`); the test-statistic multiplier policy
  — magmaan uses `N` always, reserving `N−1` only where it is genuinely the
  unbiased quantity (the sample covariance divisor is `N−1`; the χ² multiplier is
  `N`), which is why magmaan χ² relates to lavaan's `(N−G)·F` by exactly
  `(N−G)/N` for GLS/WLS while ULS already carries `N−G`; and the lavaan-parity
  implication so users are not surprised by the offset. Then consider whether the
  API should make the convention explicit/selectable (e.g. a documented
  `statistic = "N" | "N-1"` or report-both option) or stay fixed-and-documented.
  lavaan's own `N` vs `N−1` choices are partly literature-convention and not
  always principled; our value is a single stated rule. Pairs with the
  test-statistic-builder organization item under Correctness bugs.
- **M, later.** Layer CI on top only after the local commands are useful:
  `test-quick` on PRs/pushes, sanitizer validation on main or a schedule,
  heavy parity/optional optimizer lanes less often, and coverage as an
  artifact before considering badges. Avoid coverage percentage gates until
  the report has been calibrated by real maintenance work.
- **Partly landed.** Promote the FMG goodness-of-fit validation into the checked
  suite. `examples/fmg.R` (run by `just r-check`) now asserts value-for-value
  parity of the single-model FMG family (SB/SS/SF/ALL/pall/EBA/pEBA/pOLS x ML/RLS
  x biased/unbiased Gamma) against `semTests::pvalues` to <1e-6 (observed ~1e-8)
  on HS1939, gated on `semTests` being installed; this caught and now guards the
  unbiased-Gamma NT-term regression (see FMG / U-Gamma Performance). The fused
  unbiased path is unit-pinned by `reduced_gamma_nt_sample` in `robust_test.cpp`.
  Still wanted: a self-contained C++ golden for the FMG p-value transforms (fixed
  eigenvalues + chi-square -> p-value per method) so the eigenvalue-tail maths is
  gated without R. Coordinate with the exp-17 pEBA work, which owns the
  nested/multi-group side of the same machinery.

## Simulation primitives

Simulation-specific state and remaining work now lives in
[`simulation.md`](simulation.md). Keep this section as a high-level index for
cross-domain planning; put detailed generator/marginal/fixture decisions in the
simulation backlog.

- **Landed, first slice.** `magmaan::sim` now has a NORTA primitive with
  explicit calibration (`calibrate_norta`) and sampling
  (`simulate_norta_matrix`, `simulate_norta_raw`). The first marginal families
  are standard normal, standardized lognormal, and Tukey g-and-h. The same
  marginal-transform machinery also powers independent generators
  (`simulate_independent_matrix`, `simulate_independent_raw`) for simulation
  conditions where covariance is irrelevant or supplied downstream. Unit tests
  cover normal identity calibration, analytic lognormal calibration,
  infeasible lognormal correlations, independent Tukey generation, raw-data
  wrapping, and a mixed-marginal stochastic smoke.
- **Landed, IG slice.** `calibrate_ig()` implements the Foldnes-Olsson
  independent-generator moment equations for a chosen square root of `Sigma`.
  Cholesky is the default root; a symmetric square root is available for
  experiments that want the eigendecomposition convention. The calibration
  solves generator skewness and excess-kurtosis systems, fits generator
  marginals through `fit_marginal_to_moments()`, and can be sampled through
  `simulate_ig_matrix()` / `simulate_ig_raw()` or through a lower-level
  root-plus-generator overload for cached calibrations. Unit coverage checks
  both root conventions and a Tukey g-and-h stochastic covariance/moment smoke.
- **Landed, Vale-Maurelli / Fleishman slice.** `fit_fleishman_coefficients()`
  solves the classic third-order Fleishman polynomial coefficients from target
  skewness and excess kurtosis using exact normal polynomial moments.
  `calibrate_vale_maurelli()` then solves the pairwise Vale-Maurelli cubic for
  the intermediate normal correlation matrix, validates it by Cholesky, and
  feeds `simulate_vale_maurelli_matrix()` / `simulate_vale_maurelli_raw()`.
  Unit coverage checks the normal identity case, coefficient moment matching,
  the pairwise covariance equation, raw-data wrapping, infeasible moments, and
  a stochastic covariance/moment smoke.
- **Partly landed.** Moment-matching now has a reusable C++ API:
  `MomentMatchSpec` / `MomentMatchOptions` / `MomentMatchResult` plus
  `fit_marginal_to_moments()`. Tukey g-and-h fits `g,h` to target skewness and
  **excess** kurtosis, then returns a regular `MarginalSpec` that plugs into
  the existing NORTA and independent generators. Unit coverage pins the
  Kowalchuk-Headrick `(skew=3, excess-kurtosis=20)` example and verifies the
  fitted marginal feeds NORTA calibration. Pearson fitting is also landed for
  PearsonDS Types 0/I/II/III/IV/V/VI/VII: the C++ formulas mirror
  `PearsonDS::pearsonFitM()`, and the quantile path mirrors `qpearson()` through
  regularized beta/gamma/F/t inversions plus finite-integral Type IV bisection;
  `tests/tools/regen_pearson_sim_fixtures.R` generates checked PearsonDS 1.3.2
  goldens. Johnson SU/SB fitting is now landed too: it fits gamma/delta by
  deterministic quadrature against skewness and excess kurtosis and returns the
  same `MarginalSpec` used by NORTA and IG generator calibration;
  `tests/tools/regen_johnson_sim_fixtures.R` now generates checked SuppDists
  1.1.9.9 `JohnsonFit`/`qJohnson` goldens (shape pair, SU/SB type, and
  quantiles to ~1e-7), targeting the realized moments of the SuppDists fit so
  the comparison is exact rather than limited by SuppDists's loose moment
  solve. Fleishman
  fitting now routes the same moment-match API through the Vale-Maurelli
  coefficient solver, so IG can select `MomentMatchFamily::Fleishman`; the
  resulting cubic is treated as a generator transform, not as a public
  quantile. **Remaining
  simulation work is tracked in [`simulation.md`](simulation.md), including the
  Johnson SL exposure decision, special-functions policy, and NORTA calibration
  hardening.**

## FMG / U-Gamma Performance

- **Landed.** Add a fused Rcpp spectra helper for `fmg_pvalues()` that keeps
  `UFactor`, casewise contributions, reduced Gamma matrices, and eigensolves
  inside one C++ call. The current R composition repeatedly crosses the Rcpp
  boundary and reconstructs the transparent `UFactor` list, including
  per-block Cholesky factors, for each reducer.
- **Reverted (was a correctness bug).** The fused FMG unbiased path special-cased
  the NT term as the identity (`B' Gamma_NT(Sigma_hat) B = I` for the expected
  bread) and applied Browne's correction as a `-bI` diagonal shift. That is wrong:
  the Du-Bentler unbiased Gamma is a distribution-free estimator whose NT term is
  `Gamma_NT(S)` at the *sample* covariance, not the model-implied `Sigma_hat`.
  Using the identity made every `_ug` test (incl. the defaults `SB_UG_RLS`,
  `pEBA2_UG_RLS`) silently model-dependent and broke `semTests` parity by up to
  ~4e-4 in p-values (worse at small N). Fixed by `reduced_gamma_nt_sample()`
  (`B' Gamma_NT(S) B`, the Unstructured-moment reduction) fed as `M_nt`; parity
  back to ~1e-8 across the full grid, regression-guarded in `examples/fmg.R` and
  unit-pinned in `robust_test.cpp`.
- **Landed.** Add a tiled raw-data-to-reduced-Gamma path that generates
  complete-data casewise contributions in row blocks, multiplies each tile by
  `B`, and accumulates `M += Y'Y`. The reusable robust-core helpers
  `casewise_projected_rows_tiled()` and `reduced_gamma_sample_tiled()` cover
  complete-data `ProjectionExpected` U-factors, and the fused single-model R
  FMG spectra path uses them to avoid materializing and copying the full
  `N x p*` contribution matrix.
- **Partly landed.** For empirical spectra, eigensolve in row space when
  `df > N_total` on the fused single-model FMG path. Nonzero eigenvalues of
  `(ZcB)'(ZcB)` match those of `(ZcB)(ZcB)'`; large-p FMG models can have `df`
  much larger than `N`, where this avoids a much larger symmetric
  eigendecomposition. The row-space shortcut is biased-spectrum only: the
  Browne-unbiased low-rank form `M_u = -bI + (a/N)Y'Y + dww'` baked in the same
  wrong `Gamma_NT(Sigma_hat) = I` identity, so the unbiased path now always takes
  the full reduced route (`M`, `reduced_gamma_nt_sample`, `reduced_gamma_unbiased`).
  A correct row-space unbiased optimization would need the sample NT term folded
  in, not a scalar `-b` shift; defer until a tiny-N/high-df FMG case demands it.
- **Investigated, defer (and the shortcut was wrong).** Rank-one secular update
  for the unbiased spectrum once used the expected-bread identity,
  `M_unbiased = a M_sample - b I + d vv'`; with the correct sample NT term the
  `-bI` becomes `-b * B'Gamma_NT(S)B`, no longer diagonal, so the secular form
  does not apply. This could avoid the second full
  eigendecomposition, but it requires eigenvectors of `M_sample` to rotate `v`
  into the eigenspace. Eigen's `ComputeEigenvectors` path was ~4-6x slower
  than `EigenvaluesOnly` in synthetic df=300/739/1500 checks, so a perfect
  secular solve would still lose to two values-only solves at the p=40 target.
  Revisit only if a tridiagonal-level update or a LAPACK-backed eigensolver
  becomes available.

## API and R boundary

- **S/M.** Add or rename R wrappers only when the methods-developer workflow
  exposes a concrete gap in the staged API; the current `magmaan_core`,
  `magmaan_fit`, and post-fit wrapper surface is otherwise sufficient for the
  next R exploration pass.
- **S/M, experiment-motivated.** `experiments/20-deng-chan-2017-alpha-omega`
  replicates Deng & Chan's (2017) reliability-difference test, compares it to the
  existing Satorra-2000 nested LRT and a score test of tau-equivalence, and
  diagnoses a genuine non-regularity: because `omega >= alpha` with equality only
  at equal loadings, the contrast sits at an interior minimum under
  tau-equivalence, its gradient vanishes, and `omega - alpha` becomes a
  second-order (1/N) statistic with a quadratic-form null law — so the Deng-Chan
  Wald z under-rejects (Type I ~0.3-2%). The fix keeps the interpretable gap as
  the statistic but references `2n(omega - alpha)` against its weighted-chi-square
  law via Imhof (eigenvalues of the contrast Hessian times Gamma; tail from
  `CompQuadForm`), restoring calibration. Everything reachable with no new core:
  magmaan supplies the ML fits and `infer_gamma_nt`/`infer_empirical_gamma`, and
  Cronbach's alpha falls out exactly as the omega of a ULS tau-equivalent fit.
  Derivation in the experiment's `notes/alpha_omega_second_order.qmd`. If this
  earns a home in core, the natural shape is a `measures::frontier` reliability
  module (alpha, omega, the joint `omega - alpha` SE, and its second-order
  Imhof calibration) plus a thin R wrapper. Not required; the experiment runs
  today against the current surface.
- **L/XL.** Implement the ordinal SNLLS / Gamma workspace split sketched in
  [docs/design/ordinal-snlls-gamma-architecture.md](../design/ordinal-snlls-gamma-architecture.md):
  separate ordinal moments from lazy Gamma/weight construction, support
  fit-only ULS/DWLS without full Gamma materialization, add the free-threshold
  (`H = I`) profiling path first, then extend to WLS Schur-complement profiling,
  inference cache reuse, mixed ordinal data, and constrained thresholds. Landed:
  the initial C++ `OrdinalMoments` / `MixedOrdinalMoments`,
  `OrdinalGammaCache`, and `OrdinalWeightPlan` skeleton behind the legacy
  `OrdinalStats` / `MixedOrdinalStats` adapters, plus a fit-only all-ordinal
  `OrdinalMoments` path where ULS requests no Gamma and DWLS requests only
  diagonal Gamma. Landed next: free-threshold (`H = I`) delta profiling for
  that ULS/DWLS path, so the optimizer sees only the active correlation block
  and returned threshold estimates are reconstructed from observed sample
  thresholds. Full-WLS Schur-complement profiling is now also wired for the
  same free-threshold delta path. Cache-aware fit-plus-inference and
  inference-only robust DWLS/WLS reporting now reuse a full
  `OrdinalGammaCache` and match the legacy materialized `OrdinalStats` path.
  All-ordinal delta SNLLS for ULS/DWLS/WLS now has both the default
  threshold-profiled path, which fixes thresholds away before the generic
  Golub-Pereyra split, and a full-threshold path, which keeps the full
  threshold+correlation moment stack and treats threshold free parameters as
  Golub-Pereyra linear coordinates. Fixed threshold rows now stay in the
  profiled full-moment objective for ULS/DWLS/WLS and are covered for bounded
  fits plus WLS SNLLS. Threshold-local shared-label / bare-merge equality
  groups now share threshold-map columns and are covered for bounded fits plus
  WLS SNLLS; general linear threshold constraints are supported by the
  full-threshold SNLLS path. All-ordinal theta now also works through the
  cache-aware bounded path and SNLLS: the theta SNLLS path profiles threshold
  free parameters but keeps the standardized covariance block nonlinear.
  Fit-only mixed DWLS can now build `MixedOrdinalMoments` plus the exact Gamma
  diagonal through `mixed_ordinal_workspace_from_data()` and feed both bounded
  and full-threshold mixed SNLLS fits without full Gamma/WLS materialization.
  Remaining C++ estimator work on this track: lazy mixed WLS construction,
  mixed theta SNLLS, threshold-profiled general linear maps such as
  effect-coding-style constraints, reduced-Gamma robust-inference products that
  avoid full materialization where possible, and only-when-needed R/API polish.
  First benchmark slice landed as `magmaan_ordinal_workspace_bench` plus
  `experiments/06-ordinal-snlls-probe`: it covers all-ordinal delta fit-only
  ULS/DWLS/WLS across free, fixed, and shared threshold models and checks
  cache/materialization flags plus agreement with bounded/materialized paths.
  The fit-plus-inference companion also landed as
  `magmaan_ordinal_inference_workspace_bench` plus
  `experiments/10-ordinal-inference-cache-probe`, comparing legacy
  materialized robust reporting, minimal fit-only plus separate inference
  caches, and shared fit-plus-inference cache reuse for all-ordinal delta
  DWLS/WLS. A threshold-constraint support slice also landed as
  `magmaan_ordinal_threshold_constraint_bench` plus
  `experiments/12-ordinal-threshold-constraints`: shared-label thresholds fit
  through both profiled and full-threshold paths, while general linear
  threshold constraints intentionally reject in threshold-profiled fitting and
  fit through full bounded / full-threshold SNLLS. The construction-boundary
  benchmark also landed as `magmaan_ordinal_construction_bench` plus
  `experiments/13-ordinal-construction-boundary`; it now includes
  `ordinal_workspace_from_integer_data()`, where fit-only ULS returns
  `OrdinalMoments` without Gamma and fit-only DWLS returns `OrdinalMoments`
  plus the exact Gamma diagonal without full Gamma/WLS inverse materialization.
  The opt pilot shows ULS construction gets a clear reduction while exact DWLS
  diagonal construction remains dominated by score/Gamma-diagonal work. The
  ordinal SNLLS speed pilot now includes raw-to-SNLLS legacy/lazy rows that
  time construction, starts, cache setup, and profiled SNLLS together for
  ULS/DWLS, and it now stratifies all fit and end-to-end rows by delta/theta
  parameterization. Theta is tracked as cache-aware bounded plus
  threshold-only SNLLS profiling because standardized theta covariance moments
  are nonlinear in the remaining block. Landed mixed slice: materialized
  mixed continuous/ordinal delta SNLLS via
  `fit_mixed_ordinal_snlls_full_thresholds()`, matching existing bounded mixed
  DWLS/WLS on a focused unit test. Landed next: lazy fit-only mixed DWLS
  construction plus mixed-moments bounded/SNLLS overloads, with the speed pilot
  now timing legacy-versus-lazy mixed DWLS raw-to-fit rows. Landed next:
  mixed full-Gamma cache reuse for robust DWLS/WLS reporting, plus mixed
  lavaan robust scaled-test golden assertions. Remaining C++ estimator work on
  this track: tighten mixed robust scaled-test parity beyond the current loose
  guard, lazy mixed WLS construction, mixed theta SNLLS, threshold-profiled
  general linear maps such as effect-coding-style constraints, reduced-Gamma
  robust-inference products that avoid full materialization where possible, and
  only-when-needed R/API polish.
- **M/L.** Optional h-weighted polyserial path: a polyserial-only h-weighted
  moment builder — continuous-ordinal h objective, casewise threshold/rho
  estimating functions, bread/influence/Gamma construction, and splicing into
  the mixed moment stack so `NACOV`/`W_dwls`/`W_wls` rebuild. The all-ordinal
  h-score variants already have the generic Gamma machinery; the missing piece
  is the mixed polyserial estimating-equation design.
- **Landed.** FMG single-model goodness-of-fit tests are wired into the R
  inference/fit-measure surface. `magmaan_core$robust_fmg_test` and
  `magmaan_core$robust_fmg_ugamma_spectra` are canonical aliases for the C++
  primitives, `fmg_tests()` returns diagnostics (df, ML/RLS base statistic,
  method/parameter, UG flag, spectra/lambda list-columns), `fit_measures(...,
  fmg = ...)` attaches those diagnostics to ordinary fit measures, and
  `fmg_pvalues()` remains a named-vector compatibility view. Supported boundary
  is complete-data single- and multi-group ML (including mean structures) with
  retained or explicitly supplied raw data, plus **FIML/missing-data fits**
  (single- and multi-group): `fmg_tests()` accepts a `fit_fiml()` /
  `magmaan(..., estimator = "FIML")` fit and routes to
  `estimate::fiml::fiml_ugamma_spectrum` (Rcpp `infer_fiml_fmg_spectrum`), which
  builds the missing-data UΓ spectrum first-principles from the saturated EM
  information and ACOV (`U·Γ_mis`, biased gamma + FIML-LRT base only; `_ug`/`_rls`
  refused). Saturated H1 information now uses an analytic observed-row Hessian
  derived in `docs/research/notes/fiml_saturated_information.tex`, with the old
  finite-difference route kept as a C++ diagnostic comparator. It is the
  `h1.information="unstructured"` convention and reproduces lavaan's
  unstructured UGamma element-for-element on complete data (~1e-7, guarded in
  `examples/fmg.R`); semTests' unsound FIML rescale hack is not matched.
  Nested/two-model FIML FMG is also wired through
  `robust_nested_lrt()` / `nestedTest(method = "restriction_map")` when both
  fits are FIML and carry compatible `raw_data`; the C++ core uses the H1
  saturated eta-space `A/C/S` reduction with exact and delta restriction maps,
  and the R surface rejects complete-data-only nested methods and mixed
  FIML/complete pairs. Remaining FIML follow-ups: the
  `magmaan(estimator="FIML")` high-level path does not auto-enable mean
  structure (use `model_spec(..., meanstructure = TRUE)` + `fit_fiml`);
  multi-group FIML fits need explicit start/convergence care; nonlinear
  equality tangent-space support and pairwise-data FMG remain deferred.

## Benchmarks

Advisory local tooling, not a substitute for parity fixtures. Full design:
[docs/validation/benchmark_plan.md](../validation/benchmark_plan.md).

- **DONE (perf, core).** `robust::weighted_chisq::imhof_upper` was ~43x slower
  than `CompQuadForm::imhof` (fixed-grid composite Simpson over a guessed finite
  `[0,U]`). Replaced the integrator with the vendored QUADPACK `qagi` cone
  (`third_party/quadpack/`, f2c-translated `dqagie`/`dqk15i`/`dqpsrt`/`dqelg` +
  hand `d1mach`/`pow_dd`, public domain): now ~0.2 ms vs CompQuadForm's ~0.6 ms
  (3x faster, agreement to ~2e-16), with the dense-Simpson method retained as a
  fallback for the weakly-damped small-df tail (df ~ 1-2, large x) where qagi's
  epsilon extrapolation is unreliable. This was the dominant cost of every
  FMG/pEBA/pOLS p-value and the robust nested tests; experiment 17 dropped from
  11+ min to ~16 s.
  - **Option 2 (follow-up, S/M).** Consider replacing the f2c'd C under
    `third_party/quadpack/` with a clean C++23 hand-port of `dqagie`/`dqk15i`
    for readability — QUADPACK is public domain, so we can mirror it exactly.
- **Current perf note (core).** With imhof fixed and the fused spectra helper
  landed, the UΓ eigenvalue machinery remains the dominant cost of FMG/robust
  p-values at large p, but biased and `_ug` tests now share the expensive setup:
  one `UFactor`, one tiled casewise projection/reduced-Gamma accumulation, and
  the expected-bread NT identity `B'Γ_NT B = I`. Requesting any `_ug`
  (Du-Bentler / Browne unbiased Gamma) test still adds a second df×df matrix
  formation and values-only eigensolve for
  `M_unbiased = a M_sample - b I + d vv'`; the rank-one secular-update route
  was investigated above and deferred. Only compute the unbiased spectrum when a
  `_ug` test is requested (the recommended pEBA4_RLS/pOLS_RLS use biased Γ).
  Entry points: `r-package/src/robust.cpp::infer_fmg_ugamma_spectra`,
  `src/robust/robust.cpp::reduced_gamma_sample_tiled`,
  `reduced_gamma_unbiased`, and `ugamma_eigenvalues`. Validation bar remains the
  ~2e-16 CompQuadForm imhof parity and semTests pEBA/pOLS parity. Surfaced by
  experiment 17 (`experiments/17-foldnes-moss-gronneberg-peba`); see commits
  2524cf1 (QUADPACK/imhof) and e7cdb8c (experiment 17).
- **S.** Keep the build-loop timings table in `docs/architecture/roadmap.md` current after
  major workflow changes.
- **S/M.** Continue extending benchmark coverage beyond the current
  lavaan-backed complete-data ML, controlled-missingness FIML, and continuous
  ULS/GLS smoke cases to WLS, ordinal DWLS/WLS, and mixed categorical models.
- **M.** Extend ordinal SNLLS/Gamma workspace experiment instrumentation for the
  new all-ordinal delta path: report moment construction, diagonal Gamma, full
  Gamma, weight materialization, optimizer time, post-fit inference time when
  requested, objective/gradient diagnostics, iteration counts, cache
  materialization flags, and agreement with the legacy materialized ordinal
  path. Landed first: a C++ advisory benchmark target and experiment runner
  for ULS/DWLS/WLS fit-only cells across free, fixed, and shared thresholds.
  Landed next: a separate fit-plus-inference robust-reporting experiment for
  DWLS/WLS cache reuse against the legacy materialized robust path. Remaining:
  raw-data lazy construction boundaries, larger/literature-grade speed grids
  when needed, and any R/API wrapper polish justified by those results.
- **M/L.** Add a two-stage EM/saturated-covariance missing-data research path
  for comparison with direct FIML and pairwise covariance methods. Stage 1
  (saturated EM moments + sandwich ACOV ingredients) is exposed as
  `magmaan::estimate::fiml::saturated_em_moments` / R
  `magmaan_core$estimate_saturated_em_moments` — the methods-developer surface
  Savalei-Bentler (2009) needs. Stage 2 point estimate is now wrapped as
  `magmaan_core$estimate_two_stage_em(partable, raw_data, kind = c("ml","gls"))`
  (pure R glue: saturated EM → `estimate_gls`/`estimate_ml` on the EM
  moments). Used as the MSE comparator row in
  `experiments/08-pairwise-gls-efficiency/`. Still open: a packaged
  `estimator = "ML2S"` surface with the Savalei-Bentler corrected chi-square
  and SEs (consuming the Stage 1 `(H, J, ACOV)` ingredients).
- **M/L.** Van-Praag pairwise covariance machinery for the ugamma-fast
  pairwise-incomplete section and the pairwise rows of `papers/pairwise-robust-sem/`.
  Landed: `magmaan::data::pairwise_sample_stats` (per-block N-divisor pairwise
  cov, marginal means, availability π̂, overlap counts) and
  `magmaan::robust::pairwise_casewise_contributions(..., include_means)` —
  the Van Praag influence matrix Ψ̂ in the same block-stacked
  `[μ-cols | σ-vech cols]` G3b layout as `casewise_contributions`, so it
  plugs into existing `reduced_gamma_sample` / U-Gamma plumbing unchanged.
  R surface: `magmaan_core$data_pairwise_sample_stats`,
  `magmaan_core$robust_pairwise_casewise_contributions(X, mask, include_means)`.
  Pairwise normal-theory Γ_NT^pw is now materialized as a data primitive:
  `data::gamma_nt_pairwise(raw, pw)` returns the per-block p*×p* matrix via
  the pattern-grouped identity
  `Γ_NT^pw = Σ_k (n_k/n)·diag(a_k/π̂)·Γ_NT(Ŝ_pw)·diag(a_k/π̂)`. Plus a
  packaged pairwise-GLS fit `estimate::fit_gls_pairwise(pt, rep, raw, pw,
  x0, ...)` that weights by `(Γ_NT^pw)⁻¹` — the asymptotically efficient
  variant under MAR. R surface:
  `magmaan_core$data_gamma_nt_pairwise(X, mask)`,
  `magmaan_core$estimate_gls_pairwise(partable, X, mask)`. The literature
  Σ-only-weight variant remains accessible by handing
  `samp.S = pw.S` to the existing `fit_gls`; both reduce to the same fit on
  complete data (C++ unit test in tests/unit/pairwise_gls_test.cpp).
  Open: **`experiments/NN-pairwise-gls-efficiency/`** — the live research
  question on whether the asymptotically-efficient Γ_NT^pw weight pays off
  in finite samples. Scope: one-factor CFA at p ∈ {5, 10, 20} (overidentified
  df > 0 so the weight bites), n ∈ {200, 500, 1000, 2000}, MCAR rates
  {10%, 25%, 40%} plus one MAR design. Comparators: Σ-only-weight GLS,
  Γ_NT^pw GLS, complete-data ML oracle, FIML. Outcomes: per-parameter MSE,
  wall time, divergence-from-truth ratio. Now landed at
  `experiments/08-pairwise-gls-efficiency/` (with MAR, FIML oracle) and
  `experiments/09-pairwise-fit-speed/`. Inference-side Γ_NT^pw plumbing
  has also landed: `WeightMoments::Pairwise` bread in `build_u_factor`,
  `robust::reduced_gamma_nt_pairwise` meat reducer, and the matching R
  surface — see `tests/unit/pairwise_inference_test.cpp` and
  `r-package/examples/pairwise_robust_se.R` for the {bread × meat} 2×2
  validation. Remaining open items (pairwise μ ACOV, pairwise
  Browne-unbiased) stay in [`speculative.md`](speculative.md).
- **M.** Track objective value, gradient norm, iteration count, wall time, and
  agreement with lavaan-backed estimates where applicable.
- **S/M.** Extend the ordinal SNLLS speed pilot
  (`experiments/11-ordinal-snlls-speed`) when the paper needs stronger timing
  evidence. Landed: a native C++ benchmark target comparing full bounded
  DWLS/WLS, threshold-profiled bounded, full-threshold SNLLS, and
  threshold-profiled SNLLS all-ordinal delta fits across a Kreiberg-style
  repeated-measure family, one worked repeated-measure example, and threshold
  stress cells. The report now separates the full-to-profiled bounded gain,
  the full-bounded-to-full-threshold-SNLLS gain, and the incremental threshold
  profiling gain inside SNLLS; ULS has only the cache-aware comparator paths
  because no legacy full direct ULS comparator is wired into the pilot. Landed
  next: raw-to-SNLLS legacy/lazy rows using
  `ordinal_workspace_from_integer_data()` for ULS/DWLS construction. Landed
  next: the benchmark and report now split every row by delta/theta
  parameterization; theta rows intentionally use the cache-aware bounded
  comparator and threshold-only SNLLS profiling rather than delta's
  threshold-plus-covariance profiling split. Landed mixed extension: the same
  experiment now includes mixed continuous/ordinal delta rows comparing
  materialized full bounded DWLS/WLS with materialized full-threshold SNLLS,
  plus raw-to-fit mixed bounded/SNLLS construction timings. Landed next: mixed
  DWLS raw-to-fit rows now include the lazy `MixedOrdinalWorkspace` boundary,
  so the report can separate legacy mixed-stat construction from lazy
  mixed-moments-plus-Gamma-diagonal construction. The latest opt smoke pilot
  covers the compact `--smoke` grid with `q <= 4`; open follow-up is the fuller
  `q <= 12` literature-like sweep and optional lavaan context rows if the paper
  needs them.
- **S.** Extend the ordinal threshold-constraint support experiment
  (`experiments/12-ordinal-threshold-constraints`) only if the paper needs
  broader constraint evidence. The first opt pilot is single-group ordinal CFA:
  free and shared-label thresholds agree across profiled/full paths; true
  linear threshold constraints are rejected by threshold-profiled bounded/SNLLS
  and accepted by full bounded plus full-threshold SNLLS. Open follow-up:
  multi-group threshold-invariance/equality examples once the C++ or R-facing
  experiment harness can express groups without pulling in fixture generation.
- **S/M.** Extend the ordinal construction-boundary experiment
  (`experiments/13-ordinal-construction-boundary`) if the paper needs broader
  construction evidence. The first lazy opt pilot times fit-only ULS/DWLS raw
  workspace construction against eager legacy stats construction, projection
  to `OrdinalMoments`, diagonal/full Gamma cache copies, DWLS weight
  construction, and WLS reinversion across all-ordinal synthetic blocks up to
  `p = 16`, `c = 5`. Its lazy rows are now carried into the end-to-end ordinal
  SNLLS speed run.
- **M/L.** Convergence-note / start-value portfolio paper track
  (`papers/convergence-note/`). The skeleton, local resources, and first R
  simulation factories now exist for the De Jonckere-Rosseel small-sample SEM
  designs and the Ludtke-Ulitzsch-Robitzsch weak-loading CFA design. Next
  milestones: add runners that compare simple/FABIN/Guttman/Bentler/James-Stein
  starts, bounded random starts, and screened portfolios under equal budgets;
  filter genuinely hard cases by final gradient norm, objective gap,
  admissibility, PD margin, and multistart basin disagreement rather than
  optimizer failure alone.
- **M.** Extend the new near-singular ML continuation experiment
  (`experiments/04-near_singular_ml_continuation`) beyond the first
  diagonal-ridge path: the first target/profile grid now compares diagonal,
  scaled-identity, and raw-identity targets over several fixed lambda
  sequences; remaining work is direct ML vs ULS/GLS-start ladders and explicit
  cost-normalized budgets on rank-near-deficient sample covariance cases.
- **S/M.** Tighten the remaining Geiser GLS parity exceptions now documented
  in the parity-tier golden: manifest fixed.x path models need a resolved
  implied-moment comparison surface for exogenous observed moments, and
  `latent_ar_cross_lagged` needs either a stable same-basin optimizer recipe
  or a written alternate-optimum note.
- **S/M.** Extend the new Mplus SEM corpus beyond the v1 strict growth tranche.
  `corpus/textbook-corpus/raw/mplus_sem` now retains 80 first-pass translations and the tracked
  fixtures gate six continuous growth cases across ML/ULS/GLS/WLS. Remaining
  follow-ups: repair or hand-translate the skipped growth/CFA cases whose
  automatic Mplus-to-lavaan conversion is malformed, decide how to test
  observed-only path models without exercising the current saturated
  observed-path abort, and add categorical fixtures only for models that match
  magmaan's ordinal/mixed LS surface rather than Mplus logistic/probit response
  models.
- **S/M.** Extend the Little/Newsom textbook corpora beyond the initial strict
  tranche. The builders now extract 108 Little LISREL models and 142 Newsom
  lavaan fit calls, with grouped tracked manifests and strict lavaan-backed
  C++ parity for the currently supported continuous subset; the consolidated
  `magmaan_textbook_corpus_v1` manifest indexes these alongside Geiser and
  Mplus SEM, and its advisory overlap graph fingerprints same-data,
  same-syntax, same-shape, and same-oracle-structure cases for future paper
  mining. Remaining work: implement real LISREL `SE` selection and more
  complex matrix/constraint conversion for Little, promote the Newsom cases
  that now parse/lavaanify cleanly in the local 142-model sweep, and add
  ordinal/mixed parity checks once the desired categorical oracle surface is
  settled.
- **S/M.** Promote the remaining first paper-corpus seed and broaden the
  paper-corpus fixture surface. `external/paper_corpus` now owns scouting,
  minimal derived lavaan cases, validation, and magmaan JSON exports; magmaan
  consumes copied snapshots under `tests/fixtures/paper_corpus/`. `zxqvn` is
  promoted as a core complete-data ML point-estimate fixture. Remaining work:
  promote `hwkem`, document license/data-handling for that richer source,
  extract supported lavaan model/data pairs, classify RI-CLPM pieces outside
  the core parity surface where needed, and decide whether clustered-SE
  handling should become a later paper-corpus inference fixture.
- **S/M.** Add a small OpenMx tutorial corpus as an offline second-oracle
  cross-check, not as a runtime input format. Start with the dormant
  `openmx_mimic` case in `benchmarks/r/cases.R` and the OpenMx RAM examples
  noted in `docs/research/vendor-notes/benchmark_zoo.md`; hand-translate each
  retained model to lavaan syntax, harvest golden values from OpenMx
  (`mxRun`, `omxGetParameters`, `model$output`, and `mxGetExpected(.,
  "covariance")`), and tag those fixtures with `_meta.tool = "OpenMx"`.
  Keep the curated tier small (roughly 3-6 lavaan-expressible SEM/CFA cases,
  including a mixed continuous/ordinal CFA if licensing and fixture shape are
  clear), assert magmaan's implied moments against OpenMx's implied covariance
  to catch translation mistakes, and keep `mxModel`/`mxPath` parsing plus a
  runtime RAM frontend out of scope for this corpus task.
- **S/M.** Refine benchmark use of optimizer diagnostics now that fit results
  expose `optimizer_status` and final gradient norms. Benchmark scripts should
  distinguish clean convergence from line-search salvage or singular PORT
  convergence, and still avoid interpreting backend-specific missing iteration
  counts as real zero-iteration solves.
- **XL.** Design an optimizer terminal-point "ultimate verifier" track. The
  goal is to turn the provisional audit tolerance into an empirically justified
  convergence certificate for SEM fits rather than a hand-tuned cutoff. Build
  an offline verifier that records the backend-independent L1 residual
  `||projected_gradient||_inf / (1 + |f_recomputed|)`, objective/parameter
  gaps to lavaan or certified fixtures where available, cross-backend
  same-basin agreement, PD margins, active bounds, and constraint residuals
  over the Geiser/Mplus/Little/Newsom/paper corpora. Add a high-precision
  check mode (`long double`, MPFR/Boost.Multiprecision, or an R `Rmpfr`
  helper) that re-evaluates `f` and the gradient at reported terminal points
  and optionally performs a few high-precision local refinement steps. Use the
  resulting CSV/report to separate ordinary line-search noise-floor salvages
  from genuinely non-stationary same-objective points (e.g., Newsom
  `ex5_4`/`ex5_4c`) and to justify any default `TerminalAuditOptions`
  tolerance change in `docs/design/terminal-audit.md`.
- **M/L.** Decide whether `TerminalAuditOptions::stationarity_mode` should
  stay at Absolute (lavaan-matched) or switch to Relative once the verifier
  track above has data. v1 ships Absolute at `absolute_tol = 1e-3` to match
  lavaan's `check.gradient = TRUE` / `optim.dx.tol = 0.001` default; this is
  the first hard design call in magmaan and the calibration is genuinely
  unstable. The choice was made for cross-package honesty (every SEM
  package's convergence numbers should answer the same question), not from
  evidence that 1e-3 absolute is the right SEM-side noise floor. The
  Relative code path is fully wired and unit-test-covered
  (`tests/unit/terminal_audit_test.cpp`) so the experiment is one option
  flip away once the data exists; see `docs/design/terminal-audit.md`
  "Tolerance calibration" for the experimental sketch and
  `papers/snlls-constrained/reports/convergence-audit-notes.md` for the
  conversation that motivated the v1 default.
- **M/L.** Decouple terminal audits from optimizer coordinate systems so every
  fit result can be re-audited uniformly after any method-specific massage.
  The audit philosophy should be: produce a candidate result, expand/profile/
  reconstruct it as needed, then audit the resulting point against an explicit
  objective and coordinate contract. The first concrete gap is SNLLS: current
  `fit_snlls` / `fit_snlls_gls` convergence is audited in the profiled outer
  beta coordinates, relying on the inner alpha least-squares solve for the
  eliminated block. Add a post-hoc full-theta audit path, probably by
  generalizing `evaluate_at()`, so paper and benchmark harnesses can compare
  Full and SNLLS under the same expanded-theta KKT projected-gradient check.
  Record both verdicts where useful (`profiled_beta_stationary` and
  `full_theta_stationary`) plus objective/gradient norms, inner-solve
  rank/conditioning, active bounds, and any profile fallback diagnostics.
  Treat disagreements as high-value diagnostics for variable-projection
  conditioning, rank deficiency, or mismatch between the profiled and full
  objective geometry; do not make the full-theta audit a default hard gate
  until the SNLLS paper corpora and verifier track show how often it differs
  from the mathematically expected profiled verdict.
- **M.** Compare NLopt L-BFGS/SLSQP/VAR2/TNEWTON/BOBYQA, PORT/PORT-NLS,
  Ceres trust-region, Ceres dense BFGS, and SNLLS only on semantically
  appropriate cases; include shallow or Heywood-prone LS cases so bounds and
  conditioning stay visible.
- **S/M.** Extend the paper-local SNLLS benchmark package in
  `papers/snlls-constrained/r-package/` with the remaining defensible real
  cases (especially a Geiser/Eid LST covariance input and a documented MTMM
  variant) plus one Boomsma-style simulation design. Keep the runner reporting
  setup time, fit time, whole time, iterations, objective values, and errors.
- **S/M.** For the SNLLS paper, add a narrow finite-difference-gradient
  mechanism probe rather than another broad simulation grid. NLopt's
  gradient-based algorithms (`LD_*`, including L-BFGS) expect caller-supplied
  gradients; NLopt also provides derivative-free algorithms (`LN_*`, such as
  BOBYQA), but those are not the same as finite-differencing a BFGS gradient.
  If the paper needs an empirical Kreiberg-style explanation, add an
  explicit benchmark wrapper that finite-differences the existing scalar LS
  objective for ordinary LS and SNLLS under the same line-search optimizer, or
  report objective-evaluation accounting plus measured objective/Jacobian
  costs.
- **S.** In the SNLLS paper, spell out the implementation problem and solution
  more explicitly: the hard part is not proving separability, but turning the
  profiled objective into residual/Jacobian calls an optimizer can trust. Explain
  why published tensor-gradient derivations are useful as a code blueprint and
  correctness check, while keeping the main text focused on the projection
  identity, the affine constraint split, and the fact that magmaan reuses the
  ordinary LISREL moment Jacobian instead of hand-writing pages of tensor
  products.
- **M/L.** Revisit the provisional default-backend choice once the optimizer
  comparison studies land. `Backend::NloptLbfgs` is now the default and NLopt
  is a required dependency, but the final default should still be justified
  across ML, complete-data LS, bounded ordinal LS, FIML/direct optimizer
  callers, augmented-Lagrangian inner solves, and nonlinear-constraint paths
  (NLopt SLSQP and IPOPT). Document tolerance semantics (`gtol` vs NLopt
  `xtol_rel`), iteration/evaluation reporting, and bounded behavior before
  changing the default again.
- **M.** Maybe add exact Hessians for IPOPT. The first IPOPT adapter uses
  limited-memory Hessian approximation and supplies objective gradients plus
  nonlinear-constraint Jacobians only. Revisit after the optimizer comparison
  work clarifies whether exact objective / Lagrangian Hessians materially help
  ML, GLS/WLS/ULS, or nonlinear-constraint fits.
- **S/M, newsom corpus.** The Little/Newsom continuous golden
  (`tests/golden/textbook_corpus_golden_test.cpp`) is currently skipped because
  NLopt L-BFGS does not converge `newsom/ex5_5b` from `simple_start_values`.
  Same family as the documented `ex12_3` case in
  [`docs/backlog/newsom-corpus-failures.md`](newsom-corpus-failures.md): NLopt
  stalling early on a structurally awkward ML objective. Unskip once the
  starting-value path or harness-level cross-backend fallback handles it.
- **S, before shipping binary artifacts.** The repo now carries an MIT
  `LICENSE` that also notes the vendored BSD-3 PORT routines, which is
  sufficient for a source release — Eigen, Ceres, NLopt, and nlohmann_json are
  fetched at build time, not redistributed. Before shipping a
  binary or packaged artifact, extend this to a full dependency-license
  manifest (Eigen, optional Ceres, required NLopt, optional IPOPT, vendored
  PORT, and test-only nlohmann_json) with versions and redistribution
  obligations.
- **M/L, after coverage exists.** Promote the Ceres preset into regular
  validation where relevant without making the default build pay the Ceres
  dependency cost.
- **S, only if timings justify it.** Experiment with opt-in precompiled headers
  for Eigen-heavy local builds; keep them disabled unless they improve
  changed-TU rebuilds without worsening no-op or full rebuild ergonomics.

## Ordinal/SNLLS research

- **S, revisit only if the paper track needs it.** The h-score / WMA robust
  polychoric path has landed (all-ordinal h-weighted moments; see `roadmap.md`),
  so its phased plan note has been retired. Design rationale now lives in
  `docs/research/notes/h-polychorics.tex` and `robust_ordinal_gamma.tex`; the
  remaining concrete work is the h-weighted polyserial item under "API and R
  boundary". Revisit the threshold parameterization and positive-definiteness
  repair policy only if the robust-ordinal paper track demands it.
- **M/L.** Structured Gamma / model-implied fourth-order paper track
  (`papers/structured-gamma/`). A minimal `estimate::frontier` weight-matrix
  builder now exists for complete-data covariance-only pure CFA and is exposed
  to R as both a raw Gamma matrix and a WLS-compatible weight. The paper-local R
  simulation helper now generates one- and five-factor CFA scenarios, runs
  NT/ADF/fixed-mix DLS/MI4 working-weight fits, records MI4
  positive-definiteness failures as data, optionally applies eigen-floor or
  minimal NT-target shrinkage repair, and writes pilot CSV/R data from
  `scripts/sim_structured_gamma.R`. Remaining paper work: run and summarize a
  defensible grid, calibrate/report the MI4 repair policy, and decide later
  whether broader model classes are needed.
- **M/L.** Robust ordinal SEM paper track. Follow
  [docs/research/notes/robust_ordinal_sem_paper_plan.md](../research/notes/robust_ordinal_sem_paper_plan.md):
  build a paper-local simulation runner that emits tidy CSV for the Welz
  bivariate contamination design, the Welz/Foldnes-Gronneberg five-variable
  robust polychoric matrix design, an ordinal CFA downstream design, a Clayton
  copula stress test, and a small computation benchmark comparing ML, WMA hard
  cap, smooth h, and Huberized Pearson-residual moments. First milestone:
  reproduce the known Welz qualitative pattern for Designs 1--2 before adding
  SEM fits or broad copula grids.
- **M/L.** Ordinal SNLLS follow-up research. The all-ordinal delta
  ULS/DWLS/WLS prototype now exists for free, fixed, and pure-merge threshold
  models, the all-ordinal theta path covers cache-aware bounded/SNLLS point
  estimation, and the full-threshold SNLLS path supports general linear
  threshold constraints. Mixed continuous/ordinal delta SNLLS now exists for
  the materialized full-threshold DWLS/WLS path, and mixed fit-only DWLS now
  has a lazy workspace path. Mixed robust scaled-test fields are now checked
  against lavaan, but only with loose tolerances; exact mixed scale/eigen parity
  remains open. Use experiments to decide whether the next paper-facing C++
  work should be tighter mixed robust inference, lazy mixed WLS construction,
  mixed theta SNLLS, threshold-profiled general linear maps, or reduced-Gamma
  inference plumbing.
- **S/M, experiment extension.** `experiments/15-rhemtulla-2012` replicates the
  Rhemtulla, Brosseau-Liard & Savalei (2012) cat-LS-vs-continuous-ML horse race,
  but v1 covers only the symmetric-threshold, underlying-normal conditions
  (number of categories 2--7 × N). Two paper conditions are deferred: (1) the
  nonnormal underlying `y*` (skew 2, kurtosis 7 in the paper's convention),
  which is covered by the C++ cubic Fleishman / Vale-Maurelli primitive but
  still needs wiring into the experiment; this is the condition where cat-LS's
  own underlying-normality assumption breaks; (2) the moderate/extreme
  asymmetric-threshold conditions,
  whose exact threshold tables are in the paper's unavailable supplement (would
  need a documented rule validated against their Table 1 skew/kurtosis). Add
  these only if the replication needs to exercise cat-LS bias under nonnormality.

## Composite models

The C++ core and R binding slice have landed: `<~` parses as `Op::Composite`,
and `spec::build` now requires callers to choose a composite meaning explicitly
(`CompositeMode::None` is the default and errors on `<~`). The historical
Henseler-Ogasawara reflective sub-model remains available when selected, with
weights and delta-method SEs recovered post-fit, R-facing partables folded back
to `<~` shape, and R exposing a `composite_weights()` post-fit accessor. The R
lavaanify boundary selects this historical expansion explicitly. A parallel
native FC-SEM spec path now exists behind
`BuildOptions::composite_mode = CompositeMode::FcSem`: it keeps `<~` rows,
records name-free composite blocks, marker-fixes the first composite weight,
and emits fixed/derived placeholders for the composite indicator T blocks and
composite self-variances. A covariance-only `model::FcSemEvaluator` now
assembles W/T, derives composite loadings, solves derived composite disturbance
variances, and matches the pure-HS native lavaan implied-covariance fixture at
lavaan estimates. `estimate::ml_objective(FcSemEvaluator, SampleStats)` now
wraps that evaluator in the complete-data ML discrepancy, matches the HS
native lavaan chi-square fixtures at lavaan estimates, and supplies the first
central finite-difference gradient path for native `<~`. The low-level
`estimate::simple_fcsem_start_values` and `estimate::fit_ml_fcsem` path fits
the pure-composite, composite-plus-factor, and composite-structural HS native
fixtures from starts and matches lavaan's objective, implied covariance,
weights, loadings, and regressions. The composite fixtures now carry lavaan's
observed-variable order for stored covariance matrices. Native post-fit
expected information also exists: `FcSemEvaluator::dsigma_dtheta` computes a
central finite-difference covariance Jacobian, and
`inference::information_expected_fcsem` yields vcov/SEs matching lavaan for
the same reported fixture rows. Native `std.lv`/`std.all` standardization now
uses the FC-SEM evaluator's construct covariance and matches lavaan values/SEs
for the same free reported fixture rows. `standardized_rows_fcsem` also reports
native `<~`, `=~`, and `~` rows by lhs/op/rhs/group, including fixed marker
rows with lavaan-matching standardized SEs. Native FC-SEM df now subtracts the
composite indicator T-block moments from the user model, and
`measures::fit_extras_fcsem` plus the existing fit-measure helper match the
same native fixtures for chi-square, df, CFI/TLI/RMSEA, SRMR, loglik, AIC, BIC,
and sample-size-adjusted BIC. `api::frontier` now exposes the native FC-SEM
model builder, complete-data ML fit, expected SEs, fit measures, and
standardized row reporting; its model builder selects native FC-SEM and rejects
ordinary SEM syntax without `<~`. The R frontier mirrors this settled
single-group ML slice through `fcsem_model_spec()`, `df_to_fcsem_data()`,
`fit_ml_fcsem()` / `magmaan_fcsem()`, `fcsem_standard_errors()`,
`fcsem_fit_measures()`, and `fcsem_standardized_rows()`, with a checked
example under `r-package/examples/fcsem_frontier.R`. Ordinary R SEM helpers and
native FC-SEM helpers reject each other's prebuilt model/data classes and
native FC-SEM partable data.frames.
Remaining:

- **L.** Post-fit lavaan parity validation: minimal oracle fixtures exist under
  `tests/fixtures/composite/` for pure composite, composite plus common factor,
  and structural regression involving a composite. The diagnostic golden is
  wired but skipped while we decide whether native W-matrix parity belongs in
  low-level C++ fixture tests, an R frontier diagnostic, or both.
- **M/L.** Multi-group composites are in scope only after the single-group ML
  and R frontier slices stay green and only if lavaan handles them cleanly,
  including
  `group.equal = "composite.weights"`.
- **S, after parity fixtures are green.** Add composite benchmark cases.

Deferred until the ML slice is lavaan-validated: ordinal composites, FIML/LS
composites, robust corrections for composites, and composite mean-structure
rows.

## Core/frontier layout follow-ups

Deferred from the first core/frontier separation pass, which introduced
`api::frontier` and retiered `robust/fmg.hpp`, `estimate/gmm/dls_weight.hpp`,
and `estimate/pairwise.hpp` into `<domain>::frontier`. See
[docs/design/ideas.md](../design/ideas.md) for the tier model.

- **M/L.** Retier the `data/` research cluster (`h_score`, `pairwise_ordinal`,
  `pairwise_mixed`, `shrinkage`) into `data::frontier`. Blocked: core
  `data/ordinal.{hpp,cpp}` is entangled with these headers — `ordinal.cpp`
  defines `pairwise_ordinal_stats_from_integer_data` and uses
  `eval_polychoric_h_score`, the core ordinal options embed
  `PolychoricHScoreOptions`, and `ordinal.hpp` `#include`s `pairwise_mixed.hpp`.
  Moving the headers naively inverts the dependency (core → frontier). This
  work must first untangle `data/ordinal` — separating the core polychoric path
  from the research builders — then retier. `data/shrinkage.hpp` is the one
  cleanly separable header and can move first. R glue (`r-package/src/fit.cpp`)
  calls these `data::` symbols, so the retier needs one R-side requalification
  and a `just r-check`.
- **S.** Move the retiered frontier headers into `<domain>/frontier/`
  subdirectories so directory matches namespace again (`robust/frontier/fmg.hpp`,
  `estimate/frontier/dls_weight.hpp`, `estimate/frontier/pairwise.hpp`).
  Header-path moves; needs forwarding shims where `r-package/` includes a path.
- **L.** Relocate the misplaced `estimate/` files to `spec/`:
  `constraints.hpp` (24 includers), `nl_constraints.hpp`, `expr_eval.hpp`,
  `resolve_fixed_x.hpp` (13). A structural relayering — its own pass, with a
  design note settling whether constraint *evaluation* is `spec` or `estimate`.
- **M.** Move `reparameterize.hpp` to `optim/` (optimizer machinery, not an
  estimator); it is coupled to the `estimate/constraints.hpp` move above.
- **S/M.** Settle whether `cfa_utils.hpp` belongs in `spec` or `model`;
  depends on the start-values decision below.
- **M.** Gather the five start-value producers (`start_values.hpp`) into an
  `estimate::starts` sub-namespace; `start_values.hpp` has 16 includers and
  `spec::Starts` is part of the lavaanified-model triple, so this needs care.
