# magmaan TODO

Remaining-work backlog. Current state, architecture, and contracts live in
[docs/architecture/roadmap.md](../architecture/roadmap.md); this file only tracks
unfinished work. Completed items are folded into the roadmap (capabilities) and
[docs/validation/test_ledger.md](../validation/test_ledger.md) (regression notes
for fixed cross-subsystem bugs), not kept here. May-never-build ideas live in
[speculative.md](speculative.md).

Effort tags: **S** bounded docs/fixtures/wrapper cleanup · **M** focused
implementation or test slice · **L** new estimator plumbing or cross-module
semantics · **XL** statistical design/research track before implementation.

## Estimation and inference follow-ups

Small open items surfaced while fixing the standardized-solution and Kline/Guo
parity bugs (the fixes themselves are recorded in the test ledger; the ADF
`spectral_truncate` follow-up moved to [speculative.md](speculative.md)).

- **M — automatic robust fit-measure dispatch.** The lavaan-style robust/scaled
  fit-measure formulas are available for the core global-index family
  (`chisq.scaled`, scaled baseline, CFI/TLI, RMSEA CI/p-values) once the user
  and baseline scaling factors are supplied. Remaining work is estimator-specific
  plumbing: build the independence/baseline robust scaling for complete-data
  MLM/MLR and all-ordinal WLSMV baseline CATML ingredients, then let
  `fit_measures(fit, robust = "MLM"/"MLR"/"WLSMV")` compute those scalars
  automatically. The FIML corrected `XX3`/baseline reduction is implemented as
  `estimate::fiml::fiml_corrected_fit_measures`; lavaan's `MLR` fitMeasures
  fields are fixture references rather than the oracle for that explicit helper.
  The ML2S `robust.two.stage` robust/scaled CFI/TLI/RMSEA family is implemented
  as `estimate::fiml::two_stage_fit_measures` and lavaan-gated in the FIML
  golden fixtures.

- **Ordinal-SEM standardized solution / defined params / factor scores —
  mostly landed 2026-06.** Decided: `compute_defined` is valid for ordinal/mixed
  fits (a parameterization-agnostic delta-method transform) and its guard was
  removed at both the C++ api and the Rcpp binding. Ordinal/mixed factor scores
  now have their own categorical scorer instead of reusing the continuous
  regression/Bartlett path: EBM and ML are per-pattern Newton solves over the
  latent-response likelihood/posterior, and one-factor EAP uses the vendored
  QUADPACK infinite-interval integrator. Fixing the api flip surfaced a real bug:
  the ordinal api
  `Fit` carries the un-prepared structure while estimates live over the prepared
  (reduced) partable, so `standardize_lv`/`standardize_all`/`compute_defined`
  aborted on a theta-dimension mismatch — only the Rcpp path (which stores the
  prepared partable) worked. The api now reconstructs the prepared structure on
  demand (`prepared_structure` helper in `src/api/sem.cpp`). Coverage: C++ unit
  (`api_sem_test` ordinal case) and live lavaan value-parity in
  `r-package/examples/ordinal_dwls_wls.R` (ordinal `:=` value+SE to 5e-3; mixed
  std.all to 1e-3; all-ordinal std.all already covered). Remaining:
  - **Done 2026-06.** The oracle pin was realigned to `0.7-1.2691` and the
    checked C++ golden landed: `tests/golden/ordinal_golden_test.cpp`
    "ordinal/mixed standardized + := rows match lavaan" gates the `=~` loading
    std.lv/std.all rows (all-ordinal 5e-3, mixed 1e-3) and the
    `lprod := L2*L3` value+SE (5e-3) against the stored lavaan oracle
    (`ordinal/0015_defined_param_3cat_cfa` plus the per-fit
    `fits.DWLS.standardized` blocks emitted by `regen_oracle.R`).
  - **Done 2026-06-14.** Checked-in lavaan oracle for categorical factor scores
    landed. `regen_oracle.R` now emits `fits.DWLS.fscores` (via
    `ordinal_fscores_json` in `benchmarks/r/fixture_json.R`) for the single-group
    ordinal and mixed fixtures, and the golden
    `tests/golden/ordinal_golden_test.cpp` "ordinal/mixed factor scores (EBM/ML)
    match lavaan" gates `factor_scores_{ordinal,mixed_ordinal}` against
    `lavPredict()` at 5e-4 (the previously live-only parity from
    `r-package/examples/ordinal_dwls_wls.R`). The installed-lavaan method set is
    pinned by emitting only what lavaan's categorical `lavPredict()` supports:
    **EBM** single-group (all-ordinal and mixed; posterior mode, ~1e-5) and
    **ML mixed-only** (continuous indicators bound the mode). Deliberately not
    gated: all-ordinal ML (unbounded mode on extreme patterns) and EAP/precision
    (categorical `lavPredict()` rejects EAP, so no oracle; stays self-checked in
    `tests/unit/api_sem_test.cpp`).
  - **Resolved 2026-06-14 — not a magmaan bug; lavaan is the outlier.** The
    multi-group categorical EBM divergence (reference group matches lavaan ~2e-5,
    non-reference group drifts ~0.2 with `theta` matched to 1e-7) was root-caused
    to a **lavaan** defect: lavaan's multi-group categorical `lavPredict()`
    returns a *non-stationary* point for non-reference groups. Verified three
    ways: (1) magmaan's non-reference-group EBM matches an independent
    `optimize()`-based posterior-mode scorer built from lavaan's *own* extracted
    group-2 parameters to 1.9e-5; (2) at lavaan's group-2 score the posterior
    gradient is O(1) and the posterior density is *lower* than at magmaan's score
    (whose gradient is ~1e-7) — i.e. lavaan does not sit at the mode; (3) every
    ingredient lavaan feeds its scorer (prior `VETAx`, `THETA`, `TH(delta=FALSE)`,
    loadings, data, `th.idx`) is identical to magmaan's, consistent with lavaan's
    own `lav_predict.R` FIXME about categorical scores being "not identical (but
    close) to Mplus". So multi-group `fscores` are not emitted as a lavaan oracle.
    Instead the multi-group scorer is validated **transitively** against lavaan in
    `tests/golden/ordinal_golden_test.cpp`: for the unconstrained two-group
    fixture `0004` the per-group multi-group EBM equals an independent
    single-group fit on that group's data (machine-tight, ~3e-8), and the
    single-group EBM is lavaan-gated. No remaining work.
  - **M/L, only-when-needed.** Multi-factor EAP via adaptive Gauss-Hermite and
    non-diagonal residual-Theta orthant probabilities remain deferred in
    `speculative.md`; the landed scope is diagonal-Theta EBM/ML plus one-factor
    EAP for all-ordinal and mixed complete data.

- **Ordinal EAP reliability / posterior precision follow-up (2026-06).**
  Current state: `factor_score_precision()` reports one-factor ordinal/mixed
  EAP posterior variance/SE, sample-normalized PRMSE, and concrete ordinal
  reliability. Reference manifest and working notes live in
  [docs/reference/ordinal_reliability.md](../reference/ordinal_reliability.md);
  the derivation note lives in
  [docs/research/notes/ordinal_factor_score_reliability.md](../research/notes/ordinal_factor_score_reliability.md).
  Remaining:
  - **Done 2026-06-14.** One-factor ordinal EAP precision is now validated
    against simulated ground truth (not just plug-in self-consistency):
    `tests/unit/api_sem_test.cpp` "ordinal EAP factor-score precision tracks
    Monte-Carlo PRMSE" generates a five-indicator three-category one-factor
    model with retained latent `Z`, fits ordinal DWLS under `std.lv`, and pins
    (1) `pooled_prmse ≈ corr(Z, E[Z|Y])²`, (2) `mean Var(Z|Y) ≈` the realized
    EAP MSE, and (3) `concrete == 1 - mean Var(Z|Y)` exactly under unit latent
    variance (≈ `1 - MSE`). Gaps are ~1e-3 at n=8000 against 2e-2 gates. This
    surfaced and fixed a real bug: `std.lv` ordinal delta fits aborted in
    `compact_free_set` because `n_free()` (the max free index) dropped after the
    top response-scale variance was zeroed for elimination; the compaction now
    takes the original free count from the caller's `remove_free` metadata
    (`src/estimate/ordinal.cpp`, regression note at the fix site). Follow-up:
    - **Done 2026-06-15.** Checked-in lavaan oracle for `std.lv` ordinal CFA.
      `regen_oracle.R` now emits `ordinal/0016_std_lv_3cat_cfa` (model
      `f =~ NA*x1 + x2 + x3 + x4; f ~~ 1*f`, same data/structure as `0001` so it
      is a pure reparameterization: identical χ²/df, loadings rescaled by the
      latent SD). The existing delta-contract golden gates it directly —
      `magmaan` matches lavaan to χ² 3e-8 and all 12 free params (4 free
      loadings + 8 thresholds) to 4.6e-8, df=2 — and the standardized/EBM
      factor-score arms ride along. The threshold-profiled SNLLS arm is
      structurally N/A under std.lv delta (the conditionally-linear
      {Psi, Theta} block is empty: latent variance fixed to 1, response
      variances fixed by `~*~ 1`), so the SNLLS golden pins the expected
      "no conditionally linear free parameters" diagnostic rather than
      skipping; the bounded full-Newton fit carries the parity.
  - **M.** Add bootstrap CIs as the first inference surface, probably frontier
    or R-only first: parametric bootstrap from the fitted ordinal/mixed model,
    rebuild stats, refit, recompute both coefficients; optionally add
    nonparametric row bootstrap for robustness checks.
  - **L.** Analytic SEs require stable casewise ordinal moment influence
    plumbing for `theta_hat` plus finite-difference derivatives of posterior
    moments wrt free SEM parameters. Concrete reliability with fixed unit
    latent variance is the simplest analytic target; sample PRMSE needs the
    three-moment delta method. Follow Sung and Liu's IRT SE paper, replacing
    their item-parameter influence function with magmaan's ordinal SEM
    LS/GMM influence path.
  - **S/M, keep separate.** Do not collapse CTT reliability of the EAP score,
    EBM/ML determinacy summaries, and multi-factor EAP PRMSE into the same
    public coefficient; add them only with explicit names and a concrete
    downstream consumer.

- **Ordinal stats-construction perf headroom (2026-06-12 audit).** Workspace
  construction dominates ordinal/mixed wall time (fits are sub-3ms). Landed:
  cell-cached `ordinal_pair_scores`, an `x_tol` stop for the rho searches, and
  (second pass, same day) the two M items: the polychoric/polyserial ML rho
  searches are now safeguarded Newton on the closed-form score
  (`detail_rho_search.hpp`, ~6 grid evals vs ~44 golden-section), and per-rho
  bvn evaluation shares the `(K_i+1)(K_j+1)` corner grid
  (`ordinal_bvn_corner_{cdf,pdf,pdf_drho}`) instead of 4 `bvn_cdf` per cell.
  Construction-bench medians (n=900, reps=5): lazy ULS p16/c5 423→14 ms,
  lazy DWLS p16/c5 298→26 ms, legacy full stats p16/c5 376→68 ms.
  Third pass (2026-06-12, same day) closed the listed S items:
  - `bvn_cdf` is now the Genz (2004) refinement of Drezner–Wesolowsky (6/12/20
    Gauss–Legendre nodes by |rho| plus the complementary high-|rho| expansion,
    ~5e-16 absolute accuracy, unit-pinned against the asin closed form and a
    Simpson reference in `ordinal_test.cpp`); all-ordinal construction mins
    moved p16/c5 lazy ULS 11.4→4.1 ms, lazy DWLS 19.2→13.2 ms, p4/c3 lazy ULS
    1.0→0.24 ms. Lavaan parity suite unchanged.
  - `shared_ordinal_casewise_psi` is cell-cached (scaled per-cell scores
    scattered to rows via a transposed accumulator), and the shared-robust FD
    gradient and bread FD columns are restricted to the pairs touching the
    perturbed coordinate (`shared_ordinal_touched_pairs`) — exact, since
    untouched pairs difference to zero.
  - The shared-robust threshold+rho refinement and the joint polyserial DPD
    fit now delegate to the vendored NLopt L-BFGS (encoded coordinates are
    unconstrained) instead of hand-rolled steepest descent + Armijo; +inf is
    the barrier for invalid probes, and a hard optimizer failure keeps the ML
    starting values like the old bail-out. The remaining hand-rolled searches
    (1-D golden-section/bisection/Newton, Fisher scoring, EM) were audited
    2026-06-12 and are appropriate as-is.
  - `benchmarks/mixed_ordinal_construction_bench.cpp` covers the mixed
    workspace/stats paths (half-ordinal half-continuous designs). It shows
    mixed construction is dominated by the polyserial/Pearson branches:
    p16/c5 lazy ULS is ~21 ms vs ~4 ms all-ordinal at the same design.
  Remaining headroom:
  - **Done 2026-06-15 — closes this subsection.** The named target was the
    mixed polyserial casewise-influence pass. `polyserial_pair_scores` built the
    rho score column by a 3-point finite difference (three full conditional
    probability evaluations per case ≈ 6 `erfc`) plus an O(K) threshold-density
    sweep. It now uses the analytic boundary derivative
    `d log P(c|u)/drho = (top.d1 − bot.d1)/p`, reusing the two boundary normal
    densities the threshold columns already compute: one probability evaluation
    (2 `erfc`) plus at most two `normal_pdf` per case, O(1) in the category
    count. The rho score is now exact instead of a finite difference (the column
    shifts by ~1e-10 toward lavaan's analytic value); the threshold, mu, and var
    columns are bit-identical (`src/data/pairwise_mixed.cpp`). All 828 dev tests
    pass, including the mixed-DWLS NACOV/score fixtures and the bfi
    mixed-ordinal parity block. This roughly thirds the dominant `erfc` count of
    the score pass; on the score-assembly construction paths (legacy mixed
    stats, lazy DWLS/WLS workspace) wall time trends down ~15–30% at p≥12
    designs, though the i7-1355U thermal noise floor (visible as ±20–35% swings
    on the identical-code ULS control) is wide enough that the deterministic
    operation-count reduction is the solid result. The fit-only ULS path (rho
    ML search) is unchanged by design.

    No exact cell collapse exists for polyserial. The all-ordinal win came from
    both margins being discrete (a `(K_i+1)×(K_j+1)` count table where many
    cases share one cell → one set of transcendentals); the polyserial
    continuous margin `u_i` is distinct per case, so cases cannot collapse to
    cells and the rho ML search is transcendental-bound and already minimal.
    Binning `u` was considered and rejected: it would perturb the
    polyserial/polychoric estimate and break lavaan parity.

## Two-level (multilevel) SEM

V1 landed 2026-06: two-level normal-theory ML for random-intercept models over
a SHARED observed variable set, single group, complete data, no constraints,
observed + analytic-expected SE, lavaan-parity. The `level:` block header is a
real `(group, level)` axis. Entry points: `estimate::twolevel::fit_ml_twolevel`,
`api::twolevel_ml()` + `api::data_from_cluster()`, R `fit_twolevel`. Gated by
`tests/golden/twolevel_golden_test.cpp` against
`lavaan::sem(model, data, cluster=)`; fixtures from
`tests/tools/regen_oracle_twolevel.R`, with an Mplus 9.1 Demo cross-check for
unbalanced cells via `tests/tools/regen_oracle_twolevel_mplus.R`.

Remaining work:

- **L (in progress).** Multi-group two-level: replicate the `(group, level)`
  block axis across groups, with cross-group equality via shared labels as in
  the single-level multi-group path.
- **L (deferred).** Between-only and within-only observed variables: the v1
  shared-observed-set restriction assumes every observed variable decomposes
  into a within and a between part. Lift it to support level-2-only covariates
  and within-only variables (lavaan's general `%WITHIN%` / `%BETWEEN%` blocks).
- **M (deferred).** Constraints under two-level: `fit_ml_twolevel` currently
  rejects equality/inequality constraints; wire the linear-reduced /
  Jacobian-projected constraint machinery through the two-level fit.
- **XL.** Categorical / robust two-level: ordinal two-level estimators and
  robust (sandwich / scaled) two-level test statistics.
- **XL.** 3+ levels and random slopes: out of the current single-axis design;
  needs the `rv(...)` random-slope modifier (currently parser-rejected) and a
  multi-level block axis.

## Misspecification-robust SE for the moment-quadratic family (frontier)

- **Reduced-bias estimation (RBM) frontier.** V1 landed 2026-06 for raw-data
  continuous ML and FIML. The 2026-06-25 extension adds the moment-quadratic
  family: continuous ULS/GLS/WLS through `continuous_ls_rbm_parts()`, ordinal
  and mixed ordinal through `ordinal_rbm_parts()` /
  `mixed_ordinal_rbm_parts()`, and ML2S NT/DWLS/ADF/DLS through
  `two_stage_rbm_parts()`. All use the same reduced-space trace algebra as ML
  (`K' J K`, `K' E K`) and the R frontier dispatcher now routes existing fit
  objects through `magmaan_core$frontier_rbm()`. Remaining work:
  - **Done 2026-06-25 — validation breadth (FD checks for the adjusted
    estimating equations).** `tests/unit/rbm_fd_test.cpp` independently
    finite-difference-validates the RBM *driver's* penalty/correction assembly
    (the ingredients `j`/`e` are already FD-gated transitively via
    `robust_weighted_moment_ij` and the `*_ij` / Γ-influence gates). Three
    checks: **A** explicit one-step reconstruction (independently central-FD the
    penalty `P(θ)=−½tr(j⁻¹e)`, solve `j_α·δ=∇P`, assert `correction≈δ` and
    `adjustment≈∇P` — guards sign, the −½, the K-reduction, the solve), run on
    **every** family (continuous ULS/GLS/WLS, ML, FIML complete+missing, ordinal
    DWLS/WLS/ULS, mixed DWLS, ML2S Nt/Dwls/Adf); **B** explicit↔implicit
    agreement to `O(‖corr‖²)` (confirmed second-order: the gap shrinks faster
    than `‖corr‖²` as N grows); **C** implicit stationarity of
    `M(θ)=base.f(θ)+trace(θ)/(2N)` (the adjusted estimating equation verbatim).
    B and C run on the cheap-implicit families (continuous/ML/FIML): the implicit
    RBM solve is an FD-gradient-over-parts optimization that rebuilds the
    IF(Ŵ)-carrying parts ~2q times per iterate, so the estimated-weight families
    (ordinal/mixed/ML2S, especially missing-data ML2S with its per-case Stage-1
    Γ-influence) run Check A only. `trace(θ)` is read directly from the public
    `*_rbm_parts` (weighted families) or via the explicit base-swap (ML/FIML);
    `base.f(θ)` is rebuilt from each family's public objective constructor.
    13 cases, all green.
  - **M — magmaan-owned paper rerun.** Experiment
    `38-jamil-rosseel-2026-rbm-sem` now reproduces the SEM bias-reduction
    paper's main two-factor and growth-curve examples from the authors' OSF
    result objects. Remaining work is the independent magmaan rerun: implement
    the bounded-estimation choices needed for the paper design, rerun the
    normal/non-normal cells from generated data, and compare magmaan ML/eRBM/iRBM
    to the OSF summaries.

- **Done 2026-06-19.** Observed-Hessian ("robust" regime) bread for the
  moment-quadratic estimators (ordinal DWLS/WLSMV/ULSMV, mixed/polyserial,
  continuous ULS/GLS/WLS), which previously had only the Gauss-Newton/expected
  bread. The initial version used `observed_moment_bread_fd`
  (central-difference of the per-unit moment-LS gradient, reduced via `K`) as an
  optional `bread_override` into `robust_weighted_moments`; the meat (NACOV),
  df, and chi-square are unchanged. Selector `bread = "expected" | "observed"` (reusing
  `robust::Information`) threaded through `robust_ordinal` / `robust_mixed_ordinal`
  / `robust_continuous_ls`, surfaced in R as `vcov(fit, regime = "model" |
  "robust")`, which feeds `standardized(fit, vcov)`. Continuous ML already had
  the observed bread; FIML's observed-information model vcov and MLR sandwich
  now use the same `vcov(fit, regime=)` keyword. Default stays `expected`
  (lavaan parity) where an expected-bread estimator exists.
  Validation: experiment 35 (`35-misspec-robust-se`) -- under the correct model
  the two breads coincide and both match the empirical sampling SD. Focal is a
  free loading on a cross-loading-distorted factor (raw + std.all), continuous
  ML, MCAR FIML, and ordinal DWLS. Under the omitted cross-loading the
  expected-bread/model SE
  underestimates by about a third (std loading 0.69-0.78 of truth) and the
  observed bread recovers it -- essentially completely for continuous ML
  (std 1.01) and substantially for ordinal DWLS (std 0.85, the new bread), with
  FIML now included as the missing-data continuous cell. Note:
  `docs/research/notes/misspec_observed_bread.tex`. This is `frontier`, not core
  parity: the correct-model cell is the validation oracle (no mainstream
  *software* -- lavaan/Mplus -- exposes this for the ordinal/WLS family).
  This observed-bread SE *is* Lai & Simoes (2023, SEM 30:5, "Reflecting on the
  'Robust' Standard Errors..."): their new SE (Eq 34-36, `Pi_DWLS`, bread
  `H = 2 D' Gd^-1 D - 2[I (x) eps' Gd^-1] Psi`); their "robust SE" (Eq 17, Muthen
  1997 = lavaan default) is our expected bread. The initial implementation
  computed `H` by FD; the continuous, all-ordinal, and mixed ordinal production
  observed breads now use closed-form moment Hessians. Their findings match exp
  35 (robust SE biased 15-40% under misspec, not improving with n; new SE
  consistent; ULS closes better than DWLS).
  The DWLS finite-sample residual is the data-dependent weight `Wd = Gd^-1`: both
  their new SE and ours treat the Stage-2 weight as fixed, so neither captures its
  higher-order variability (ULS, W=I, has none -> closes cleanly); a
  case-resampling bootstrap recovers that term.
  Follow-up landed 2026-06-19: complete all-ordinal ULS/DWLS/WLS now has
  `robust_ordinal_ij`, an explicit infinitesimal-jackknife covariance that carries
  the estimated diagonal-weight and full dense-weight terms. The data-direct
  Γ-diagonal and full-Γ channels are regression-tested against case-weight
  finite differences, and the dense diagonal extraction is pinned against the
  DWLS helper. This is the
  Hall-Inoue estimated-weight correction for misspecified moment-GMM, specialized
  to the moment-quadratic SEM stack. The reusable `robust_weighted_moment_ij`
  primitive is now the shared transport for observed-bread weighted-moment IJ
  covariance: callers provide per-case moment influence rows plus optional
  per-case estimated-weight corrections. Robust/experimental mixed stage-1
  variants, MCAR/pairwise-missing variants, and ML2S adapters remain out of
  scope.
  Derivation and implementation grid:
  `docs/research/notes/weighted_moment_ij_grid.tex`.
  Open follow-ups:
  - **Hall-Inoue estimated-weight IJ grid** (`weighted_moment_ij_grid.tex`).
    Verification gates for every adapter: fixed-weight reduction to
    `robust_weighted_moments`; derivative checks for each new weight-influence
    channel; deterministic resampling-jackknife fixture; misspecification
    simulation with ULS/fixed-weight negative controls. Slices: complete
    continuous LS fixed-weight/ULS reduction **landed 2026-06-19** as
    `robust_continuous_ls_fixed_weight_ij`; sample-built continuous GLS
    normal-theory-weight correction **landed 2026-06-19** as
    `robust_continuous_ls_gls_ij` (this is not the ML/FIML robust-score NT
    path); complete-data continuous WLS/ADF empirical-weight correction
    **landed 2026-06-19** as `robust_continuous_ls_wls_ij`, with
    covariance-only and meanstructure finite-difference contamination gates.
    Complete-data continuous DWLS empirical diagonal-weight correction
    **landed 2026-06-19** as `robust_continuous_ls_dwls_ij`, with the same
    covariance-only and meanstructure contamination gates on
    `diag(Gamma)^{-1}`. Complete-data continuous DLS fixed-mixing correction
    **landed 2026-06-19** as `robust_continuous_ls_dls_ij`, with
    covariance-only and meanstructure contamination gates on
    `((1-a) Gamma_NT + a Gamma_ADF)^{-1}`. EB-DLS scalar-selection uncertainty
    is a separate future scalar-IF term, not part of the fixed-`a` adapter.
    Complete all-ordinal full WLS **landed 2026-06-19** in
    `robust_ordinal_ij`, using dense `IF(Gamma)` from integer data plus the
    finite-difference stage-1 kappa channel; tests gate the full case-weight
    derivative and diagonal extraction against the DWLS path.
    Complete mixed ordinal/polyserial fixed-weight ULS **landed 2026-06-19** as
    `robust_mixed_ordinal_ij`, with `MixedOrdinalStats::moment_influence` rows
    reconstructed from the mixed moment builder and an exact reduction test
    against the observed-bread fixed-weight sandwich.
    Complete mixed ordinal/polyserial DWLS diagonal estimated-weight correction
    **landed 2026-06-19** in `robust_mixed_ordinal_ij`, using ordinary
    complete-data ML/polyserial raw mixed blocks, mixed moment influence rows,
    data-direct diagonal `IF(Gamma)`, and finite-difference
    `d diag(Gamma) / d kappa` in the mixed moment order.
    Complete mixed ordinal/polyserial full WLS estimated-weight correction
    **landed 2026-06-19** in `robust_mixed_ordinal_ij`, using dense mixed
    data-direct `IF(Gamma)` plus finite-difference `d Gamma / d kappa`; tests
    gate the full case-weight derivative and diagonal extraction against the
    DWLS path.
    ML2S observed-bread Stage-2 regime **landed 2026-06-20** as
    `TwoStageBread::Observed` on `two_stage_em_ml_inference`, with
    complete-data reductions against unstructured observed-bread robust SEM
    for `TwoStageWeight::Nt` and observed-bread `robust_continuous_ls` for
    `TwoStageWeight::Adf`.
    ML2S saturated-EM casewise moment influence primitive **landed
    2026-06-20** as `saturated_em_moment_influence`, with complete-data
    reduction to centered mean/covariance moment rows and missing-data
    `IF'IF == SaturatedMoments::acov` gates.
    ML2S raw complete-data observed-bread covariance for non-NT Stage-2 weights
    **landed 2026-06-20** by routing
    `TwoStageWeight::{Dwls,Adf,Dls}` through the continuous-LS IJ adapters;
    tests gate ADF/DWLS/DLS against
    `robust_continuous_ls_{wls,dwls,dls}_ij`. The scaled-test fields remain
    the fixed-weight Satorra-Bentler quantities.
    ML2S raw missing-data observed-bread covariance for non-NT Stage-2 weights
    **landed 2026-06-20** by adding the Stage-1 FIML sandwich-Gamma
    influence through a case-weight finite-difference over the saturated EM
    `(H,J,ACOV)` stack; the adapter covers `TwoStageWeight::{Dwls,Adf,Dls}`
    and tests that the IJ path changes `vcov`/`se` while preserving the
    fixed-weight scaled-test fields.
    Analytic replacement **landed 2026-06-23**: the per-case Stage-1
    sandwich-Γ influence `n·dΓ_b/dw_i` (Γ_b = n·H⁻¹JH⁻¹) is now closed-form,
    `dΓ/dw_i = H⁻¹JH⁻¹ + n(−H⁻¹dH H⁻¹JH⁻¹ + H⁻¹dJ H⁻¹ − sym)` with the
    direct (explicit case-weight) and θ-movement parts of `dH = I_i +
    (∂H/∂θ)Δθ_i`, `dJ = g_ig_iᵀ + (∂J/∂θ)Δθ_i`, where `Δθ_i = H⁻¹g_i` is the
    `saturated_em_moment_influence` row. The J-movement collapses to
    `¼(dSᵀS + SᵀdS)` (directional deviance score `dS`, no per-case info
    matrix); the H-movement is the per-pattern directional third derivative of
    the analytic saturated Hessian (`fiml_saturated_hessian_directional_block`,
    differentiating `A, Zsum, M` and reusing `basis_trace_xy`). This is the
    "score-product / observed-information / eta-movement" decomposition the
    remaining-slice flagged; it removes the per-case EM re-fits (was O(n) EM
    solves per block) and the ε step. `two_stage_saturated_gamma_influence(...,
    GammaInfluenceRegime::{Analytic,FiniteDifference})` exposes both; the FD is
    retained as the validation oracle. Gated in `fiml_test.cpp` (analytic == FD
    to ~1e-5 on complete and missing data, symmetric) plus the existing
    missing-data DWLS/ADF/DLS IJ vcov tests now run through the analytic path.
    Continuous-LS analytic observed bread **landed 2026-06-20** by replacing
    the production finite-difference Hessian with the closed-form
    `J' W J + residual-curvature` contraction from
    `weighted_moment_ij_grid.tex`; tests keep the finite-difference bread as
    the validation oracle for fixed-weight, GLS, WLS, DWLS, and DLS IJ paths.
    All-ordinal and mixed ordinal/polyserial analytic observed bread **landed
    2026-06-20** by adding the same `Delta' W Delta + residual-curvature`
    contraction for threshold, correlation, variance, mean, and polyserial
    association moments, including theta/released-scale standardization terms.
    All-ordinal MCAR/pairwise-overlap ULS IJ **landed 2026-06-20** by
    materializing case-aligned sparse moment-influence rows in
    `ordinal_stats_from_observed_integer_data(..., Overlap)`; tests gate
    complete-data reduction, `G'G/N == NACOV`, and observed-bread ULS reduction
    under deterministic MCAR.
    All-ordinal MCAR/pairwise-overlap DWLS/WLS IJ **landed 2026-06-20** by
    adding support-aware observed Gamma data-influence and Jacobian helpers;
    tests gate missing-pattern case-weight finite differences, complete-data
    reduction, and deterministic-MCAR DWLS/WLS fit-level IJ execution.
    Mixed ordinal/polyserial MCAR/pairwise-overlap ULS IJ **landed
    2026-06-20** by adding
    `mixed_ordinal_stats_from_observed_data`, which materializes support-aligned
    mixed moment influence rows over thresholds, continuous means/variances,
    polychorics, polyserial covariances, and Pearson covariances; tests gate
    complete-data reduction, `G'G/N == NACOV`, and observed-bread ULS reduction
    under deterministic MCAR.
    Mixed ordinal/polyserial MCAR/pairwise-overlap DWLS/WLS IJ **landed
    2026-06-20** by adding support-aware observed mixed Gamma
    data-influence and finite-difference `d Gamma / d kappa` helpers; observed
    mixed stats retain NaN-coded raw blocks so `robust_mixed_ordinal_ij` can
    route DWLS and dense WLS through the MCAR estimated-weight correction.
    Tests gate complete-data reduction of the observed helpers, diagonal/full
    consistency, `IF(Gamma)` centering at fixed kappa, and deterministic-MCAR
    DWLS/WLS fit-level IJ execution.
    Follow-up landed 2026-06-23: observed mixed stats now also materialize
    support-aligned `gamma_diag_influence` / `gamma_full_influence` rows, and
    `robust_mixed_ordinal_ij` plus `mixed_ordinal_dwls_profile_rmsea` consume
    those rows directly, falling back to raw-data reconstruction only for older
    complete-data stats. This makes missing-data mixed IJ/profile inference
    explicit-data-object driven like the all-ordinal overlap path.
    Basic hybrid mixed observed-data stats **landed 2026-06-23** as
    `estimate::fiml::mixed_ordinal_stats_hybrid_fiml_from_observed_data`:
    ordinal thresholds/polychorics and polyserials remain observed-pairwise,
    while continuous means/covariances and their influence rows come from
    saturated continuous FIML. R exposes this through
    `magmaan_core$data_mixed_ordinal_stats_hybrid_fiml_from_raw/_from_df`,
    with observed-pairwise companions and `robust_mixed_ordinal_ij`. Gated by
    C++ MCAR smoke tests and an R smoke. MCAR/MAR efficiency validated
    2026-06-23 by `experiments/37-mixed-fiml-pairwise-efficiency`: the hybrid
    leaves the ordinal block bit-for-bit identical, modestly reduces continuous
    sampling variance under MCAR (grows with the missing rate), and removes the
    pairwise continuous-moment bias under MAR (worst continuous-parameter |bias|
    ~0.10 pairwise vs ~0.01 hybrid at 35% missing). Remaining validation is
    finite-sample calibration and stress tests for singular full WLS Gamma.
    Remaining
    slices: the default `TwoStageWeight::Nt` path remains ordinary normal-theory
    ML robust-score inference, while the moment-quadratic GLS IJ correction is
    covered by complete continuous LS. (Robust/experimental mixed stage-1
    variants under missing data — WMA/DPD/Huber polyserial — are cut to
    [speculative.md](speculative.md): blocked on a recipe choice, no consumer.)
  - **Done 2026-06-20.** Analytic moment-Hessian remainder: complete continuous
    LS, all-ordinal, and mixed ordinal/polyserial observed breads now have the
    closed-form `J' W J + residual-curvature` contraction; FD remains the
    diagnostic validation helper rather than the production path.
  - **Done 2026-06-22.** First fixed-misspecification profile-RMSEA / profile
    LRT slices: `weighted_moment_profile_rmsea` builds the dense profile Hessian
    `Q = W - W D B^{-1} D' W` from the observed-Hessian bread and reports the
    signed `tr(QΓ)` RMSEA centering, positive `QΓ` mixture-tail spectrum,
    negative count, rank, nominal df, and actual positive `spectrum_size`;
    `continuous_ls_profile_rmsea` wires this for complete
    continuous ULS/GLS/WLS/DWLS-style moment-quadratic fits with Γ from raw data
    or caller blocks. `weighted_moment_profile_lrt` and
    `continuous_ls_profile_lrt` now compare nested dense profile Hessians in a
    common first-stage moment space and report both nominal `df_diff` and actual
    positive `spectrum_size`. `weighted_moment_profile_rmsea_two_metric` and
    the complete-data ML wrappers `ml_profile_rmsea` / `ml_profile_lrt` now
    implement the basic dense two-metric profile Hessian
    `Q = V0 - W* D B^{-1} D' W*` with raw-data or caller Γ. This is the basic
    research surface only.
    - **Done 2026-06-22.** Estimated-weight (diagonal DWLS) profile-Hessian
      core primitive `weighted_moment_profile_rmsea_estimated_weight` (+ the
      `WeightedEstimatedWeightProfileBlock` ingredient struct). It assembles the
      value-function Hessian over the *extended* first-stage vector `x = (u, γ)`,
      `Q = φ_xx - φ_xθ B^{-1} φ_θx = [[W,R],[R,S]] - [[WD],[RD]] B^{-1} [D'W,D'R]`
      with `W=diag(1/γ)`, `R=diag(r/γ²)`, `S=diag(r²/γ³)`, routing the extended
      block (`jacobian=[D;D]`, `V0=[[W,R],[R,S]]`, `W*=blkdiag(W,R)`,
      `gamma=Γ_x`) through the existing two-metric core and restating `df` to the
      classical u-moment count `Σ m_b − n_alpha`. At `r=0` the γ channel is
      dormant and `Q` collapses to the fixed-weight `W − W D B^{-1} D' W`; under
      fixed misspecification the γ block reshapes the law (matches the
      `docs/research/notes/ordinal_dwls_profile_*` prototype). `weighted_moment_profile_lrt`
      already consumes the nested pair unchanged. Gated by FD-vs-analytic,
      `r=0` reduction, and nested-LRT cases in `weighted_inference_test.cpp`.
    - **Done 2026-06-22 (rank accounting).** The structured inertia bookkeeping
      for the profile spectrum is characterized in
      `docs/research/notes/profile_hessian_rank_accounting.tex` (+ the numpy
      check `profile_hessian_rank_check.py`). Writing the profile Hessian as
      `Q = V_o^{1/2}(I − Ĝ B^{-1} Ĝ') V_o^{1/2}` with `Ĝ = V_o^{-1/2} W_p D` and
      the metric-gap form `Ã = D' W_p V_o^{-1} W_p D`, `col(Ĝ)` is invariant and
      `spec(I − Ĝ B^{-1} Ĝ') = {1}^{(m−p)} ∪ {1 − ν_j}`, `ν_j = eig(B^{-1} Ã)`.
      With full-rank `Γ`, Sylvester gives the same inertia for `Q`, the
      symmetric `Γ^{1/2}QΓ^{1/2}` form, and the inner matrix, so
      `spectrum_size = n_+(QΓ) = (m−p) + #{ν_j < 1}`, `n_−(QΓ) = #{ν_j > 1}`,
      `rank(QΓ) = (m−p) + #{ν_j ≠ 1}`; with singular `Γ`, the count is restricted
      to `range(Γ)`. These counts are `Γ`-free only conditional on fixed `Q`;
      `Q` can itself move with population moments / weights. Classical df
      survives iff `ν_j ≡ 1`, i.e.
      `B = Ã`, which needs `K = 0` (observed-bread/curvature channel off) **and**
      `W_p = V_o` (estimated-weight/metric channel off); both hold under correct
      spec and at `e=0`. ULS exposes only the curvature channel, ML/GLS both, and
      positive residual-curvature directions make `QΓ` indefinite (negative
      mixture weights). The note also flags that the positive-only `bias_trace`
      overcounts the signed `tr(QΓ)` of `higher_order_discrepancy_misspec.tex`
      when `n_−>0` (a convention, not a bug), and that the extra weights are
      `O(‖e‖)` so the integer rank is brittle (read `spectrum_size` as an
      effective rank at a floor). Numerically verified to machine precision (ULS
      inertia `(13,7,1)`, ML `(17,1,3)` at `df=9`; e→0 collapse to df).
    - **Done 2026-06-22 (signed trace fields).** Profile RMSEA/LRT result
      structs expose `trace_signed = tr(QΓ)`, `negative_trace_abs`,
      `negative_spectrum_size`, and `spectrum_rank` alongside the existing
      positive-tail `eigvals` / `bias_trace` / `spectrum_size`. RMSEA centering
      uses `trace_signed`; `rmsea_positive_trace` preserves the old positive-only
      comparator. LRT p-value summaries remain positive-tail approximations and
      warn when the contrast is indefinite.
    - **Done 2026-06-22 (small-pencil diagnostics).** Single-model profile
      RMSEA results now expose the parameter-space pencil
      `ν_j = eig(B^{-1} Ã)` when the stacked data metric `V_o` is positive
      definite, with `max|ν_j−1|` and predicted structural positive/negative/rank
      counts. Estimated-weight extended metrics can be singular, so they leave
      the pencil empty and continue through the dense `QΓ` path.
    - **Done 2026-06-22 (categorical DWLS wiring).**
      `estimate::ordinal_dwls_profile_rmsea` / `ordinal_dwls_profile_lrt`
      (`estimate/ordinal.{hpp,cpp}`) extract `(D, γ, r, B)` from an all-ordinal
      DWLS fit (`ordinal_moment_jacobian`, `NACOV.diagonal()`,
      `ordinal_block_residual`, `ordinal_observed_bread_analytic`) and build the
      joint NACOV `Γ_x` of `(u, γ)` as the cross-product of the stacked per-case
      influence rows `[g_i | IF_i(γ)]`, with `IF_i(γ) = IFG + G·(dΓ̂/dκ)'`
      reusing the same `ordinal_gamma_diag_data_influence` /
      `ordinal_gamma_diag_jacobian_fd` channels (and `*observed*` variants for
      pairwise-overlap MCAR) that `robust_ordinal_ij` uses. `fmin` is the
      standard DWLS `F`, so `chisq_standard` and `df` match `robust_ordinal`
      exactly; the γ channel is the first-stage estimated-weight direction the
      rank note flags. Single- and multi-group. Gated in `ordinal_test.cpp`
      (single-fit `Γ_x` uu-block == NACOV + df/chisq vs `robust_ordinal` + live
      γ channel; nested LRT df_diff / spectrum / mixture-p).
    - **Done 2026-06-22 (mixed categorical DWLS wiring).**
      `estimate::mixed_ordinal_dwls_profile_rmsea` /
      `mixed_ordinal_dwls_profile_lrt` now run the same estimated-weight profile
      construction for mixed continuous/ordinal DWLS. The wrapper prepares the
      mixed delta partable, builds `D` with `mixed_moment_jacobian`, takes
      `(γ, r)` from `MixedOrdinalStats::NACOV.diagonal()` and
      `mixed_model_moments() - stats.moments`, uses
      `mixed_observed_bread_analytic` for `B`, and assembles the extended
      `Γ_x` from `[g_i | IF_i(γ)]` using the complete or observed
      `mixed_gamma_diag_*` influence/Jacobian channels. Gated in
      `ordinal_test.cpp` by the live mixed γ block, uu-block == NACOV,
      standard χ²/df parity with `robust_mixed_ordinal`, raw-data requirement,
      and a nested mixed DWLS profile-LRT smoke.
      Follow-up 2026-06-23: the mixed profile accepts precomputed
      `MixedOrdinalStats::gamma_diag_influence` and only falls back to raw
      mixed data when those rows are absent; the observed-missing mixed builder
      now provides the precomputed rows.
    - **Done 2026-06-22 (R ordinal profile surface).**
      The R package now exposes the all-ordinal and mixed DWLS profile methods
      through `magmaan_core$ordinal_profile_rmsea` /
      `ordinal_profile_lrt` and
      `magmaan_core$mixed_ordinal_profile_rmsea` /
      `mixed_ordinal_profile_lrt`, backed by `infer_*` Rcpp wrappers. These
      take the same explicit `magmaan_ordinal_data` /
      `magmaan_mixed_ordinal_data` object used for fitting, enforce DWLS, and
      return the profile Hessian, Gamma, dense spectrum, signed/positive trace
      summaries, RMSEA or nested-LRT statistics, and any warnings. Verified by
      `just r-dev` and an R smoke over all-ordinal and mixed nested DWLS fits.
    - **Done 2026-06-25 (estimator-aware LRT modification indices).**
      The R `modification_indices_lrt()` observed-bread column `lrt_p_obs` now
      dispatches the profile-LRT on the anchor's estimator instead of hardcoding
      `ml_profile_lrt`: complete-data ML keeps `ml_profile_lrt`, all-ordinal DWLS
      uses `ordinal_profile_lrt` (sharing the anchor's `fit$ordinal_stats`), and
      mixed-ordinal DWLS uses `mixed_ordinal_profile_lrt` (`fit$mixed_ordinal_stats`)
      — making `modification_indices_lrt` the first R consumer of those bindings.
      `.lrt_refit` now propagates `ordered=` so an ordinal augmented model refits
      under DWLS (it previously errored "requires ordered variables"). Estimators
      without a wired profile-LRT path report `lrt_p_obs = NA`; the plain
      `lrt`/`lrt_p` Δχ² columns still report for any complete-data estimator.
      `.lrt_refit` now propagates `ordered=` so an ordinal augmented model refits
      under DWLS (it previously errored "requires ordered variables").
    - **Done 2026-06-25 (continuous ULS/GLS + FIML + ML2S LRT MI).** Extended the
      `lrt_p_obs` dispatch to the remaining estimator families via three new Rcpp
      bindings (in `fit.cpp`, co-located with the fit-local weight/pack helpers;
      the shared `profile_lrt_to_list` serializer was hoisted to `internal.hpp`):
      `infer_continuous_ls_profile_lrt` (continuous **ULS/GLS** — one shared weight
      built at the anchor θ, empirical Γ from the raw data),
      `infer_fiml_profile_lrt` (**FIML** — one shared missingness/saturated stage,
      per-model χ² from `fiml_extras`), and `infer_two_stage_nt_profile_lrt`
      (**ML2S** — Stage-1 EM moments reconstructed from `fit$stage1`). The FIML
      `stop()` is gone. For FIML/ML2S the plain `lrt`/`lrt_p` come from the
      profile-LRT's own `T_diff`/`p_unscaled` (the incomplete-data saturated term
      cancels in the difference), not `2N·fmin`; `robust_nested_lrt` is unusable
      here because its Satorra-2000 restriction map requires equal npar, not the
      parameter additions a modindex sweep makes. ML2S also needed an ML2S arm in
      `inference_modification_indices` for candidate enumeration (one-step ML score
      MI on the EM moments). Validated by GLS/ULS/FIML/ML2S arms in
      `r-package/examples/modification_indices_lrt.R` (each table value matches a
      direct `*_profile_lrt` call to ~1e-9).
    - **Done 2026-06-25 (continuous WLS LRT MI — family complete).** Continuous WLS
      is now wired too: `modification_indices_lrt()` gains a `weight=` argument (the
      fitting W, which is not retained on the fit), threaded through the candidate
      sweep, the augmented refit (`magmaan(..., W=)`), and the profile-LRT binding.
      A WLS fit without `weight=` errors clearly. Pure-R change (the
      `infer_continuous_ls_profile_lrt` binding already accepted `weight=`). The
      `lrt_p_obs` dispatch now covers **every** estimator magmaan fits: ML, ordinal
      DWLS, mixed-ordinal DWLS, continuous ULS/GLS/WLS, FIML, and ML2S. Validated by
      a WLS arm (ADF weight from `robust_empirical_gamma`) in the example.
    - **Done 2026-06-22 (mean-structure ML wiring).**
      `estimate::ml_profile_rmsea` / `ml_profile_lrt` now handle complete-data
      mean-structure ML in addition to covariance-only ML. The two-metric
      profile blocks use the stacked `[mean; vech(S)]` Jacobian, block-diagonal
      normal-theory metrics (`S^{-1}` or `Σ(θ̂)^{-1}` for means plus
      `Γ_NT^{-1}` for vech covariances), observed ML bread scaled per unit, and
      either caller-supplied stacked Γ or raw-data Γ from
      `data::empirical_gamma_with_means`. Gated in
      `weighted_inference_test.cpp` by raw-vs-supplied Γ parity for single-fit
      RMSEA and nested LRT.
    - **Done 2026-06-22 (ML2S-NT profile wiring).**
      `estimate::fiml::two_stage_nt_profile_rmsea` /
      `two_stage_nt_profile_lrt` now adapt the ML2S normal-theory Stage-2
      likelihood to the same two-metric profile-Hessian engine: saturated EM
      moments become the complete-data ML sample statistics, and Stage-1
      uncertainty is the block-diagonal stacked `[mean; vech(cov)]` Gamma from
      `two_stage_gamma_from_acov(sm, false)`. The public overloads cover
      precomputed `SaturatedMoments`, raw data, and raw data plus precomputed
      `FIMLPack`/`FIMLH1`; `fiml_test.cpp` gates raw/precomputed parity for
      single-model RMSEA and nested LRT.
    - **Done 2026-06-22 (FIML profile wiring).**
      `estimate::fiml::fiml_profile_rmsea` / `fiml_profile_lrt` now adapt
      raw-data FIML to the same two-metric profile-Hessian engine. The
      first-stage space is the EM saturated `[mean; vech(cov)]` coordinate:
      `V0 = H/n_b`, `W*` is the model-implied observed-pattern H1 information
      per block, Γ is the n-scaled saturated ACOV from
      `two_stage_gamma_from_acov(sm, false)`, and the observed FIML information
      divided by N supplies the profile bread. The API takes the already-known
      FIML LRT chi-square so `chisq_standard = chi2_lrt`; overloads cover raw,
      precomputed `FIMLPack`/`FIMLH1`, and precomputed `SaturatedMoments`.
      `fiml_test.cpp` gates raw/precomputed parity for single-model RMSEA and
      nested LRT.
    - **Done 2026-06-22 (finite-sample calibration experiment).**
      `experiments/36-ordinal-dwls-profile-lrt` (C++ Monte-Carlo, paper-sim).
      On the exact C4 binary pseudo-null it shows the standard fixed-weight
      DWLS nested test (Satorra-2000) is anti-conservative under fixed
      misspecification of the larger model — rising to 10.5% rejection at a
      nominal 5% (strong design, n=1500) as ε grows — while the estimated-weight
      profile law stays flat at ~4.2%. The empirical mean of the difference
      statistic tracks the full-profile trace, not the fixed-weight trace
      (2.63 vs 2.13 at the strongest setting). This is the nested-LRT companion
      to experiment 35 / `papers/estimated-weight-se`.
    - **Done 2026-06-22 (population cross-check).** Advisory check
      `tests/checks/ordinal_dwls_profile/` reproduces the symmetry-protected C4
      binary pseudo-null of `ordinal_dwls_profile_exploration.tex` in magmaan
      (tally→expand population dataset, congeneric vs tau-equivalent DWLS). Two
      verifications: (1) at `ε=0` the estimated-weight channel is dormant
      (`gap=0`, `tr_full==tr_fixed`) and the law reduces to the fixed-weight DWLS
      law (eigenvalues = the single SB scaling `c≈0.954`, *not* χ²₃ — DWLS≠WLS);
      (2) for `ε∈[0.02,0.10]` magmaan matches the prototype's population
      `tr_fixed` to ~1e-3 and the `ε=0.10` eigenvalues to ~1e-3, with the γ
      channel positive and monotone. Surfaced a prototype defect: its
      double-finite-differenced `tr_full` at `ε=0.06` is non-monotonic; magmaan's
      analytic-influence law is the corrected reference (as the note anticipated).
    - **Done 2026-06-22 (CRMR/SRMR estimated-weight inference).** The
      absolute-fit (single-model) companion. `estimate::ordinal_crmr` (point) +
      `OrdinalFitMeasures.crmr`; `estimate::ordinal_crmr_misspec_inference`
      (`OrdinalCrmrInference`) returns a bias-corrected point, exact-fit mixture
      p-value, and a confidence interval for CRMR/SRMR that propagates the
      estimated-weight γ channel. It is a criterion-at-estimator sandwich
      `Q_G = Dφᵀ V0 Dφ`, `V0` = correlation-selector, `Dφ` the full extended
      `(u,γ)` residual jacobian (u-channel `−(I−P)`; γ-channel the estimator
      weight-sensitivity), reusing the catml projector and the
      `ordinal_dwls_profile_rmsea` `Γ_x`; CI uses the normal-theory `g_Gᵀ Γ_x g_G`
      under misspec and the `weighted_chisq` mixture at the null. `estimated_weight`
      flag toggles the fixed-weight comparator; `srmr_denominator` switches the
      scaling. Single-group only this pass (multi-group coupling deferred). Gated
      by `ordinal_test.cpp` (denominator relation, point==fm.crmr, γ channel
      active, SRMR scaling) and the C4 MC harness
      `tests/checks/ordinal_crmr_inference/` (bias/variance/coverage vs
      Monte-Carlo + ε=0 dormancy). **Finding:** the CI is calibrated, but the γ
      channel is only ~2–3% of the CRMR variance under misspecification (vs the
      metric-dominating channel in RMSEA/the nested test) — CRMR's fixed metric
      makes it largely robust to weight estimation. Deferred: R bindings, lavaan
      `crmr` oracle/goldens, threshold-inclusive SRMR variants, the residual-map
      curvature term. (Multi-group landed in `b82b1f7`; the open piece is the
      pooling *convention*, below.)
    - **TODO: survey the multigroup CRMR/SRMR pooling convention** (short lit
      survey). Multi-group is implemented, but the CRMR point has two defensible
      forms that diverge for G>1: lavaan / `ordinal_crmr` use the size-weighted
      *mean of per-group roots* `Σ_b (n_b/N)·√(‖r_b‖²/k)` (oracle-pinned for the
      core fit measure), while `ordinal_crmr_misspec_inference` reports the literal
      RMS, the *root of the size-weighted mean* `√(Σ_b (n_b/N)‖r_b‖²/k)`, forced by
      the `T = N·G` statistic its CI is built on; by Jensen the former ≤ the latter,
      equal at G=1 (pinned in `ordinal_test.cpp`). Two questions: (1) mean-of-roots
      vs root-of-mean (root-of-mean is the literal "root mean square residual" and
      coheres with the CI; lavaan's is the nonstandard form); (2) the weighting
      itself: CRMR is not a discrepancy (identity metric, no `W`), so `π_g = n_g/N`
      is not derived; it enters legitimately only via the pseudo-true
      `θ_0 = argmin Σ_g π_g F_g`, whereas the explicit `π_g`-weighting of the pooled
      residuals is inherited convention plus the independence-additivity that keeps
      `Var(T) = Σ_g n_g(·)` a clean block sum. Unweighted stacked-RMS
      `√(Σ_b‖r_b‖²/Σ_b k_b)` is a third, equally-legit target. Survey what
      lavaan / Mplus / EQS and the SRMR literature actually do, then decide which
      the frontier index should report (current: weighted root-of-mean) and whether
      to expose a `crmr_lavaan` companion. From the 2026-06-22 audit discussion.
    - **Done 2026-06-22 (RMSEA estimated-weight CI).** The absolute-fit
      large-γ case. `estimate::ordinal_rmsea_misspec_inference`
      (`OrdinalRmseaInference`) returns the bias-corrected RMSEA (=
      `ordinal_dwls_profile_rmsea.rmsea`), an exact-fit mixture p-value, and a
      CI. RMSEA's criterion is the discrepancy `F = rᵀWr` itself, so by the
      envelope theorem the gradient is the bare profile score `g_F = (−2Wr,
      −r²/γ²)` (no projector, unlike CRMR); the CI is the normal-theory interval
      on `F₀` with `Var(N·F)=N·g_Fᵀ Γ_x g_F`, reusing the profile `Q`/`Γ_x`/bias/
      spectrum. `estimated_weight=false` is the fixed-weight comparator.
      Single-group only. Gated by `ordinal_test.cpp` (point==profile rmsea, CI
      ordering, γ active) and `tests/checks/ordinal_rmsea_inference/`
      (bias/variance/coverage vs Monte-Carlo). **Finding:** the γ channel is
      large (−0.12 to −0.80 of the variance) and **variance-reducing** (`r` and
      `γ` co-vary negatively), so the estimated-weight CI is calibrated (~nominal)
      while the fixed-weight CI is increasingly **conservative / too wide**
      (coverage 0.93→0.98 as ε grows) — accounting for the estimated weight
      *tightens* RMSEA's interval. Same theme as the rest of the program, opposite
      direction from the nested test (conservative, not anti-conservative).
      Deferred: R bindings, close-fit p-value, classical-noncentral comparator,
      multi-group.
    - **Done 2026-06-22 (CFI/TLI estimated-weight inference).** The program's
      first *incremental* (two-model) index. `estimate::ordinal_cfi_tli_misspec_-
      inference` (`OrdinalIncrementalFitInference`) returns misspecification-robust
      CFI and TLI with confidence intervals. The user model runs through
      `ordinal_dwls_profile_rmsea`; the independence baseline (free thresholds,
      zero correlations) is an analytic block (`D_b=[I;0]`, Gauss-Newton bread
      `W_tt` since σ_b is linear) routed through the *same*
      `weighted_moment_profile_rmsea_estimated_weight` with the *shared* `Γ_x`, so
      the joint law of `(T_u,T_b)` is one bilinear form: `Cov(T_u,T_b)=N gᵤᵀΓ_x g_b`
      with the envelope-score gradients `g=(−2Wd,−d²/γ²)`. CFI `=1−δ_u/δ_b` and
      TLI `=1−(Q̄_b/Q̄_u)·δ_u/δ_b` (noncentralities `δ=T−Q̄`, generalized df
      `Q̄=tr(QΓ_x)`) are a ratio delta-method; the TLI interval is the CFI
      interval scaled by `Q̄_b/Q̄_u`. `estimated_weight=false` zeros the γ channel.
      Single-group only. Derivation in
      `docs/research/notes/cfi_tli_misspec_inference.tex` (joint CLT + ratio
      delta-method + the baseline-dominated `Var(CFI)≈V_uu/δ_b²` simplification,
      which the test confirms). Gated by `ordinal_test.cpp` (T_u/T_b pass-through,
      baseline==`fm.baseline`, CFI∈[0,1] ordered CI, TLI=1−c·r and
      `Var(TLI)=c²Var(CFI)`, leading-order check, γ channel active vs fixed-weight
      comparator) and the C4 MC harness `tests/checks/ordinal_cfi_inference/`
      (point unbiasedness, coverage, γ-share, leading-order ratio). **Findings**
      (reps=2000, n=1000): CFI's CI is **calibrated** (coverage 0.90–0.91, var
      matches MC); CFI is **largely robust to weight estimation** (γ-share only
      ≈±3–8%, CRMR-like, *opposite* to RMSEA's −12…−80%); the baseline-dominated
      `Var(CFI)≈V_uu/δ_b²` simplification is accurate only at weak misfit (ratio
      0.99→0.84→0.35 as ε grows — the full bivariate form is load-bearing). TLI's
      point is unbiased and its CI calibrated at weak/moderate misfit but its
      analytic variance **over-states at strong misfit** (≈8× MC, over-covers)
      because `c=Q̄_b/Q̄_u` is ill-conditioned when `Q̄_u` is small — conservative,
      not anti-conservative; CFI is the index to trust for an interval. Deferred:
      a stabilized-`c` TLI variance, close-fit boundary calibration, multi-group.
    - **Done 2026-06-22 (consolidated R binding).** The estimated-weight
      fit-index family is now R-exposed through one surface.
      `estimate::ordinal_fit_measures_misspec_inference` (`OrdinalMisspecFitMeasures`)
      bundles RMSEA + CRMR/SRMR + CFI/TLI with their CIs (delegating to the
      per-index entry points; SRMR is the CRMR statistic rescaled to the vech
      denominator), gated by `ordinal_test.cpp`. The Rcpp entry
      `infer_ordinal_fit_measures_misspec` (`r-package/src/robust.cpp`,
      dotted-key list) and the `@export`ed wrapper `fit_measures_misspec(fit,
      ordinal_stats, estimated_weight, conf_level)` (`r-package/R/fmg.R`) expose
      it; `ordinal_stats` is passed explicitly (a fit does not retain `int_data`).
      Validated by `r-package/examples/fit_measures_misspec.R` (RMSEA == the
      standalone profile binding, SRMR==CRMR·√(ncorr/vech), ordered/in-range
      intervals, CFI near weight-invariant). The individual `ordinal_{rmsea,crmr,
      cfi_tli}_misspec_inference` stay C++-only. Deferred: per-index R bindings
      (covered by the bundle), the `.scaled`/`.robust` integration into the main
      `fit_measures()` table.
    - **Done 2026-06-22 (multi-group).** RMSEA, CRMR/SRMR, CFI and TLI
      estimated-weight inference now handle multi-group all-ordinal DWLS fits
      (the single-group guards are lifted). The criteria pool as `Σ_b n_b·crit_b`;
      the underlying profile already returns the block-diagonal `Γ_x` and pooled
      signed trace, so the new work is the stacked envelope gradient (`√(n_b/N)`
      per-group weights), a per-block γ-zeroing helper (`zero_extended_gamma_-
      channel`) for the fixed-weight comparator, the stacked CRMR sandwich
      (block-diagonal `Q_G`, statistic `Σ_b n_b G_b`), and a multi-group baseline
      (direct sum of per-group independence models; its bread is scaled `(n_b/N)·
      W_tt` so it pools correctly). The R `fit_measures_misspec` is group-agnostic
      and works unchanged. Note's "Multi-group" section derives the pooling.
      Gated by a duplicate-group reduction in `ordinal_test.cpp` (points
      invariant, statistic/df double, RMSEA interval tightens) plus a different-
      two-group run and the R example's two-group block. CRMR/SRMR require a
      common per-group `p` (shared denominator). The MC harness
      `tests/checks/ordinal_cfi_inference` gained a two-group cell (heterogeneous
      ε/size, configural): CFI coverage ≈0.92 (calibrated, point unbiased), TLI
      conservative — the pooling holds under genuine multi-group sampling, not
      just the exact-arithmetic reduction. A stabilized-`c` TLI variance remains
      the one outstanding quality gap.
    Remaining profile-Hessian fit/test work:
    using the small-pencil `max|ν_j−1|` diagnostic as an actual
    runtime gate to skip dense profile-curvature work when negligible (still use
    dense `QΓ` when actual positive mixture weights are needed); an *a-priori*
    analytic sign count of `#{ν_j < 1}` from model structure (the note settles
    the inertia identity but still reads the signs off an eigendecomposition).
  - **M — mixed continuous/ordinal misspec fit-measure bundle.** Mixed DWLS
    already has estimated-weight profile RMSEA/LRT over the extended
    `(u, gamma)` moment stack and observed/pairwise-missing stats now materialize
    the needed gamma-influence rows. First reportable slice **landed
    2026-06-23**: `mixed_ordinal_rmsea_misspec_inference` adds the same
    estimated-weight envelope-score CI and exact-fit mixture p-value as the
    all-ordinal RMSEA path, exposed through
    `magmaan_core$mixed_ordinal_rmsea_misspec`. Second slice **landed
    2026-06-23**: `mixed_ordinal_crmr_misspec_inference` adds CRMR/SRMR
    inference over the existing standardized mixed association residual
    convention (continuous-continuous and polyserial residuals divided by
    observed standard-deviation scales, with scale derivatives carried in the
    sandwich), exposed through `magmaan_core$mixed_ordinal_crmr_misspec`. Third
    slice **landed 2026-06-23**:
    `mixed_ordinal_cfi_tli_misspec_inference` adds incremental CFI/TLI inference
    using the mixed DWLS independence-baseline convention, exposed through
    `magmaan_core$mixed_ordinal_cfi_tli_misspec`. Consolidated surface **landed
    2026-06-23**: `mixed_ordinal_fit_measures_misspec_inference` bundles
    RMSEA + CRMR/SRMR + CFI/TLI into the same reportable schema as the
    all-ordinal table, exposed through
    `magmaan_core$mixed_ordinal_fit_measures_misspec` and the explicit exported R
    companion `fit_measures_misspec_mixed_ordinal(fit, mixed_stats, ...)`.
  - **FIML slice landed 2026-06-23.** R now exposes
    `magmaan_core$fiml_observed_vcov()` / `inference_fiml_observed_vcov()` and
    `vcov(fit, regime = "model" | "robust")` dispatches FIML fits to inverse
    observed information or the existing MLR sandwich respectively, without a
    separate raw-data argument. `r-package/examples/fiml.R` checks the
    model-vs-robust route on a non-saturated FIML fit, and
    experiment 35 includes an MCAR FIML cell under both correct and omitted
    cross-loading conditions.
  - A continuous-LS R `bread` binding for `vcov(fit, regime=)`; the
    single-source-of-truth refactor of `ordinal_block_residual`.

## Robust score / modification-index tests (frontier)

`inference::frontier::{modification_indices,score_tests}_robust` has landed for
complete-data ML (both breads), continuous ULS/GLS/WLS/DWLS (the
`gmm::Weight` overloads, expected bread, moment-metric sandwich), FIML/MLR
(`{modification_indices,score_tests}_fiml_robust`, observed bread + casewise
meat), and — as
`estimate::frontier::{modification_indices,score_tests}_{ordinal,mixed_ordinal}_robust`
— all-ordinal ULS/DWLS/WLS and mixed DWLS/WLS over the polychoric NACOV meat.
The continuous ML/LS, ordinal/mixed-ordinal, and FIML/MLR tiers are single- or
multi-group; a df>1 total-release (`score_tests_robust_joint`, mean-scaled +
imhof mixture) covers joint releases.
Validated by `tests/unit/score_robust_test.cpp`,
`tests/golden/score_robust_golden_test.cpp` fixtures 0006-0012, the R-internals
oracle from `tests/tools/regen_robust_score.R`, and the advisory
`tests/checks/robust_score/`. Remaining work:

- **Done 2026-06-13.** Multi-group robust MI/score tests now cover the
  continuous ML/LS tiers, the ordinal/mixed-ordinal tiers, and the FIML/MLR tier.
  The ordinal guard in `estimate::frontier` and the FIML guard in
  `inference::frontier` are removed after validating the same block-stacked
  nuisance projection: per-block `n_b/N` sandwich over polychoric NACOV with the
  two-group WLSMV golden 0012, exact WLS reductions for all-ordinal MI/score and
  mixed-ordinal MI/score, and a two-group FIML raw/pack parity regression over
  unequal raw block sizes and MCAR patterns.
- **R wrappers done 2026-06-14.** `modification_indices_robust()` /
  `score_tests_robust()` (R/`context.R`, exported) now wrap the LS and ordinal
  robust tiers: new Rcpp glue `inference_{modification_indices,score_tests}_robust`
  (`r-package/src/fit.cpp`) dispatches ordinal/mixed -> `estimate::frontier`
  (intrinsic NACOV-meat scaling) and continuous ML/ULS/GLS/WLS ->
  `inference::frontier`, reconstructing the LS weight in glue (ULS identity / GLS
  normal-theory / WLS explicit) and taking `bread`/`moments`/`cov` plus the raw
  fitting data for the empirical meat. The `bread/moments/cov` -> `InferenceSpec`
  parsers moved to the shared `r-package/src/internal.hpp`. Concrete consumer:
  `experiments/22-robust-score-modification-indices` (ordinal DWLS + continuous
  GLS misspecification demo, with the exact `c -> 1` reductions as correctness
  gates). Still open / only-when-needed: a C++ `api::frontier` entry point taking
  `api::Fit` for the LS tiers (the api `Fit`'s `EstimatorSpec.weight` is empty for
  GLS/DWLS, so it would have to recompute the weight; the ML `api::frontier`
  robust overloads already exist). The R path sidesteps this by reconstructing the
  weight in glue, so no consumer needs the C++ api entry point yet.
- **Done 2026-06-22 — estimated-weight ("complete-sandwich") robust MI.** The
  per-direction robust scaling `c = gᵀB1g / gᵀA1g` can now use the complete
  (Hall-Inoue infinitesimal-jackknife) meat — with the data-dependent-weight
  `IF(Ŵ)` term — instead of the fixed-weight `Δ'WΓ̂WΔ`, via a new
  `estimated_weight` flag. This is the per-parameter denominator lavaan never
  builds (it scales MI only by the global SB scalar `c`). New core primitives
  `estimate::weighted_param_space_sandwich_ij` and the IJ adapters
  `continuous_ls_param_space_sandwich_ij` (continuous GLS/WLS/DWLS/DLS, via the
  shared `build_continuous_ls_ij_blocks`) and `ordinal_param_space_sandwich_ij`
  (all-ordinal DWLS/WLS, via the shared `build_ordinal_ij_blocks` factored out of
  `robust_ordinal_ij` so the SE and MI paths build identical meat). Threaded
  through `inference::frontier::RobustScoreOptions.{estimated_weight,
  ij_weight_mode}` and the ordinal/mixed `estimate::frontier` entry points
  (mixed-ordinal errors: not yet wired); ML/FIML reject the flag (no estimated
  second-stage weight). R: `modification_indices_robust()` /
  `score_tests_robust()` gain `estimated_weight = FALSE`
  (`r-package/examples/estimated_weight_modindices.R`). The correction is
  leading-order only under misspecification (Hall-Inoue order-promotion; ULS has
  a fixed weight and is unaffected). Reduction anchors and a misspec-shift check
  in `tests/unit/score_robust_test.cpp`. Open follow-ups: mixed-ordinal IJ meat;
  an MC calibration cell in `tests/checks/robust_score/` showing the
  estimated-weight `mi.scaled` is better-calibrated than fixed-weight under DWLS
  misspecification; these new IJ sandwich primitives ride the pending
  `<domain>::frontier` retier below.

## Residual summary (lavResiduals parity)

- **Done 2026-06-23 — `lavResiduals(fit)$summary` table.** `measures::
  standardized_residuals` now fills a per-block `ResidualSummary` (`cov`, and
  `mean`/`total` under a mean structure), the analogue of
  `lavResiduals(fit)$summary` with the default `type = "cor.bentler"`: the SRMR
  family (SRMR, its asymptotic SE, the exact-fit z-test against 0) and the
  bias-corrected USRMR with a close-fit CI and a close-fit z-test against 0.05.
  The cor.bentler residual ACOV is the existing raw-metric `acov_res`
  (`Q·acov_obs·Qᵀ` in `fill_residual_z`) congruence-scaled by the sample SDs
  (Ogasawara 2001 eq. 13; verified `GG·acov_raw·GG == lavaan cor.bentler acov` to
  machine zero), so no new projection — just the RMS summary port of lavaan's
  `lav_residuals_summary_rms`. R: `lav_residuals(fit)` (exported) and
  `residuals(fit, standardized = TRUE)$summary` carry it; one C++ call.
  Gated full-precision vs lavaan 0.7.1.2691 in `tests/unit/residuals_test.cpp`
  (no-mean cov column; meanstructure cov/mean/total) and end-to-end vs live
  lavaan in `r-package/examples/lav_residuals_summary.R` (single- and
  multi-group, ~1e-6/1e-5). Note: only the default cor.bentler/SRMR type is
  built; the raw (RMR) and cor.bollen (CRMR, needs the `lav_deriv_cov2cor_b`
  Jacobian) summary types are not — add if a consumer appears. Ordinal residual
  summaries are a separate object (`OrdinalMisspecFitMeasures`), left untouched.

- **Done 2026-06-23 — estimated-weight ("complete-sandwich") standardized
  residuals (frontier).** `measures::frontier::standardized_residuals_estimated_
  weight` (new `measures::frontier` namespace) gives the residual SE/z and the
  `$summary` inference an estimated-weight residual ACOV for continuous LS fits
  (GLS/WLS/DWLS/DLS), instead of the NT projection + Γ_NT that lavaan and the NT
  path use. The residual influence function is `IF_r = M·Qᵀ − C·P` (residual-
  maker `Q = I − P·W`, `P = n_b·Δ_b·A⁻¹·Δ_bᵀ`, `A` the pooled LS bread, `C =
  weight_correction` the Hall-Inoue IF(Ŵ) rows); the ACOV is the empirical
  covariance of those rows. New core primitive `estimate::continuous_ls_residual_
  acov_ij` reuses the Stage-A `build_continuous_ls_ij_blocks`; `fill_residual_z`'s
  standardization was factored into a shared `apply_residual_standardization` the
  override path also calls. With a fixed weight (`C = 0`) it collapses to
  `Q·(Γ̂/n)·Qᵀ`, gated EXACTLY in `tests/unit/residuals_test.cpp` against a hand-
  built projector (1e-8); a DWLS weight-correction shift test isolates the `C`
  term. R: `residuals(fit, standardized = TRUE, estimated_weight = TRUE, data =)`
  / `lav_residuals(fit, estimated_weight = TRUE, data =)`
  (`r-package/examples/estimated_weight_residuals.R`); ML/FIML/ordinal reject the
  flag. Advisory bootstrap MC `tests/checks/residual_estimated_weight/` confirms
  the calibration: under t₆ heavy tails the estimated-weight SE / bootstrap-SD
  ratio ≈ 1.02 while the NT ratio ≈ 0.68 (NT under-states the residual
  variability by ~⅓). Open follow-ups: multi-group couples blocks only through
  the shared bread (per-block meat, the lavaan residual-ACOV convention);
  mixed-ordinal / categorical estimated-weight residuals are not wired; these
  primitives ride the pending `<domain>::frontier` retier below.

## Case-level influence diagnostics (semfindr parity)

The **exact leave-one-out engine and the approximate parameter-change engine
landed 2026-06-23** (`r-package/R/case_influence.R`: `case_rerun` /
`est_change_raw` / `est_change` (+`gcd`) / `fit_measures_change` /
`mahalanobis_rerun` exact; `est_change_raw_approx` / `est_change_approx`
one-step), single-group continuous ML/ULS/GLS, semfindr-format, validated by
`r-package/examples/case_influence_semfindr.R` and frozen fixtures under
`tests/fixtures/case_influence/`. The one-step engine rides a new core accessor
`inference::casewise_scores` (the N×n_free per-case score matrix; bound as
`infer_casewise_scores_fit`, == `lavaan::lavScores`); `information_cross_products`
is now its Gram. Remaining:

- **Upstream PR to semfindr (est_change_approx scaling).** semfindr 0.2.0
  `est_change_approx()` applies `N/(N-1)` twice to DFTHETAS and once (too few)
  inside `gcd_approx`; both are spurious O(1/N) factors with no first-principles
  basis (`est_change_raw_approx` is correct). magmaan uses the correct scaling
  and gates transitively up to the documented factors; see the
  `docs/validation/oracle-defects.md` entry. File a PR / issue against
  `sfcheung/semfindr` with the influence-function derivation. **S**
- **`fit_measures_change_approx` — NOT supported (by design).** semfindr's
  no-refit fit-measure change exists to dodge expensive refits; magmaan's refits
  are cheap summary-stat fits and are already reused from `case_rerun`, so the
  exact `fit_measures_change` covers this leg at machine precision for no extra
  cost. The approximation would be lossier (first-order `2·(ll1_i − ll0_i)`) and
  its CFI/TLI would inherit the `fixed.x` baseline gap. It is implementable in
  pure R (casewise MVN log-density at model/saturated/baseline moments — no core
  accessor needed); build only if a concrete large-N, approximate-only consumer
  ever appears.
- **Done 2026-06-23 — `fixed.x` baseline df gap.** `fit_measures()` reported
  `baseline.df` one too high per exogenous covariance vs lavaan (path / `pa_dat`:
  magmaan 6, lavaan 5), so CFI/TLI diverged on `fixed.x` models (chisq/rmsea were
  fine). The core correction already existed (`measures::baseline_chi2(pt, samp)`
  frees the exo (co)variances); the R `fit_measures()` was just calling the
  partable-unaware `infer_baseline(ss)`. Fixed by adding `infer_baseline_fit(fit)`
  (calls the partable-aware overload; a no-op without exo, so safe for all fits)
  and pointing `fit_measures()` at it. Path-model CFI/TLI now match lavaan to
  machine precision; the case-influence example and fixture use all four
  measures; full `just r-examples` clean.
- **Done 2026-06-23 — multiple-group `case_rerun` / `est_change`.** Pass the
  original `data` frame to `case_rerun(fit, data = …)`; it refits on `data[-i, ]`
  through the canonical `df_to_data` pipeline for any number of groups (case ids
  = data rows, as in semfindr). Columns are suffixed with the group *label*
  (magmaan and lavaan order groups differently — magmaan by factor level, lavaan
  by data appearance — so a label keeps comparisons tool-independent).
  `mahalanobis_rerun(fit, data)` computes per-group distances placed at the
  original rows. Gated vs semfindr in `case_influence_semfindr.R` (2-group HS
  CFA, `meanstructure = FALSE` to match): est_change_raw 7e-6, est_change 5e-5,
  fit_measures_change 2e-9, Mahalanobis exact. The one-step `*_approx` engine is
  still single-group (errors clearly on multigroup; the block-stacked scores
  would need original-row reordering — extend if needed).
- **Done 2026-06-23 — robust-regime `est_change`.** `est_change(rerun, se = )`
  takes `"standard"` (default, naive ML), `"robust.sem"` (Satorra-Bentler /
  expected-bread sandwich), or `"robust.huber.white"` (MLR / observed-bread
  sandwich); the robust regimes standardize and form gCD from the per-refit
  robust vcov (`robust_se_raw_fit(refit, refit$raw_data$X, bread)`, reusing the
  raw data every fit carries). Gated vs semfindr on robust-SE fits in
  `case_influence_semfindr.R` (~1e-5).
- **Done 2026-06-23 — misspecification-robust ("complete-sandwich") case
  influence (frontier).** The casewise dual of the estimated-weight SE: the
  one-step leave-one-out change carries the per-case data-dependent-weight term
  the naive (semfindr / Pek-MacCallum) influence drops by treating the estimator
  weight as fixed. New core accessor
  `estimate::continuous_ls_casewise_influence_ij` returns the `N×q` per-case
  influences `c_i = (1/N)·K·A⁻¹·(Δ_b K)ᵀ·v_i` (observed bread; `v_i = g_i·W + IF(Ŵ)`)
  plus the fixed-weight `influence_naive`; by construction `Σ_i c_i c_iᵀ`
  reproduces the `robust_continuous_ls_*_ij` vcov exactly (pinned at 1e-9 in
  `weighted_inference_test.cpp`). Reuses the shared `build_continuous_ls_ij_blocks`
  from the estimated-weight stream (no new IJ math). Bound as
  `infer_casewise_influence_ij_fit`; R surface
  `est_change_raw_approx(fit, type = "estimated.weight")` /
  `est_change_approx(..., type = "estimated.weight")` (continuous GLS/WLS/ULS),
  the raw output carrying `"naive"` / `"weight_diagnostic"` (`Δ'W'_d`) attributes.
  Self-validated against the EXACT GLS leave-one-out engine (which re-estimates
  the weight per drop) in `r-package/examples/case_influence_estimated_weight.R`:
  the complete one-step tracks the exact refit at RMSE ~3e-4 in both regimes,
  while the naive error grows with misfit (2.8e-3 at cfi 0.94 → 1.3e-2 at cfi
  0.85) and the complete error stays flat — the Hall-Inoue order promotion
  (`O_p(N⁻¹)` at the null → `O_p(N^{-1/2})` off it). Writeup in
  `papers/estimated-weight-se` (the casewise-dual section).
- **Done 2026-06-23 — full estimator/group coverage for the case-influence
  one-step engine.** Both regimes (`standard` and `estimated.weight`) are now
  **multiple-group**: the bindings already block-stack the per-group cases, so
  the R surface passes the full per-group raw list and labels rows `g{b}_{row}`
  with group-suffixed columns (the standard multigroup one-step tracks the exact
  ML refit at RMSE 3.6e-3, cor 0.93). The per-case row extraction was factored
  into the shared `estimate::casewise_influence_from_ij_blocks(blocks, K, bread)`
  (the per-case dual of `robust_weighted_moment_ij`), reused by both the
  continuous accessor and the new **ordinal** `estimate::ordinal_casewise_influence_ij`
  — the categorical (DWLS/WLSMV/ULSMV) headline cell, riding the existing
  `build_ordinal_ij_blocks` + `ordinal_observed_bread_analytic`, bound as
  `infer_ordinal_casewise_influence_ij_fit` and routed automatically when
  `fit$ordinal`. Continuous WLS/ADF (`SampleEmpiricalWls`) is also covered. Every
  cell self-checks `Σ_i c_i c_iᵀ ≡ robust_{continuous_ls,ordinal}_*_ij` vcov to
  1e-8/1e-9 (`weighted_inference_test.cpp`, `ordinal_test.cpp`), and the R
  example exercises multigroup + ordinal.
- **Done 2026-06-23 — two-stage (ML2S) case influence**, the missing-data member
  of the estimated-weight family. `estimate::fiml::two_stage_casewise_influence_ij`
  decomposes the ML2S complete sandwich: each case influences θ̂ through the
  Stage-1 saturated-moment influence (`saturated_em_moment_influence`) AND the
  Stage-2 data-dependent-weight term (`ml2s_weight_correction_block`). It factors
  the same IJ-block assembly the missing-data SE sandwich uses (`build_ml2s_ij_blocks`)
  and ends in the shared `casewise_influence_from_ij_blocks`; non-NT complete data
  routes through `continuous_ls_casewise_influence_ij` (the SE's complete-data
  route). The NT weight (lavaan robust.two.stage) treats the weight as fixed, so
  its correction is exactly zero (complete == naive); the non-NT Stage-2 weights
  (DWLS/ADF/DLS) carry the live term (HS missing+misspecified DWLS: naive-diff
  0.036). Bound `infer_ml2s_casewise_influence_ij_fit`, routed by the `^ML2S`
  estimator label (Stage-2 weight parsed from `ML2S_{DWLS,ADF,DLS,WLS}`).
  Self-check `Σ_i c_i c_iᵀ ≡` the observed-bread ML2S IJ vcov to 1e-8
  (`fiml_test.cpp`); R example exercises NT (zero) vs DWLS (live).
- **Remaining (case-influence one-step):** the per-drop exact leave-one-out
  *simulation figures* for the categorical/DWLS and ML2S cells (paper
  deliverables, not library code — `case_rerun` is ML/ULS/GLS; re-estimating
  polychorics / re-running the Stage-1 EM per drop is expensive; the GLS
  exact-LOO already validates the one-step semantics and the self-checks pin the
  math). Mixed ordinal-continuous case influence stays blocked on its
  estimated-weight SE (not yet wired). Ties to the model-free DOCR reference
  (`external/refs/case-influence/`).

## Local hardening and validation tooling

Local-first safety tooling for an AI-assisted repo. Design note:
[docs/validation/local_hardening.md](../validation/local_hardening.md). The test
ledger, risk map, regression-note convention, and JUnit/health recipes have
landed; remaining open items:

- **Done 2026-06-15 — coverage lane calibrated.** A full `just coverage` run
  (clang/llvm-cov 21, all nine `magmaan_test_*` binaries) was interpreted and
  written up in
  [docs/validation/local_hardening.md](../validation/local_hardening.md)
  ("Interpreting a coverage run"). Findings: (1) the `171 functions have
  mismatched data` warning is benign — header-defined inline/template functions
  carry different structural hashes across test TUs; the count scales with the
  number of combined binaries (0/19/171 for smoke/ordinal/all-nine) and is
  emitted at load time, so it cannot be suppressed via `--ignore-filename-regex`
  and must not be piped away; line/region coverage of out-of-line `src/*.cpp`
  (one consistent hash from the single static lib) is accurate, only the function
  denominator is mildly understated. (2) Object list (the nine targets) and
  ignore list (`/_deps/|/third_party/|/tests/|/usr/`) reviewed and correct;
  no changes. (3) Domain-level snapshot table added with dark-room reading —
  `api`/`compat` read artificially low because they are R-boundary-validated
  (lane is C++-test-only), while `robust`/`sim` low spots track known backlog
  gaps. Refresh the snapshot after major test/source changes; the ordering, not
  the absolute %, is the durable output.
- **M, audit.** Audit test tolerances so the suite is not "faked" by loose gates
  that pass a known divergence instead of pinning the right answer. Template
  already fixed: continuous GLS/WLS chisq was gated at `max(5e-2, 2·N·fmin·2e-3)`
  (≈1.2) in `lavaan_parity_golden_test` and `ls_golden_test`, absorbing the whole
  `(N−G)/N` multiplier gap; both now pin `magmaan·(N−G)/N == lavaan` to 5e-3.
  The ordinal goldens are now swept too (2026-06: same multiplier found under
  the 8e-2 ordinal chisq gates; rescaled and tightened to 5e-3, with the
  per-group `(n_g−1)/n_g` estimator-weighting consequence documented in
  numerical-conventions exception 4 and the test ledger). Follow-up audit:
  the real-data bfi ordinal block in `lavaan_parity_golden_test` had the same
  stale multiplier gate (and used relative tolerances); it now rescales to
  lavaan's statistic and pins χ²-family quantities to 5e-3, and finite lavaan
  fit-measure fields now fail if magmaan returns non-finite values, which
  surfaced and fixed the saturated-model TLI convention (`df_user = 0` → TLI =
  1). Soft-gap hygiene pass (2026-06): ptable tolerated divergences and the
  matrix-rep deferred set now have exact count checks, and the skipped
  Little/Newsom broad corpus check no longer aborts on Little's null mean
  placeholders; forcing it with `--no-skip` reaches the single documented
  `newsom/ex5_5b` backend failure. The remaining golden soft sets are now
  swept too: `fit_implied`, `fit_measures`, `fit_theta`, `inference`,
  `test_stats`, and `observed_inference` count-pin their skip/defer/
  needs-regen/no-oracle buckets, and the old silent `test_stats` `pe_z`
  pre-regen path is now recorded and pinned empty. The self-referential-fixture
  audit came back clean: all fixture families are anchored to lavaan,
  lavaan-internals, PearsonDS, SuppDists, rvinecopulib, robcat, or analytic
  references; `paper_corpus`'s misleading `export_magmaan.R` name still runs
  `lavaan::sem()`. Observed-inference currently has no no-oracle fixtures
  under the regenerated lavaan-backed corpus; that bucket is literal-count
  pinned at zero so future structural missing-oracle cases must be explicit.
- **M, later.** Layer CI on top only after the local commands are useful: run
  `test-quick` on PRs/pushes, sanitizer validation on main or a schedule, heavy
  parity/optional optimizer lanes less often, and coverage as an artifact before
  considering badges. Avoid coverage-percentage gates until the report has been
  calibrated by real maintenance work.

## Simulation primitives

The `magmaan::sim` surface (NORTA / IG / Vale-Maurelli-Fleishman / PLSIM /
t-copula / Archimedean copula / C-vine / elliptical generators, marginal
families, and observed-variable projection) is summarized in the roadmap; its
work queue and decision log live in [`simulation.md`](simulation.md). Keep this
section as a high-level index for cross-domain planning — including the Johnson
SL exposure decision, the special-functions dependency policy, NORTA calibration
hardening, group-specific projection/population metadata, and model-implied
simulation lowering — and put detailed generator, marginal, and fixture
decisions in the simulation backlog.

## API and R boundary

- **S/M.** Add or rename R wrappers only when the methods-developer workflow
  exposes a concrete gap in the staged API; the current `magmaan_core`,
  `magmaan_fit`, and post-fit wrapper surface is otherwise sufficient for the
  next R exploration pass.
- **S/M, experiment-motivated.** Decide whether the Deng-Chan reliability-
  difference test (`experiments/20-deng-chan-2017-alpha-omega`) earns a home in
  core. The experiment already runs against the current surface (ML fits plus
  `infer_gamma_nt`/`infer_empirical_gamma`; Cronbach's alpha falls out as the
  omega of a ULS tau-equivalent fit) and diagnoses a genuine non-regularity:
  because `omega ≥ alpha` with equality only at equal loadings, `omega − alpha`
  is a second-order (1/N) statistic with a weighted-chi-square null, so a naive
  Wald z under-rejects and the fix references `2n(omega − alpha)` against its
  Imhof tail. If promoted, the natural shape is a `measures::frontier`
  reliability module (alpha, omega, the joint `omega − alpha` SE, and its
  second-order Imhof calibration) plus a thin R wrapper. First slice landed
  2026-06-26: `measures::frontier::reliability` now provides covariance-only
  alpha, Guttman's lambda6, and Spearman-Guttman covariance omega with
  delta-method SEs, plus the exploratory R primitive
  `magmaan_core$measures_reliability_cov` and
  `experiments/41-reliability-lambda6`. The non-regular joint
  `omega - alpha` test and Imhof calibration remain unpromoted. Not required.
  Prior-art oracle for the Spearman-Guttman covariance omega is Hancock & An
  (2020) (closed-form single-factor omega; see
  [paper-eval](../research/paper-evals/2020-hancock-closed-form-omega.md)): their
  Spearman-1927 ratio-of-sums loading aggregation is more numerically stable than
  the average-of-ratios communality and is the parity target for exp 41. Their
  256-cell sim is the validation oracle. Open lane beyond Hancock & An (single
  factor only): the multi-factor / multi-group / weighted `ω_G(σ;w)` form and
  omega-hierarchical via a second-stage Schmid-Leiman centroid on `Φ_G` (k>=3),
  derived in the `guttman_cfa_asymptotics.tex` note.
- **L/XL.** Remaining C++ estimator work on the ordinal SNLLS / Gamma workspace
  track (the landed split is in the roadmap and
  [docs/design/ordinal-snlls-gamma-architecture.md](../design/ordinal-snlls-gamma-architecture.md);
  threshold-profiled general linear maps, joint multi-group threshold
  profiling/invariance, their lavaan oracle fixtures 0013/0014, exact mixed
  robust parity — the muthen1984-faithful mixed Gamma sandwich, gated at
  all-ordinal tightness — mixed theta SNLLS, and lazy mixed WLS (the
  workspace defers the O(m³) inverse to the cache ensure helpers) landed
  2026-06): only-when-needed R/API polish remains; reduced-Gamma
  robust-inference products moved to [speculative.md](speculative.md) with a
  size-triggered build-if. **`group.equal` ordinal measurement invariance landed
  2026-06-15 (theta).** The `BuildOptions::group_equal` / `group_partial` keyword
  ties the requested families across groups (synthetic shared labels → the same
  `compute_eq_groups` merge as explicit labels), and the Wu-Estabrook (2016)
  identification frees the group-2+ ordinal residual variances (the released
  latent-response *scale*, binary-vetoed) and indicator intercepts in
  `prepare_ordinal_*_partable`, with `build` forcing a mean structure so the
  intercept rows exist. The gated parameterization is **theta** (the standard for
  ordinal invariance): under delta lavaan's released `~*~` scale is unidentified
  (it stays pinned at 1 with a singular vcov), so lavaan-delta invariance is
  *not* gated; the explicit Mplus-style delta probe below covers the released
  branch. lavaan-gated by `cfa(..., parameterization="theta",
  group.equal=...)` fixtures 0017 (3-cat thresholds+loadings), 0018 (binary
  scale-veto), 0019 (thresholds-only), and 0020
  (thresholds+loadings+intercepts / scalar), df/chisq/theta_hat parity in the
  `ordinal invariance (group.equal) theta fits match lavaan` golden. The
  released-scale moment Jacobian is now shared between the fit and robust nested
  paths: theta subtracts the released μ and threads `J_mu`, while released delta
  differentiates `(τ−μ)δ_i` plus the implied association rows so explicit
  latent-mean and response-scale columns are not rank-dropped. At scalar,
  lavaan fixes the group-2+ indicator intercepts back to 0 and frees group-2+
  latent means; magmaan mirrors that in `prepare_ordinal_*_partable`. **The
  Satorra-2000 nested LRT ladder also landed 2026-06-18** (delta A-method,
  theta), using the same shared moment-Jacobian block for freed intercept and
  latent-mean columns. Gated by the `ordinal invariance nested LRT
  (satorra.2000 delta) matches lavaan` golden: configural→metric and
  thresholds→metric scaled Δχ² 3.025 / Δdf 3 / p 0.388, metric→scalar scaled
  Δχ² 3.765 / Δdf 3 / p 0.288 (scalar-only 1.5e-2 tolerance on the scaled
  statistic for the documented `(n_g−1)/n_g` LS-weight gap), and
  configural→thresholds recorded as a df=0 χ²-equivalence because lavaan cannot
  form a positive-df `lavTestLRT` there. The explicit Mplus-style delta scalar
  probe in `experiments/33-mplus-demo-wlsmv-difftest` also now passes: with
  overlap pairwise Gamma, magmaan's scaled-shifted restriction-map statistic is
  22.365850 on 22 df (p = 0.438242), matching Mplus Demo DIFFTEST 22.366000 on
  22 df (p = 0.438200). The remaining paper work is its own item below.
- **M. Ordinal `group.equal` R surface + paper (the C++ core is done, gated).**
  The keyword + Wu-Estabrook theta release + nested satorra.2000 LRT all landed
  and match lavaan in C++ (commits 2aae694 / be27d01 / 00f2373; see the ordinal
  SNLLS bullet above). The R surface (single-fit + nested) is now done and
  lavaan-gated, and `mplus_wlsmv_invariance()` now provides the ergonomic
  Mplus-style delta ladder over all-ordinal pairwise/listwise data plus mixed
  continuous/ordinal listwise data. It is fixture-gated against the Mplus Demo
  DIFFTEST scalar probe (`tests/fixtures/mplus_wlsmv_invariance`) and a small
  mixed deterministic regression; mixed pairwise missing remains deferred until
  mixed pairwise NACOV construction exists. Only the paper remains.
  - **R surface — LANDED 2026-06-15, lavaan-gated.** `model_spec()` /
    `magmaan(...)` take `group_equal` / `group_partial` (snake_case, lavaan
    family strings); Rcpp `lavaan_lavaanify` maps strings → `GroupEqual` and ties
    the families at build (`RcppExports` regenerated). The build round-trip drops
    `LatentStructure::group_equal` (a lavaan partable has no such column), so
    `lavaan_lavaanify` now stamps the resolved families as an integer-index
    `magmaan.group_equal` attribute and the three ordinal fit impls
    (`fit_{dwls,uls,wls}_ordinal_impl`) read it back (`group_equal_attr`) to
    re-apply the fit-time Wu-Estabrook release. Because the ordinal threshold
    rows are materialized in R `augment_ordinal_partable` (after the data-free
    build, so build can't tie them), augment now (a) ties equated thresholds via
    shared `.theq.` labels — the same shared-label path build uses for loadings,
    so `from_lavaan_partable` emits the cross-group `==` rows — and (b) leaves the
    group-2+ non-binary ordinal `~~` free (binary-vetoed) for the C++ release
    instead of pinning it. The scalar/intercept rung now suppresses the indicator
    intercept release and frees the group-2+ latent mean, matching lavaan's
    theta scalar convention. Gated by `r-package/examples/group_equal_ordinal.R`
    (0017/0019/0020 analogues, theta): npar, LS χ², released-row count, and every
    free estimate match `lavaan::cfa(group.equal=, parameterization="theta")`
    (est diff ≤ 2.4e-4; `~*~` excluded — a fixed theta row lavaan reports as a
    derived implied scale, magmaan keeps the nominal 1.0 and carries the scale on
    `~~`).
  - **Nested-ordinal FMG wrapper — LANDED 2026-06-15, lavaan-gated.** No new
    Rcpp export was needed: `infer_ordinal_lr_test_satorra2000` already returns
    the difference triple (`T_diff`, `df_diff`, `eigenvalues`), so
    `fmg_nested_ordinal(fit_H1, fit_H0, ordinal_stats, ...)` (`r-package/R/fmg.R`)
    is a thin R wrapper that runs `robust_nested_lrt(method="restriction_map")`
    and feeds that triple through the same `.fmg_result_rows_ordinal` /
    `infer_fmg_test` path as `fmg_tests_ordinal`. Its `A.method` defaults to
    `"delta"` (the configural→metric Wu-Estabrook pair is non-nested);
    `robust_nested_lrt()` keeps its `"exact"` default so nested complete-data /
    FIML pairs are unaffected. The threading gotcha was real and is fixed:
    `ctx_from_fit` → `from_lavaan_partable` drops `group_equal`, which the H0
    re-prep inside `lr_test_satorra2000_ordinal` needs (it re-preps both
    structures), so `ordinal_fit_result` now stamps `magmaan.group_equal` on
    `fit$partable` (`stamp_group_equal_attr`) and `ctx_from_fit` re-attaches it.
    Gated by `r-package/examples/group_equal_nested_ordinal.R`: configural→metric
    and metric→scalar rescaled satorra.2000 Δχ²/Δdf/p match
    `lavTestLRT(method="satorra.2000")` on `se="robust.sem"` fits (≤ 5e-3 for
    metric, ≤ 1.5e-2 scalar scaled statistic, reproducing the C++ nested
    golden), and the FMG `sb_ls` row reproduces `nestedTest()`'s `T_scaled`.
  - **Paper** (`papers/ordinal-fmg/`, private leaf): switch
    `harness-population.R::invariance_syntax` from loadings-only to
    `group.equal = c("thresholds","loadings")` under theta; add a
    `lavTestLRT(method="satorra.2000")` cross-check in `harness-oracle.R`; route
    the nested arm through `fmg_nested_ordinal`.
  - **Parameterization note (deliberate):** the lavaan-gated invariance path is
    **theta**, not delta. lavaan's *delta* threshold+loading release is
    structurally degenerate — the freed `~*~` scale stays pinned at 1 for any
    latent-scale difference, the vcov is singular, and chisq cannot absorb the
    difference — so magmaan does not gate lavaan-delta invariance. The released
    delta moment branch is tested through the explicit Mplus-style scalar probe
    above, not through lavaan's degenerate delta `group.equal` convention. The
    paper's arm should use `parameterization = "theta"`.
  - **Mixed-ordinal release not started** (all-ordinal only).
  - **Adjacent finding (continuous `group.equal`).** The same keyword now reaches
    continuous fits: `group.equal = "loadings"` ties at the npar level and the
    free estimates match lavaan (only magmaan's synthetic `.eqg` labels vs
    lavaan's `.pN.` differ cosmetically). But families that require a
    *compensating* release are not wired: `c("loadings","intercepts")` gives
    npar 60 vs lavaan's 63 — lavaan frees the group-2 latent means when
    indicator intercepts are equated, and `build` does not. This is a separate
    continuous strong-invariance feature (a `Means`/`Intercepts` build release,
    analogous to Wu-Estabrook), not part of the gated ordinal scope; promote to
    its own item if a continuous-invariance consumer appears.
- **M/L.** Optional h-weighted polyserial path: a polyserial-only h-weighted
  moment builder — continuous-ordinal h objective, casewise threshold/rho
  estimating functions, bread/influence/Gamma construction, and splicing into the
  mixed moment stack so `NACOV`/`W_dwls`/`W_wls` rebuild. The all-ordinal h-score
  variants already have the generic Gamma machinery; the missing piece is the
  mixed polyserial estimating-equation design.
- **FIML FMG follow-ups.** The landed FMG R wiring (single-model, FIML/missing,
  and the nested restriction-map route; see roadmap) leaves: multi-group FIML
  fits need explicit start/convergence care, and pairwise-data FMG remains
  deferred. Nonlinear equality tangent-space support for the single-model FIML
  UGamma spectrum and nested FIML restriction-map route landed 2026-06. The
  high-level `magmaan(estimator = "FIML")` path now auto-enables mean structure
  for syntax-backed models and rejects explicit `meanstructure = FALSE`.
  Saturated-EM reuse extended to the FIML nested path (2026-06):
  `lr_test_satorra2000/2001_fiml_from_data` take an optional `sm_precomputed` and
  `infer_fiml_lr_test_satorra2000` reuses a fit's `$stage1`, and
  `fit_ml2s(stage1=)` skips the rung-independent Stage-1 EM; this is what
  `experiments/25-fiml-invariance-fmg-power` uses to build one saturated EM per
  masked dataset and thread it through both estimators, all four ladder rungs,
  every FMG battery, and every nested test (kills the ~72s-at-p=30 FIML-nested
  rebuild and the 4x ML2S Stage-1 redundancy; bit-identical, verified). Deferred
  residual: `fit_fiml` still recomputes its cheap mu/Sigma-only `fiml_h1_moments`
  per rung; eliminating it needs a `fit_fiml` h1-injection arg + a FIMLH1-from-R
  reconstructor (the structured optimization dominates, so it is low priority).
- **Measurement-invariance nested run (exp 25) gaps.** Two items block the full
  Brace-Savalei-style invariance Type-I run; the overnight run uses the p=6
  one-factor model with the weak + strict nested steps only.
  (1) **metric->scalar ("strong") nested test is not nestable**: the scalar model
  ties intercepts AND frees the group-2 latent mean, so `npar(scalar) = 37 >
  npar(metric) = 36` and the exact Satorra-2000 route
  (`restriction_alpha_from_K`) rejects `K_H1.npar != K_H0.npar`. Needs a
  mean-structure-aware nested path (reference both models to the saturated model
  the way lavaan does), the same item the pEBA-nested paper reviewers raise.
  (2) **Brace-Savalei multi-factor populations** (two-factor, p in {8,16,30}) are
  not implemented; `build_population()` errors for p != 6. The p=30 cell is
  pEBA's home turf and the high-dimensional head-to-head; it is the build-out.
  Note: pEBA-on-the-difference IS already exposed (`fmg_nested()`, FIML + ML2S),
  so the spectrum nested battery is harvested now; only the two items above
  remain.
- **Done 2026-06-24 — structured-h1 FIML FMG ditched.**
  `estimate::FIMLH1Information::Structured` (the FIML FMG U·Γ-spectrum knob that
  evaluated the U-side information `V` at the model-implied moments `ξ(θ̂)`
  instead of the saturated `ξ̃`) is removed: the `FIMLH1Information` enum, the
  `h1_information` parameter on the `fiml_ugamma_spectrum` overloads, the
  `FIMLUGammaSpectrum::h1_information` field, the `infer_fiml_fmg_spectrum` Rcpp
  arg, and the R `fmg_tests(h1_information=)` / `fmg_pvalues()` argument. `V` is
  now always the saturated observed H1 information — the FMG-spectrum convention,
  PD by second-order optimality. Rationale: structured-h1 was *not*
  asymptotically advantageous (both consistent under H₀; Satorra-Bentler varies
  only `U⁰`, and Xia et al. 2016 via Savalei & Rosseel 2022 mildly prefer the
  saturated/unstructured information), and the model-implied curvature is not
  guaranteed PD off H₀ with nothing guarding the sign of the retained spectrum
  fed to the FMG/pEBA mixture (which assumes `λᵢ ≥ 0`). No new sim was needed:
  experiment 23 already carried both variants across its grid, so the
  keep-or-ditch evaluation was in hand. Two things survive the removal: the
  `fiml_structured_h1_information` helper stays (the ML2S Stage-2 `W*` uses it),
  and magmaan's MLR/Yuan-Bentler reproduction is the independent
  `fiml_robust_mlr` (`estimate_fiml_robust_mlr`), which never used this knob.
  Experiment 23's structured columns were dropped from the harness and the
  variant is marked legacy in its report; `examples/fmg.R` and the
  `fiml_ugamma_spectrum` C++ test no longer exercise it.
- **Ordinal/polychoric FMG (`papers/ordinal-fmg/` Paper 2).** Core gate **landed
  2026-06-13** (commit b8c6dcb): `fmg_tests_ordinal()` / `fmg_tests_mixed_ordinal()`
  apply the FMG eigenvalue-tail transforms to the `robust_ordinal()` /
  `robust_mixed_ordinal()` polychoric UGamma spectrum (`eigvals` + `chisq_standard`
  + `df`), single- and multi-group, `_ml`/`_ug` rejected, anchored by the ordinal
  C++ FMG test and `r-package/examples/fmg_ordinal.R` (single-group plus
  two-group all-ordinal/mixed SB parity); no new C++ production code (see
  roadmap). The nested-test gate **also landed 2026-06-13**:
  all-ordinal DWLS/WLS `nestedTest(..., data = ordinal_stats,
  method = "satorra.2000")` now builds the direct ordinal Satorra-2000 reduced
  spectrum for exact or delta restrictions, including two-group
  configural-vs-metric invariance, with lavaan WLSMV parity in
  `r-package/examples/nested_test_ordinal.R`. What Paper 2 still lacks, to
  write the paper:
  - **Scaffold + reframed design landed 2026-06-13.** The paper-local harness
    exists at `papers/ordinal-fmg/` (its own nested git repo, gitignored by the
    outer repo), mirroring the `papers/fiml-fmg/` scaffold. **Thesis correction:**
    the paper is NOT about robustness to non-normality. For ordinal DWLS/ULS the
    fit statistic is weighted-chi-square (`T → Σ λ_j χ²₁`, `λ_j ≠ 1`) *even at the
    correct normal model*, because the diagonal weight is not `Γ⁻¹` — so the
    correction is on-model, and a Gaussian-copula non-normal generator
    (Vale-Maurelli, NORTA) is invisible after thresholding (collapses to normal
    theory). The harness therefore generates **normal** underlying data and varies
    the spectrum-shaping axes: categories {2,3,5} × model size p {6,12} × N. It
    carries the single-group GOF arm (omitted residual covariance) and the
    two-group nested arm (configural-vs-metric via `nestedTest(satorra.2000)`),
    comparing naive / SB / WLSMV mean-variance / WLSMV scaled-shifted / FMG
    (pEBA, pOLS). Smoke-verified: pipeline runs, lavaan WLSMV parity holds (UGamma
    spectrum ~3.8e-9, DWLS chi-square ~1e-6 after the documented `(N-G)/N`
    rescale). The `sim_vm_*` core exposure stays (useful for the continuous
    papers) but is unused here. Remaining (author, in the paper repo's
    `dev/todo.md`): the full `just parity` run, and the live empirical question —
    do the FMG transforms beat the WLSMV mean-variance adjustment anywhere (only
    possible where the spectrum spreads: binary items, larger models), or is
    WLSMV hard to beat for ordinal?
  - **Chen-style pairwise-missing scalar probe (experiment 34, 2026-06).**
    Full-spectrum p-values do not repair the WLSMV_PD inflation once the
    pairwise-missing ordinal summary statistics are already distorted. In a
    Mplus-free 10-indicator invariant scalar test (`N=1000`, symmetric
    thresholds, overlap Gamma, 200 reps), scaled-shifted rejects 2.5% / 11.0% /
    31.0% at 0% / 30% / 50% missing; `all`/mixture is only slightly lower
    (2.5% / 10.5% / 30.0%); pEBA4 is more liberal (4.0% / 12.5% / 35.0%).
    Treat this as evidence that the Chen WLSMV_PD failure is not primarily a
    low-moment tail approximation problem; the missing-data moment/Gamma
    construction is upstream of the FMG transform.
    **MCAR vs MAR decomposition (added 2026-06-19, `--missing-mechanism`).**
    Re-running the same design under MCAR isolates the cause: every method holds
    nominal Type-I even at 50% missing (scaled-shifted 5.5% / 3.5% at 30% / 50%
    MCAR; whole battery .035-.050), while MAR inflates to ~.30-.35. The
    difference-statistic center is flat across MCAR (37.3 -> 37.4) and climbs
    across MAR (37.6 / 41.4 / 50.2) with the scale factor unchanged (~0.98), so
    the inflation is a bias-driven non-centrality from pairwise deletion being
    inconsistent under MAR, not a calibration/tail-approximation defect. Confirms
    the failure is the missing-data technique (PD), not the WLSMV estimator or
    the p-value family; the fix is full information or multiple imputation.
  - **Done 2026-06-13.** Direct ordinal UGamma-spectrum oracle: the paper
    parity pipeline now emits an explicit `ordinal_wlsmv_ugamma_spectrum_maxabs`
    row comparing magmaan's public DWLS + `robust_ordinal()` eigenvalues against
    lavaan WLSMV `lavInspect(., "UGamma")`. The core C++ fixture gate already
    compares `robust_ordinal().eigvals` to lavaan's stored UGamma eigenvalues in
    `tests/golden/ordinal_golden_test.cpp`; the paper row makes that provenance
    visible alongside the FIML parity table.
  - **Decided 2026-06-13.** Separate `papers/ordinal-fmg/` folder (not a second
    part of `papers/fiml-fmg/`).
  - **Done 2026-06-13.** Multi-group ordinal/mixed GOF FMG is first-class at the
    R boundary: the wrappers reuse the multi-block robust ordinal sandwich and
    `r-package/examples/fmg_ordinal.R` checks two-group all-ordinal and mixed
    SB parity against `robust_ordinal()` / `robust_mixed_ordinal()`.
  - **Only-when-needed.** A C++ methods-developer convenience entry point is not
    needed for the paper, since the spectrum-once-then-loop R orchestration
    avoids a per-method `robust_ordinal` recompute. (`_ug`/unbiased-Gamma is N/A
    for polychoric: the NACOV is already the asymptotic Gamma.)
## Benchmarks

Advisory local tooling, not a substitute for parity fixtures. Full design:
[docs/validation/benchmark_plan.md](../validation/benchmark_plan.md). The imhof
integrator is now QUADPACK-backed (~3x faster, ~2e-16 parity; roadmap), which was
the dominant cost of every FMG/pEBA/pOLS p-value. Deferred unbiased-spectrum perf
work lives in [`speculative.md`](speculative.md). Open work:

- **S.** Keep the build-loop timings table in
  [docs/architecture/roadmap.md](../architecture/roadmap.md) current after major
  workflow changes.
- **M.** Track objective value, gradient norm, iteration count, wall time, and
  agreement with lavaan-backed estimates where applicable.
- **S/M.** Continue extending benchmark coverage beyond the current
  lavaan-backed complete-data ML, controlled-missingness FIML, and continuous
  ULS/GLS smoke cases to WLS, ordinal DWLS/WLS, and mixed categorical models.
- **S/M.** Remaining ordinal SNLLS / Gamma workspace benchmark polish: lazy
  mixed WLS and mixed theta rows for the estimators landed 2026-06, and any
  R/API wrapper polish justified by the paper results. (All-ordinal delta
  fit-only, fit-plus-inference cache reuse, threshold-constraint,
  construction-boundary, raw-to-SNLLS legacy/lazy, delta/theta, mixed-DWLS
  lazy, two-group invariance, naive corr-block-WLS rows, and experimental
  `OrdinalStats` stage-2 weight reuse helpers for ULS/DWLS/WLS/NT/DLS have landed
  across `experiments/_archive/06/10/11/12/13` and the benchmark itself; the
  literature-grade `q ≤ 12` grid now runs from `papers/ordinal-snlls/`.)
- **Landed.** Two-stage EM / saturated-covariance missing-data path. Stage 1
  (`estimate::fiml::saturated_em_moments` / `estimate_saturated_em_moments`) and
  Stage 2 (`estimate_two_stage_em(partable, raw_data, kind = c("ml","gls"))`)
  have landed and feed the MSE comparator in
  `experiments/08-pairwise-gls-efficiency/`. The packaged ML2S path has also
  landed: `fit_ml2s()` / `magmaan(..., estimator = "ML2S")` run Stage-2 ML on
  the saturated EM moments and attach Savalei-Bentler-style corrected SEs plus
  scaled chi-square from the Stage-1 `(H, J, ACOV)` ingredients. The C++ post-fit
  layer also exposes lavaan's `robust.two.stage` scaled/robust CFI/TLI/RMSEA
  family, including baseline scaling and RMSEA interval/p-value variants. The
  GLS branch remains an explicit research comparator, not a named high-level
  estimator.
  `fmg_tests()` now accepts an ML2S fit: the eigenvalue-tail family (SS/all/pall/
  pEBA/pOLS, plus the SB/SS/SF/MV low-moment matches, where MV = `mv` =
  Satterthwaite mean.var.adjusted, a new `FmgMethod` matching lavaan to ~1e-12)
  is applied to the df-dim two-stage UGamma spectrum + Stage-2 ML base on
  `fit$ml2s`. The two-stage scaling and SEs match lavaan's
  `missing="robust.two.stage"` (sandwich ACOV) convention to machine precision
  (≲1e-4 across the exp-24 grid, typically ~1e-7; EM/optimizer-limited), and the
  robust/scaled global indices are lavaan-gated to fixture tolerances. This is
  not the plain `missing="two.stage"`
  (normal-theory ACOV) scaling, which collapses under non-normality; the base
  matches both. `trace(UGamma) = E[T]` (normal-data ncp ~ 0) is an independent
  first-principles check. Calibration study + lavaan parity oracle:
  `experiments/24-fiml-twostage-fmg-chisq`; unit gate:
  `two_stage_em_ml_inference` self-consistency in `tests/unit/fiml_test.cpp` and
  the `ml2s_*` rows of `tests/golden/fiml_golden_test.cpp`.
  - **Structured/unstructured weight axis carries to ML2S (resolved 2026-06-17).**
    The Satorra-Bentler U-metric weight choice - `WeightMoments::Structured`
    (model-implied Σ̂(θ̂)) vs `Unstructured` (sample/saturated h1, =
    `h1.information="unstructured"`) - applies to the two-stage path just as it
    does to FIML/robust.sem. ML2S now uses **Unstructured** for both the test
    spectrum and the SEs (`two_stage_em_ml_inference` in `src/estimate/fiml.cpp`),
    matching lavaan `robust.two.stage` (which hard-forces unstructured) and
    magmaan's own FIML FMG convention. It was briefly `Structured`, which left a
    1-3% trace/SE gap that grew with non-normality. Note: on complete data this is
    `robust.two.stage`, NOT `robust.sem`/MLM - those differ on the same axis when
    Σ̂ ≠ Σ(θ̂).
- **Landed; remainder in speculative.** The Van-Praag pairwise covariance
  machinery (`data::pairwise_sample_stats`,
  `robust::pairwise_casewise_contributions`, `data::gamma_nt_pairwise`,
  `estimate::fit_gls_pairwise`, the inference-side `WeightMoments::Pairwise` bread
  plus `robust::reduced_gamma_nt_pairwise` meat, and the matching R surface) has
  landed for `papers/pairwise-robust-sem/` and `experiments/08`/`09`. The
  remaining pairwise μ ACOV and pairwise Browne-unbiased items live in
  [`speculative.md`](speculative.md).
- **M.** Extend the near-singular ML continuation experiment
  (`experiments/04-near-singular-ml-continuation`) beyond the first
  diagonal-ridge path: the first target/profile grid compares diagonal,
  scaled-identity, and raw-identity targets over several fixed lambda sequences;
  remaining work is direct ML vs ULS/GLS-start ladders and explicit
  cost-normalized budgets on rank-near-deficient sample covariance cases.
- **Done 2026-06-15.** All three Geiser GLS/ULS parity exceptions are closed
  (`tests/golden/geiser_golden_test.cpp`, regression note in the test ledger):
  (1) manifest fixed.x path models (`manifest_regression`, `manifest_path`,
  `manifest_path_non_saturated`) resolve exogenous observed moments from the
  sample before the implied-moment check and gate Σ/μ against lavaan; (2) the
  latent AR cross-lagged family is rescued by a multi-start recipe (`best_start`:
  take the lower-objective of a `simple_start_values` fit and an ML-warm-started
  fit), reaching lavaan's Σ/μ to ~1e-7, with the scalar objective gate relaxed to
  `max(2e-4, 2.5e-3·|fx|)` (GLS) / `max(5e-3, 5e-3·|fx|)` (ULS) to absorb the
  documented ~1/N mean-structure scale convention; (3) the two *manifest* fixed.x
  cross-lagged path models (`manifest_ar_cross_lagged`, `…_extended`) now gate
  Σ/μ against lavaan to ~1e-8. The earlier "worse global optimum" symptom was
  **not** a core propagation bug: it was an observed-order mismatch in the golden
  harness. magmaan orders observed variables `[ov.y, ov.x]` (classify) and
  `SampleStats` is name-free / positionally aligned to that `ov_order`, but the
  fixtures supply `sample_cov`/`sample_mean` and lavaan's Σ/μ in their own data
  column order. The two cross-lagged models are the only Geiser cases where the
  exogenous variables (`d11`, `c11`) are not already last in the data, so
  `resolve_fixed_x` read exogenous moments from the wrong sample positions and
  the objective compared mis-ordered matrices. The harness now reconciles by
  variable name (`perm_to_magmaan` over `rep.ov_names`), reordering the sample
  and the lavaan moments into magmaan's `ov_order` (identity for the other
  cases). Still open from before: the Geiser per-parameter GLS comparison surface
  is implied-moment-based, not θ̂/SE-keyed.
- **S/M.** Per-parameter θ̂/SE parity for the Kline/Guo measurement-invariance
  corpus. The order-free chisq/df parity is gated; per-parameter parity needs a
  lavaan→magmaan free-parameter-order map (the submodule oracle stores
  `theta`/`se` in lavaan's free-parameter order).
- **S/M.** Extend the Mplus SEM corpus beyond the v1 strict growth tranche.
  `corpus/textbook-corpus/raw/mplus_sem` retains 80 first-pass translations and
  the tracked fixtures gate six continuous growth cases across ML/ULS/GLS/WLS.
  Remaining: repair or hand-translate the skipped growth/CFA cases whose
  automatic Mplus-to-lavaan conversion is malformed, decide how to test
  observed-only path models without exercising the saturated observed-path abort,
  and add categorical fixtures only for models that match magmaan's ordinal/mixed
  LS surface rather than Mplus logistic/probit response models.
- **S/M.** Extend the Little/Newsom textbook corpora beyond the initial strict
  tranche. The builders extract 108 Little LISREL models and 142 Newsom lavaan
  fit calls, with grouped tracked manifests and strict lavaan-backed C++ parity
  for the supported continuous subset; the consolidated
  `magmaan_textbook_corpus_v1` manifest indexes these alongside Geiser and Mplus
  SEM, with an advisory overlap graph for future paper mining. Remaining:
  implement real LISREL `SE` selection and more complex matrix/constraint
  conversion for Little, promote the Newsom cases that now parse/lavaanify
  cleanly, and add ordinal/mixed parity checks once the categorical oracle
  surface is settled.
- **S/M.** Promote the remaining first paper-corpus seed and broaden the
  paper-corpus fixture surface. `external/paper_corpus` owns scouting, minimal
  derived lavaan cases, validation, and magmaan JSON exports; magmaan consumes
  copied snapshots under `tests/fixtures/paper_corpus/`. `zxqvn` is promoted as a
  core complete-data ML point-estimate fixture. Remaining: promote `hwkem`,
  document license/data-handling for that richer source, extract supported lavaan
  model/data pairs, classify RI-CLPM pieces outside the core parity surface, and
  decide whether clustered-SE handling should become a later paper-corpus
  inference fixture.
- **S/M.** Add a small OpenMx tutorial corpus as an offline second-oracle
  cross-check, not as a runtime input format. Start with the dormant
  `openmx_mimic` case in `benchmarks/r/cases.R` and the OpenMx RAM examples noted
  in `docs/research/vendor-notes/benchmark_zoo.md`; hand-translate each retained
  model to lavaan syntax, harvest golden values from OpenMx (`mxRun`,
  `omxGetParameters`, `model$output`, `mxGetExpected(., "covariance")`), and tag
  those fixtures with `_meta.tool = "OpenMx"`. Keep the curated tier small
  (roughly 3-6 lavaan-expressible SEM/CFA cases, including a mixed
  continuous/ordinal CFA if licensing and fixture shape are clear), assert
  magmaan's implied moments against OpenMx's implied covariance, and keep
  `mxModel`/`mxPath` parsing plus a runtime RAM frontend out of scope.
- **S/M.** Refine benchmark use of optimizer diagnostics now that fit results
  expose `optimizer_status` and final gradient norms. Benchmark scripts should
  distinguish clean convergence from line-search salvage or singular PORT
  convergence, and still avoid interpreting backend-specific missing iteration
  counts as real zero-iteration solves.
- **XL.** Design an optimizer terminal-point "ultimate verifier" track. Turn the
  provisional audit tolerance into an empirically justified convergence
  certificate rather than a hand-tuned cutoff. Build an offline verifier that
  records the backend-independent L1 residual
  `||projected_gradient||_inf / (1 + |f_recomputed|)`, objective/parameter gaps
  to lavaan or certified fixtures, cross-backend same-basin agreement, PD margins,
  active bounds, and constraint residuals over the
  Geiser/Mplus/Little/Newsom/paper corpora. Add a high-precision check mode
  (`long double`, MPFR/Boost.Multiprecision, or an R `Rmpfr` helper) that
  re-evaluates `f` and the gradient at terminal points and optionally performs a
  few high-precision local refinement steps. Use the resulting CSV/report to
  separate ordinary line-search noise-floor salvages from genuinely non-stationary
  same-objective points (e.g. Newsom `ex5_4`/`ex5_4c`; see
  [newsom-corpus-failures.md](newsom-corpus-failures.md)) and to justify any
  default `TerminalAuditOptions` tolerance change in
  `docs/design/terminal-audit.md`.
- **M/L.** Decide whether `TerminalAuditOptions::stationarity_mode` should stay at
  Absolute (lavaan-matched) or switch to Relative once the verifier track above
  has data. v1 ships Absolute at `absolute_tol = 1e-3` to match lavaan's
  `check.gradient = TRUE` / `optim.dx.tol = 0.001` default; this is the first hard
  design call in magmaan and the calibration is genuinely unstable. The Relative
  code path is fully wired and unit-test-covered
  (`tests/unit/terminal_audit_test.cpp`), so the experiment is one option flip
  away once the data exists; see `docs/design/terminal-audit.md` "Tolerance
  calibration".
- **M/L.** Decouple terminal audits from optimizer coordinate systems so every
  fit result can be re-audited uniformly after any method-specific massage. The
  first concrete gap is SNLLS: `fit_snlls` / `fit_snlls_gls` convergence is
  audited in the profiled outer beta coordinates, relying on the inner alpha
  least-squares solve for the eliminated block. Add a post-hoc full-theta audit
  path, probably by generalizing `evaluate_at()`, so paper and benchmark harnesses
  can compare Full and SNLLS under the same expanded-theta KKT projected-gradient
  check. Record both verdicts (`profiled_beta_stationary` and
  `full_theta_stationary`) plus objective/gradient norms, inner-solve
  rank/conditioning, active bounds, and profile fallback diagnostics. Do not make
  the full-theta audit a default hard gate until the SNLLS corpora and verifier
  track show how often it differs from the expected profiled verdict.
- **M.** Compare NLopt L-BFGS/SLSQP/VAR2/TNEWTON/BOBYQA, PORT/PORT-NLS, Ceres
  trust-region, Ceres dense BFGS, and SNLLS only on semantically appropriate
  cases; include shallow or Heywood-prone LS cases so bounds and conditioning stay
  visible.
- **S/M.** Extend the paper-local SNLLS benchmark package in
  `papers/snlls-constrained/r-package/` with the remaining defensible real cases
  (especially a Geiser/Eid LST covariance input and a documented MTMM variant)
  plus one Boomsma-style simulation design. Keep the runner reporting setup time,
  fit time, whole time, iterations, objective values, and errors.
- **S/M.** For the SNLLS paper, add a narrow finite-difference-gradient mechanism
  probe rather than another broad simulation grid. NLopt's gradient-based
  algorithms (`LD_*`, including L-BFGS) expect caller-supplied gradients; the
  derivative-free `LN_*` algorithms (e.g. BOBYQA) are not the same as
  finite-differencing a BFGS gradient. If the paper needs an empirical
  Kreiberg-style explanation, add an explicit benchmark wrapper that
  finite-differences the existing scalar LS objective for ordinary LS and SNLLS
  under the same line-search optimizer, or report objective-evaluation accounting
  plus measured objective/Jacobian costs.
- **S.** In the SNLLS paper, spell out the implementation problem and solution
  more explicitly: the hard part is not proving separability but turning the
  profiled objective into residual/Jacobian calls an optimizer can trust. Explain
  why published tensor-gradient derivations are useful as a code blueprint and
  correctness check, while keeping the main text focused on the projection
  identity, the affine constraint split, and the fact that magmaan reuses the
  ordinary LISREL moment Jacobian instead of hand-writing pages of tensor
  products.
- **M/L.** Revisit the provisional default-backend choice once the optimizer
  comparison studies land. `Backend::NloptLbfgs` is now the default and NLopt is
  a required dependency, but the final default should still be justified across
  ML, complete-data LS, bounded ordinal LS, FIML/direct optimizer callers,
  augmented-Lagrangian inner solves, and nonlinear-constraint paths (NLopt SLSQP
  and IPOPT). Document tolerance semantics (`gtol` vs NLopt `xtol_rel`),
  iteration/evaluation reporting, and bounded behavior before changing the default
  again.
- **S/M, newsom corpus.** The Little/Newsom continuous golden
  (`tests/golden/textbook_corpus_golden_test.cpp`) is currently skipped because
  NLopt L-BFGS does not converge `newsom/ex5_5b` from `simple_start_values`. Same
  family as the documented `ex12_3` case in
  [newsom-corpus-failures.md](newsom-corpus-failures.md): NLopt stalling early on
  a structurally awkward ML objective. Unskip once the starting-value path or a
  harness-level cross-backend fallback handles it.

## Ordinal/SNLLS research

- **M/L.** Robust ordinal SEM paper track. Follow
  [docs/research/notes/robust_ordinal_sem_paper_plan.md](../research/notes/robust_ordinal_sem_paper_plan.md):
  build a paper-local simulation runner that emits tidy CSV for the Welz bivariate
  contamination design, the Welz/Foldnes-Gronneberg five-variable robust
  polychoric matrix design, an ordinal CFA downstream design, a Clayton copula
  stress test, and a small computation benchmark comparing ML, WMA hard cap,
  smooth h, and Huberized Pearson-residual moments. First milestone: reproduce the
  known Welz qualitative pattern for Designs 1-2 before adding SEM fits or broad
  copula grids.
  - **Robust-variant cleanup / evaluation harness (decide which to keep).** The
    full variant family is wired end-to-end (C++ core + R `data_ordinal_stats_-
    from_raw(robust=, h_kind=, clip=, ...)`), and `ros_method_specs()` now
    sweeps all 8: `ml`, `wma_hard_cap`, `smooth_cap`, `exp_cap`, `dpd`,
    `hard_huber`, `pseudo_huber`, `tukey_biweight`. Open question is which earn
    a main-text line vs. supplement vs. removal: we don't necessarily want all
    of them, but the call should come from a proper comparison, not a hunch.
    Build the evaluation setup before pruning: contamination sweep (bias/RMSE/SE
    calibration/convergence/Γ-conditioning/runtime) AND the copula
    distributional stress (Welz et al. evaluated on copulas too — Clayton main
    text, Gumbel/t supplement) so we separate "robust to tail contamination"
    from "robust to nonnormal copula." Prior signal: a limited pilot
    (`robust_ordinal_pilot_n1000_r5.csv`) had WMA hard cap doing best, and hard
    cap's φ is closed-form (the others — smooth/exp — carry Gauss-Legendre
    quadrature in `phi_from_h`). Hard cap alone is a sufficient paper; the
    cleanup decides whether the rest add enough to report.
    - **DONE 2026-06-24: copula distributional stress (Design 4).**
      `docs/research/sims/r/robust_ordinal_copula.R` runs the full 8-estimator
      family on a faithful Welz §8.2 replica (Clayton/Gumbel/Frank × rho_G
      {0.3,0.9}, N=1000, 5000 reps); our WMA hard cap reproduces robcat and ML
      reproduces the MLE to Table-5 precision. Findings in
      [robust_ordinal_copula_results.md](../research/notes/robust_ordinal_copula_results.md):
      WMA's robustness is *directional* (wins on Clayton, over-corrects on
      Gumbel rho=0.9 to worse-than-ML); the Huber/Tukey residual-clip family is
      dominated everywhere (Tukey SEs unusable); DPD is the only cross-copula-
      stable robust recipe. Still to run: the contamination sweep (Designs 1-2
      at scale) and the SEM-downstream design (Design 3).
- **M/L.** Ordinal SNLLS follow-up research. The all-ordinal delta ULS/DWLS/WLS
  path covers free, fixed, merged (including cross-group invariant), and
  general linearly constrained thresholds through both the threshold-profiled
  and full-threshold SNLLS routes; the all-ordinal theta path covers
  cache-aware bounded/SNLLS point estimation. Mixed continuous/ordinal delta
  and theta SNLLS exist for the materialized full-threshold DWLS/WLS path,
  mixed fit-only DWLS has a lazy workspace path, and mixed robust scaled tests
  now match lavaan at all-ordinal tightness. Use experiments to decide whether
  the next paper-facing C++ work should be lazy mixed WLS construction or
  reduced-Gamma inference plumbing.

## Composite models

The single-group ML slice has landed end to end: `<~` parsing (`Op::Composite`),
the historical Henseler-Ogasawara expansion and the native FC-SEM spec/evaluator
path (`CompositeMode::FcSem`, `model::FcSemEvaluator`, `estimate::fit_ml_fcsem`),
native expected SEs, standardization, fit measures, df, the `api::frontier`
surface, and the R frontier mirror (`fcsem_model_spec()`,
`fit_ml_fcsem()`/`magmaan_fcsem()`, `fcsem_standard_errors()`,
`fcsem_fit_measures()`, `fcsem_standardized_rows()`). Details are in the roadmap.
Single-group native ML lavaan parity for the pure-composite,
composite-plus-factor, and composite-structural HS fixture trio is now gated by
`tests/golden/composite_golden_test.cpp`.
Remaining:

- **M/L.** Multi-group composites are in scope only after the single-group ML and
  R frontier slices stay green and only if lavaan handles them cleanly, including
  `group.equal = "composite.weights"`.
- **S, after parity fixtures are green.** Add composite benchmark cases.

Deferred beyond the lavaan-validated single-group ML slice: ordinal composites,
FIML/LS composites, robust corrections for composites, and composite
mean-structure rows.

## Core/frontier layout follow-ups

Deferred from the first core/frontier separation pass, which introduced
`api::frontier` and retiered FMG, DLS, pairwise-composite, and shrinkage helpers
into `<domain>::frontier`. Canonical public headers now live under
`<domain>/frontier/`, with old public paths kept as forwarding shims. See
[docs/design/ideas.md](../design/ideas.md) for the tier model.

- **M/L.** Retier the remaining `data/` research cluster (`h_score`,
  `pairwise_ordinal`, `pairwise_mixed`) into `data::frontier`. Blocked: core
  `data/ordinal.{hpp,cpp}` is entangled with these headers - `ordinal.cpp` defines
  `pairwise_ordinal_stats_from_integer_data` and uses `eval_polychoric_h_score`,
  the core ordinal options embed `PolychoricHScoreOptions`, and `ordinal.hpp`
  `#include`s `pairwise_mixed.hpp`. Moving the headers naively inverts the
  dependency (core -> frontier). This work must first untangle `data/ordinal` -
  separating the core polychoric path from the research builders - then retier.
  `data::frontier::shrinkage` is already retiered, and `r-package/src/fit.cpp`
  calls the frontier namespace while keeping the R surface stable.
- **L.** Relocate the misplaced `estimate/` files to `spec/`: `constraints.hpp`
  (24 includers), `nl_constraints.hpp`, `expr_eval.hpp`, `resolve_fixed_x.hpp`
  (13). A structural relayering — its own pass, with a design note settling
  whether constraint *evaluation* is `spec` or `estimate`.
- **S/M.** Settle whether `cfa_utils.hpp` belongs in `spec` or `model`; depends on
  the start-values decision below.
- **M.** Gather the five start-value producers (`start_values.hpp`) into an
  `estimate::starts` sub-namespace; `start_values.hpp` has 16 includers and
  `spec::Starts` is part of the lavaanified-model triple, so this needs care.
- **M/L.** Retier the moment-quadratic misspecification research surface into
  `<domain>::frontier`. Currently in plain core namespaces: the estimated-weight
  fit-index inference (`ordinal_{crmr,rmsea,cfi_tli,fit_measures}_misspec_inference`
  in `estimate/ordinal`), the profile-Hessian primitives
  (`weighted_moment_profile_*`, `continuous_ls_profile_*`, `ml_profile_*`,
  `observed_moment_bread_fd` in `robust/weighted_inference`; `fiml_profile_*` /
  `two_stage_nt_profile_*` in `estimate/fiml`; `compute_profile_contrast_spectrum`
  in `robust/satorra2000`), the pre-existing exp-35 misspec-SE machinery
  (`robust_continuous_ls`, the Hall-Inoue `*_ij` family, `weighted_param_space_sandwich`
  and its estimated-weight MI counterparts `weighted_param_space_sandwich_ij` /
  `continuous_ls_param_space_sandwich_ij` / `ordinal_param_space_sandwich_ij`,
  also in `weighted_inference`/`ordinal`), and the older `OrdinalCatmlDwlsRmsea` probe. All
  are non-lavaan research yet sit in `magmaan::{estimate,robust,estimate::fiml}`.
  This must be one deliberate pass, not piecemeal: `weighted_inference.{hpp,cpp}`
  interleaves the new profile primitives with the widely-used `robust_continuous_ls`
  (callers across `api/`, tests, R glue), so wrapping only the audited stream would
  leave a half-migrated header. Do it with the `<domain>/frontier/` directory move
  + forwarding shims used by the first separation pass, updating R glue
  (`r-package/src/{robust,fit}.cpp`), `experiments/36`, the unit tests, and the
  vendored mirrors (`just vendor`). Flagged by the 2026-06-22 audit of the
  estimated-weight fit-index stream.
