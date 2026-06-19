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

## Misspecification-robust SE for the moment-quadratic family (frontier)

- **Done 2026-06-19.** Observed-Hessian ("robust" regime) bread for the
  moment-quadratic estimators (ordinal DWLS/WLSMV/ULSMV, mixed/polyserial,
  continuous ULS/GLS/WLS), which previously had only the Gauss-Newton/expected
  bread. Built as `observed_moment_bread_fd` (central-difference of the per-unit
  moment-LS gradient, reduced via `K`) injected as an optional `bread_override`
  into `robust_weighted_moments`; the meat (NACOV), df, and chi-square are
  unchanged. Selector `bread = "expected" | "observed"` (reusing
  `robust::Information`) threaded through `robust_ordinal` / `robust_mixed_ordinal`
  / `robust_continuous_ls`, surfaced in R as `vcov(fit, regime = "model" |
  "robust")`, which feeds `standardized(fit, vcov)`. Continuous ML/FIML already
  had the observed bread (unchanged). Default stays `expected` (lavaan parity).
  Validation: experiment 35 (`35-misspec-robust-se`) -- under the correct model
  the two breads coincide and both match the empirical sampling SD. Focal is a
  free loading on a cross-loading-distorted factor (raw + std.all), continuous ML
  and ordinal DWLS. Under the omitted cross-loading the expected-bread SE
  underestimates by about a third (std loading 0.69-0.78 of truth) and the
  observed bread recovers it -- essentially completely for continuous ML
  (std 1.01) and substantially for ordinal DWLS (std 0.85, the new bread). Note:
  `docs/research/notes/misspec_observed_bread.tex`. This is `frontier`, not core
  parity: the correct-model cell is the validation oracle (no mainstream
  *software* -- lavaan/Mplus -- exposes this for the ordinal/WLS family).
  This observed-bread SE *is* Lai & Simoes (2023, SEM 30:5, "Reflecting on the
  'Robust' Standard Errors..."): their new SE (Eq 34-36, `Pi_DWLS`, bread
  `H = 2 D' Gd^-1 D - 2[I (x) eps' Gd^-1] Psi`); their "robust SE" (Eq 17, Muthen
  1997 = lavaan default) is our expected bread. We compute `H` by FD; they have
  the closed form. Their findings match exp 35 (robust SE biased 15-40% under
  misspec, not improving with n; new SE consistent; ULS closes better than DWLS).
  The DWLS finite-sample residual is the data-dependent weight `Wd = Gd^-1`: both
  their new SE and ours treat the Stage-2 weight as fixed, so neither captures its
  higher-order variability (ULS, W=I, has none -> closes cleanly); a
  case-resampling bootstrap recovers that term.
  Follow-up landed 2026-06-19: complete all-ordinal ULS/DWLS now has
  `robust_ordinal_ij`, an explicit infinitesimal-jackknife covariance that carries
  the estimated diagonal-weight term. The data-direct Γ-diagonal channel is
  regression-tested against case-weight finite differences. The reusable
  `robust_weighted_moment_ij` primitive is now the shared transport for
  observed-bread weighted-moment IJ covariance: callers provide per-case moment
  influence rows plus optional per-case estimated-weight corrections. Full WLS,
  mixed ordinal/polyserial, MCAR/pairwise-missing variants, and ML2S adapters
  remain out of scope. Derivation and implementation grid:
  `docs/research/notes/weighted_moment_ij_grid.tex`.
  Open follow-ups:
  - **Estimator-grid expansion for estimated-weight IJ.** Implement in small
    verified slices, all using the note's four gates: fixed-weight reduction to
    `robust_weighted_moments`, derivative checks for each new weight-influence
    channel, resampling-jackknife agreement on deterministic fixtures, and a
    misspecification simulation with ULS/fixed-weight negative controls.
    1. Complete continuous LS: expose the ULS/fixed-weight IJ reduction first,
       then use raw mean/covariance moment influence to validate GLS/NT,
       DWLS/WLS, and DLS estimated-weight formulas.
    2. Complete mixed ordinal/polyserial: add casewise mixed moment influence in
       the exact mixed moment order; gate ULS first, then diagonal estimated
       weights for polyserial rows, then full WLS.
    3. Complete all-ordinal full WLS: generalize the current diagonal
       `robust_ordinal_ij` path from `diag(IF(Gamma))` to full `IF(Gamma)`,
       with diagonal extraction required to reproduce DWLS.
    4. ML2S: add casewise saturated-EM moment influence and route
       `TwoStageWeight::{Nt,Dwls,Adf,Dls}` through the same formulas; the NT
       weight still has a derivative when it is built from saturated sample
       moments.
    5. MCAR/pairwise-missing ordinal and polyserial: design a case-aligned sparse
       moment-row primitive with per-moment support scaling before coding; the
       current disjoint-block IJ primitive is not enough for overlapping
       pairwise supports.
  - **Analytic moment-Hessian** for the observed bread (closed-form `H` of Lai
    Eq 36 / its mixed analogue; FD becomes the validation gate). Also removes the
    per-resample-SE cost that makes the studentized bootstrap expensive.
  - **Two-stage FIML (ML2S)**: also two-stage, so its robust SE likely uses the
    Gauss-Newton/expected Stage-2 bread -- give it the observed-bread regime too.
  - **FIML**: verify it really is misspecification-robust (its bread is the
    observed Hessian by construction); add an expected-vs-observed comparison and
    a `vcov(fit, regime=)` route so the regime keyword is uniform, plus a
    misspecified-model FIML cell in exp 35.
  - **ULS-vs-DWLS finite-sample probe (CONFIRMED 2026-06-19)**: on a structural
    `f3~f1` parameter, the DWLS observed-bread SE/empirical-SD ratio plateaus at
    ~0.945 and does NOT improve with n (0.944 at N=2500, 0.946 at N=25000), while
    ULS is ~1.0 at every n. So the DWLS gap does not vanish -- it is the omitted
    leading-order weight term `~Wd'(u)*eps` (zero under the null, O(1) under
    misspec); ULS (W=I fixed) has none. Resolves Lai's "unclear whether it
    vanishes": for DWLS it doesn't. (Not yet a tracked script -- /tmp only.)
  - **(Research / "insanely cool") explicit second-order corrections**: Edgeworth /
    Cornish-Fisher / saddlepoint / Nagar variance expansion for the two-stage
    ordinal estimator. All need the same hard ingredient -- the third cumulants of
    the Stage-1 polychorics/thresholds (Olsson's `Gamma` is only second-order) plus
    `d^3 eta/d theta^3`. Big project; scope a 1-parameter or continuous-ML proof of
    concept first. The iterated/double bootstrap reaches the same second-order
    coverage *without* deriving cumulants (B^2 cost, but magmaan is fast enough).
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
  second-order Imhof calibration) plus a thin R wrapper. Not required.
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
- **M, evaluation — structured-h1 FIML FMG: keep or ditch.** The FIML FMG U·Γ
  spectrum exposes `estimate::FIMLH1Information::Structured`, which evaluates the
  U-side information `V` (the saturated observed Hessian `J`) at the model-implied
  moments `ξ(θ̂)` instead of the saturated `ξ̃` (`fiml_structured_h1_information`,
  `src/estimate/fiml.cpp:2155`; consumed in `fiml_ugamma_spectrum_impl`, `:2245`).
  Concern: at `ξ(θ̂)` the observed Hessian is not guaranteed positive definite
  off-H₀ (the `ȳ_j − μ_j(θ̂)` terms, `:842-846`), and nothing guards it. The only
  definiteness check is on the model-implied `Σ_oo` (always PD); `invert_symmetric`
  silently LDLT-falls-back on an indefinite `Δ'VΔ` (`:500-517`); and the rank
  sanity check (`:2374`) is necessary-not-sufficient (one boundary eigenvalue, no
  sign check on the retained spectrum, which then feeds the FMG/pEBA mixture that
  assumes `λᵢ ≥ 0`). The default `Saturated` evaluates `V` at the saturated
  optimum `ξ̃`, PD by second-order optimality, so it is unaffected. Structured-h1
  is *not* asymptotically advantageous (both consistent under H₀); the
  Satorra-Bentler convention keeps the meat/`Ω` saturated and varies only `U⁰`,
  and the complete-data evidence (Xia et al. 2016, via Savalei & Rosseel 2022)
  mildly prefers the saturated/unstructured information. Evaluation: under the
  exp-23 misspecified conditions, measure (a) how often structured `V` / `Δ'VΔ`
  goes indefinite, and (b) whether structured-h1 ever beats saturated on Type-I /
  power. If it never helps and risks indefiniteness, remove the `Structured`
  option and keep `Saturated` as the sole FIML-FMG path (the FMG-spectrum
  convention anyway). Caveat to weigh before removal: structured-h1 + observed-info
  is the combination that gives lavaan-MLR bread parity, so dropping it ends exact
  MLR reproduction from magmaan (the exp-23 MLR baseline runs via lavaan, so the
  comparison itself is unaffected).
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
