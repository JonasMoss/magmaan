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

- **Ordinal-SEM standardized solution / defined params — mostly landed
  2026-06.** Decided: `compute_defined` is valid for ordinal/mixed fits (a
  parameterization-agnostic delta-method transform) and its guard was removed at
  both the C++ api and the Rcpp binding; `factor_scores` stays guarded (lavaan
  uses latent-response EBM integration, a distinct estimator deferred in
  `speculative.md`). Fixing the api flip surfaced a real bug: the ordinal api
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
  - **S.** Decide whether delta-method defined parameters and `factor_scores`
    need any further ordinal-specific handling once a downstream consumer asks.

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
  - **S.** The mixed polyserial pair fits recompute per-case quantities the
    way the all-ordinal path did before cell caching; profile
    `pairwise_mixed.cpp` (ML polyserial scores, casewise influence) for the
    analogous case→cell collapse where the per-case loop is over a smooth
    continuous coordinate (no exact cell structure; needs binning or a
    different factorization, so verify accuracy first).

## Robust score / modification-index tests (frontier)

`inference::frontier::{modification_indices,score_tests}_robust` has landed for
complete-data ML (both breads), continuous ULS/GLS/WLS/DWLS (the
`gmm::Weight` overloads, expected bread, moment-metric sandwich), FIML/MLR
(`{modification_indices,score_tests}_fiml_robust`, observed bread + casewise
meat), and — as
`estimate::frontier::{modification_indices,score_tests}_{ordinal,mixed_ordinal}_robust`
— all-ordinal ULS/DWLS/WLS and mixed DWLS/WLS over the polychoric NACOV meat.
The continuous ML/LS and ordinal/mixed-ordinal tiers are single- or multi-group;
FIML/MLR remains single-group; a df>1 total-release (`score_tests_robust_joint`,
mean-scaled + imhof mixture) covers joint releases.
Validated by `tests/unit/score_robust_test.cpp`,
`tests/golden/score_robust_golden_test.cpp` fixtures 0006-0012, the R-internals
oracle from `tests/tools/regen_robust_score.R`, and the advisory
`tests/checks/robust_score/`. Remaining work:

- **Done 2026-06-13.** Multi-group robust MI/score tests now cover the
  continuous ML/LS tiers and the ordinal/mixed-ordinal tiers. The ordinal guard
  in `estimate::frontier` is removed after validating the same per-block `n_b/N`
  sandwich over polychoric NACOV with the two-group WLSMV golden 0012, plus
  exact WLS reductions for all-ordinal MI/score and mixed-ordinal MI/score.
  FIML/MLR robust score tests intentionally remain single-group v1.
- **S, only-when-needed.** `api::frontier` / R wrappers for the LS and ordinal
  robust tiers (the api `Fit` does not currently carry the LS estimation
  weight; the `estimate::frontier` / `inference::frontier` functions are the
  methods-developer surface). Add when a concrete consumer appears.

## Local hardening and validation tooling

Local-first safety tooling for an AI-assisted repo. Design note:
[docs/validation/local_hardening.md](../validation/local_hardening.md). The test
ledger, risk map, regression-note convention, and JUnit/health recipes have
landed; remaining open items:

- **Partly calibrated.** Calibrate the local LLVM coverage lane (`coverage`
  preset, `just coverage` / `just coverage-html`, surfaced by `just health`, all
  under ignored `build/coverage/`) after more real runs. Still open: interpret
  the `llvm-cov` "mismatched data" warning, decide whether any ignore/object-list
  adjustments are needed, and add domain-level interpretation notes.
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
  size-triggered build-if. `group.equal = "thresholds"` parity is
  documented out of scope: lavaan's Wu-Estabrook identification frees group-2+
  delta scales and intercepts, which the magmaan ordinal delta path does not
  model (explicit shared threshold labels are the supported contract,
  fixture 0013).
- **M/L.** Optional h-weighted polyserial path: a polyserial-only h-weighted
  moment builder — continuous-ordinal h objective, casewise threshold/rho
  estimating functions, bread/influence/Gamma construction, and splicing into the
  mixed moment stack so `NACOV`/`W_dwls`/`W_wls` rebuild. The all-ordinal h-score
  variants already have the generic Gamma machinery; the missing piece is the
  mixed polyserial estimating-equation design.
- **FIML FMG follow-ups.** The landed FMG R wiring (single-model, FIML/missing,
  and the nested restriction-map route; see roadmap) leaves: the
  `magmaan(estimator = "FIML")` high-level path does not auto-enable mean
  structure (use `model_spec(..., meanstructure = TRUE)` + `fit_fiml`);
  multi-group FIML fits need explicit start/convergence care; and nonlinear
  equality tangent-space support plus pairwise-data FMG remain deferred.
- **Ordinal/polychoric FMG (`papers/fiml-fmg/` Paper 2).** Core gate **landed
  2026-06-13** (commit b8c6dcb): `fmg_tests_ordinal()` / `fmg_tests_mixed_ordinal()`
  apply the FMG eigenvalue-tail transforms to the `robust_ordinal()` /
  `robust_mixed_ordinal()` polychoric UGamma spectrum (`eigvals` + `chisq_standard`
  + `df`), single-group v1, `_ml`/`_ug` rejected, anchored by the ordinal C++ FMG
  test and `r-package/examples/fmg_ordinal.R`; no new C++ production code (see
  roadmap). What Paper 2 still lacks, to write the paper:
  - **L, paper sim harness.** A paper-local runner: an ordinal measurement-
    invariance population (threshold + loading invariance across groups), an
    ordinal data generator (categories × underlying skew × N), DWLS/WLS GOF +
    nested fits, and the naive-WLS vs SB vs FMG-winner (pEBA/pOLS/PALL) rejection
    grid. Mirror `papers/fiml-fmg/` Paper-1 scaffold.
  - **M, direct UGamma-spectrum oracle.** A paper parity check of the ordinal
    UGamma spectrum vs lavaan WLSMV (`lavInspect(., "UGamma")` or the internal
    scaling), analogous to Paper 1's FIML parity table. The spectrum is already
    gated indirectly via the ordinal robust SB/scaled-shifted goldens, but the
    paper wants the explicit spectrum-vs-lavaan row.
  - **S, author decision.** Single paper with a FIML + polychoric pair of parts,
    or a separate `papers/<slug>/` folder.
  - **Only-when-needed.** Multi-group ordinal FMG (lift the single-group v1 cap
    in `robust_ordinal`), and a C++ methods-developer convenience entry point —
    not needed for the paper, since the spectrum-once-then-loop R orchestration
    avoids a per-method `robust_ordinal` recompute. (`_ug`/unbiased-Gamma is N/A
    for polychoric: the NACOV is already the asymptotic Gamma.)
- **FIML cross-call pack: remaining consumers.** The `FIMLPack`/`FIMLH1`
  value-based precomputation (roadmap, Continuous FIML) is threaded through
  the post-fit helpers and `api::sem`. Not yet pack-aware: the R bindings
  (stateless per call — each `estimate_fiml_*()` round-trip rebuilds the pack
  and re-runs the H1 EM; threading it needs the R fit object to carry an XPtr
  to fit-time C++ state), `inference::{modification_indices,score_tests}_fiml`
  and `robust::lr_test_satorra2000_fiml_from_data` (rebuild pattern grouping
  and start stats per call; no H1 EM, so the win is one data pass, not an EM).

## Benchmarks

Advisory local tooling, not a substitute for parity fixtures. Full design:
[docs/validation/benchmark_plan.md](../validation/benchmark_plan.md). The imhof
integrator is now QUADPACK-backed (~3x faster, ~2e-16 parity; roadmap), which was
the dominant cost of every FMG/pEBA/pOLS p-value. Deferred unbiased-spectrum perf
work lives in [`speculative.md`](speculative.md). Open work:

- **S/M.** Consider replacing the f2c'd C under `third_party/quadpack/` with a
  clean C++23 hand-port of `dqagie`/`dqk15i` for readability — QUADPACK is public
  domain, so we can mirror it exactly.
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
  lazy, two-group invariance, and naive corr-block-WLS rows have landed
  across `experiments/_archive/06/10/11/12/13` and the benchmark itself; the
  literature-grade `q ≤ 12` grid now runs from `papers/ordinal-snlls/`.)
- **M/L.** Two-stage EM / saturated-covariance missing-data path. Stage 1
  (`estimate::fiml::saturated_em_moments` / `estimate_saturated_em_moments`) and
  Stage 2 (`estimate_two_stage_em(partable, raw_data, kind = c("ml","gls"))`)
  have landed and feed the MSE comparator in
  `experiments/08-pairwise-gls-efficiency/`. Still open: a packaged
  `estimator = "ML2S"` surface with the Savalei-Bentler (2009) corrected
  chi-square and SEs, consuming the Stage 1 `(H, J, ACOV)` ingredients.
- **Landed; remainder in speculative.** The Van-Praag pairwise covariance
  machinery (`data::pairwise_sample_stats`,
  `robust::pairwise_casewise_contributions`, `data::gamma_nt_pairwise`,
  `estimate::fit_gls_pairwise`, the inference-side `WeightMoments::Pairwise` bread
  plus `robust::reduced_gamma_nt_pairwise` meat, and the matching R surface) has
  landed for `papers/pairwise-robust-sem/` and `experiments/08`/`09`. The
  remaining pairwise μ ACOV and pairwise Browne-unbiased items live in
  [`speculative.md`](speculative.md).
- **M/L.** Convergence-note / start-value portfolio paper track
  (`papers/convergence-note/`). The skeleton, local resources, and first R
  simulation factories exist for the De Jonckere-Rosseel small-sample SEM designs
  and the Ludtke-Ulitzsch-Robitzsch weak-loading CFA design. Next: add runners
  that compare simple/FABIN/Guttman/Bentler/James-Stein starts, bounded random
  starts, and screened portfolios under equal budgets; filter genuinely hard
  cases by final gradient norm, objective gap, admissibility, PD margin, and
  multistart basin disagreement rather than optimizer failure alone.
- **M.** Extend the near-singular ML continuation experiment
  (`experiments/04-near-singular-ml-continuation`) beyond the first
  diagonal-ridge path: the first target/profile grid compares diagonal,
  scaled-identity, and raw-identity targets over several fixed lambda sequences;
  remaining work is direct ML vs ULS/GLS-start ladders and explicit
  cost-normalized budgets on rank-near-deficient sample covariance cases.
- **S/M.** Tighten the remaining Geiser GLS parity exceptions documented in the
  parity-tier golden: manifest fixed.x path models need a resolved implied-moment
  comparison surface for exogenous observed moments, and `latent_ar_cross_lagged`
  needs either a stable same-basin optimizer recipe or a written alternate-optimum
  note.
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

- **M/L.** Structured Gamma / model-implied fourth-order paper track
  (`papers/structured-gamma/`). A minimal `estimate::frontier` weight-matrix
  builder exists for complete-data covariance-only pure CFA and is exposed to R as
  both a raw Gamma matrix and a WLS-compatible weight; the paper-local R
  simulation helper generates one- and five-factor CFA scenarios, runs
  NT/ADF/fixed-mix DLS/MI4 working-weight fits, records MI4 positive-definiteness
  failures as data, optionally applies eigen-floor or minimal NT-target shrinkage
  repair, and writes pilot CSV/R data from `scripts/sim_structured_gamma.R`.
  Remaining paper work: run and summarize a defensible grid, calibrate/report the
  MI4 repair policy, and decide later whether broader model classes are needed.
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
- **M.** Move `reparameterize.hpp` to `optim/` (optimizer machinery, not an
  estimator); it is coupled to the `estimate/constraints.hpp` move above.
- **S/M.** Settle whether `cfa_utils.hpp` belongs in `spec` or `model`; depends on
  the start-values decision below.
- **M.** Gather the five start-value producers (`start_values.hpp`) into an
  `estimate::starts` sub-namespace; `start_values.hpp` has 16 includers and
  `spec::Starts` is part of the lavaanified-model triple, so this needs care.
