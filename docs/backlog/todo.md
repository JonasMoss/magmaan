# magmaan TODO

Remaining-work backlog. Current state, architecture, and contracts live in
[docs/architecture/roadmap.md](../architecture/roadmap.md); this file only tracks unfinished work.

Effort tags: **S** bounded docs/fixtures/wrapper cleanup · **M** focused
implementation or test slice · **L** new estimator plumbing or cross-module
semantics · **XL** statistical design/research track before implementation.

## Correctness bugs

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

- **S.** **ADF/WLS weight builder should tolerance-trim Γ̂ before inversion** —
  At small N with a binary covariate, the empirical Browne NACOV is
  positive-definite to working precision but has condition number ≈ 10¹⁷
  along one cross-third-moment direction. Today's
  `snlls_adf_weight(include_means=TRUE)` path computes `chol2inv(chol(Γ̂))`
  unconditionally; that succeeds (smallest eigenvalue technically > 0) but
  amplifies the singular direction by ~10¹⁶, so a machine-precision
  residual `||r|| ≈ 10⁻⁸` at a saturated fit produces `||W·r|| ≈ 10⁸` and
  the KKT projected-gradient audit fires on what is analytically a zero
  gradient. Replace with a tolerance-aware pseudoinverse at cutoff
  `tol · max(eig(Γ̂))` (default e.g. `tol = 1e-12`); zero out below-cutoff
  eigendirections before inverting. Same fix likely applies to the
  block-diagonal `snlls_adf_weight(include_means=FALSE)` path. Surface:
  `src/estimate/gmm/dls_weight.cpp` (and the paper-side wrapper
  `r-package/R/core-cases.R::snlls_adf_weight`). Reproducer:
  `muthen_2017_ch2_ex2_1__adf` (Hayes PROTEST mediation; N=129; binary
  treatment 41/88 split; saturated path model; Γ̂ has 8 well-conditioned
  eigenvalues 10.4 → 0.09 plus one at 7.5×10⁻¹⁷). Per-element diagnosis at
  `papers/snlls-constrained/reports/lcs-objective-gap.md` (the §B‴ writeup
  also covers this side finding).

## API and R boundary

- **S/M.** Add or rename R wrappers only when the methods-developer workflow
  exposes a concrete gap in the staged API; the current `magmaan_core`,
  `magmaan_fit`, and post-fit wrapper surface is otherwise sufficient for the
  next R exploration pass.
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
  All-ordinal delta SNLLS for ULS/DWLS/WLS now reuses the profiled correlation
  objective, fixes thresholds away before the generic Golub-Pereyra split, and
  returns estimates in the ordinary prepared ordinal partable coordinate, with
  WLS using the Schur-complement profiled weight. Fixed threshold rows now stay
  in the profiled full-moment objective for ULS/DWLS/WLS and are covered for
  bounded fits plus WLS SNLLS. Threshold-local shared-label / bare-merge
  equality groups now share threshold-map columns and are covered for bounded
  fits plus WLS SNLLS. Remaining C++ estimator work on this track:
  cache-aware mixed continuous/ordinal fitting; general linear threshold maps
  such as effect-coding-style constraints; theta parameterization in the
  cache-aware/profiled/SNLLS path; reduced-Gamma robust-inference products that
  avoid full materialization where possible; and only-when-needed R/API polish.
  First benchmark slice landed as `magmaan_ordinal_workspace_bench` plus
  `experiments/06-ordinal-snlls-probe`: it covers all-ordinal delta fit-only
  ULS/DWLS/WLS across free, fixed, and shared threshold models and checks
  cache/materialization flags plus agreement with bounded/materialized paths.
  Next practical slices are fit-plus-inference timing, raw/lazy Gamma
  construction boundaries, then the remaining estimator generalizations.
- **M/L.** Optional h-weighted polyserial path: a polyserial-only h-weighted
  moment builder — continuous-ordinal h objective, casewise threshold/rho
  estimating functions, bread/influence/Gamma construction, and splicing into
  the mixed moment stack so `NACOV`/`W_dwls`/`W_wls` rebuild. The all-ordinal
  h-score variants already have the generic Gamma machinery; the missing piece
  is the mixed polyserial estimating-equation design.

## Benchmarks

Advisory local tooling, not a substitute for parity fixtures. Full design:
[docs/validation/benchmark_plan.md](../validation/benchmark_plan.md).

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
  Remaining: fit-plus-inference/robust reuse timing, raw-data lazy construction
  boundaries, opt-build larger grids, and any R/API wrapper polish justified by
  those results.
- **M/L.** Add a two-stage EM/saturated-covariance missing-data research path
  for comparison with direct FIML and pairwise covariance methods. Stage 1
  (saturated EM moments + sandwich ACOV ingredients) is now exposed as
  `magmaan::estimate::fiml::saturated_em_moments` / R
  `magmaan_core$estimate_saturated_em_moments` — the methods-developer surface
  Savalei-Bentler (2009) needs. Stage 2 (second-stage SEM fit + Savalei-Bentler
  corrected chi-square / SEs as a packaged `estimator = "ML2S"`) and the
  `pairwise-robust-sem` simulation row that consumes Stage 1 are still open.
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
  wall time, divergence-from-truth ratio. Remaining inference-side
  Γ_NT^pw plumbing, pairwise μ ACOV, and pairwise Browne-unbiased live in
  [`speculative.md`](speculative.md) — none committed until a downstream
  consumer needs them.
- **M.** Track objective value, gradient norm, iteration count, wall time, and
  agreement with lavaan-backed estimates where applicable.
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
  models. Use experiments to decide whether the next paper-facing C++ work
  should be theta-profiled SNLLS, mixed continuous/ordinal SNLLS, general
  linear threshold maps, or reduced-Gamma inference plumbing.

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
