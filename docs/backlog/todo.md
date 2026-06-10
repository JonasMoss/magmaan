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

Small open items surfaced while fixing the standardized-solution, Kline/Guo
parity, and ADF inversion bugs (the fixes themselves are recorded in the test
ledger).

- **S.** Ordinal-SEM standardized solution: add a checked C++ golden fixture for
  ordinal / mixed-ordinal standardized rows (currently only `r-package/examples/
  ordinal_dwls_wls.R` plus the experiment `--lavaan-parity` runs cover it), and
  decide whether delta-method defined parameters (`compute_defined`) and
  `factor_scores` are valid for ordinal fits — both still carry the
  `require_not_ordinal` guard that was removed from `standardize_lv`/`_all`.
- **M, research-tier.** Optional non-default `spectral_truncate` weight policy
  for degenerate saturated continuous ADF/WLS Γ̂: a pseudo-inverse returning
  dropped rank / retained weighted residual / conditioned projected-gradient
  norm, as a parity-restoring alternative to the current rank-deficiency refusal
  (`detail::symmetric_inverse_pd_gated`). `experiments/00-lavaan-parity`'s
  `conditioned_adf_weight()` is the advisory telemetry for it.

## Robust score / modification-index tests (frontier)

`inference::frontier::{modification_indices,score_tests}_robust` has landed for
complete-data ML, both breads, single group (see roadmap; validated by
`tests/unit/score_robust_test.cpp`, `tests/golden/score_robust_golden_test.cpp`,
the R-internals oracle from `tests/tools/regen_robust_score.R`, and the advisory
`tests/checks/robust_score/`). Remaining work:

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
  Sweep the remaining golden/parity tests for similar slack tolerances,
  `MESSAGE`-only soft checks, and fixtures regenerated from magmaan's own output
  rather than an external oracle; for each, either tighten to the oracle or write
  down explicitly why the looseness is principled.
- **S, deferred.** A `statistic = "N" | "N-1"` (or report-both) selector for the
  GLS/WLS test multiplier, deferred until a concrete methods workflow needs it.
  The convention itself is fixed and documented in
  [docs/design/numerical-conventions.md](../design/numerical-conventions.md).
- **M, later.** Layer CI on top only after the local commands are useful:
  `test-quick` on PRs/pushes, sanitizer validation on main or a schedule, heavy
  parity/optional optimizer lanes less often, and coverage as an artifact before
  considering badges. Avoid coverage-percentage gates until the report has been
  calibrated by real maintenance work.
- **S/M.** Add a self-contained C++ golden for the FMG p-value transforms (fixed
  eigenvalues + chi-square → p-value per method) so the eigenvalue-tail maths is
  gated without R. The single-model FMG family is already value-for-value
  parity-checked against `semTests::pvalues` in `examples/fmg.R` (run by
  `just r-check`); coordinate with the exp-17 pEBA work, which owns the
  nested/multi-group side of the same machinery.

## Simulation primitives

The `magmaan::sim` surface (NORTA / IG / Vale-Maurelli-Fleishman / PLSIM /
t-copula / Archimedean copula / C-vine / elliptical generators, marginal
families, and observed-variable projection) is summarized in the roadmap; its
work queue and decision log live in [`simulation.md`](simulation.md). Keep this
section as a high-level index for cross-domain planning — including the Johnson
SL exposure decision, the special-functions dependency policy, NORTA calibration
hardening, and the remaining R `calibrate_*`/`draw_*` split for NORTA and the
copula/vine paths — and put detailed generator/marginal/fixture decisions in the
simulation backlog.

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
  [docs/design/ordinal-snlls-gamma-architecture.md](../design/ordinal-snlls-gamma-architecture.md)):
  tighten mixed robust scaled-test parity beyond the current loose guard, lazy
  mixed WLS construction, mixed theta SNLLS, threshold-profiled general linear
  maps (e.g. effect-coding-style constraints), reduced-Gamma robust-inference
  products that avoid full materialization where possible, and only-when-needed
  R/API polish.
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
- **S/M.** Extend ordinal SNLLS / Gamma workspace benchmark coverage: larger /
  literature-grade speed grids, tighter mixed robust scaled-test diagnostics,
  lazy mixed WLS and mixed theta rows once those estimators exist, and any R/API
  wrapper polish justified by the results. (All-ordinal delta fit-only,
  fit-plus-inference cache reuse, threshold-constraint, construction-boundary,
  raw-to-SNLLS legacy/lazy, delta/theta, and mixed-DWLS lazy rows have landed
  across `experiments/_archive/06/10/11/12/13`.)
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
- **S/M.** Extend the ordinal SNLLS speed pilot
  (`experiments/_archive/11-ordinal-snlls-speed`) when the paper needs stronger
  timing evidence. The native C++ benchmark now compares full bounded DWLS/WLS,
  threshold-profiled bounded, full-threshold SNLLS, and threshold-profiled SNLLS
  across a Kreiberg-style family, splits rows by delta/theta parameterization,
  and includes mixed continuous/ordinal rows plus the lazy `MixedOrdinalWorkspace`
  boundary. The latest opt smoke pilot covers the compact `--smoke` grid with
  `q ≤ 4`; open follow-up is the fuller `q ≤ 12` literature-like sweep and
  optional lavaan context rows if the paper needs them.
- **S.** Extend the ordinal threshold-constraint support experiment
  (`experiments/_archive/12-ordinal-threshold-constraints`) only if the paper
  needs broader constraint evidence. Single-group ordinal CFA is covered (free /
  shared-label thresholds agree across profiled/full paths; true linear threshold
  constraints are rejected by threshold-profiled fitting and accepted by full
  bounded plus full-threshold SNLLS). Open follow-up: multi-group
  threshold-invariance/equality examples once the harness can express groups
  without pulling in fixture generation.
- **S/M.** Extend the ordinal construction-boundary experiment
  (`experiments/_archive/13-ordinal-construction-boundary`) if the paper needs
  broader construction evidence. The lazy opt pilot times fit-only ULS/DWLS raw
  workspace construction against eager legacy stats construction, projection to
  `OrdinalMoments`, diagonal/full Gamma cache copies, DWLS weight construction,
  and WLS reinversion across all-ordinal synthetic blocks up to `p = 16`, `c = 5`.
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
- **M.** Maybe add exact Hessians for IPOPT. The first IPOPT adapter uses
  limited-memory Hessian approximation and supplies objective gradients plus
  nonlinear-constraint Jacobians only. Revisit after the optimizer comparison work
  clarifies whether exact objective / Lagrangian Hessians materially help ML,
  GLS/WLS/ULS, or nonlinear-constraint fits.
- **S/M, newsom corpus.** The Little/Newsom continuous golden
  (`tests/golden/textbook_corpus_golden_test.cpp`) is currently skipped because
  NLopt L-BFGS does not converge `newsom/ex5_5b` from `simple_start_values`. Same
  family as the documented `ex12_3` case in
  [newsom-corpus-failures.md](newsom-corpus-failures.md): NLopt stalling early on
  a structurally awkward ML objective. Unskip once the starting-value path or a
  harness-level cross-backend fallback handles it.
- **S, before shipping binary artifacts.** The repo carries an MIT `LICENSE` that
  also notes the vendored BSD-3 PORT routines, sufficient for a source release —
  Eigen, Ceres, NLopt, and nlohmann_json are fetched at build time, not
  redistributed. Before shipping a binary or packaged artifact, extend this to a
  full dependency-license manifest (Eigen, optional Ceres, required NLopt,
  optional IPOPT, vendored PORT and QUADPACK, and test-only nlohmann_json) with
  versions and redistribution obligations.
- **M/L, after coverage exists.** Promote the Ceres preset into regular validation
  where relevant without making the default build pay the Ceres dependency cost.
- **S, only if timings justify it.** Experiment with opt-in precompiled headers
  for Eigen-heavy local builds; keep them disabled unless they improve changed-TU
  rebuilds without worsening no-op or full rebuild ergonomics.

## Ordinal/SNLLS research

- **S, revisit only if the paper track needs it.** The h-score / WMA robust
  polychoric path has landed (all-ordinal h-weighted moments; see roadmap), so its
  phased plan note has been retired. Design rationale lives in
  `docs/research/notes/h-polychorics.tex` and `robust_ordinal_gamma.tex`; the
  remaining concrete work is the h-weighted polyserial item under "API and R
  boundary". Revisit the threshold parameterization and positive-definiteness
  repair policy only if the robust-ordinal paper track demands it.
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
  prototype exists for free, fixed, and pure-merge threshold models, the
  all-ordinal theta path covers cache-aware bounded/SNLLS point estimation, and
  the full-threshold SNLLS path supports general linear threshold constraints.
  Mixed continuous/ordinal delta SNLLS exists for the materialized full-threshold
  DWLS/WLS path, and mixed fit-only DWLS has a lazy workspace path. Mixed robust
  scaled-test fields are checked against lavaan only with loose tolerances; exact
  mixed scale/eigen parity remains open. Use experiments to decide whether the
  next paper-facing C++ work should be tighter mixed robust inference, lazy mixed
  WLS construction, mixed theta SNLLS, threshold-profiled general linear maps, or
  reduced-Gamma inference plumbing.
- **S/M, experiment extension.** `experiments/15-rhemtulla-2012` replicates the
  Rhemtulla, Brosseau-Liard & Savalei (2012) cat-LS-vs-continuous-ML horse race,
  but v1 covers only the symmetric-threshold, underlying-normal conditions
  (number of categories 2-7 × N). Two paper conditions are deferred: (1) the
  nonnormal underlying `y*` (skew 2, kurtosis 7 in the paper's convention),
  covered by the C++ cubic Fleishman / Vale-Maurelli primitive but still needing
  wiring into the experiment — the condition where cat-LS's own
  underlying-normality assumption breaks; (2) the moderate/extreme
  asymmetric-threshold conditions, whose exact threshold tables are in the paper's
  unavailable supplement (would need a documented rule validated against their
  Table 1 skew/kurtosis). Add these only if the replication needs to exercise
  cat-LS bias under nonnormality.

## Composite models

The single-group ML slice has landed end to end: `<~` parsing (`Op::Composite`),
the historical Henseler-Ogasawara expansion and the native FC-SEM spec/evaluator
path (`CompositeMode::FcSem`, `model::FcSemEvaluator`, `estimate::fit_ml_fcsem`),
native expected SEs, standardization, fit measures, df, the `api::frontier`
surface, and the R frontier mirror (`fcsem_model_spec()`,
`fit_ml_fcsem()`/`magmaan_fcsem()`, `fcsem_standard_errors()`,
`fcsem_fit_measures()`, `fcsem_standardized_rows()`). Details are in the roadmap.
Remaining:

- **L.** Post-fit lavaan parity validation: minimal oracle fixtures exist under
  `tests/fixtures/composite/` for pure composite, composite plus common factor,
  and structural regression involving a composite. The diagnostic golden is wired
  but skipped while we decide whether native W-matrix parity belongs in low-level
  C++ fixture tests, an R frontier diagnostic, or both.
- **M/L.** Multi-group composites are in scope only after the single-group ML and
  R frontier slices stay green and only if lavaan handles them cleanly, including
  `group.equal = "composite.weights"`.
- **S, after parity fixtures are green.** Add composite benchmark cases.

Deferred until the ML slice is lavaan-validated: ordinal composites, FIML/LS
composites, robust corrections for composites, and composite mean-structure rows.

## Core/frontier layout follow-ups

Deferred from the first core/frontier separation pass, which introduced
`api::frontier` and retiered `robust/fmg.hpp`, `estimate/gmm/dls_weight.hpp`, and
`estimate/pairwise.hpp` into `<domain>::frontier`. See
[docs/design/ideas.md](../design/ideas.md) for the tier model.

- **M/L.** Retier the `data/` research cluster (`h_score`, `pairwise_ordinal`,
  `pairwise_mixed`, `shrinkage`) into `data::frontier`. Blocked: core
  `data/ordinal.{hpp,cpp}` is entangled with these headers — `ordinal.cpp` defines
  `pairwise_ordinal_stats_from_integer_data` and uses `eval_polychoric_h_score`,
  the core ordinal options embed `PolychoricHScoreOptions`, and `ordinal.hpp`
  `#include`s `pairwise_mixed.hpp`. Moving the headers naively inverts the
  dependency (core → frontier). This work must first untangle `data/ordinal` —
  separating the core polychoric path from the research builders — then retier.
  `data/shrinkage.hpp` is the one cleanly separable header and can move first. R
  glue (`r-package/src/fit.cpp`) calls these `data::` symbols, so the retier needs
  one R-side requalification and a `just r-check`.
- **S.** Move the retiered frontier headers into `<domain>/frontier/`
  subdirectories so directory matches namespace again (`robust/frontier/fmg.hpp`,
  `estimate/frontier/dls_weight.hpp`, `estimate/frontier/pairwise.hpp`).
  Header-path moves; needs forwarding shims where `r-package/` includes a path.
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
