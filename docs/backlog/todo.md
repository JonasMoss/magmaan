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
  finalized `external/kline` corpus into an end-to-end parity test.

## API and R boundary

- **S/M.** Add or rename R wrappers only when the methods-developer workflow
  exposes a concrete gap in the staged API; the current `magmaan_core`,
  `magmaan_fit`, and post-fit wrapper surface is otherwise sufficient for the
  next R exploration pass.
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
- **M.** Track objective value, gradient norm, iteration count, wall time, and
  agreement with lavaan-backed estimates where applicable.
- **S/M.** Tighten the remaining Geiser GLS parity exceptions now documented
  in the parity-tier golden: manifest fixed.x path models need a resolved
  implied-moment comparison surface for exogenous observed moments, and
  `latent_ar_cross_lagged` needs either a stable same-basin optimizer recipe
  or a written alternate-optimum note.
- **S/M.** Extend the new Mplus SEM corpus beyond the v1 strict growth tranche.
  `external/mplus_sem` now retains 80 first-pass translations and the tracked
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
  complex matrix/constraint
  conversion for Little, promote additional Newsom continuous cases whose
  lavaan syntax uses parser features magmaan does not yet accept, and add
  ordinal/mixed parity checks once the desired categorical oracle surface is
  settled.
- **S/M.** Refine benchmark use of optimizer diagnostics now that fit results
  expose `optimizer_status` and final gradient norms. Benchmark scripts should
  distinguish clean convergence from line-search salvage or singular PORT
  convergence, and still avoid interpreting backend-specific missing iteration
  counts as real zero-iteration solves.
- **M.** Compare LBFGS, LBFGS-B, Ceres trust-region, Ceres dense BFGS, and
  SNLLS only on semantically appropriate cases; include shallow or
  Heywood-prone LS cases so bounds and conditioning stay visible.
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
- **M/L.** Decide whether NLopt L-BFGS should replace LBFGS++ as the default
  scalar optimizer. This is not just a search/replace: first add a comparison
  tier showing `Backend::NloptLbfgs` matches the current LBFGS++ path on ML,
  complete-data LS, bounded ordinal LS, FIML/direct `optim::lbfgs` callers,
  and augmented-Lagrangian inner solves; document any differences in tolerance
  semantics (`gtol` vs NLopt `xtol_rel`), iteration/evaluation reporting, and
  bounded behavior. Only then decide whether the ordinary/default build should
  pay the NLopt dependency cost or keep NLopt as an optional cross-check.
- **S, before shipping binary artifacts.** The repo now carries an MIT
  `LICENSE` that also notes the vendored BSD-3 PORT routines, which is
  sufficient for a source release — LBFGS++, Eigen, Ceres, NLopt, and
  nlohmann_json are fetched at build time, not redistributed. Before shipping a
  binary or packaged artifact, extend this to a full dependency-license
  manifest (LBFGS++, optional Ceres, optional NLopt, vendored PORT) with
  versions and redistribution obligations.
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
- **XL.** Design and prototype SNLLS for all-ordinal delta DWLS/WLS on
  polychoric moments. The likely scope is a Golub-Pereyra profile over free
  thresholds plus conditionally linear latent-response covariance pieces, with
  loadings/structural coefficients as the outer nonlinear block. This needs its
  own design note before implementation: classify threshold rows outside the
  current `ModelEvaluator::param_locations()` matrix-cell scheme, prove the
  delta residual/Jacobian is affine in the proposed profiled block, spell out
  equality-constraint compatibility, and gate against full ordinal LS plus
  lavaan-backed fixtures. Theta parameterization and mixed continuous/ordinal
  SNLLS stay out of this first scope unless the separability argument is made
  explicit.

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
- **M.** Move `auglag.hpp` and `reparameterize.hpp` to `optim/` (optimizer
  machinery, not estimators); `reparameterize.hpp` is coupled to the
  `estimate/constraints.hpp` move above.
- **S/M.** Settle whether `cfa_utils.hpp` belongs in `spec` or `model`;
  depends on the start-values decision below.
- **M.** Gather the five start-value producers (`start_values.hpp`) into an
  `estimate::starts` sub-namespace; `start_values.hpp` has 16 includers and
  `spec::Starts` is part of the lavaanified-model triple, so this needs care.
