# magmaan TODO

Remaining-work backlog. Current state, architecture, and contracts live in
[docs/architecture/roadmap.md](../architecture/roadmap.md); this file only tracks unfinished work.

Effort tags: **S** bounded docs/fixtures/wrapper cleanup · **M** focused
implementation or test slice · **L** new estimator plumbing or cross-module
semantics · **XL** statistical design/research track before implementation.

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
- **S/M.** Fix unbounded L-BFGS convergence diagnostics on the line-search
  salvage path: accepted last iterates can currently surface without a reliable
  solver iteration count, so benchmark scripts should avoid treating a reported
  zero as a real optimizer iteration count.
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
- **M/L.** Decide whether NLopt L-BFGS should replace LBFGS++ as the default
  scalar optimizer. This is not just a search/replace: first add a comparison
  tier showing `Backend::NloptLbfgs` matches the current LBFGS++ path on ML,
  complete-data LS, bounded ordinal LS, FIML/direct `optim::lbfgs` callers,
  and augmented-Lagrangian inner solves; document any differences in tolerance
  semantics (`gtol` vs NLopt `xtol_rel`), iteration/evaluation reporting, and
  bounded behavior. Only then decide whether the ordinary/default build should
  pay the NLopt dependency cost or keep NLopt as an optional cross-check.
- **S/M.** Consolidate optimizer/dependency license accounting before any
  public artifact: document the exact third-party optimizer libraries in use
  (currently LBFGS++ via FetchContent/system package, optional Ceres,
  optional NLopt, and vendored PORT — AMPL/ASL `cport/` plus Fermi-LAT's
  `drmnfb_routines.c`, both BSD-3, manifest in `third_party/port/README.md`),
  their versions, and their redistribution obligations.
- **M/L, after coverage exists.** Promote the Ceres preset into regular
  validation where relevant without making the default build pay the Ceres
  dependency cost.
- **S, only if timings justify it.** Experiment with opt-in precompiled headers
  for Eigen-heavy local builds; keep them disabled unless they improve
  changed-TU rebuilds without worsening no-op or full rebuild ergonomics.

## Ordinal/SNLLS research

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
`spec::build` desugars each composite into a Henseler-Ogasawara reflective
sub-model, weights and delta-method SEs are recovered post-fit, R-facing
partables are folded back to `<~` shape, and R exposes a `composite_weights()`
post-fit accessor.
Remaining:

- **L.** Lavaan parity validation: minimal oracle fixtures now exist under
  `tests/fixtures/composite/` for pure composite, composite plus common factor,
  and structural regression involving a composite. The diagnostic golden is
  wired but skipped because the current Henseler-Ogasawara expansion is not yet
  lavaan-native equivalent on point estimates, implied covariance, chi-square,
  SEs, standardization, and fit measures. Next implementation step: make the
  complete-data ML composite path match lavaan's native `<~` W-matrix
  semantics (or prove and implement an equivalent objective/parameterization),
  then unskip the golden. Multi-group composites are in scope only after the
  single-group ML slice is green and only if lavaan handles them cleanly,
  including `group.equal = "composite.weights"`.
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
