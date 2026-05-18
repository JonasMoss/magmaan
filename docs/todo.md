# magmaan TODO

Remaining-work backlog. Current state, architecture, and contracts live in
[docs/roadmap.md](roadmap.md); this file only tracks unfinished work.

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
[docs/benchmark_plan.md](benchmark_plan.md).

- **S.** Keep the build-loop timings table in `docs/roadmap.md` current after
  major workflow changes.
- **S/M.** Extend benchmark coverage beyond complete-data ML to FIML, ULS,
  GLS, WLS, ordinal DWLS/WLS, and mixed categorical models.
- **M.** Track objective value, gradient norm, iteration count, wall time, and
  agreement with lavaan-backed estimates where applicable.
- **M.** Compare LBFGS, LBFGS-B, Ceres, and SNLLS only on semantically
  appropriate cases; include shallow or Heywood-prone LS cases so bounds and
  conditioning stay visible.
- **M/L, after coverage exists.** Promote the Ceres preset into regular
  validation where relevant without making the default build pay the Ceres
  dependency cost.
- **S, only if timings justify it.** Experiment with opt-in precompiled headers
  for Eigen-heavy local builds; keep them disabled unless they improve
  changed-TU rebuilds without worsening no-op or full rebuild ergonomics.

## Composite models

The C++ core landed (uncommitted, unit-tested): `<~` parses as `Op::Composite`,
`spec::build` desugars each composite into a Henseler-Ogasawara reflective
sub-model, weights and delta-method SEs are recovered post-fit, and
`compat::lavaan::fold_composites` re-folds the partable back to `<~` shape.
Remaining:

- **M.** R-binding wiring: expose `<~` through the R model surface, call
  `fold_composites` so R sees `<~` rows rather than the H-O expansion, and add
  a `composite_weights` post-fit accessor.
- **L.** Lavaan parity validation: collect minimal oracle fixtures — pure
  composite, composite plus common factor, structural regression involving a
  composite — and gate point estimates, implied covariance, df, chi-square,
  SEs, standardization, and fit measures for the complete-data ML composite
  slice. Multi-group composites are in scope only if lavaan handles them
  cleanly, including `group.equal = "composite.weights"`.
- **S, after parity fixtures are green.** Add composite benchmark cases.

Deferred until the ML slice is lavaan-validated: ordinal composites, FIML/LS
composites, robust corrections for composites, and composite mean-structure
rows.

## lavaan tutorial parity

Close the gaps from the live checklist:
[docs/lavaan_tutorial_parity.md](lavaan_tutorial_parity.md).

- **M.** `lavResiduals()` z-statistics: the deterministic residual metrics
  landed (`measures::standardized_residuals`). The asymptotic-SE standardized
  residuals (`lavResiduals()$cov.z`) still need the residual-ACOV convention
  (`Γ_NT/N − Δ·vcov·Δᵀ`) pinned against a lavaan oracle before landing.
