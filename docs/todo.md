# magmaan TODO

This is the active human-readable backlog for work that remains after the
current implementation state summarized in [docs/roadmap.md](roadmap.md). Keep
it focused on unfinished work, acceptance checks, and the next useful paths for
methods development.

Effort is approximate:

- **S**: bounded docs, fixtures, or wrapper cleanup.
- **M**: focused implementation or test slice.
- **L**: new estimator plumbing, cross-module semantics, or inference work.
- **XL**: separate statistical design/research track before implementation.

Global contracts:

- `docs/roadmap.md` is the state and architecture summary; this file is only
  the remaining-work backlog.
- Lavaan-backed support claims need checked-in fixtures and tolerances. CI
  never depends on running R.
- R bindings stay methods-developer oriented: estimate first, then explicit
  post-fit inference, robust tests, fit measures, defined parameters, and
  nested tests.
- Unsupported statistical combinations should fail explicitly rather than
  falling through to nearby lavaan-compatible behavior.
- Local iteration should default to the non-sanitized `fast` C++ preset; run
  sanitizer, optimized, Ceres, and R workflows explicitly when they are relevant
  validation gates.

## 0. Validation, tests, and examples

Intent: keep validation fixture-first, broaden structural test coverage, and
make supported workflows legible without turning docs into a second roadmap.

Contracts:

- Fixture-backed parity remains the bar for supported estimator/model slices.
- The synthetic corpus goldens cover breadth; the lavaan-parity layer
  (`tests/golden/lavaan_parity_golden_test.cpp`, `tests/fixtures/parity/`)
  covers depth on real datasets. See `docs/roadmap.md` → Testing and validation.
- Property and boundary tests should catch structural mistakes before they turn
  into hard-to-debug lavaan parity failures.
- Examples should demonstrate the explicit post-fit workflow rather than hide
  inference behind `magmaan()`.

Remaining work, in suggested order:

- [ ] **S.** Keep ordinal R documentation current around `model_spec()` with
  `ordered` and `parameterization = "delta"`, ordinal sample-stat builders, and
  ordinal DWLS/WLS fitting wrappers.
- [ ] **S/M.** Add examples for the intended workflow: estimation first, then
  optional SEs, robust tests, fit measures, defined parameters, and nested
  tests. Include public fixed.x and missing-data boundaries where users are
  likely to hit them.
- [x] **M.** Expand property tests around Jacobians, moment-vector ordering,
  block stacking, equality constraints, and group weighting.
  (`tests/unit/property_test.cpp` — structural invariants independent of any
  one lavaan fixture.)
- [ ] **M.** Add ordinal/mixed boundary fixtures. A threshold-heavy 6-category
  case is in (`0011_sixcat_threshold_heavy_cfa`, 20 thresholds); near-empty
  categories (`0005`) and empty-category hard errors (ordinal_test.cpp) are
  already covered. Remaining: a lavaan-backed complete/listwise mixed
  categorical case — blocked on the mixed NACOV/weight path, which currently
  drifts past the documented loose tolerance for a sparse 4-category indicator.
- [ ] **M/L.** Add multi-group LS weighting and equality-constraint cases
  across continuous and ordinal estimators, plus mean-structure LS cases that
  exercise Ceres and SNLLS where semantically appropriate. Continuous
  multi-group LS parity has landed — weighting (`hs_3factor_ls_mg_configural`)
  and cross-group equal loadings (`hs_3factor_ls_mg_metric`), ULS/GLS/WLS, HS
  by school. `fit_bounded`'s LS path K-reparameterizes *every* equality
  constraint (`θ = θ₀ + K·α`, optimizing the reduced bounded problem) —
  pure-merge with mapped box bounds, general-linear (`a == 2*b + c`) with an
  unbounded α and a post-fit bound check — so the old 1e10 penalty residual
  (ill-conditioned enough to hang LBFGS-B) is gone entirely; general-linear LS
  is gated by the `0007_general_linear_hs` golden. `browne_residual_nt`
  K-reduces its Jacobian so constrained ULS chi-square is correct. Remaining:
  ordinal multi-group LS, and mean-structure LS exercising Ceres/SNLLS.
- [ ] **S.** Add scalar-invariance latent mean rescaling fixtures. The
  line-search salvage (`docs/convergence_diagnostics.md`) unblocked
  mean-structure ML convergence on the unbounded path; confirm the bounded
  optimizer also fits these models before relying on it there.
- [x] **M.** Extend the lavaan-parity golden layer
  (`tests/golden/lavaan_parity_golden_test.cpp`): four estimator-family
  `TEST_CASE`s now gate real data — ML, FIML (`bfi_fiml`, raw bfi with genuine
  missingness), continuous LS (`hs_3factor_ls`, ULS/GLS/WLS), and ordinal
  DWLS/WLS (`bfi_ordinal_dwls`). Each gates every quantity lavaan exposes that
  magmaan produces; see the per-family "not gated" table in
  `tests/fixtures/parity/README.md`. Shared lavaan→JSON regen helpers were
  factored into `benchmarks/r/fixture_json.R`.
- [ ] **S/M.** Deepen the parity layer's non-ML coverage. `bfi_fiml` is a
  2-factor 10-item model — add a full 5-factor bfi FIML case once convergence
  on the wider model is confirmed. The parity test also leaves FIML/LS
  standard SEs and LS/ordinal CFI/TLI/RMSEA/SRMR ungated, because magmaan has
  no standard-SE path for FIML/LS and no LS/ordinal fit-measure path; extend
  the gate as those post-fit surfaces land.
- [ ] **M/L.** Bring `demo_growth_linear` to lavaan parity. magmaan has no
  `growth()` equivalent, so its free set does not align with lavaan's
  (`magmaan_aligned = false` in the fixture); the case is committed as a soft
  known gap and auto-promotes to a hard gate once the parameterization matches.

Done when: representative supported estimator/model combinations have fixtures
or structural tests at the right level, and contributors can discover the
intended workflow from docs/examples rather than reverse engineering tests.

## 1. API, R boundary, and namespace cleanup

Intent: keep the public surface coherent while the namespace transition and R
methods-developer interface settle.

Contracts:

- Thin R wrappers should mirror C++ primitive signatures over time; fit-list
  helpers may remain as explicit convenience adapters.
- New C++ code should use `spec`, `lavaan`, `estimate`, `optim`, `nt`, `gls`,
  and `data` directly.

Remaining work, in suggested order:

- [ ] **S/M.** Continue migrating R post-fit helpers from opaque fit-list
  unpacking toward explicit primitive-shaped entry points, keeping convenience
  aliases only where they do not obscure the contract.
- [ ] **M, later transition cleanup.** After the compatibility window, remove
  the old `fit/*` and `partable/*` shim headers and their transition tests.

Done when: new code naturally uses the target namespaces and R users can choose
primitive post-fit operations without depending on hidden fit-list unpacking.

## 2. Benchmarks and performance baselines

Intent: make refactors and backend choices measurable instead of anecdotal.

Contracts:

- Benchmarks are advisory local tooling, not a substitute for lavaan parity
  fixtures or correctness tests.
- `docs/benchmark_plan.md` is the full benchmarking design; this section tracks
  the actionable near-term slice.
- Compare only semantically appropriate backends for each estimator/model slice.

Remaining work, in suggested order:

- [ ] **S.** Record local build-loop timings after major workflow changes:
  no-op build, touched core TU, touched test TU, labeled `ctest`, `test-dev`,
  and the three R install modes.
- [ ] **S/M.** The benchmark scaffold exists — `benchmarks/cases.yml`, the
  `benchmarks/r/` harness, and five active complete-data ML cases with
  checked-in `reference_lavaan.json`. Extend case coverage to FIML, ULS, GLS,
  WLS, ordinal DWLS/WLS, and mixed categorical models.
- [ ] **M.** Track objective value, gradient norm, iteration count, wall time,
  and agreement with lavaan-backed estimates where applicable.
- [ ] **M.** Compare LBFGS, LBFGS-B, Ceres, and SNLLS only on cases where each
  backend is semantically appropriate; include shallow or Heywood-prone LS
  cases so bounds and conditioning stay visible.
- [ ] **M/L, after benchmark coverage exists.** Promote the Ceres preset into
  regular validation where relevant without making the default build pay the
  Ceres dependency cost.
- [ ] **S, only if timings justify it.** Experiment with opt-in precompiled
  headers for Eigen-heavy local builds; keep them disabled unless they improve
  changed-TU rebuilds without worsening no-op or full rebuild ergonomics.

Done when: backend recommendations and performance-sensitive refactors can be
checked against repeatable local benchmark scenarios.

## 3. Statistical feature backlog

Intent: keep unsupported statistical work explicit so it is not mistaken for
polish.

Contracts:

- Small-sample stabilizers, shrinkage, and DLS should be explicit builders or
  transformations, not string-mode knobs that alter lavaan-compatible defaults.
- Theta ordinal support is a separate lavaan compatibility slice from the
  current delta boundary.
- Nonlinear constraints, inequality constraints, and nonstandard active-bound
  inference stay outside the regular inference surface until their theory and
  reporting are explicit.

Remaining work, in suggested order:

- [ ] **M/L.** Add named small-sample stabilizer builders: variance-only bounds,
  wider/literature variance bounds, statistically justified loading bounds,
  start projection into the chosen box, active-bound diagnostics, and
  Heywood/small-N fixtures across representative ML/LS/ordinal slices.
- [ ] **M/L.** Add covariance shrinkage as sample-moment transformations that
  return `SampleStats` or `MixedOrdinalStats`-compatible objects plus repair
  diagnostics, without changing lavaan-compatible defaults.
- [ ] **L.** Add DLS and empirical-Bayes DLS as weight-matrix builders layered
  on the existing LS discrepancy and weighted-moment sandwich surfaces.
- [ ] **L.** Add theta-parameterization support for ordinal models as a
  separate lavaan-backed compatibility slice.
- [ ] **M.** Add absent-row generation and standardized EPC for modification
  indices after the raw fixed-row/equality-release contract remains stable.
- [ ] **XL, deferred.** Reopen nonlinear equality constraints, inequality
  constraints, or active-bound inference only with an explicit statistical
  design and reporting contract.

Done when: each unsupported statistical feature is either implemented with
oracle/fixture backing or remains clearly outside the public contract.

## 4. Robust pairwise estimators

Intent: turn the planning notes in `resources/alternative_estimators/` into
incremental robust polychoric estimators without changing default
lavaan-compatible behavior.

Contracts:

- Default ordinal behavior remains ML with lavaan-style marginal thresholds and
  must continue to reproduce existing ordinal fixtures.
- Robust methods are experimental until diagnostics, influence/Gamma, and at
  least one SEM integration mode are fixture-backed.
- Mixed continuous/ordinal robust estimators wait behind the all-ordinal robust
  Gamma milestone.
- R should expose only predefined robust methods; arbitrary C++ h-functions
  remain internal until a concrete methods use case exists.

Remaining work, in suggested order:

- [x] **M.** Add an enum-backed h-score API in `magmaan::data` for `ml`,
  `wma_hard_cap`, `smooth_cap`, and `exp_cap`, with focused tests for score
  values, derivatives, and objective contributions where available.
  (`include/magmaan/data/h_score.hpp`, `tests/unit/ordinal_test.cpp`.)
- [x] **M.** Implement experimental fixed-threshold h-weighted rho estimation
  for one bivariate table, returning rho, convergence status, residuals, and
  per-cell weights.
  (`include/magmaan/data/pairwise_ordinal.hpp`,
  `src/data/pairwise_ordinal.cpp`, `tests/unit/ordinal_test.cpp`.)
- [x] **S/M.** Validate fixed-threshold robustness with constructed
  contaminated tables, including the `hard_cap(k = Inf)` equals ML limit.
  (`tests/unit/ordinal_test.cpp` pins the ML and hard-cap-infinity limits and
  a discordant-corner contamination case.)
- [ ] **L.** Implement full pair-local WMA/h-score estimation with thresholds
  and rho estimated jointly, validated against robcat/WMA or independent
  reference calculations for selected bivariate tables.
- [ ] **L.** Add casewise influence and sandwich/Gamma calculations for robust
  pairwise moments, including the hard-cap kink convention and scaling.
- [ ] **M/L.** Add the first SEM integration mode: shared lavaan-style marginal
  thresholds plus robust rhos as experimental Option A before attempting
  shared-threshold composite h-score estimation.
- [ ] **M.** Handle indefinite robust correlation matrices explicitly with
  minimum eigenvalue and optional ridge/shrinkage diagnostics.
- [ ] **L, comparator track.** Implement density power divergence as the main
  non-h-score comparator; keep Hellinger and Huberized residual fitting as
  lower-priority experimental comparators.

Done when: robust polychoric alternatives are selectable, default ML fixtures
are unchanged, diagnostics make robustness visible, and at least one robust
method has a Gamma path usable by ordinal DWLS/WLS robust reporting.

## 5. Composite models

Intent: decide whether magmaan supports lavaan's new composite-variable
semantics for `<~`, then implement the smallest lavaan-backed complete-data ML
slice without confusing composites with ordinary latent factors.

Contracts:

- Composites are not ordinary reflective factors. If supported, they need a
  distinct representation in the lavaanified model triple and matrix layer.
- Grammar changes start in `docs/grammar/grammar.ebnf`, followed by parser
  comments/tests and oracle fixtures.
- Ordinal, FIML missing-data, robust corrections, and LS composite support stay
  out of scope until the complete-data ML composite contract is stable.

Remaining work, in suggested order:

- [ ] **S/M.** Document lavaan's current `<~` contract from the bundled
  reference and collect minimal oracle cases: pure composite, composite plus
  common factor, structural regression involving a composite, and multi-group
  only if lavaan supports it cleanly.
- [ ] **L, prerequisite.** Design where composites live in the lavaanified model
  triple and matrix representation, including weights, names, starts, partable
  rows, variance behavior, covariance behavior, and regressions.
- [ ] **M.** If support is accepted, update the grammar and parser from
  rejected-operator behavior to `<~` support with fixture-backed tests.
- [ ] **L.** Extend lavaan partable projection for `<~` rows, weights, labels,
  plabels, group equality labels, and `group.equal = "composite.weights"`.
- [ ] **L.** Implement matrix/implied-moment behavior for composite weights,
  composite variance handling, and mean-structure rows.
- [ ] **L.** Validate point estimates, implied covariance/mean, df,
  chi-square, SEs, standardization, and fit measures for the supported
  complete-data ML slice before exposing composites through R helpers.
- [ ] **S, after semantic fixtures are green.** Add benchmark cases.

Done when: `<~` no longer parses as a rejected operator for the supported
slice, magmaan partables and implied moments match lavaan composite examples,
and unsupported composite combinations fail with explicit errors.
