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
  covers depth on real datasets. See `docs/roadmap.md` -> Testing and
  validation.
- Property and boundary tests should catch structural mistakes before they turn
  into hard-to-debug lavaan parity failures.
- Examples should demonstrate the explicit post-fit workflow rather than hide
  inference behind `magmaan()`.

Completed checks:

- [x] **M/L. Growth parity.** `demo_growth_linear` now uses the growth
  lavaanification defaults: observed-variable intercepts fixed at zero,
  latent growth-factor means free, and endogenous growth factors covaried
  under `auto_cov_y`. The fixture is hard-gated with `magmaan_aligned = true`.

Remaining work, in suggested order:

- None currently queued in this section.

Done when: representative supported estimator/model combinations have fixtures
or structural tests at the right level, and contributors can discover the
intended workflow from docs/examples rather than reverse engineering tests.

## 1. API, R boundary, and namespace cleanup

Intent: keep the public surface coherent while the staged API and R
methods-developer interface settle.

Contracts:

- Thin R wrappers should mirror C++ primitive signatures over time; fit-list
  helpers may remain as explicit convenience adapters.
- User-facing APIs should be staged and explicit: model/data construction,
  point estimation, and post-fit inference/reporting are separate choices.
- Primitive APIs should remain available for methods work without crowding the
  friendly R namespace.
- New C++ code should use the target namespaces directly: `parse`, `spec`,
  `model`, `data`, `estimate`, `inference`, `robust`, `measures`, `optim`,
  and `compat::lavaan`.

Remaining work, in suggested order:

- **M/L.** Formalize the staged C++ facade after the prototype settles. Keep it
  as value objects over the existing primitives (`Model`, `Data`, `Fit`,
  `Analysis`, `Summary`) rather than a second estimator implementation.
- **S/M.** Continue migrating R post-fit helpers from opaque fit-list unpacking
  toward explicit primitive-shaped entry points, keeping convenience aliases
  only where they do not obscure the contract.
- **M/L.** Design the Python binding surface around a friendly top-level API
  and a `magmaan.core` submodule for primitives, keeping names aligned with the
  C++ namespace layout where possible.
- **S/M.** Continue the staged-API examples. `observed_information_se.R` now
  covers complete-data ML observed-information SEs; Satorra-Bentler and ordinal
  DWLS/WLS robust reporting are in `constraints_and_satorra_bentler.R` and
  `ordinal_dwls_wls.R`. FIML with MLR-style robust reporting still needs an R
  example, blocked on exposing the C++ FIML MLR sandwich / Yuan-Bentler
  machinery through a `magmaan_core` binding — no FIML-robust entry point
  exists today.
- **M/L.** Land explicit primitive surfaces for post-fit quantities that are
  not currently produced by magmaan: standard SEs for FIML/LS, FIML SRMR, and
  estimator-appropriate LS/ordinal CFI/TLI/RMSEA/SRMR. Add lavaan parity gates
  as each surface becomes semantically defined.

Done when: new code naturally uses the target namespaces, friendly R/Python
users see a small staged API, and methods developers can still choose primitive
post-fit operations without depending on hidden fit-list unpacking.

## 2. Benchmarks and performance baselines

Intent: make refactors and backend choices measurable instead of anecdotal.

Contracts:

- Benchmarks are advisory local tooling, not a substitute for lavaan parity
  fixtures or correctness tests.
- `docs/benchmark_plan.md` is the full benchmarking design; this section tracks
  the actionable near-term slice.
- Compare only semantically appropriate backends for each estimator/model slice.

Remaining work, in suggested order:

- **S.** Keep the build-loop timings table in `docs/roadmap.md` (Local build
  workflow) current after major workflow changes: no-op build, touched/edited
  core and test TUs, `ctest`, `test-dev`, and the three R install modes.
- **S/M.** Extend benchmark case coverage beyond the current complete-data ML
  scaffold to FIML, ULS, GLS, WLS, ordinal DWLS/WLS, and mixed categorical
  models.
- **M.** Track objective value, gradient norm, iteration count, wall time, and
  agreement with lavaan-backed estimates where applicable.
- **M.** Compare LBFGS, LBFGS-B, Ceres, and SNLLS only on cases where each
  backend is semantically appropriate; include shallow or Heywood-prone LS
  cases so bounds and conditioning stay visible.
- **M/L, after benchmark coverage exists.** Promote the Ceres preset into
  regular validation where relevant without making the default build pay the
  Ceres dependency cost.
- **S, only if timings justify it.** Experiment with opt-in precompiled headers
  for Eigen-heavy local builds; keep them disabled unless they improve
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
  delta boundary.
- Nonlinear constraints, inequality constraints, and nonstandard active-bound
  inference stay outside the regular inference surface until their theory and
  reporting are explicit.

Remaining work, in suggested order:

- **M/L.** Extend covariance shrinkage to `MixedOrdinalStats`. Continuous
  `SampleStats` shrinkage already exists; mixed shrinkage needs the
  `src/data/ordinal.cpp` moment/NACOV rebuild factored into a shared helper so
  R and C++ can shrink consistently.
- **M/L.** Finish the theta ordinal compatibility slice beyond all-ordinal
  point estimates: mixed continuous/ordinal theta, theta robust inference,
  theta modification indices, and theta standardized-solution reporting.
- **XL, deferred.** Reopen nonlinear equality constraints, inequality
  constraints, or active-bound inference only with an explicit statistical
  design and reporting contract.

Done when: each unsupported statistical feature is either implemented with
oracle/fixture backing or remains clearly outside the public contract.

## 4. Composite models

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

- **S/M.** Document lavaan's current `<~` contract from the bundled reference
  and collect minimal oracle cases: pure composite, composite plus common
  factor, structural regression involving a composite, and multi-group only if
  lavaan supports it cleanly.
- **L, prerequisite.** Design where composites live in the lavaanified model
  triple and matrix representation, including weights, names, starts, partable
  rows, variance behavior, covariance behavior, and regressions.
- **M.** If support is accepted, update the grammar and parser from
  rejected-operator behavior to `<~` support with fixture-backed tests.
- **L.** Extend lavaan partable projection for `<~` rows, weights, labels,
  plabels, group equality labels, and `group.equal = "composite.weights"`.
- **L.** Implement matrix/implied-moment behavior for composite weights,
  composite variance handling, and mean-structure rows.
- **L.** Validate point estimates, implied covariance/mean, df, chi-square,
  SEs, standardization, and fit measures for the supported complete-data ML
  slice before exposing composites through R helpers.
- **S, after semantic fixtures are green.** Add benchmark cases.

Done when: `<~` no longer parses as a rejected operator for the supported
slice, magmaan partables and implied moments match lavaan composite examples,
and unsupported composite combinations fail with explicit errors.
