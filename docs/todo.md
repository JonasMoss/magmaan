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

## 0. Pairwise threshold/composite-likelihood model

Intent: add a direct pairwise likelihood/minimum-disparity path for threshold
data, starting with ordinary polychoric replication and only then expanding to
missing data, mixed data, and robust divergences.

Contracts:

- The lavaan-compatible SEM-facing ordinal path remains shared marginal
  thresholds plus polychoric/polyserial/covariance moments consumed by
  DWLS/WLS. `PairwiseOrdinalStats` is the diagnostic, influence, and Gamma
  wrapper for that path; it must stay bitwise-equivalent to `OrdinalStats`
  where the moment outputs overlap.
- Joint bivariate threshold/rho ML is a pair-local primitive for diagnostics,
  robust-pair experiments, and future composite likelihood. It does not feed
  the current shared-threshold SEM moment vector because pair-local thresholds
  would duplicate each variable's nuisance parameters across pairs.
- Missing ordinal pair wrappers use observed-pair counts only. SEM-level
  missing ordinal estimation is not supported until a likelihood target and
  scaling convention are explicit. The first missing-data target should be an
  observed-pair composite-likelihood path, not an implicit MAR/FIML claim.
- Shared-threshold multivariate missing ordinal modeling is out of scope for
  this pairwise milestone. If reopened later, it needs a separate design note
  covering the likelihood target, threshold sharing rules, and scaling before
  implementation.
- Continuous normal pair likelihood is a mixed-pair primitive and
  benchmark/sanity check against complete-data ML/FIML, not a supported
  standalone SEM estimator.

Remaining work, in suggested order:

- [x] **L, prerequisite.** Design the SEM-level pairwise composite-likelihood
  API as a separate estimator from lavaan-compatible `OrdinalStats`/DWLS.
  Specify scaling, per-pair sample-size weighting, SEM-implied bivariate margin
  mapping, boundary diagnostics, and reported chi-square/df behavior. Initial
  surface: `estimate::pairwise_ordinal_composite_objective()` evaluates the
  shared-threshold all-ordinal objective, supports observed-pair-count or
  equal-pair weighting, exposes boundary diagnostics, and explicitly reports no
  chi-square/df until a calibrated composite-likelihood test exists.
- [x] **M/L, depends on API.** Add a complete/listwise all-ordinal
  composite-likelihood prototype using the existing complete bivariate joint ML
  kernel and diagnostics comparable to the current polychoric/DWLS path.
  Initial surface: `estimate::pairwise_ordinal_joint_composite_objective()`
  consumes `PairwiseOrdinalStats`, refits each complete pair with joint
  threshold/rho ML, preserves the same weighting/scaling options, and keeps
  chi-square/df unreported until a calibrated composite test is designed.
- [x] **L, harder than complete/listwise.** Add an observed-pair all-ordinal
  prototype behind the same API. Preserve per-pair `n_obs`/`n_missing`; reject
  all-missing pairs and empty marginal categories; document that this is
  pairwise observed-data likelihood, not multivariate MAR ordinal FIML.
  Initial surface:
  `estimate::pairwise_ordinal_observed_joint_composite_objective()` takes raw
  ordinal blocks with `NaN` missingness and declared level counts, fits each
  pair on its observed cases, and returns the shared composite result shape.
- [x] **M, after each implementation slice.** Add explicit pairwise
  diagnostics fixtures: current complete all-ordinal polychoric wrappers first,
  mixed pair primitives second, complete/listwise composite-likelihood
  prototypes third, and observed-pair scenarios only after that target is fixed.
  Initial fixture: `tests/fixtures/pairwise/0001_pairwise_diagnostics.json`
  covers all four slices with a golden fixture test.
- [ ] **L, follow-on inference work.** Extend pairwise influence/Gamma exposure
  to complete/listwise and missing-data all-ordinal composite-likelihood paths,
  then to mixed continuous/ordinal pairwise paths once their likelihood targets
  and moment ordering are fixed.

Done when: the pairwise path reproduces existing ML polychorics in its default
slice, has an explicit composite-likelihood API for pair-local SEM work, and
documents missing-data semantics without implying unsupported multivariate
ordinal FIML behavior.

## 1. Validation, tests, and examples

Intent: keep validation fixture-first, broaden structural test coverage, and
make supported workflows legible without turning docs into a second roadmap.

Contracts:

- Fixture-backed parity remains the bar for supported estimator/model slices.
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
- [ ] **M.** Expand property tests around Jacobians, moment-vector ordering,
  block stacking, equality constraints, and group weighting.
- [ ] **M.** Add ordinal/mixed boundary fixtures: threshold-heavy sparse or
  near-empty categories, complete/listwise mixed categorical sample stats, and
  explicit empty-category hard errors.
- [ ] **M/L.** Add multi-group LS weighting and equality-constraint cases
  across continuous and ordinal estimators, plus mean-structure LS cases that
  exercise Ceres and SNLLS where semantically appropriate.
- [ ] **S, blocked on optimizer behavior.** Add scalar-invariance latent mean
  rescaling fixtures once the bounded optimizer reliably fits the needed
  mean-structure models.

Done when: representative supported estimator/model combinations have fixtures
or structural tests at the right level, and contributors can discover the
intended workflow from docs/examples rather than reverse engineering tests.

## 2. API, R boundary, and namespace cleanup

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

## 3. Benchmarks and performance baselines

Intent: make refactors and backend choices measurable instead of anecdotal.

Contracts:

- Benchmarks are advisory local tooling, not a substitute for lavaan parity
  fixtures or correctness tests.
- Compare only semantically appropriate backends for each estimator/model slice.

Remaining work, in suggested order:

- [ ] **S/M.** Add benchmark fixtures for representative complete-data ML,
  FIML, ULS, GLS, WLS, ordinal DWLS/WLS, and mixed categorical models.
- [ ] **M.** Track objective value, gradient norm, iteration count, wall time,
  and agreement with lavaan-backed estimates where applicable.
- [ ] **M.** Compare LBFGS, LBFGS-B, Ceres, and SNLLS only on cases where each
  backend is semantically appropriate; include shallow or Heywood-prone LS
  cases so bounds and conditioning stay visible.
- [ ] **M/L, after benchmark coverage exists.** Promote the Ceres preset into
  regular validation where relevant without making the default build pay the
  Ceres dependency cost.
- [ ] **S, after scenarios exist.** Document benchmark usage.

Done when: backend recommendations and performance-sensitive refactors can be
checked against repeatable local benchmark scenarios.

## 4. Statistical feature backlog

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

## 5. Robust pairwise estimators

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

- [ ] **M.** Add an enum-backed h-score API in `magmaan::data` for `ml`,
  `wma_hard_cap`, `smooth_cap`, and `exp_cap`, with focused tests for score
  values, derivatives, and objective contributions where available.
- [ ] **M.** Implement experimental fixed-threshold h-weighted rho estimation
  for one bivariate table, returning rho, convergence status, residuals, and
  per-cell weights.
- [ ] **S/M.** Validate fixed-threshold robustness with constructed
  contaminated tables, including the `hard_cap(k = Inf)` equals ML limit.
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

## 6. Composite models

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
