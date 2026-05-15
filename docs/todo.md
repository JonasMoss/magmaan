# magmaan TODO

This is the active human-readable backlog for work that remains after the
current implementation state summarized in [docs/roadmap.md](roadmap.md). Keep
it focused on unfinished work, acceptance checks, and the next useful seams for
methods development.

## 0. Pairwise threshold/composite-likelihood model

Intent: add a direct pairwise likelihood/minimum-disparity path for threshold
data, starting with ordinary polychoric replication and only then expanding to
missing data, mixed data, and robust divergences.

- [x] Define the statistical contract separately from the current
  limited-information DWLS/WLS path: pairwise table likelihood or divergence
  over bivariate ordinal margins, with thresholds and latent correlations as
  the first target.
- [x] Factor the bivariate ordinal probability, score, and table bookkeeping
  out of the current polychoric implementation so pairwise likelihood, robust
  polychorics, and NACOV construction share one kernel.
- [x] Implement the smallest all-ordinal slice first: complete/listwise
  bivariate tables, fixed lavaan-style marginal thresholds, and pairwise rho
  estimates that reproduce current ML polychorics.
- [x] Add an explicit all-ordinal `PairwiseOrdinalStats` wrapper over
  `OrdinalStats` with pair diagnostics, lower-triangle pair labels,
  adjusted-count reporting, minimum eigenvalue diagnostics, and
  bitwise-equivalent NACOV/DWLS/WLS moment output.
- [x] Add the first joint pairwise threshold/rho estimator at the bivariate
  kernel level: pair-local nuisance thresholds plus rho for one complete
  ordinal table, separate from the shared-moment `OrdinalStats` path.
- [ ] Decide how joint pairwise threshold/rho estimation integrates with SEM
  moments: pair-local nuisance thresholds for composite likelihood, shared
  threshold moments, or both as separate explicit APIs.
- [ ] Define missing-data semantics before coding: observed-pair composite
  likelihood may not need EM, but shared-threshold/multivariate MAR handling
  needs a documented likelihood target and scaling convention.
- [ ] Add mixed continuous/ordinal pair kernels after all-ordinal works:
  polyserial pairs, continuous-continuous normal pairs, and consistent moment
  ordering with current `MixedOrdinalStats`.
- [ ] Decide whether normal-data pairwise likelihood is a real supported
  estimator or only a benchmark/sanity check against complete-data ML/FIML.
- [x] Expose complete/listwise all-ordinal casewise influence and Gamma for
  the threshold-plus-polychoric moment vector so downstream DWLS, WLS, robust
  SEs, and scaled tests can reuse the existing weighted-moment sandwich path
  without recomputing sample-stat internals.
- [ ] Extend pairwise influence/Gamma exposure to future missing-data and
  mixed continuous/ordinal pairwise paths once their likelihood targets and
  moment ordering are fixed.
- [x] Extend complete-data pair diagnostics beyond labels, convergence,
  boundary, adjusted-count, and minimum-eigenvalue fields: include fitted
  expected tables, residual tables, zero missingness counts, and explicit
  no-ridge/no-shrinkage repair fields.
- [ ] Add lavaan-backed or internally cross-checked fixtures for complete
  all-ordinal polychorics first, then mixed data, then missing-data scenarios.

Done when: the pairwise path reproduces existing ML polychorics in its default
slice, has explicit missing-data semantics, and returns a moment/Gamma bundle
that downstream ordinal DWLS/WLS code can consume without special SEM-side
logic.

## 1. Validation and lavaan comparison

Intent: keep validation fixture-first and make parity failures cheap to
diagnose.

- [ ] Add fixture cases for scalar-invariance latent mean rescaling once the
  bounded optimizer reliably fits the needed mean-structure models.

Done when: representative supported estimator/model combinations have
checked-in lavaan-backed fixtures, clear tolerances, and no hidden fixture-free
parity claims.

## 2. R and public API polish

Intent: make the methods-developer interface reasonable without turning the R
package into a second SEM implementation.

- [ ] Keep ordinal R documentation current around `model_spec()` with
  `ordered` and `parameterization = "delta"`, `data_ordinal_stats_from_df()`,
  `fit_dwls_ordinal()`, and `fit_wls_ordinal()`.

Done when: a methods developer can fit supported complete, FIML, LS, and
ordinal/mixed models from R through thin wrappers while still choosing
post-fit inference explicitly.

## 3. Public namespace and header cleanup

Intent: finish the namespace transition without breaking compatibility
unnecessarily.

- [ ] After the transition window, remove the old `fit/*` and `partable/*`
  compatibility shims.

Done when: new code naturally uses `spec`, `lavaan`, `estimate`, `optim`, `nt`,
`gls`, and `data`, while old include paths still have a tested transition path.

## 4. Benchmarks and performance baselines

Intent: make refactors and backend choices measurable instead of anecdotal.

- [x] Seed an R-first benchmark scaffold with a case manifest, per-case model
  syntax/source metadata, ignored local data/result caches, and lavaan
  references for the initial built-in complete-data cases.
- [ ] Add benchmark fixtures for representative complete-data ML, FIML, ULS,
  GLS, WLS, ordinal DWLS/WLS, and mixed categorical models.
- [ ] Compare LBFGS, LBFGS-B, Ceres, and SNLLS only where each backend is
  semantically appropriate.
- [ ] Track objective value, gradient norm, iteration count, wall time, and
  agreement with lavaan-backed estimates where applicable.
- [ ] Promote the Ceres preset into regular validation where relevant without
  making the default build pay the Ceres dependency cost.
- [ ] Include shallow or Heywood-prone LS cases so lower bounds and backend
  conditioning stay visible.

Done when: backend recommendations and performance-sensitive refactors can be
checked against repeatable local benchmark scenarios.

## 5. Larger and more principled tests

Intent: complement narrow golden fixtures with broader tests that catch
structural mistakes.

- [ ] Expand property tests around Jacobians, moment-vector ordering, block
  stacking, equality constraints, and group weighting.
- [ ] Add stress cases for threshold-heavy ordinal models with sparse or
  near-empty categories; empty categories should remain hard errors.
- [ ] Add multi-group LS weighting and equality-constraint cases across
  continuous and ordinal estimators.
- [ ] Add mean-structure LS cases that exercise Ceres and SNLLS backends.
- [ ] Add complete/listwise boundary tests for mixed categorical sample stats.

Done when: new estimator work fails fast on structural invariants before it
drifts into hard-to-debug lavaan parity failures.

## 6. Remaining statistical gaps

Intent: keep unsupported statistical work explicit so it is not mistaken for a
polish task.

- [ ] Add explicit small-sample stabilizer primitives instead of string-mode
  knobs. Current code already has `Bounds` in full free-parameter space,
  automatic lower bounds for free variance diagonals, bounded ML/LS optimizer
  paths, and ordinal/mixed bounded DWLS/WLS fits. The missing work is a set of
  named C++ builders for published bounded-estimation recipes: variance-only
  bounds, wider/literature bounds for residual and latent variances, loading
  bounds when statistically justified, start projection into the chosen box,
  active-bound diagnostics, and fixtures for Heywood/small-N cases across
  complete ML, continuous LS, and ordinal/mixed LS.
- [ ] Add covariance shrinkage as explicit sample-moment transformations, not
  estimator flags. Current continuous `SampleStats` carries empirical
  covariances, while ordinal/mixed stats already build lavaan-style thresholds,
  polychorics, polyserials, NACOV, and DWLS/WLS weights. Add separate
  ridge-covariance and model-based shrinkage target builders that return new
  `SampleStats`/`MixedOrdinalStats`-compatible objects plus diagnostics
  (target, intensity, eigenvalues, repaired dimensions), then validate
  convergence and estimate movement on small-sample and nearly singular
  covariance/correlation cases without changing lavaan-compatible defaults.
- [ ] Add DLS and empirical-Bayes DLS as weight-matrix builders layered on the
  existing LS discrepancy surface. `gls::WLS` already consumes explicit full
  weights and the weighted-moment sandwich path accepts arbitrary block
  weights/NACOV matrices; the missing piece is constructing and validating the
  DLS weight matrices from normal-theory and ADF/Gamma components, including
  ridge/PSD repairs, block ordering, diagonal variants if useful, and robust
  reporting semantics for continuous and ordinal/mixed moment stacks.
- [ ] Add theta-parameterization support for ordinal models as a separate
  compatibility slice. The public R/C++ boundary now accepts the
  parameterization distinction and fails explicitly for `theta`; actual
  theta fitting still needs the separate lavaan-backed compatibility slice.
- [ ] Add absent-row generation and standardized EPC for modification indices
  only after the raw fixed-row/equality-release contract is fully validated.
- [ ] Keep nonlinear equality constraints, inequality constraints, and
  nonstandard active-bound inference out of the regular inference surface
  until their theory and reporting are explicit.

Done when: each unsupported statistical feature is either implemented with
oracle/fixture backing or remains clearly outside the public contract.

## 7. Documentation and examples

Intent: make the codebase easier to steer for future methods work.

- [ ] Keep `docs/roadmap.md` current as the state and architecture summary.
- [ ] Keep this file limited to remaining work; remove or rewrite completed
  checklist items when milestones land.
- [ ] Add examples that show the intended explicit post-fit workflow:
  estimation first, then optional SEs, robust tests, fit measures, defined
  parameters, and nested tests.
- [ ] Document public fixed.x and missing-data boundaries where users are most
  likely to encounter them.
- [ ] Document benchmark usage once benchmark scenarios exist.

Done when: a new contributor can read the roadmap for context, this TODO for
remaining work, and examples for the intended workflow without reverse
engineering current state from tests.

## 8. Alternative robust polychoric and mixed estimators

Intent: turn the planning notes in `resources/alternative_estimators/` into
incremental, testable robust polychoric estimators without changing default
lavaan-compatible behavior.

- [ ] Add an enum-backed h-score API in `magmaan::data` for `ml`,
  `wma_hard_cap`, `smooth_cap`, and `exp_cap`, with tests for `h`, `dh`, and
  objective contributions where available.
- [ ] Preserve current behavior as the default: `ml` with lavaan-style
  marginal thresholds must reproduce existing ordinal fixtures to current
  tolerances.
- [ ] Implement experimental fixed-threshold h-weighted rho estimation for one
  bivariate table, returning rho, convergence status, residuals, and per-cell
  weights.
- [ ] Validate fixed-threshold behavior with constructed contaminated tables:
  hard/smooth caps should move less than ML under inflated low-probability
  cells, while `hard_cap(k = Inf)` matches ML.
- [ ] Implement full pair-local WMA/h-score estimation with thresholds and rho
  estimated jointly; validate against robcat/WMA or independent reference
  calculations for selected bivariate tables.
- [ ] Add casewise influence and sandwich/Gamma calculations for robust
  pairwise moments; document the hard-cap kink convention and scaling.
- [ ] Decide and implement the first SEM integration mode: lavaan-style shared
  marginal thresholds plus robust rhos as an experimental Option A, before
  attempting shared-threshold composite h-score estimation.
- [ ] Handle indefinite robust correlation matrices explicitly: report minimum
  eigenvalues and optional ridge/shrinkage diagnostics rather than silently
  projecting by default.
- [ ] Implement density power divergence as the primary non-h-score comparator,
  with ML recovered as `alpha -> 0` and tests over a small alpha grid.
- [ ] Implement Hellinger and Huberized residual fitting only as experimental
  comparators; keep one-sided and symmetric Huber options clearly named.
- [ ] Keep mixed continuous/ordinal robust estimators behind the all-ordinal
  robust Gamma milestone; polyserial robustness should not advance before the
  all-ordinal influence path is stable.
- [ ] Expose only predefined robust methods through R controls; keep arbitrary
  C++ h-functions internal until there is a concrete methods use case.

Done when: robust polychoric alternatives are selectable, default ML fixtures
are unchanged, diagnostics make robustness visible, and at least one robust
method has a Gamma path usable by ordinal DWLS/WLS robust reporting.

## 9. Composite models

Intent: decide whether magmaan supports lavaan's new composite-variable
semantics for `<~`, then implement the smallest lavaan-backed slice without
confusing composites with ordinary latent factors.

- [ ] Document lavaan's current `<~` contract from the bundled reference:
  composites are weighted linear combinations of composite indicators; lavaan
  0.6-20+ treats them specially, while the old fallback rewrote `f <~ rhs` as
  `f =~ 0; f ~~ 0*f; f ~ rhs`.
- [ ] Collect minimal lavaan oracle cases before implementation: one pure
  composite, one model mixing composites and common factors, one structural
  regression involving a composite, and one multi-group case if lavaan
  supports it cleanly.
- [ ] Decide where composites live in the lavaanified model triple:
  `LatentStructure` must distinguish composite variables and composite
  indicators; `LatentNames` must preserve `<~` rows; `Starts` must carry weight
  starts without treating them as ordinary loadings.
- [ ] Update `docs/grammar/grammar.ebnf` first if `<~` becomes supported, then
  parser comments/tests, replacing the current rejected-operator behavior.
- [ ] Extend lavaan partable projection so `<~` rows, free/fixed weights,
  labels, plabels, group equality labels, and `group.equal =
  "composite.weights"` round-trip against lavaan.
- [ ] Extend matrix representation with lavaan's composite handling rather than
  pretending composites are reflective factors; identify the needed `WMAT`,
  variance, covariance, and regression matrix behavior.
- [ ] Implement composite variance handling, including the lavaan behavior that
  fixes/sets composite total or residual variances from the composite weights
  and indicator covariance structure.
- [ ] Implement mean-structure behavior explicitly: composite means/intercepts
  should follow the weighted indicator means, and fixtures should pin when mean
  rows appear in `parTable()`.
- [ ] Validate point estimates, implied covariance/mean, df, chi-square, SEs,
  standardization, and fit measures only for the supported complete-data ML
  slice before exposing composites through R helpers.
- [ ] Keep ordinal, FIML missing-data, robust corrections, and LS composite
  support out of scope until the complete-data ML composite contract is stable.
- [ ] Add benchmark cases only after the semantic fixture suite is green.

Done when: `<~` no longer parses as a rejected operator for the supported
slice, magmaan partables and implied moments match lavaan composite examples,
and unsupported composite combinations fail with explicit errors.
