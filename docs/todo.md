# magmaan TODO

This is the active human-readable backlog for work that remains after the
current implementation state summarized in [docs/roadmap.md](roadmap.md). Keep
it focused on unfinished work, acceptance checks, and the next useful seams for
methods development.

## 0. Pairwise threshold/composite-likelihood model

Intent: add a direct pairwise likelihood/minimum-disparity path for threshold
data, starting with ordinary polychoric replication and only then expanding to
missing data, mixed data, and robust divergences.

- [ ] Define the statistical contract separately from the current
  limited-information DWLS/WLS path: pairwise table likelihood or divergence
  over bivariate ordinal margins, with thresholds and latent correlations as
  the first target.
- [ ] Factor the bivariate ordinal probability, score, and table bookkeeping
  out of the current polychoric implementation so pairwise likelihood, robust
  polychorics, and NACOV construction share one kernel.
- [ ] Implement the smallest all-ordinal slice first: complete/listwise
  bivariate tables, fixed lavaan-style marginal thresholds, and pairwise rho
  estimates that reproduce current ML polychorics.
- [ ] Add a joint pairwise threshold/rho estimator after the fixed-threshold
  slice is stable; decide explicitly whether thresholds are pair-local nuisance
  parameters or shared SEM moments.
- [ ] Define missing-data semantics before coding: observed-pair composite
  likelihood may not need EM, but shared-threshold/multivariate MAR handling
  needs a documented likelihood target and scaling convention.
- [ ] Add mixed continuous/ordinal pair kernels after all-ordinal works:
  polyserial pairs, continuous-continuous normal pairs, and consistent moment
  ordering with current `MixedOrdinalStats`.
- [ ] Decide whether normal-data pairwise likelihood is a real supported
  estimator or only a benchmark/sanity check against complete-data ML/FIML.
- [ ] Compute casewise influence/Gamma for the pairwise moment vector so DWLS,
  WLS, robust SEs, and scaled tests can reuse the existing weighted-moment
  sandwich path.
- [ ] Add diagnostics: pair labels, convergence, boundary hits, table
  residuals, missingness counts, minimum eigenvalue of the assembled
  correlation matrix, and any ridge/shrinkage applied.
- [ ] Add lavaan-backed or internally cross-checked fixtures for complete
  all-ordinal polychorics first, then mixed data, then missing-data scenarios.

Done when: the pairwise path reproduces existing ML polychorics in its default
slice, has explicit missing-data semantics, and returns a moment/Gamma bundle
that downstream ordinal DWLS/WLS code can consume without special SEM-side
logic.

## 1. Trust and validation hardening

Intent: reach a state where the current implementation can be trusted as a
stable base for API design, refactoring, and larger research-facing work.

- [x] Extend continuous FIML lavaan fixtures beyond the original CFA,
  structural, path, fixed.x, equality, and dense missingness tranche. The FIML
  stream now includes complete-row equivalence coverage and multi-group
  `fixed.x` with complete exogenous variables plus missing outcomes.
- [x] Add targeted complete-data equivalence checks where FIML gradients match
  complete-data ML gradients up to objective constants when every row is fully
  observed.
- [x] Verify `fit_extras()` likelihood accounting for complete-data FIML and
  fixture-backed multi-group `fixed.x` / meanstructure-plus-`fixed.x` cases.
- [x] Keep robust FIML expansion fixture-first; unsupported missing-data
  corrections remain outside the public claim beyond the current
  MLR/Yuan-Bentler Mplus slice.
- [x] Harden mixed continuous/ordinal categorical validation and fixture
  reporting. Mixed NACOV/weight parity remains intentionally tolerance-loose
  while polyserial Gamma details are refined.

Done when: representative supported estimator/model combinations have checked-in
lavaan-backed fixtures, clear tolerances, and no hidden fixture-free parity
claims.

## 2. Lavaan comparison tooling

Intent: make parity failures easier to diagnose and fixture expansion cheaper.

- [x] Add lavaan partable comparison helpers that match by semantic row keys
  rather than raw row position.
- [x] Report missing rows, extra fixed-zero rows, label/plabel drift, group
  drift, and estimate drift as distinct comparison failures.
- [x] Make fitted-object partables returned through R helpers match the
  corresponding lavaan call under the same options, with special attention to
  fixed-zero intercept and mean rows.
- [x] Keep fixture regeneration centralized in `tools/regen_oracle.R`, and
  document any new tolerance classes near the fixture code.
- [ ] Add fixture cases for scalar-invariance latent mean rescaling once the
  bounded optimizer reliably fits the needed mean-structure models.

Done when: a failing lavaan parity test points to the semantic reason for the
drift, not just a raw row or numeric mismatch.

## 3. R and public API polish

Intent: make the methods-developer interface reasonable without turning the R
package into a second SEM implementation.

- [x] Add `magmaan(model, data, estimator, groups)` as the high-level R
  convenience for parse/lavaanify, sample-stat construction, and parameter
  estimation.
- [x] Keep `magmaan()` estimate-only: SEs, robust tests, fit measures, defined
  parameters, and nested tests remain explicit post-fit calls.
- [ ] Expose `compute_defined()` through R and add lavaan-backed goldens for
  chained `:=` rows and `.pN.` references.
- [ ] Add high-level shortcuts equivalent to `se = "none"` and `test = "none"`
  for point-estimate-only workflows.
- [ ] Audit exported R wrappers that take a whole fit list and split them into
  primitive-shaped wrappers plus explicit fit-list convenience adapters where
  useful. Priority examples are `infer_vcov()`, `infer_z_test()`,
  `infer_wald_test()`, `infer_rls_chi2()`, `infer_build_u_factor()`, and robust
  SE/test helpers.
- [x] Tighten validation and documentation for the sample-moment R path,
  including accepted shapes for `list(S = , nobs = , mean = )`.
- [ ] Keep ordinal R documentation current around `model_spec()` with
  `ordered` and `parameterization = "delta"`, `data_ordinal_stats_from_df()`,
  `fit_dwls_ordinal()`, and `fit_wls_ordinal()`.

Done when: a methods developer can fit supported complete, FIML, LS, and
ordinal/mixed models from R through thin wrappers while still choosing
post-fit inference explicitly.

## 4. Public namespace and header cleanup

Intent: finish the namespace transition without breaking compatibility
unnecessarily.

- [x] Move primary definitions out of old `magmaan::fit` and
  `magmaan::partable` namespaces into the target namespaces.
- [x] Convert repo code and R binding internals to target headers and
  namespaces directly.
- [x] Keep old `include/magmaan/fit/*` and `include/magmaan/partable/*`
  headers as compatibility shims for one transition window.
- [x] Keep one focused compatibility test for representative old headers.
- [x] Update stale docs and examples after the migration lands.
- [ ] After the transition window, remove the old `fit/*` and `partable/*`
  compatibility shims.

Done when: new code naturally uses `spec`, `lavaan`, `estimate`, `optim`, `nt`,
`gls`, and `data`, while old include paths still have a tested transition path.

## 5. Benchmarks and performance baselines

Intent: make refactors and backend choices measurable instead of anecdotal.

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

## 6. Larger and more principled tests

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

## 7. Remaining statistical gaps

Intent: keep unsupported statistical work explicit so it is not mistaken for a
polish task.

- [x] Generalize robust/inference helpers that assume the normal-theory ML
  weight so ULS/GLS/WLS can share sandwich paths with arbitrary per-block
  weights. A shared weighted-moment sandwich/U-Gamma primitive now backs
  ordinal, mixed ordinal, and continuous ULS/GLS/WLS robust adapters. ULS
  `robust.sem` has lavaan-backed non-fixed.x fixtures; GLS/WLS robust paths
  remain shape/scaling covered because lavaan does not expose matching robust
  scaled-test targets for those estimators.
- [x] Close the continuous LS standard golden gate for ULS/GLS/WLS, including
  lavaan's fixed.x Browne residual NT convention for observed exogenous
  moments.
- [x] Decide whether WLS robust ordinal reporting should get a lavaan-backed
  target or stay shape-only. Decision: keep robust WLS scaled-test reporting
  shape-only for now because lavaan rejects Satorra-Bentler-family `test=`
  requests with categorical `estimator = "WLS"`; DWLS robust reporting remains
  lavaan-backed.
- [ ] Add theta-parameterization support for ordinal models as a separate
  compatibility slice. The public R/C++ boundary now accepts the
  parameterization distinction and fails explicitly for `theta`; actual
  theta fitting still needs the separate lavaan-backed compatibility slice.
- [x] Lavaan-back and broaden the new modification-index / score-test surface:
  representative `modindices()` fixed-row and `lavTestScore()` equality-release
  fixtures now cover ML, observed-information FIML, continuous ULS, ordinal
  DWLS, and mixed ordinal DWLS. Mixed-ordinal MI remains tolerance-loose with
  the current polyserial/NACOV hardening state.
- [ ] Add absent-row generation and standardized EPC for modification indices
  only after the raw fixed-row/equality-release contract is fully validated.
- [ ] Keep nonlinear equality constraints, inequality constraints, and
  nonstandard active-bound inference out of the regular inference surface
  until their theory and reporting are explicit.

Done when: each unsupported statistical feature is either implemented with
oracle/fixture backing or remains clearly outside the public contract.

## 8. Documentation and examples

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

## 9. Alternative robust polychoric and mixed estimators

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

## 10. Composite models

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
