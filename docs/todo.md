# magmaan TODO

This is the active human-readable backlog for work that remains after the
current implementation state summarized in [docs/roadmap.md](roadmap.md). Keep
it focused on unfinished work, acceptance checks, and the next useful seams for
methods development.

## 1. Trust and validation hardening

Intent: reach a state where the current implementation can be trusted as a
stable base for API design, refactoring, and larger research-facing work.

- [ ] Extend continuous FIML lavaan fixtures beyond the current CFA,
  structural, path, fixed.x, equality, and dense missingness tranche.
- [ ] Add targeted complete-data equivalence checks where FIML gradients match
  complete-data ML gradients up to objective constants when every row is fully
  observed.
- [ ] Verify `fit_extras()` conditional log-likelihood for multi-group
  `fixed.x` and meanstructure plus `fixed.x` once targeted fixtures exist.
- [ ] Keep robust FIML expansion fixture-first; do not claim unsupported
  missing-data corrections beyond the current MLR/Yuan-Bentler Mplus slice.
- [ ] Harden mixed continuous/ordinal categorical parity, especially
  NACOV/weight construction, DWLS/WLS fits, and robust reporting tolerances.

Done when: representative supported estimator/model combinations have checked-in
lavaan-backed fixtures, clear tolerances, and no hidden fixture-free parity
claims.

## 2. Lavaan comparison tooling

Intent: make parity failures easier to diagnose and fixture expansion cheaper.

- [ ] Add lavaan partable comparison helpers that match by semantic row keys
  rather than raw row position.
- [ ] Report missing rows, extra fixed-zero rows, label/plabel drift, group
  drift, and estimate drift as distinct comparison failures.
- [ ] Make fitted-object partables returned through R helpers match the
  corresponding lavaan call under the same options, with special attention to
  fixed-zero intercept and mean rows.
- [ ] Keep fixture regeneration centralized in `tools/regen_oracle.R`, and
  document any new tolerance classes near the fixture code.
- [ ] Add fixture cases for scalar-invariance latent mean rescaling once the
  bounded optimizer reliably fits the needed mean-structure models.

Done when: a failing lavaan parity test points to the semantic reason for the
drift, not just a raw row or numeric mismatch.

## 3. R and public API polish

Intent: make the methods-developer interface reasonable without turning the R
package into a second SEM implementation.

- [ ] Add `magmaan(model, data, estimator, groups)` as the high-level R
  convenience for parse/lavaanify, sample-stat construction, and parameter
  estimation.
- [ ] Keep `magmaan()` estimate-only: SEs, robust tests, fit measures, defined
  parameters, and nested tests remain explicit post-fit calls.
- [ ] Expose `compute_defined()` through R and add lavaan-backed goldens for
  chained `:=` rows and `.pN.` references.
- [ ] Add high-level shortcuts equivalent to `se = "none"` and `test = "none"`
  for point-estimate-only workflows.
- [ ] Tighten validation and documentation for the sample-moment R path,
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

- [ ] Move primary definitions out of old `magmaan::fit` and
  `magmaan::partable` namespaces into the target namespaces.
- [ ] Convert repo code and R binding internals to target headers and
  namespaces directly.
- [ ] Keep old `include/magmaan/fit/*` and `include/magmaan/partable/*`
  headers as compatibility shims for one transition window.
- [ ] Keep one focused compatibility test for representative old headers.
- [ ] Update stale docs and examples after the migration lands.

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

- [ ] Add Browne residual ADF by swapping empirical gamma into the existing
  Browne residual projection machinery.
- [ ] Generalize robust/inference helpers that assume the normal-theory ML
  weight so ULS/GLS/WLS can share sandwich paths with arbitrary per-block
  weights.
- [ ] Decide whether WLS robust ordinal reporting should get a lavaan-backed
  target or stay shape-only.
- [ ] Add theta-parameterization support for ordinal models as a separate
  compatibility slice.
- [ ] Add modification indices, univariate score tests, and EPC using scores
  and information for fixed or equality-constrained parameters at the optimum.
- [ ] Add RMSEA close-fit p-value support.
- [ ] Check `std.all` beta-coefficient rescaling for structural regressions.
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
