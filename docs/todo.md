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
- User-facing APIs should be staged and explicit: model/data construction,
  point estimation, and post-fit inference/reporting are separate choices.
- Primitive APIs should remain available for methods work without crowding the
  friendly R namespace.
- New C++ code should use `spec`, `lavaan`, `estimate`, `optim`, `nt`, `gls`,
  and `data` directly.

Remaining work, in suggested order:

- [ ] **M/L.** Formalize the staged C++ facade after the prototype settles.
  Keep it as value objects over the existing primitives (`Model`, `Data`,
  `Fit`, `Analysis`, `Summary`) rather than a second estimator implementation.
- [ ] **S/M.** Continue migrating R post-fit helpers from opaque fit-list
  unpacking toward explicit primitive-shaped entry points, keeping convenience
  aliases only where they do not obscure the contract.
- [ ] **M.** Replace the broad R primitive export strategy with a small
  friendly namespace plus a single `magmaan_core` primitive escape hatch.
  `magmaan_core` should mirror C++-shaped entry points while ordinary tab
  completion stays focused on staged user workflows.
- [ ] **M/L.** Design the Python binding surface around a friendly top-level
  API and a `magmaan.core` submodule for primitives, keeping names aligned with
  the C++ namespace layout where possible.
- [ ] **S/M.** Add examples for the staged API design: ordinal polychoric DWLS
  with WLSMV-style robust reporting, complete-data ML with Satorra-Bentler,
  complete-data ML with observed-information SEs, and FIML with MLR-style
  robust reporting.
- [x] **M, later transition cleanup.** After the compatibility window, remove
  the old `fit/*` and `partable/*` shim headers and their transition tests.
  (`fit/*` and `gls/*` were removed with the GMM-core refactor; the orphaned
  `partable/*` forwarding shims were removed once nothing referenced them.)

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

- [x] **M/L.** Add named small-sample stabilizer builders: variance-only bounds,
  wider/literature variance bounds, statistically justified loading bounds,
  start projection into the chosen box, active-bound diagnostics, and
  Heywood/small-N fixtures across representative ML/LS/ordinal slices.
  (`estimate::variance_bounds` / `standard_bounds` / `wide_bounds` /
  `loading_bounds` port lavaan's `optim.bounds` `pos.var`/`standard`/`wide`
  presets — explicit builders, no default change; `intersect_bounds`,
  `project_start_into_bounds`, `active_bounds` round out the surface.
  `tests/unit/stabilizer_bounds_test.cpp` pins the bound values against a
  lavaan 0.6.22 oracle and exercises ML + ULS Heywood slices.)
- [ ] **M/L.** Add covariance shrinkage as sample-moment transformations that
  return `SampleStats` or `MixedOrdinalStats`-compatible objects plus repair
  diagnostics, without changing lavaan-compatible defaults.
  (`data::shrink_sample_stats` shrinks continuous `SampleStats` toward Ridge /
  identity / diagonal / constant-correlation targets — fixed or normal-theory
  Ledoit-Wolf intensity — returning `ShrunkSampleStats` with per-block
  eigenvalue diagnostics; `tests/unit/shrinkage_test.cpp`. Remaining (2b):
  `MixedOrdinalStats` shrinkage needs the `src/data/ordinal.cpp` moment/NACOV
  rebuild factored into a shared helper before R can be shrunk consistently.)
- [ ] **L.** Add DLS and empirical-Bayes DLS as weight-matrix builders layered
  on the existing LS discrepancy and weighted-moment sandwich surfaces.
  (`estimate::dls_weight` builds Browne's distributionally-weighted weight
  `W = ((1-a)·Γ_NT + a·Γ_ADF)⁻¹` over the `[mean ; vech(cov)]` layout — reuses
  `data::gamma_nt` / `data::empirical_gamma`; `a = 0` reproduces
  `gmm::normal_theory_weight` bit-exactly, `a = 1` the ADF weight;
  `tests/unit/dls_weight_test.cpp`. Remaining: `eb_dls_weight` — the
  empirical-Bayes mixing-scalar estimator needs a methods decision + citation;
  and a `checks/dls/` Monte-Carlo driver for the statistical-benefit check.)
- [x] **L.** Add theta-parameterization support for ordinal models as a
  separate lavaan-backed compatibility slice.
  (`fit_ordinal_bounded` takes an `OrdinalParameterization` argument; `Theta`
  fixes the latent-response residual variances to 1 and fits the *standardized*
  implied moments — `ordinal_residuals` / `ordinal_jacobian` gained a theta
  branch that standardizes the implied correlations Σ*ᵢⱼ/√(Σ*ᵢᵢΣ*ⱼⱼ) and
  rescales the implied thresholds by √Σ*ᵢᵢ. The prepared partable is
  parameterization-independent. `tests/unit/ordinal_test.cpp` pins
  delta/theta reparameterization invariance (equal fmin); the
  `ordinal_golden_test.cpp` theta case matches magmaan's theta loadings to a
  lavaan 0.6.22 `parameterization="theta"` oracle. Remaining: mixed
  continuous/ordinal theta, and theta robust inference / modification indices /
  standardized-solution reporting.)
- [ ] **M.** Add absent-row generation and standardized EPC for modification
  indices after the raw fixed-row/equality-release contract remains stable.
  (`nt::infer::modification_indices` gains a `ModificationIndexOptions` overload
  — `ScoreCandidateSet::WithAbsentRows` enumerates absent cross-loadings and
  covariances as fixed-at-0 candidates; `ScoreTestResult` now carries
  standardized EPC `epc_lv` / `epc_all`. `tests/unit/score_test.cpp` gates
  absent-row MI/EPC against a lavaan 0.6.22 `modindices()` oracle and pins
  absent-gen ≡ explicit-fixed-row scoring. Remaining: thread the options
  through the LS / FIML / ordinal `modification_indices` overloads;
  structural-regression absent rows (a `~` row changes the model form, beyond
  a partable append); golden-fixture regen of `tests/fixtures/score/`.)
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
- Mixed continuous/ordinal robust estimators should stay explicitly separated
  from the default lavaan-compatible mixed path.
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
- [x] **L.** Implement full pair-local WMA/h-score estimation with thresholds
  and rho estimated jointly, validated against robcat/WMA or independent
  reference calculations for selected bivariate tables.
  (`data::fit_ordinal_pair_joint_h_weighted()` estimates pair-local thresholds
  and rho with the same predefined h-score family; unit coverage pins the ML
  and `hard_cap(k = Inf)` limits, a contaminated discordant-corner case, and
  an independently recomputed h-objective/local-perturbation check.)
- [x] **L.** Add casewise influence and sandwich/Gamma calculations for robust
  pairwise moments, including the hard-cap kink convention and scaling.
  (`data::ordinal_pair_h_weighted_influence()` returns casewise
  centered `h'(t)`-weighted meat rows, h-score diagnostics, analytic
  probability-Hessian bread, influence rows, score Gamma, and sandwich Gamma;
  unit coverage pins ML and `hard_cap(k = Inf)` limits, `S'S / n` scaling,
  downweighting, and `dh(t) = 0` at the WMA hard-cap kink.)
- [x] **M/L.** Replace the old fixed-threshold robust-rho Option A with a
  shared-threshold h-score SEM path.
  (`data::pairwise_ordinal_stats_h_weighted_from_integer_data()` now optimizes
  one threshold block per ordinal variable plus one polychoric correlation per
  pair under the h-weighted composite objective, then rebuilds
  moment-influence/Gamma and DWLS/WLS weights from the shared robust estimating
  equations. Tests pin the ML limit, shared-threshold movement under finite
  caps, Gamma scaling, correlation repair, and `fit_ordinal_bounded()`
  compatibility.)
- [x] **M.** Handle indefinite/low-eigen robust correlation matrices
  explicitly with minimum-eigen diagnostics and optional error, ridge, or
  shrinkage policies.
  (`PairwiseOrdinalCorrelationRepairOptions` records the raw/final minimum
  eigenvalues, repairs robust `R` toward the identity when requested, applies
  the matching correlation-column influence scaling before rebuilding Gamma,
  and exposes block/pair diagnostics.)
- [x] **M.** Add pair-local density power divergence as the main non-h-score
  bivariate comparator.
  (`data::fit_ordinal_pair_joint_dpd()` estimates pair-local thresholds and
  rho, delegates `alpha = 0` to joint ML, and exposes `p^alpha` attenuation
  weights plus the same table diagnostics as the h-weighted path.)
- [x] **M/L.** Add all-ordinal shared-threshold DPD SEM stats.
  (`data::pairwise_ordinal_stats_dpd_from_integer_data()` uses the same
  shared threshold/correlation parameter layout as the h-score SEM path,
  optimizes the density-power-divergence composite objective, delegates
  `alpha = 0` to the default ML stats path, and exposes fit-ready
  `PairwiseOrdinalStats` with sandwich Gamma.)
- [ ] **M/L.** After shared-threshold h/DPD simulation use, decide whether
  Hellinger and Huberized residual fitting are worth adding as lower-priority
  experimental comparators.
- [x] **M.** Pin the h-score estimators against the canonical robcat R package
  (Welz, Mair & Alfons, 2026; vendored in `external/robcat`). The WMA hard cap
  is the same C-estimator as robcat `polycor`: `robcat polycor(c=C)` ≡ magmaan
  `WmaHardCap, k = C+1`, and `polycor_mle` ≡ `ML`.
  (`tools/regen_robcat_fixtures.R` freezes robcat MLE + robust-sweep oracles
  into `tests/fixtures/robcat/`; `tests/golden/robcat_parity_golden_test.cpp`
  — target `magmaan_test_robcat`, `just test-area robcat` — hard-gates rho,
  thresholds, and robust sandwich SE magnitudes for `c > 0` on clean, skewed,
  and contaminated tables. Point estimates match robcat to <5e-4 everywhere;
  the `c = 0` boundary remains a finite/positive SE diagnostic because the WMA
  hard cap is degenerate there.)
- [x] **M.** Fix the robust h-weighted sandwich meat in
  `data::ordinal_pair_h_weighted_influence()`. The C-estimation meat now uses
  centered `W·Λ·Wᵀ` rows with `W_k = h'(t_k)·s_k` instead of the diagnostic
  weight `h(t_k)/t_k`; the bread is analytic from bivariate rectangle
  probability Hessians rather than finite differences. The SEM Option A
  ordinal Gamma path uses the same `h'(t)` rho linearization.
- [x] **S/M.** Add non-CI simulation sanity checks for the robust all-ordinal
  Gamma path. `checks/robust_polychoric/` has its own local `justfile` and
  standalone C++ driver for pair-local robust polychorics, HS-style ordinal
  moment Gamma, and HS-style ordinal CFA robust vcov checks; generated
  binaries/results stay ignored and are not wired into the root package
  workflow.
- [x] **S/M.** Keep the terminology clean: DPD means density power divergence
  and is independent of robcat; robcat parity applies to the h-score/WMA
  C-estimators only. Package names keep this split (`*_dpd` versus
  `*_h_weighted`), and the mixed robust polyserial path no longer reuses
  robcat wording.
- [x] **M.** Promote the full pair-local DPD polyserial primitive and retire
  the fixed-marginal/fixed-threshold robust polyserial prototypes. The exposed
  DPD path is now `data::fit_polyserial_pair_joint_dpd()`, which jointly
  estimates continuous mean/scale, ordinal thresholds, and rho, returning
  probabilities, joint densities, and `f(x, y)^alpha` attenuation weights. The
  retired rho-only DPD and fixed-reference h-weighted mixed-stat builders were
  misleading for the intended robustness track because they kept the ordinal
  thresholds fixed.
- [x] **M.** Make the full polyserial DPD question visible in simulation
  checks. The local `checks/robust_polychoric` driver now runs a pair-level
  contaminated polyserial experiment comparing lavaan-style fixed-marginal ML
  rho against package full DPD. On the current quick run, full DPD worked and
  sharply reduced average `|rho - rho_true|` relative to ML under continuous
  tail/category discordance.
- [ ] **M/L.** Design SEM-level mixed DPD integration for polyserial pairs.
  The current `MixedOrdinalStats` layout has shared marginal thresholds, while
  the current full DPD polyserial primitive is only bivariate/pair-local. Do
  not silently inject pair-specific thresholds into shared-threshold moments;
  implement a mixed shared-threshold DPD estimator with one threshold block per
  ordinal variable, one mean/variance block per continuous variable, and the
  matching sandwich Gamma.
- [ ] **M/L.** Add a sandwich/Gamma calculation for full pair-local
  polyserial DPD only if it remains useful as a bivariate diagnostic. It cannot
  be checked against robcat because DPD is not a robcat estimator, so
  validation should use finite-difference estimating-equation derivatives plus
  Monte Carlo covariance checks in `checks/robust_polychoric`.

Done when: robust polychoric/polyserial alternatives are selectable where their
moment contracts are designed, default ML fixtures are unchanged, diagnostics
make robustness visible, and at least one robust method has a Gamma path usable
by ordinal DWLS/WLS robust reporting.

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
