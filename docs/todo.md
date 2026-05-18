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

Intent: use the R package as an exploratory workbench for the C++ primitive
surface while the staged public API and namespace layout settle.

Contracts:

- R should expose virtually every relevant C++ primitive used for model
  construction, data construction, estimation, testing, inference, robust
  reporting, fit measures, standardization, and diagnostics. "Relevant" means
  functions that methods work would naturally call directly, not only
  implementation details hidden inside another primitive.
- `magmaan_core` is an exploratory audit surface during this phase, not yet a
  stability promise. It should be broad enough to reveal awkward names,
  misplaced namespaces, missing primitives, and over-composed helpers before
  the small friendly API is blessed.
- Direct C++ exports should be thin wrappers named
  `<namespace>_<function_name>` from the R side, reflecting the C++ namespace
  layout. Pure R helpers that compose those primitives should use ordinary
  helper names without a namespace prefix.
- Fit-list helpers may remain as explicit convenience adapters, but every
  convenience path should be decomposable into visible primitive calls through
  the R scaffold.
- User-facing APIs should be staged and explicit: model/data construction,
  point estimation, and post-fit inference/reporting are separate choices.
- Primitive APIs should remain available for methods work without crowding the
  friendly R namespace; low-level exports live behind `magmaan_core` while
  friendly helpers stay small and inspectable.
- R examples and development scripts should avoid `library("magmaan")` and
  `library("lavaan")`; use explicit package-qualified calls so examples make
  API boundaries visible.
- New C++ code should use the target namespaces directly: `parse`, `spec`,
  `model`, `data`, `estimate`, `inference`, `robust`, `measures`, `optim`,
  and `compat::lavaan`.

Remaining work, in suggested order:

- **S/M, as R exploration reveals it.** Add or rename wrappers only when the
  methods-developer workflow exposes a concrete gap in the current staged API.
  The current `magmaan_core`, friendly fit object, and explicit post-fit
  wrapper surface are sufficient for the next R exploration pass.

Completed checks:

- [x] **M/L. Staged C++ facade.** `magmaan::api` is now a compiling facade
  over the implemented model/data/fit and explicit post-fit primitives, with
  dedicated `api` tests covering ML, LS, FIML, ordinal DWLS/WLS, and
  `Analysis` chaining.
- [x] **S/M. R scaffold audit seed.** `docs/r-api-audit.md` now records the
  current `magmaan_core` primitive surface and classifies names as good,
  awkward, misplaced, missing, or compatibility-only.
- [x] **S/M. FIML MLR R scaffold.** `magmaan_core` exposes
  `estimate_fiml_robust_mlr()` for observed-pattern sandwich SEs plus
  Yuan-Bentler/Mplus scaled-test traces, with
  `r-package/examples/r_scaffold_fiml_mlr.R` as a package-qualified legibility
  example.
- [x] **S/M. Post-fit R scaffold slice.** `magmaan_core` exposes
  `measures_standardize_lv()`, `measures_standardize_all()`,
  `inference_modification_indices()`, and `inference_score_tests()` over the
  existing C++ primitives.
- [x] **S/M. Residual and factor-score R scaffold slice.** `magmaan_core`
  exposes `measures_residuals()`, `measures_standardized_residuals()`, and
  `measures_factor_scores()` over the C++ post-fit accessors; the
  package-qualified `post_fit_primitives.R` example now exercises them.
- [x] **S/M. R primitive alias normalization slice.** `magmaan_core` now
  publishes canonical `compat_lavaan_*`, `estimate_*`, `inference_*`,
  `robust_*`, and `measures_*` aliases while keeping the old `lavaan_*`,
  `fit_*`, `infer_*`, and `_impl` names available for compatibility; the
  broad post-fit and staged workflow examples use the canonical names.
- [x] **S. Example migration slice.** The observed-information,
  defined-parameter, ordinal DWLS/WLS, FIML MLR, and tutorial
  modification-index / extraction examples now use package-qualified calls and
  the canonical `magmaan_core` names where they exercise direct primitives.
- [x] **S. High-level R fit object polish.** `magmaan()` now returns a
  `magmaan_fit` list with source model/spec/options/grouping metadata and a
  restrained print method, while keeping SEs, tests, robust corrections, and
  fit measures as explicit post-fit calls.
- [x] **S/M. Friendly explicit post-fit wrappers.** Exported
  `standardized()`, `stats::residuals.magmaan_fit()`, `factor_scores()`, and
  `modification_indices()` as thin wrappers over `magmaan_core`, preserving
  explicit covariance and raw-data choices.
- [x] **S/M. R scaffold third pass closure.** The audit has no current missing
  primitive bindings, copy-worthy examples use package-qualified calls and
  canonical `magmaan_core` names, and the friendly `magmaan_fit` / post-fit
  wrappers are ready for exploratory R use. Future R exploration should open
  specific follow-up tasks instead of keeping this broad scaffold item open.
- [x] **M/L. Robust-test naming policy slice.** R now exposes
  `robust_nested_lrt()` as the statistical nested-model LRT helper, keeps
  `nestedTest()` and lavaan Satorra labels as compatibility aliases, and
  publishes both `robust_nested_lrt_restriction_map` and
  `compat_lavaan_nested_lrt_*` aliases through `magmaan_core`.
- [x] **M/L. Ordinal post-fit fit measures.** All-ordinal DWLS/WLS now expose
  categorical CFI/TLI/RMSEA/SRMR via the polychoric independence baseline,
  including WLS threshold nuisance minimization under the full weight matrix;
  the bfi ordinal parity gate checks DWLS/WLS values against lavaan.

Done when: new code naturally uses the target namespaces, R scripts can reach
the relevant C++ primitive graph through predictable names, friendly R users
see a small staged API, and methods developers can choose primitive post-fit
operations without depending on hidden fit-list unpacking.

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

## 3. Composite models

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

## 4. lavaan tutorial parity — deferred items

Intent: close the gaps the lavaan-tutorial audit
([docs/lavaan_tutorial_parity.md](lavaan_tutorial_parity.md)) surfaced. The
audit doc is the live checklist; this section is the actionable backlog.

Contracts:

- Each item must stay fixture-backed (or example-backed) on landing, like the
  rest of magmaan's lavaan-parity claims.
- Unsupported combinations keep failing explicitly rather than approximating.

Remaining work, in suggested order:

- **L. Bootstrapping.** `se = "bootstrap"`, Bollen-Stine `test = "bootstrap"`,
  and bootstrap confidence intervals for `:=` defined parameters. Principled
  RNG design (settled): the C++ resampling engine takes an explicit integer
  `seed` and is deterministic given it — no language-specific code in the
  core. Each binding forwards its own seed: the R wrapper draws one integer
  under the active `set.seed()` so reproducibility is automatic; Python and
  the C++ API pass a seed directly. The engine owns the resample + refit loop
  and the Bollen-Stine data transform.
- **M. `lavResiduals()` z-statistics.** The deterministic residual metrics
  landed (`measures::standardized_residuals` — raw, correlation-metric, and
  SRMR). The asymptotic-SE standardized residuals (`lavResiduals()$cov.z`)
  still need the residual-ACOV convention (`Γ_NT/N − Δ·vcov·Δᵀ`) pinned
  against a lavaan oracle before landing.
- **M/L. CFI/TLI/RMSEA/SRMR for ordinal.** The FIML and continuous-LS slices
  landed; ordinal DWLS/WLS still needs a polychoric independence-model
  baseline and a correlation-metric ordinal SRMR. `api::fit_measures` fails
  explicitly for ordinal fits until then.
Completed:

- [x] **S. `residuals()` accessor.** `measures::residuals` / `api::residuals`
  — the raw residual moment matrices `S − Σ̂(θ̂)` (plus the mean residuals
  under a mean structure).
- [x] **M. `standardized_residuals()` table.** `measures::standardized_residuals`
  / `api::standardized_residuals` — magmaan's `lavResiduals()`: the raw and
  correlation-metric (Bentler) residual matrices and the SRMR. The
  asymptotic-SE z-statistics are tracked above.
- [x] **M. `lavPredict()` factor scores.** `measures::factor_scores` /
  `api::factor_scores` — regression (Thurstone) and Bartlett scores.
- [x] **M. `exp()` / `log()` in expressions.** Function calls now parse,
  canonical-format, classify (`analyze_linear`), and evaluate with
  forward-mode AD in both the `:=` evaluator (`expr_eval`) and the
  nonlinear-constraint evaluator (`nl_constraints`).
- [x] **L. Nonlinear equality + linear equality in one model.** The
  augmented-Lagrangian path now runs in the linear-constraint-reduced α-space
  when a model carries both kinds of equality.
- [x] **M. Standardized EPC (`sepc.all`).** Already produced in the core /
  `magmaan::api` modification-index output (`ScoreTestResult.epc_lv` /
  `epc_all`, filled by `fill_standardized_epc`) and exposed through the
  `magmaan_core$inference_modification_indices()` scaffold.
- [x] **M. CFI/TLI/RMSEA/SRMR for FIML and continuous LS**, and **standard
  (non-robust) SEs for FIML and LS.** `api::fit_measures` and
  `api::standard_errors` now dispatch per estimator; FIML also reports SRMR
  against the saturated EM moments, and the FIML observed information is
  `−∂²logl/∂θ²`.

Done when: the audit doc shows no ◐/✗ rows for the in-scope tutorial sections,
or each remaining gap is a deliberate, documented out-of-scope decision.
