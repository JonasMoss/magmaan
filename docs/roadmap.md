# magmaan roadmap

This is the live roadmap for complete-data linear SEM work. It is the source of
truth for current implementation state and near-term work; old phase notes should
be folded here or deleted when they stop being actionable.

Out of scope for this track: FIML/missing data beyond listwise handling for
the ordinal LS path, Bayesian, multilevel, latent interactions/mixtures, EFA,
and end-user `cfa(model, data)` ergonomics.

## Current State

The core parser-to-fit pipeline is in place:

- C++23 library built under `-fno-exceptions -fno-rtti`, with fallible APIs
  returning `std::expected<T, Error>`.
- Lavaan-style syntax parser, normative EBNF in `docs/grammar/`, checked-in
  lexer/parser/partable/matrix/fit oracle fixtures, and R-only fixture
  regeneration through `tools/regen_oracle.R`.
- The lavaanified model contract is the triple `LatentStructure`,
  `LatentNames`, and `Starts`, with `LavaanParTable` as the boundary projection
  for R and oracle comparison.
- Single- and multi-group LISREL matrix representation, fixed.x resolution,
  mean structures, marker/std.lv/effect-coding identification, start hints, and
  linear equality constraints.
- ML fitting through LBFGS, including affine reparameterization for linear
  equalities.
- ULS, GLS, and explicit-weight WLS discrepancies, each with scalar
  value/gradient and least-squares residual/Jacobian interfaces.
- Bounded least-squares fitting through LBFGS-B and optional Ceres, including
  automatic nonnegative variance bounds and equality-penalty residuals on the
  LS path.
- First-pass ordinal LS support: parser and partable projection for threshold
  (`|`) and response-scale (`~*~`) rows, integer all-ordinal complete/listwise
  sample statistics, pairwise polychoric correlations, Muthen-style all-ordinal
  NACOV construction for thresholds plus polychorics, DWLS diagonal weights,
  full WLS weights, bounded ordinal LS fitting, and thin R wrappers for ordinal
  sample stats plus DWLS/WLS fits. Checked-in ordinal fixtures now validate
  thresholds, polychoric `R`, `NACOV`, `WLS.V`, `WLS.VD`, and DWLS/WLS
  delta-parameterization free sets, point estimates, degrees of freedom, and
  chi-square statistics across single-group, multi-group, skewed, near-empty
  category, and sparse nonempty ordinal cases, with equality-constrained DWLS
  fit coverage. The C++ ordinal fitter shares the bounded LS optimizer surface,
  with LBFGS-B as the default and Ceres objective parity coverage in Ceres
  builds. This path currently implements lavaan's delta-style response scale
  boundary; theta parameterization, mixed polyserial statistics, ordinal SNLLS,
  and robust ordinal reporting remain open.
- Separable nonlinear least squares (SNLLS) profiling for LS estimators where
  conditionally linear parameters can be profiled out.
- Expected information, finite-difference observed information, analytic
  observed information for covariance-only models, vcov/SE, Wald/z tests,
  chi-square/df helpers, LR/Satorra-2000 nested tests, robust U-Gamma
  machinery, Satorra-Bentler-family statistics, robust SEs, Browne residual NT,
  fit measures, standardization, and C++ defined-parameter evaluation.
- Exploratory R bindings for lavaanify, fitting, sample-stat bundles, robust
  inference, fit measures, model implied moments, LS estimators, SNLLS, Ceres
  paths when enabled, and data-frame-to-model sample statistics.
- The intended R boundary is thin wrappers over C++ plus small composition
  helpers. A future `magmaan(model, data, estimator, groups)` convenience should
  perform estimation only; SEs, tests, robust corrections, fit measures, and
  defined parameters stay explicit post-fit steps.
- Transitional public namespaces exist (`spec`, `lavaan`, `estimate`, `optim`,
  `nt`, `gls`, `data`) as aliases over remaining canonical definitions in
  `fit` and `partable`.

## Immediate Priorities

### 1. Turn LS estimator support into lavaan-parity fixtures

ULS/GLS/WLS are implemented as discrepancies and exercise the bounded LS path
in unit/integration tests. The next step is parity work, not basic
implementation.

Open work:

- Add checked-in lavaan golden fixtures for `estimator = "ULS"`, `"GLS"`, and
  `"WLS"` across single-group CFA, multi-group CFA, equality constraints,
  fixed.x, and mean-structure cases.
- Decide which LS backend is the default public recommendation for each
  estimator: LBFGS-B, Ceres, or SNLLS. ULS landscapes are shallow enough that
  backend defaults matter.
- Promote the Ceres preset into the regular validation loop where relevant
  (`cmake --preset ceres`, `ctest --preset ceres`) without making the default
  build pay the Ceres dependency cost.
- Restore or add the Holzinger 3-factor mean-structure fixture once the chosen
  bounded LS path fits it reliably.
- Compare LS fit statistics against lavaan's estimator-specific reporting,
  especially Browne residual tests versus raw LS objective statistics.

Validation targets:

- Saturated one-factor ULS, which exposes shallow-landscape conditioning.
- Heywood-prone ULS/GLS/WLS, where lower bounds must be enforced.
- Multi-group LS weighting and equality constraints.
- Mean-structure LS models with Ceres and SNLLS backends.

### 2. Make ordinal LS lavaan-parity-grade

The first implementation path is intentionally narrow: complete/listwise
ordinal indicators, threshold starts from marginal proportions, pairwise
polychorics, diagonal weights, and bounded LS over model-implied correlations
plus thresholds. It is enough for methods exploration, but not yet a claim of
lavaan ordinal estimator parity.

Open work:

- Extend the checked-in ordinal sample-stat fixtures as new edge cases are
  added. Current fixtures compare thresholds, polychoric `R`, `NACOV`,
  `WLS.V`, and `WLS.VD` against lavaan for representative all-ordinal
  single-group, multi-group, skewed, sparse-but-nonempty, near-empty category,
  and equality-constrained cases. Keep numeric tolerances documented
  separately for threshold estimates, bivariate-normal integration, optimizer
  convergence, and weight inversion.
- Keep `parameterization = "delta"` as the asserted ordinal boundary for now.
  The R helper accepts only `"delta"` and emits fixed response-scale rows;
  lavaan's `parameterization = "theta"` remains a later compatibility slice.
- Maintain ordinal DWLS/WLS fit parity. The fixtures include lavaan fit outputs
  and now assert `fit_ordinal_bounded()` free-parameter counts, estimates,
  degrees of freedom, and chi-square statistics under lavaan's delta contract.
- Maintain R-boundary coverage for `data_ordinal_stats_from_df()`,
  `fit_dwls_ordinal()`, and `fit_wls_ordinal()`. The example smoke tests now
  cover sample-stat construction, result reconstruction, lavaan chi-square
  parity, and empty-category validation.
- Keep polychoric construction internal to `ordinal_stats_from_integer_data()`
  until the sample-stat contract is stable. A public polychoric API should
  expose the moment-vector ordering, category metadata, sample-size scaling,
  and error policy explicitly rather than returning only a correlation matrix.
- Decide the public backend policy for ordinal fitting. C++ now shares the
  generic bounded LS optimizer surface and Ceres builds check objective parity,
  but R Ceres wrappers and any public recommendation should wait until
  parameter/statistic parity is resolved. Consider an ordinal SNLLS variant
  only after the conditionally linear block remains valid with threshold
  residuals included.
- Extend the data path beyond all-ordinal complete/listwise indicators only
  when the mixed continuous/ordinal contract is clear. Polyserial correlations
  and mixed thresholds should not be inferred ad hoc in R.
- Connect robust ordinal reporting deliberately, reusing the existing
  continuous-data robust vocabulary where the algebra matches but validating
  the categorical moment vector separately. Lavaan's relevant LS-family targets
  are `se = "robust.sem"` / `"robust.sem.nt"` and tests
  `satorra.bentler` (WLSM/ULSM), `scaled.shifted` (WLSMV/ULSMV), and
  `mean.var.adjusted` (WLSMVS/ULSMVS). Plain categorical DWLS keeps a
  standard test by default in lavaan, so WLSMV-style reporting should be an
  explicit post-fit call, not silently implied by the DWLS point estimator.
  Robust ordinal SEs and scaled tests must consume the threshold-plus-polychoric
  `NACOV`/`WLS.V` layout, not raw continuous-data casewise Gamma helpers.

Validation targets:

- Single-group ordinal CFA with three or more categories per indicator.
- Multi-group ordinal models with group-specific thresholds.
- Equality constraints involving loadings while thresholds remain free.
- Threshold-heavy edge cases with empty or near-empty categories, where the
  current sample-stat builder must return explicit errors instead of infinities.
  Empty categories should stay hard errors; near-empty categories need fixture
  coverage so finite adjustments, convergence, and NACOV conditioning are
  visible instead of accidental.
- Backend parity for ordinal LS if Ceres is enabled: LBFGS-B and Ceres already
  agree on objective values for representative DWLS/WLS cases. Do not expose a
  public Ceres ordinal wrapper until the remaining ordinal reporting surface is
  stable.

### 3. Close remaining inference and robust gaps

Expected-info inference, finite-difference observed inference, covariance-only
analytic observed inference, robust SEs, U-Gamma/Satorra-Bentler families,
Browne residual NT, Wald/LR/z tests, fit measures, standardization, and
Satorra-2000 nested tests are implemented and covered.

Open gaps:

- Extend `information_observed_analytic()` to mean-structure models. The
  missing terms are the `dmu` and `d2mu` contributions involving Lambda-alpha,
  Lambda-Beta, alpha-Beta, and Beta-Beta pairs.
- Generalize observed-bread robust SE and observed-Hessian U-factor handling to
  multi-block models. The expected-bread path is multi-block; observed bread is
  still single-block.
- Generalize Browne's unbiased reduced gamma correction to multi-block models.
- Add Browne residual ADF by swapping empirical gamma into the existing Browne
  residual projection machinery.
- Verify `fit_extras()` conditional log-likelihood for multi-group `fixed.x`
  and meanstructure times `fixed.x` once targeted fixtures exist.
- Generalize robust/inference helpers that assume the normal-theory ML weight
  so ULS/GLS/WLS can share sandwich paths with arbitrary per-block weights.

### 4. Finish R/API parity polish

The C++ surface is ahead of the R boundary in several places.

Open items:

- Add `magmaan(model, data, estimator, groups)` as the single high-level R
  convenience for parse/lavaanify plus sample-stat construction plus parameter
  estimation. Keep it estimate-only; do not automatically compute SEs, robust
  MLM/Satorra-Bentler tests, fit measures, defined parameters, or nested tests.
- Keep the rest of the R API as thin wrappers around C++ entry points plus
  small R-side composition helpers. Avoid R implementations of SEM behavior
  that can drift away from the C++ oracle path.
- Expose `compute_defined()` through R and add a golden comparison against
  lavaan `parameterEstimates()` for chained `:=` rows and `.pN.` references.
- Add high-level shortcuts equivalent to `se = "none"` and `test = "none"` so
  callers can skip post-fit inference when they only need point estimates.
- Make fitted-object partables round-trip to the same lavaan-shaped table that
  the corresponding lavaan call would return when constructed through the R
  helpers. The known suspect is mean-structure handling: fixed-zero intercept
  or mean rows may currently be retained or reconstructed differently from
  lavaan.
- Add lavaan partable comparison helpers that compare by semantic row keys
  rather than raw row position, so parity failures identify whether the issue is
  missing rows, extra fixed-zero rows, labels/plabels, groups, or estimates.
- Tighten the sample-moment R path around validation and documentation. The
  binding already accepts `list(S = , nobs = , mean = )`; it should be easier
  to discover and harder to mis-shape.
- Document the ordinal R boundary around `model_spec(..., ordered = ,
  parameterization = "delta")`, `data_ordinal_stats_from_df()`, and
  `fit_dwls_ordinal()` / `fit_wls_ordinal()` once the first lavaan fixtures
  lock down behavior.
- Keep fit-list extraction helpers as R boundary conveniences, while keeping
  C++ APIs explicit over primitive values.

### 5. Finish namespace cleanup opportunistically

The target public namespaces exist and are covered by
`tests/unit/namespace_alias_test.cpp`:

- `spec`: model specification, lavaanify options, equality groups, linear
  constraints, starts.
- `lavaan`: lavaan-shaped partable projections and compatibility views.
- `estimate`: estimates, fitting orchestration, bounds, fixed.x resolution,
  equality reparameterization, SNLLS.
- `optim`: optimizer concepts and implementations.
- `nt`: complete-data normal-theory ML, inference, robust tests, measures,
  standardization, and defined parameters.
- `gls`: ULS/GLS/WLS discrepancy functions.
- `data`: sample statistics, raw data, and moment/gamma helpers.

What remains:

- Move primary definitions out of the old `magmaan::fit` and
  `magmaan::partable` namespaces into the target namespaces.
- Convert repo code and R binding internals to include and call target
  headers/namespaces directly.
- Keep old `include/magmaan/fit/*` and `include/magmaan/partable/*` headers as
  compatibility shims for one transition window.
- Keep one focused compatibility test for representative old headers.
- Update stale source references in docs after the migration lands.

This is API organization work, not the next statistical blocker.

## Reporting and Exploration

These are lavaan-parity features above the core estimator:

- Modification indices, univariate score tests, and EPC are not implemented.
  They need scores and information for fixed or equality-constrained
  parameters at the optimum.
- RMSEA close-fit p-value (`rmsea.pvalue`, H0: RMSEA <= 0.05) is not computed.
- `std.all` beta-coefficient rescaling remains to be checked for structural
  regressions.
- Add targeted goldens for scalar-invariance latent mean rescaling once the
  bounded optimizer reliably fits the needed mean-structure models.

## Constraint Boundary

Linear equality constraints are implemented through affine reparameterization
for ML and penalty residuals on the bounded LS path. Effect coding for loadings
is implemented. The current boundary is intentional:

- Nonlinear equality constraints remain unsupported.
- Inequality constraints remain unsupported.
- Active bound/inequality inference must not silently report ordinary
  chi-square/SE theory; boundary cases need explicit warnings or
  chi-bar-square style treatment before being presented as regular inference.

## Design Invariants

- No global mutable state.
- No in-place mutation of shared public structures. Entry points operate on
  values or local copies and return values/errors.
- No groups == one group. Single-group models use the same block/group shape
  as multi-group models.
- Lavaan-shaped partables are a boundary format for oracle comparison,
  interchange, and compatibility projection. Core work should prefer the model
  triple plus explicit derived structures.
- Extension points remain concepts and free functions, not virtual hot-path
  interfaces.
