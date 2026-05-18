# magmaan roadmap

This document summarizes the current implementation state and architectural
contracts for magmaan. It is not the active backlog. Remaining work lives in
[docs/todo.md](todo.md).

Out of scope for this track: Bayesian SEM, multilevel SEM, latent
interactions/mixtures, EFA, inequality constraints (and active-bound
inference), and end-user lavaan
replacement ergonomics.

## Current State

magmaan is a C++23 library for methods developers working on linear SEM. It is
built under `-fno-exceptions -fno-rtti`, Eigen runs under
`EIGEN_NO_EXCEPTIONS`, and fallible APIs return `std::expected<T, Error>`.
Extension points are concepts and free function templates rather than virtual
hot-path interfaces.

Lavaan remains the oracle. Parser output, lavaanified partables, point
estimates, standard errors, and chi-square statistics are compared against
checked-in lavaan fixtures where support is claimed. Fixture regeneration is
done with `tools/regen_oracle.R`; CI does not invoke R.

The lavaanified model contract is the triple:

- `LatentStructure`: the estimable model, name-free except where estimator or
  identification choices require structure.
- `LatentNames`: the verbal model, including variable names, labels, groups,
  group levels, and `.pN.` plabels.
- `Starts`: free-parameter start hints.

`to_lavaan_partable()` and `from_lavaan_partable()` project this model to and
from `LavaanParTable`, which is the compatibility format used by R bindings and
golden `parTable()` fixtures.

## Implemented Capabilities

### Parser, lavaanify, and matrix representation

- Lavaan-style syntax parser with normative grammar in `docs/grammar/`.
- Checked-in lexer, parser, partable, matrix, and fit oracle fixtures.
- Single- and multi-group LISREL matrix representation.
- Fixed.x resolution, mean structures, marker/std.lv/effect-coding
  identification, start hints, and linear equality constraints.
- Linear equality constraints through affine reparameterization for ML and
  penalty residuals for bounded LS.
- Nonlinear equality constraints (`a == b*c`, `b1 == (b2+b3)^2`) for the ML
  and complete-data LS paths: compiled to name-free expression trees by
  `resolve_lin_constraints`, enforced by an augmented-Lagrangian outer loop
  over L-BFGS, with the constrained vcov / df projected through the constraint
  Jacobian H(θ̂). They may be combined with linear equality constraints in the
  same model — the augmented-Lagrangian loop then runs in the
  linear-constraint-reduced α-space. FIML, ordinal, and the separable (SNLLS)
  path reject them.
- The expression sub-language shared by `:=` defined parameters and `==`
  constraints supports `+ - * / ^`, unary `+ -`, and the unary functions
  `exp` / `log`; both the defined-parameter evaluator and the
  nonlinear-constraint evaluator evaluate them with forward-mode AD.
- Effect coding for loadings.

### Complete-data ML and inference

- Normal-theory ML fitting through LBFGS. Heuristic start values sign each
  free loading by its indicator's covariance with the factor's marker, and the
  unbounded path salvages a converged iterate when the strong-Wolfe line search
  stalls in the flat neighbourhood of the optimum instead of reporting a hard
  line-search error (see `docs/convergence_diagnostics.md`).
- Expected information, finite-difference observed information, and analytic
  observed information for covariance and mean-structure models.
- Vcov/SE, Wald/z tests, chi-square/df helpers, LR/Satorra-2000 and
  Satorra-Bentler 2001/2010 compatibility nested tests, robust U-Gamma
  machinery, Satorra-Bentler-family statistics, robust SEs,
  FMG eigenvalue p-value tests (explicit method/options API, no parser),
  Browne residual NT/ADF, fixed-parameter modification indices,
  equality-release score tests, fit measures including RMSEA close-fit
  p-values, structural-aware standardization, and C++ defined-parameter
  evaluation.
- Observed-bread robust SEs and observed-Hessian U-factors use total-N scaling
  and work on block-stacked multi-block covariance and mean-structure models.
- Browne's unbiased reduced gamma has a single-block reduced-matrix shorthand
  and a casewise multi-block primitive.

### Continuous FIML

- Direct observed-pattern ML over raw continuous data with missingness masks.
- Rows are compressed into observed-value patterns; the observed-pattern
  objective and analytic gradient reuse `ModelEvaluator` Jacobians.
- `fit_fiml()` optimizes with LBFGS.
- Current checked-in fixtures cover single- and multi-group CFA, three-factor
  CFA, labeled equality CFA, latent structural models, observed-variable path
  models under random-x and complete fixed.x policies, equality-constrained
  structural regressions, dense non-monotone missingness, complete observed-row
  equivalence, multi-group fixed.x with complete exogenous variables and
  missing outcomes, explicit mean structures, and a full 25-item five-factor
  bfi real-data FIML parity case.
- Post-fit FIML extras include observed-data normal constants, saturated/H1
  likelihood, baseline/independence likelihood accounting, chi-square,
  information criteria, and fit-index inputs for the current fixture tranche.
- Robust FIML MLR post-fit reporting computes observed-pattern casewise
  sandwich SEs and Yuan-Bentler Mplus scaled-test traces for fixture-backed
  non-saturated single- and multi-group cases.
- The public fixed.x policy rejects missing observed exogenous variables rather
  than approximating lavaan's conditional likelihood behavior.
- The R boundary exposes `df_to_fiml_data()` and estimate-only `fit_fiml()`.

### Staged C++ facade

- `magmaan::api` provides a tested staged facade over the currently
  implemented core primitives. Model construction, data construction, fitting,
  standard SEs, robust reporting, tests, modification indices, score tests,
  Wald/z tests, standardization, defined parameters, fit measures, and nested
  tests remain explicit calls rather than a lavaan-style one-shot summary.
- The facade supports complete-data ML, raw continuous FIML, continuous
  ULS/GLS/explicit-weight WLS, and ordinal/mixed DWLS/WLS through the same
  underlying fit and post-fit primitives described above. Unsupported
  combinations fail with `api::ErrorStage::UnsupportedCombination`.

### Least-squares estimators

- ULS, GLS, and explicit-weight WLS discrepancies with scalar
  value/gradient and residual/Jacobian interfaces.
- Bounded LS fitting through LBFGS-B and optional Ceres.
- Automatic nonnegative variance bounds.
- Equality-penalty residuals on the LS path.
- Fixed-parameter modification indices and equality-release score tests reuse
  the estimator-specific LS residual/Jacobian weighting for ULS/GLS/WLS.
- Representative modification-index and equality-release score-test fixtures
  now pin fixed rows plus generated absent covariance rows against lavaan across
  complete ML, observed-information FIML, continuous ULS, ordinal DWLS, and
  mixed ordinal DWLS. `ModificationIndexOptions` exposes absent cross-loadings
  and covariances for those paths; generated structural-regression rows remain
  deliberately out of scope.
- Continuous LS fixtures cover point estimates, degrees of freedom, and
  estimator-specific chi-square reporting for representative CFA,
  multi-group, labeled-equality, mean-structure, and observed-exogenous
  fixed.x cases. Mean-structure LS fixtures also cross-check SNLLS against
  the full LS path, and Ceres against LBFGS-B when the Ceres backend is
  enabled.
- Continuous ULS/GLS/WLS robust adapters reuse the shared weighted-moment
  sandwich/U-Gamma primitive with either supplied Gamma blocks or raw-data
  Gamma construction. ULS `robust.sem` SEs and Satorra-Bentler-family
  statistics are lavaan-backed for the non-fixed.x continuous LS fixtures;
  GLS/WLS robust paths have shape and scaling coverage because lavaan does not
  expose matching robust scaled-test targets for those estimators.
- GLS/WLS reporting follows lavaan's `2 * N * fmin` convention; ULS
  standard chi-square is pinned to lavaan's Browne residual NT statistic,
  including lavaan's `fixed.x` convention of zeroing the fixed exogenous
  rows/columns in the NT inverse weight. ULS robust scaled-test reporting
  follows lavaan's robust LS `2 * N * fmin` base statistic.
- Continuous ULS/GLS/WLS objectives can be evaluated explicitly at any
  supplied parameter vector via `estimate::evaluate_ls_objective()`, keeping
  discrepancy evaluation separate from chi-square reporting scale. RLS exposes
  a matching theta-based `nt::infer::rls_chi2()` overload that builds implied
  moments from the model structure before applying Browne's model-based
  residual quadratic.
- Distributionally weighted least squares (DLS) is available as an explicit
  weight-matrix builder over the existing moment-quadratic LS surface. The
  fixed-scalar builder mixes covariance-moment Gamma matrices between
  normal-theory GLS and empirical ADF/WLS endpoints, while
  `empirical_bayes_dls_mixing_scalar()` estimates a global reliability-style
  scalar from the observed `Gamma_ADF - Gamma_NT` departure and casewise
  fourth-moment product noise, following the empirical-Bayes weight-selection
  direction of Du and Wu (2024). `empirical_bayes_dls_weight()` then delegates
  to the same DLS builder, so modification-index and weighted-moment sandwich
  paths consume the result exactly like any supplied WLS weight. Local
  simulation checks live under `checks/dls/`.
- Separable nonlinear least squares profiling exists for LS estimators where
  conditionally linear parameters can be profiled out.

### Ordinal and mixed categorical LS

- Threshold (`|`) and response-scale (`~*~`) parser/partable projection.
- Integer all-ordinal complete/listwise sample statistics.
- Pairwise polychoric correlations.
- Public enum-backed polychoric h-score API under `data::eval_polychoric_h_score()`
  for ML, WMA hard cap, smooth cap, and exponential cap experiments, returning
  score values, derivatives, and objective contributions while leaving the
  default lavaan-compatible ordinal sample-stat path unchanged.
- Experimental fixed-threshold all-ordinal bivariate h-weighted rho fitting
  under `data::fit_ordinal_pair_rho_h_weighted()`. It keeps thresholds fixed,
  accepts the predefined h-score options, returns rho/objective/score,
  convergence and bound diagnostics, adjusted counts, fitted probabilities,
  expected/residual/Pearson tables, and per-cell robust weights. ML and
  `WmaHardCap(k = Inf)` delegate to the existing ML rho path, preserving the
  lavaan-compatible limit; finite caps remain diagnostics/prototype machinery,
  not the default ordinal moment builder.
- Experimental pair-local all-ordinal bivariate joint h-weighted fitting under
  `data::fit_ordinal_pair_joint_h_weighted()`. It estimates both pair-local
  thresholds and rho by minimizing the h-score/minimum-disparity objective,
  reusing ordered-threshold and bounded-rho transforms. It returns pair-local
  thresholds, rho, objective/gradient diagnostics, adjusted counts, fitted
  probabilities, expected/residual/Pearson tables, and per-cell robust weights.
  ML and `WmaHardCap(k = Inf)` delegate to the existing joint ML kernel; finite
  caps are validated as experimental bivariate diagnostics, not SEM moment
  construction or calibrated robust inference.
- Experimental pair-local all-ordinal h-weighted influence diagnostics under
  `data::ordinal_pair_h_weighted_influence()`. Given integer bivariate counts,
  pair-local thresholds, and rho, it expands casewise estimating-function rows,
  returns h-score ratios/values/derivatives/diagnostic weights, analytic
  probability-Hessian bread, centered `h'(t)`-weighted meat rows, influence
  rows, score Gamma, and sandwich Gamma with `S'S / n` scaling. The WMA
  hard-cap derivative convention is pinned as `dh(t) = 1[t < k]`, so the exact
  kink uses derivative zero. This remains the robust bivariate Gamma primitive
  used for pair-level validation.
- Experimental pair-local all-ordinal density power divergence fitting under
  `data::fit_ordinal_pair_joint_dpd()`. It estimates pair-local thresholds and
  rho with DPD tuning `alpha`, delegates `alpha = 0` to joint ML, and returns
  probabilities, expected/residual/Pearson tables, and `p^alpha` attenuation
  weights. This is the main non-h-score bivariate comparator and is not yet
  wired into SEM moment/Gamma construction.
- Experimental all-ordinal SEM integration under
  `data::pairwise_ordinal_stats_h_weighted_from_integer_data()` now uses a
  shared-threshold composite h-score estimator: one threshold block per ordinal
  variable plus one polychoric correlation per pair are optimized jointly under
  the h-weighted objective. It rebuilds casewise moment influence/Gamma from
  the shared robust estimating equations and refreshes DWLS/WLS weights.
  Optional robust-R handling records the raw minimum eigenvalue and can either
  fail explicitly or repair low-eigen correlation matrices by ridge/shrinkage
  toward the identity, with the same transformation applied to the correlation
  influence columns before Gamma and weights are rebuilt.
- Experimental all-ordinal shared-threshold DPD stats are available under
  `data::pairwise_ordinal_stats_dpd_from_integer_data()`. The parameter and
  output contract matches the h-weighted SEM path, but the composite objective
  is density power divergence with tuning `alpha`; `alpha = 0` delegates to
  the default ML/lavaan-compatible stats path.
- Experimental all-ordinal shared-threshold Huberized Pearson-residual stats
  are available under
  `data::pairwise_ordinal_stats_huber_residual_from_integer_data()`. The
  clipped residual is the cell Pearson residual; hard Huber, pseudo-Huber,
  Tukey biweight, and no-clip options share the same moment/Gamma contract as
  h-weighted and DPD stats, with no-clip delegating to the default
  ML/lavaan-compatible stats path. Robust-R ridge/shrinkage repair applies the
  same correlation-column influence transformation before Gamma and weights
  are rebuilt.
- Experimental mixed continuous/ordinal Huberized Pearson-residual stats are
  available under `data::mixed_ordinal_stats_huber_residual_from_data()`.
  Hard Huber, pseudo-Huber, Tukey biweight, and no-clip residual options are
  exposed for ordinal-containing threshold/correlation/polyserial rows while
  continuous-only means, variances, and covariances remain ordinary moments.
  The path rebuilds casewise moment influence, Gamma/NACOV, and DWLS/WLS
  weights, including the single-ordinal case where threshold uncertainty enters
  only through polyserial links. Unit-level simulation checks cover the no-clip
  ML limit, contaminated polyserial tails, sparse ordinal margins, positive
  DWLS diagonals, finite Gamma conditioning, and DPD comparator stability. This
  remains a methods comparator rather than a lavaan-backed compatibility claim.
- Robust ordinal moment builders are experimental methods-developer surfaces,
  not changes to the default lavaan-compatible ordinal builders. SEM-facing
  robust builders use shared ordinal thresholds and rebuild moment influence,
  Gamma/NACOV, and DWLS/WLS weights under the robust equations. Pair-local
  threshold estimators remain diagnostics/prototypes rather than SEM moment
  constructors. Mixed robust builders currently robustify ordinal-only and
  continuous-ordinal links only; continuous marginal moments and
  continuous-continuous covariances remain ordinary unless a separate design
  reopens them. R exposes predefined robust methods only; arbitrary C++
  h-functions remain internal.
- Public all-ordinal pairwise ML kernel and `PairwiseOrdinalStats` wrapper for
  complete/listwise ordinal data, exposing pair labels, count/adjustment
  diagnostics, fitted fixed-threshold rho diagnostics, fitted expected/residual
  tables, missingness/repair diagnostics, casewise moment influence, Gamma,
  and minimum eigenvalue diagnostics while preserving the existing
  `OrdinalStats` moment/weight path.
- Bivariate ordinal observed-pair table kernel for explicit pairwise
  observed-data composite-likelihood work. `NaN` skips the pair and increments
  missing-pair diagnostics; finite observed values must be positive integer
  categories inside the declared level ranges. Observed-pair wrappers feed
  those counts into both fixed-threshold rho ML and pair-local joint
  threshold/rho ML while preserving `n_obs` and `n_missing`.
- Experimental joint bivariate ordinal ML kernel for one complete pairwise
  table, estimating pair-local nuisance thresholds and rho. This kernel backs
  the pair-local complete/listwise and observed-pair composite objective
  prototypes, but it is not wired into the lavaan-compatible SEM moment
  construction path.
- Public SEM-facing all-ordinal pairwise composite objective API under
  `estimate::pairwise_ordinal_composite_objective()`. It consumes
  `PairwiseOrdinalStats` plus SEM-implied shared thresholds and correlations,
  maps each pair to its bivariate threshold margins, exposes per-pair boundary
  diagnostics plus score/Gamma rows for the bivariate threshold/rho margin,
  and makes scaling/weighting explicit. The current reporting contract
  deliberately does not report chi-square or degrees of freedom until a
  calibrated composite-likelihood test is implemented.
- Complete/listwise all-ordinal pair-local joint composite prototype under
  `estimate::pairwise_ordinal_joint_composite_objective()`. It consumes the
  same `PairwiseOrdinalStats` diagnostics and options surface, refits every
  pair with the complete bivariate joint threshold/rho ML kernel, and returns
  pair-local thresholds, rho, adjusted counts, fitted counts, residuals,
  score contributions, score Gamma, boundary flags, and objective scaling.
  This is a saturated/reference composite target for future SEM fitting, not a
  lavaan-compatible DWLS moment builder and not a calibrated global chi-square
  test.
- Observed-pair all-ordinal joint composite prototype under
  `estimate::pairwise_ordinal_observed_joint_composite_objective()`. It takes
  ordinal data blocks with `NaN` missingness plus declared level counts, fits
  each bivariate observed-pair table independently with the joint threshold/rho
  ML kernel, preserves per-pair `n_obs`/`n_missing`, and rejects all-missing
  pairs or empty marginal categories. It exposes the same bivariate
  threshold/rho score contributions and score Gamma as the complete/listwise
  path. This is pairwise observed-data likelihood, not multivariate MAR
  ordinal FIML.
- Checked-in pairwise diagnostic fixture coverage under
  `tests/fixtures/pairwise/`: complete all-ordinal polychoric diagnostics,
  mixed pair labels and primitive ML helpers, complete/listwise joint
  composite diagnostics, and observed-pair composite missingness/count
  semantics.
- The SEM-facing ordinal moment path remains the lavaan-compatible
  shared-threshold path. Pair-local joint threshold/rho ML is reserved for
  diagnostics, robust pair experiments, and future composite likelihood rather
  than for constructing the current `OrdinalStats` moment vector.
- Shared-threshold multivariate missing ordinal modeling remains out of scope:
  the implemented missing-data ordinal path is pairwise observed-data
  composite likelihood, not multivariate MAR ordinal FIML.
- Muthen-style all-ordinal NACOV construction for thresholds plus
  polychorics.
- DWLS diagonal weights, full WLS weights, bounded ordinal LS fitting, and
  thin R wrappers for ordinal stats plus DWLS/WLS fits.
- The R ordinal data boundary exposes consolidated dispatchers:
  `data_ordinal_stats_from_raw(robust = ...)` for all-ordinal data and
  `data_mixed_ordinal_stats_from_raw(polyserial = ..., ordered_mask = ...)`
  for mixed continuous/ordinal data. Method-specific Rcpp names remain
  callable compatibility aliases rather than entries in the displayed
  `magmaan_core` data group.
- Current ordinal fixtures validate thresholds, polychoric `R`, `NACOV`,
  `WLS.V`, `WLS.VD`, free sets, point estimates, degrees of freedom, and
  chi-square statistics across representative single-group, multi-group,
  skewed, sparse, near-empty, equality-constrained, and multi-group loading
  equality cases.
- The implemented ordinal LS boundary supports both lavaan delta and theta
  parameterizations for all-ordinal and mixed continuous/ordinal DWLS/WLS point
  estimates. Theta post-fit support uses parameterization-aware threshold and
  association Jacobians for robust ordinal reporting, modification indices,
  score tests, and standardized-solution reporting.
- Explicit post-fit robust ordinal reporting returns sandwich SEs plus
  Satorra-Bentler, mean/variance-adjusted, and scaled/shifted statistics from
  the threshold-plus-polychoric moment vector. The implementation now uses a
  shared weighted-moment sandwich/U-Gamma primitive that can be reused by other
  LS moment stacks with arbitrary block weights and NACOV matrices.
- All-ordinal DWLS/WLS fit measures are exposed through
  `estimate::fit_measures_ordinal()` and `api::fit_measures()`: CFI/TLI/RMSEA
  use the categorical independence baseline over the polychoric moment stack,
  with WLS minimizing threshold nuisance residuals under the full weight
  matrix, and ordinal SRMR uses the lavaan correlation-metric denominator that
  includes zero diagonal residuals. The bfi ordinal parity fixture gates DWLS
  and WLS CFI/TLI/RMSEA/SRMR against lavaan.
- Weighted-χ² reducer formulas are shared across eigenvalue and trace-summary
  callers: Satorra-Bentler, mean/variance-adjusted, and scaled/shifted tests
  can consume either the UΓ spectrum or `(Σλ, Σλ²)` when a low-rank trick has
  already computed the traces.
- A first mixed continuous/ordinal path builds lavaan-ordered thresholds,
  continuous means/variances, polychoric/polyserial/covariance moments,
  NACOV/DWLS/WLS weights, and DWLS/WLS delta/theta fits. The lavaan-backed
  fixtures include a complete/listwise sparse 4-category boundary case; the
  continuous-ordinal covariance influence rows include the variance
  delta-method term needed for stable mixed WLS weighting.
- Covariance shrinkage is available for both continuous `SampleStats` and
  mixed continuous/ordinal `MixedOrdinalStats`. Mixed shrinkage leaves
  thresholds and continuous means in place, transforms the lower-triangle
  association/covariance block, propagates the moment transformation through
  `NACOV`, and rebuilds DWLS/WLS weights so C++ and R consume the same shrunk
  moment stack.
- Public complete-data polyserial pair kernel for mixed continuous/ordinal
  work, exposing fixed-threshold rho ML, likelihood, casewise threshold/rho
  scores, and pairwise score Gamma. The mixed sample-stat builder now reuses
  this kernel for polyserial associations.
- Experimental fixed-marginal polyserial DPD is available under
  `data::fit_polyserial_pair_rho_dpd()` and
  `data::polyserial_pair_dpd_scores()`. It keeps the shared ordinal thresholds
  and standardized continuous marginal fixed, estimates only the polyserial
  association, delegates `alpha = 0` to the ML kernel, and returns DPD
  attenuation weights plus score/Gamma/bread diagnostics.
- Experimental SEM-facing mixed polyserial DPD stats are available under
  `data::mixed_ordinal_stats_polyserial_dpd_from_data()`. The builder preserves
  the existing mixed moment order, shared ordinal thresholds, continuous
  means/variances, ordinal-ordinal polychorics, and continuous-continuous
  covariance moments, while replacing only continuous-ordinal association
  equations with fixed-marginal DPD and rebuilding NACOV/DWLS/WLS weights from
  the mixed casewise influence rows.
- Experimental pair-local full DPD polyserial fitting is available under
  `data::fit_polyserial_pair_joint_dpd()`. It jointly estimates continuous
  mean/scale, ordinal thresholds, and rho with DPD tuning `alpha`, and returns
  probabilities, joint densities, and `f(x, y)^alpha` attenuation weights. DPD
  here means density power divergence and is not part of robcat parity.
- Pair-local full polyserial DPD remains a bivariate diagnostic only and is not
  used to construct `MixedOrdinalStats`; SEM-facing robust mixed moments use
  the shared-marginal fixed-threshold contract instead.
- Public complete-data mixed pair helpers also expose continuous-continuous
  normal pair likelihood/diagnostics, casewise mean/variance/covariance scores,
  score Gamma, and labels for the exact threshold, negative-mean, variance,
  and lower-triangle pair order used by `MixedOrdinalStats`.
- The continuous normal pair likelihood is currently a mixed-pair primitive
  and benchmark against complete-data ML/FIML, not a supported standalone
  normal-data pairwise SEM estimator.
- Ordinal and mixed delta DWLS/WLS expose fixed-parameter modification indices
  and equality-release score tests over the same threshold/correlation moment
  vectors and weights used by fitting.
- Ordinal and mixed categorical entry points validate block counts, threshold
  metadata, ordered masks, moment/weight/NACOV dimensions, finite values,
  positive `n_obs`, and positive NACOV diagonals before fitting or robust
  reporting.

### R bindings and public namespace transition

- Exploratory R bindings cover lavaanify, fitting, sample-stat bundles, robust
  inference, fit measures, model implied moments, LS estimators, SNLLS, Ceres
  paths when enabled, and data-frame-to-model sample statistics.
- `magmaan(model, data, estimator, groups)` is the high-level estimate-only
  R convenience. It composes `model_spec()`, data-frame sample-stat/raw-data
  construction, and the matching point-estimation wrapper for complete-data
  ML/ULS/GLS/WLS, FIML, and ordinal/mixed DWLS/WLS where the lower-level
  inputs are available. Lavaan-style `se = "none"` and `test = "none"` are
  accepted as explicit point-estimate-only shortcuts; other values error and
  point users to explicit post-fit inference calls. It returns a `magmaan_fit`
  list: the raw primitive fit fields plus the source `model_spec`, syntax,
  estimator options, ordered-variable metadata, parameterization, and grouping
  metadata. Its print method reports only point-fit status and directs users
  to explicit post-fit primitives.
- `compute_defined(model, fit, vcov)` exposes C++ defined-parameter evaluation
  for `:=` rows through R. It keeps covariance selection explicit, supports
  chained definitions, and resolves `.pN.` plabel references using the fitted
  lavaanified model.
- Friendly R post-fit wrappers expose routine inspection without hiding
  statistical choices: `standardized(fit, vcov, type)` requires an explicit
  covariance matrix, `stats::residuals(fit, standardized)` wraps raw or
  deterministic standardized residuals, `factor_scores(fit, data, method)`
  requires complete raw data, and `modification_indices(fit, candidates)`
  forwards to the explicit scaffold primitive.
- R post-fit inference helpers now separate primitive-shaped entry points from
  fit-list adapters for the first audited slice: vcov, z tests, Wald tests,
  RLS chi-square, U-factor construction, and robust SE helpers can be called
  with explicit `partable` / `sample_stats` / `theta` pieces, while existing
  fit-list calls remain available with explicit `*_fit` aliases.
- The R sample-moment path accepts `list(S = , nobs = , mean = )`, reorders
  named covariance matrices to model observed-variable order, and rejects
  malformed group counts, non-square or wrong-sized covariance matrices,
  non-finite moments, nonpositive `nobs`, and wrong-length means before calling
  C++ fitters.
- Future public API direction is a staged, explicit workflow rather than
  lavaan-style compound estimator strings. Users should build a model, build
  data or sample moments, fit point estimates, and then explicitly request
  standard errors, robust corrections, test statistics, fit measures, defined
  parameters, or summaries. High-level objects may retain model/data/fit
  context for ergonomic chaining, but each statistical choice must remain
  inspectable.
- The planned C++ facade sits above the existing primitive namespaces as value
  objects such as `Model`, `Data`, `Fit`, `Analysis`, and `Summary`. It may
  support fluent usage, while the primitive namespaces remain the
  methods-developer surface. Convenience recipes can be added later only if
  they expand to visible choices such as estimator, moment builder, standard
  error method, test statistic, and fit-measure inputs.
- Bindings should keep the same split between a small friendly surface and an
  inspectable primitive layer. Python can expose a friendly top-level API plus a
  `magmaan.core` submodule for C++-shaped primitives. R should avoid exporting
  every primitive into the package namespace; instead, the friendly staged API
  should be exported directly and low-level functions should live behind a
  single `magmaan_core` object.
- The staged C++ facade and R `magmaan_core` surface expose the experimental
  robust ordinal moment builders explicitly: all-ordinal h-weighted,
  all-ordinal DPD, mixed continuous/ordinal fixed-marginal polyserial DPD, and
  mixed continuous/ordinal Huberized residual stats. The staged C++ facade also
  exposes all-ordinal Huberized residual stats. Default ordinal and mixed data
  builders remain lavaan-compatible ML paths.
- The R `magmaan_core` surface also exposes mixed continuous/ordinal covariance
  shrinkage as an explicit data transformation, rebuilding mixed moments,
  `NACOV`, and DWLS/WLS weights before fitting rather than hiding shrinkage
  inside estimator strings.
- The R package is intended as a methods-developer interface over the C++
  library, not a second SEM implementation.
- Standard errors, information matrices, Wald/z tests, robust corrections,
  fit measures, defined parameters, and nested tests remain explicit post-fit
  calls outside `magmaan()`.
- Primary public declarations and internal implementation now live in the
  target namespaces: `parse`, `spec`, `model`, `data`, `estimate`,
  `inference`, `robust`, `measures`, `optim`, and `compat::lavaan`. Repository
  code and R binding internals use those namespaces directly.
- Old `fit/*` and `partable/*` compatibility headers have been removed; use the
  target namespace headers directly.

### Local build workflow

- The normal C++ edit loop is the `fast` preset and `just test-fast` / `just
  test`: Debug, no sanitizers, Ceres off, with ccache and mold enabled.
- The `dev` preset is the sanitizer validation loop: Debug plus ASan/UBSan.
  Use `just test-dev` before handing back risky core changes, and prefer
  targeted `ctest --preset dev -R ...` while narrowing failures.
- The C++ doctest suite is split into labeled executables (`smoke`, `spec`,
  `estimate`, `inference`, and `ordinal`) behind the aggregate
  `magmaan_tests` target. This keeps `cmake --build --preset <p> --target
  magmaan_tests` working while allowing narrower relinks and `ctest -L`.
- R bindings are intentionally outside the default C++ loop. `just r-install`
  links the optimized non-Ceres core, `just r-install-fast` is for interactive
  wrapper work, and `just r-install-ceres` is the explicit Ceres-enabled R
  path.

#### Build-loop timings

Snapshot taken after the directory/namespace refactor (commit `11e67f4`), on a
13th-gen i7-1355U (12 threads), clang 19.1.7, with ccache and mold enabled.
Wall-clock; approximate orientation, not a benchmark.

| Loop step (`just` recipe)                   | Time   | Notes |
|---------------------------------------------|--------|-------|
| no-op `fast` build (`just fast`)            | 0.01 s | nothing changed |
| touched core TU, `fast`                     | 1.2 s  | mtime only — ccache hit, relink `libmagmaan` + test exes |
| edited core TU, `fast`                      | 7.5 s  | content change — cold clang compile of one core TU |
| touched test TU, `fast`                     | 0.5 s  | mtime only — ccache hit, relink one test exe |
| edited test TU, `fast`                      | 5.6 s  | content change — cold compile of one test TU |
| C++ suite (`just test-fast`)                | 90 s   | 394 tests; no-op build + `ctest` |
| C++ suite minus parity (`just test-quick`)  | 58 s   | 390 tests; the parity cases are ~half the full run |
| sanitizer suite (`just test-dev`)           | 287 s  | 399 ASan/UBSan tests, plus a one-time 158 s build to carry the `dev` tree past the refactor |
| R install, opt (`just r-install`)           | 12 s   | warm core; rebuilds the 5 R-glue TUs + link |
| R install, fast (`just r-install-fast`)     | 15 s   | warm core |
| R install, Ceres (`just r-install-ceres`)   | 123 s  | includes a one-time post-refactor rebuild of the Ceres core; ~15 s once warm |

The everyday inner loop (edit a core file, `just test-quick`) is about a
minute; the sanitizer suite and the Ceres R path are minutes-scale and run
deliberately rather than on every change.

### Argument-minimality sweep

One core architecture rule is that functions should take exactly the data they
need, and no more. The current C++ core mostly follows this:

- Fit results are plain `Estimates` and carry no back-pointer to the model.
- Continuous fitters take the estimable structure, matrix representation,
  sample/raw data, starts, discrepancy, and optimizer. They do not take
  parameter names.
- Information, covariance, SE, chi-square, degrees of freedom, Wald/z tests,
  robust U-Gamma reducers, and fit measures are exposed as separate primitives
  rather than as methods on one fitted-object bundle.
- Satorra-Bentler-family reducers take scalar test statistics, degrees of
  freedom, and eigenvalues or trace summaries rather than fit objects.
- The Satorra-2000 C++ helper takes the model, reparameterization matrices,
  raw-data pieces, chi-square values, degrees of freedom, and an explicit
  `A.method` option (`exact` default, `delta` for lavaan-style
  moment-Jacobian column-space restrictions).

The remaining pressure point is the R boundary. Several exported R wrappers
currently accept the transparent fit list for convenience and then unpack the
partable, sample statistics, and estimates internally. That is acceptable as an
interactive convenience while the R package is exploratory, but it is not the
final architectural ideal. Thin R wrappers should gradually mirror the C++
primitive signatures; fit-list helpers can remain as explicit convenience
adapters layered on top.

### Robust-test naming and compatibility

The core robust-test API should use statistical names for the object being
computed, not historical SEM estimator labels. The preferred core surface is:

- weighted chi-square mixture tests from explicit eigenvalues or trace
  summaries;
- moment reductions of that same mixture: mean-scaled, mean/variance-adjusted,
  and scaled/shifted;
- likelihood-ratio / nested-model helpers named for the statistical contract
  they implement, for example exact restriction-map LRT rather than a default
  public name centred on "Satorra-2000";
- robust SEs as named sandwich constructions.

Historical labels remain important for lavaan/Mplus parity, but they belong in
`compat::lavaan` or similarly explicit compatibility wrappers. This includes
estimator shortcuts such as `MLM`, `MLMV`, `MLMVS`, and `MLR`, lavaan test
labels such as `satorra.bentler`, `scaled.shifted`,
`mean.var.adjusted`, `yuan.bentler`, and `yuan.bentler.mplus`, and legacy
nested-test formulas such as Satorra-Bentler 2001/2010. The compatibility
wrappers should bind each label to the exact lavaan/Mplus bundle it denotes;
they should not expose a combinatorial menu that lets callers freely pair a
test label with unrelated bread, gamma, vcov, or scaling choices.

`yuan.bentler.mplus` is especially a compatibility target rather than a core
weighted-mixture reducer. Lavaan's MLR default computes a scalar trace
difference, H1 minus H0, relying on Mplus-style approximations such as
`A0 ~= Delta' A1 Delta` and the analogous first-order matrix. The plain
Yuan-Bentler formula instead constructs the H1 residual-space trace directly.
Both should be documented when exposed, but neither should force historical
names into the core naming scheme.

The R boundary now stages this policy with `robust_nested_lrt()` as the
friendly statistical name for nested robust likelihood-ratio work. Its default
`method = "restriction_map"` names the exact restriction-map contract, while
`"lavaan_sb2001"` and `"lavaan_sb2010"` are explicit compatibility methods.
The older `nestedTest()` spelling and lavaan labels (`"satorra.2000"`,
`"satorra.bentler.2001"`, `"satorra.bentler.2010"`) remain as compatibility
aliases during exploration. Low-level `magmaan_core` exposes both
`robust_nested_lrt_restriction_map` and `compat_lavaan_nested_lrt_*` aliases
over the same C++ primitives so methods scripts can choose the intended
surface directly.

### Testing and validation

Validation has three deliberately separate surfaces:

- **Corpus golden tests** — breadth. 26 small synthetic models in
  `tests/fixtures/corpus.json` exercise every parser, lavaanify, matrix, fit,
  and inference stage against checked-in lavaan fixtures. Oracle:
  `tools/regen_oracle.R`.
- **Parity golden tests** — depth. `tests/golden/lavaan_parity_golden_test.cpp`
  gates magmaan against lavaan on the real-data benchmark cases
  (HolzingerSwineford1939, PoliticalDemocracy, Demo.growth, bfi, Mplus ex5.1):
  real sample sizes, real conditioning, real SEs and fit measures, and the
  raw-data ingestion path. The FIML tranche includes both a quick two-factor
  bfi missing-data sentinel with robust MLR post-fit reporting and a full
  25-item five-factor bfi missing-data convergence/global-fit gate.
  Self-contained fixtures live in
  `tests/fixtures/parity/`; oracle: `tools/regen_parity_fixtures.R`.
- **Benchmarks** — `benchmarks/` fits *live lavaan* on every run and gates
  timing, not CI correctness. It is R-dependent and advisory; the parity layer
  is the bridge that freezes its correctness checks into the gated C++ suite.

The suite builds as six doctest executables — `magmaan_test_{smoke, spec,
estimate, inference, ordinal, parity}` — so areas build and run independently
(`ctest -L parity`). CI never invokes R; fixture regeneration is a manual
developer step. Property and boundary tests are expected to catch structural
mistakes early, before they surface as hard-to-debug parity failures.

## Current Boundaries

- Complete-data and FIML support are fixture-backed only for the documented
  estimator/model slices.
- The complete-data ML parity layer includes the `lavaan::growth()` defaults
  for linear latent growth models: observed-variable intercepts are fixed to
  zero, latent growth-factor means are freely estimated, and the relevant
  growth-factor covariance is auto-added. The R boundary exposes this through
  `model_spec(..., model_type = "growth")`.
- FIML with missing observed exogenous variables under `fixed.x = TRUE`
  remains unsupported.
- FIML score tests and modification indices use observed-pattern gradients
  plus a finite-difference observed information matrix; fixture parity is
  against lavaan's observed-information score output for the covered fixed-row
  and equality-release cases.
- Delta and theta parameterizations are supported for all-ordinal and mixed
  continuous/ordinal DWLS/WLS point estimates. Ordinal robust reporting,
  modification indices, score tests, and standardized reporting use the fitted
  parameterization; remaining lavaan parity is fixture-backed for the covered
  ordinal slices and smoke-tested where lavaan fixture coverage is not yet
  available.
- WLS ordinal point estimates and standard chi-square are lavaan-backed.
  Robust WLS scaled-test reporting remains shape-only because lavaan rejects
  Satorra-Bentler-family `test=` requests with `estimator = "WLS"` for the
  current categorical fixtures; DWLS robust reporting remains lavaan-backed.
- Continuous LS robust ULS reporting is lavaan-backed for non-fixed.x cases;
  fixed.x robust LS still follows lavaan's conditional exogenous bookkeeping
  and is not part of the generic continuous adapter coverage.
- Mixed categorical NACOV/weight, fit, and robust-reporting parity remains
  looser than the all-ordinal path, but now includes a sparse listwise
  4-category boundary fixture. Pairwise polyserial ML, fixed-marginal
  polyserial DPD, pair-local full DPD polyserial diagnostics, and
  continuous-normal score/Gamma primitives are exposed for follow-on inference
  work. The SEM-facing mixed polyserial DPD path robustifies only
  continuous-ordinal associations; robust continuous marginals and
  continuous-continuous covariances remain outside that contract.
- Observed-pair ordinal table kernels are pair-level primitives only. Missing
  ordinal SEM estimation is limited to the explicit pairwise observed-data
  composite prototype; shared-threshold multivariate missing ordinal modeling
  remains unsupported and requires a separate design note before
  implementation.
- Browne residual ADF is complete-data only.
- Score-test EPC is raw, unstandardized EPC only; standardized EPC and absent
  row-generation helpers are not yet part of the public contract.
- Nonlinear *equality* constraints are supported for ML and complete-data LS
  (augmented-Lagrangian fit, Jacobian-projected vcov/df), including in
  combination with linear equality constraints in the same model — but not for
  FIML, ordinal, or the separable SNLLS path; those combinations fail
  explicitly.
- Inequality constraints (`<` / `>`) and active-bound inference remain
  unsupported: inequality-constrained estimation needs boundary
  (chi-bar-squared) asymptotics magmaan does not implement. They fail with an
  explicit early error rather than silently reporting ordinary χ²/SE theory.

## Design Invariants

- No global mutable state.
- No in-place mutation of shared public structures; entry points operate on
  values or local copies and return values/errors.
- No groups == one group. Single-group models use the same block/group shape
  as multi-group models.
- Unsupported statistical combinations fail with an explicit error rather than
  silently approximating nearby lavaan-compatible behavior.
- Lavaan-shaped partables are boundary formats for oracle comparison,
  interchange, and compatibility projection. Core work should prefer the model
  triple plus explicit derived structures.
- Extension points remain concepts and free functions, not virtual hot-path
  interfaces.
- Parser behavior follows `docs/grammar/grammar.ebnf`; if parser code and the
  EBNF disagree, the parser is wrong.
- Function signatures should be argument-minimal. A computation should receive
  the smallest structure or primitive values it needs: no names for numeric
  fitters, no fitted object for degrees of freedom, no full fit list where a
  parameter vector or sample-size vector is enough. Convenience adapters may
  unpack larger objects at the boundary, but core APIs should not depend on
  those bundles.

## Planning Documents

Use this file to understand the current state and contracts before structural
changes. Use [docs/todo.md](todo.md) to choose or update remaining work. See
[docs/lavaan_tutorial_parity.md](lavaan_tutorial_parity.md) for the
section-by-section audit of magmaan against the lavaan tutorial.
