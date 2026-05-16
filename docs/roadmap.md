# magmaan roadmap

This document summarizes the current implementation state and architectural
contracts for magmaan. It is not the active backlog. Remaining work lives in
[docs/todo.md](todo.md).

Out of scope for this track: Bayesian SEM, multilevel SEM, latent
interactions/mixtures, EFA, nonlinear constraints, and end-user lavaan
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
- Effect coding for loadings.

### Complete-data ML and inference

- Normal-theory ML fitting through LBFGS. Heuristic start values sign each
  free loading by its indicator's covariance with the factor's marker, and the
  unbounded path salvages a converged iterate when the strong-Wolfe line search
  stalls in the flat neighbourhood of the optimum instead of reporting a hard
  line-search error (see `docs/convergence_diagnostics.md`).
- Expected information, finite-difference observed information, and analytic
  observed information for covariance and mean-structure models.
- Vcov/SE, Wald/z tests, chi-square/df helpers, LR/Satorra-2000 nested tests,
  robust U-Gamma machinery, Satorra-Bentler-family statistics, robust SEs,
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
  missing outcomes, and explicit mean structures.
- Post-fit FIML extras include observed-data normal constants, saturated/H1
  likelihood, baseline/independence likelihood accounting, chi-square,
  information criteria, and fit-index inputs for the current fixture tranche.
- Robust FIML MLR post-fit reporting computes observed-pattern casewise
  sandwich SEs and Yuan-Bentler Mplus scaled-test traces for fixture-backed
  non-saturated single- and multi-group cases.
- The public fixed.x policy rejects missing observed exogenous variables rather
  than approximating lavaan's conditional likelihood behavior.
- The R boundary exposes `df_to_fiml_data()` and estimate-only `fit_fiml()`.

### Least-squares estimators

- ULS, GLS, and explicit-weight WLS discrepancies with scalar
  value/gradient and residual/Jacobian interfaces.
- Bounded LS fitting through LBFGS-B and optional Ceres.
- Automatic nonnegative variance bounds.
- Equality-penalty residuals on the LS path.
- Fixed-parameter modification indices and equality-release score tests reuse
  the estimator-specific LS residual/Jacobian weighting for ULS/GLS/WLS.
- Representative fixed-row modification-index and equality-release score-test
  fixtures now pin the shared score surface against lavaan across complete ML,
  observed-information FIML, continuous ULS, ordinal DWLS, and mixed ordinal
  DWLS; absent-row generation and standardized EPC remain future work.
- Continuous LS fixtures cover point estimates, degrees of freedom, and
  estimator-specific chi-square reporting for representative CFA,
  multi-group, labeled-equality, mean-structure, and observed-exogenous
  fixed.x cases.
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
  returns h-score ratios/values/derivatives/weights, finite-difference bread,
  influence rows, score Gamma, and sandwich Gamma with `S'S / n` scaling. The
  WMA hard-cap derivative convention is pinned as `dh(t) = 1[t < k]`, so the
  exact kink uses derivative zero. This is the robust pairwise Gamma primitive;
  SEM moment integration remains a follow-on step.
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
- Current ordinal fixtures validate thresholds, polychoric `R`, `NACOV`,
  `WLS.V`, `WLS.VD`, free sets, point estimates, degrees of freedom, and
  chi-square statistics across representative single-group, multi-group,
  skewed, sparse, near-empty, and equality-constrained cases.
- The implemented ordinal boundary is lavaan's delta parameterization.
- Explicit post-fit robust ordinal reporting returns sandwich SEs plus
  Satorra-Bentler, mean/variance-adjusted, and scaled/shifted statistics from
  the threshold-plus-polychoric moment vector. The implementation now uses a
  shared weighted-moment sandwich/U-Gamma primitive that can be reused by other
  LS moment stacks with arbitrary block weights and NACOV matrices.
- A first mixed continuous/ordinal path builds lavaan-ordered thresholds,
  continuous means/variances, polychoric/polyserial/covariance moments,
  NACOV/DWLS/WLS weights, and DWLS/WLS delta fits.
- Public complete-data polyserial pair kernel for mixed continuous/ordinal
  work, exposing fixed-threshold rho ML, likelihood, casewise threshold/rho
  scores, and pairwise score Gamma. The mixed sample-stat builder now reuses
  this kernel for polyserial associations.
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
  point users to explicit post-fit inference calls.
- `compute_defined(model, fit, vcov)` exposes C++ defined-parameter evaluation
  for `:=` rows through R. It keeps covariance selection explicit, supports
  chained definitions, and resolves `.pN.` plabel references using the fitted
  lavaanified model.
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
- The R package is intended as a methods-developer interface over the C++
  library, not a second SEM implementation.
- Standard errors, information matrices, Wald/z tests, robust corrections,
  fit measures, defined parameters, and nested tests remain explicit post-fit
  calls outside `magmaan()`.
- Primary public declarations and internal implementation now live in the
  target namespaces: `spec`, `lavaan`, `estimate`, `optim`, `nt`, `gls`, and
  `data`. Repository code and R binding internals use those namespaces
  directly.
- Old `fit/*` and `partable/*` headers remain tested compatibility shims for
  one transition window; they re-export target namespace names but no longer
  own the primary API definitions.

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
  freedom, and eigenvalues rather than fit objects.
- The Satorra-2000 C++ helper takes the model, reparameterization matrices,
  raw-data pieces, chi-square values, and degrees of freedom explicitly.

The remaining pressure point is the R boundary. Several exported R wrappers
currently accept the transparent fit list for convenience and then unpack the
partable, sample statistics, and estimates internally. That is acceptable as an
interactive convenience while the R package is exploratory, but it is not the
final architectural ideal. Thin R wrappers should gradually mirror the C++
primitive signatures; fit-list helpers can remain as explicit convenience
adapters layered on top.

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
  raw-data ingestion path. Self-contained fixtures live in
  `tests/fixtures/parity/`; oracle: `tools/regen_parity_fixtures.R`.
- **Benchmarks** — `benchmarks/` fits *live lavaan* on every run and gates
  timing, not CI correctness. It is R-dependent and advisory; the parity layer
  is the bridge that freezes its correctness checks into the gated C++ suite.

The suite builds as six doctest executables — `magmaan_test_{smoke, spec,
estimate, inference, ordinal, parity}` — so areas build and run independently
(`ctest -L parity`). CI never invokes R; fixture regeneration is a manual
developer step.

## Current Boundaries

- Complete-data and FIML support are fixture-backed only for the documented
  estimator/model slices.
- The `lavaan::growth()` identification defaults (observed-variable intercepts
  fixed to zero, latent growth-factor means freely estimated) are not
  replicated. Latent growth models fit, but the mean-structure rows must be
  spelled out explicitly in the model syntax; magmaan has no `growth()`-
  equivalent option, so for the same syntax its free set differs from lavaan's
  (`demo_growth_linear` is a tracked parity gap — see `docs/todo.md`).
- FIML with missing observed exogenous variables under `fixed.x = TRUE`
  remains unsupported.
- FIML score tests and modification indices use observed-pattern gradients
  plus a finite-difference observed information matrix; fixture parity is
  against lavaan's observed-information score output for the covered fixed-row
  and equality-release cases.
- Theta parameterization for ordinal models remains unsupported.
- The ordinal parameterization boundary accepts `delta` only for estimation;
  requested `theta` models fail explicitly instead of falling through to
  delta-shaped response-scale assumptions.
- WLS ordinal point estimates and standard chi-square are lavaan-backed.
  Robust WLS scaled-test reporting remains shape-only because lavaan rejects
  Satorra-Bentler-family `test=` requests with `estimator = "WLS"` for the
  current categorical fixtures; DWLS robust reporting remains lavaan-backed.
- Continuous LS robust ULS reporting is lavaan-backed for non-fixed.x cases;
  fixed.x robust LS still follows lavaan's conditional exogenous bookkeeping
  and is not part of the generic continuous adapter coverage.
- Mixed categorical NACOV/weight, fit, and robust-reporting parity is
  intentionally looser than the all-ordinal path, but pairwise polyserial and
  continuous-normal score/Gamma primitives are exposed for follow-on inference
  work.
- Observed-pair ordinal table kernels are pair-level primitives only. Missing
  ordinal SEM estimation is limited to the explicit pairwise observed-data
  composite prototype; shared-threshold multivariate missing ordinal modeling
  remains unsupported and requires a separate design note before
  implementation.
- Browne residual ADF is complete-data only.
- Score-test EPC is raw, unstandardized EPC only; standardized EPC and absent
  row-generation helpers are not yet part of the public contract.
- Nonlinear equality constraints and inequality constraints are unsupported.
- Active bound or inequality inference must not silently report ordinary
  chi-square/SE theory.

## Design Invariants

- No global mutable state.
- No in-place mutation of shared public structures; entry points operate on
  values or local copies and return values/errors.
- No groups == one group. Single-group models use the same block/group shape
  as multi-group models.
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
changes. Use [docs/todo.md](todo.md) to choose or update remaining work.
