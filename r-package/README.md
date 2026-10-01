# magmaanlab: methods-development R bindings

`magmaanlab` is the compiled R package over the magmaan C++ core, for methods
development and research. The opinionated one-call package for ordinary users
is `magmaan` in [`../r-magmaan/`](../r-magmaan/), which builds on this one; see
[the two-package vision](../project/design/r-interface-vision.md).

The friendly helpers compose a staged SEM workflow, while C++-shaped primitives
are available through the `magmaan_core` object for interactive methods work.

Convenience helpers are limited to R-side composition:

- `eqs_model()` parses explicit EQS equations, variances and covariances into
  the same model specification. Supply `observed_names` in EQS data-column
  order to map V-number variables. The initial single-group continuous subset
  and validation limits are documented in the [EQS contract](../project/grammar/eqs.md).

- `model_spec()` calls the parser/lavaanify wrapper and stores the syntax plus
  lavaanify options, including `model_type = "growth"` for lavaan-style linear
  growth defaults. `meanstructure = "default"` enables means for grouped models;
  ordered variables and explicit intercept syntax imply a mean structure as in
  lavaan. Explicit `FALSE` keeps continuous grouped models covariance-only.
  A saved spec remembers an omitted default when groups or ordered variables
  are supplied later. FIML/ML2S fitting enables their required mean structure.
- `df_to_data()` selects model variables from a data frame, handles optional
  grouping, and calls the C++ raw-data sample-statistics wrapper.
- `fit_model(model, data, estimator, groups)` is the high-level estimate-only
  convenience. It parses/lavaanifies syntax strings, builds sample statistics
  or FIML raw-data objects from data frames, and dispatches to the matching
  point-estimation wrapper. It returns a `magmaan_fit` list with the raw
  primitive fit fields plus the source model spec, syntax, estimator options,
  ordered variables, parameterization, and grouping metadata. SEs, robust
  corrections, fit measures, defined parameters, and nested tests remain
  explicit post-fit calls. Lavaan-style `se = "none"` and `test = "none"` are
  accepted to make point-estimate-only workflows explicit; other values error.
  Fit lists include `optimizer_status` and `grad_norm` so methods work can
  distinguish clean stationary convergence from usable but salvaged or singular
  optimizer exits. The `converged` boolean is true only for the clean
  stationary case. Continuous normal-theory fits also expose the post-fit
  reduced-LISREL covariance audit at
  `fit$diagnostics$admissibility`; `print()` reports its overall status and an
  inadmissible high-level fit emits one warning without changing `converged`.
- `compute_defined(model, fit, vcov)` evaluates `:=` rows after fitting. The
  caller supplies the covariance matrix explicitly, so expected/observed/robust
  covariance choices stay visible.
- Friendly post-fit wrappers keep routine inspection explicit without forcing
  callers through `magmaan_core`: `standardized(fit, vcov, type)` requires the
  caller-supplied covariance matrix, `stats::residuals(fit, standardized)`,
  `factor_scores(fit, data, method)` requires complete raw data, and
  `modification_indices(fit, data, candidates)` / `score_tests(fit, data)`
  forward to the scaffold primitives. Categorical fit objects retain the
  ordinal/mixed-ordinal stats object used for fitting, so `data` is optional for
  the ordinary categorical MI/score path. Factor scores use
  regression/Bartlett for continuous fits and EBM/ML/EAP for ordinal or
  mixed-ordinal fits; EBM is the categorical default. For one-factor
  ordinal/mixed EAP scores, `factor_score_precision(fit, data)` reports
  posterior variance/SE, sample-moment PRMSE, and the direct concrete ordinal
  reliability `1 - mean(Var(Z | y)) / Var(Z)`.
  `fit_measures(fit, fmg = ...)` reports the ordinary fit-measure set and can
  attach Foldnes-Moss-Gronneberg (FMG) robust p-value diagnostics.
- `vcov(fit, regime = ...)` names the covariance formula explicitly:
  `information_expected` / `information_observed` are inverse information
  matrices (ML and FIML), while `sandwich_expected` / `sandwich_observed`
  use empirical meat with the named bread. Sandwiches use retained raw data
  when `data` is omitted; categorical fits use retained categorical statistics.
  FIML uses its own retained data and rejects replacement data. Non-iterative
  CFA uses `delta_nt` / `delta_empirical`; SAM exposes `stored` covariance.
  Unsupported combinations error. Omitting `regime` preserves the old defaults:
  expected-bread sandwich for continuous/categorical iterative fits, observed
  inverse information for FIML, normal-theory delta for non-iterative CFA, and
  stored covariance for SAM. Legacy `model` and `robust` aliases retain their
  old estimator-dependent meanings; prefer the explicit names.
- FIML methods work can compare all three lavaan information conventions with
  `magmaan_core$inference_fiml_information_vcov(fit)`. It returns expected
  Fisher, observed-H1, and full observed-Hessian information; each carries a
  model-based covariance and an empirical-score sandwich covariance built
  from the same meat. `vcov()`'s observed regimes use the full Hessian;
  observed-H1 remains available through this diagnostic primitive.

Low-level functions such as `compat_lavaan_lavaanify()`,
`model_matrix_rep()`, `estimate_fit()`, `estimate_*()`,
`data_sample_stats_from_raw()`, `data_ordinal_stats_from_raw()`,
`data_mixed_ordinal_stats_from_raw()`, and the `inference_*` / `robust_*`
families are available as `magmaan_core$...` entries so the C++ architecture is
still directly inspectable from R without flooding ordinary tab completion.
Frontier structured-Gamma helpers are similarly explicit:
`magmaan_core$estimate_structured_gamma()` returns the raw MI4 Gamma matrix, and
`magmaan_core$estimate_structured_gamma_weight()` returns its direct inverse for
continuous WLS when it is already positive definite.
Older spellings such as `lavaan_lavaanify()`, `fit_fit()`, `fit_*_impl()`,
method-specific ordinal data builders, and `infer_*` remain available as
compatibility aliases during exploration, but they are no longer listed in
`attr(magmaan_core, "groups")` or the compact `magmaan_core` printout.
Model-dependent post-fit helpers expose primitive-shaped entry points such as
`magmaan_core$inference_vcov_partable(info, partable)`,
`magmaan_core$inference_z_test_theta(theta, se)`,
`magmaan_core$inference_wald_test_theta(theta, R, vcov)`,
`magmaan_core$inference_nt_moment_quadratic_sample(sample_stats, implied)`,
`magmaan_core$robust_build_u_factor_parts(partable, sample_stats, theta)`, and
`magmaan_core$robust_reduced_gamma_sample_zc(...)` /
`magmaan_core$robust_reduced_gamma_sample_gamma(...)`,
`magmaan_core$robust_test_moments_both_breads_zc(...)` /
`magmaan_core$robust_test_moments_both_breads_gamma(...)`, and
`magmaan_core$robust_se*_parts(...)`. Fit-list calls remain available in
`magmaan_core`, with explicit `*_fit` aliases for scripts that prefer
adapter-style names.

## Reusable model, data and weights

For repeated estimation, use the native prepared interface:

```r
model <- prepare_model("f =~ x1 + x2 + x3 + x4")
data <- prepare_data(model, df)
fit <- estimate(model, data, estimator = "ML")

# A new replication reuses the model, but prepares its own data and weights.
data2 <- prepare_data(model, df2)
weight2 <- prepare_weight(data2, "DWLS")
fit2 <- estimate(model, data2, weight = weight2)
```

`prepare_model()` retains the native model triple and matrix representation.
`prepare_data()` builds per-dataset moments or a FIML missingness pack, without
estimation weights. `prepare_weight()` owns the weight separately. `estimate()`
refreshes data-dependent starts, fits and audits; inference remains explicit.
Returned fits use the existing fit-list interface.

| Data kind | Prepared estimators |
|---|---|
| `"moments"` (default for continuous models) | ML, ULS, GLS, WLS, DWLS |
| `"raw"` (requires a model with `meanstructure = TRUE`) | FIML |
| `"ordinal"` (all observed variables ordered) | ULS, DWLS, WLS; delta/theta |
| `"mixed"` (some observed variables ordered) | DWLS, WLS; delta/theta |

Continuous data can also be supplied as `list(S = ..., nobs = ..., mean = ...)`
with N-divisor covariances. Empirical continuous weights need raw observations;
otherwise supply `W` explicitly. The prepared weight exposes `$W` for inspection
and for post-fit functions taking an explicit weight.

Fixed observed covariates are unsupported because conditional moments
(`conditional.x`) are not implemented. Such models fail explicitly. Setting
`fixed_x = FALSE` specifies a joint random-covariate model and changes the
statistical model; it does not request conditional estimation.

For categorical models, declare the category schema once:

```r
model <- prepare_model("f =~ x1 + x2 + x3 + x4",
                       ordered = names(df), prototype = df)
data <- prepare_data(model, df)
weight <- prepare_weight(data, "DWLS", full = FALSE)
fit <- estimate(model, data, weight = weight)
```

The prototype contributes category labels/order only. It contributes no empirical
thresholds or starting values. Each dataset must contain all declared categories;
changed levels, missing model variables and incompatible groups fail explicitly.
Declare group labels in `model_spec()`/`prepare_model()`; columns are matched by
name. Use `missing = "listwise"` explicitly for incomplete moment data.

For categorical weights, `full = TRUE` (the default) retains full Gamma and
influence ingredients for existing post-fit inference; DWLS does not invert the
full Gamma. `full = FALSE` builds only the DWLS diagonal. The resulting fit lacks
full-Gamma inference ingredients. These can be requested separately, without
refitting: `inference_weight <- prepare_weight(data, "DWLS", full = TRUE)` and
then, for example,
`magmaan_core$robust_ordinal(fit, inference_weight$stats)`. ULS can similarly use
`prepare_weight(data, "ULS", full = TRUE)` when full-Gamma inference is wanted.
Weight preparation currently calls the existing categorical builders, which
recompute stage-one moments; that cost is outside estimation but remains an
optimization opportunity. Measure model, data, weight, fit and inference stages
separately.

Handles are immutable and process-local: rebuild them in each worker, and after
reading serialized objects. Reusing data across models requires identical
observable order, category/group schema and mean-structure convention. A weight
is tied to its dataset and cannot silently be reused for another replication.

Existing `model_spec()`, `fit_model()`, `fit_*` and `magmaan_core$estimate_*` calls
remain supported without deprecation warnings. `model_spec()` alone retains an R
partable, not a compiled native model. Specialized ML2S, two-level, FC-SEM, SAM
and frontier paths still use their existing entry points. Deprecation and removal
wait for coverage and caller migration; see the
[interface audit and rollout status](../project/design/r-model-preparation.md).

## Sample-moment data

Complete-data ML/ULS/GLS/WLS fit wrappers accept sample moments as
`list(S = , nobs = , mean = )`:

- `S` is a covariance matrix for one group, or a list of covariance matrices
  for multiple groups. Each covariance must be square, finite, and match the
  model's observed-variable dimension.
- `nobs` is a positive integer scalar for one group, or a positive integer
  vector with one entry per group.
- `mean` is optional. When supplied, it is a length-`p` vector for one group,
  or a list of per-group vectors. Omit `mean` entirely when means are
  unavailable.
- Named covariance columns are reordered to the model's observed-variable
  order. Without names, matrices and means are assumed to already be in model
  order.
- `data_sample_stats_from_raw()` and `df_to_data()` use the N-divisor
  covariance convention expected by the C++ discrepancies. `df_to_data()` can
  rescale to `n-1` for inspection, but lavaan parity fixtures use N-divisor
  moments.

## FMG robust p-values

FMG single-model goodness-of-fit tests are exposed as a post-fit inference
family:

- `fmg_tests(fit, tests = ...)` returns one row per requested test with the
  p-value, df, source statistic (`base = "ml"`, `"rls"`, or `"ls"`), method, parameter,
  UG flag, chi-square-equivalent diagnostic, truncation count, and list-columns
  for the UGamma spectrum and method-specific lambda vectors.
- `fit_measures(fit, fmg = TRUE)` attaches the default FMG table to the
  ordinary fit-measure list. Passing a character vector to `fmg` chooses the
  exact FMG tests.
- `fmg_pvalues(fit, data = NULL, tests = ...)` remains as the compatibility
  named-vector view over `fmg_tests()`.
- Fits built from `fit_model(..., data.frame, estimator = "ML")` or
  `fit_ml(model, df_to_data(...))` retain the listwise-complete raw blocks in
  `fit$raw_data`, so FMG calls normally do not need a separate `data` argument.
  Sample-stat-only fits can still pass complete raw `data =` explicitly.
- Current support is complete-data ML and continuous ULS/GLS/WLS for single-
  and multi-group fits, including mean structures, plus FIML/missing-data and
  ML2S fits. Continuous LS uses its standard quadratic-form statistic and
  accepts `gamma = "empirical"` (default) or `"normal"`; WLS also requires the
  fitting `weight =` because fit objects do not retain a caller-supplied matrix.
  Covariance scaling remains explicit:
  `df_to_data(..., scaling = "n-1")` matches lavaan's continuous-LS convention,
  while magmaan's data-frame default is `"n"`. Under FIML only the ML/LRT base
  and biased Gamma are defined; explicit `_rls` and `_ug` labels are rejected.
  FIML uses saturated H1 information for the U projector metric. Listwise-
  deleted input is supported only after it has become complete-data sample
  moments through `df_to_data(..., missing = "listwise")`. ML2S uses the
  Stage-2 ML statistic and cached spectrum attached to the fit; like FIML, it
  rejects explicit `_rls` and `_ug` labels.
- Nested FIML model-pair tests are available through
  `nestedTest(..., method = "restriction_map")` / `robust_nested_lrt()` when
  both fits are FIML fits from the same raw-data shape and mask. The
  `satorra.bentler.2001` and `satorra.bentler.2010` compatibility methods are
  available too. FIML takes raw data from the fit; caller-supplied `data =`,
  unbiased/paired Gamma, and mixed FIML/complete-data pairs are rejected.
- ML2S model pairs support the restriction-map nested test across the NT, ULS,
  DWLS, ADF/WLS, and DLS Stage-2 weights. Scalar SB2001/SB2010 compatibility
  methods are NT-only, and unbiased/paired Gamma is not defined for ML2S.
- Complete-data ML restriction-map pairs accept both empirical and
  Browne/Du--Bentler unbiased Gamma (`gamma = "unbiased"` or `_ug` FMG
  labels). The correction is group- and mean-structure-aware; it remains
  undefined for FIML, ordinal, and continuous least-squares pairs.
  `robust_nested_lrt(..., gamma = "both")` obtains both spectra in one shared
  streaming pass: `eigenvalues` and all scalar test summaries use empirical
  Gamma, while `eigenvalues_unbiased` holds the auxiliary unbiased spectrum.
  `fmg_nested()` selects this paired path automatically when a request mixes
  empirical and `_ug` tests. Numerical backend diagnostics are retained in
  `attr(result, "warnings")`.
- Continuous ULS/GLS/WLS restriction-map pairs are available through
  `nestedTest()` / `robust_nested_lrt()` and `fmg_nested()`. ULS defaults to
  normal-theory Gamma, while GLS/WLS default to empirical Gamma; WLS requires
  the original fitting `weight =`. Their canonical FMG labels use the `_ls`
  base suffix.
- Nested FMG convenience wrappers are `fmg_nested_ordinal()` and
  `fmg_nested_mixed_ordinal()` for all-ordinal and mixed-ordinal LS pairs.
- The test-name grammar mirrors semTests-style labels:
  `std`, `sb`, `ss`, `sf`, `all`, `pall`, `eba<j>`, `peba<j>`, and
  `pols<gamma>`, with optional `_ug` and `_ml` / `_rls` suffixes. Single-model
  complete-data tests default to RLS; nested complete-data tests default to the
  ML/LRT difference; ordinal and continuous-LS tests use their LS statistic.
  Bare `peba` and `pols` use parameter 2.

## Ordinal LS boundary

Ordinal support is intentionally narrow and mirrors the C++ ordinal LS path:

- Declare ordered indicators with
  `model_spec(model, ordered = ..., parameterization = "delta")` or
  `parameterization = "theta"`. Both parameterizations are accepted for
  all-ordinal and mixed continuous/ordinal DWLS/WLS point estimation, and the
  fitted parameterization is reused by explicit post-fit robust reporting.
- For all-ordinal models, build sample statistics with
  `magmaan_core$data_ordinal_stats_from_df()`. Every observed model variable
  must be listed in `ordered`; otherwise use the mixed builder.
- For mixed continuous/ordinal models, use
  `magmaan_core$data_mixed_ordinal_stats_from_df()`. Ordered variables produce
  thresholds and categorical association rows; continuous variables contribute
  ordinary means, variances, and covariances. Declaring `ordered` in
  `model_spec()` enables the required mean structure automatically.
- Missing observed values are handled listwise by default. Use
  `missing = "error"` to reject missing observed values instead.
- Empty ordinal categories are hard errors. Near-empty but nonempty categories
  are allowed when the C++ sample-stat builder can produce finite thresholds,
  polychorics, `NACOV`, and weights.
- Returned ordinal data includes `thresholds`, polychoric `R`, `moments`,
  `NACOV`, `W_dwls`, and `W_wls`. `moments[[b]]` is ordered as thresholds
  first, then lower-triangle polychorics by columns, and all covariance/weight
  matrices use that same row/column order.
- Returned mixed data follows lavaan's categorical WLS moment order and
  includes `ordered_mask`, `thresholds`, `R`, continuous means, `moments`,
  `NACOV`, `W_dwls`, and `W_wls`.
- Simulation from pre-estimated categorical summaries uses
  `magmaan_core$sim_ordcorr_summary_calibrate(R, kinds, thresholds)` or the
  multi-group `sim_ordcorr_mg_summary_calibrate()`, then the existing
  `sim_ordcorr_draw()` / `sim_ordcorr_mg_draw()` calls. This path treats `R` as
  the latent polychoric/polyserial summary matrix and skips the
  proportions-plus-target inversion step.
- Mixed continuous/ordinal covariance shrinkage is explicit:
  `magmaan_core$shrink_mixed_ordinal_stats(x, kind = "diagonal",
  intensity = ...)` transforms the association/covariance block and rebuilds
  `moments`, `NACOV`, `W_dwls`, and `W_wls` before fitting.
- Fit all-ordinal data with `magmaan_core$fit_dwls_ordinal()` or
  `magmaan_core$fit_wls_ordinal()`. Fit mixed continuous/ordinal data with
  `magmaan_core$fit_dwls_mixed_ordinal()` or
  `magmaan_core$fit_wls_mixed_ordinal()`. These are point-estimate and
  standard chi-square statistic workflows.
- The high-level `fit_model()` helper dispatches to the same all-ordinal or
  mixed path for `estimator = "DWLS"` / `"WLS"` when `ordered =` is supplied
  with a data frame.
- Complete-data covariance-honest ML uses `frontier_fit_ml_psd(model, data)`.
  Its default `preconditioning = "diagonal"` freezes separate expected-information
  scales for original model coordinates and auxiliary covariance factors at the
  start. The returned partable, objective and PSD domain are unchanged, and
  terminal audits run in original coordinates. Set `"none"` to disable scaling;
  this option is not exposed for other estimator families. Inspect
  `fit$psd_preconditioning` for the selected mode.
  The default uses transported std.lv FABIN starts where supported and
  `nlopt-slsqp`. There is no automatic chart selection or targeted
  restart. A passing numerical verdict checks feasibility and stationarity;
  it does not certify a local or global optimum at singular boundaries.
- `frontier_fit_ml_psd_fallback(model, data)` explicitly tries ordinary
  L-BFGS first and accepts it only if both the common accuracy verdict and
  covariance admissibility pass. Otherwise it runs PSD-SLSQP once. Finite
  ordinary estimates with a valid implied covariance and satisfied model
  equalities supply the warm start; after an error or unusable return it uses
  the original initializer. A warm start transfers parameter values, not
  optimizer state. PSD factor initialization repairs covariance starts; the
  link equalities can initially be violated and are enforced during fitting.
  The two stages have separate `ordinary_control` and `psd_control` lists
  and `ordinary_optimizer` / `psd_optimizer` selections. Initial start-policy
  selection belongs in `ordinary_control$start`.
  Inspect `result$fit` for the accepted fit (NULL if neither attempt passes),
  `converged`, `fallback_used`, `fallback_reason`, and `warm_start_used`.
  `result$ordinary` and `result$psd` retain each attempt's `fit` or structured
  `error`; `psd` is NULL if skipped. Parsing/data/start-construction errors
  still raise ordinary R errors before the policy can run. This explicit
  policy leaves `fit_model()` and ordinary ML defaults unchanged, and does not
  add automatic post-fit inference.
- Covariance-honest research fits are explicit. Use
  `frontier_fit_ml2s_psd()` for saturated-EM Stage 1 followed by PSD ML or a
  fixed ULS/DWLS/ADF/DLS Stage 2, and `frontier_fit_catml_psd()` for
  normal-theory ML on a positive-definite Stage-1 polychoric matrix. Both
  preserve their Stage-1 objects; neither attaches automatic boundary
  inference.
- Experimental robust moment builders are opt-in on the data step:
  `magmaan_core$data_ordinal_stats_from_df(..., robust = "h_weighted")`,
  `robust = "dpd"`, or `robust = "huber_residual"`. The raw primitive
  `magmaan_core$data_ordinal_stats_from_raw(X, robust = ...)` uses the same
  options, with `"ml"`/`"none"` as the lavaan-compatible default. For mixed
  continuous/ordinal data, use
  `magmaan_core$data_mixed_ordinal_stats_from_df(..., polyserial = "dpd")` or
  `polyserial = "huber_residual"`; the raw primitive takes the same
  `polyserial =` option plus `ordered_mask`. The Huber residual comparator
  accepts `clip = "hard_huber"`, `"pseudo_huber"`, `"tukey_biweight"`, or
  `"none"`. In that path, ordinal-containing threshold/correlation/polyserial
  rows are rebuilt from clipped residual influence; continuous-only moments
  remain ordinary. Defaults remain the lavaan-compatible ML moment builders.
- Robust ordinal reporting is explicit: call
  `magmaan_core$infer_ordinal_robust(fit, ordinal_stats, weight = "")` after a
  DWLS/WLS ordinal fit to compute sandwich SEs and SB-family scaled statistics
  from the threshold-plus-polychoric `NACOV` and the selected DWLS/WLS weight
  matrix. For all-ordinal ULS/DWLS fits, `magmaan_core$robust_ordinal_ij`
  adds the estimated diagonal-weight infinitesimal-jackknife covariance term.
- Mixed robust reporting is explicit too:
  `magmaan_core$infer_mixed_ordinal_robust(fit, mixed_stats, weight = "")`.
  Mixed `NACOV`/weight and robust reporting parity is still looser than the
  all-ordinal path. Empty `weight` reuses `fit$estimator`.

## Continuous SNLLS contract

Use `magmaan_core$fit_uls_snlls(model, data)`, `fit_gls_snlls(model, data)`,
or `fit_wls_snlls(model, data, W)` for fixed-weight separable fits. These
paths are unbounded: supplied bounds are rejected, including variance-bound
presets. Choose the corresponding ordinary LS fit when bounds are required.
`NULL` or the friendly wrapper's `bounds = "none"` requests unbounded fitting.
Covariance admissibility is reported separately; profiling does not impose PSD.

`fit$diagnostics$geometric_stationarity` recomputes stationarity in the full
model coordinates, including the eliminated linear parameters. It is additive
and does not replace `fit$audit`, which retains the optimizer's driven-coordinate
verdict. GP uses Kaufman's approximate residual Jacobian: scalar gradients are
exact locally at fixed rank with accurate inner solves, while residual-based
backends use approximate Gauss–Newton curvature.

### Sharing NTML work across tests

For complete-data ML, use one inference dataset across the fitted models. The
native objects retain the existing NTML U-factor setup, casewise contributions,
information and each test's reduced covariance/spectrum:

```r
d <- prepare_inference_data(fit1, data)
g0 <- prepare_inference(fit0, d)
g1 <- prepare_inference(fit1, d)
h <- prepare_hypothesis(g0, g1)

# Global score and ML GOF at one fit:
qs <- inference_quadratic(g0, "score")
ql <- inference_quadratic(g0, "lr")
# Nested score and exact Satorra–2000 LR:
ns <- inference_quadratic(h, "score")
nl <- inference_quadratic(h, "lr")
calibrate_quadratic(ns, c("sb", "peba4"))
calibrate_quadratic(nl, c("sb", "peba4"))

V <- inference_covariance(g0, robust = TRUE) # reusable by wald_test()
inference_reuse(g0)                         # construction counts
```

Repeated calls reuse the native results, including spectra. SB-only calibration
uses the trace. `storage = "auto"` caches casewise moment contributions up to
64 MiB, then uses tiled projections. The large-N global path accumulates score
and GOF reductions together; the smaller row-space spectrum is used when
appropriate. Neither empirical Gamma nor full U is required. Tiled storage may
revisit raw data for distinct hypotheses; an explicit unbiased GOF request can
materialize casewise contributions for the existing correction.

This path currently supports structured expected-information geometry, affine
exact parameter nesting, interior fits and random X. Score uses likelihood
contributions at H0; LR uses its own H1-anchored reduction. Their finite-sample
spectra remain distinct. Compatible prepared fits also feed the existing GOF,
expected-information, expected sandwich-SE and exact empirical streaming LR
wrappers. Other conventions, FIML and ML2S retain their existing interfaces.
See `examples/inference_reuse.R` for parity and reuse checks.

### Reusable score and inference objects

Scores are available without running a score test. For ML/FIML, prepare an
immutable inference snapshot once and choose the subsequent work explicitly:

```r
ctx <- prepare_inference(fit, data) # omit data for FIML/ML2S
s <- scores(ctx, space = "parameter")
c <- score_components(ctx)         # global saturated complement
u <- project_scores(c, retain_rows = TRUE)
e <- score_spectrum(u)             # compute once, reuse below
calibrate_quadratic(e, c("sb", "peba2", "peba4"))
resample_scores(u, n_flips = 999, seed = 7)
```

`score_components(ctx, H1 = unrestricted_model)` constructs affine nested
ML/FIML ingredients without fitting H1. Neither component construction nor
projection computes an exact-mixture p-value. `calibrate_quadratic(u, "sb")`
uses a trace without an eigendecomposition; request `"all"` explicitly for the
exact mixture. `score_sandwich(u)` is separate, so singular empirical meat can
still support mixture calibration. `center = TRUE` on projection explicitly
centers covariance rows globally, not within missingness patterns.

`score_components_from_matrices()` accepts existing scores, rows, sensitivity,
metric and chosen nuisance/test directions for method development. It uses the
same C++ projection and downstream consumers as the fit adapters.

NT-ML2S has a separate adapter: its observed Stage-2 score is not the sum of
its Stage-1 influence rows. Regularized Stage 1 and non-NT weights remain
unsupported. Full casewise likelihood scores and parameter information use the
ML/FIML adapters; no universal likelihood-score interpretation is imposed on
other estimators.

Information and covariance can also be computed once and reused:

```r
I <- inference_information(ctx, "expected")
V <- parameter_covariance(ctx, I)   # optionally supply a total-scale meat
wald_test(ctx, R, vcov = V, q = target)
```

These matrices carry snapshot provenance, and using one with a different
snapshot is rejected. Supplied matrices and parameter vectors remain available
for methods development. Snapshots are process-local and immutable; changes to
the source fit require a new snapshot. The prepared fit context is also reused
by `fmg_tests(ctx)`, `fmg_nested(ctx1, ctx0)`, and
`robust_nested_lrt(ctx1, ctx0)`. Legacy LR/GOF routines retain their own geometry
construction; reuse their returned statistic/spectrum explicitly when only the
calibration changes:

```r
lr <- robust_nested_lrt(ctx1, ctx0)
r <- quadratic_reference(lr$T_diff, lr$df_diff, lr$eigenvalues)
calibrate_quadratic(r, c("sb", "peba2", "peba4"))
```

See `examples/scores.R` for executable parity and ownership checks, and
`?scores` for the coordinate and normalization contracts.
