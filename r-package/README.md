# magmaan R bindings

These bindings keep the exported user namespace intentionally small. The
friendly helpers compose a staged SEM workflow, while C++-shaped primitives are
available through the `magmaan_core` object for interactive methods work.

Convenience helpers are limited to R-side composition:

- `model_spec()` calls the parser/lavaanify wrapper and stores the syntax plus
  lavaanify options, including `model_type = "growth"` for lavaan-style linear
  growth defaults.
- `df_to_data()` selects model variables from a data frame, handles optional
  grouping, and calls the C++ raw-data sample-statistics wrapper.
- `magmaan(model, data, estimator, groups)` is the high-level estimate-only
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
  stationary case.
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
  the ordinary categorical MI/score path.
  `fit_measures(fit, fmg = ...)` reports the ordinary fit-measure set and can
  attach Foldnes-Moss-Gronneberg (FMG) robust p-value diagnostics.

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
`magmaan_core$inference_rls_chi2_sample(sample_stats, implied)`,
`magmaan_core$robust_build_u_factor_parts(partable, sample_stats, theta)`, and
`magmaan_core$robust_reduced_gamma_sample_zc(...)` /
`magmaan_core$robust_reduced_gamma_sample_gamma(...)`,
`magmaan_core$robust_test_moments_both_breads_zc(...)` /
`magmaan_core$robust_test_moments_both_breads_gamma(...)`, and
`magmaan_core$robust_se*_parts(...)`. Fit-list calls remain available in
`magmaan_core`, with explicit `*_fit` aliases for scripts that prefer
adapter-style names.

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
  p-value, df, source statistic (`base = "ml"` or `"rls"`), method, parameter,
  UG flag, chi-square-equivalent diagnostic, truncation count, and list-columns
  for the UGamma spectrum and method-specific lambda vectors.
- `fit_measures(fit, fmg = TRUE)` attaches the default FMG table to the
  ordinary fit-measure list. Passing a character vector to `fmg` chooses the
  exact FMG tests.
- `fmg_pvalues(fit, data = NULL, tests = ...)` remains as the compatibility
  named-vector view over `fmg_tests()`.
- Fits built from `magmaan(..., data.frame, estimator = "ML")` or
  `fit_ml(model, df_to_data(...))` retain the listwise-complete raw blocks in
  `fit$raw_data`, so FMG calls normally do not need a separate `data` argument.
  Sample-stat-only fits can still pass complete raw `data =` explicitly.
- Current support is complete-data ML for single- and multi-group fits,
  including mean structures, plus FIML/missing-data fits with retained
  `fit$raw_data`. Under FIML only the ML/LRT base and biased Gamma are defined;
  explicit `_rls` and `_ug` labels are rejected. Listwise-deleted input is
  supported only after it has become complete-data sample moments through
  `df_to_data(..., missing = "listwise")`.
- Nested FIML model-pair tests are available through
  `nestedTest(..., method = "restriction_map")` / `robust_nested_lrt()` when
  both fits are FIML fits from the same raw-data shape and mask. Complete-data
  SB2001/SB2010 compatibility methods, caller-supplied `data =`, and mixed
  FIML/complete-data pairs are rejected for this path.
- The test-name grammar mirrors semTests-style labels:
  `std`, `sb`, `ss`, `sf`, `all`, `pall`, `eba<j>`, `peba<j>`, and
  `pols<gamma>`, with optional `_ug` and `_ml` / `_rls` suffixes. The default
  source statistic is RLS; bare `peba` and `pols` use parameter 2.

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
  ordinary means, variances, and covariances. Mixed models currently require
  `meanstructure = TRUE` so lavaan-style categorical WLS moment order is
  explicit.
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
- The high-level `magmaan()` helper dispatches to the same all-ordinal or
  mixed path for `estimator = "DWLS"` / `"WLS"` when `ordered =` is supplied
  with a data frame.
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
  matrix.
- Mixed robust reporting is explicit too:
  `magmaan_core$infer_mixed_ordinal_robust(fit, mixed_stats, weight = "")`.
  Mixed `NACOV`/weight and robust reporting parity is still looser than the
  all-ordinal path. Empty `weight` reuses `fit$estimator`.
