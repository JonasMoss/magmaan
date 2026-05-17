# magmaan R bindings

These bindings keep the exported user namespace intentionally small. The
friendly helpers compose a staged SEM workflow, while C++-shaped primitives are
available through the `magmaan_core` object for interactive methods work.

Convenience helpers are limited to R-side composition:

- `model_spec()` calls the parser/lavaanify wrapper and stores the syntax plus
  lavaanify options.
- `df_to_data()` selects model variables from a data frame, handles optional
  grouping, and calls the C++ raw-data sample-statistics wrapper.
- `fit_ml()`, `fit_uls()`, `fit_gls()`, and `fit_wls()` accept those helper
  objects, then delegate to the matching C++ fit wrapper.
- `magmaan(model, data, estimator, groups)` is the high-level estimate-only
  convenience. It parses/lavaanifies syntax strings, builds sample statistics
  or FIML raw-data objects from data frames, and dispatches to the matching
  point-estimation wrapper. SEs, robust corrections, fit measures, defined
  parameters, and nested tests remain explicit post-fit calls. Lavaan-style
  `se = "none"` and `test = "none"` are accepted to make point-estimate-only
  workflows explicit; other values error.
- `compute_defined(model, fit, vcov)` evaluates `:=` rows after fitting. The
  caller supplies the covariance matrix explicitly, so expected/observed/robust
  covariance choices stay visible.

Low-level functions such as `lavaan_lavaanify()`, `model_matrix_rep()`,
`fit_fit()`, `fit_*_impl()`, `data_sample_stats_from_raw()`, and the `infer_*`
family are available as `magmaan_core$...` entries so the C++ architecture is
still directly inspectable from R without flooding ordinary tab completion.
Model-dependent post-fit helpers expose primitive-shaped entry points such as
`magmaan_core$infer_vcov_partable(info, partable)`,
`magmaan_core$infer_z_test_theta(theta, se)`,
`magmaan_core$infer_wald_test_theta(theta, R, vcov)`,
`magmaan_core$infer_rls_chi2_sample(sample_stats, implied)`,
`magmaan_core$infer_build_u_factor_parts(partable, sample_stats, theta)`, and
`magmaan_core$infer_robust_se*_parts(...)`. Fit-list calls remain available in
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

## Ordinal LS boundary

Ordinal support is intentionally narrow and mirrors the C++ delta-path:

- Declare ordered indicators with
  `model_spec(model, ordered = ..., parameterization = "delta")`.
  `"delta"` is the only accepted ordinal parameterization at the R boundary.
- Build sample statistics with `data_ordinal_stats_from_df()`. In this v1
  path, every observed model variable must be ordered; mixed
  continuous/ordinal data and polyserial correlations are not inferred in R.
- Missing observed values are handled listwise by default. Use
  `missing = "error"` to reject missing observed values instead.
- Empty ordinal categories are hard errors. Near-empty but nonempty categories
  are allowed when the C++ sample-stat builder can produce finite thresholds,
  polychorics, `NACOV`, and weights.
- Returned ordinal data includes `thresholds`, polychoric `R`, `moments`,
  `NACOV`, `W_dwls`, and `W_wls`. `moments[[b]]` is ordered as thresholds
  first, then lower-triangle polychorics by columns, and all covariance/weight
  matrices use that same row/column order.
- Fit with `magmaan_core$fit_dwls_ordinal()` or
  `magmaan_core$fit_wls_ordinal()`. These are
  point-estimate and standard chi-square statistic workflows.
- Experimental robust moment builders are opt-in on the data step:
  `magmaan_core$data_ordinal_stats_from_df(..., robust = "h_weighted")`,
  `robust = "dpd"`, or
  `magmaan_core$data_mixed_ordinal_stats_from_df(..., polyserial = "dpd")`.
  The mixed path also exposes the experimental Pearson-residual clipping
  comparator with
  `polyserial = "huber_residual"` and `clip = "hard_huber"`,
  `"pseudo_huber"`, `"tukey_biweight"`, or `"none"`. In that path,
  ordinal-containing threshold/correlation/polyserial rows are rebuilt from
  clipped residual influence; continuous-only moments remain ordinary.
  Defaults remain the lavaan-compatible ML moment builders.
- Robust ordinal reporting is explicit: call
  `magmaan_core$infer_ordinal_robust(fit, ordinal_stats, weight = "")` after a
  DWLS/WLS ordinal fit to compute sandwich SEs and SB-family scaled statistics
  from the threshold-plus-polychoric `NACOV` and the selected DWLS/WLS weight
  matrix.
- Mixed continuous/ordinal categorical data now has a separate first-pass path:
  use `model_spec(..., ordered = ..., parameterization = "delta",
  meanstructure = TRUE)`, `data_mixed_ordinal_stats_from_df()`, and
  `fit_dwls_mixed_ordinal()` / `fit_wls_mixed_ordinal()`. Mixed moment values
  follow lavaan's categorical WLS order; mixed `NACOV`/weight and robust
  reporting parity is still looser than the all-ordinal path.
  Empty `weight` reuses `fit$estimator`.
