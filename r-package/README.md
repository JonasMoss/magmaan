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
  point-estimation wrapper. SEs, robust corrections, fit measures, defined
  parameters, and nested tests remain explicit post-fit calls. Lavaan-style
  `se = "none"` and `test = "none"` are accepted to make point-estimate-only
  workflows explicit; other values error.
- `compute_defined(model, fit, vcov)` evaluates `:=` rows after fitting. The
  caller supplies the covariance matrix explicitly, so expected/observed/robust
  covariance choices stay visible.

Low-level functions such as `compat_lavaan_lavaanify()`,
`model_matrix_rep()`, `estimate_fit()`, `estimate_*()`,
`data_sample_stats_from_raw()`, and the `inference_*` / `robust_*` families are
available as `magmaan_core$...` entries so the C++ architecture is still
directly inspectable from R without flooding ordinary tab completion. Older
spellings such as `lavaan_lavaanify()`, `fit_fit()`, `fit_*_impl()`, and
`infer_*` remain available as compatibility aliases during exploration.
Model-dependent post-fit helpers expose primitive-shaped entry points such as
`magmaan_core$inference_vcov_partable(info, partable)`,
`magmaan_core$inference_z_test_theta(theta, se)`,
`magmaan_core$inference_wald_test_theta(theta, R, vcov)`,
`magmaan_core$inference_rls_chi2_sample(sample_stats, implied)`,
`magmaan_core$robust_build_u_factor_parts(partable, sample_stats, theta)`, and
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
  `robust = "dpd"`, or
  `magmaan_core$data_mixed_ordinal_stats_from_df(..., polyserial = "dpd")`.
  The mixed path also exposes the experimental Pearson-residual clipping
  comparator with `polyserial = "huber_residual"` and `clip = "hard_huber"`,
  `"pseudo_huber"`, `"tukey_biweight"`, or `"none"`. In that path,
  ordinal-containing threshold/correlation/polyserial rows are rebuilt from
  clipped residual influence; continuous-only moments remain ordinary.
  Defaults remain the lavaan-compatible ML moment builders.
- Robust ordinal reporting is explicit: call
  `magmaan_core$infer_ordinal_robust(fit, ordinal_stats, weight = "")` after a
  DWLS/WLS ordinal fit to compute sandwich SEs and SB-family scaled statistics
  from the threshold-plus-polychoric `NACOV` and the selected DWLS/WLS weight
  matrix.
- Mixed robust reporting is explicit too:
  `magmaan_core$infer_mixed_ordinal_robust(fit, mixed_stats, weight = "")`.
  Mixed `NACOV`/weight and robust reporting parity is still looser than the
  all-ordinal path. Empty `weight` reuses `fit$estimator`.
