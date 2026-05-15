# magmaan R bindings

These bindings are intentionally thin. Most exported functions are direct
`.Call` wrappers around one C++ entry point and keep the C++ argument shape
visible for interactive methods work.

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
  parameters, and nested tests remain explicit post-fit calls.

Low-level functions such as `lavaan_lavaanify()`, `model_matrix_rep()`,
`fit_fit()`, `fit_*_impl()`, `data_sample_stats_from_raw()`, and the `infer_*`
family remain exported so the C++ architecture is still directly inspectable
from R.

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
- Fit with `fit_dwls_ordinal()` or `fit_wls_ordinal()`. These are
  point-estimate and standard chi-square statistic workflows.
- Robust ordinal reporting is explicit: call
  `infer_ordinal_robust(fit, ordinal_stats, weight = "")` after a DWLS/WLS
  ordinal fit to compute sandwich SEs and SB-family scaled statistics from the
  threshold-plus-polychoric `NACOV` and the selected DWLS/WLS weight matrix.
- Mixed continuous/ordinal categorical data now has a separate first-pass path:
  use `model_spec(..., ordered = ..., parameterization = "delta",
  meanstructure = TRUE)`, `data_mixed_ordinal_stats_from_df()`, and
  `fit_dwls_mixed_ordinal()` / `fit_wls_mixed_ordinal()`. Mixed moment values
  follow lavaan's categorical WLS order; mixed `NACOV`/weight and robust
  reporting parity is still looser than the all-ordinal path.
  Empty `weight` reuses `fit$estimator`.
