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

Low-level functions such as `lavaan_lavaanify()`, `model_matrix_rep()`,
`fit_fit()`, `fit_*_impl()`, `data_sample_stats_from_raw()`, and the `infer_*`
family remain exported so the C++ architecture is still directly inspectable
from R.

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
  Empty `weight` reuses `fit$estimator`.
