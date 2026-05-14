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
