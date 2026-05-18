# R API Scaffold Audit

This is the working audit log for the exploratory R primitive surface. During
this phase `magmaan_core` is broad on purpose: it is a workbench for testing
whether the C++ primitive graph and namespace layout feel coherent from R, not
yet a stability promise.

Status meanings:

- `good`: keep this shape unless implementation pressure says otherwise.
- `awkward`: usable, but the name or signature points at an API cleanup.
- `misplaced`: likely belongs under another namespace or layer.
- `missing`: useful primitive exists or should exist, but R cannot reach it yet.
- `compatibility-only`: retained as an alias or historical spelling while the
  canonical name settles.

| C++ primitive | R name | Status | Action | Notes |
|---|---|---:|---|---|
| `parse::Parser::parse` | `parse_parse` | good | Keep. | Low-level syntax inspection. |
| `compat::lavaan::to_lavaan_partable` via build | `lavaan_lavaanify` | misplaced | Rename later to `compat_lavaan_lavaanify`; keep alias. | Current spelling is readable but does not show target namespace. |
| `model::build_matrix_rep` | `model_matrix_rep` | good | Keep. | Useful model-inspection primitive. |
| `data::sample_stats_from_raw` | `data_sample_stats_from_raw` | good | Keep. | Main complete-data moment builder. |
| `data::ordinal_stats_from_integer_data` | `data_ordinal_stats_from_raw_impl` | awkward | Rename later without `_impl`; keep helper `data_ordinal_stats_from_df`. | Direct C++ binding currently exposes implementation suffix. |
| `estimate::fit_ml` / `estimate::fit_fiml` / LS fitters | `fit_*_impl`, `fit_fit` | awkward | Add canonical `estimate_*` names in a later sweep; keep current aliases while examples settle. | Existing names predate namespace policy. |
| `estimate::fiml::fiml_robust_mlr` | `estimate_fiml_robust_mlr` | good | Added in scaffold pass. | FIML MLR sandwich SEs and Yuan-Bentler/Mplus scaled-test traces. |
| `inference::information_*`, `vcov`, `se`, tests | `infer_*` | awkward | Rename later to `inference_*`; keep aliases. | `infer` is concise but not the target namespace. |
| `inference::modification_indices` | `inference_modification_indices` | good | Added in scaffold pass. | Supports ML/FIML/continuous LS; WLS requires explicit weight. |
| `inference::score_tests` | `inference_score_tests` | good | Added in scaffold pass. | Supports ML/FIML/continuous LS; WLS requires explicit weight. |
| `robust::*` U-Gamma / weighted-chi-square helpers | `infer_*` robust names | misplaced | Move to `robust_*` names later; keep aliases. | The functions are robust-test primitives, not generic inference. |
| `measures::fit_measures` | `measures_fit` | good | Keep. | Already namespace-shaped. |
| `measures::standardize::*` | `measures_standardize_lv`, `measures_standardize_all` | good | Added in scaffold pass. | Returns standardized estimates and delta-method SEs. |
| `measures::effects::compute_defined` | `compute_defined_impl` / `compute_defined` | awkward | Keep helper; rename direct binding later. | Friendly helper is plain R composition; direct binding still has `_impl`. |
| `measures::residuals`, `measures::standardized_residuals` | `measures_residuals`, `measures_standardized_residuals` | good | Added in scaffold pass. | Raw and correlation-metric residual accessors; residual z-statistics remain a separate lavaan-oracle task. |
| `measures::factor_scores` | `measures_factor_scores` | good | Added in scaffold pass. | Complete-data regression and Bartlett scores over caller-supplied raw blocks. |
