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
| `compat::lavaan::to_lavaan_partable` via build | `compat_lavaan_lavaanify` / `lavaan_lavaanify` | good | Prefer canonical name; keep old spelling as compatibility alias. | Current scripts can migrate without losing the readable historical spelling. |
| `model::build_matrix_rep` | `model_matrix_rep` | good | Keep. | Useful model-inspection primitive. |
| `data::sample_stats_from_raw` | `data_sample_stats_from_raw` | good | Keep. | Main complete-data moment builder. |
| `data::ordinal_stats_from_integer_data` plus robust ordinal builders | `data_ordinal_stats_from_raw(robust = ...)` / compatibility method-specific names | good | Prefer canonical dispatcher; keep `_impl` and old method-specific names as compatibility aliases. | Helper `data_ordinal_stats_from_df` remains plain R composition. Supports `robust = "ml"`/`"none"`, `"h_weighted"`, `"dpd"`, and `"huber_residual"`. |
| `data::mixed_ordinal_stats_from_data` plus robust mixed builders | `data_mixed_ordinal_stats_from_raw(polyserial = ...)` / compatibility method-specific names | good | Prefer canonical dispatcher; keep `_impl` and old method-specific names as compatibility aliases. | Helper `data_mixed_ordinal_stats_from_df` remains plain R composition. The mask-bearing mixed interface supports `polyserial = "ml"`, `"dpd"`, and `"huber_residual"`. |
| `estimate::fit_ml` / `estimate::fit_fiml` / LS fitters | `estimate_*` / `fit_*_impl`, `estimate_fit` / `fit_fit` | good | Prefer canonical names; keep old spellings as compatibility aliases. | Existing names predate namespace policy but remain reachable during exploration. |
| `estimate::fiml::fiml_robust_mlr` | `estimate_fiml_robust_mlr` | good | Added in scaffold pass. | FIML MLR sandwich SEs and Yuan-Bentler/Mplus scaled-test traces. |
| `inference::information_*`, `vcov`, `se`, tests | `inference_*` / `infer_*` | good | Prefer canonical names; keep old spellings as compatibility aliases. | `infer` is concise but not the target namespace. |
| `inference::modification_indices` | `inference_modification_indices` | good | Added in scaffold pass. | Supports ML/FIML/continuous LS; WLS requires explicit weight. |
| `inference::score_tests` | `inference_score_tests` | good | Added in scaffold pass. | Supports ML/FIML/continuous LS; WLS requires explicit weight. |
| `robust::*` U-Gamma / weighted-chi-square helpers | `robust_*` / old `infer_*` robust names | good | Prefer canonical names; keep old spellings as compatibility aliases. | The functions are robust-test primitives, not generic inference. Nested-test aliases now include statistical `robust_nested_lrt_restriction_map` plus explicit `compat_lavaan_nested_lrt_*` labels. |
| `measures::fit_measures` / `estimate::fit_measures_ordinal` | `measures_fit` / API ordinal fit measures | good | Keep; add an R primitive only when ordinal R scripts need direct low-level access. | Continuous/FIML R fit measures remain through `measures_fit`; all-ordinal C++ API now covers categorical CFI/TLI/RMSEA/SRMR. |
| `measures::standardize::*` | `measures_standardize_lv`, `measures_standardize_all` | good | Added in scaffold pass. | Returns standardized estimates and delta-method SEs. |
| `measures::effects::compute_defined` | `measures_compute_defined` / `compute_defined_impl` / `compute_defined` | good | Prefer canonical direct primitive; keep helper and old spelling. | Friendly helper is plain R composition. |

Presentation rule: `attr(magmaan_core, "groups")` lists only the canonical
surface plus plain R helpers. Historical aliases, method-specific Rcpp bridges,
and `_impl` names remain callable from `magmaan_core` for old scripts but are
not included in the displayed groups.
| `measures::residuals`, `measures::standardized_residuals` | `measures_residuals`, `measures_standardized_residuals` | good | Added in scaffold pass. | Raw and correlation-metric residual accessors; residual z-statistics remain a separate lavaan-oracle task. |
| `measures::factor_scores` | `measures_factor_scores` | good | Added in scaffold pass. | Complete-data regression and Bartlett scores over caller-supplied raw blocks. |
