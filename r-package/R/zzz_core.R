compat_lavaan_lavaanify <- lavaan_lavaanify
compat_lavaan_nested_lrt_satorra2000 <- infer_lr_test_satorra2000
compat_lavaan_nested_lrt_satorra_bentler2001 <- infer_lr_test_satorra_bentler2001
compat_lavaan_nested_lrt_satorra_bentler2010 <- infer_lr_test_satorra_bentler2010

data_ordinal_stats_from_raw <- function(X, robust = c("ml", "none", "h_weighted", "dpd", "huber_residual"),
                                        alpha = 0.3,
                                        h_kind = c("wma_hard_cap", "ml", "smooth_cap", "exp_cap"),
                                        h_k = 1.5, h_a = 1.6, h_b = 2.2,
                                        h_lambda = 0.2,
                                        clip = c("hard_huber", "pseudo_huber",
                                                 "tukey_biweight", "none"),
                                        k = NULL) {
  robust <- match.arg(robust)
  h_kind <- match.arg(h_kind)
  clip <- match.arg(clip)
  if (is.null(k)) {
    k <- if (identical(clip, "tukey_biweight")) 4.685 else 1.345
  }
  switch(
    robust,
    ml = data_ordinal_stats_from_raw_impl(X),
    none = data_ordinal_stats_from_raw_impl(X),
    h_weighted = data_ordinal_stats_h_weighted_from_raw_impl(
      X, h_kind = h_kind, k = h_k, a = h_a, b = h_b, lambda = h_lambda),
    dpd = data_ordinal_stats_dpd_from_raw_impl(X, alpha = alpha),
    huber_residual = data_ordinal_stats_huber_residual_from_raw_impl(
      X, clip = clip, k = k)
  )
}
data_ordinal_stats_h_weighted_from_raw <- data_ordinal_stats_h_weighted_from_raw_impl
data_ordinal_stats_dpd_from_raw <- data_ordinal_stats_dpd_from_raw_impl
data_ordinal_stats_huber_residual_from_raw <- data_ordinal_stats_huber_residual_from_raw_impl
data_mixed_ordinal_stats_from_raw <- function(X, ordered_mask,
                                              polyserial = c("ml", "dpd", "huber_residual"),
                                              alpha = 0.3,
                                              clip = c("hard_huber", "pseudo_huber",
                                                       "tukey_biweight", "none"),
                                              k = NULL) {
  polyserial <- match.arg(polyserial)
  clip <- match.arg(clip)
  if (is.null(k)) {
    k <- if (identical(clip, "tukey_biweight")) 4.685 else 1.345
  }
  switch(
    polyserial,
    ml = data_mixed_ordinal_stats_from_raw_impl(X, ordered_mask),
    dpd = data_mixed_ordinal_stats_polyserial_dpd_from_raw_impl(
      X, ordered_mask, alpha = alpha),
    huber_residual = data_mixed_ordinal_stats_huber_residual_from_raw_impl(
      X, ordered_mask, clip = clip, k = k)
  )
}
data_shrink_mixed_ordinal_stats <- data_shrink_mixed_ordinal_stats_impl
data_mixed_ordinal_stats_polyserial_dpd_from_raw <- data_mixed_ordinal_stats_polyserial_dpd_from_raw_impl
data_mixed_ordinal_stats_huber_residual_from_raw <- data_mixed_ordinal_stats_huber_residual_from_raw_impl

estimate_fit <- fit_fit
estimate_ml <- fit_ml_impl
estimate_fiml <- fit_fiml_impl
estimate_uls <- fit_uls_impl
estimate_gls <- fit_gls_impl
estimate_wls <- fit_wls_impl
estimate_dwls_ordinal <- fit_dwls_ordinal_impl
estimate_wls_ordinal <- fit_wls_ordinal_impl
estimate_dwls_mixed_ordinal <- fit_dwls_mixed_ordinal_impl
estimate_wls_mixed_ordinal <- fit_wls_mixed_ordinal_impl
estimate_uls_snlls <- fit_uls_snlls_impl
estimate_gls_snlls <- fit_gls_snlls_impl
estimate_wls_snlls <- fit_wls_snlls_impl
estimate_bounds_variance <- bounds_variance
estimate_bounds_pos_var <- bounds_pos_var
estimate_bounds_standard <- bounds_standard
estimate_bounds_wide <- bounds_wide
estimate_bounds_loading <- bounds_loading
# Phase 4 retired estimate_*_ceres (separate Rcpp shims per Backend); the
# unified estimate_uls/gls/wls/*_snlls now take an `optimizer = "..."`
# string that maps to any of the C++ Backend values (see backend_strings.hpp).
estimate_start_values <- fit_start_values
frontier_fcsem_model_spec <- fcsem_model_spec_impl
frontier_fit_ml_fcsem <- fit_ml_fcsem_impl
frontier_is_std_lv_admissible <- is_std_lv_admissible_impl
frontier_backconvert_std_lv_to_marker <- backconvert_std_lv_to_marker_impl
frontier_fit_ml_auto_identification <- fit_ml_auto_identification_impl

inference_information_expected <- infer_information_expected
inference_information_observed_fd <- infer_information_observed_fd
inference_information_observed_analytic <- infer_information_observed_analytic
inference_vcov <- infer_vcov
inference_vcov_partable <- infer_vcov_partable
inference_vcov_fit <- infer_vcov_fit
inference_se <- infer_se
inference_chi2_stat <- infer_chi2_stat
inference_df_stat <- infer_df_stat
inference_z_test <- infer_z_test
inference_z_test_theta <- infer_z_test_theta
inference_z_test_fit <- infer_z_test_fit
inference_chi2_pvalue <- infer_chi2_pvalue
inference_wald_test <- infer_wald_test
inference_wald_test_theta <- infer_wald_test_theta
inference_wald_test_fit <- infer_wald_test_fit
inference_browne_residual_nt <- infer_browne_residual_nt
inference_rls_chi2 <- infer_rls_chi2
inference_rls_chi2_sample <- infer_rls_chi2_sample
inference_rls_chi2_fit <- infer_rls_chi2_fit

robust_lr_test_satorra2000 <- infer_lr_test_satorra2000
robust_lr_test_satorra_bentler2001 <- infer_lr_test_satorra_bentler2001
robust_lr_test_satorra_bentler2010 <- infer_lr_test_satorra_bentler2010
robust_nested_lrt_restriction_map <- infer_lr_test_satorra2000
robust_build_u_factor <- infer_build_u_factor
robust_build_u_factor_parts <- infer_build_u_factor_parts
robust_build_u_factor_fit <- infer_build_u_factor_fit
robust_reduced_gamma_nt <- infer_reduced_gamma_nt
robust_reduced_gamma_sample <- infer_reduced_gamma_sample
robust_reduced_gamma_sample_materialized <- infer_reduced_gamma_sample_materialized
robust_reduced_gamma_unbiased <- infer_reduced_gamma_unbiased
robust_ugamma_eigenvalues <- infer_ugamma_eigenvalues
robust_satorra_bentler <- infer_satorra_bentler
robust_mean_var_adjusted <- infer_mean_var_adjusted
robust_scaled_shifted <- infer_scaled_shifted
robust_casewise_contributions <- infer_casewise_contributions
robust_empirical_gamma <- infer_empirical_gamma
robust_gamma_nt <- infer_gamma_nt
robust_ordinal <- infer_ordinal_robust
robust_mixed_ordinal <- infer_mixed_ordinal_robust
robust_se <- infer_robust_se
robust_se_parts <- infer_robust_se_parts
robust_se_fit <- infer_robust_se_fit
robust_se_raw <- infer_robust_se_raw
robust_se_raw_parts <- infer_robust_se_raw_parts
robust_se_raw_fit <- infer_robust_se_raw_fit

measures_baseline <- infer_baseline
measures_compute_defined <- compute_defined_impl
frontier_fcsem_standard_errors <- fcsem_standard_errors_impl
frontier_fcsem_fit_measures <- fcsem_fit_measures_impl
frontier_fcsem_standardized_rows <- fcsem_standardized_rows_impl

magmaan_core <- local({
  groups <- list(
    parse = c(
      "parse_parse"
    ),
    compat_lavaan = c(
      "compat_lavaan_lavaanify",
      "compat_lavaan_nested_lrt_satorra2000",
      "compat_lavaan_nested_lrt_satorra_bentler2001",
      "compat_lavaan_nested_lrt_satorra_bentler2010",
      "lavaan_compare_partable"
    ),
    model = c(
      "model_matrix_rep",
      "model_implied"
    ),
    data = c(
      "data_sample_stats_from_raw",
      "data_ordinal_stats_from_raw",
      "data_mixed_ordinal_stats_from_raw",
      "data_shrink_mixed_ordinal_stats"
    ),
    estimate = c(
      "estimate_fit",
      "estimate_ml",
      "estimate_fiml",
      "estimate_uls",
      "estimate_gls",
      "estimate_wls",
      "estimate_dwls_ordinal",
      "estimate_wls_ordinal",
      "estimate_dwls_mixed_ordinal",
      "estimate_wls_mixed_ordinal",
      "estimate_uls_snlls",
      "estimate_gls_snlls",
      "estimate_wls_snlls",
      "estimate_bounds_variance",
      "estimate_bounds_pos_var",
      "estimate_bounds_standard",
      "estimate_bounds_wide",
      "estimate_bounds_loading",
      "estimate_start_values",
      "estimate_structured_gamma",
      "estimate_structured_gamma_weight",
      "estimate_fiml_robust_mlr"
    ),
    inference = c(
      "inference_information_expected",
      "inference_information_observed_fd",
      "inference_information_observed_analytic",
      "inference_vcov",
      "inference_vcov_partable",
      "inference_vcov_fit",
      "inference_se",
      "inference_chi2_stat",
      "inference_df_stat",
      "inference_z_test",
      "inference_z_test_theta",
      "inference_z_test_fit",
      "inference_chi2_pvalue",
      "inference_wald_test",
      "inference_wald_test_theta",
      "inference_wald_test_fit",
      "inference_browne_residual_nt",
      "inference_rls_chi2",
      "inference_rls_chi2_sample",
      "inference_rls_chi2_fit",
      "inference_modification_indices",
      "inference_score_tests"
    ),
    robust = c(
      "robust_lr_test_satorra2000",
      "robust_lr_test_satorra_bentler2001",
      "robust_lr_test_satorra_bentler2010",
      "robust_nested_lrt_restriction_map",
      "robust_build_u_factor",
      "robust_build_u_factor_parts",
      "robust_build_u_factor_fit",
      "robust_reduced_gamma_nt",
      "robust_reduced_gamma_sample",
      "robust_reduced_gamma_sample_materialized",
      "robust_reduced_gamma_unbiased",
      "robust_ugamma_eigenvalues",
      "robust_satorra_bentler",
      "robust_mean_var_adjusted",
      "robust_scaled_shifted",
      "robust_casewise_contributions",
      "robust_empirical_gamma",
      "robust_gamma_nt",
      "robust_ordinal",
      "robust_mixed_ordinal",
      "robust_se",
      "robust_se_parts",
      "robust_se_fit",
      "robust_se_raw",
      "robust_se_raw_parts",
      "robust_se_raw_fit"
    ),
    measures = c(
      "measures_baseline",
      "measures_fit",
      "measures_standardize_lv",
      "measures_standardize_all",
      "measures_composite_weights",
      "measures_residuals",
      "measures_standardized_residuals",
      "measures_factor_scores",
      "measures_compute_defined"
    ),
    frontier = c(
      "frontier_fcsem_model_spec",
      "frontier_fit_ml_fcsem",
      "frontier_fcsem_standard_errors",
      "frontier_fcsem_fit_measures",
      "frontier_fcsem_standardized_rows",
      "frontier_is_std_lv_admissible",
      "frontier_backconvert_std_lv_to_marker",
      "frontier_fit_ml_auto_identification",
      "frontier_fit_ml_ridge_continuation"
    ),
    helpers = c(
      "df_to_data",
      "df_to_fiml_data",
      "df_to_fcsem_data",
      "data_ordinal_stats_from_df",
      "data_mixed_ordinal_stats_from_df",
      "shrink_mixed_ordinal_stats",
      "fit_ml",
      "fit_fiml",
      "fit_uls",
      "fit_gls",
      "fit_wls",
      "bounds_variance",
      "bounds_pos_var",
      "bounds_standard",
      "bounds_wide",
      "bounds_loading",
      "fit_dwls_ordinal",
      "fit_wls_ordinal",
      "fit_dwls_mixed_ordinal",
      "fit_wls_mixed_ordinal",
      "fit_uls_snlls",
      "fit_gls_snlls",
      "fit_wls_snlls",
      "fit_ml_fcsem",
      "magmaan_fcsem",
      "fcsem_standard_errors",
      "fcsem_fit_measures",
      "fcsem_standardized_rows"
    )
  )
  compatibility_names <- c(
      "lavaan_lavaanify",
      "data_ordinal_stats_h_weighted_from_raw",
      "data_ordinal_stats_dpd_from_raw",
      "data_ordinal_stats_huber_residual_from_raw",
      "data_mixed_ordinal_stats_polyserial_dpd_from_raw",
      "data_mixed_ordinal_stats_huber_residual_from_raw",
      "data_ordinal_stats_from_raw_impl",
      "data_ordinal_stats_h_weighted_from_raw_impl",
      "data_ordinal_stats_dpd_from_raw_impl",
      "data_ordinal_stats_huber_residual_from_raw_impl",
      "data_mixed_ordinal_stats_from_raw_impl",
      "data_shrink_mixed_ordinal_stats_impl",
      "data_mixed_ordinal_stats_polyserial_dpd_from_raw_impl",
      "data_mixed_ordinal_stats_huber_residual_from_raw_impl",
      "fit_fit",
      "fit_ml_impl",
      "bounds_variance_impl",
      "bounds_standard_impl",
      "bounds_wide_impl",
      "bounds_loading_impl",
      "fit_fiml_impl",
      "fit_uls_impl",
      "fit_gls_impl",
      "fit_wls_impl",
      "fit_dwls_ordinal_impl",
      "fit_wls_ordinal_impl",
      "fit_dwls_mixed_ordinal_impl",
      "fit_wls_mixed_ordinal_impl",
      "fit_uls_snlls_impl",
      "fit_gls_snlls_impl",
      "fit_wls_snlls_impl",
      "fcsem_model_spec_impl",
      "fit_ml_fcsem_impl",
      "frontier_fit_ml_ridge_continuation_impl",
      "fcsem_standard_errors_impl",
      "fcsem_fit_measures_impl",
      "fcsem_standardized_rows_impl",
      "is_std_lv_admissible_impl",
      "backconvert_std_lv_to_marker_impl",
      "fit_ml_auto_identification_impl",
      "fit_start_values",
      "fit_sample_stats",
      "infer_information_expected",
      "infer_information_observed_fd",
      "infer_information_observed_analytic",
      "infer_vcov",
      "infer_vcov_partable",
      "infer_se",
      "infer_chi2_stat",
      "infer_df_stat",
      "infer_z_test",
      "infer_z_test_theta",
      "infer_chi2_pvalue",
      "infer_wald_test",
      "infer_wald_test_theta",
      "infer_browne_residual_nt",
      "infer_rls_chi2",
      "infer_rls_chi2_sample",
      "infer_lr_test_satorra2000",
      "infer_lr_test_satorra_bentler2001",
      "infer_lr_test_satorra_bentler2010",
      "infer_build_u_factor",
      "infer_build_u_factor_parts",
      "infer_reduced_gamma_nt",
      "infer_reduced_gamma_sample",
      "infer_reduced_gamma_sample_materialized",
      "infer_reduced_gamma_unbiased",
      "infer_ugamma_eigenvalues",
      "infer_satorra_bentler",
      "infer_mean_var_adjusted",
      "infer_scaled_shifted",
      "infer_casewise_contributions",
      "infer_empirical_gamma",
      "infer_gamma_nt",
      "infer_ordinal_robust",
      "infer_mixed_ordinal_robust",
      "infer_robust_se",
      "infer_robust_se_parts",
      "infer_robust_se_raw",
      "infer_robust_se_raw_parts",
      "infer_baseline",
      "compute_defined_impl",
      "measures_composite_weights",
      "infer_vcov_fit",
      "infer_z_test_fit",
      "infer_wald_test_fit",
      "infer_rls_chi2_fit",
      "infer_build_u_factor_fit",
      "infer_robust_se_fit",
      "infer_robust_se_raw_fit"
  )
  core_names <- unique(c(unlist(groups, use.names = FALSE), compatibility_names))

  ns <- parent.env(environment())
  out <- list2env(mget(core_names, envir = ns, inherits = FALSE),
                  parent = emptyenv())
  attr(out, "groups") <- groups
  class(out) <- "magmaan_core"
  lockEnvironment(out, bindings = TRUE)
  out
})

print.magmaan_core <- function(x, ...) {
  groups <- attr(x, "groups", exact = TRUE)
  cat("magmaan_core primitive environment\n")
  for (nm in names(groups)) {
    cat("  ", nm, ": ", length(groups[[nm]]), "\n", sep = "")
  }
  cat("Use attr(magmaan_core, \"groups\")$<group> to list canonical names.\n")
  cat("Compatibility aliases remain callable but are not displayed.\n")
  invisible(x)
}
