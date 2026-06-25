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
                                        k = NULL, full_wls_weight = TRUE) {
  robust <- match.arg(robust)
  h_kind <- match.arg(h_kind)
  clip <- match.arg(clip)
  if (is.null(k)) {
    k <- if (identical(clip, "tukey_biweight")) 4.685 else 1.345
  }
  switch(
    robust,
    ml = data_ordinal_stats_from_raw_impl(X, full_wls_weight = full_wls_weight),
    none = data_ordinal_stats_from_raw_impl(X, full_wls_weight = full_wls_weight),
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
data_ordinal_stats_observed_from_raw <- data_ordinal_stats_observed_from_raw_impl
data_mixed_ordinal_stats_from_raw <- function(X, ordered_mask,
                                              polyserial = c("ml", "dpd", "huber_residual"),
                                              alpha = 0.3,
                                              clip = c("hard_huber", "pseudo_huber",
                                                       "tukey_biweight", "none"),
                                              k = NULL,
                                              full_wls_weight = TRUE) {
  polyserial <- match.arg(polyserial)
  clip <- match.arg(clip)
  if (is.null(k)) {
    k <- if (identical(clip, "tukey_biweight")) 4.685 else 1.345
  }
  switch(
    polyserial,
    ml = data_mixed_ordinal_stats_from_raw_impl(X, ordered_mask, full_wls_weight),
    dpd = data_mixed_ordinal_stats_polyserial_dpd_from_raw_impl(
      X, ordered_mask, alpha = alpha),
    huber_residual = data_mixed_ordinal_stats_huber_residual_from_raw_impl(
      X, ordered_mask, clip = clip, k = k)
  )
}
data_shrink_mixed_ordinal_stats <- data_shrink_mixed_ordinal_stats_impl
data_mixed_ordinal_stats_observed_from_raw <- data_mixed_ordinal_stats_observed_from_raw_impl
data_mixed_ordinal_stats_hybrid_fiml_from_raw <- data_mixed_ordinal_stats_hybrid_fiml_from_raw_impl
data_mixed_ordinal_stats_polyserial_dpd_from_raw <- data_mixed_ordinal_stats_polyserial_dpd_from_raw_impl
data_mixed_ordinal_stats_huber_residual_from_raw <- data_mixed_ordinal_stats_huber_residual_from_raw_impl

estimate_fit <- fit_fit
estimate_ml <- fit_ml_impl
estimate_ml_fisher <- fit_ml_fisher_impl
estimate_ml_fisher_snlls <- fit_ml_fisher_snlls_impl
estimate_ml_irls <- fit_ml_irls_impl
estimate_ml_irls_snlls <- fit_ml_irls_snlls_impl
estimate_fiml <- fit_fiml_impl
estimate_saturated_em_moments <- saturated_em_moments_impl
estimate_two_stage_em <- estimate_two_stage_em_impl
estimate_uls <- fit_uls_impl
estimate_gls <- fit_gls_impl
estimate_gls_pairwise <- fit_gls_pairwise_impl
estimate_wls <- fit_wls_impl
estimate_dwls_ordinal <- fit_dwls_ordinal_impl
estimate_wls_ordinal <- fit_wls_ordinal_impl
estimate_uls_ordinal <- fit_uls_ordinal_impl
estimate_ordinal_stage2 <- fit_ordinal_stage2_impl
estimate_ordinal_stage2_weight_blocks <- ordinal_stage2_weight_blocks_impl
estimate_dwls_mixed_ordinal <- fit_dwls_mixed_ordinal_impl
estimate_wls_mixed_ordinal <- fit_wls_mixed_ordinal_impl
estimate_uls_snlls <- fit_uls_snlls_impl
estimate_gls_snlls <- fit_gls_snlls_impl
estimate_wls_snlls <- fit_wls_snlls_impl
estimate_evaluate_at <- evaluate_at_impl
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
frontier_pairwise_ordinal_composite_nested <- frontier_pairwise_ordinal_composite_nested_impl
frontier_rbm <- function(fit, raw_data = NULL, weight = NULL,
                         stage2_weight = "nt", dls_a = 0.5,
                         method = c("explicit", "implicit"),
                         optimizer = NULL, control = NULL, bounds = NULL) {
  method <- match.arg(method)
  frontier_rbm_impl(fit, raw_data, weight, stage2_weight, dls_a,
                    method, optimizer, control, bounds)
}

inference_information_expected <- infer_information_expected
inference_information_observed_fd <- infer_information_observed_fd
inference_information_observed_analytic <- infer_information_observed_analytic
inference_fiml_observed_vcov <- infer_fiml_observed_vcov
fiml_observed_vcov <- infer_fiml_observed_vcov
inference_information_cross_products <- infer_information_cross_products
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
robust_lr_test_satorra2000_fiml <- infer_fiml_lr_test_satorra2000
robust_lr_test_satorra2000_ml2s <- infer_ml2s_lr_test_satorra2000
robust_lr_test_satorra2000_ordinal <- infer_ordinal_lr_test_satorra2000
robust_lr_test_satorra_bentler2001 <- infer_lr_test_satorra_bentler2001
robust_lr_test_satorra_bentler2010 <- infer_lr_test_satorra_bentler2010
robust_nested_lrt_restriction_map <- infer_lr_test_satorra2000
robust_nested_lrt_fiml_restriction_map <- infer_fiml_lr_test_satorra2000
robust_nested_lrt_ml2s_restriction_map <- infer_ml2s_lr_test_satorra2000
robust_nested_lrt_ordinal_restriction_map <- infer_ordinal_lr_test_satorra2000
robust_build_u_factor <- infer_build_u_factor
robust_build_u_factor_parts <- infer_build_u_factor_parts
robust_build_u_factor_fit <- infer_build_u_factor_fit
robust_build_u_factor_pairwise <- infer_build_u_factor_pairwise
robust_reduced_gamma_nt <- infer_reduced_gamma_nt
robust_reduced_gamma_nt_pairwise <- infer_reduced_gamma_nt_pairwise
robust_reduced_gamma_sample <- infer_reduced_gamma_sample
robust_reduced_gamma_sample_zc <- infer_reduced_gamma_sample
robust_reduced_gamma_sample_from_gamma <- infer_reduced_gamma_sample_from_gamma
robust_reduced_gamma_sample_gamma <- infer_reduced_gamma_sample_from_gamma
robust_test_moments_both_breads_zc <- infer_robust_test_moments_both_breads_zc
robust_test_moments_both_breads_gamma <- infer_robust_test_moments_both_breads_gamma
robust_reduced_gamma_sample_materialized <- infer_reduced_gamma_sample_materialized
robust_reduced_gamma_unbiased <- infer_reduced_gamma_unbiased
robust_ugamma_eigenvalues <- infer_ugamma_eigenvalues
robust_fmg_ugamma_spectra <- infer_fmg_ugamma_spectra
robust_fmg_test <- infer_fmg_test
robust_satorra_bentler <- infer_satorra_bentler
robust_mean_var_adjusted <- infer_mean_var_adjusted
robust_scaled_shifted <- infer_scaled_shifted
# Trace-form path for SB / MV-adj / SS without an eigendecomposition: use
# `robust_test_moments_both_breads_{zc,gamma}` to get the moments straight
# from C++ and apply the closed-form scaling in R (see Maydeu experiment).
robust_casewise_contributions <- infer_casewise_contributions
robust_pairwise_casewise_contributions <- infer_pairwise_casewise_contributions
robust_empirical_gamma <- infer_empirical_gamma
robust_empirical_gamma_with_means <- infer_empirical_gamma_with_means
robust_gamma_nt <- infer_gamma_nt
robust_ordinal <- infer_ordinal_robust
robust_ordinal_ij <- infer_ordinal_robust_ij
robust_mixed_ordinal <- infer_mixed_ordinal_robust
robust_mixed_ordinal_ij <- infer_mixed_ordinal_robust_ij
ordinal_profile_rmsea <- infer_ordinal_profile_rmsea
ordinal_profile_lrt <- infer_ordinal_profile_lrt
ml_profile_lrt <- infer_ml_profile_lrt
continuous_ls_profile_lrt <- infer_continuous_ls_profile_lrt
fiml_profile_lrt <- infer_fiml_profile_lrt
mixed_ordinal_fit_measures_misspec <- infer_mixed_ordinal_fit_measures_misspec
mixed_ordinal_rmsea_misspec <- infer_mixed_ordinal_rmsea_misspec
mixed_ordinal_crmr_misspec <- infer_mixed_ordinal_crmr_misspec
mixed_ordinal_cfi_tli_misspec <- infer_mixed_ordinal_cfi_tli_misspec
mixed_ordinal_profile_rmsea <- infer_mixed_ordinal_profile_rmsea
mixed_ordinal_profile_lrt <- infer_mixed_ordinal_profile_lrt
robust_se <- infer_robust_se
robust_se_parts <- infer_robust_se_parts
robust_se_fit <- infer_robust_se_fit
robust_se_raw <- infer_robust_se_raw
robust_se_raw_parts <- infer_robust_se_raw_parts
robust_se_raw_fit <- infer_robust_se_raw_fit
robust_se_zc <- infer_robust_se_zc
# Pair entry points — run robust_se setup once and return both
# `bread = expected` and `bread = observed` sandwich SEs. Prefer these in
# simulation hot loops; the single-bread variants above re-run setup.
robust_se_both_breads <- infer_robust_se_both_breads
robust_se_both_breads_raw <- infer_robust_se_both_breads_raw
robust_se_both_breads_zc <- infer_robust_se_both_breads_zc

sim_ig_batch <- sim_ig_batch_impl
sim_ig_calibrate <- sim_ig_calibrate_impl
sim_ig_draw <- sim_ig_draw_impl
sim_norta_batch <- sim_norta_batch_impl
sim_norta_calibrate <- sim_norta_calibrate_impl
sim_norta_draw <- sim_norta_draw_impl
sim_vm_batch <- sim_vm_batch_impl
sim_vm_calibrate <- sim_vm_calibrate_impl
sim_vm_draw <- sim_vm_draw_impl
sim_bicop_batch <- sim_bicop_batch_impl
sim_bicop_calibrate <- sim_bicop_calibrate_impl
sim_bicop_draw <- sim_bicop_draw_impl
sim_cvine_batch <- sim_cvine_batch_impl
sim_cvine_calibrate <- sim_cvine_calibrate_impl
sim_cvine_draw <- sim_cvine_draw_impl
sim_cvine3_batch <- sim_cvine3_batch_impl
sim_cvine3_calibrate <- sim_cvine3_calibrate_impl
sim_cvine3_draw <- sim_cvine3_draw_impl
sim_plsim_batch <- sim_plsim_batch_impl
sim_plsim_calibrate <- sim_plsim_calibrate_impl
sim_plsim_draw <- sim_plsim_draw_impl
sim_model_batch <- sim_model_batch_impl
sim_model_calibrate <- sim_model_calibrate_impl
sim_model_draw <- sim_model_draw_impl
sim_ordcorr_batch <- sim_ordcorr_batch_impl
sim_ordcorr_calibrate <- sim_ordcorr_calibrate_impl
sim_ordcorr_draw <- sim_ordcorr_draw_impl
sim_ordcorr_mg_batch <- sim_ordcorr_mg_batch_impl
sim_ordcorr_mg_calibrate <- sim_ordcorr_mg_calibrate_impl
sim_ordcorr_mg_draw <- sim_ordcorr_mg_draw_impl
sim_ordcorr_summary_calibrate <- sim_ordcorr_summary_calibrate_impl
sim_ordcorr_mg_summary_calibrate <- sim_ordcorr_mg_summary_calibrate_impl

measures_baseline <- infer_baseline
measures_baseline_fit <- infer_baseline_fit
measures_ordinal_catml_dwls_rmsea <- ordinal_catml_dwls_rmsea_impl
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
      "data_pairwise_sample_stats",
      "data_gamma_nt_pairwise",
      "data_ordinal_stats_from_raw",
      "data_ordinal_stats_observed_from_raw",
      "data_mixed_ordinal_stats_from_raw",
      "data_mixed_ordinal_stats_observed_from_raw",
      "data_mixed_ordinal_stats_hybrid_fiml_from_raw",
      "data_shrink_mixed_ordinal_stats"
    ),
    estimate = c(
      "estimate_fit",
      "estimate_ml",
      "estimate_ml_fisher",
      "estimate_ml_fisher_snlls",
      "estimate_ml_irls",
      "estimate_ml_irls_snlls",
      "estimate_fiml",
      "estimate_saturated_em_moments",
      "estimate_two_stage_em",
      "estimate_uls",
      "estimate_gls",
      "estimate_gls_pairwise",
      "estimate_wls",
      "estimate_dwls_ordinal",
      "estimate_wls_ordinal",
      "estimate_uls_ordinal",
      "estimate_dwls_mixed_ordinal",
      "estimate_wls_mixed_ordinal",
      "estimate_uls_snlls",
      "estimate_gls_snlls",
      "estimate_wls_snlls",
      "estimate_evaluate_at",
      "estimate_bounds_variance",
      "estimate_bounds_pos_var",
      "estimate_bounds_standard",
      "estimate_bounds_wide",
      "estimate_bounds_loading",
      "estimate_start_values",
      "estimate_structured_gamma",
      "estimate_structured_gamma_weight",
      "fiml_observed_vcov",
      "estimate_fiml_robust_mlr",
      "estimate_two_stage_em_ml_inference"
    ),
    inference = c(
      "inference_information_expected",
      "inference_information_observed_fd",
      "inference_information_observed_analytic",
      "inference_fiml_observed_vcov",
      "inference_information_cross_products",
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
      "inference_score_tests",
      "inference_modification_indices_robust",
      "inference_score_tests_robust"
    ),
    robust = c(
      "robust_lr_test_satorra2000",
      "robust_lr_test_satorra2000_fiml",
      "robust_lr_test_satorra2000_ml2s",
      "robust_lr_test_satorra2000_ordinal",
      "robust_lr_test_satorra_bentler2001",
      "robust_lr_test_satorra_bentler2010",
      "robust_nested_lrt_restriction_map",
      "robust_nested_lrt_fiml_restriction_map",
      "robust_nested_lrt_ml2s_restriction_map",
      "robust_nested_lrt_ordinal_restriction_map",
      "robust_build_u_factor",
      "robust_build_u_factor_parts",
      "robust_build_u_factor_fit",
      "robust_build_u_factor_pairwise",
      "robust_reduced_gamma_nt",
      "robust_reduced_gamma_nt_pairwise",
      "robust_reduced_gamma_sample",
      "robust_reduced_gamma_sample_zc",
      "robust_reduced_gamma_sample_from_gamma",
      "robust_reduced_gamma_sample_gamma",
      "robust_test_moments_both_breads_zc",
      "robust_test_moments_both_breads_gamma",
      "robust_reduced_gamma_sample_materialized",
      "robust_reduced_gamma_unbiased",
      "robust_ugamma_eigenvalues",
      "robust_fmg_ugamma_spectra",
      "robust_fmg_test",
      "robust_satorra_bentler",
      "robust_mean_var_adjusted",
      "robust_scaled_shifted",
      "robust_casewise_contributions",
      "robust_pairwise_casewise_contributions",
      "robust_empirical_gamma",
      "robust_empirical_gamma_with_means",
      "robust_gamma_nt",
      "robust_ordinal",
      "robust_ordinal_ij",
      "robust_mixed_ordinal",
      "robust_mixed_ordinal_ij",
      "ordinal_profile_rmsea",
      "ordinal_profile_lrt",
      "ml_profile_lrt",
      "continuous_ls_profile_lrt",
      "fiml_profile_lrt",
      "mixed_ordinal_fit_measures_misspec",
      "mixed_ordinal_rmsea_misspec",
      "mixed_ordinal_crmr_misspec",
      "mixed_ordinal_cfi_tli_misspec",
      "mixed_ordinal_profile_rmsea",
      "mixed_ordinal_profile_lrt",
      "robust_se",
      "robust_se_parts",
      "robust_se_fit",
      "robust_se_raw",
      "robust_se_raw_parts",
      "robust_se_raw_fit",
      "robust_se_zc",
      "robust_se_both_breads",
      "robust_se_both_breads_raw",
      "robust_se_both_breads_zc"
    ),
    sim = c(
      "sim_ig_batch",
      "sim_ig_calibrate",
      "sim_ig_draw",
      "sim_norta_batch",
      "sim_norta_calibrate",
      "sim_norta_draw",
      "sim_vm_batch",
      "sim_vm_calibrate",
      "sim_vm_draw",
      "sim_bicop_batch",
      "sim_bicop_calibrate",
      "sim_bicop_draw",
      "sim_cvine_batch",
      "sim_cvine_calibrate",
      "sim_cvine_draw",
      "sim_cvine3_batch",
      "sim_cvine3_calibrate",
      "sim_cvine3_draw",
      "sim_plsim_batch",
      "sim_plsim_calibrate",
      "sim_plsim_draw",
      "sim_model_batch",
      "sim_model_calibrate",
      "sim_model_draw",
      "sim_ordcorr_batch",
      "sim_ordcorr_calibrate",
      "sim_ordcorr_draw",
      "sim_ordcorr_mg_batch",
      "sim_ordcorr_mg_calibrate",
      "sim_ordcorr_mg_draw",
      "sim_ordcorr_summary_calibrate",
      "sim_ordcorr_mg_summary_calibrate"
    ),
    measures = c(
      "measures_baseline",
      "measures_baseline_fit",
      "measures_fit",
      "measures_ordinal_catml_dwls_rmsea",
      "measures_standardize_lv",
      "measures_standardize_all",
      "measures_composite_weights",
      "measures_residuals",
      "measures_standardized_residuals",
      "measures_standardized_residuals_estimated_weight",
      "measures_factor_scores",
      "measures_factor_score_precision",
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
      "frontier_fit_ml_ridge_continuation",
      "frontier_rbm",
      "frontier_pairwise_ordinal_composite_nested"
    ),
    helpers = c(
      "df_to_data",
      "df_to_fiml_data",
      "df_to_fcsem_data",
      "data_ordinal_stats_from_df",
      "data_mixed_ordinal_stats_from_df",
      "data_mixed_ordinal_stats_observed_from_df",
      "data_mixed_ordinal_stats_hybrid_fiml_from_df",
      "shrink_mixed_ordinal_stats",
      "fit_ml",
      "fit_ml_fisher",
      "fit_ml_fisher_snlls",
      "fit_ml_irls",
      "fit_ml_irls_snlls",
      "fit_fiml",
      "fit_ml2s",
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
      "fit_uls_ordinal",
      "fit_ordinal_stage2",
      "ordinal_stage2_weight_blocks",
      "fit_dwls_mixed_ordinal",
      "fit_wls_mixed_ordinal",
      "fit_uls_snlls",
      "fit_gls_snlls",
      "fit_wls_snlls",
      "evaluate_at",
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
      "data_ordinal_stats_observed_from_raw",
      "data_mixed_ordinal_stats_observed_from_raw",
      "data_mixed_ordinal_stats_hybrid_fiml_from_raw",
      "data_mixed_ordinal_stats_polyserial_dpd_from_raw",
      "data_mixed_ordinal_stats_huber_residual_from_raw",
      "data_ordinal_stats_from_raw_impl",
      "data_ordinal_stats_observed_from_raw_impl",
      "data_ordinal_stats_h_weighted_from_raw_impl",
      "data_ordinal_stats_dpd_from_raw_impl",
      "data_ordinal_stats_huber_residual_from_raw_impl",
      "data_mixed_ordinal_stats_from_raw_impl",
      "data_mixed_ordinal_stats_observed_from_raw_impl",
      "data_mixed_ordinal_stats_hybrid_fiml_from_raw_impl",
      "data_shrink_mixed_ordinal_stats_impl",
      "data_mixed_ordinal_stats_polyserial_dpd_from_raw_impl",
      "data_mixed_ordinal_stats_huber_residual_from_raw_impl",
      "fit_fit",
      "fit_ml_impl",
      "fit_ml_fisher_impl",
      "fit_ml_fisher_snlls_impl",
      "fit_ml_irls_impl",
      "fit_ml_irls_snlls_impl",
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
      "fit_uls_ordinal_impl",
      "fit_ordinal_stage2_impl",
      "ordinal_stage2_weight_blocks_impl",
      "fit_dwls_mixed_ordinal_impl",
      "fit_wls_mixed_ordinal_impl",
      "fit_uls_snlls_impl",
      "fit_gls_snlls_impl",
      "fit_wls_snlls_impl",
      "evaluate_at_impl",
      "fcsem_model_spec_impl",
      "fit_ml_fcsem_impl",
      "frontier_fit_ml_ridge_continuation_impl",
      "frontier_rbm_impl",
      "frontier_pairwise_ordinal_composite_nested_impl",
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
      "infer_information_cross_products",
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
      "infer_ordinal_lr_test_satorra2000",
      "infer_ml2s_lr_test_satorra2000",
      "infer_lr_test_satorra_bentler2001",
      "infer_lr_test_satorra_bentler2010",
      "infer_build_u_factor",
      "infer_build_u_factor_parts",
      "infer_build_u_factor_pairwise",
      "infer_reduced_gamma_nt",
      "infer_reduced_gamma_nt_pairwise",
      "infer_reduced_gamma_sample",
      "infer_reduced_gamma_sample_from_gamma",
      "infer_robust_test_moments_both_breads_zc",
      "infer_robust_test_moments_both_breads_gamma",
      "infer_reduced_gamma_sample_materialized",
      "infer_reduced_gamma_unbiased",
      "infer_ugamma_eigenvalues",
      "infer_satorra_bentler",
      "infer_mean_var_adjusted",
      "infer_scaled_shifted",
      "infer_casewise_contributions",
      "infer_pairwise_casewise_contributions",
      "data_pairwise_sample_stats",
      "data_gamma_nt_pairwise",
      "fit_gls_pairwise_impl",
      "infer_empirical_gamma",
      "infer_gamma_nt",
      "infer_ordinal_robust",
      "infer_mixed_ordinal_robust",
      "infer_mixed_ordinal_robust_ij",
      "infer_ordinal_profile_rmsea",
      "infer_ordinal_profile_lrt",
      "infer_ml_profile_lrt",
      "infer_continuous_ls_profile_lrt",
      "infer_fiml_profile_lrt",
      "infer_mixed_ordinal_fit_measures_misspec",
      "infer_mixed_ordinal_rmsea_misspec",
      "infer_mixed_ordinal_crmr_misspec",
      "infer_mixed_ordinal_cfi_tli_misspec",
      "infer_mixed_ordinal_profile_rmsea",
      "infer_mixed_ordinal_profile_lrt",
      "infer_robust_se",
      "infer_robust_se_parts",
      "infer_robust_se_raw",
      "infer_robust_se_raw_parts",
      "infer_baseline",
      "ordinal_catml_dwls_rmsea_impl",
      "compute_defined_impl",
      "measures_composite_weights",
      "infer_vcov_fit",
      "infer_z_test_fit",
      "infer_wald_test_fit",
      "infer_rls_chi2_fit",
      "infer_build_u_factor_fit",
      "infer_robust_se_fit",
      "infer_robust_se_raw_fit",
      "infer_casewise_scores_fit",
      "infer_casewise_influence_ij_fit",
      "infer_ordinal_casewise_influence_ij_fit",
      "infer_ml2s_casewise_influence_ij_fit",
      "infer_robust_se_zc",
      "infer_robust_se_both_breads",
      "infer_robust_se_both_breads_raw",
      "infer_robust_se_both_breads_zc"
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
