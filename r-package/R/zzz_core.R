magmaan_core <- local({
  groups <- list(
    parse = c(
      "parse_parse"
    ),
    compat_lavaan = c(
      "lavaan_lavaanify",
      "lavaan_compare_partable"
    ),
    model = c(
      "model_matrix_rep",
      "model_implied"
    ),
    data = c(
      "data_sample_stats_from_raw",
      "data_ordinal_stats_from_raw_impl",
      "data_ordinal_stats_h_weighted_from_raw_impl",
      "data_ordinal_stats_dpd_from_raw_impl",
      "data_mixed_ordinal_stats_from_raw_impl",
      "data_shrink_mixed_ordinal_stats_impl",
      "data_mixed_ordinal_stats_polyserial_dpd_from_raw_impl",
      "data_mixed_ordinal_stats_huber_residual_from_raw_impl"
    ),
    estimate = c(
      "fit_fit",
      "fit_ml_impl",
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
      "fit_uls_ceres_impl",
      "fit_gls_ceres_impl",
      "fit_wls_ceres_impl",
      "fit_uls_snlls_ceres_impl",
      "fit_gls_snlls_ceres_impl",
      "fit_wls_snlls_ceres_impl",
      "fit_start_values",
      "fit_sample_stats",
      "estimate_fiml_robust_mlr"
    ),
    inference = c(
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
      "inference_modification_indices",
      "inference_score_tests"
    ),
    robust = c(
      "infer_lr_test_satorra2000",
      "infer_lr_test_satorra_bentler2001",
      "infer_lr_test_satorra_bentler2010",
      "infer_build_u_factor",
      "infer_build_u_factor_parts",
      "infer_reduced_gamma_nt",
      "infer_reduced_gamma_sample",
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
      "infer_robust_se_raw_parts"
    ),
    measures = c(
      "infer_baseline",
      "measures_fit",
      "measures_standardize_lv",
      "measures_standardize_all",
      "compute_defined_impl"
    ),
    helpers = c(
      "df_to_data",
      "df_to_fiml_data",
      "data_ordinal_stats_from_df",
      "data_mixed_ordinal_stats_from_df",
      "shrink_mixed_ordinal_stats",
      "fit_ml",
      "fit_fiml",
      "fit_uls",
      "fit_gls",
      "fit_wls",
      "fit_dwls_ordinal",
      "fit_wls_ordinal",
      "fit_dwls_mixed_ordinal",
      "fit_wls_mixed_ordinal",
      "fit_uls_snlls",
      "fit_gls_snlls",
      "fit_wls_snlls",
      "fit_uls_ceres",
      "fit_gls_ceres",
      "fit_wls_ceres",
      "fit_uls_snlls_ceres",
      "fit_gls_snlls_ceres",
      "fit_wls_snlls_ceres",
      "infer_vcov_fit",
      "infer_z_test_fit",
      "infer_wald_test_fit",
      "infer_rls_chi2_fit",
      "infer_build_u_factor_fit",
      "infer_robust_se_fit",
      "infer_robust_se_raw_fit"
    )
  )
  core_names <- unique(unlist(groups, use.names = FALSE))

  ns <- parent.env(environment())
  out <- list2env(mget(core_names, envir = ns, inherits = FALSE),
                  parent = emptyenv())
  attr(out, "groups") <- groups
  lockEnvironment(out, bindings = TRUE)
  out
})
