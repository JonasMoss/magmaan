model <- "visual  =~ x1 + x2 + x3
          textual =~ x4 + x5 + x6
          speed   =~ x7 + x8 + x9"
data <- lavaan::HolzingerSwineford1939

magmaan_model <- \() {
  vars <- paste0("x", 1:9)

  Xg   <- lapply(split(data[vars], data$school), as.matrix)
  ssg  <- data_sample_stats_from_raw(Xg)

  pt  <- lavaan_lavaanify(model, n_groups = 2L, group_var = "school")
  fit <- fit_fit(pt, ssg)
  uf  <- infer_build_u_factor(fit)
  Zc  <- infer_casewise_contributions(pt, Xg)
  M   <- infer_reduced_gamma_sample(uf, Zc, fit$nobs)
  ev  <- infer_ugamma_eigenvalues(M)
  df <- infer_df_stat(fit$partable, fit_sample_stats(fit))
  chisq <- infer_chi2_stat(fit_sample_stats(fit), fit$fmin)
  sb <- infer_satorra_bentler(chisq, df, ev)$chi2_scaled
  #infer_robust_se_raw(fit, Xg)$se
  fit$partable$est
}

lavaan_model <- \() {
  xx <- lavaan::cfa(model, data, estimator = "MLM", baseline = FALSE, group = "school")
  sb <- xx@test$satorra.bentler$stat
  xx@Fit@est
}

lavaan_model()
magmaan_model()

#microbenchmark::microbenchmark(magmaan_model(), lavaan_model())
