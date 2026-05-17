model <- "visual  =~ x1 + x2 + x3
          textual =~ x4 + x5 + x6
          speed   =~ x7 + x8 + x9"
data <- lavaan::HolzingerSwineford1939

magmaan_model <- \() {
  vars <- paste0("x", 1:9)
  X <- as.matrix(data[vars])
  ss <- magmaan_core$data_sample_stats_from_raw(X)
  pt <- magmaan_core$lavaan_lavaanify(model)
  fit <- magmaan_core$fit_fit(pt, ss)
  uf <- magmaan_core$infer_build_u_factor(fit)
  df <- magmaan_core$infer_df_stat(fit$partable, magmaan_core$fit_sample_stats(fit))
  chisq <- magmaan_core$infer_chi2_stat(magmaan_core$fit_sample_stats(fit), fit$fmin)
  Zc <- magmaan_core$infer_casewise_contributions(pt, X)  # N × p* centred contributions
  M  <- magmaan_core$infer_reduced_gamma_sample(uf, Zc, ss$nobs)
  ev <- magmaan_core$infer_ugamma_eigenvalues(M)
  sb <- magmaan_core$infer_satorra_bentler(chisq, df, ev)$chi2_scaled
  fit$partable$est
}

lavaan_model <- \() {
  xx <- lavaan::cfa(model, data, estimator = "MLM", baseline = FALSE)
  sb <- xx@test$satorra.bentler$stat
  xx@Fit@est
}

lavaan_model()
magmaan_model()
#microbenchmark::microbenchmark(magmaan_model(), lavaan_model())
