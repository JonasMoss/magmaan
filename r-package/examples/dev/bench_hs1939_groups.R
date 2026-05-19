library(magmaan)

model <- "visual  =~ x1 + x2 + x3
          textual =~ x4 + x5 + x6
          speed   =~ x7 + x8 + x9"
data <- lavaan::HolzingerSwineford1939

magmaan_model <- \() {
  spec <- model_spec(model, group = "school",
                     group_labels = levels(data$school))
  dat <- df_to_data(data, spec, group = "school")
  fit <- fit_ml(spec, dat)
  uf  <- magmaan_core$infer_build_u_factor(fit)
  Zc  <- magmaan_core$infer_casewise_contributions(spec$partable, dat$X)
  M   <- magmaan_core$infer_reduced_gamma_sample(uf, Zc, fit$nobs)
  ev  <- magmaan_core$infer_ugamma_eigenvalues(M)
  df <- magmaan_core$infer_df_stat(fit$partable, magmaan_core$fit_sample_stats(fit))
  chisq <- magmaan_core$infer_chi2_stat(magmaan_core$fit_sample_stats(fit), fit$fmin)
  sb <- magmaan_core$infer_satorra_bentler(chisq, df, ev)$chi2_scaled
  sb
}

lavaan_model <- \() {
  xx <- lavaan::cfa(model, data, estimator = "MLM", baseline = FALSE, group = "school")
  sb <- xx@test$satorra.bentler$stat
  sb
}

lavaan_model()
magmaan_model()

microbenchmark::microbenchmark(magmaan_model(), lavaan_model())
