library(magmaan)

model <- "visual  =~ x1 + x2 + x3
          textual =~ x4 + x5 + x6
          speed   =~ x7 + x8 + x9"
data <- lavaan::HolzingerSwineford1939

magmaan_model <- \() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec)
  fit <- fit_ml(spec, dat)
  uf <- infer_build_u_factor(fit)
  df <- infer_df_stat(fit$partable, fit_sample_stats(fit))
  chisq <- infer_chi2_stat(fit_sample_stats(fit), fit$fmin)
  Zc <- infer_casewise_contributions(spec$partable, dat$X)
  M  <- infer_reduced_gamma_sample(uf, Zc, dat$nobs)
  ev <- infer_ugamma_eigenvalues(M)
  sb <- infer_satorra_bentler(chisq, df, ev)$chi2_scaled
  sb
}

lavaan_model <- \() {
  xx <- lavaan::cfa(model, data, estimator = "MLM", baseline = FALSE)
  sb <- xx@test$satorra.bentler$stat
  sb
}

lavaan_model()
magmaan_model()
microbenchmark::microbenchmark(magmaan_model(), lavaan_model())
