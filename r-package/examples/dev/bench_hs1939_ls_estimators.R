library(magmaan)
library(lavaan)

model <- "visual  =~ x1 + x2 + x3
          textual =~ x4 + x5 + x6
          speed   =~ x7 + x8 + x9"
data <- lavaan::HolzingerSwineford1939

times <- 100L
lbfgsb <- list(max_iter = 5000L, ftol = 1e-12, gtol = 1e-8)
ceres <- list(max_iter = 500L, ftol = 1e-10, gtol = 1e-7, ptol = 1e-8)

lav_wls_ref <- lavaan::cfa(model, data, estimator = "WLS", baseline = FALSE)
W_wls <- lavaan::lavInspect(lav_wls_ref, "wls.v")
if (is.list(W_wls)) W_wls <- W_wls[[1L]]

ls_chisq <- function(fit, dat, multiplier) {
  multiplier * sum(dat$nobs - 1L) * fit$fmin
}

uls_chisq <- function(fit) {
  ## lavaan::fitMeasures(fit, "chisq") for estimator = "ULS" reports
  ## Browne's residual NT test, not the raw ULS objective statistic.
  ## magmaan's C++ Browne helper is N-scaled; lavaan's ULS test is on the
  ## same n - 1 scale as the standard ULS statistic for raw-data fits.
  magmaan_core$infer_browne_residual_nt(fit)$statistic *
    sum(fit$nobs - 1L) / sum(fit$nobs)
}

magmaan_uls_lbfgsb <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- fit_uls(spec, dat, lbfgsb = lbfgsb)
  uls_chisq(fit)
}

magmaan_uls_ceres <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- fit_uls_ceres(spec, dat, ceres = ceres)
  uls_chisq(fit)
}

magmaan_uls_snlls_lbfgsb <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- fit_uls_snlls(spec, dat, lbfgsb = lbfgsb)
  uls_chisq(fit)
}

magmaan_uls_snlls_ceres <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- fit_uls_snlls_ceres(spec, dat, ceres = ceres)
  uls_chisq(fit)
}

magmaan_gls_lbfgsb <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- fit_gls(spec, dat, lbfgsb = lbfgsb)
  ls_chisq(fit, dat, multiplier = 1)
}

magmaan_gls_ceres <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- fit_gls_ceres(spec, dat, ceres = ceres)
  ls_chisq(fit, dat, multiplier = 1)
}

magmaan_gls_snlls_lbfgsb <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- fit_gls_snlls(spec, dat, lbfgsb = lbfgsb)
  ls_chisq(fit, dat, multiplier = 1)
}

magmaan_gls_snlls_ceres <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- fit_gls_snlls_ceres(spec, dat, ceres = ceres)
  ls_chisq(fit, dat, multiplier = 1)
}

magmaan_wls_lbfgsb <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- fit_wls(spec, dat, W_wls, lbfgsb = lbfgsb)
  ls_chisq(fit, dat, multiplier = 2)
}

magmaan_wls_ceres <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- fit_wls_ceres(spec, dat, W_wls, ceres = ceres)
  ls_chisq(fit, dat, multiplier = 2)
}

magmaan_wls_snlls_lbfgsb <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- fit_wls_snlls(spec, dat, W_wls, lbfgsb = lbfgsb)
  ls_chisq(fit, dat, multiplier = 2)
}

magmaan_wls_snlls_ceres <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- fit_wls_snlls_ceres(spec, dat, W_wls, ceres = ceres)
  ls_chisq(fit, dat, multiplier = 2)
}

lavaan_uls <- function() {
  fit <- lavaan::cfa(model, data, estimator = "ULS", baseline = FALSE)
  unname(lavaan::fitMeasures(fit, "chisq"))
}

lavaan_gls <- function() {
  fit <- lavaan::cfa(model, data, estimator = "GLS", baseline = FALSE)
  unname(lavaan::fitMeasures(fit, "chisq"))
}

lavaan_wls <- function() {
  fit <- lavaan::cfa(model, data, estimator = "WLS", baseline = FALSE)
  unname(lavaan::fitMeasures(fit, "chisq"))
}

snlls_diagnostics <- function(estimator, backend) {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- switch(
    paste(estimator, backend, sep = "_"),
    ULS_lbfgsb = fit_uls_snlls(spec, dat, lbfgsb = lbfgsb),
    ULS_ceres = fit_uls_snlls_ceres(spec, dat, ceres = ceres),
    GLS_lbfgsb = fit_gls_snlls(spec, dat, lbfgsb = lbfgsb),
    GLS_ceres = fit_gls_snlls_ceres(spec, dat, ceres = ceres),
    WLS_lbfgsb = fit_wls_snlls(spec, dat, W_wls, lbfgsb = lbfgsb),
    WLS_ceres = fit_wls_snlls_ceres(spec, dat, W_wls, ceres = ceres)
  )
  data.frame(
    method = paste("magmaan", tolower(estimator), "snlls", backend, sep = "_"),
    iterations = fit$iterations,
    nonlinear = fit$snlls_nonlinear_npar,
    linear = fit$snlls_linear_npar,
    profiles = fit$snlls_profile_evaluations,
    profile_cache_hits = fit$snlls_profile_cache_hits,
    gradients = fit$snlls_gradient_evaluations,
    jacobians = fit$snlls_jacobian_evaluations
  )
}

cat("\nQuick returned statistics\n")
print(data.frame(
  method = c(
    "magmaan_uls_lbfgsb", "magmaan_uls_ceres",
    "magmaan_uls_snlls_lbfgsb", "magmaan_uls_snlls_ceres", "lavaan_uls",
    "magmaan_gls_lbfgsb", "magmaan_gls_ceres",
    "magmaan_gls_snlls_lbfgsb", "magmaan_gls_snlls_ceres", "lavaan_gls",
    "magmaan_wls_lbfgsb", "magmaan_wls_ceres",
    "magmaan_wls_snlls_lbfgsb", "magmaan_wls_snlls_ceres", "lavaan_wls"
  ),
  statistic = c(
    magmaan_uls_lbfgsb(), magmaan_uls_ceres(),
    magmaan_uls_snlls_lbfgsb(), magmaan_uls_snlls_ceres(), lavaan_uls(),
    magmaan_gls_lbfgsb(), magmaan_gls_ceres(),
    magmaan_gls_snlls_lbfgsb(), magmaan_gls_snlls_ceres(), lavaan_gls(),
    magmaan_wls_lbfgsb(), magmaan_wls_ceres(),
    magmaan_wls_snlls_lbfgsb(), magmaan_wls_snlls_ceres(), lavaan_wls()
  )
), digits = 5, row.names = FALSE)

cat("\nSNLLS diagnostics\n")
print(do.call(rbind, list(
  snlls_diagnostics("ULS", "lbfgsb"),
  snlls_diagnostics("ULS", "ceres"),
  snlls_diagnostics("GLS", "lbfgsb"),
  snlls_diagnostics("GLS", "ceres"),
  snlls_diagnostics("WLS", "lbfgsb"),
  snlls_diagnostics("WLS", "ceres")
)), row.names = FALSE)

cat("\nULS benchmark\n")
print(microbenchmark::microbenchmark(
  magmaan_lbfgsb = magmaan_uls_lbfgsb(),
  magmaan_ceres = magmaan_uls_ceres(),
  magmaan_snlls_lbfgsb = magmaan_uls_snlls_lbfgsb(),
  magmaan_snlls_ceres = magmaan_uls_snlls_ceres(),
  lavaan = lavaan_uls(),
  times = times
))

cat("\nGLS benchmark\n")
print(microbenchmark::microbenchmark(
  magmaan_lbfgsb = magmaan_gls_lbfgsb(),
  magmaan_ceres = magmaan_gls_ceres(),
  magmaan_snlls_lbfgsb = magmaan_gls_snlls_lbfgsb(),
  magmaan_snlls_ceres = magmaan_gls_snlls_ceres(),
  lavaan = lavaan_gls(),
  times = times
))

cat("\nWLS benchmark\n")
print(microbenchmark::microbenchmark(
  magmaan_lbfgsb = magmaan_wls_lbfgsb(),
  magmaan_ceres = magmaan_wls_ceres(),
  magmaan_snlls_lbfgsb = magmaan_wls_snlls_lbfgsb(),
  magmaan_snlls_ceres = magmaan_wls_snlls_ceres(),
  lavaan = lavaan_wls(),
  times = times
))
