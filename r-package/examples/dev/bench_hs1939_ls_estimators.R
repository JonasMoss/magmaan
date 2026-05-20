library(magmaan)
library(lavaan)

core <- magmaan_core

model <- "visual  =~ x1 + x2 + x3
          textual =~ x4 + x5 + x6
          speed   =~ x7 + x8 + x9"
data <- lavaan::HolzingerSwineford1939

times <- as.integer(Sys.getenv("MAGMAAN_BENCH_TIMES", "100"))
times <- 100
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
  core$infer_browne_residual_nt(fit)$statistic *
    sum(fit$nobs - 1L) / sum(fit$nobs)
}

magmaan_uls_lbfgsb <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- core$fit_uls(spec, dat, optimizer = "lbfgs", control = lbfgsb)
  uls_chisq(fit)
}

magmaan_uls_ceres <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- core$fit_uls(spec, dat, optimizer = "ceres", control = ceres)
  uls_chisq(fit)
}

magmaan_uls_snlls_lbfgsb <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- core$fit_uls_snlls(spec, dat, optimizer = "lbfgs", control = lbfgsb)
  uls_chisq(fit)
}

magmaan_uls_snlls_ceres <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- core$fit_uls_snlls(spec, dat, optimizer = "ceres", control = ceres)
  uls_chisq(fit)
}

magmaan_gls_lbfgsb <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- core$fit_gls(spec, dat, optimizer = "lbfgs", control = lbfgsb)
  ls_chisq(fit, dat, multiplier = 2)
}

magmaan_gls_ceres <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- core$fit_gls(spec, dat, optimizer = "ceres", control = ceres)
  ls_chisq(fit, dat, multiplier = 2)
}

magmaan_gls_snlls_lbfgsb <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- core$fit_gls_snlls(spec, dat, optimizer = "lbfgs", control = lbfgsb)
  ls_chisq(fit, dat, multiplier = 2)
}

magmaan_gls_snlls_ceres <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- core$fit_gls_snlls(spec, dat, optimizer = "ceres", control = ceres)
  ls_chisq(fit, dat, multiplier = 2)
}

magmaan_wls_lbfgsb <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- core$fit_wls(spec, dat, W_wls, optimizer = "lbfgs", control = lbfgsb)
  ls_chisq(fit, dat, multiplier = 2)
}

magmaan_wls_ceres <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- core$fit_wls(spec, dat, W_wls, optimizer = "ceres", control = ceres)
  ls_chisq(fit, dat, multiplier = 2)
}

magmaan_wls_snlls_lbfgsb <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- core$fit_wls_snlls(spec, dat, W_wls, optimizer = "lbfgs", control = lbfgsb)
  ls_chisq(fit, dat, multiplier = 2)
}

magmaan_wls_snlls_ceres <- function() {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  fit <- core$fit_wls_snlls(spec, dat, W_wls, optimizer = "ceres", control = ceres)
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

fit_for <- function(estimator, backend, snlls) {
  spec <- model_spec(model)
  dat <- df_to_data(data, spec, scaling = "n-1")
  key <- paste(estimator, if (snlls) "snlls" else "ordinary", backend, sep = "_")
  switch(
    key,
    ULS_ordinary_lbfgsb = core$fit_uls(spec, dat, optimizer = "lbfgs", control = lbfgsb),
    ULS_ordinary_ceres = core$fit_uls(spec, dat, optimizer = "ceres", control = ceres),
    ULS_snlls_lbfgsb = core$fit_uls_snlls(spec, dat, optimizer = "lbfgs", control = lbfgsb),
    ULS_snlls_ceres = core$fit_uls_snlls(spec, dat, optimizer = "ceres", control = ceres),
    GLS_ordinary_lbfgsb = core$fit_gls(spec, dat, optimizer = "lbfgs", control = lbfgsb),
    GLS_ordinary_ceres = core$fit_gls(spec, dat, optimizer = "ceres", control = ceres),
    GLS_snlls_lbfgsb = core$fit_gls_snlls(spec, dat, optimizer = "lbfgs", control = lbfgsb),
    GLS_snlls_ceres = core$fit_gls_snlls(spec, dat, optimizer = "ceres", control = ceres),
    WLS_ordinary_lbfgsb = core$fit_wls(spec, dat, W_wls, optimizer = "lbfgs", control = lbfgsb),
    WLS_ordinary_ceres = core$fit_wls(spec, dat, W_wls, optimizer = "ceres", control = ceres),
    WLS_snlls_lbfgsb = core$fit_wls_snlls(spec, dat, W_wls, optimizer = "lbfgs", control = lbfgsb),
    WLS_snlls_ceres = core$fit_wls_snlls(spec, dat, W_wls, optimizer = "ceres", control = ceres)
  )
}

iteration_diagnostics <- function() {
  rows <- expand.grid(
    estimator = c("ULS", "GLS", "WLS"),
    backend = c("lbfgsb", "ceres"),
    snlls = c(FALSE, TRUE),
    stringsAsFactors = FALSE
  )
  rows <- rows[order(rows$estimator, rows$backend, rows$snlls), ]
  diagnostics <- lapply(seq_len(nrow(rows)), function(i) {
    row <- rows[i, ]
    fit <- fit_for(row$estimator, row$backend, row$snlls)
    data.frame(
      estimator = row$estimator,
      method = if (row$snlls) "SNLLS" else "ordinary",
      backend = row$backend,
      iterations = fit$iterations,
      fmin = fit$fmin
    )
  })
  do.call(rbind, diagnostics)
}

cat("\nOptimizer iterations\n")
print(iteration_diagnostics(), digits = 6, row.names = FALSE)

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
