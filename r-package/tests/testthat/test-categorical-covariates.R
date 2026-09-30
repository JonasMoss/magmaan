categorical_covariate_data <- function() {
  set.seed(41)
  x <- rnorm(300)
  z <- 0.8 * x + rnorm(300)
  data.frame(y = as.integer(cut(z, c(-Inf, -0.5, 0.5, Inf))), x = x)
}

covariate_error <- "fixed observed covariates are unsupported.*conditional moments"

test_that("categorical fixed-x fits fail before constructing marginal moments", {
  d <- categorical_covariate_data()
  spec <- model_spec("y ~ x", ordered = "y")
  for (estimator in c("DWLS", "WLS", "ULS")) {
    for (psd in c(FALSE, TRUE)) {
      expect_error(fit_model(spec, d, estimator = estimator, psd = psd), covariate_error)
    }
  }
  expect_error(fit_model("y ~ x", d, ordered = "y", estimator = "DWLS"), covariate_error)
  d$g <- rep(c("a", "b"), each = 150)
  expect_error(fit_model("y ~ x", d, ordered = "y", groups = "g",
                         estimator = "DWLS"), covariate_error)
  expect_error(data_mixed_ordinal_stats_from_df(d, spec), covariate_error)
  expect_error(data_mixed_ordinal_stats_observed_from_df(d, spec), covariate_error)
  expect_error(data_mixed_ordinal_stats_hybrid_fiml_from_df(d, spec), covariate_error)
  expect_error(prepare_model(spec, prototype = d), covariate_error)

  d$x <- as.integer(cut(d$x, c(-Inf, -0.5, 0.5, Inf)))
  spec <- model_spec("y ~ x", ordered = c("y", "x"))
  expect_error(data_ordinal_stats_from_df(d, spec), covariate_error)
})

test_that("precomputed categorical data cannot bypass the fixed-x check", {
  d <- categorical_covariate_data()
  joint <- model_spec("y ~ x", ordered = "y", fixed_x = FALSE)
  stats <- data_mixed_ordinal_stats_from_df(d, joint, full_wls_weight = FALSE)
  spec <- model_spec("y ~ x", ordered = "y")
  expect_error(magmaanlab:::fit_dwls_mixed_ordinal_impl(spec$partable, stats), covariate_error)
  for (model in list(spec, spec$partable)) {
    expect_error(fit_model(model, stats, estimator = "DWLS"), covariate_error)
    expect_error(fit_dwls_mixed_ordinal(model, stats), covariate_error)
    expect_error(fit_wls_mixed_ordinal(model, stats), covariate_error)
    expect_error(frontier_fit_mixed_ordinal_psd(model, stats), covariate_error)
  }
  d$x <- as.integer(cut(d$x, c(-Inf, -0.5, 0.5, Inf)))
  joint <- model_spec("y ~ x", ordered = c("y", "x"), fixed_x = FALSE)
  stats <- data_ordinal_stats_from_df(d, joint, full_wls_weight = FALSE)
  spec <- model_spec("y ~ x", ordered = c("y", "x"))
  expect_error(magmaanlab:::fit_dwls_ordinal_impl(spec$partable, stats), covariate_error)
  expect_error(fit_dwls_ordinal(spec, stats), covariate_error)
  expect_error(fit_wls_ordinal(spec, stats), covariate_error)
  expect_error(fit_uls_ordinal(spec, stats), covariate_error)
  expect_error(frontier_fit_ordinal_psd(spec, stats), covariate_error)
})

test_that("explicit joint categorical models and continuous fixed-x remain usable", {
  skip_if_not_installed("lavaan")
  d <- categorical_covariate_data()
  oracle <- lavaan::sem("y ~ x", d, ordered = "y", estimator = "DWLS")
  expect_true(lavaan::lavInspect(oracle, "options")$conditional.x)
  expect_lt(lavaan::fitMeasures(oracle, "fmin"), 1e-12)
  spec <- model_spec("y ~ x", ordered = "y", fixed_x = FALSE,
                     parameterization = "theta")
  fit <- fit_model(spec, d, estimator = "DWLS")
  joint_oracle <- lavaan::sem("y ~ x", d, ordered = "y", estimator = "DWLS",
                             fixed.x = FALSE, conditional.x = FALSE,
                             parameterization = "theta")
  expect_true(fit$converged)
  slope <- function(pt) pt$est[pt$op == "~" & pt$lhs == "y" & pt$rhs == "x"]
  expect_equal(slope(fit$partable), slope(lavaan::parTable(joint_oracle)), tolerance = 1e-5)

  d$y <- 0.8 * d$x + rnorm(nrow(d))
  expect_true(fit_model("y ~ x", d, estimator = "ML")$converged)
})
