test_that("continuous LS scaling advice is core-owned and weight-aware", {
  set.seed(331053)
  dat <- data.frame(x = rnorm(200), y = 40 * rnorm(200))
  syntax <- "x ~~ x\ny ~~ y\nx ~~ y"
  advice <- "observed variances differ by a factor.*consider rescaling"
  for (estimator in c("ULS", "DWLS", "WLS")) {
    expect_warning(fit <- fit_model(syntax, dat, estimator = estimator), advice)
    expect_gt(fit$diagnostics$observed_variance_ratio, 1000)
    expect_match(fit$diagnostics$numerical_scaling_message, advice)
    expect_no_warning(standardized <- fit_model(syntax, as.data.frame(scale(dat)),
                                                 estimator = estimator))
    expect_equal(standardized$diagnostics$observed_variance_ratio, 1)
  }
  for (estimator in c("ML", "GLS", "DLS")) {
    expect_no_warning(fit <- fit_model(syntax, dat, estimator = estimator))
    expect_gt(fit$diagnostics$observed_variance_ratio, 1000)
    expect_identical(fit$diagnostics$numerical_scaling_message, "")
  }
  # Literal covariance avoids floating-point sample-variance ambiguity at 1000.
  spec <- model_spec(syntax)
  for (ratio in c(1000, 1001)) {
    stats <- list(S = list(diag(c(1, ratio))), nobs = 200L)
    fit <- magmaan_core$fit_uls(spec, stats)
    expect_equal(fit$diagnostics$observed_variance_ratio, ratio)
    expect_identical(nzchar(fit$diagnostics$numerical_scaling_message), ratio > 1000)
  }
})

test_that("ML2S and mixed LS retain caller-unit scaling diagnostics", {
  set.seed(331054)
  dat <- data.frame(x = rnorm(200), y = 40 * rnorm(200))
  syntax <- "x ~~ x\ny ~~ y\nx ~~ y"
  spec <- model_spec(syntax, meanstructure = TRUE)
  raw <- list(X = list(as.matrix(dat)))
  sm <- magmaan_core$estimate_saturated_em_moments(raw)
  for (weight in c("uls", "dwls", "adf", "dls", "nt")) {
    fit <- magmaan_core$fit_ml2s(spec, raw, stage1 = sm, stage2_weight = weight)
    expect_gt(fit$diagnostics$observed_variance_ratio, 1000)
    expect_identical(nzchar(fit$diagnostics$numerical_scaling_message),
                     weight %in% c("uls", "dwls", "adf"))
  }
  dat$z <- as.integer(cut(rnorm(200), breaks = c(-Inf, 0, Inf)))
  mixed_syntax <- paste(syntax, "z ~~ z\nz ~~ x + y", sep = "\n")
  mixed_spec <- model_spec(mixed_syntax, ordered = "z", meanstructure = TRUE)
  expect_warning(fit <- fit_model(mixed_spec, dat, estimator = "DWLS"),
                         "observed variances differ by a factor")
  expect_gt(fit$diagnostics$observed_variance_ratio, 1000)
  ordinal <- dat
  ordinal$x <- as.integer(cut(dat$x, c(-Inf, 0, Inf)))
  ordinal$y <- as.integer(cut(dat$y, c(-Inf, 0, Inf)))
  all_spec <- model_spec(mixed_syntax, ordered = names(ordinal), meanstructure = TRUE)
  expect_no_warning(fit <- fit_model(all_spec, ordinal, estimator = "DWLS"))
  expect_equal(fit$diagnostics$observed_variance_ratio, 1)
  expect_identical(fit$diagnostics$numerical_scaling_message, "")
})
