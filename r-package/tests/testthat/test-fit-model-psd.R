cfa <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"

test_that("fit_model(psd = TRUE) dispatches to the PSD-constrained fitters", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  psd <- fit_model(cfa, d, psd = TRUE)
  direct <- frontier_fit_ml_psd(model_spec(cfa), d)
  expect_true(psd$options$psd)
  expect_false(fit_model(cfa, d)$options$psd)
  expect_equal(psd$theta, direct$theta, tolerance = 1e-8)
  fiml <- fit_model(cfa, d, estimator = "FIML", psd = TRUE)
  expect_true(isTRUE(fiml$converged))
  expect_true(fiml$options$psd)
})

test_that("fit_model(psd = TRUE) refuses conflicting options", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  expect_error(fit_model(cfa, d, psd = NA), "TRUE or FALSE")
  expect_error(fit_model(cfa, d, psd = TRUE, bounds = "standard"),
               "replaces `bounds`")
})

test_that("frontier PSD square factor agrees with triangular on three-factor CFA", {
  skip_if_not_installed("lavaan")
  model <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6\nspeed =~ x7 + x8 + x9"
  d <- lavaan::HolzingerSwineford1939
  triangular <- frontier_fit_ml_psd(model, d, psd_factor = "triangular")
  square <- frontier_fit_ml_psd(model, d, psd_factor = "square")
  expect_lt(max(abs(square$theta - triangular$theta)), 1e-5)
  expect_lte(abs(square$fmin - triangular$fmin),
             1e-8 * max(abs(square$fmin), abs(triangular$fmin)))
  expect_error(frontier_fit_ml_psd(model, d, psd_factor = "unknown"),
               "triangular.*square")
  expect_error(frontier_fit_ml_psd_impl(partable_arg(model_spec(model)),
               list(S = cov(d[paste0("x", 1:9)]), nobs = nrow(d)),
               psd_factor = "unknown"), "triangular or square")
})
