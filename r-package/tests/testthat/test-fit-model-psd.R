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
