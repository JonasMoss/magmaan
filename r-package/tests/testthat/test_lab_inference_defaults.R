test_that("generic ML inference uses observed empirical covariance and nested geometry", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  syntax <- "visual =~ x1 + a*x2 + b*x3\ntextual =~ x4+x5+x6"
  h1 <- fit_model(syntax, d)
  h0 <- fit_model(paste(syntax, "a == b", sep = "\n"), d)
  shared <- prepare_inference_data(h1, d)
  c1 <- prepare_inference(h1, shared)
  c0 <- prepare_inference(h0, shared)
  hyp <- prepare_hypothesis(c0, c1)
  expect_equal(inference_information(c1), inference_information(c1, "observed"))
  expect_equal(inference_covariance(c1), inference_covariance(c1, TRUE))
  expect_equal(as.numeric(inference_covariance(c1)),
    as.numeric(vcov(h1, "sandwich_observed")), tolerance = 1e-8)
  for (kind in c("score", "lr")) {
    expect_equal(inference_quadratic(hyp, kind)$statistic,
      inference_quadratic(hyp, kind, "observed")$statistic)
    expect_true(all(is.finite(score_spectrum(inference_quadratic(hyp, kind))$eigenvalues)))
    expect_equal(inference_quadratic(c1, kind)$statistic,
      inference_quadratic(c1, kind, "expected")$statistic)
  }
  expect_true(all(is.finite(modification_indices(h1, d)$mi.scaled)))
  expect_true(all(is.finite(score_tests(h0, d)$mi.scaled)))
  expect_identical(nested_score_test(h1, h0, d)$sensitivity, "observed")
})
