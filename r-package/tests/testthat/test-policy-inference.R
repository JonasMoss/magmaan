cfa <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"

test_that("policy_inference() equals the explicit composition of primitives", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  fit <- fit_model(cfa, d)
  res <- policy_inference(fit)
  expect_true(res$covariance_available)
  # Without a mean structure the observed-bread sandwich of centered moment
  # contributions is the exact-score sandwich.
  expect_equal(res$covariance, vcov(fit, regime = "robust", data = d),
               tolerance = 1e-7, ignore_attr = TRUE)
  ctx <- prepare_inference(fit)
  for (test in c("score", "lr")) {
    explicit <- calibrate_quadratic(inference_quadratic(ctx, test), c("sb", "peba4"))
    got <- res[[test]]
    expect_true(got$available)
    expect_equal(got$statistic, explicit$statistic[1])
    expect_equal(got$df, explicit$df[1])
    expect_equal(got$p_sb, explicit$p_value[explicit$method == "sb"], tolerance = 1e-10)
    expect_equal(got$p_peba4, explicit$p_value[explicit$method == "peba4"], tolerance = 1e-10)
  }
})

test_that("policy_inference() reports estimators outside its scope", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  res <- policy_inference(fit_model(cfa, d, estimator = "FIML"))
  expect_false(res$covariance_available)
  expect_null(res$covariance)
  expect_equal(c(res$covariance_reason, res$score$reason, res$lr$reason),
               rep("unsupported_model", 3))
  uls <- policy_inference(fit_model(cfa, d, estimator = "ULS"))
  expect_equal(uls$lr$reason, "unsupported_model")
})
