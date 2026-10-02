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

test_that("penalized fits get no policy inference, for every estimator", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  for (estimator in c("ML", "FIML", "ULS")) {
    fit <- fit_model(cfa, d, estimator = estimator, covariance = "barrier")
    res <- policy_inference(fit)
    expect_false(res$covariance_available)
    expect_null(res$covariance)
    expect_equal(c(res$covariance_reason, res$score$reason, res$lr$reason),
                 rep("penalized", 3), label = estimator)
  }
  barrier <- fit_model(cfa, d, covariance = "barrier")
  plain <- fit_model(cfa, d)
  for (nested in list(policy_nested(plain, barrier), policy_nested(barrier, plain))) {
    expect_equal(c(nested$lr$reason, nested$score$reason), rep("penalized", 2))
  }
})

test_that("inference_rows() rebuilds the score statistic and its spectrum", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  m1 <- "visual =~ x1 + a*x2 + b*x3\ntextual =~ x4 + x5 + x6\nspeed =~ x7 + x8 + x9"
  f1 <- fit_model(m1, d)
  f0 <- fit_model(paste(m1, "a == b", sep = "\n"), d)
  shared <- prepare_inference_data(f1, d, storage = "casewise")
  c1 <- prepare_inference(f1, shared)
  c0 <- prepare_inference(f0, shared)
  for (q in list(inference_quadratic(c1, "score"),
                 inference_quadratic(prepare_hypothesis(c0, c1), "score"))) {
    rows <- inference_rows(q)
    expect_equal(nrow(rows), nrow(d))
    expect_equal(sum(colSums(rows)^2), q$statistic, tolerance = 1e-10)
    expect_equal(sort(eigen(crossprod(rows), symmetric = TRUE, only.values = TRUE)$values),
                 sort(score_spectrum(q)$eigenvalues), tolerance = 1e-8)
  }
})
