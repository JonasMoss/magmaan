test_that("ML robust MI defaults infer the absence of a second-stage weight", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  f <- fit_model("f =~ x1 + x2 + x3 + x4", d)
  default <- modification_indices_robust(f)
  explicit <- modification_indices_robust(f, data = d, estimated_weight = FALSE)
  expect_equal(default, explicit)
  expect_error(modification_indices_robust(f, data = d, estimated_weight = TRUE), "not ML")
  expect_equal(policy_modification_indices(f)$statistic[policy_modification_indices(f)$reason == "available"],
    modification_indices_robust(f, data = d, information = "observed", estimated_weight = FALSE)$mi.scaled)
})
