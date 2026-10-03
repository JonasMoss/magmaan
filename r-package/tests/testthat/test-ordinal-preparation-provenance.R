test_that("released response scales retain preparation through post-fit reconstruction", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  ordered <- paste0("x", 1:4)
  for (v in ordered) d[[v]] <- as.integer(cut(d[[v]], 3))
  syntax <- "f =~ x1+x2+x3+x4\nx1 ~*~ c(1, NA)*x1"
  fit_one <- function(extra = "") {
    spec <- model_spec(paste(syntax, extra, sep = "\n"), ordered = ordered,
      parameterization = "delta", group = "school",
      group_labels = unique(as.character(d$school)), group_equal = "thresholds")
    fit_model(spec, d, estimator = "DWLS")
  }
  fit <- fit_one()
  expect_true(fit$converged)
  stamp <- attr(fit$partable, "magmaan.ordinal_preparation")
  expect_identical(stamp, rep(list(rep(2L, 4)), 2))
  n <- max(fit$partable$free)
  released <- with(fit$partable, op == "~~" & lhs == "x1" & rhs == "x1" & group == 2)
  expect_true(fit$partable$free[released] > 0L)
  robust <- robust_ordinal(fit, fit$ordinal_stats)
  ij <- vcov(fit, regime = "sandwich_ij")
  expect_equal(nrow(robust$vcov), n)
  expect_equal(nrow(ij), n)
  expect_true(all(is.finite(ij)))
  policy <- policy_inference(fit)
  expect_true(policy$covariance_available)
  expect_equal(nrow(policy$covariance), n)
  null <- fit_one("x2 ~~ c(1, 1)*x2")
  expect_true(null$converged)
  nested <- robust_nested_lrt(fit, null, data = fit$ordinal_stats, A.method = "delta")
  expect_equal(nested$df_diff, 1)
  expect_true(is.finite(nested$T_scaled))
  expect_identical(attr(fit$partable, "magmaan.ordinal_preparation"), stamp)
})
