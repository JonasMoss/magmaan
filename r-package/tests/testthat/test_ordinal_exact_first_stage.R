test_that("all-ordinal exact first-stage comparator preserves OPG and policy defaults", {
  set.seed(74)
  n <- 400L
  f <- rnorm(n)
  d <- as.data.frame(sapply(c(.82,.76,.70,.64), function(l) {
    y <- l*f + sqrt(1-l*l)*rnorm(n)
    1L + (y > -.45) + (y > .55)
  }))
  names(d) <- paste0("x", 1:4)
  d$g <- rep(c("a", "b"), each = n/2)
  for (parameterization in c("delta", "theta")) {
    for (grouped in c(FALSE, TRUE)) {
      fit <- fit_model("f =~ x1 + .9*x2 + x3 + x4", d, estimator = "DWLS",
                       ordered = paste0("x", 1:4), parameterization = parameterization,
                       groups = if (grouped) "g" else NULL)
      baseline <- magmaan_core$robust_ordinal_ij(fit, fit$ordinal_stats)
      expect_identical(baseline, magmaan_core$robust_ordinal_ij(fit, fit$ordinal_stats, first_stage = "opg"))
      policy <- policy_inference(fit)
      exact <- magmaan_core$robust_ordinal_ij(fit, fit$ordinal_stats, first_stage = "exact")
      expect_true(all(is.finite(exact$vcov)))
      expect_gt(max(abs(exact$vcov-baseline$vcov)), 0)
      expect_length(exact$sampling_gamma, if (grouped) 2L else 1L)
      for (g in seq_along(exact$sampling_gamma)) {
        rows <- exact$sampling_moment_influence[[g]]
        expect_equal(exact$sampling_gamma[[g]], crossprod(rows)/nrow(rows), tolerance = 1e-12)
        expect_lt(max(abs(colMeans(rows))), 1e-10)
      }
      expect_identical(policy_inference(fit), policy)
      expect_identical(magmaan_core$robust_ordinal_ij(fit, fit$ordinal_stats), baseline)
      expect_error(magmaan_core$robust_ordinal_ij(fit, fit$ordinal_stats, first_stage = "unknown"), "first_stage")
    }
  }
})
