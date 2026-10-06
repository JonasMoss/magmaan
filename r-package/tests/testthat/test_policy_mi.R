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


test_that("observed likelihood MI retains saddle candidates and expected EPCs", {
  d <- lavaan::HolzingerSwineford1939
  syntax <- "visual =~ x1+x2+x3
textual =~ x4+x5+x6
speed =~ x7+x8+x9"
  f <- fit_model(syntax, d)
  robust <- modification_indices_robust(f, bread = "observed")
  expect_equal(nrow(robust), 54L)
  expect_true(all(robust$reason == "available"))
  expect_true(all(is.finite(robust$mi.scaled)))
  expected <- modification_indices(f, bread = "expected", cov = "model_implied", estimated_weight = FALSE)
  key <- function(x) paste(x$lhs, x$op, x$rhs)
  ix <- match(key(robust), key(expected))
  expect_equal(robust$epc, expected$epc[ix], tolerance = 1e-10)
  expect_equal(robust$epc.lv, expected$epc.lv[ix], tolerance = 1e-10)
  expect_equal(robust$epc.all, expected$epc.all[ix], tolerance = 1e-10)
  expect_equal(robust$mi.scaled, robust$score^2 / robust$v.eff, tolerance = 1e-10)
  expect_equal(robust, modification_indices_robust(f, bread = "observed", information = "observed"))
  # Centered moment meat isolates a genuinely zero variance. Likelihood meat
  # also retains the nonzero mean score at a misspecified evaluation point.
  zero <- modification_indices_robust(f, gamma = matrix(0, 45, 45),
                                     moments = "structured", estimated_weight = FALSE)
  expect_equal(nrow(zero), 54L)
  expect_true(all(zero$reason == "numeric_failure"))
  expect_true(all(is.na(zero$mi.scaled)))
  expect_true(all(nzchar(zero$detail)))
})


test_that("FIML robust MI accepts its expected metric with observed sensitivity", {
  d <- lavaan::HolzingerSwineford1939
  d$x2[d$x1 > median(d$x1)] <- NA_real_
  f <- fit_model("f =~ x1+x2+x3+x4", d, estimator = "FIML")
  expect_equal(modification_indices_robust(f),
               modification_indices_robust(f, information = "expected"))
  expect_equal(modification_indices_robust(f),
               modification_indices_robust(f, information = "observed"))
})
