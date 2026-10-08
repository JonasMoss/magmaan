# Interior population; this fixed draw has an indefinite fitted factor block.
nested_boundary_sample <- function() {
  set.seed(129230105)
  n <- 100
  factors <- matrix(rnorm(n * 2), n, 2) %*% chol(matrix(c(1, .9, .9, 1), 2))
  setNames(as.data.frame(.7 * factors[, rep(1:2, each = 3)] +
    matrix(rnorm(n * 6, sd = sqrt(.51)), n, 6)), paste0("x", 1:6))
}

nested_boundary_alternative <- "f =~ x1 + x2 + x3\ng =~ x4 + x5 + x6"
nested_boundary_null <- "f =~ x1 + a*x2 + a*x3\ng =~ x4 + x5 + x6"

test_that("released-label tests compute at an improper null and match expected SB geometry", {
  skip_if_not_installed("lavaan")
  skip_if(as.character(utils::packageVersion("lavaan")) != "0.7.2")
  d <- nested_boundary_sample()
  f0 <- suppressWarnings(magmaan(nested_boundary_null, d, inference = FALSE))
  f1 <- suppressWarnings(magmaan(nested_boundary_alternative, d, inference = FALSE))
  pt <- as_lab_fit(f0)$partable
  value <- function(lhs, rhs) pt$est[pt$op == "~~" & pt$lhs == lhs & pt$rhs == rhs]
  expect_lt(value("f", "f") * value("g", "g") - value("f", "g")^2, 0)
  result <- anova(f0, f1, references = c("sb", "peba4"))
  expect_true(all(is.finite(result$statistic)))
  expect_true(all(is.finite(result$pvalue)))
  expect_true(all(is.na(result$reason)))
  expect_false(attr(result, "psd_boundary"))

  # The policy uses observed geometry. Its expected-geometry lab comparator
  # is Satorra-2000 with the exact restriction map, as in test-magmaan.R.
  shared <- magmaanlab::prepare_inference_data(as_lab_fit(f1))
  hypothesis <- magmaanlab::prepare_hypothesis(
    magmaanlab::prepare_inference(as_lab_fit(f0), shared),
    magmaanlab::prepare_inference(as_lab_fit(f1), shared))
  expected <- magmaanlab::calibrate_quadratic(
    magmaanlab::inference_quadratic(hypothesis, "lr", geometry = "expected"), "sb")
  lav0 <- suppressWarnings(lavaan::cfa(nested_boundary_null, d,
    meanstructure = TRUE, estimator = "MLM"))
  lav1 <- suppressWarnings(lavaan::cfa(nested_boundary_alternative, d,
    meanstructure = TRUE, estimator = "MLM"))
  oracle <- lavaan::lavTestLRT(lav0, lav1, method = "satorra.2000",
    A.method = "exact", scaled.shifted = FALSE)
  lab <- magmaanlab::robust_nested_lrt(as_lab_fit(f1), as_lab_fit(f0),
    data = d, A.method = "exact", method = "restriction_map")
  expect_equal(lab$T_scaled, oracle[["Chisq diff"]][2L], tolerance = 1e-5)
  expect_equal(expected$p_value, oracle[["Pr(>Chisq)"]][2L], tolerance = 1e-5)
})

test_that("PSD-boundary nested inputs retain the global boundary diagnostic", {
  d <- nested_boundary_sample()
  f0 <- suppressWarnings(magmaan(nested_boundary_null, d, covariance = "psd"))
  f1 <- suppressWarnings(magmaan(nested_boundary_alternative, d, covariance = "psd"))
  expect_true(f0$inference$psd_boundary)
  expect_false(as_lab_fit(f0)$diagnostics$newton_accuracy$covariance_interior)
  for (result in list(anova(f0, f1), anova(f1, f0),
                      anova(f0, f1, lavaan_compat = "MLM"))) {
    expect_true(all(is.finite(result$statistic)))
    expect_true(all(is.finite(result$pvalue)))
    expect_true(attr(result, "psd_boundary"))
    expect_true(all(result$reason == "psd_boundary"))
    expect_output(print(result), "assumes the population is interior")
  }
  lab <- magmaanlab::policy_nested(as_lab_fit(f1), as_lab_fit(f0))
  expect_true(lab$psd_boundary)
  expect_true(lab$score$available)
  expect_true(lab$lr$available)
  expect_true(magmaanlab::convention_nested(as_lab_fit(f1), as_lab_fit(f0), "MLM")$psd_boundary)
})
