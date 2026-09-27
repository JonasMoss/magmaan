# Frontier fits are first-class magmaan fits: they carry the magmaan_fit class,
# so vcov() and residuals() apply as for ordinary ML, and a route, so
# refit-based methods refit them with their own constraint or penalty.

# Raw data whose sample covariance (divisor n) is exactly a one-factor
# covariance with a Heywood case on x1 (loading 1.03 at unit variances).
heywood_frame <- function(n = 80L) {
  l <- c(1.03, 0.8, 0.6, 0.5)
  S <- tcrossprod(l); diag(S) <- 1
  set.seed(11)
  Z <- matrix(stats::rnorm(n * 4L), n)
  Z <- scale(Z, scale = FALSE)
  Z <- Z %*% solve(chol(crossprod(Z) / n))
  X <- Z %*% chol(S)
  colnames(X) <- paste0("x", 1:4)
  as.data.frame(X)
}
one_factor <- "f =~ x1 + x2 + x3 + x4"

test_that("frontier fitters return magmaan_fit objects with a route", {
  dat <- heywood_frame()
  fits <- list(
    frontier_fit_ml_psd = frontier_fit_ml_psd(one_factor, dat),
    frontier_fit_ml_multiinfo = frontier_fit_ml_multiinfo(
      one_factor, dat, weight = 0.5, target = "determinacy"),
    frontier_fit_uls_psd = frontier_fit_uls_psd(one_factor, dat),
    frontier_fit_gls_psd = frontier_fit_gls_psd(one_factor, dat))
  for (name in names(fits)) {
    fit <- fits[[name]]
    expect_s3_class(fit, "magmaan_fit")
    expect_identical(fit$options$route$fitter, name)
    expect_false("model" %in% names(fit$options$route$args))
    # Information regimes are ML-only; least squares fits use the sandwich.
    V <- if (identical(fit$options$estimator, "ML")) {
      vcov(fit, regime = "information_expected")
    } else vcov(fit)
    expect_equal(dim(V), rep(length(fit$theta), 2L))
    expect_true(all(is.finite(V)))
  }
  fb <- frontier_fit_ml_psd_fallback(one_factor, dat)
  expect_s3_class(fb$fit, "magmaan_fit")
  expect_identical(fb$fit$options$route$extract, "fit")
})

test_that("vcov() of a barrier fit is the ML formula at the barrier estimate", {
  dat <- heywood_frame()
  fit <- frontier_fit_ml_multiinfo(one_factor, dat, weight = 0.5, target = "determinacy")
  direct <- magmaan_core$inference_vcov(magmaan_core$inference_information_expected(fit), fit)
  expect_equal(unname(vcov(fit, regime = "information_expected")), unname(direct))
  # The default regime (the expected-bread sandwich) uses the retained raw data.
  expect_true(all(is.finite(vcov(fit))))
})

test_that("case_rerun() refits a barrier fit with its penalty", {
  dat <- heywood_frame()
  fit <- frontier_fit_ml_multiinfo(one_factor, dat, weight = 0.5, target = "determinacy")
  cr <- case_rerun(fit, dat, to_rerun = 1:2)
  expect_true(all(cr$converged))
  for (j in 1:2) {
    expect_equal(cr$rerun[[j]]$penalty$weight, 0.5)
    expect_identical(cr$rerun[[j]]$penalty$type, "determinacy")
    hand <- frontier_fit_ml_multiinfo(one_factor, dat[-j, ], weight = 0.5,
                                      target = "determinacy")
    expect_equal(cr$rerun[[j]]$theta, hand$theta, tolerance = 1e-4)
  }
})

test_that("case_rerun() keeps the PSD constraint of a fit_model(psd = TRUE) fit", {
  dat <- heywood_frame()
  fit <- fit_model(one_factor, dat, psd = TRUE)
  cr <- case_rerun(fit, dat, to_rerun = 1:2)
  for (j in 1:2) {
    hand <- fit_model(one_factor, dat[-j, ], psd = TRUE)
    expect_true(isTRUE(cr$rerun[[j]]$options$psd))
    expect_equal(cr$rerun[[j]]$theta, hand$theta, tolerance = 1e-4)
    # An ordinary refit would put x1's residual variance below zero.
    x1 <- cr$rerun[[j]]$partable
    expect_gte(x1$est[x1$lhs == "x1" & x1$op == "~~" & x1$rhs == "x1"], 0)
  }
})

test_that("case_rerun() refuses a route it cannot reproduce", {
  dat <- heywood_frame()
  fit <- fit_model(one_factor, dat, estimator = "FIML", psd = TRUE)
  expect_error(case_rerun(fit, dat, to_rerun = 1), "cannot refit")
})

test_that("likelihood-ratio refits keep the barrier and the PSD constraint", {
  dat <- heywood_frame()
  bar <- frontier_fit_ml_multiinfo(one_factor, dat, weight = 0.5, target = "determinacy")
  refit <- magmaanlab:::.lrt_refit(bar, dat, extra_syntax = "x3 ~~ x4")
  expect_equal(refit$penalty$weight, 0.5)
  expect_identical(refit$penalty$type, "determinacy")
  psd <- fit_model(one_factor, dat, psd = TRUE)
  expect_true(isTRUE(magmaanlab:::.lrt_refit(psd, dat, extra_syntax = "x3 ~~ x4")$options$psd))
})
