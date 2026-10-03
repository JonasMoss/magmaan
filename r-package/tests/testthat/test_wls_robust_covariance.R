wls_robust_covariance_data <- function() {
  set.seed(20261001)
  n <- 900L
  eta <- rt(n, df = 6) * sqrt(2 / 3)
  nuisance <- rnorm(n)
  x <- sapply(seq_len(4L), function(j) {
    c(1, 0.8, 0.7, 0.9)[j] * eta + rnorm(n, sd = 0.75) +
      if (j <= 2L) 0.4 * nuisance else 0
  })
  colnames(x) <- paste0("x", 1:4)
  as.data.frame(x)
}

test_that("WLS robust MI and releases preserve empirical covariance and fitting W", {
  for (means in c(FALSE, TRUE)) {
    for (grouped in c(FALSE, TRUE)) {
      d <- wls_robust_covariance_data()
      if (grouped) d$g <- rep(c("a", "b"), c(520L, 380L))
      blocks <- if (grouped) {
        lapply(c("a", "b"), function(g) as.matrix(d[d$g == g, paste0("x", 1:4)]))
      } else list(as.matrix(d))
      gamma <- lapply(blocks, if (means) magmaan_core$robust_empirical_gamma_with_means else {
        magmaan_core$robust_empirical_gamma
      })
      for (full in c(FALSE, TRUE)) {
        weights <- lapply(gamma, function(g) {
          if (full) solve(g) else diag(1 / diag(g))
        })
        fit <- fit_model("f =~ x1 + c*x2 + c*x3 + x4", d, estimator = "WLS",
                         groups = if (grouped) "g" else NULL,
                         meanstructure = means, W = weights,
                         control = list(max_iter = 1200L, ftol = 1e-12, gtol = 1e-8))
        expect_true(fit$converged)
        for (release in c(FALSE, TRUE)) {
          worker <- if (release) score_tests_robust else modification_indices_robust
          ordinary <- if (release) score_tests(fit, weight = weights, bread = "expected", estimated_weight = FALSE, cov = "model_implied") else {
            modification_indices(fit, weight = weights, bread = "expected", estimated_weight = FALSE, cov = "model_implied")
          }
          empirical <- worker(fit, data = d, weight = weights, estimated_weight = FALSE, bread = "expected")
          normal <- worker(fit, weight = weights, cov = "model_implied", estimated_weight = FALSE, bread = "expected")
          expect_gt(nrow(empirical), 0L)
          expect_equal(empirical$mi, ordinary$mi, tolerance = 1e-8)
          expect_equal(normal$mi, empirical$mi, tolerance = 1e-8)
          expect_equal(worker(fit, data = blocks, weight = weights, estimated_weight = FALSE, bread = "expected"), empirical,
                       tolerance = 1e-10)
          expect_gt(max(abs(empirical$scaling.factor - normal$scaling.factor)), 1e-3)
          expect_true(all(is.finite(empirical$mi.scaled)))
          expect_equal(empirical$pvalue, pchisq(empirical$mi.scaled, empirical$df,
                                               lower.tail = FALSE), tolerance = 1e-12)
          if (full) {
            # W = Gamma_empirical^-1 gives meat = bread in the expected metric.
            expect_equal(empirical$scaling.factor, rep(1, nrow(empirical)),
                         tolerance = 1e-8)
            expect_equal(empirical$mi.scaled, ordinary$mi, tolerance = 1e-8)
          }
          scaled <- worker(fit, data = d, weight = lapply(weights, function(w) 3 * w), estimated_weight = FALSE, bread = "expected")
          expect_equal(scaled$mi, 3 * empirical$mi, tolerance = 1e-8)
          expect_equal(scaled$scaling.factor, 3 * empirical$scaling.factor,
                       tolerance = 1e-8)
          expect_equal(scaled$mi.scaled, empirical$mi.scaled, tolerance = 1e-8)
        }
      }
    }
  }
})

test_that("WLS covariance choices fail explicitly when unavailable", {
  d <- wls_robust_covariance_data()
  weight <- solve(magmaan_core$robust_empirical_gamma(as.matrix(d)))
  fit <- fit_model("f =~ x1 + c*x2 + c*x3 + x4", d, estimator = "WLS", W = weight)
  # The built-in ADF recipe builds the same weight and records its recipe.
  recipe <- fit_model("f =~ x1 + c*x2 + c*x3 + x4", d, estimator = "WLS")
  expect_equal(recipe$theta, fit$theta, tolerance = 1e-10)
  for (worker in list(modification_indices_robust, score_tests_robust)) {
    expect_error(worker(fit, weight = weight, estimated_weight = FALSE, bread = "expected"), "require.*fitting data")
    # The fit records its weight, so `weight` is optional and must match it.
    expect_equal(worker(fit, data = d, estimated_weight = FALSE, bread = "expected"), worker(fit, data = d, weight = weight, estimated_weight = FALSE, bread = "expected"),
                 tolerance = 0)
    expect_error(worker(fit, data = d, weight = diag(diag(weight)), estimated_weight = FALSE, bread = "expected"), "fit\\$W")
    expect_error(worker(fit, data = d, weight = weight, cov = "unknown", estimated_weight = FALSE, bread = "expected"),
                 "`cov` must be")
    expect_error(worker(fit, data = d, weight = weight, cov = "browne_unbiased", estimated_weight = FALSE, bread = "expected"),
                 "Browne-unbiased.*not implemented")
    observed <- worker(fit, data = d, weight = weight, bread = "observed", estimated_weight = FALSE)
    expect_true(nrow(observed) > 0L)
    expect_true(all(is.finite(observed$mi.scaled)))
    incomplete <- d
    incomplete$x2[1] <- NA_real_
    expect_error(worker(fit, data = incomplete, weight = weight, estimated_weight = FALSE, bread = "expected"), "non-finite")
    # A supplied W has no recipe, so its data influence is unknown.
    expect_error(worker(fit, data = d, estimated_weight = TRUE, bread = "expected"),
                 "UnsupportedInference")
    estimated <- worker(recipe, data = d, estimated_weight = TRUE, bread = "expected")
    expect_gt(nrow(estimated), 0L)
    expect_true(all(is.finite(estimated$mi.scaled)))
    expect_equal(estimated$mi, worker(fit, data = d, estimated_weight = FALSE, bread = "expected")$mi, tolerance = 1e-8)
    expect_error(worker(recipe, estimated_weight = TRUE, bread = "expected"),
                 "require.*fitting data")
    expect_error(worker(recipe, data = d, estimated_weight = TRUE,
                        cov = "model_implied", bread = "expected"), "requires empirical")
    expect_error(worker(recipe, data = d, estimated_weight = TRUE,
                        cov = "browne_unbiased", bread = "expected"), "Browne-unbiased.*not implemented")
  }
})
