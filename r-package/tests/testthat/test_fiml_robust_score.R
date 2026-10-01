fiml_robust_score_data <- function(n = 600L, seed = 8123L) {
  set.seed(seed)
  eta <- rt(n, df = 8) * sqrt(0.75)
  x <- sapply(c(1, 0.8, 0.7, 0.9), function(loading) {
    loading * eta + rnorm(n, sd = 0.8)
  })
  colnames(x) <- paste0("x", 1:4)
  x[seq(1L, n, 9L), 2L] <- NA_real_
  x[seq(3L, n, 11L), 3L] <- NA_real_
  as.data.frame(x)
}

test_that("FIML robust MI and releases agree across retained and explicit data", {
  for (grouped in c(FALSE, TRUE)) {
    d <- fiml_robust_score_data()
    if (grouped) d$g <- rep(c("a", "b"), c(340L, 260L))
    fit <- fit_model("f =~ x1 + c*x2 + c*x3 + x4", d,
                     groups = if (grouped) "g" else NULL,
                     estimator = "FIML", meanstructure = TRUE,
                     control = list(max_iter = 1200L, ftol = 1e-12, gtol = 1e-8))
    expect_true(fit$converged)

    for (release in c(FALSE, TRUE)) {
      ordinary <- if (release) score_tests(fit) else modification_indices(fit)
      robust <- if (release) score_tests_robust(fit) else modification_indices_robust(fit)
      explicit <- if (release) score_tests_robust(fit, data = d) else {
        modification_indices_robust(fit, data = d)
      }
      expect_gt(nrow(robust), 0L)
      expect_equal(explicit, robust, tolerance = 1e-10)
      expect_equal(robust[c("kind", "row", "lhs", "op", "rhs", "group")],
                   ordinary[c("kind", "row", "lhs", "op", "rhs", "group")])
      expect_equal(robust$mi, ordinary$mi, tolerance = 1e-8)
      expect_true(all(is.finite(robust$scaling.factor) & robust$scaling.factor > 0))
      expect_equal(robust$mi.scaled, robust$mi / robust$scaling.factor,
                   tolerance = 1e-12)
      expect_true(all(is.finite(robust$pvalue)))
      expect_equal(robust$pvalue, pchisq(robust$mi.scaled, robust$df,
                                        lower.tail = FALSE), tolerance = 1e-12)
      # The cache is optional and must not be needed for an explicit-data call.
      uncached <- fit
      uncached$fiml_pack <- NULL
      uncached$fiml_h1 <- NULL
      rebuilt <- if (release) score_tests_robust(uncached) else {
        modification_indices_robust(uncached)
      }
      expect_equal(rebuilt, robust, tolerance = 1e-10)
      uncached$raw_data <- NULL
      supplied <- if (release) score_tests_robust(uncached, data = d) else {
        modification_indices_robust(uncached, data = d)
      }
      expect_equal(supplied, robust, tolerance = 1e-10)
    }
    mi <- modification_indices_robust(fit)
    marker <- mi$lhs == "f" & mi$op == "=~" & mi$rhs == "x1"
    if (grouped) {
      # Shared loading labels anchor each group's scale through the other group.
      expect_true(any(marker))
      unlabelled <- fit_model("f =~ x1+x2+x3+x4", d, groups = "g",
                              estimator = "FIML", meanstructure = TRUE)
      expect_true(unlabelled$converged)
      mi_unlabelled <- modification_indices_robust(unlabelled)
      expect_equal(nrow(mi_unlabelled), 12L)
      expect_true(all(mi_unlabelled$op == "~~" &
                       mi_unlabelled$lhs != mi_unlabelled$rhs))
    } else {
      expect_false(any(marker))
    }
  }
})

test_that("FIML explicit score data rebuild their missingness pack", {
  d <- fiml_robust_score_data()
  fit <- fit_model("f =~ x1 + c*x2 + c*x3 + x4", d, estimator = "FIML",
                   meanstructure = TRUE)
  changed <- d
  changed$x4[seq(4L, nrow(d), 7L)] <- NA_real_
  changed$x2 <- 0.95 * changed$x2
  masked <- list(X = as.matrix(d), mask = !is.na(as.matrix(d)))
  masked$X[!masked$mask] <- 1e6
  for (worker in list(modification_indices_robust, score_tests_robust)) {
    expect_equal(worker(fit, data = masked), worker(fit), tolerance = 1e-10)
    explicit <- worker(fit, data = changed)
    rebuilt <- fit
    rebuilt$raw_data <- magmaanlab:::raw_data_arg(fit, changed)
    rebuilt$fiml_pack <- NULL
    rebuilt$fiml_h1 <- NULL
    expect_equal(explicit, worker(rebuilt), tolerance = 1e-10)
    expect_false(isTRUE(all.equal(explicit$mi, worker(fit)$mi)))
  }
})

test_that("FIML robust MI retains identified fixed loadings", {
  d <- fiml_robust_score_data()
  fit <- fit_model("f =~ 1*x1+x2+x3+x4\nf ~~ 1*f", d,
                   estimator = "FIML", meanstructure = TRUE)
  expect_true(fit$converged)
  mi <- modification_indices_robust(fit)
  expect_true(any(mi$lhs == "f" & mi$op == "=~" & mi$rhs == "x1"))
  expect_equal(mi$mi, modification_indices(fit)$mi, tolerance = 1e-8)
})

test_that("FIML robust score wrappers reject incompatible conventions", {
  fit <- fit_model("f =~ x1 + c*x2 + c*x3 + x4", fiml_robust_score_data(),
                   estimator = "FIML", meanstructure = TRUE)
  for (worker in list(modification_indices_robust, score_tests_robust)) {
    expect_error(worker(fit, bread = "expected"), "bread='observed'")
    expect_error(worker(fit, bread = "invalid"), "bread='observed'")
    expect_error(worker(fit, moments = "unstructured"), "observed-pattern")
    expect_error(worker(fit, cov = "model_implied"), "observed-pattern")
    expect_error(worker(fit, cov = "browne_unbiased"), "observed-pattern")
    expect_error(worker(fit, estimated_weight = TRUE), "no second-stage weight")
    expect_error(worker(fit, weight = diag(10)), "no second-stage weight")
    no_data <- fit
    no_data$raw_data <- NULL
    expect_error(worker(no_data), "requires fit\\$raw_data or data=")
  }
  expect_error(modification_indices_robust(fit, information = "expected"),
               "information='observed'")
  expect_error(modification_indices_robust(fit, information = "invalid"),
               "information must be")
})
