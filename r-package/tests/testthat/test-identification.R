library(magmaanlab)

# Structural identification (board TASK-33.3): an unidentified model is never
# reported as converged, whatever the local Newton check says, and the report
# names the parameter combination that leaves the implied moments unchanged.

interior <- list(S = list(matrix(c(1, .5, .4, .5, 1, .32, .4, .32, 1), 3)),
                 nobs = 300L)

testthat::test_that("the free-marker PSD ridge is rejected as unidentified", {
  ridge <- model_spec("f =~ x1 + x2 + x3", auto_fix_first = FALSE)
  e <- frontier_fit_ml_psd_fallback(ridge, interior,
                                    ordinary_control = list(max_iter = 1L))
  testthat::expect_false(e$converged)
  testthat::expect_null(e$fit)
  psd <- e$psd$fit
  testthat::expect_false(isTRUE(psd$converged))
  id <- psd$diagnostics$identification
  testthat::expect_identical(id$status, "unidentified")
  testthat::expect_identical(id$reason, "counting_rule")
  testthat::expect_identical(id$n_parameters, 7L)
  testthat::expect_identical(id$n_moments, 6L)
  testthat::expect_identical(id$rank, 6L)
  testthat::expect_identical(psd$verdict$identification, "failed")
  testthat::expect_identical(psd$verdict$status, "failed")
  testthat::expect_identical(dim(id$null_directions), c(7L, 1L))
  testthat::expect_true(all(c("f=~x1", "f~~f") %in% rownames(id$null_directions)))
  testthat::expect_match(id$null_direction_text, "f=~x1", fixed = TRUE)
  testthat::expect_match(id$null_direction_text, "f~~f", fixed = TRUE)

  marker <- frontier_fit_ml_psd_fallback(model_spec("f =~ x1 + x2 + x3"), interior)
  testthat::expect_true(marker$converged)
  testthat::expect_identical(marker$fit$diagnostics$identification$status, "identified")
  testthat::expect_identical(marker$fit$verdict$identification, "passed")
})

testthat::test_that("the direct-FIML ridge passes the counting rule but not the rank", {
  set.seed(20261007)
  x <- outer(rnorm(300), c(.8, .7, .6, .5)) + matrix(rnorm(1200, sd = .6), 300, 4)
  x <- sweep(x, 2, c(2, -1, .5, 4), "+")
  colnames(x) <- paste0("x", 1:4)
  raw <- list(X = list(x), mask = list(matrix(TRUE, 300, 4)))
  ridge <- model_spec("f =~ x1 + x2 + x3 + x4", auto_fix_first = FALSE,
                      meanstructure = TRUE, fixed_x = FALSE)
  fit <- suppressWarnings(magmaan_core$fit_fiml(ridge, raw))
  testthat::expect_false(fit$converged)
  id <- fit$diagnostics$identification
  testthat::expect_identical(id$status, "unidentified")
  testthat::expect_identical(id$reason, "rank")
  testthat::expect_true(id$counting_rule)
  testthat::expect_identical(id$map, "covariance_mean")
  testthat::expect_identical(id$n_parameters - id$rank, 1L)
  testthat::expect_true(id$directions_at_estimate)
  # The loading/variance rescaling at the estimate: d lambda = lambda,
  # d psi = -2 psi, nothing else.
  d <- id$null_directions[, 1]
  pt <- fit$partable[fit$partable$free > 0, ]
  load <- pt$op == "=~"
  psi <- pt$op == "~~" & pt$lhs == "f" & pt$rhs == "f"
  expected <- numeric(length(d))
  expected[pt$free[load]] <- fit$theta[pt$free[load]]
  expected[pt$free[psi]] <- -2 * fit$theta[pt$free[psi]]
  testthat::expect_equal(abs(sum(d * expected)) / sqrt(sum(expected^2)), 1,
                         tolerance = 1e-8)

  audit <- magmaan_core$frontier_fiml_newton_audit(fit)
  testthat::expect_identical(audit$identification$status, "unidentified")

  marker <- model_spec("f =~ x1 + x2 + x3 + x4", meanstructure = TRUE,
                       fixed_x = FALSE)
  control <- suppressWarnings(magmaan_core$fit_fiml(marker, raw))
  testthat::expect_true(control$converged)
  testthat::expect_identical(control$diagnostics$identification$status, "identified")
})

testthat::test_that("identified and constrained controls keep their verdicts under unit changes", {
  set.seed(20261008)
  eta1 <- rnorm(400)
  eta2 <- .5 * eta1 + sqrt(.75) * rnorm(400)
  x <- cbind(outer(eta1, c(.8, .7)), outer(eta2, c(.6, .5))) +
    matrix(rnorm(1600, sd = .6), 400, 4)
  colnames(x) <- paste0("x", 1:4)
  for (units in list(c(1, 1, 1, 1), c(10, .1, 1, 10))) {
    d <- as.data.frame(sweep(x, 2, units, "*"))
    for (syntax in c("f =~ x1 + x2 + x3 + x4",
                     "f =~ NA*a*x1 + b*x2 + c*x3 + d*x4\na + b + c + d == 4",
                     "f1 =~ x1 + x2\nf2 =~ x3 + x4")) {
      fit <- suppressWarnings(fit_model(syntax, d))
      testthat::expect_identical(fit$diagnostics$identification$status, "identified",
                                 info = syntax)
      testthat::expect_true(fit$converged, info = paste(syntax, units[1]))
    }
  }
})

testthat::test_that("standalone reports and prepared fits reuse construction checks without refusing", {
  before <- prepared_identification_count_impl()
  ridge <- prepare_model(model_spec("f =~ NA*x1 + x2 + x3"))
  testthat::expect_equal(prepared_identification_count_impl(), before + 1)
  report <- structural_identification(ridge)
  testthat::expect_identical(report, ridge$identification_report)
  testthat::expect_identical(report$status, "unidentified")
  testthat::expect_false(report$directions_at_estimate)
  for (n in c(300L, 400L)) {
    ss <- interior; ss$nobs <- n
    fit <- suppressWarnings(estimate(ridge, prepare_data(ridge, ss), estimator = "ML"))
    testthat::expect_s3_class(fit, "magmaan_fit")
    testthat::expect_identical(fit$diagnostics$identification, report)
    testthat::expect_false(fit$converged)
    testthat::expect_equal(prepared_identification_count_impl(), before + 1)
  }
  testthat::expect_identical(structural_identification("f =~ x1 + x2 + x3")$status,
                            "identified")
  testthat::expect_identical(structural_identification(model_spec(
    "f =~ x1 + a*x2 + b*x3\nb == a*a"))$status, "unchecked")
})

testthat::test_that("ordinal construction uses category schema without prototype values", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  vars <- paste0("x", 1:3)
  d[vars] <- lapply(d[vars], function(x) ordered(cut(x, breaks = quantile(x,
    c(0, 1/3, 2/3, 1)), include.lowest = TRUE, labels = FALSE), levels = 1:3))
  spec <- model_spec("f =~ x1 + x2 + x3", ordered = vars)
  m <- prepare_model(spec, prototype = d[0, vars])
  report <- structural_identification(m)
  expect_identical(report$status, "identified")
  expect_identical(report$map, "ordinal")
  expect_identical(structural_identification(spec, prototype = d), report)
  fit <- estimate(m, prepare_data(m, d), estimator = "DWLS")
  expect_identical(fit$diagnostics$identification, report)
})
