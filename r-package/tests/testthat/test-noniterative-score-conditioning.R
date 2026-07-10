test_that("non-iterative score conditioning is recorded and reused", {
  set.seed(20260710)
  Lambda <- matrix(0, 6, 2)
  Lambda[1:3, 1] <- c(1, 0.8, 1.2)
  Lambda[4:6, 2] <- c(1, 0.7, 1.3)
  Phi <- matrix(c(1, 0.3, 0.3, 1), 2, 2)
  Sigma <- Lambda %*% Phi %*% t(Lambda) + diag(0.5, 6)
  X <- matrix(rnorm(400L * 6L), 400L, 6L) %*% chol(Sigma)
  colnames(X) <- paste0("x", 1:6)
  model <- "f1 =~ x1 + x2 + x3
            f2 =~ x4 + x5 + x6
            f1 ~~ f2"
  pt <- magmaan_core$lavaan_lavaanify(model)
  ss <- magmaan_core$data_sample_stats_from_raw(X)

  fit <- fit_noniterative_cfa(
    pt, ss, estimator = "guttman_aligned", composite = "auto",
    score_conditioning = "hard", score_floor0 = 16, score_rate = 0.5)

  expect_equal(fit$composite, "standardized")
  expect_equal(fit$score_conditioning, "hard")
  expect_equal(fit$score_floor0, 16)
  expect_equal(fit$score_rate, 0.5)
  expect_s3_class(fit$score_conditioning_diagnostics, "data.frame")
  expect_true(all(c(
    "target_floor", "raw_min_eigenvalue", "repaired_min_eigenvalue",
    "raw_normalized_min_eigenvalue", "repaired_normalized_min_eigenvalue",
    "shrinkage", "hard_violation", "min_score_variance",
    "min_abs_marker") %in% names(fit$score_conditioning_diagnostics)))
  expect_gte(
    fit$score_conditioning_diagnostics$repaired_normalized_min_eigenvalue,
    fit$score_conditioning_diagnostics$target_floor - 1e-10)

  se <- noniterative_cfa_se(fit, gamma = "nt")
  expect_true(all(is.finite(se$se)))
})

test_that("older non-iterative fits reconstruct raw score conditioning", {
  set.seed(20260711)
  X <- matrix(rnorm(500L * 6L), 500L, 6L)
  X[, 2] <- 0.8 * X[, 1] + 0.6 * X[, 2]
  X[, 3] <- 0.7 * X[, 1] + 0.7 * X[, 3]
  X[, 5] <- 0.8 * X[, 4] + 0.6 * X[, 5]
  X[, 6] <- 0.7 * X[, 4] + 0.7 * X[, 6]
  colnames(X) <- paste0("x", 1:6)
  pt <- magmaan_core$lavaan_lavaanify(
    "f1 =~ x1 + x2 + x3
     f2 =~ x4 + x5 + x6
     f1 ~~ f2")
  ss <- magmaan_core$data_sample_stats_from_raw(X)
  fit <- fit_noniterative_cfa(
    pt, ss, estimator = "guttman_aligned", composite = "standardized")
  se <- noniterative_cfa_se(fit, gamma = "nt")

  old <- fit
  old$score_conditioning <- NULL
  old$score_floor0 <- NULL
  old$score_rate <- NULL
  old$score_conditioning_diagnostics <- NULL
  old_se <- noniterative_cfa_se(old, gamma = "nt")
  expect_identical(old_se$se, se$se)
  expect_identical(old_se$vcov, se$vcov)
})

test_that("nested pseudo-LRT rejects mismatched score conditioning", {
  set.seed(20260712)
  X <- matrix(rnorm(400L * 6L), 400L, 6L)
  X[, 2] <- X[, 1] + 0.5 * X[, 2]
  X[, 3] <- X[, 1] + 0.5 * X[, 3]
  X[, 5] <- X[, 4] + 0.5 * X[, 5]
  X[, 6] <- X[, 4] + 0.5 * X[, 6]
  colnames(X) <- paste0("x", 1:6)
  pt <- magmaan_core$lavaan_lavaanify(
    "f1 =~ x1 + x2 + x3
     f2 =~ x4 + x5 + x6
     f1 ~~ f2")
  ss <- magmaan_core$data_sample_stats_from_raw(X)
  fit <- fit_noniterative_cfa(
    pt, ss, estimator = "guttman_aligned", composite = "standardized")
  mismatch <- fit
  mismatch$score_conditioning <- "soft"
  mismatch$score_floor0 <- 4
  expect_error(
    noniterative_cfa_pseudo_lrt(fit, mismatch),
    "score conditioning configurations differ")
})

test_that("non-raw score conditioning rejects legacy and adaptive maps", {
  set.seed(20260713)
  X <- matrix(rnorm(300L * 6L), 300L, 6L)
  colnames(X) <- paste0("x", 1:6)
  pt <- magmaan_core$lavaan_lavaanify(
    "f1 =~ x1 + x2 + x3
     f2 =~ x4 + x5 + x6
     f1 ~~ f2")
  ss <- magmaan_core$data_sample_stats_from_raw(X)
  expect_error(
    fit_noniterative_cfa(
      pt, ss, estimator = "guttman_lavaan", score_conditioning = "hard"),
    "supported only for guttman_aligned")
  expect_error(
    fit_noniterative_cfa(
      pt, ss, estimator = "guttman_aligned", composite = "adaptive",
      score_conditioning = "soft"),
    "not supported for adaptive")
})

test_that("fixed-diagonal H repair is recorded as a point-fit feasibility map", {
  set.seed(20260714)
  X <- matrix(rnorm(400L * 6L), 400L, 6L)
  X[, 2] <- 0.8 * X[, 1] + 0.6 * X[, 2]
  X[, 3] <- 0.7 * X[, 1] + 0.7 * X[, 3]
  X[, 5] <- 0.8 * X[, 4] + 0.6 * X[, 5]
  X[, 6] <- 0.7 * X[, 4] + 0.7 * X[, 6]
  colnames(X) <- paste0("x", 1:6)
  pt <- magmaan_core$lavaan_lavaanify(
    "f1 =~ x1 + x2 + x3
     f2 =~ x4 + x5 + x6
     f1 ~~ f2")
  ss <- magmaan_core$data_sample_stats_from_raw(X)

  fit <- fit_noniterative_cfa(
    pt, ss, estimator = "guttman_aligned", composite = "standardized",
    h_conditioning = "hard", h_floor0 = 16, h_rate = 0.5)
  expect_equal(fit$h_conditioning, "hard")
  expect_equal(fit$h_floor0, 16)
  expect_s3_class(fit$h_conditioning_diagnostics, "data.frame")
  expect_gte(
    fit$h_conditioning_diagnostics$repaired_normalized_min_eigenvalue,
    fit$h_conditioning_diagnostics$target_floor - 1e-10)
  expect_error(noniterative_cfa_se(fit, gamma = "nt"),
               "point-estimation feasibility prototype")
})
