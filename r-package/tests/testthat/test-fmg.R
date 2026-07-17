test_that("ML2S weighted fits use the cached ML2S FMG spectrum", {
  calls <- new.env(parent = emptyenv())
  calls$fmg <- NULL

  local_mocked_bindings(
    infer_fiml_fmg_spectrum = function(...) {
      stop("wrong FIML branch")
    },
    estimate_two_stage_em_ml_inference = function(...) {
      stop("cached ML2S spectrum was not used")
    },
    infer_fmg_test = function(chi2_source, df, eigvals, method = "peba",
                              param = 4.0, truncate_negative = TRUE) {
      calls$fmg <- list(
        chi2_source = chi2_source, df = df, eigvals = eigvals,
        method = method, param = param,
        truncate_negative = truncate_negative)
      list(
        p_value = 0.42, df = df, chi2_source = chi2_source,
        method = method, param = param, chi2_equiv = chi2_source,
        n_truncated = 0L, lambdas_raw = eigvals, lambdas = eigvals,
        lambdas_reference = eigvals)
    },
    .package = "magmaan"
  )

  raw <- structure(list(), class = "magmaan_fiml_data")
  fit <- list(
    estimator = "ML2S_DWLS",
    raw_data = raw,
    ml2s = list(eigvals = c(0.5, 1.25), chisq = 7, df = 2L)
  )

  res <- fmg_tests(fit, tests = "SB")

  expect_equal(calls$fmg$chi2_source, 7)
  expect_equal(calls$fmg$df, 2L)
  expect_equal(calls$fmg$eigvals, c(0.5, 1.25))
  expect_equal(calls$fmg$method, "sb")
  expect_equal(res$label, "sb_ml")
  expect_equal(res$base, "ml")
  expect_equal(res$eigenvalues[[1]], c(0.5, 1.25))
})

test_that("continuous least-squares fits use the C++ robust spectrum", {
  calls <- new.env(parent = emptyenv())
  calls$weight <- NULL

  local_mocked_bindings(
    infer_continuous_ls_robust = function(fit, raw_data, weight = NULL,
                                          bread = "expected",
                                          gamma = "empirical") {
      calls$weight <- weight
      calls$gamma <- gamma
      list(df = 2L, eigvals = c(0.75, 1.25), chisq_standard = 6)
    },
    infer_fmg_test = function(chi2_source, df, eigvals, method = "peba",
                              param = 4.0, truncate_negative = TRUE) {
      list(
        p_value = 0.31, df = df, chi2_source = chi2_source,
        method = method, param = param, chi2_equiv = chi2_source,
        n_truncated = 0L, lambdas_raw = eigvals, lambdas = eigvals,
        lambdas_reference = eigvals
      )
    },
    .package = "magmaan"
  )

  fit <- list(
    estimator = "WLS",
    raw_data = matrix(seq_len(6), nrow = 3),
    nobs = 3L
  )
  W <- diag(2)
  res <- fmg_tests(fit, tests = "SB", weight = W, gamma = "normal")

  expect_equal(calls$weight, W)
  expect_equal(calls$gamma, "normal")
  expect_equal(res$label, "sb_ls")
  expect_equal(res$base, "ls")
  expect_equal(res$base_statistic, 6)
  expect_equal(res$eigenvalues[[1]], c(0.75, 1.25))
  expect_error(fmg_tests(fit, tests = "SB_UG", weight = W),
               "only for complete-data normal-theory ML")
  expect_error(fmg_tests(fit, tests = "SB_RLS", weight = W),
               "has no RLS")
})

test_that("mixed-ordinal nested FMG composes the existing nested spectrum", {
  calls <- new.env(parent = emptyenv())
  calls$data <- NULL

  local_mocked_bindings(
    robust_nested_lrt = function(fit_H1, fit_H0, data = NULL, ...) {
      calls$data <- data
      list(T_diff = 4.5, df_diff = 2L, eigenvalues = c(0.8, 1.2))
    },
    infer_fmg_test = function(chi2_source, df, eigvals, method = "peba",
                              param = 4.0, truncate_negative = TRUE) {
      list(
        p_value = 0.27, df = df, chi2_source = chi2_source,
        method = method, param = param, chi2_equiv = chi2_source,
        n_truncated = 0L, lambdas_raw = eigvals, lambdas = eigvals,
        lambdas_reference = eigvals
      )
    },
    .package = "magmaan"
  )

  stats <- structure(list(marker = TRUE),
                     class = "magmaan_mixed_ordinal_data")
  res <- fmg_nested_mixed_ordinal(
    list(estimator = "DWLS"), list(estimator = "DWLS"), stats,
    tests = "pEBA2", weight = "DWLS"
  )

  expect_identical(calls$data, stats)
  expect_equal(res$label, "peba2_ls")
  expect_equal(res$base_statistic, 4.5)
  expect_equal(res$eigenvalues[[1]], c(0.8, 1.2))
})

test_that("generic nested FMG uses the continuous-LS restriction spectrum", {
  calls <- new.env(parent = emptyenv())
  calls$gamma <- NULL
  local_mocked_bindings(
    robust_nested_lrt = function(fit_H1, fit_H0, data = NULL, gamma, ...) {
      calls$gamma <- gamma
      list(T_diff = 5, df_diff = 2L, eigenvalues = c(0.7, 1.3))
    },
    infer_fmg_test = function(chi2_source, df, eigvals, method = "peba",
                              param = 4.0, truncate_negative = TRUE) {
      list(
        p_value = 0.24, df = df, chi2_source = chi2_source,
        method = method, param = param, chi2_equiv = chi2_source,
        n_truncated = 0L, lambdas_raw = eigvals, lambdas = eigvals,
        lambdas_reference = eigvals
      )
    },
    .package = "magmaan"
  )
  fit <- list(estimator = "ULS")
  res <- fmg_nested(fit, fit, data = matrix(1:8, ncol = 2), tests = "SB")
  expect_equal(calls$gamma, "NT")
  expect_equal(res$label, "sb_ls")
  expect_equal(res$base, "ls")
  expect_equal(res$eigenvalues[[1]], c(0.7, 1.3))
})

test_that("nested ML FMG selects biased and unbiased spectra per test", {
  calls <- character()
  local_mocked_bindings(
    robust_nested_lrt = function(fit_H1, fit_H0, data = NULL, gamma, ...) {
      calls <<- c(calls, gamma)
      list(
        T_diff = 6, df_diff = 2L,
        eigenvalues = if (gamma == "unbiased") c(0.8, 1.4) else c(0.7, 1.2),
        eigenvalues_unbiased = if (gamma == "both") c(0.8, 1.4) else numeric()
      )
    },
    infer_fmg_test = function(chi2_source, df, eigvals, method = "peba",
                              param = 4.0, truncate_negative = TRUE) {
      list(
        p_value = sum(eigvals) / 10, df = df, chi2_source = chi2_source,
        method = method, param = param, chi2_equiv = chi2_source,
        n_truncated = 0L, lambdas_raw = eigvals, lambdas = eigvals,
        lambdas_reference = eigvals
      )
    },
    .package = "magmaan"
  )
  fit <- list(estimator = "ML")
  res <- fmg_nested(
    fit, fit, data = matrix(1:8, ncol = 2),
    tests = c("SB", "SB_UG")
  )
  expect_equal(calls, "both")
  expect_equal(res$label, c("sb_ml", "sb_ug_ml"))
  expect_equal(res$ug, c(FALSE, TRUE))
  expect_equal(res$eigenvalues[[1]], c(0.7, 1.2))
  expect_equal(res$eigenvalues[[2]], c(0.8, 1.4))
})
