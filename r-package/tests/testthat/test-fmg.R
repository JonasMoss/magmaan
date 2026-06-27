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
