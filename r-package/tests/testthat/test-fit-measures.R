test_that("FIML fit_measures uses the FIML-specific bridge", {
  calls <- new.env(parent = emptyenv())
  calls$robust <- NULL

  local_mocked_bindings(
    fiml_fit_measures_impl = function(fit, robust = FALSE) {
      calls$robust <- robust
      list(
        chisq = 2, df = 1L, pvalue = 0.16,
        baseline.chisq = 10, baseline.df = 3L,
        cfi = 0.9, rmsea = 0.1, ntotal = 50)
    },
    fit_sample_stats = function(...) {
      stop("wrong complete-data fit-measure path")
    },
    measures_fit = function(...) {
      stop("wrong complete-data fit-measure path")
    },
    .package = "magmaan"
  )

  raw <- structure(list(), class = "magmaan_fiml_data")
  fit <- list(estimator = "FIML", fiml = TRUE, raw_data = raw)

  res <- fit_measures(fit, robust = "MLR")

  expect_true(calls$robust)
  expect_equal(res$cfi, 0.9)
  expect_equal(res$baseline.chisq, 10)
})

test_that("FIML fit_measures exposes robust baseline CFI and RMSEA fields", {
  set.seed(11)
  n <- 120L
  eta <- rnorm(n)
  dat <- data.frame(
    x1 = 0.9 * eta + rnorm(n, sd = 0.35),
    x2 = 0.8 * eta + rnorm(n, sd = 0.40),
    x3 = 0.7 * eta + rnorm(n, sd = 0.45),
    x4 = 0.6 * eta + rnorm(n, sd = 0.50)
  )
  model <- "f =~ x1 + x2 + x3 + x4"

  fit_fiml <- magmaan(model, dat, estimator = "FIML")
  fit_ml <- magmaan(model, dat, estimator = "ML")
  fm <- fit_measures(fit_fiml, robust = TRUE)
  fm_ml <- fit_measures(fit_ml)

  ordinary <- c("chisq", "df", "baseline.chisq", "baseline.df",
                "cfi", "tli", "rmsea")
  robust <- c("chisq.scaled", "baseline.chisq.scaled",
              "baseline.chisq.scaling.factor", "cfi.robust",
              "tli.robust", "rmsea.robust")
  expect_true(all(c(ordinary, robust) %in% names(fm)))
  expect_true(all(is.finite(unlist(fm[ordinary]))))
  expect_true(all(is.finite(unlist(fm[robust]))))
  expect_gte(fm$cfi, 0)
  expect_lte(fm$cfi, 1)
  expect_gte(fm$cfi.robust, 0)
  expect_lte(fm$cfi.robust, 1)
  expect_gte(fm$rmsea, 0)
  expect_gte(fm$rmsea.robust, 0)

  same <- c("chisq", "df", "baseline.chisq", "baseline.df", "cfi", "rmsea")
  expect_equal(unlist(fm[same]), unlist(fm_ml[same]), tolerance = 1e-8)
})
