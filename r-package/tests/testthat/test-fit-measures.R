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

test_that("non-iterative fit measures use empirical baseline scaling and no IC", {
  skip_if_not_installed("lavaan")

  hs <- lavaan::HolzingerSwineford1939
  vars <- paste0("x", 1:9)
  X <- as.matrix(hs[vars])
  model <- "visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9"

  pt <- magmaan_core$lavaan_lavaanify(model)
  ss <- magmaan_core$data_sample_stats_from_raw(X)
  fit <- fit_noniterative_cfa(pt, ss, estimator = "guttman_gls_aligned",
                              composite = "standardized")

  fm <- fit_measures_noniterative(fit, discrepancy = "uls",
                                  gamma = "empirical", data = X)

  S <- fit$S[[1L]]
  ut <- upper.tri(S)
  Xc <- scale(X, center = TRUE, scale = FALSE)
  ii <- row(S)[ut]
  jj <- col(S)[ut]
  gdiag <- vapply(seq_along(ii), function(k) {
    cp <- Xc[, ii[k]] * Xc[, jj[k]]
    mean((cp - mean(cp))^2)
  }, numeric(1))
  c0_hand <- sum(2 * gdiag) / sum(ut)

  expect_equal(fm$baseline.scale.c, c0_hand, tolerance = 1e-12)
  expect_true(all(is.na(unlist(
    fm[c("logl", "unrestricted.logl", "aic", "bic", "bic2")]))))
  expect_true(is.finite(fm$srmr))
  expect_true(is.finite(fm$cfi.robust))
})

test_that("non-iterative fit measures use grouped inference and baselines", {
  skip_if_not_installed("lavaan")

  hs <- lavaan::HolzingerSwineford1939
  vars <- paste0("x", 1:9)
  dat <- hs[c("school", vars)]
  labels <- unique(as.character(dat$school))
  model <- "visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9"
  spec <- model_spec(model, group = "school", group_labels = labels)
  ss <- df_to_data(dat, spec, group = "school", missing = "error")
  fit <- fit_noniterative_cfa(spec$partable, ss,
                              estimator = "guttman_gls_aligned",
                              composite = "standardized")

  fm <- fit_measures_noniterative(fit, discrepancy = "uls",
                                  gamma = "empirical", data = ss)
  inf <- noniterative_cfa_grouped_inference(fit, discrepancy = "uls",
                                            gamma = "empirical", data = ss)

  baseline_chisq <- 0
  baseline_df <- 0L
  baseline_scale_num <- 0
  for (b in seq_along(ss$S)) {
    S <- ss$S[[b]]
    X <- ss$X[[b]]
    ut <- upper.tri(S)
    baseline_chisq <- baseline_chisq +
      ss$nobs[[b]] * 2 * sum(S[ut]^2)
    baseline_df <- baseline_df + sum(ut)

    Xc <- scale(X, center = TRUE, scale = FALSE)
    ii <- row(S)[ut]
    jj <- col(S)[ut]
    gdiag <- vapply(seq_along(ii), function(k) {
      cp <- Xc[, ii[k]] * Xc[, jj[k]]
      mean((cp - mean(cp))^2)
    }, numeric(1))
    baseline_scale_num <- baseline_scale_num + sum(2 * gdiag)
  }

  expect_equal(fm$chisq, inf$T, tolerance = 1e-10)
  expect_equal(fm$df, inf$df)
  expect_equal(fm$baseline.chisq, baseline_chisq, tolerance = 1e-10)
  expect_equal(fm$baseline.df, baseline_df)
  expect_equal(fm$baseline.scale.c, baseline_scale_num / baseline_df,
               tolerance = 1e-12)
  expect_true(is.finite(fm$cfi.robust))
  expect_true(is.finite(fm$rmsea.robust))
})
