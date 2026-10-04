test_that("refit_from_null replays ML, FIML and ordinal DWLS fits", {
  skip_if_not_installed("lavaan")
  original <- lavaan::HolzingerSwineford1939[paste0("x", 1:4)]
  for (estimator in c("ML", "FIML", "DWLS")) {
    data <- original
    if (estimator == "FIML") data$x2[seq(1, nrow(data), 9)] <- NA_real_
    ordered <- if (estimator == "DWLS") names(data) else NULL
    if (estimator == "DWLS") data[] <- lapply(data, function(x)
      ordered(cut(x, quantile(x, c(0, .33, .67, 1)), include.lowest = TRUE)))
    null <- fit_model(model_spec("f =~ x1 + x2 + x3 + x4", ordered = ordered),
      data, estimator = estimator)
    # The larger model deliberately has covariance-inadmissible endpoints.
    alternative <- suppressWarnings(fit_model(model_spec("f =~ x1 + x2 + x3 + x4\nx1 ~~ x2", ordered = ordered),
      data, estimator = estimator))
    before <- list(alternative$theta, null$theta, alternative$fmin, null$fmin)
    # Explicit refits keep ordinary fit warnings; inspect the result below.
    if (estimator == "ML") {
      expect_warning(retry <- refit_from_null(alternative, null), "covariance-admissible")
    } else {
      retry <- suppressWarnings(refit_from_null(alternative, null))
    }
    expect_s3_class(retry, "magmaan_fit")
    expect_true(isTRUE(retry$converged))
    expect_identical(retry$diagnostics$verdict$status, "passed")
    expect_lte(retry$fmin, null$fmin + 1e-8)
    expect_identical(retry$estimator, alternative$estimator)
    expect_identical(retry$raw_data, alternative$raw_data)
    if (estimator == "DWLS") expect_identical(retry$ordinal_stats, alternative$ordinal_stats)
    expect_identical(list(alternative$theta, null$theta, alternative$fmin, null$fmin), before)
    expect_error(refit_from_null(null, alternative), "nested|embedding|restriction|nesting")
  }
})
