vcov_syntax <- "visual =~ x1+x2+x3\ntextual =~ x4+x5+x6"

test_that("ML covariance regimes identify information and sandwich formulas", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  fit <- fit_model(vcov_syntax, d)
  for (bread in c("expected", "observed")) {
    regime <- paste0("sandwich_", bread)
    explicit <- magmaan_core$robust_se_raw_fit(fit, fit$raw_data$X, bread = bread)$vcov
    expect_equal(vcov(fit, regime), explicit, tolerance = 1e-12)
    expect_equal(vcov(fit, regime), vcov(fit, regime, data = d), tolerance = 1e-12)
    oracle <- lavaan::cfa(vcov_syntax, d, information = bread)
    expect_equal(unname(vcov(fit, paste0("information_", bread))),
                 unname(unclass(lavaan::lavInspect(oracle, "vcov"))), tolerance = 1e-5)
  }
  expect_equal(vcov(fit), vcov(fit, "sandwich_observed"))
  expect_equal(vcov(fit, "model"), vcov(fit, "sandwich_expected"))
  expect_equal(vcov(fit, "robust"), vcov(fit, "sandwich_observed"))
  expect_gt(max(abs(vcov(fit, "information_expected") - vcov(fit, "sandwich_expected"))), 1e-5)
  expect_error(vcov(fit, "stored"), "not supported")
  expect_error(vcov(fit, data = d[setdiff(names(d), "x1")]), "vcov\\(\\):.*missing observed")
  retained <- vcov(fit)
  fit$raw_data <- NULL
  expect_error(vcov(fit), "supply `data`")
  expect_equal(vcov(fit, data = d), retained)
  expect_true(is.matrix(vcov(fit, "information_expected")))
})

test_that("grouped retained observations preserve covariance ordering", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  fit <- fit_model(vcov_syntax, d, groups = "school", group_equal = "loadings")
  for (regime in c("sandwich_expected", "sandwich_observed")) {
    expect_equal(vcov(fit, regime), vcov(fit, regime, d), tolerance = 1e-12)
  }
  expect_error(vcov(fit, data = d[setdiff(names(d), "school")]), "vcov\\(\\): grouped")
})

test_that("FIML explicit regimes reuse matching information conventions", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  d$x2[seq(1L, nrow(d), 7L)] <- NA_real_
  fit <- fit_model(vcov_syntax, d, estimator = "FIML")
  refs <- magmaan_core$inference_fiml_information_vcov(fit)
  for (bread in c("expected", "observed")) {
    ref <- refs[[if (bread == "expected") "expected" else "observed_hessian"]]
    expect_true(ref$ok)
    expect_equal(vcov(fit, paste0("information_", bread)), ref$vcov_model, tolerance = 1e-10)
    expect_equal(vcov(fit, paste0("sandwich_", bread)), ref$vcov_sandwich, tolerance = 1e-10)
  }
  expect_equal(vcov(fit), vcov(fit, "sandwich_observed"))
  expect_equal(vcov(fit, "model"), vcov(fit, "information_observed"))
  expect_equal(vcov(fit, "robust"), vcov(fit, "sandwich_observed"))
  expect_error(vcov(fit, "delta_nt"), "not supported")
  expect_error(vcov(fit, data = d), "retained data")
})

test_that("categorical and closed-form fits do not silently substitute regimes", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  x <- as.matrix(d[paste0("x", 1:6)])
  ss <- magmaan_core$data_sample_stats_from_raw(x)
  fit <- fit_noniterative_cfa(model_spec(vcov_syntax)$partable, ss,
                            estimator = "guttman_aligned", composite = "standardized")
  expect_equal(vcov(fit, "delta_nt"), noniterative_cfa_se(fit, gamma = "nt")$vcov)
  expect_error(vcov(fit, "delta_empirical"), "raw observations")
  fit$raw_data <- list(X = list(x))
  expect_equal(vcov(fit, "delta_empirical"), vcov(fit, "delta_empirical", x))
  expect_error(vcov(fit, "information_expected"), "not supported")
  for (v in paste0("x", 1:6)) d[[v]] <- as.integer(cut(d[[v]], 3))
  ordinal <- fit_model(vcov_syntax, d, estimator = "DWLS", ordered = paste0("x", 1:6))
  expect_equal(vcov(ordinal), vcov(ordinal, "sandwich_ij"))
  expect_equal(vcov(ordinal, "robust"), vcov(ordinal, "sandwich_ij"))
  expect_error(vcov(ordinal, "information_observed"), "not supported")
  sam <- structure(list(vcov = diag(2)), class = c("magmaan_sam_fit", "magmaan_fit"))
  expect_equal(vcov(sam, "stored"), diag(2))
  expect_error(vcov(sam, "sandwich_observed"), "not supported")
})
