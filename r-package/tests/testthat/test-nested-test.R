test_that("ML2S weighted fits use the ML2S nested-test branch", {
  calls <- new.env(parent = emptyenv())
  calls$ml2s <- NULL

  local_mocked_bindings(
    infer_ml2s_lr_test_satorra2000 = function(
        fit_H1, fit_H0, gamma = "empirical", a_method = "exact",
        h_step = 1e-4, ud_method = "2000", stage2_weight = "nt",
        dls_a = 0.5) {
      calls$ml2s <- list(
        gamma = gamma, a_method = a_method, ud_method = ud_method,
        stage2_weight = stage2_weight, dls_a = dls_a)
      list(statistic = 1, df = 1, p_value = 0.3)
    },
    infer_fiml_lr_test_satorra2000 = function(...) {
      stop("wrong FIML branch")
    },
    .package = "magmaan"
  )

  raw <- structure(list(), class = "magmaan_fiml_data")
  h1 <- list(estimator = "ML2S_DWLS", raw_data = raw,
             stage2_weight = "dwls", stage2_dls_a = 0.5)
  h0 <- list(estimator = "ML2S_DWLS", raw_data = raw,
             stage2_weight = "dwls", stage2_dls_a = 0.5)

  res <- robust_nested_lrt(h1, h0, method = "restriction_map")

  expect_equal(calls$ml2s$stage2_weight, "dwls")
  expect_equal(calls$ml2s$dls_a, 0.5)
  expect_equal(res$stage2_weight, "dwls")
  expect_equal(res$computation, "ml2s_eta")
})

test_that("ML2S stage2 weight can be inferred from estimator labels", {
  calls <- new.env(parent = emptyenv())
  calls$stage2_weight <- NULL

  local_mocked_bindings(
    infer_ml2s_lr_test_satorra2000 = function(
        fit_H1, fit_H0, gamma = "empirical", a_method = "exact",
        h_step = 1e-4, ud_method = "2000", stage2_weight = "nt",
        dls_a = 0.5) {
      calls$stage2_weight <- stage2_weight
      list(statistic = 1, df = 1, p_value = 0.3)
    },
    infer_fiml_lr_test_satorra2000 = function(...) {
      stop("wrong FIML branch")
    },
    .package = "magmaan"
  )

  raw <- structure(list(), class = "magmaan_fiml_data")
  h1 <- list(estimator = "ML2S_ULS", raw_data = raw)
  h0 <- list(estimator = "ML2S_ULS", raw_data = raw)

  res <- robust_nested_lrt(h1, h0)

  expect_equal(calls$stage2_weight, "uls")
  expect_equal(res$stage2_weight, "uls")
})

test_that("ML2S nested tests reject mismatched stage2 weights", {
  raw <- structure(list(), class = "magmaan_fiml_data")
  h1 <- list(estimator = "ML2S_DWLS", raw_data = raw, stage2_weight = "dwls")
  h0 <- list(estimator = "ML2S", raw_data = raw, stage2_weight = "nt")

  expect_error(
    robust_nested_lrt(h1, h0),
    "same stage2_weight"
  )
})

test_that("non-NT ML2S scalar SB methods are rejected", {
  raw <- structure(list(), class = "magmaan_fiml_data")
  h1 <- list(estimator = "ML2S_DWLS", raw_data = raw, stage2_weight = "dwls")
  h0 <- list(estimator = "ML2S_DWLS", raw_data = raw, stage2_weight = "dwls")

  expect_error(
    robust_nested_lrt(h1, h0, method = "lavaan_sb2001"),
    "NT-only"
  )
})
