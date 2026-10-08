test_that("quadratic calibration grammar dispatches every name to native FMG", {
  methods <- c("std", "sb", "ss", "mv", "scaled_f", "all", "pall",
               "peba1", "peba2", "peba4", "peba6", "eba2", "eba4")
  eigenvalues <- c(0, .2, .5, 1.7, 3)
  object <- quadratic_reference(8, length(eigenvalues), eigenvalues)
  out <- calibrate_quadratic(object, toupper(methods))
  expect_identical(out$method, methods)
  for (i in seq_along(methods)) {
    method <- methods[i]
    block <- startsWith(method, "peba") || startsWith(method, "eba")
    native <- magmaan_core$robust_fmg_test(8, 5L, eigenvalues,
      if (method == "std") "standard" else if (block) sub("[0-9]+$", "", method) else method,
      if (block) as.numeric(sub("^p?eba", "", method)) else 0)
    expect_identical(out$p_value[i], native$p_value)
  }
  # Blocks include zeros: exceeding the positive rank alone need not saturate.
  for (k in c(4L, 5L, 6L, 100L)) {
    for (family in c("eba", "peba")) {
      native <- magmaan_core$robust_fmg_test(8, 5L, eigenvalues, family, k)
      expect_identical(calibrate_quadratic(object, paste0(family, k))$p_value, native$p_value)
      expect_equal(native$blocks_effective, ceiling(5 / ceiling(5/k)))
    }
  }
  expect_identical(calibrate_quadratic(object, "eba6")$p_value,
                   calibrate_quadratic(object, "all")$p_value)
  expect_identical(calibrate_quadratic(object, "peba6")$p_value,
                   calibrate_quadratic(object, "pall")$p_value)
  for (bad in list(NULL, character(), NA_character_, 2L, "peba0", "eba01", "peba2.0", "pOLS", "std\n"))
    expect_error(calibrate_quadratic(object, bad), "integer k >= 1", fixed = TRUE)
})

test_that("projected-score calibration obtains spectra for all new laws", {
  projected <- score_quadratic(c(2, 1), diag(2), diag(c(1, 3)))
  explicit <- score_spectrum(projected)
  methods <- c("std", "sb", "ss", "mv", "scaled_f", "pall", "eba2", "peba6")
  expect_identical(calibrate_quadratic(projected, methods)$p_value,
                   calibrate_quadratic(explicit, methods)$p_value)
})

test_that("explicit spectra share projected-score roundoff validation", {
  values <- c(-6e-15, .2, .5, 1.7, 3)
  zero <- replace(values, 1, 0)
  actual <- quadratic_reference(8, 5, values)
  expect_equal(actual$eigenvalues, zero, tolerance = 0)
  expect_identical(calibrate_quadratic(actual, c("sb", "peba4")),
                   calibrate_quadratic(quadratic_reference(8, 5, zero), c("sb", "peba4")))
  projected <- score_quadratic(rep(1, 5), diag(5), diag(values))
  expect_equal(score_spectrum(projected)$eigenvalues, zero, tolerance = 0)
  expect_error(quadratic_reference(8, 5, replace(values, 1, -.01)),
               "[NumericIssue]: quadratic spectrum: eigenvalues are not positive semidefinite", fixed = TRUE)
  expect_error(score_spectrum(score_quadratic(rep(1, 5), diag(5),
               diag(replace(values, 1, -.01)))), "[NumericIssue]: quadratic spectrum: eigenvalues are not positive semidefinite", fixed = TRUE)
})
