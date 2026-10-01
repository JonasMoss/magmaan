test_that("ordinary advanced fitting choices use the shared engine", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  m <- "f =~ x1+x2+x3+x4"
  fit <- magmaan(m, d, inference = FALSE,
      options = list(preset = "lavaan-0.7.2"))
  lab <- as_lab_fit(fit)
  expect_equal(lab$fitting$effective$convergence, "lavaan-0.7.2")
  expect_equal(lab$verdict$policy, "lavaan-0.7.2")
  hybrid <- magmaan(m, d, inference = FALSE,
      options = list(preset = "lavaan-0.7.2", convergence = "newton"))
  expect_true(as_lab_fit(hybrid)$fitting$modified_preset)
  expect_equal(as_lab_fit(hybrid)$verdict$status, as_lab_fit(hybrid)$diagnostics$verdict$status)
  expect_error(magmaan(m, d, inference = FALSE, start = "fabin3",
      options = list(preset = "lavaan-0.7.2")), "constructor conflicts")
  expect_error(magmaan(m, d, inference = FALSE, psd = TRUE,
      options = list(preset = "lavaan-0.7.2")), "ordinary complete")
})
