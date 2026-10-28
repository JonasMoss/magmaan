# Pairwise-deletion ordinal stats carry their Gamma convention (pd_gamma) and
# moment supports as attributes. Every C++ inference route must read them back;
# before TASK-167 the robust routes saw missing category codes without the
# overlap convention and refused.
pairwise_ordinal_case <- function() {
  d <- lavaan::HolzingerSwineford1939[, paste0("x", 1:6)]
  for (v in names(d)) {
    d[[v]] <- ordered(cut(d[[v]], c(-Inf, stats::quantile(d[[v]], c(.25, .5, .75)), Inf)))
  }
  set.seed(167)
  for (v in names(d)) d[[v]][sample.int(nrow(d), 30)] <- NA
  d
}

test_that("pairwise ordinal stats keep pd_gamma through robust and nested inference", {
  skip_if_not_installed("lavaan")
  d <- pairwise_ordinal_case()
  ord <- names(d)
  h1 <- "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6"
  h0 <- paste0(h1, "\nf1 ~~ 0.5*f2")
  spec1 <- model_spec(h1, ordered = ord)
  spec0 <- model_spec(h0, ordered = ord)
  st <- magmaan_core$data_ordinal_stats_from_df(
    d, spec1, ordered = ord, missing = "pairwise", pd_gamma = "overlap",
    full_wls_weight = FALSE)
  expect_identical(attr(st, "pd_gamma"), "overlap")
  f1 <- fit_model(spec1, st, estimator = "DWLS")
  f0 <- fit_model(spec0, st, estimator = "DWLS")

  single <- fmg_tests_ordinal(f1, st, tests = c("SB", "SS"), weight = "DWLS")
  expect_true(all(is.finite(single$p_value)))
  nested <- fmg_nested_ordinal(f1, f0, st, tests = "SB", weight = "DWLS")
  expect_true(all(is.finite(nested$p_value)))

  # lavaan 0.7-2's pairwise UGamma uses the same overlap Gamma.
  oracle <- lavaan_categorical_fit(lavaan::cfa, h1, data = d, ordered = ord,
                                   estimator = "DWLS", missing = "pairwise")
  df <- as.integer(lavaan::fitMeasures(oracle, "df"))
  ug <- sort(Re(eigen(lavaan::lavInspect(oracle, "UGamma"),
                      only.values = TRUE)$values), decreasing = TRUE)[seq_len(df)]
  ours <- sort(single$eigenvalues[[1L]], decreasing = TRUE)
  expect_equal(ours, ug, tolerance = 1e-5)
})
