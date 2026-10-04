test_that("policy_nested routes Wu-Estabrook moment-nested threshold steps", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  ord <- paste0("x", 1:6)
  syntax <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"
  for (categories in c(3L, 5L)) {
    input <- d
    for (v in ord) input[[v]] <- ordered(cut(input[[v]],
      quantile(input[[v]], seq(0, 1, length.out = categories + 1L)),
      include.lowest = TRUE, labels = FALSE), levels = seq_len(categories))
    spec <- function(equal = NULL) model_spec(syntax, ordered = ord,
      parameterization = "theta", group = "school",
      group_labels = levels(input$school), group_equal = equal)
    h1 <- fit_model(spec(), input, estimator = "DWLS")
    h0 <- fit_model(spec("thresholds"), input, estimator = "DWLS")
    expect_true(h1$converged); expect_true(h0$converged)
    out <- policy_nested(h1, h0)$lr
    if (categories == 3L) {
      expect_identical(out$reason, "equivalent_models")
    } else {
      expect_true(out$available, info = out$detail)
      expect_equal(out$df, 12L)
      expect_length(out$eigenvalues, 12L)
      expect_true(all(is.finite(out$eigenvalues)))
      expect_equal(out$statistic, 2 * h1$ntotal * (h0$fmin - h1$fmin), tolerance = 1e-8)
      expect_identical(policy_nested(h0, h1)$lr$reason, "not_nested")
    }
  }
})
