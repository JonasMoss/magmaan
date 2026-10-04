test_that("anova routes Wu-Estabrook moment-nested threshold steps", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  ord <- paste0("x", 1:6)
  syntax <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"
  for (categories in c(3L, 5L)) {
    input <- d
    for (v in ord) input[[v]] <- ordered(cut(input[[v]],
      quantile(input[[v]], seq(0, 1, length.out = categories + 1L)),
      include.lowest = TRUE, labels = FALSE), levels = seq_len(categories))
    spec <- function(equal = NULL) magmaan_model(syntax, prototype = input, ordered = ord,
      parameterization = "theta", group = "school",
      group.equal = equal)
    h1 <- magmaan(spec(), input, estimator = "DWLS")
    h0 <- magmaan(spec("thresholds"), input, estimator = "DWLS")
    out <- anova(h1, h0)
    if (categories == 3L) {
      expect_match(attr(out, "unavailable")[["lr"]], "equivalent_models")
    } else {
      expect_equal(out$df[2L], 12L)
      expect_true(is.finite(out$statistic[2L]))
      expect_true(is.finite(out$p.sb[2L]))
      expect_equal(anova(h0, h1)$statistic, out$statistic)
    }
  }
})
