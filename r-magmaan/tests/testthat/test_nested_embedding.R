test_that("anova accepts omitted, fixed-zero and equality-written null paths", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  three <- paste("visual =~ x1 + x2 + x3", "textual =~ x4 + x5 + x6",
                 "speed =~ x7 + x8 + x9", sep = "\n")
  h1 <- magmaan(paste(three, "visual =~ x9", sep = "\n"), d)
  nulls <- lapply(c(three, paste(three, "visual =~ 0*x9", sep = "\n"),
                   paste(three, "visual =~ a*x9\na == 0", sep = "\n")), magmaan, data = d)
  results <- lapply(nulls, function(h0) anova(h1, h0))
  for (x in results) {
    expect_s3_class(x, "magmaan_anova")
    expect_length(attr(x, "unavailable"), 0L)
    expect_equal(x$statistic, results[[3L]]$statistic, tolerance = 1e-5)
    expect_equal(x$pvalue[x$reference %in% "sb"], results[[3L]]$pvalue[results[[3L]]$reference %in% "sb"], tolerance = 1e-6)
    expect_equal(x$pvalue[x$reference %in% "peba4"], results[[3L]]$pvalue[results[[3L]]$reference %in% "peba4"], tolerance = 1e-6)
  }
  expect_equal(anova(nulls[[1L]], h1)$statistic, results[[1L]]$statistic)
})
