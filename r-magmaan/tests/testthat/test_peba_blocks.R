test_that("nested PEBA4 reports formed blocks without adding a column", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  vars <- paste0("x", 1:5)
  variances <- paste(vars, "~~", vars)
  pairs <- combn(vars, 2, function(x) paste(x, collapse = " ~~ "))
  alternative <- magmaan(paste(c(variances, pairs), collapse = "\n"), d)
  for (df in c(1L, 4L, 5L, 8L)) {
    null <- magmaan(paste(c(variances, pairs[-seq_len(df)]), collapse = "\n"), d)
    result <- anova(alternative, null)
    blocks <- if (df == 5L) 3L else min(df, 4L)
    expect_equal(result$df, rep(df, 4))
    expect_equal(unname(attr(result, "peba_blocks")), rep(blocks, 4))
    expect_false("peba_blocks" %in% names(result))
    printed <- capture.output(print(result))
    expect_equal(sum(grepl("PEBA4 formed", printed)), as.integer(blocks < 4L))
    if (blocks < 4L) expect_match(paste(printed, collapse = "\n"),
      paste0("PEBA4 formed ", blocks, " eigenvalue blocks (df = ", df, ")"), fixed = TRUE)
    global <- summary(null)$tests
    expect_equal(unname(attr(global, "peba_blocks")), rep(blocks, nrow(global)))
    expect_equal(sum(grepl("PEBA4 formed", capture.output(print(summary(null))))),
                 as.integer(blocks < 4L))
  }
})

test_that("lab FMG and unavailable policy expose block diagnostics", {
  for (df in c(1L, 2L, 3L, 4L, 5L, 6L, 8L)) {
    out <- magmaanlab:::infer_fmg_test(6, df, seq_len(df), "peba", 4)
    expect_equal(out$blocks_effective, ceiling(df / ceiling(df / 4)))
  }
  table <- magmaanlab:::.fmg_result_rows_ordinal(
    list(chisq_standard = 6, df = 5L, eigvals = 1:5),
    list(magmaanlab:::.fmg_parse_test("PEBA4")))
  expect_equal(table$blocks_effective, 3L)
  expect_identical(magmaanlab:::.policy_unavailable("unsupported_model", "test")$score$peba_blocks, 0L)
})
