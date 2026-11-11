.ml_expected_h1 <- paste("visual =~ x1 + a*x2 + b*x3",
  "textual =~ x4 + x5 + x6", "speed =~ x7 + x8 + x9", sep = "\n")

.ml_expected_pair <- function(d, identification = "marker", grouped = FALSE) {
  h1 <- if (grouped) sub("a*x2 + b*x3", "c(a,c)*x2 + c(b,d)*x3", .ml_expected_h1, fixed = TRUE) else .ml_expected_h1
  h0 <- paste(h1, if (grouped) "a == c" else "a == b", sep = "\n")
  fit <- function(s) magmaan(magmaan_model(s, prototype = d,
    identification = identification, group = if (grouped) "school" else NULL), d)
  list(null = fit(h0), alternative = fit(h1))
}

test_that("ML nested policy uses explicit expected sensitivity in one and two groups", {
  d <- hs()
  for (grouped in c(FALSE, TRUE)) {
    pair <- .ml_expected_pair(d, grouped = grouped)
    f0 <- pair$null; f1 <- pair$alternative
    policy <- magmaanlab::policy_nested(as_lab_fit(f1), as_lab_fit(f0))$score
    explicit <- magmaanlab::nested_score_test(as_lab_fit(f1), as_lab_fit(f0), data = d,
      sensitivity = "expected")
    expect_true(policy$available, info = policy$detail)
    expect_equal(policy$statistic, explicit$statistic_effective, tolerance = 1e-8)
    expect_equal(policy$df, explicit$df)
    expect_equal(policy$eigenvalues, explicit$eigenvalues, tolerance = 1e-8)
    peba <- magmaanlab:::infer_fmg_test(policy$statistic, policy$df,
      policy$eigenvalues, "peba", 4)$p_value
    table <- anova(f0, f1)
    row <- table[table$test == "score", ]
    expect_equal(row$statistic, policy$statistic, tolerance = 1e-8)
    expect_equal(row$pvalue, peba, tolerance = 1e-8)
    if (!grouped) {
      # Raw expected-metric quadratic (mi), not the mean-scaled 3.846701.
      expect_equal(row$statistic, 2.862175, tolerance = 1e-6)
    }
  }
})

test_that("ML nested scores and candidate MIs are identification invariant", {
  d <- hs()
  pairs <- lapply(c("marker", "std.lv"), function(id) .ml_expected_pair(d, id))
  scores <- lapply(pairs, function(p) {
    x <- anova(p$null, p$alternative)
    x[x$test == "score", c("statistic", "pvalue")]
  })
  expect_equal(scores[[1]], scores[[2]], tolerance = 1e-6)
  tables <- lapply(pairs, function(p) {
    f <- p$alternative
    out <- modindices(f, releases = FALSE)
    lab <- magmaanlab::modification_indices_robust(as_lab_fit(f), data = d,
      bread = "expected", information = "expected", estimated_weight = FALSE)
    key <- function(x) paste(x$lhs, x$op, x$rhs, x$group)
    idx <- match(key(out), key(lab))
    expect_false(anyNA(idx))
    expect_true(all(out$reason == "available"))
    expect_equal(out$statistic, lab$mi.scaled[idx], tolerance = 1e-8)
    expect_equal(out$pvalue, lab$pvalue[idx], tolerance = 1e-8)
    # Marker-fixed loadings differ between identifications; absent cross-loadings
    # and residual covariance candidates have the same identities.
    out <- out[out$op == "=~" | (out$op == "~~" & out$lhs != out$rhs), ]
    out <- out[order(key(out)), c("lhs", "op", "rhs", "group", "statistic", "pvalue")]
    rownames(out) <- NULL
    out
  })
  expect_equal(tables[[1]], tables[[2]], tolerance = 1e-6)
  mi <- modindices(pairs[[1]]$alternative, releases = FALSE)
  for (key in c("visual =~ x9", "textual =~ x1", "x7 ~~ x8")) {
    row <- mi[paste(mi$lhs, mi$op, mi$rhs) == key, ]
    expect_equal(nrow(row), 1L)
    expect_identical(row$reason, "available")
    expect_true(is.finite(row$statistic))
  }
})
