# Live lavaan gates belong to the R suite; the C++ mean-row tests run offline.
scalar_syntax <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"

scalar_pair <- function(data, syntax = scalar_syntax, std_lv = FALSE,
                        equal = c("loadings", "intercepts"), partial = NULL,
                        estimator = "ML", growth = FALSE) {
  spec <- model_spec(syntax, meanstructure = TRUE, group = "school",
                     group_labels = unique(as.character(data$school)),
                     group_equal = equal, group_partial = partial,
                     std_lv = std_lv, model_type = if (growth) "growth" else "sem")
  ours <- fit_model(spec, data, estimator = estimator)
  oracle <- if (growth) lavaan::growth else lavaan::cfa
  theirs <- oracle(syntax, data, group = "school", meanstructure = TRUE,
                   group.equal = equal, group.partial = partial, std.lv = std_lv,
                   missing = if (estimator == "FIML") "ml" else "listwise")
  list(ours = ours, theirs = theirs)
}

expect_scalar_parity <- function(pair) {
  expect_true(isTRUE(pair$ours$converged))
  expect_true(lavaan::lavInspect(pair$theirs, "converged"))
  p <- pair$ours$partable
  q <- lavaan::parTable(pair$theirs)
  p <- p[p$op != "==", , drop = FALSE]
  q <- q[q$op != "==", , drop = FALSE]
  key <- function(x) paste(x$lhs, x$op, x$rhs, x$group)
  expect_false(anyDuplicated(key(p)) > 0L)
  expect_setequal(key(p), key(q))
  q <- q[match(key(p), key(q)), , drop = FALSE]
  expect_identical(p$free > 0L, q$free > 0L)
  expect_true(all(is.finite(p$est)))
  # Absolute tolerance: do not let large estimates loosen the parity gate.
  expect_lt(max(abs(p$est - q$est)), 1e-4)
  fm <- fit_measures(pair$ours, robust = FALSE)
  ref <- lavaan::fitMeasures(pair$theirs, c("df", "chisq"))
  expect_equal(as.numeric(fm$df), unname(ref["df"]))
  expect_lt(abs(fm$chisq - unname(ref["chisq"])), 1e-4)
}

test_that("keyword scalar invariance preserves mean identification in ML and FIML", {
  skip_if_not_installed("lavaan")
  for (estimator in c("ML", "FIML")) {
    d <- lavaan::HolzingerSwineford1939
    if (estimator == "FIML") d$x2[seq(1L, nrow(d), 7L)] <- NA_real_
    for (std_lv in c(FALSE, TRUE)) {
      expect_scalar_parity(scalar_pair(d, std_lv = std_lv, estimator = estimator))
    }
    expect_scalar_parity(scalar_pair(d, partial = "x1~1", estimator = estimator))
    expect_scalar_parity(scalar_pair(d,
      syntax = paste(scalar_syntax, "visual ~ c(0, 0.25)*1", sep = "\n"),
      estimator = estimator))
    expect_scalar_parity(scalar_pair(d, equal = c("loadings", "intercepts", "means"),
                                    estimator = estimator))
  }
})

test_that("three-group scalar invariance releases every nonreference mean", {
  skip_if_not_installed("lavaan")
  set.seed(250925)
  # One fixed draw exercises unequal sizes and nonzero group means, not size/power.
  d <- do.call(rbind, lapply(seq_len(3L), function(g) {
    n <- 120L + 30L * g
    eta <- rnorm(n, mean = c(0, 0.5, -0.4)[g])
    x <- outer(eta, c(1, 0.8, 1.2, 0.9)) + matrix(rnorm(n * 4L, sd = 0.6), n)
    out <- as.data.frame(x)
    names(out) <- paste0("x", 1:4)
    out$school <- LETTERS[g]
    out
  }))
  for (std_lv in c(FALSE, TRUE)) {
    pair <- scalar_pair(d, syntax = "f =~ x1 + x2 + x3 + x4", std_lv = std_lv)
    expect_scalar_parity(pair)
    p <- pair$ours$partable
    means <- p[p$op == "~1" & p$lhs == "f", ]
    expect_equal(means$group, 1:3)
    expect_identical(means$free > 0L, c(FALSE, TRUE, TRUE))
    expect_gt(means$est[2], 0.2)
    expect_lt(means$est[3], -0.2)
  }
})

test_that("growth identification retains free reference-group means", {
  skip_if_not_installed("lavaan")
  d <- lavaan::Demo.growth
  d$school <- rep(c("A", "B"), length.out = nrow(d))
  syntax <- "i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4\ns =~ 0*t1 + 1*t2 + 2*t3 + 3*t4"
  for (equal in list(c("loadings", "intercepts"),
                     c("loadings", "intercepts", "means"))) {
    pair <- scalar_pair(d, syntax = syntax, equal = equal, growth = TRUE)
    expect_scalar_parity(pair)
    p <- pair$ours$partable
    expect_true(all(p$free[p$op == "~1" & p$lhs %in% c("i", "s")] > 0L))
    expect_true(all(p$est[p$op == "~1" & p$lhs %in% paste0("t", 1:4)] == 0))
  }
})

for (std_lv in c(FALSE, TRUE)) for (estimator in c("ML", "FIML")) test_that(paste(estimator,
  if (std_lv) "std.lv" else "marker",
  "keyword scalar nested tests use the delta restriction map"), {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  if (estimator == "FIML") d$x2[seq(1L, nrow(d), 7L)] <- NA_real_
  if (estimator == "FIML" && std_lv) d$x5[seq(2L, nrow(d), 11L)] <- NA_real_
  metric <- scalar_pair(d, equal = "loadings", estimator = estimator, std_lv = std_lv)
  scalar <- scalar_pair(d, estimator = estimator, std_lv = std_lv)
  raw <- if (estimator == "ML") lapply(unique(as.character(d$school)), function(g) {
    as.matrix(d[d$school == g, paste0("x", 1:6)])
  }) else NULL
  nt <- robust_nested_lrt(metric$ours, scalar$ours, data = raw,
                           A.method = "delta", convention = "lavaan", method = "restriction_map")
  # Complete-data restriction-map geometry uses expected information (MLM).
  # FIML uses MLR; requesting MLM would silently select listwise deletion.
  oracle_estimator <- if (estimator == "ML") "MLM" else "MLR"
  lav_metric <- lavaan::cfa(scalar_syntax, d, group = "school",
    std.lv = std_lv,
    group.equal = "loadings", meanstructure = TRUE, estimator = oracle_estimator,
    missing = if (estimator == "FIML") "ml" else "listwise")
  lav_scalar <- lavaan::cfa(scalar_syntax, d, group = "school",
    std.lv = std_lv,
    group.equal = c("loadings", "intercepts"), meanstructure = TRUE,
    estimator = oracle_estimator, missing = if (estimator == "FIML") "ml" else "listwise")
  lr <- lavaan::lavTestLRT(lav_metric, lav_scalar, method = "satorra.2000",
                          A.method = "delta", scaled.shifted = FALSE)
  expect_equal(as.numeric(nt$df_diff), as.numeric(lr[2, "Df diff"]))
  delta <- lavaan::fitMeasures(lav_scalar, "chisq") -
    lavaan::fitMeasures(lav_metric, "chisq")
  expect_lt(abs(nt$T_diff - unname(delta)), 1e-4)
  expect_lt(abs(nt$T_scaled - lr[2, "Chisq diff"]), 5e-3)
  expect_lt(abs(nt$scale_c - unname(delta) / lr[2, "Chisq diff"]), 1e-4)
  expect_lt(abs(nt$p_scaled - lr[2, "Pr(>Chisq)"]), 1e-5)
})
