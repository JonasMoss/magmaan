# Retained-fit mixed WLSMV gates, doctest-relative 1e-5.
test_that("mixed WLSMV global and nested bundles match lavaan", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  ord <- paste0("x", 1:3)
  for (v in ord) d[[v]] <- ordered(cut(d[[v]],
    quantile(d[[v]], c(0, 1/3, 2/3, 1)), include.lowest = TRUE, labels = FALSE))
  syntax <- "f =~ x1 + x2 + x3\ng =~ x4 + x5 + x6"
  relative <- function(x, y) expect_lte(max(abs(x-y)),
    1e-5 * max(abs(x), abs(y)) + 1e-14)
  for (p in c("delta", "theta")) for (grouped in c(FALSE, TRUE)) {
    fits <- refs <- vector("list", 2)
    for (i in 1:2) {
      eq <- if (grouped && i == 2) "loadings" else NULL
      s <- if (!grouped && i == 2) paste(syntax, "f ~~ 0*g", sep = "\n") else syntax
      spec <- model_spec(s, ordered = ord, meanstructure = TRUE,
        parameterization = p, group = if (grouped) "school" else NULL,
        group_labels = if (grouped) levels(d$school) else NULL, group_equal = eq)
      fits[[i]] <- fit_model(spec, d, estimator = "DWLS", control = list(ftol = 1e-14, gtol = 1e-10, max_iter = 5000))
      refs[[i]] <- lavaan::cfa(s, d, ordered = ord, estimator = "WLSMV",
        parameterization = p, group = if (grouped) "school" else NULL,
        group.label = if (grouped) levels(d$school) else NULL, group.equal = eq)
      expect_true(fits[[i]]$converged)
      ours <- convention_inference(fits[[i]], "WLSMV")
      expect_true(ours$covariance_available, info = ours$covariance_detail)
      a <- fits[[i]]$partable; b <- lavaan::parTable(refs[[i]])
      a <- a[a$free > 0 & !duplicated(a$free), ]; a <- a[order(a$free), ]
      b <- b[b$free > 0 & !duplicated(b$free), ]; b <- b[order(b$free), ]
      key <- function(x) paste(x$lhs, x$op, x$rhs, x$group)
      ix <- match(key(a), key(b)); expect_false(anyNA(ix))
      relative(a$est, b$est[ix])
      relative(ours$covariance, lavaan::vcov(refs[[i]])[ix, ix])
      t <- lavaan::lavInspect(refs[[i]], "test")$scaled.shifted
      relative(ours$test$statistic, t$stat); expect_equal(ours$test$df, t$df)
      relative(ours$test$pvalue, t$pvalue)
      relative(ours$test$scale, t$scaling.factor)
      relative(ours$test$shift, t$shift.parameter)
    }
    ref <- lavaan::lavTestLRT(refs[[1]], refs[[2]])
    explicit <- lavaan::lavTestLRT(refs[[1]], refs[[2]],
      method = "satorra.2000", A.method = "delta", scaled.shifted = TRUE)
    expect_equal(ref, explicit)
    ours <- convention_nested(fits[[1]], fits[[2]], "WLSMV")$test
    expect_true(ours$available, info = ours$detail)
    relative(ours$statistic, ref[2, "Chisq diff"])
    expect_equal(ours$df, ref[2, "Df diff"])
    relative(ours$pvalue, ref[2, "Pr(>Chisq)"])
    relative(ours$scale, 1/attr(ref, "scale")[[2]])
    relative(ours$shift, attr(ref, "shift")[[2]])
    expect_identical(convention_inference(fits[[1]], "ULSMV")$test$reason, "unsupported_model")
  }
})
