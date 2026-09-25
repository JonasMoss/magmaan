mean_default_syntax <- "f =~ x1 + x2 + x3"

expect_mean_rows <- function(spec, oracle, check_free = TRUE) {
  rows <- function(pt) {
    pt <- pt[pt$op == "~1", ]
    key <- paste(pt$lhs, pt$group)
    data.frame(key = key, free = pt$free > 0L)[order(key), , drop = FALSE]
  }
  p <- rows(spec$partable)
  q <- rows(lavaan::parTable(oracle))
  rownames(p) <- rownames(q) <- NULL
  if (!check_free) { p$free <- NULL; q$free <- NULL }
  expect_equal(p, q)
}

test_that("model_spec resolves lavaan mean defaults and explicit overrides", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  for (grouped in c(FALSE, TRUE)) {
    for (means in list("default", FALSE, TRUE)) {
      args <- list(model = mean_default_syntax, data = d, do.fit = FALSE)
      if (grouped) args$group <- "school"
      if (!identical(means, "default")) args$meanstructure <- means
      spec <- model_spec(mean_default_syntax, meanstructure = means,
        group = if (grouped) "school" else NULL,
        group_labels = if (grouped) unique(as.character(d$school)) else NULL)
      expect_mean_rows(spec, do.call(lavaan::cfa, args))
    }
  }
  for (means in list("default", FALSE)) {
    syntax <- paste(mean_default_syntax, "x1 ~ 1", sep = "\n")
    expect_mean_rows(model_spec(syntax, meanstructure = means),
      lavaan::cfa(syntax, d, meanstructure = FALSE, do.fit = FALSE))
  }
  expect_false(any(model_spec(paste(mean_default_syntax, "# x1 ~ 1"))$partable$op == "~1"))
  expect_error(model_spec(mean_default_syntax, meanstructure = NA), "meanstructure")
})

test_that("late grouping preserves automatic defaults and explicit FALSE", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  for (means in list("default", FALSE)) {
    spec <- model_spec(mean_default_syntax, meanstructure = means)
    fit <- fit_model(spec, d, groups = "school")
    args <- list(model = mean_default_syntax, data = d, group = "school")
    if (!identical(means, "default")) args$meanstructure <- means
    oracle <- do.call(lavaan::cfa, args)
    expect_mean_rows(fit, oracle)
    expect_true(fit$converged)
    p <- fit$partable
    q <- lavaan::parTable(oracle)
    key <- function(x) paste(x$lhs, x$op, x$rhs, x$group)
    expect_setequal(key(p), key(q))
    expect_lt(max(abs(p$est - q$est[match(key(p), key(q))])), 1e-4)
  }
})

test_that("ordered and missing-data preparation resolve means in the lab", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  ordered <- paste0("x", 1:3)
  for (v in ordered) d[[v]] <- as.integer(cut(d[[v]], 3))
  oracle <- lavaan::cfa(mean_default_syntax, d, ordered = ordered, do.fit = FALSE)
  prepare <- magmaanlab:::.magmaan_prepare_spec
  for (means in list("default", FALSE)) {
    spec <- model_spec(mean_default_syntax, meanstructure = means, ordered = ordered)
    # Categorical augmentation sets intercept identification at fit preparation.
    # The syntax spec must already contain every required mean row.
    expect_mean_rows(spec, oracle, check_free = FALSE)
    late <- prepare(model_spec(mean_default_syntax, meanstructure = means),
                    d, "DWLS", NULL, list(), ordered, "delta")$spec
    expect_mean_rows(late, oracle, check_free = FALSE)
  }
  for (estimator in c("FIML", "ML2S")) {
    for (model in list(mean_default_syntax, model_spec(mean_default_syntax))) {
      spec <- prepare(model, d, estimator, NULL, list(), NULL, "delta")$spec
      expect_mean_rows(spec, lavaan::cfa(mean_default_syntax, d,
                       meanstructure = TRUE, do.fit = FALSE))
    }
    expect_true(any(prepare(mean_default_syntax, d, estimator, NULL,
      list(meanstructure = "default"), NULL, "delta")$spec$partable$op == "~1"))
  }
})
