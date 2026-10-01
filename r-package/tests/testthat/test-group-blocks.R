group_block_syntax <- paste(
  "group: 1", "f =~ x1 + x2 + x3",
  "group: 2", "f =~ x1 + 0.6*x2 + x3", sep = "\n")

group_block_data <- function() {
  d <- lavaan::HolzingerSwineford1939
  for (v in paste0("x", 1:3)) d[[v]] <- as.integer(cut(d[[v]], 3))
  d
}

test_that("ordered group blocks fit distinct templates with lavaan parity", {
  skip_if_not_installed("lavaan")
  d <- group_block_data()
  ordered <- paste0("x", 1:3)
  for (parameterization in c("delta", "theta")) {
    fit <- fit_model(group_block_syntax, d, groups = "school", ordered = ordered,
                     estimator = "DWLS", parameterization = parameterization)
    oracle <- lavaan::cfa(group_block_syntax, d, group = "school", ordered = ordered,
                          estimator = "DWLS", parameterization = parameterization)
    expect_true(fit$converged)
    expect_equal(sort(unique(fit$partable$block)), 1:2)
    expect_equal(fit$group_labels, unique(as.character(d$school)))
    key <- function(pt) paste(pt$lhs, pt$op, pt$rhs, pt$block)
    p <- fit$partable
    q <- lavaan::parTable(oracle)
    expect_setequal(key(p), key(q))
    expect_equal(p$est, q$est[match(key(p), key(q))], tolerance = 1e-4)
    expect_equal(p$free > 0, q$free[match(key(p), key(q))] > 0)
  }
})

test_that("prebuilt and staged ordinal group specs preserve data group ordering", {
  skip_if_not_installed("lavaan")
  d <- group_block_data()
  labels <- rev(unique(as.character(d$school)))
  spec <- model_spec(group_block_syntax, group = "school", group_labels = labels,
                     ordered = paste0("x", 1:3))
  stats <- data_ordinal_stats_from_df(d, spec, full_wls_weight = FALSE)
  expect_equal(stats$group_labels, labels)
  expect_length(stats$R, 2)
  # A prebuilt spec already owns its grouping column; no `groups` is needed.
  fit <- fit_model(spec, d, estimator = "DWLS")
  expect_true(fit$converged)
  expect_equal(fit$group_labels, labels)
  model <- prepare_model(spec, prototype = d)
  data <- prepare_data(model, d)
  weight <- prepare_weight(data, "DWLS", full = FALSE)
  staged <- estimate(model, data, weight = weight)
  expect_true(staged$converged)
  expect_equal(staged$partable$est, fit$partable$est, tolerance = 1e-5)
})

test_that("group blocks also preserve mixed and continuous model grouping", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  for (ordered in list("x1", character())) {
    if (length(ordered)) d$x1 <- as.integer(cut(d$x1, 3))
    fit <- fit_model(group_block_syntax, d, groups = "school", ordered = ordered,
                     estimator = if (length(ordered)) "DWLS" else "ML")
    expect_true(fit$converged)
    expect_equal(sort(unique(fit$partable$block)), 1:2)
  }
})
