test_that("theta fitted tables reconstruct scales for ordinal and mixed models", {
  skip_if_not_installed("lavaan")
  syntax <- "f =~ x1 + x2 + x3"
  for (mixed in c(FALSE, TRUE)) for (grouped in c(FALSE, TRUE)) {
    d <- lavaan::HolzingerSwineford1939
    ordered <- if (mixed) "x1" else paste0("x", 1:3)
    for (v in ordered) d[[v]] <- as.integer(cut(d[[v]], 3))
    fit <- fit_model(syntax, d, ordered = ordered,
                     groups = if (grouped) "school" else NULL,
                     estimator = "DWLS", parameterization = "theta")
    oracle <- lavaan::cfa(syntax, d, ordered = ordered,
                          group = if (grouped) "school" else NULL,
                          estimator = "DWLS", parameterization = "theta")
    expect_true(fit$converged)
    p <- fit$partable
    q <- lavaan::parTable(oracle)
    key <- function(pt) paste(pt$lhs, pt$op, pt$rhs, pt$group)
    expect_setequal(key(p), key(q))
    expect_equal(p$est, q$est[match(key(p), key(q))], tolerance = 1e-4)
    scales <- p$op == "~*~"
    expect_true(all(p$est[scales] > 0 & p$est[scales] < 1))
    expect_equal(p$ustart[scales], rep(1, sum(scales)))
    expect_true(all(p$free[scales] == 0L))
    expect_equal(fit$parameterization, "theta")
    # Rebuilding for staged fitting must use the model's unit preparation,
    # even though the reported scale is derived from its fitted variance.
    spec <- model_spec(syntax, ordered = ordered, parameterization = "theta",
      group = if (grouped) "school" else NULL,
      group_labels = if (grouped) unique(as.character(d$school)) else NULL)
    model <- prepare_model(spec, prototype = d)
    data <- prepare_data(model, d)
    weight <- prepare_weight(data, "DWLS", full = FALSE)
    staged <- estimate(model, data, weight = weight)
    expect_true(staged$converged)
    expect_equal(staged$partable$est, p$est, tolerance = 1e-5)
  }
})
