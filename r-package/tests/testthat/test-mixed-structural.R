test_that("mixed structural models report delta residuals and theta scales", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  ordered <- paste0("x", 1:3)
  for (v in ordered) d[[v]] <- as.integer(cut(d[[v]], 3))
  syntax <- "f =~ x1 + x2 + x3\ng =~ x4 + x5 + x6\ng ~ f"
  key <- function(p) paste(p$lhs, p$op, p$rhs, p$group)
  for (param in c("delta", "theta")) {
    spec <- model_spec(syntax, ordered = ordered, parameterization = param,
                       fixed_x = FALSE, auto_cov_y = TRUE)
    fit <- fit_model(spec, d, estimator = "DWLS")
    oracle <- lavaan::sem(syntax, d, ordered = ordered, parameterization = param,
                          fixed.x = FALSE, estimator = "DWLS")
    p <- fit$partable
    q <- lavaan::parTable(oracle)
    expect_true(fit$converged)
    expect_setequal(key(p), key(q))
    expect_equal(p$est, q$est[match(key(p), key(q))], tolerance = 1e-4)
    expect_equal(model_matrix_rep(p)$form, "Reduced")
    stats <- data_mixed_ordinal_stats_from_df(d, spec, full_wls_weight = FALSE)
    ov <- stats$ov_names[[1]]
    sample <- lavaan::lavInspect(oracle, "sampstat")
    expect_equal(unname(stats$R[[1]]), unname(sample$cov[ov, ov]), tolerance = 2e-6)
    expect_equal(unname(stats$mean[[1]]), unname(sample$mean[ov]), tolerance = 1e-10)
    expect_equal(unname(stats$thresholds[[1]]), unname(sample$th), tolerance = 1e-8)
    continuous <- ov[stats$ordered_mask[[1]] == 0L]
    keys <- c(paste0(ov[stats$threshold_ov[[1]]], "|t", stats$threshold_level[[1]]),
              paste0(continuous, "~1"), paste0(continuous, "~~", continuous))
    for (j in seq_len(length(ov) - 1L)) for (i in (j + 1L):length(ov)) {
      keys <- c(keys, paste0(ov[i], "~~", ov[j]))
    }
    canonical <- function(keys) vapply(strsplit(keys, "~~", fixed = TRUE),
      function(k) paste(sort(k), collapse = "~~"), character(1))
    observed <- lavaan::lavInspect(oracle, "wls.obs")
    ix <- match(canonical(keys), canonical(names(observed)))
    expect_false(anyNA(ix))
    expect_equal(unname(stats$moments[[1]]), unname(observed[ix]), tolerance = 2e-6)
    expect_equal(unname(stats$NACOV[[1]]),
                 unname(lavaan::lavInspect(oracle, "gamma")[ix, ix]), tolerance = 2e-5)
    expect_equal(unname(stats$W_dwls[[1]]),
                 unname(lavaan::lavInspect(oracle, "wls.v")[ix, ix]), tolerance = 2e-5)
    model <- prepare_model(spec, prototype = d)
    data <- prepare_data(model, d)
    staged <- estimate(model, data, weight = prepare_weight(data, "DWLS", full = FALSE))
    expect_true(staged$converged)
    expect_equal(staged$partable$est, p$est, tolerance = 1e-5)
    residuals <- p$op == "~~" & p$lhs == p$rhs & p$lhs %in% ordered
    expect_equal(p$ustart[residuals], rep(1, sum(residuals)))
    if (param == "delta") expect_true(all(p$est[residuals] < 1))
  }
})
