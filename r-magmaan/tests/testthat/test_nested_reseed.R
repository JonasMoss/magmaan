# Independently reconstruct decisions/04 cell 17, rep 542, seed 726270583.
# The test owns its formulas and never reads the experiment leaf.
nested_reseed_data <- function(seed = 726270583L, skewed = FALSE) {
  loading <- rep(c(.7, .8, .75, .65), 2)
  residual <- 1 - loading^2
  residual[5] <- residual[5] - .4^2 - 2*.4*loading[5]*.4
  population <- paste(c(
    paste0('f1 =~ ', paste0(loading[1:4], '*x', 1:4, collapse = ' + ')),
    paste0('f2 =~ ', paste0(loading[5:8], '*x', 5:8, collapse = ' + ')),
    'f1 =~ .4*x5', 'f1 ~~ 1*f1', 'f2 ~~ 1*f2', 'f1 ~~ .4*f2',
    paste0('x', 1:8, ' ~~ ', residual, '*x', 1:8),
    paste0('x1 ~~ ', .3 * sqrt(residual[1]*residual[5]), '*x5')), collapse = '\n')
  set.seed(seed)
  do.call(rbind, lapply(c('a', 'b'), function(g) {
    x <- lavaan::simulateData(population, sample.nobs = 100,
      skewness = if (skewed) rep(2, 8) else NULL,
      kurtosis = if (skewed) rep(7, 8) else NULL)
    x$group <- g
    x
  }))
}

 test_that("anova recovers the decisions/04 larger local optimum without replacing fits", {
  skip_if_not_installed("lavaan")
  data <- nested_reseed_data()
  syntax <- 'f1 =~ x1 + x2 + x3 + x4\nf2 =~ x5 + x6 + x7 + x8'
  fit <- function(equal = NULL) magmaan(magmaan_model(syntax,
    prototype = data, group = 'group', group.equal = equal), data, inference = FALSE)
  # This fixture deliberately reaches covariance-inadmissible endpoints.
  h1 <- suppressWarnings(fit()); h0 <- suppressWarnings(fit('loadings'))
  before <- lapply(list(h1, h0), function(x) list(theta = x$lab$theta, fmin = x$lab$fmin))
  failed <- magmaanlab::policy_nested(h1$lab, h0$lab)
  expect_identical(failed$lr$reason, "not_converged")
  expect_match(failed$lr$detail, "alternative fits worse")
  expect_warning(out <- anova(h1, h0), NA)
  expect_match(attr(out, "reseed")$warnings, "covariance-admissible")
  expect_output(print(out), "covariance-admissible")
  expect_length(attr(out, "unavailable"), 0L)
  expect_true(all(is.finite(out$statistic)))
  expect_true(all(is.finite(out$pvalue[out$reference %in% "sb"])))
  expect_lt(attr(out, "refit")$objective_after, attr(out, "refit")$objective_before)
  expect_identical(attr(out, "refit")$verdict$status, "passed")
  expect_output(print(out), "larger model refit from the restricted estimate")
  expect_equal(anova(h0, h1)$statistic, out$statistic, tolerance = 0)
  alternatives <- anova(h1, h0, references = c("all", "mv"))
  expect_identical(attr(alternatives, "refit"), attr(out, "refit"))
  expect_identical(attr(alternatives, "spectra"), attr(out, "spectra"))
  for (component in c("score", "lr")) {
    label <- if (component == "score") "score" else "lr"
    rows <- alternatives[alternatives$test == label, ]
    explicit <- magmaanlab::calibrate_quadratic(magmaanlab::quadratic_reference(
      rows$statistic[1], rows$df[1], attr(out, "spectra")[[component]]), c("all", "mv"))
    expect_identical(rows$pvalue, explicit$p_value)
  }
  expect_warning(compat <- anova(h1, h0, lavaan_compat = "ML"), NA)
  expect_identical(attr(compat, "reseed")$warnings, attr(out, "reseed")$warnings)
  expect_true(all(is.finite(compat$statistic)))
  expect_false(is.null(attr(compat, "refit")))
  expect_identical(lapply(list(h1, h0), function(x) list(theta = x$lab$theta, fmin = x$lab$fmin)), before)
})

test_that("well behaved nested results are unchanged by recovery", {
  skip_if_not_installed("lavaan")
  data <- lavaan::HolzingerSwineford1939
  h0 <- magmaan('visual =~ x1 + x2 + x3 + x4', data, inference = FALSE)
  expect_warning(h1 <- magmaan('visual =~ x1 + x2 + x3 + x4\nx1 ~~ x2',
    data, inference = FALSE), "covariance-admissible")
  expected <- magmaanlab::policy_nested(h1$lab, h0$lab)
  out <- anova(h1, h0)
  expect_null(attr(out, "refit"))
  expect_identical(out$statistic, rep(c(expected$score$statistic, expected$lr$statistic), each = 2))
  expect_identical(out$pvalue[out$reference %in% "sb"], c(expected$score$p_sb, expected$lr$p_sb))
})

test_that("FIML recovers a deliberately supplied bad basin and DWLS refits a bad start", {
  skip_if_not_installed("lavaan")
  syntax <- 'f1 =~ x1 + x2 + x3 + x4\nf2 =~ x5 + x6 + x7 + x8'
  for (estimator in c("FIML", "DWLS")) {
    data <- nested_reseed_data()
    ordered_names <- NULL
    if (estimator == "DWLS") {
      ordered_names <- paste0("x", 1:8)
      data[ordered_names] <- lapply(data[ordered_names], function(x)
        ordered(cut(x, quantile(x, c(0, .33, .67, 1)), include.lowest = TRUE)))
    }
    fit <- function(equal = NULL, options = NULL) suppressWarnings(magmaan(
      magmaan_model(syntax, prototype = data, group = "group",
        group.equal = equal, ordered = ordered_names), data,
      estimator = estimator, inference = FALSE, options = options))
    h0 <- fit("loadings"); h1 <- fit()
    start <- h1$lab$partable
    start$start <- start$est
    if (estimator == "DWLS") {
      set.seed(1)
      loading <- start$op == "=~" & start$free > 0L
      start$start[loading] <- runif(sum(loading), -2, 2)
      start$start[start$op == "~~" & start$lhs != start$rhs] <- 0
    }
    bad <- fit(options = list(start = start))
    before <- list(bad$lab$theta, bad$lab$fmin, h0$lab$theta)
    expect_gt(bad$lab$fmin, h0$lab$fmin)
    retry <- suppressWarnings(magmaanlab::refit_from_null(bad$lab, h0$lab))
    expect_true(isTRUE(retry$converged))
    expect_identical(retry$diagnostics$verdict$status, "passed")
    expect_lte(retry$fmin, h0$lab$fmin)
    nested <- magmaanlab::policy_nested(retry, h0$lab)
    expect_true(nested$lr$available)
    expect_true(is.finite(nested$lr$p_sb))
    if (estimator == "FIML") {
      expect_warning(result <- anova(bad, h0), NA)
      expect_true(all(is.finite(result$statistic)))
      expect_false(is.null(attr(result, "refit")))
      expect_output(print(result), "larger model refit from the restricted estimate")
      expect_warning(compat <- anova(bad, h0, lavaan_compat = "ML"), NA)
      expect_true(all(is.finite(compat$statistic)))
      expect_false(is.null(attr(compat, "refit")))
    } else {
      # A failed native verdict keeps its typed failure in ordinary anova();
      # the lab's explicit reseed remains available for this endpoint.
      expect_false(isTRUE(bad$lab$converged))
      expect_match(attr(anova(bad, h0), "unavailable")[["lr"]], "not_converged")
      expect_null(attr(anova(bad, h0), "refit"))
    }
    expect_identical(list(bad$lab$theta, bad$lab$fmin, h0$lab$theta), before)
  }
})

test_that("an unsuccessful nested retry preserves the original typed failure", {
  skip_if_not_installed("lavaan")
  data <- nested_reseed_data()
  syntax <- 'f1 =~ x1 + x2 + x3 + x4\nf2 =~ x5 + x6 + x7 + x8'
  fit <- function(equal = NULL) suppressWarnings(magmaan(magmaan_model(syntax,
    prototype = data, group = 'group', group.equal = equal), data, inference = FALSE))
  h1 <- fit(); h0 <- fit('loadings')
  original <- magmaanlab::policy_nested(h1$lab, h0$lab)
  calls <- 0L
  testthat::local_mocked_bindings(refit_from_null = function(...) {
    calls <<- calls + 1L
    warning("internal refit warning")
    stop("refit unavailable")
  }, .package = "magmaanlab")
  expect_warning(result <- anova(h1, h0), NA)
  expect_identical(attr(result, "reseed")$warnings, "internal refit warning")
  expect_output(print(result), "internal refit warning")
  expect_identical(calls, 1L)
  expect_null(attr(result, "refit"))
  expect_identical(attr(result, "unavailable")[["lr"]],
    paste0(original$lr$reason, ": ", original$lr$detail))
  expect_identical(result$statistic, c(rep(original$score$statistic, 2), original$lr$statistic))
})
