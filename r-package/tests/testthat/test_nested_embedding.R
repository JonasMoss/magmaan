nested_three <- paste("visual =~ x1 + x2 + x3", "textual =~ x4 + x5 + x6",
                      "speed =~ x7 + x8 + x9", sep = "\n")
nested_spellings <- c(nested_three,
  paste(nested_three, "visual =~ 0*x9", sep = "\n"),
  paste(nested_three, "visual =~ a*x9\na == 0", sep = "\n"))

# Different optimizers/constraint coordinates can produce slightly different
# estimates. First check independently fitted models, then use the identical
# numerical null point to gate the inference embedding to machine precision.
nested_at_same_point <- function(fit, reference) {
  key <- function(pt) paste(pt$lhs, pt$op, pt$rhs, pt$group)
  p <- fit$partable
  q <- reference$partable
  values <- q$est[match(key(p), key(q))]
  values[is.na(values) & p$op == "=~" & p$rhs == "x9"] <- 0
  rows <- which(p$free > 0L)
  fit$theta[p$free[rows]] <- values[rows]
  fit$partable$est[rows] <- values[rows]
  fit
}

nested_numeric <- function(x) {
  unlist(x[c("T_diff", "df_diff", "scale_c", "p_scaled", "p_adjusted",
             "p_mixture", "eigenvalues")], use.names = FALSE)
}

test_that("exact ML path embedding agrees across spellings and with lavaan", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  h1 <- fit_model(paste(nested_three, "visual =~ x9", sep = "\n"), d)
  nulls <- lapply(nested_spellings, fit_model, data = d)
  expected <- policy_nested(h1, nulls[[3L]])
  expect_true(expected$score$available)
  expect_true(expected$lr$available)
  for (h0 in nulls) {
    independent <- policy_nested(h1, h0)
    expect_true(independent$score$available)
    expect_equal(independent$score$statistic, expected$score$statistic, tolerance = 1e-5)
    same <- nested_at_same_point(h0, nulls[[3L]])
    out <- policy_nested(h1, same)
    for (component in c("score", "lr")) {
      expect_equal(out[[component]]$statistic, expected[[component]]$statistic, tolerance = 1e-10)
      expect_equal(out[[component]]$eigenvalues, expected[[component]]$eigenvalues, tolerance = 1e-10)
      expect_equal(out[[component]]$p_sb, expected[[component]]$p_sb, tolerance = 1e-10)
      expect_equal(out[[component]]$p_peba4, expected[[component]]$p_peba4, tolerance = 1e-10)
    }
  }
  lr <- lapply(nulls, function(h0) robust_nested_lrt(h1,
    nested_at_same_point(h0, nulls[[3L]]), data = d, A.method = "exact", method = "restriction_map"))
  for (x in lr[-1L]) expect_equal(nested_numeric(x), nested_numeric(lr[[1L]]), tolerance = 1e-10)
  scores <- lapply(nulls, function(h0) nested_score_test(h1,
    nested_at_same_point(h0, nulls[[3L]]), data = d))
  for (x in scores[-1L]) expect_equal(x$statistic, scores[[1L]]$statistic, tolerance = 1e-10)
  # Complete-data Satorra uses expected information; match that option in the oracle.
  lav1 <- lavaan::cfa(paste(nested_three, "visual =~ x9", sep = "\n"), d,
                      estimator = "MLR", information = "expected")
  lav0 <- lavaan::cfa(nested_spellings[[2L]], d, estimator = "MLR", information = "expected")
  oracle <- lavaan::lavTestLRT(lav1, lav0, method = "satorra.2000", A.method = "exact")
  compat <- robust_nested_lrt(h1, nulls[[2L]], data = d,
                              A.method = "exact", convention = "lavaan", method = "restriction_map")
  expect_equal(compat$T_scaled, oracle[["Chisq diff"]][2L], tolerance = 1e-4)
  # lavTestScore() is the expected nested geometry; the policy uses the
  # observed one. The lab reproduces both.
  shared <- prepare_inference_data(h1, d)
  hyp <- prepare_hypothesis(prepare_inference(nulls[[3L]], shared), prepare_inference(h1, shared))
  expect_equal(inference_quadratic(hyp, "score", geometry = "observed")$statistic,
               expected$score$statistic, tolerance = 1e-10)
  score_null <- lavaan::cfa(nested_spellings[[3L]], d)
  expect_equal(inference_quadratic(hyp, "score", geometry = "expected")$statistic,
               lavaan::lavTestScore(score_null, release = 1)$test$X2, tolerance = 1e-4)
})

test_that("FIML, ordinal and weighted exact paths share key embedding", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  for (estimator in c("FIML", "DWLS", "ULS", "ML2S")) {
    input <- d
    if (estimator %in% c("FIML", "ML2S")) input$x2[seq(1L, nrow(input), 9L)] <- NA_real_
    ordered <- NULL
    if (estimator == "DWLS") {
      ordered <- paste0("x", 1:9)
      input[ordered] <- lapply(input[ordered], function(x) ordered(cut(x,
        breaks = c(-Inf, quantile(x, c(1/3, 2/3)), Inf), labels = FALSE)))
    }
    spec1 <- model_spec(paste(nested_three, "visual =~ x9", sep = "\n"), ordered = ordered)
    if (estimator == "DWLS") input <- magmaanlab:::data_ordinal_stats_from_df(input, spec1)
    h1 <- fit_model(spec1, input, estimator = estimator)
    nulls <- lapply(nested_spellings, function(syntax)
      fit_model(model_spec(syntax, ordered = ordered), input, estimator = estimator))
    lr <- lapply(nulls, function(h0) robust_nested_lrt(h1,
      nested_at_same_point(h0, nulls[[3L]]),
      data = if (estimator %in% c("FIML", "ML2S")) NULL else input,
      A.method = "exact", method = "restriction_map"))
    for (x in lr[-1L]) expect_equal(nested_numeric(x), nested_numeric(lr[[1L]]), tolerance = 1e-9)
    if (estimator == "FIML") {
      scores <- lapply(nulls, function(h0) nested_score_test(h1,
        nested_at_same_point(h0, nulls[[3L]])))
      for (x in scores[-1L]) expect_equal(x$statistic, scores[[1L]]$statistic, tolerance = 1e-10)
    }
  }
})

test_that("multigroup drop/fix embedding retains group-specific constraints", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  groups <- unique(as.character(d$school))
  raw <- lapply(groups, function(g) as.matrix(d[d$school == g, paste0("x", 1:9)]))
  h1 <- fit_model(model_spec(paste(nested_three, "visual =~ x9", sep = "\n"),
                             group = "school", group_labels = groups), d)
  fixed <- fit_model(model_spec(paste(nested_three, "visual =~ c(0, NA)*x9", sep = "\n"),
                                group = "school", group_labels = groups), d)
  equal <- fit_model(model_spec(paste(nested_three, "visual =~ c(b, a)*x9\nb == 0", sep = "\n"),
                                group = "school", group_labels = groups), d)
  fixed <- nested_at_same_point(fixed, equal)
  result <- policy_nested(h1, fixed)
  expected <- policy_nested(h1, equal)
  expect_true(result$score$available)
  expect_equal(result$score$statistic, expected$score$statistic, tolerance = 1e-10)
  expect_equal(result$score$eigenvalues, expected$score$eigenvalues, tolerance = 1e-10)
  lav1 <- lavaan::cfa(paste(nested_three, "visual =~ x9", sep = "\n"), d,
                      group = "school", estimator = "MLR", information = "expected")
  lav0 <- lavaan::cfa(paste(nested_three, "visual =~ c(b,a)*x9\nb == 0", sep = "\n"), d,
                      group = "school", estimator = "MLR", information = "expected")
  oracle <- lavaan::lavTestLRT(lav1, lav0, method = "satorra.2000", A.method = "exact")
  expect_equal(robust_nested_lrt(h1, equal, data = raw, method = "restriction_map")$T_scaled,
               oracle[["Chisq diff"]][2L], tolerance = 1e-4)
  expect_equal(nested_numeric(robust_nested_lrt(h1, fixed, data = raw, method = "restriction_map")),
               nested_numeric(robust_nested_lrt(h1, equal, data = raw, method = "restriction_map")), tolerance = 1e-10)
})

test_that("covariance-only nested scores profile the mean (structural-path constants)", {
  skip_if_not_installed("lavaan")
  # The research/52 report: on a covariance-only structural model with two
  # structural paths fixed to zero, score_components() disagreed with
  # lavTestScore(). Its complete-data scores fixed the mean at zero; they now
  # profile it, so data far from mean zero give lavaan's statistic.
  d <- lavaan::HolzingerSwineford1939[, paste0("x", 1:9)] + 10
  h1_syntax <- paste("visual =~ x1 + x2 + x3", "textual =~ x4 + x5 + x6",
                     "speed =~ x7 + x8 + x9", "speed ~ r1*visual + r2*textual", sep = "\n")
  h0_syntax <- paste(h1_syntax, "r1 == 0", "r2 == 0", sep = "\n")
  h1 <- fit_model(h1_syntax, d)
  h0 <- fit_model(h0_syntax, d)
  projected <- project_scores(score_components(prepare_inference(h0, d), H1 = model_spec(h1_syntax), sensitivity = "expected"))
  shared <- prepare_inference_data(h1, d)
  hyp <- prepare_hypothesis(prepare_inference(h0, shared), prepare_inference(h1, shared))
  expect_true(is.finite(inference_quadratic(hyp, "score")$statistic))
  reference <- lavaan::lavTestScore(lavaan::sem(h0_syntax, d))$test$X2
  expect_equal(projected$statistic, reference, tolerance = 1e-5)
  expect_equal(inference_quadratic(hyp, "score", geometry = "expected")$statistic, reference, tolerance = 1e-5)
})
