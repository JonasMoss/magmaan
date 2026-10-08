.post_check_oracle <- function() {
  skip_if_not_installed("lavaan")
  skip_if(as.character(utils::packageVersion("lavaan")) != "0.7.2")
}
.post_check_compare <- function(model, data, estimator = "ML", ordered = NULL) {
  fit <- suppressWarnings(fit_model(model, data, meanstructure = TRUE,
      fixed_x = FALSE, options = list(preset = "lavaan-0.7.2"), estimator = estimator,
      ordered = ordered))
  oracle <- suppressWarnings(lavaan::sem(model, data, meanstructure = TRUE,
      fixed.x = FALSE, estimator = if (estimator == "FIML") "ML" else estimator,
      missing = if (estimator == "FIML") "ml" else "listwise", ordered = ordered))
  mp <- fit$partable
  lp <- lavaan::parTable(oracle)
  keys <- function(p) paste(p$lhs, p$op, p$rhs, p$group, sep = "\r")
  expect_equal(mp$est, lp$est[match(keys(mp), keys(lp))], tolerance = 1e-5)
  expect_equal(fit$fitting$post_check$ok,
      suppressWarnings(lavaan::lavInspect(oracle, "post.check")))
  list(fit = fit, oracle = oracle)
}

test_that("post.check matches proper and literal improper ML fits", {
  .post_check_oracle()
  set.seed(1292)
  d <- as.data.frame(matrix(rnorm(600), 200, 3))
  names(d) <- c("x1", "x2", "x3")
  cases <- list(
    proper = list("f =~ 1*x1+1*x2\nf ~~ 1*f\nx1 ~~ 1*x1\nx2 ~~ 1*x2", NULL),
    heywood = list("f =~ 1*x1+1*x2\nf ~~ 2*f\nx1 ~~ -1*x1\nx2 ~~ 5*x2", "ov_variance_negative"),
    negative_lv = list("f =~ 1*x1+1*x2\nf ~~ -1*f\nx1 ~~ 5*x1\nx2 ~~ 5*x2", "lv_variance_negative"),
    correlation = list("f =~ 1*x1\ng =~ 1*x2\nf ~~ 1*f\ng ~~ 1*g\nf ~~ 2*g\nx1 ~~ 5*x1\nx2 ~~ 5*x2", "cov_lv_not_pd"),
    residual = list("f =~ 1*x1+-1*x2\nf ~~ 2*f\nx1 ~~ 1*x1\nx2 ~~ 1*x2\nx1 ~~ 2*x2", "theta_not_pd"))
  for (case in cases) {
    result <- .post_check_compare(case[[1]], d)
    p <- result$fit$fitting$post_check
    expect_false(p$var_na)
    if (is.null(case[[2]])) expect_true(p$ok)
    else {
      expect_false(p$ok)
      expect_true(p[[case[[2]]]])
      expect_equal(sum(unlist(p[-c(1,2)])), 1)
    }
  }
})

test_that("post.check uses propagated latent covariance and removes dummy latents", {
  .post_check_oracle()
  d <- lavaan::HolzingerSwineford1939
  for (m in c("f =~ x1+x2+x3\ng =~ x4+x5+x6\ng ~ f",
              "f =~ x1+x2+x3\nf ~ x4")) {
    result <- .post_check_compare(m, d)
    expect_true(result$fit$fitting$post_check$ok)
    covariance <- lavaan::lavTech(result$oracle, "cov.lv")[[1]]
    expect_equal(nrow(covariance), if (grepl("g =~", m, fixed = TRUE)) 2L else 1L)
    if (nrow(covariance) == 2L)
      expect_gt(max(abs(covariance - result$oracle@Model@GLIST[[which(names(result$oracle@Model@GLIST) == "psi")[1]]])), 0.01)
  }
  native <- fit_model("f =~ x1+x2+x3", d,
      options = list(preset = "lavaan-0.7.2", convergence = "newton"))
  expect_null(native$fitting$post_check)
})

test_that("post.check is retained for FIML and categorical DWLS", {
  .post_check_oracle()
  d <- lavaan::HolzingerSwineford1939
  d$x1[seq(1, nrow(d), 5)] <- NA_real_
  result <- .post_check_compare("f =~ x1+x2+x3+x4", d, estimator = "FIML")
  expect_true(result$fit$fitting$post_check$ok)
  d <- lavaan::HolzingerSwineford1939
  for (v in c("x1", "x2", "x3", "x4"))
    d[[v]] <- ordered(cut(d[[v]], breaks = quantile(d[[v]], c(0, 1/3, 2/3, 1)), include.lowest = TRUE))
  for (ordered_names in list(c("x1", "x2", "x3", "x4"), c("x1", "x2"))) {
    # Keep the remaining mixed responses continuous.
    mixed_data <- d
    remaining <- setdiff(c("x1", "x2", "x3", "x4"), ordered_names)
    for (v in remaining) mixed_data[[v]] <- lavaan::HolzingerSwineford1939[[v]]
    result <- .post_check_compare("f =~ x1+x2+x3+x4", mixed_data,
        ordered = ordered_names, estimator = "DWLS")
    expect_false(result$fit$fitting$post_check$theta_not_pd)
  }
})
