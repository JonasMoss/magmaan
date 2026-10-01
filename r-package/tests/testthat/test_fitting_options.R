# Pin the behavioral contract rather than silently following installed versions.
.fitting_oracle <- function() {
  skip_if_not_installed("lavaan")
  skip_if(as.character(utils::packageVersion("lavaan")) != "0.7.2",
          "versioned fitting oracle requires lavaan 0.7.2")
}
.fitting_keys <- function(pt) paste(pt$lhs, pt$op, pt$rhs, pt$group, sep = "\r")
.fitting_match <- function(fit, oracle, column) {
  mp <- fit$partable
  lp <- lavaan::parTable(oracle)
  free <- mp$free > 0L
  lp[[column]][match(.fitting_keys(mp[free, ]), .fitting_keys(lp))]
}

test_that("versioned starts and complete ML fits agree with lavaan", {
  .fitting_oracle()
  d <- lavaan::HolzingerSwineford1939
  cases <- list(
    list(model = "f =~ x1+x2+x3+x4", std = FALSE, fixed = FALSE),
    list(model = "f =~ x1+x2+x3+x4", std = TRUE, fixed = FALSE),
    list(model = "x4 ~ x1+x2+x3", std = FALSE, fixed = FALSE),
    list(model = "x4 ~ x1+x2+x3", std = FALSE, fixed = TRUE),
    list(model = "f =~ x1+x2+x3\ng =~ x4+x5+x6\ng ~ f", std = FALSE, fixed = FALSE),
    list(model = "f =~ NA*x1+1*x2+x3+x4", std = FALSE, fixed = FALSE),
    list(model = "f =~ x1\nx2 ~ f", std = FALSE, fixed = FALSE),
    list(model = "f1 =~ x1+x2+x3\nf2 =~ x4+x5+x6\nf3 =~ x7+x8+x9\ng =~ f1+f2+f3", std = FALSE, fixed = FALSE),
    list(model = "x1 ~ x2\nx2 ~ x3", std = FALSE, fixed = FALSE))
  for (case in cases) {
    fit <- suppressWarnings(fit_model(case$model, d, meanstructure = TRUE,
        std_lv = case$std, fixed_x = case$fixed,
        options = list(preset = "lavaan-0.7.2")))
    lv <- suppressWarnings(lavaan::sem(case$model, d, meanstructure = TRUE,
        std.lv = case$std, fixed.x = case$fixed))
    expect_equal(as.numeric(fit$fitting$attempts[[1]]$start),
        as.numeric(.fitting_match(fit, lv, "start")), tolerance = 1e-9)
    expect_equal(fit$converged, lavaan::lavInspect(lv, "converged"))
    expect_equal(fit$verdict$status, if (fit$converged) "passed" else "failed")
    expect_equal(as.numeric(fit$theta), as.numeric(.fitting_match(fit, lv, "est")), tolerance = 1e-5)
    expect_equal(fit$fmin, as.numeric(lavaan::fitMeasures(lv, "fmin")), tolerance = 1e-9)
    expect_false(fit$sample_normalized)
    expect_true(isTRUE(fit$diagnostics$newton_accuracy$checked))
    attempt <- fit$fitting$attempts[[fit$fitting$selected_attempt]]
    expect_true(attempt$raw_status %in% 3:6)
    expect_equal(attempt$gradient_max, max(abs(lv@optim$dx)), tolerance = 1e-6)
  }
})

test_that("grouped fitting retains the same native starts and estimates", {
  .fitting_oracle()
  d <- lavaan::HolzingerSwineford1939
  m <- "f =~ NA*x1+1*x2+x3+x4"
  fit <- fit_model(m, d, groups = "school", meanstructure = TRUE,
      options = list(preset = "lavaan-0.7.2"))
  lv <- lavaan::sem(m, d, group = "school", meanstructure = TRUE)
  expect_equal(as.numeric(fit$fitting$attempts[[1]]$start), as.numeric(.fitting_match(fit, lv, "start")), tolerance = 1e-9)
  expect_equal(as.numeric(fit$theta), as.numeric(.fitting_match(fit, lv, "est")), tolerance = 1e-5)
  expect_equal(fit$converged, lavaan::lavInspect(lv, "converged"))
})

test_that("lavaan retries retain rejection of the original invalid covariance", {
  .fitting_oracle()
  d <- lavaan::HolzingerSwineford1939
  model <- "f =~ x1+x2+x3+x4"
  spec <- model_spec(model, meanstructure = TRUE)
  n <- max(spec$partable$free)
  fit <- suppressWarnings(fit_model(spec, d, control = list(start = rep(0, n)),
      options = list(preset = "lavaan-0.7.2")))
  lv <- suppressWarnings(lavaan::sem(model, d, meanstructure = TRUE, start = rep(0, n)))
  expect_equal(length(fit$fitting$attempts), 4L)
  expect_equal(fit$fitting$selected_attempt, 4L)
  expect_true(fit$fitting$attempts[[3]]$simple_start)
  expect_equal(fit$converged, lavaan::lavInspect(lv, "converged"))
  expect_equal(as.numeric(fit$theta), as.numeric(.fitting_match(fit, lv, "est")), tolerance = 1e-5)
  expect_equal(as.numeric(fit$fitting$attempts[[4]]$optimizer_start), as.numeric(lv@optim$x), tolerance = 1e-12)
  expect_true(is.na(fit$fmin))
  expect_true(all(vapply(fit$fitting$attempts, function(a) !a$accepted && a$iterations == 0L, logical(1))))
  expect_true(all(vapply(fit$fitting$attempts, function(a) is.na(a$fmin), logical(1))))
})

test_that("components resolve separately and refits preserve the setup", {
  .fitting_oracle()
  d <- lavaan::HolzingerSwineford1939
  options <- list(preset = "lavaan-0.7.2", convergence = "newton")
  fit <- fit_model("f =~ x1+x2+x3+x4", d, options = options)
  expect_true(fit$fitting$modified_preset)
  expect_equal(fit$fitting$effective$convergence, "newton")
  expect_equal(fit$verdict$status, fit$diagnostics$verdict$status)
  refit <- getFromNamespace(".route_refit_fun", "magmaanlab")(fit)(fit$model, d)
  expect_equal(refit$fitting$effective, fit$fitting$effective)
  expect_equal(refit$theta, fit$theta, tolerance = 1e-10)
  expect_equal(fit$fitting$attempts[[1]]$controls$port$max_eval, 20000L)
  expect_equal(fit$fitting$attempts[[1]]$controls$port$x_tol, 1.5e-8)
  native <- fit_model("f =~ x1+x2+x3+x4", d,
      options = list(starts = "lavaan-0.7.2", optimizer = "port", convergence = "newton"))
  expect_equal(native$fitting$effective$optimizer, "port")
  expect_true(native$sample_normalized)
})

test_that("fully fixed models retain the selected acceptance convention", {
  .fitting_oracle()
  d <- lavaan::HolzingerSwineford1939
  m <- "x1 ~~ 1*x1\nx2 ~~ 1*x2\nx1 ~~ 0*x2"
  fit <- fit_model(m, d, options = list(preset = "lavaan-0.7.2"))
  lv <- lavaan::sem(m, d, se = "none", test = "none")
  expect_equal(fit$converged, lavaan::lavInspect(lv, "converged"))
  expect_equal(fit$fmin, lv@optim$fx, tolerance = 1e-9)
  expect_length(fit$fitting$attempts, 0L)
  expect_true(is.na(fit$fitting$selected_attempt))
  native <- fit_model(m, d, options = list(preset = "lavaan-0.7.2", convergence = "newton"))
  expect_true(native$converged)
})

test_that("zero variance bounds use lavaan's exact bound-masked gradient", {
  .fitting_oracle()
  s <- matrix(c(1, .7, .7, .7, 1, .3, .7, .3, 1), 3)
  dimnames(s) <- list(paste0("x", 1:3), paste0("x", 1:3))
  m <- "f =~ x1+x2+x3"
  ss <- structure(list(S = list(s), nobs = 200L), class = c("magmaan_data", "list"))
  fit <- getFromNamespace("fit_ml", "magmaanlab")(model_spec(m), ss,
      bounds = "variance", options = list(preset = "lavaan-0.7.2"))
  lv <- lavaan::sem(m, sample.cov = s, sample.nobs = 200,
      sample.cov.rescale = FALSE, bounds = "pos.var")
  expect_equal(as.numeric(fit$theta), as.numeric(.fitting_match(fit, lv, "est")), tolerance = 1e-5)
  expect_equal(fit$fmin, lv@optim$fx, tolerance = 1e-9)
  expect_equal(fit$converged, lavaan::lavInspect(lv, "converged"))
  lp <- lavaan::parTable(lv)
  lp <- lp[lp$free > 0L, ]
  active <- lp$est == lp$lower | lp$est == lp$upper
  expect_true(any(active))
  expect_gt(max(abs(lv@optim$dx)), .001)
  expect_equal(fit$fitting$attempts[[fit$fitting$selected_attempt]]$gradient_max,
      max(abs(lv@optim$dx[!active])), tolerance = 1e-6)
})

test_that("fitting options reject unknown versions, conflicts and unsupported sources", {
  d <- data.frame(x1 = 1:10, x2 = c(2,4,1,5,3,7,9,6,10,8), x3 = c(7,1,3,2,6,4,8,10,9,5))
  m <- "f =~ x1+x2+x3"
  expect_error(fit_model(m, d, options = list(preset = "lavaan")), "supported fitting preset")
  expect_error(fit_model(m, d, options = list(preset = "lavaan-0.7.3")), "supported fitting preset")
  expect_error(fit_model(m, d, options = list(convergence = "magmaan")), "convergence must be")
  expect_error(fit_model(m, d, options = list(convergence = "lavaan-0.7.2")), "requires the PORT")
  expect_error(fit_model(m, d, options = list(wut = "newton")), "unknown fitting option")
  expect_error(fit_model(m, d, options = list(preset = "lavaan-0.7.2"), optimizer = "port"), "conflicts")
  expect_error(fit_model(m, d, options = list(preset = "lavaan-0.7.2"), control = list(start = "fabin3")), "constructor conflicts")
  expect_error(fit_model(m, d, options = list(preset = "lavaan-0.7.2"), psd = TRUE), "ordinary complete")
  expect_error(fit_model(m, d, options = list(preset = "lavaan-0.7.2"), estimator = "FIML"), "ordinary complete")
  expect_error(fit_model(m, d, options = list(preset = "lavaan-0.7.2"), missing = "pairwise"), "ordinary complete")
  expect_error(fit_model("f =~ x1 + a*x2 + a*x3", d, options = list(preset = "lavaan-0.7.2")), "constraints")
  expect_error(fit_model("f =~ x1 + a*x2 + a*x3", d,
      options = list(optimizer = "port", convergence = "lavaan-0.7.2")), "constraints")
})
