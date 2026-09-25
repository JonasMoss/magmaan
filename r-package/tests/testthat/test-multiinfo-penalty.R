heywood_stats <- function(n = 50L) {
  S <- matrix(c(1.0, 0.9, 0.6,
                0.9, 1.0, 0.5,
                0.6, 0.5, 1.0), 3, 3)
  nm <- c("x1", "x2", "x3")
  dimnames(S) <- list(nm, nm)
  list(S = list(S), nobs = n)
}

row_est <- function(fit, lhs, op, rhs) {
  pt <- fit$partable
  pt$est[pt$lhs == lhs & pt$op == op & pt$rhs == rhs]
}

test_that("multi-information penalty pulls a Heywood case inside", {
  model <- model_spec("f =~ x1 + x2 + x3")
  ml <- magmaan_core$fit_ml(model, heywood_stats())
  expect_lt(row_est(ml, "x1", "~~", "x1"), 0)

  fit <- frontier_fit_ml_multiinfo(model, heywood_stats())
  expect_true(fit$converged)
  expect_true(fit$diagnostics$admissibility$admissible)
  expect_gt(row_est(fit, "x1", "~~", "x1"), 0)
  expect_equal(fit$penalty$type, "multiinfo")
  expect_equal(fit$penalty$weight, 0.25)
  expect_true(fit$penalty$recursive)
  expect_lte(fit$penalty$value, 0)
  # fmin is the unpenalized criterion; the penalized one adds -(lambda/N) P.
  expect_gte(fit$fmin, ml$fmin)
  expect_equal(fit$penalty$penalized_fmin,
               fit$fmin - fit$penalty$weight / 50 * fit$penalty$value)
  terms <- fit$penalty$terms
  expect_setequal(terms$variable, c("f", "x1", "x2", "x3"))
  expect_equal(sum(terms$log_one_minus_r2) + fit$penalty$residual_log_det_corr,
               fit$penalty$value, tolerance = 1e-10)

  heavy <- frontier_fit_ml_multiinfo(model, heywood_stats(), eta = 3)
  expect_equal(heavy$penalty$weight, 2)
  expect_gt(row_est(heavy, "x1", "~~", "x1"), row_est(fit, "x1", "~~", "x1"))
})

test_that("fixed.x covariates stay in the barrier set", {
  set.seed(11)
  n <- 120
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  dat <- data.frame(y = 0.6 * x1 + 0.3 * x2 + rnorm(n, sd = 0.5), x1, x2)
  fit <- frontier_fit_ml_multiinfo("y ~ x1 + x2", dat)
  expect_true(fit$converged)
  terms <- fit$penalty$terms
  expect_setequal(terms$variable, c("y", "x1", "x2"))
  expect_true(all(terms$kind == "latent"))
  y_r2 <- terms$r2[terms$variable == "y"]
  expect_gt(y_r2, 0.5)
  expect_lt(y_r2, 1)
})

test_that("penalized FIML equals penalized ML on complete data", {
  set.seed(12)
  n <- 150
  f <- rnorm(n)
  dat <- data.frame(x1 = 0.8 * f + rnorm(n, sd = 0.6),
                    x2 = 0.7 * f + rnorm(n, sd = 0.7),
                    x3 = 0.6 * f + rnorm(n, sd = 0.8),
                    x4 = 0.5 * f + rnorm(n, sd = 0.9))
  model <- model_spec("f =~ x1 + x2 + x3 + x4", meanstructure = TRUE)
  ml <- frontier_fit_ml_multiinfo(model, dat)
  fiml <- frontier_fit_fiml_multiinfo(model, dat)
  expect_true(fiml$converged)
  expect_equal(fiml$penalty$value, ml$penalty$value, tolerance = 1e-5)
  expect_equal(fiml$theta, ml$theta, tolerance = 1e-4)
})

test_that("nonrecursive models warn but fit", {
  set.seed(13)
  n <- 300
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  e1 <- rnorm(n, sd = 0.7)
  e2 <- rnorm(n, sd = 0.7)
  # y1 = .3 y2 + .5 x1 + e1, y2 = .2 y1 + .5 x2 + e2 solved jointly.
  y1 <- (0.5 * x1 + e1 + 0.3 * (0.5 * x2 + e2)) / (1 - 0.06)
  y2 <- 0.2 * y1 + 0.5 * x2 + e2
  dat <- data.frame(y1, y2, x1, x2)
  expect_warning(
    fit <- frontier_fit_ml_multiinfo("y1 ~ y2 + x1\ny2 ~ y1 + x2", dat),
    "nonrecursive")
  expect_false(fit$penalty$recursive)
  expect_lte(fit$penalty$value, 0)
})

test_that("determinacy penalty pulls a Heywood case inside and reports its parts", {
  model <- model_spec("f =~ x1 + x2 + x3")
  fit <- frontier_fit_ml_multiinfo(model, heywood_stats(), target = "determinacy")
  expect_true(fit$converged)
  expect_true(fit$diagnostics$admissibility$admissible)
  expect_gt(row_est(fit, "x1", "~~", "x1"), 0)
  expect_equal(fit$penalty$type, "determinacy")
  expect_lte(fit$penalty$value, 0)
  terms <- fit$penalty$terms
  expect_equal(terms$variable, "f")
  expect_equal(sum(terms$log_one_minus_rho2) + fit$penalty$residual_log_det_corr,
               fit$penalty$value, tolerance = 1e-10)
  expect_equal(-2 * (fit$penalty$total_correlation + fit$penalty$mutual_information),
               fit$penalty$block_value, tolerance = 1e-10)
  # One factor: log det Q = log(1 - rho^2) = -log(1 + sum lambda^2 phi / theta).
  pt <- fit$partable
  lam <- pt$est[pt$op == "=~"]
  th <- sapply(c("x1", "x2", "x3"), function(v) row_est(fit, v, "~~", v))
  phi <- row_est(fit, "f", "~~", "f")
  expect_equal(fit$penalty$value, -log1p(phi * sum(lam^2 / th)), tolerance = 1e-8)
})

test_that("determinacy penalty leaves manifest models at ordinary ML", {
  set.seed(11)
  n <- 120
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  dat <- data.frame(y = 0.6 * x1 + 0.3 * x2 + rnorm(n, sd = 0.5), x1, x2)
  fit <- frontier_fit_ml_multiinfo("y ~ x1 + x2", dat, target = "determinacy")
  ml <- magmaan_core$fit_ml(model_spec("y ~ x1 + x2"), df_to_data(dat, model_spec("y ~ x1 + x2")))
  expect_true(fit$converged)
  expect_equal(fit$penalty$value, 0)
  expect_equal(nrow(fit$penalty$terms), 0L)
  expect_equal(fit$theta, ml$theta, tolerance = 1e-6)
})

test_that("determinacy penalty: marker and std.lv agree, FIML equals ML", {
  set.seed(12)
  n <- 150
  f <- rnorm(n)
  g <- 0.5 * f + rnorm(n, sd = 0.8)
  dat <- data.frame(x1 = 0.8 * f + rnorm(n, sd = 0.6),
                    x2 = 0.7 * f + rnorm(n, sd = 0.7),
                    x3 = 0.6 * f + rnorm(n, sd = 0.8),
                    x4 = 0.8 * g + rnorm(n, sd = 0.6),
                    x5 = 0.7 * g + rnorm(n, sd = 0.7),
                    x6 = 0.6 * g + rnorm(n, sd = 0.8))
  syntax <- "f =~ x1 + x2 + x3\ng =~ x4 + x5 + x6"
  marker <- frontier_fit_ml_multiinfo(syntax, dat, target = "determinacy")
  stdlv <- frontier_fit_ml_multiinfo(model_spec(syntax, std_lv = TRUE), dat,
                                     target = "determinacy")
  expect_equal(marker$penalty$value, stdlv$penalty$value, tolerance = 1e-6)
  expect_equal(marker$fmin, stdlv$fmin, tolerance = 1e-6)
  model <- model_spec(syntax, meanstructure = TRUE)
  ml <- frontier_fit_ml_multiinfo(model, dat, target = "determinacy")
  fiml <- frontier_fit_fiml_multiinfo(model, dat, target = "determinacy")
  expect_true(fiml$converged)
  expect_equal(fiml$penalty$value, ml$penalty$value, tolerance = 1e-5)
  expect_equal(fiml$theta, ml$theta, tolerance = 1e-4)
  joint <- frontier_fit_ml_multiinfo(syntax, dat)
  expect_equal(joint$penalty$type, "multiinfo")
})
