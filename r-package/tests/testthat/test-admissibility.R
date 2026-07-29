test_that("complete-data fits expose structured covariance admissibility", {
  set.seed(17)
  n <- 120L
  eta <- rnorm(n)
  dat <- data.frame(
    x1 = eta + rnorm(n, sd = 0.6),
    x2 = 0.8 * eta + rnorm(n, sd = 0.7),
    x3 = 0.7 * eta + rnorm(n, sd = 0.8)
  )

  fit <- expect_no_warning(magmaan("f =~ x1 + x2 + x3", dat,
                                   estimator = "ML"))
  a <- fit$diagnostics$admissibility

  expect_true(a$checked)
  expect_true(a$covariance_matrices_psd)
  expect_true(a$implied_sigma_pd)
  expect_true(a$admissible)
  expect_length(a$theta, 1L)
  expect_length(a$psi, 1L)
  expect_true(a$theta[[1L]]$psd)
  expect_true(a$psi[[1L]]$psd)
  expect_true(all(a$theta[[1L]]$covariance_rows >= 1L))
})

test_that("Heywood solution warns once without changing convergence", {
  S <- matrix(c(
    1.0, 0.7, 0.7,
    0.7, 1.0, 0.3,
    0.7, 0.3, 1.0
  ), 3L, 3L, byrow = TRUE)
  n <- 8L
  H <- stats::contr.helmert(n)[, 1:3, drop = FALSE]
  Q <- sweep(H, 2L, sqrt(colSums(H^2)), "/")
  X <- sqrt(n) * Q %*% chol(S)
  dat <- as.data.frame(X)
  names(dat) <- c("x1", "x2", "x3")

  fit <- NULL
  expect_warning(
    fit <- magmaan("f =~ x1 + x2 + x3", dat, estimator = "ML"),
    "not covariance-admissible"
  )

  expect_true(fit$converged)
  expect_false(fit$diagnostics$admissibility$admissible)
  expect_true(fit$diagnostics$admissibility$implied_sigma_pd)
  theta <- fit$diagnostics$admissibility$theta[[1L]]
  expect_false(theta$psd)
  expect_length(theta$negative_variance_rows, 1L)
  expect_lt(fit$partable$est[theta$negative_variance_rows], 0)

  printed <- paste(capture.output(print(fit)), collapse = "\n")
  expect_match(printed, "covariance-admissible: FALSE")
})

test_that("FIML fits run the same covariance admissibility audit", {
  set.seed(23)
  n <- 100L
  eta <- rnorm(n)
  dat <- data.frame(
    x1 = eta + rnorm(n, sd = 0.5),
    x2 = 0.8 * eta + rnorm(n, sd = 0.6),
    x3 = 0.6 * eta + rnorm(n, sd = 0.7)
  )
  dat$x2[seq(5L, n, by = 11L)] <- NA_real_

  fit <- expect_no_warning(magmaan("f =~ x1 + x2 + x3", dat,
                                   estimator = "FIML"))
  expect_true(fit$diagnostics$admissibility$checked)
  expect_true(fit$diagnostics$admissibility$admissible)
})
