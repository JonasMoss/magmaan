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

test_that("frontier PSD ML returns an ordinary admissible R fit", {
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

  fit <- expect_no_warning(frontier_fit_ml_psd(
    "f =~ x1 + x2 + x3", dat,
    control = list(max_iter = 5000L, gtol = 1e-8)
  ))

  expect_type(fit, "list")
  expect_true(fit$converged)
  expect_true(fit$diagnostics$admissibility$admissible)
  expect_true(fit$diagnostics$admissibility$covariance_matrices_psd)
  expect_true(fit$audit$constrained)
  expect_true(fit$audit$stationary)
  expect_lte(fit$audit$constraint_violation_inf, 1e-6)
  expect_gte(fit$audit$constraint_jacobian_rank, 1L)
  expect_true(is.finite(fit$audit$raw_grad_inf_norm))
  expect_lte(fit$audit$grad_inf_norm, fit$audit$stationarity_rhs)
  expect_equal(nrow(fit$partable),
               nrow(model_spec("f =~ x1 + x2 + x3")$partable))
})

test_that("frontier PSD FIML repairs a missing-data Heywood solution", {
  S <- matrix(c(
    1.0, 0.7, 0.7,
    0.7, 1.0, 0.3,
    0.7, 0.3, 1.0
  ), 3L, 3L, byrow = TRUE)
  n <- 80L
  H <- stats::contr.helmert(n)[, 1:3, drop = FALSE]
  Q <- sweep(H, 2L, sqrt(colSums(H^2)), "/")
  X <- sqrt(n) * Q %*% chol(S)
  dat <- as.data.frame(X)
  names(dat) <- c("x1", "x2", "x3")
  dat$x1[1L] <- NA_real_

  fit <- expect_no_warning(frontier_fit_fiml_psd(
    "f =~ x1 + x2 + x3", dat,
    control = list(max_iter = 5000L, gtol = 1e-8)
  ))

  expect_true(fit$converged)
  expect_true(fit$fiml)
  expect_identical(fit$estimator, "FIML")
  expect_identical(fit$covariance_policy, "psd")
  expect_true(fit$diagnostics$admissibility$admissible)
  expect_true(fit$audit$constrained)
  expect_true(fit$audit$stationary)
  expect_lte(fit$audit$constraint_violation_inf, 1e-6)
  expect_true(anyNA(fit$raw_data$X[[1L]]))
})

test_that("frontier PSD ordinal fit preserves the Stage-1 polychorics", {
  set.seed(43)
  n <- 260L
  eta <- rnorm(n)
  latent <- cbind(
    eta + rnorm(n, sd = 0.65),
    0.8 * eta + rnorm(n, sd = 0.7),
    0.7 * eta + rnorm(n, sd = 0.75)
  )
  dat <- as.data.frame(apply(latent, 2L, function(x) {
    ordered(cut(x, c(-Inf, -0.5, 0.5, Inf), labels = FALSE))
  }))
  names(dat) <- c("x1", "x2", "x3")
  spec <- model_spec("f =~ x1 + x2 + x3", ordered = names(dat))
  stats <- magmaan_core$data_ordinal_stats_from_df(dat, spec)
  original_R <- stats$R

  fit <- expect_no_warning(frontier_fit_ordinal_psd(
    spec, stats, estimator = "DWLS",
    control = list(max_iter = 5000L, gtol = 1e-8)
  ))

  expect_true(fit$converged)
  expect_true(fit$ordinal)
  expect_identical(fit$estimator, "DWLS")
  expect_identical(fit$covariance_policy, "psd")
  expect_true(fit$diagnostics$admissibility$admissible)
  expect_true(fit$audit$constrained)
  expect_equal(unname(fit$polychoric[[1L]]),
               unname(original_R[[1L]]), tolerance = 0)
  expect_equal(stats$R, original_R, tolerance = 0)
})

test_that("frontier PSD mixed ordinal fit preserves Stage-1 moments", {
  set.seed(44)
  n <- 280L
  eta <- rnorm(n)
  latent_1 <- 0.85 * eta + rnorm(n, sd = 0.6)
  latent_2 <- 0.72 * eta + rnorm(n, sd = 0.7)
  dat <- data.frame(
    x1 = ordered(cut(latent_1, c(-Inf, -0.55, 0.45, Inf),
                     labels = FALSE)),
    x2 = ordered(cut(latent_2, c(-Inf, -0.45, 0.55, Inf),
                     labels = FALSE)),
    x3 = 0.78 * eta + rnorm(n, sd = 0.68) + 0.2,
    x4 = 0.62 * eta + rnorm(n, sd = 0.8) - 0.1
  )
  spec <- model_spec(
    "f =~ x1 + x2 + x3 + x4",
    ordered = c("x1", "x2"), meanstructure = TRUE
  )
  stats <- magmaan_core$data_mixed_ordinal_stats_from_df(dat, spec)
  original_R <- stats$R
  original_moments <- stats$moments

  fit <- expect_no_warning(frontier_fit_mixed_ordinal_psd(
    spec, stats, estimator = "DWLS",
    control = list(max_iter = 5000L, gtol = 1e-8)
  ))

  expect_true(fit$converged)
  expect_true(fit$mixed_ordinal)
  expect_identical(fit$estimator, "DWLS")
  expect_identical(fit$covariance_policy, "psd")
  expect_true(fit$diagnostics$admissibility$admissible)
  expect_true(fit$audit$constrained)
  expect_equal(lapply(fit$mixed_ordinal_stats$R, unname),
               lapply(original_R, unname), tolerance = 0)
  expect_equal(fit$mixed_ordinal_stats$moments,
               original_moments, tolerance = 0)
  expect_equal(stats$R, original_R, tolerance = 0)
  expect_equal(stats$moments, original_moments, tolerance = 0)
})

test_that("frontier PSD continuous LS wrappers preserve covariance admissibility", {
  set.seed(29)
  n <- 160L
  eta <- rnorm(n)
  dat <- data.frame(
    x1 = eta + rnorm(n, sd = 0.7),
    x2 = 0.8 * eta + rnorm(n, sd = 0.8),
    x3 = 0.65 * eta + rnorm(n, sd = 0.75)
  )
  model <- "f =~ x1 + x2 + x3"
  control <- list(max_iter = 5000L, gtol = 1e-8)

  fits <- list(
    ULS = frontier_fit_uls_psd(model, dat, control = control),
    GLS = frontier_fit_gls_psd(model, dat, control = control),
    WLS = frontier_fit_wls_psd(
      model, dat, W = diag(seq(0.7, 1.7, length.out = 6L)),
      control = control
    ),
    GMM_FITTED_WEIGHT = frontier_fit_gmm_fitted_weight_psd(
      model, dat, control = control
    )
  )
  for (estimator in names(fits)) {
    fit <- fits[[estimator]]
    expect_true(fit$converged, info = estimator)
    expect_true(fit$diagnostics$admissibility$admissible, info = estimator)
    expect_true(fit$audit$constrained, info = estimator)
    expect_true(fit$audit$stationary, info = estimator)
    expect_true(fit$audit$constraint_violation_inf <= 1e-6,
                info = estimator)
    expect_identical(fit$estimator, estimator)
  }
  expect_identical(
    fits$GMM_FITTED_WEIGHT$weight_policy,
    "expected_information_fixed_point"
  )
  expect_gt(fits$GMM_FITTED_WEIGHT$iterations, 1L)
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
