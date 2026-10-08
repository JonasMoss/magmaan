association_ml_data <- function(n = 350L) {
  set.seed(261001L)
  loading <- c(.75, .65, .55, .45)
  factor <- rnorm(n)
  x <- sapply(loading, function(l) l * factor + sqrt(1 - l^2) * rnorm(n))
  out <- as.data.frame(apply(x, 2L, function(v)
    ordered(cut(v, c(-Inf, -.5, .4, Inf), labels = FALSE))))
  names(out) <- paste0("x", 1:4)
  out
}

test_that("ML shares staged and convenience ordinal association fitting", {
  dat <- association_ml_data()
  spec <- model_spec("f =~ x1 + x2 + x3 + x4", ordered = names(dat))
  stats <- magmaan_core$data_ordinal_stats_from_df(dat, spec)
  original <- stats[c("R", "thresholds")]
  fit <- fit_model(spec, stats, estimator = "ML")
  raw <- fit_model(spec, dat, estimator = "ML")
  prepared <- prepare_model(spec, prototype = dat)
  staged <- estimate(prepared, prepare_data(prepared, dat), estimator = "ML")
  direct <- magmaan_core$fit_ml(spec, stats)
  expect_identical(fit$estimator, "ML")
  expect_true(fit$ordinal)
  expect_identical(fit$composition$moment_source, "polychoric")
  expect_identical(fit$composition$moment_target, "correlation")
  expect_identical(fit$composition$thresholds, "saturated_stage1")
  expect_identical(fit$composition$inference, "not_validated")
  expect_identical(fit$npar_active, 4L)
  expect_identical(fit$association$rank, 4L)
  expect_identical(fit$df, 2L)
  for (other in list(raw, staged, direct)) {
    expect_equal(other$fmin, fit$fmin, tolerance = 1e-8)
    expect_equal(other$theta, fit$theta, tolerance = 1e-5)
    expect_identical(other$association, fit$association)
  }
  expect_equal(stats[c("R", "thresholds")], original, tolerance = 0)
  expect_equal(fit$partable$est[fit$partable$op == "|"],
               unlist(stats$thresholds, use.names = FALSE), tolerance = 0)
  implied <- magmaan_core$model_implied(fit)$sigma[[1L]]
  expect_equal(diag(implied), rep(1, 4), tolerance = 0)
  reference <- .5 * (as.numeric(determinant(implied, logarithm = TRUE)$modulus) -
    as.numeric(determinant(stats$R[[1L]], logarithm = TRUE)$modulus) +
    sum(diag(stats$R[[1L]] %*% solve(implied))) - 4)
  expect_equal(fit$fmin, reference, tolerance = 1e-10)
  expect_error(vcov(fit), "sampling/inference contract")
  expect_error(fit_measures(fit), "sampling/inference contract")
})

test_that("ordinal ML retains PSD composition on refit and preserves delta/theta semantics", {
  dat <- association_ml_data()
  values <- numeric()
  for (parameterization in c("delta", "theta")) {
    spec <- model_spec("f =~ x1 + x2 + x3 + x4", ordered = names(dat),
                       parameterization = parameterization)
    stats <- magmaan_core$data_ordinal_stats_from_df(dat, spec)
    ordinary <- fit_model(spec, stats, estimator = "ML")
    fit <- fit_model(spec, stats, estimator = "ML", psd = TRUE,
                     control = list(max_iter = 5000L, gtol = 1e-8))
    refit <- getFromNamespace(".route_refit_fun", "magmaanlab")(fit)(spec, stats)
    expect_true(fit$converged)
    expect_true(fit$audit$constrained)
    expect_identical(fit$estimator, "ML")
    expect_identical(fit$covariance_policy, "psd")
    expect_identical(fit$composition$covariance_domain, "psd")
    expect_identical(fit$parameterization, parameterization)
    expect_identical(fit$options$route$fitter, "fit_model")
    expect_identical(refit$composition, fit$composition)
    expect_equal(refit$theta, fit$theta, tolerance = 1e-7)
    expect_equal(fit$fmin, ordinary$fmin, tolerance = 1e-7)
    expect_equal(fit$polychoric, stats$R, tolerance = 0)
    values <- c(values, fit$fmin)
  }
  expect_equal(values[1L], values[2L], tolerance = 1e-9)
  expect_false("frontier_fit_catml_psd" %in% getNamespaceExports("magmaanlab"))
})

test_that("ordinal ML supports loading equalities and rejects inactive constraints", {
  dat <- association_ml_data(700L)
  dat$group <- rep(c("a", "b"), each = 350L)
  ordered <- paste0("x", 1:4)
  syntax <- "f =~ x1 + x2 + x3 + x4"
  spec <- model_spec(syntax, ordered = ordered, group = "group",
                     group_labels = c("a", "b"), group_equal = "loadings")
  fit <- fit_model(spec, dat, estimator = "ML")
  expect_identical(fit$association$n_coordinates, 5L)
  expect_identical(fit$association$rank, 5L)
  expect_identical(fit$df, 7L)
  threshold_spec <- model_spec(syntax, ordered = ordered, group = "group",
                               group_labels = c("a", "b"), group_equal = "thresholds")
  expect_error(fit_model(threshold_spec, dat, estimator = "ML"), "saturated thresholds")
  for (extra in c("x1 ~ 1", "x1 ~~ NA*x1", "x1 ~~ 2*x1", "x1 ~*~ NA*x1",
                  "x1 | 0*t1 + t2")) {
    bad <- model_spec(paste(syntax, extra, sep = "\n"), ordered = ordered)
    expect_error(fit_model(bad, dat[ordered], estimator = "ML"),
                 "means|mean/intercept|response|saturated")
    prepared <- prepare_model(bad, prototype = dat[ordered])
    if (identical(extra, "x1 | 0*t1 + t2")) {
      report <- structural_identification(prepared)
      expect_identical(report$status, "unchecked")
      expect_identical(report$reason, "unsupported_model")
    }
    expect_error(estimate(prepared, prepare_data(prepared, dat[ordered]), estimator = "ML"),
                 "means|mean/intercept|response|saturated")
  }
  mixed <- model_spec(syntax, ordered = ordered[1:3])
  mixed_data <- dat[ordered]
  mixed_data$x4 <- as.numeric(mixed_data$x4)
  expect_error(fit_model(mixed, mixed_data, estimator = "ML"), "all-ordinal")
})
