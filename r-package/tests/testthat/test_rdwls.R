rdwls_data <- function(mixed = TRUE) {
  d <- lavaan::HolzingerSwineford1939[paste0("x", 1:6)]
  for (j in seq_len(if (mixed) 3L else 6L)) d[[j]] <- cut(d[[j]], 3, labels = FALSE)
  d
}

test_that("RDWLS agrees with lavaan DWLS supplied exact weights", {
  skip_if_not_installed("lavaan")
  for (mixed in c(FALSE, TRUE)) for (parameterization in c("delta", "theta")) {
    d <- rdwls_data(mixed)
    ordered <- paste0("x", seq_len(if (mixed) 3L else 6L))
    fit <- fit_model("f =~ x1+x2+x3+x4+x5+x6", d, estimator = "RDWLS",
                     ordered = ordered, parameterization = parameterization)
    expect_true(fit$converged)
    expect_identical(fit$estimator, "RDWLS")
    stats <- if (mixed) fit$mixed_ordinal_stats else fit$ordinal_stats
    expect_identical(stats$gamma_source, "exact")
    rows <- stats$moment_influence[[1]]
    expect_equal(stats$NACOV[[1]], crossprod(rows)/nrow(rows), tolerance = 1e-12)
    oracle <- lavaan::cfa("f =~ x1+x2+x3+x4+x5+x6", d, ordered = ordered,
      parameterization = parameterization, estimator = "DWLS", se = "none", test = "none",
      WLS.V = list(stats$W_dwls[[1]]))
    pt <- lavaan::parameterEstimates(oracle)
    own <- fit$partable
    key <- function(x) paste(x$lhs, x$op, x$rhs)
    idx <- match(key(own), key(pt))
    keep <- own$free > 0 & !is.na(idx)
    gap <- max(abs(own$est[keep]-pt$est[idx[keep]]))
    message("RDWLS lavaan point maximum difference: ", format(gap, digits = 10))
    expect_lt(gap, 1e-5)
    model <- prepare_model(model_spec("f =~ x1+x2+x3+x4+x5+x6", ordered = ordered,
                         parameterization = parameterization), prototype = d)
    data <- prepare_data(model, d)
    weight <- prepare_weight(data, "RDWLS", full = FALSE)
    staged <- estimate(model, data, weight = weight)
    expect_equal(staged$theta, fit$theta, tolerance = 1e-7)
    expect_identical(weight$stats$gamma_source, "exact")
    policy <- policy_inference(fit)
    ij <- if (mixed) infer_mixed_ordinal_robust_ij(fit, stats) else infer_ordinal_robust_ij(fit, stats)
    expect_equal(policy$covariance, ij$vcov, tolerance = 1e-12, ignore_attr = TRUE)
    if (!mixed) expect_equal(ij$vcov, infer_ordinal_robust_ij(fit, stats, first_stage = "exact")$vcov, tolerance = 1e-12)
    compatibility <- convention_inference(fit, "WLSMV")
    expect_identical(compatibility$covariance_reason, "unsupported_model")
  }
})

test_that("RDWLS refuses missing and pairwise observations", {
  skip_if_not_installed("lavaan")
  d <- rdwls_data(FALSE)
  expect_error(fit_model("f =~ x1+x2+x3+x4+x5+x6", d, estimator = "RDWLS",
    ordered = names(d), missing = "pairwise"), "RDWLS")
  expect_error(fit_model("f =~ x1+x2+x3+x4+x5+x6", d, estimator = "RDWLS",
    ordered = names(d), options = list(preset = "lavaan-0.7.2")), "RDWLS")
  d[1,1] <- NA
  expect_error(magmaan_core$data_ordinal_stats_from_raw(list(as.matrix(d)), gamma = "exact"), "RDWLS")
  expect_error(fit_model("f =~ x1+x2+x3+x4+x5+x6", d, estimator = "RDWLS", ordered = names(d)), "RDWLS")
})

test_that("RDWLS policy covariance reconstructs the observed sandwich", {
  skip_if_not_installed("lavaan")
  fit <- fit_model("f =~ x1+x2+x3+x4+x5+x6", rdwls_data(), estimator = "RDWLS",
                   ordered = paste0("x", 1:3))
  stats <- fit$mixed_ordinal_stats
  gradient <- function(f) as.numeric(evaluate_mixed_ordinal_at_impl(f)$gradient)
  h <- 1e-5
  k <- length(fit$theta)
  m <- ncol(stats$moment_influence[[1]])
  H <- vapply(seq_len(k), function(j) {
    plus <- minus <- fit
    plus$theta[j] <- plus$theta[j] + h
    minus$theta[j] <- minus$theta[j] - h
    (gradient(plus)-gradient(minus))/(2*h)
  }, numeric(k))
  moment_gradient <- weight_gradient <- matrix(0, k, m)
  for (j in seq_len(m)) {
    plus <- minus <- fit
    plus$mixed_ordinal_stats$moments[[1]][j] <- stats$moments[[1]][j]+h
    minus$mixed_ordinal_stats$moments[[1]][j] <- stats$moments[[1]][j]-h
    moment_gradient[,j] <- (gradient(plus)-gradient(minus))/(2*h)
    plus <- minus <- fit
    gamma <- stats$NACOV[[1]][j,j]
    plus$mixed_ordinal_stats$W_dwls[[1]][j,j] <- 1/(gamma+h)
    minus$mixed_ordinal_stats$W_dwls[[1]][j,j] <- 1/(gamma-h)
    weight_gradient[,j] <- (gradient(plus)-gradient(minus))/(2*h)
  }
  # Differentiating the score gives both channels of the empirical IJ meat;
  # the Hessian above is observed, including the fitted residual curvature.
  score <- stats$moment_influence[[1]] %*% t(moment_gradient) +
    stats$gamma_diag_influence[[1]] %*% t(weight_gradient)
  influence <- score %*% t(solve(H))
  explicit <- crossprod(influence)/nrow(score)^2
  policy <- policy_inference(fit)$covariance
  gap <- norm(policy-explicit, "F")/norm(explicit, "F")
  message("RDWLS explicit sandwich relative difference: ", format(gap, digits = 10))
  expect_lt(gap, 1e-5)
})

test_that("RDWLS grouped constrained fits retain policy global and nested inference", {
  skip_if_not_installed("lavaan")
  d <- rdwls_data()
  d$school <- lavaan::HolzingerSwineford1939$school
  syntax <- "f =~ x1 + a*x2 + b*x3\ng =~ x4+x5+x6\ng ~ f"
  for (parameterization in c("delta", "theta")) {
    args <- list(ordered = paste0("x", 1:3), group = "school",
      group_labels = unique(as.character(d$school)), parameterization = parameterization,
      fixed_x = FALSE, auto_cov_y = TRUE)
    alternative <- do.call(model_spec, c(list(syntax), args))
    null <- do.call(model_spec, c(list(paste(syntax, "a == b", sep = "\n")), args))
    f1 <- fit_model(alternative, d, estimator = "RDWLS")
    f0 <- fit_model(null, d, estimator = "RDWLS")
    expect_true(f1$converged)
    expect_true(f0$converged)
    global <- policy_inference(f1)
    expect_true(global$covariance_available, info = global$covariance_detail)
    expect_true(global$score$available, info = global$score$detail)
    nested <- policy_nested(f1, f0)
    expect_true(nested$lr$available, info = nested$lr$detail)
    expect_true(is.finite(nested$lr$p_all))
    indices <- policy_fit_measures(f1)
    expect_true(all(is.na(indices$reason)))
  }
})
