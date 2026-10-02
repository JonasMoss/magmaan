#!/usr/bin/env Rscript
# Reproduce the pinned affine-constraint scaling defect without magmaan.
# Run from the repository root with lavaan 0.7.2 installed.
stopifnot(as.character(utils::packageVersion("lavaan")) == "0.7.2")

lambda <- c(1, .8, .6, .9)
sample_cov <- tcrossprod(lambda) + diag(.7, 4)
dimnames(sample_cov) <- list(paste0("x", 1:4), paste0("x", 1:4))
model <- "f =~ x1+a*x2+b*x3+x4\na+b == 1.5"

fit_lavaan <- function(scale, syntax = model) {
  suppressWarnings(lavaan::sem(syntax, sample.cov = sample_cov,
      sample.nobs = 200, sample.cov.rescale = FALSE, meanstructure = FALSE,
      fixed.x = FALSE, se = "none", test = "none", optim.parscale = scale))
}

loading_values <- function(fit) {
  pt <- lavaan::parTable(fit)
  vapply(c("a", "b"), function(label) {
    pt$est[pt$op == "=~" & pt$label == label]
  }, numeric(1))
}

ordinary <- fit_lavaan("none")
scaled <- fit_lavaan("standardized")
ordinary_sum <- sum(loading_values(ordinary))
scaled_sum <- sum(loading_values(scaled))
stopifnot(lavaan::lavInspect(ordinary, "converged"),
    lavaan::lavInspect(scaled, "converged"),
    abs(ordinary_sum - 1.5) < 1e-10, abs(scaled_sum - 1.5) > .5)

# Independent constrained Gaussian ML reference. The parameterization enforces
# b=1.5-a exactly; positive residual/latent variances ensure a PD covariance.
# Neither its covariance, objective nor analytic gradient calls lavaan.
logdet_sample <- as.numeric(determinant(sample_cov, logarithm = TRUE)$modulus)
independent_components <- function(alpha, ratio = FALSE) {
  loading <- if (ratio) c(1, 2 * alpha[1], alpha[1], alpha[2]) else
      c(1, alpha[1], 1.5 - alpha[1], alpha[2])
  residual <- exp(alpha[3:6])
  latent_variance <- exp(alpha[7])
  covariance <- latent_variance * tcrossprod(loading) + diag(residual)
  factor <- chol(covariance)
  inverse <- chol2inv(factor)
  covariance_gradient <- (inverse - inverse %*% sample_cov %*% inverse) / 2
  loading_gradient <- as.numeric(covariance_gradient %*% loading)
  list(objective = (2 * sum(log(diag(factor))) + sum(sample_cov * inverse) -
      logdet_sample - 4) / 2,
      gradient = c(2 * latent_variance * if (ratio)
          (2 * loading_gradient[2] + loading_gradient[3]) else
          (loading_gradient[2] - loading_gradient[3]),
          2 * latent_variance * loading_gradient[4],
          diag(covariance_gradient) * residual,
          latent_variance * sum(loading * loading_gradient)))
}
independent <- stats::optim(c(.75, .9, rep(log(.7), 4), 0),
    function(alpha) independent_components(alpha)$objective,
    function(alpha) independent_components(alpha)$gradient,
    method = "BFGS", control = list(reltol = 1e-12, maxit = 10000))
independent_sum <- independent$par[1] + (1.5 - independent$par[1])
independent_gradient <- max(abs(independent_components(independent$par)$gradient))
stopifnot(independent$convergence == 0L, independent_gradient < 1e-7,
    abs(independent_sum - 1.5) < 1e-14,
    abs(independent$value - as.numeric(ordinary@optim$fx)) < 1e-10)

# Explain which equation the scaled oracle actually solved. z=D*theta and
# A*z=d imply A*D*theta=d, not the original A*theta=d.
pt <- lavaan::parTable(scaled)
theta <- numeric(max(pt$free))
free_rows <- pt$free > 0L
theta[pt$free[free_rows]] <- pt$est[free_rows]
jacobian <- scaled@Model@ceq.JAC
rhs <- jacobian %*% scaled@Model@eq.constraints.k0
original_residual <- max(abs(jacobian %*% theta - rhs))
scaled_residual <- max(abs(jacobian %*% (scaled@optim$parscale * theta) - rhs))
stopifnot(original_residual > .5, scaled_residual < 1e-10)

cat("lavaan:", as.character(utils::packageVersion("lavaan")), "\n")
cat(sprintf("ordinary: a+b=%.17g, fmin=%.17g\n", ordinary_sum, ordinary@optim$fx))
cat(sprintf("standardized: a+b=%.17g, fmin=%.17g\n", scaled_sum, scaled@optim$fx))
cat(sprintf("independent BFGS: a+b=%.17g, fmin=%.17g, max gradient=%.17g\n",
    independent_sum, independent$value, independent_gradient))
cat(sprintf("constraint residual: original=%.17g, scaled=%.17g\n",
    original_residual, scaled_residual))

# Zero RHS is insufficient: a=2*b also changes under the projected scale.
# An independent feasible parameterization is a=2*t, b=t.
ratio_model <- "f =~ x1+a*x2+b*x3+x4\na == 2*b"
ratio_ordinary <- fit_lavaan("none", ratio_model)
ratio_scaled <- fit_lavaan("standardized", ratio_model)
ratio_values <- loading_values(ratio_ordinary)
ratio_scaled_values <- loading_values(ratio_scaled)
ratio_residual <- ratio_scaled_values[1] - 2 * ratio_scaled_values[2]
stopifnot(lavaan::lavInspect(ratio_ordinary, "converged"),
    lavaan::lavInspect(ratio_scaled, "converged"),
    abs(ratio_values[1] - 2 * ratio_values[2]) < 1e-10,
    abs(ratio_residual) > .5)
ratio_independent <- stats::optim(c(.5, .9, rep(log(.7), 4), 0),
    function(alpha) independent_components(alpha, TRUE)$objective,
    function(alpha) independent_components(alpha, TRUE)$gradient,
    method = "BFGS", control = list(reltol = 1e-12, maxit = 10000))
ratio_gradient <- max(abs(independent_components(ratio_independent$par, TRUE)$gradient))
stopifnot(ratio_independent$convergence == 0L, ratio_gradient < 5e-7,
    abs(ratio_independent$value - as.numeric(ratio_ordinary@optim$fx)) < 1e-10)
basis <- ratio_scaled@Model@eq.constraints.K
transported_basis <- sweep(basis, 1, ratio_scaled@optim$parscale, "/")
transport_residual <- max(abs(ratio_scaled@Model@ceq.JAC %*% transported_basis))
stopifnot(transport_residual > .1)
cat(sprintf("homogeneous ordinary: a/b=%.17g, fmin=%.17g\n",
    ratio_values[1] / ratio_values[2], ratio_ordinary@optim$fx))
cat(sprintf("homogeneous standardized: a/b=%.17g, original residual=%.17g\n",
    ratio_scaled_values[1] / ratio_scaled_values[2], ratio_residual))
cat(sprintf("homogeneous independent BFGS: fmin=%.17g, max gradient=%.17g\n",
    ratio_independent$value, ratio_gradient))
cat(sprintf("homogeneous null-space transport residual=%.17g\n", transport_residual))
