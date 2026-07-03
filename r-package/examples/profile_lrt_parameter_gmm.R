library(magmaan)

core <- magmaan_core

set.seed(20260703)
n <- 200
eta <- rnorm(n)
dat <- data.frame(
  x1 = eta + rnorm(n, sd = 0.60),
  x2 = 0.80 * eta + rnorm(n, sd = 0.70),
  x3 = 0.70 * eta + rnorm(n, sd = 0.80)
)
X <- as.matrix(dat)

fit <- magmaan("f =~ x1 + x2 + x3", dat, estimator = "ULS")
loading_row <- which(fit$partable$op == "=~" & fit$partable$rhs == "x2")
free_id <- fit$partable$free[loading_row]
stopifnot(length(free_id) == 1L, free_id > 0L)

target <- 0.95 * fit$theta[free_id]
lrt <- core$frontier_profile_lrt_parameter_gmm(fit, free_id, target)
lrt_robust <- core$frontier_profile_lrt_parameter_gmm(
  fit, free_id, target, raw_data = X, robust = TRUE
)
lrt_fitw <- core$frontier_profile_lrt_parameter_gmm_fitted_weight(
  fit, free_id, target
)
ci <- core$frontier_profile_lrt_ci_parameter_gmm(
  fit, free_id, initial_step = 0.1 * abs(fit$theta[free_id]),
  root_tol = 1e-4, statistic_tol = 1e-4
)
ci_robust <- core$frontier_profile_lrt_ci_parameter_gmm(
  fit, free_id, initial_step = 0.1 * abs(fit$theta[free_id]),
  root_tol = 1e-5, statistic_tol = 1e-5,
  raw_data = X, robust = TRUE
)
ci_fitw <- core$frontier_profile_lrt_ci_parameter_gmm_fitted_weight(
  fit, free_id, initial_step = 0.1 * abs(fit$theta[free_id]),
  root_tol = 1e-4, statistic_tol = 1e-4
)

stopifnot(
  is.finite(lrt$T),
  lrt$df == 1L,
  abs(lrt$constrained_value - target) < 1e-5,
  abs(lrt$T - 2 * lrt$nobs *
        (lrt$fmin_constrained - lrt$fmin_unrestricted)) < 1e-8,
  is.finite(lrt_robust$scaling_factor),
  lrt_robust$scaling_factor > 0,
  abs(lrt_robust$T - lrt$T) < 1e-8,
  abs(lrt_robust$T_scaled -
        lrt_robust$T / lrt_robust$scaling_factor) < 1e-8,
  is.finite(lrt_fitw$T),
  lrt_fitw$df == 1L,
  abs(lrt_fitw$constrained_value - target) < 1e-5,
  ci$lower < fit$theta[free_id],
  ci$upper > fit$theta[free_id],
  abs(ci$lower_profile$T - ci$cutoff) < 1e-3,
  abs(ci$upper_profile$T - ci$cutoff) < 1e-3,
  ci_robust$lower < fit$theta[free_id],
  ci_robust$upper > fit$theta[free_id],
  abs(ci_robust$lower_profile$T_scaled - ci_robust$cutoff) < 1e-3,
  abs(ci_robust$upper_profile$T_scaled - ci_robust$cutoff) < 1e-3,
  ci_fitw$lower < ci_fitw$estimate,
  ci_fitw$upper > ci_fitw$estimate
)

print(lrt[c("parameter", "target", "T", "p_value")])
print(lrt_robust[c("parameter", "target", "T_scaled", "p_value_scaled",
                   "scaling_factor")])
print(ci[c("parameter", "estimate", "lower", "upper", "cutoff")])
print(ci_robust[c("parameter", "estimate", "lower", "upper", "cutoff")])
