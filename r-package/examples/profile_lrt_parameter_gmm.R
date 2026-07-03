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
lrt_fitw_robust <- core$frontier_profile_lrt_parameter_gmm_fitted_weight(
  fit, free_id, target, raw_data = X, robust = TRUE
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
ci_fitw_robust <- core$frontier_profile_lrt_ci_parameter_gmm_fitted_weight(
  fit, free_id, initial_step = 0.1 * abs(fit$theta[free_id]),
  root_tol = 1e-5, statistic_tol = 1e-5,
  raw_data = X, robust = TRUE
)

fit_gls <- magmaan("f =~ x1 + x2 + x3", dat, estimator = "GLS")
free_gls <- fit_gls$partable$free[loading_row]
target_gls <- 0.95 * fit_gls$theta[free_gls]
lrt_gls_robust <- core$frontier_profile_lrt_parameter_gmm(
  fit_gls, free_gls, target_gls, raw_data = X, robust = TRUE
)
lrt_gls_estw <- core$frontier_profile_lrt_parameter_gmm(
  fit_gls, free_gls, target_gls, raw_data = X, robust = TRUE,
  estimated_weight = TRUE
)
ci_gls_estw <- core$frontier_profile_lrt_ci_parameter_gmm(
  fit_gls, free_gls, initial_step = 0.1 * abs(fit_gls$theta[free_gls]),
  root_tol = 1e-5, statistic_tol = 1e-5,
  raw_data = X, robust = TRUE, estimated_weight = TRUE
)

W_adf <- solve(core$robust_empirical_gamma(X))
fit_wls <- magmaan("f =~ x1 + x2 + x3", dat, estimator = "WLS", W = W_adf)
free_wls <- fit_wls$partable$free[loading_row]
target_wls <- 0.95 * fit_wls$theta[free_wls]
lrt_wls_robust <- core$frontier_profile_lrt_parameter_gmm(
  fit_wls, free_wls, target_wls, weight = W_adf, raw_data = X, robust = TRUE
)
lrt_wls_estw <- core$frontier_profile_lrt_parameter_gmm(
  fit_wls, free_wls, target_wls, weight = W_adf, raw_data = X, robust = TRUE,
  estimated_weight = TRUE
)
err_estw_without_robust <- tryCatch(
  core$frontier_profile_lrt_parameter_gmm(
    fit_gls, free_gls, target_gls, raw_data = X, estimated_weight = TRUE),
  error = conditionMessage)

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
  is.finite(lrt_fitw_robust$scaling_factor),
  lrt_fitw_robust$scaling_factor > 0,
  abs(lrt_fitw_robust$T - lrt_fitw$T) < 1e-8,
  abs(lrt_fitw_robust$T_scaled -
        lrt_fitw_robust$T / lrt_fitw_robust$scaling_factor) < 1e-8,
  ci$lower < fit$theta[free_id],
  ci$upper > fit$theta[free_id],
  abs(ci$lower_profile$T - ci$cutoff) < 1e-3,
  abs(ci$upper_profile$T - ci$cutoff) < 1e-3,
  ci_robust$lower < fit$theta[free_id],
  ci_robust$upper > fit$theta[free_id],
  abs(ci_robust$lower_profile$T_scaled - ci_robust$cutoff) < 1e-3,
  abs(ci_robust$upper_profile$T_scaled - ci_robust$cutoff) < 1e-3,
  ci_fitw$lower < ci_fitw$estimate,
  ci_fitw$upper > ci_fitw$estimate,
  ci_fitw_robust$lower < ci_fitw_robust$estimate,
  ci_fitw_robust$upper > ci_fitw_robust$estimate,
  abs(ci_fitw_robust$lower_profile$T_scaled -
        ci_fitw_robust$cutoff) < 1e-3,
  abs(ci_fitw_robust$upper_profile$T_scaled -
        ci_fitw_robust$cutoff) < 1e-3,
  abs(lrt_gls_estw$T - lrt_gls_robust$T) < 1e-8,
  is.finite(lrt_gls_estw$scaling_factor),
  lrt_gls_estw$scaling_factor > 0,
  abs(lrt_gls_estw$scaling_factor -
        lrt_gls_robust$scaling_factor) > 1e-6,
  ci_gls_estw$lower < fit_gls$theta[free_gls],
  ci_gls_estw$upper > fit_gls$theta[free_gls],
  abs(ci_gls_estw$lower_profile$T_scaled -
        ci_gls_estw$cutoff) < 1e-3,
  abs(ci_gls_estw$upper_profile$T_scaled -
        ci_gls_estw$cutoff) < 1e-3,
  abs(lrt_wls_estw$T - lrt_wls_robust$T) < 1e-8,
  is.finite(lrt_wls_estw$scaling_factor),
  lrt_wls_estw$scaling_factor > 0,
  is.character(err_estw_without_robust),
  grepl("estimated_weight", err_estw_without_robust, fixed = TRUE)
)

print(lrt[c("parameter", "target", "T", "p_value")])
print(lrt_robust[c("parameter", "target", "T_scaled", "p_value_scaled",
                   "scaling_factor")])
print(lrt_fitw_robust[c("parameter", "target", "T_scaled", "p_value_scaled",
                        "scaling_factor")])
print(ci[c("parameter", "estimate", "lower", "upper", "cutoff")])
print(ci_robust[c("parameter", "estimate", "lower", "upper", "cutoff")])
print(ci_fitw_robust[c("parameter", "estimate", "lower", "upper", "cutoff")])
print(lrt_gls_estw[c("parameter", "target", "T_scaled", "p_value_scaled",
                     "scaling_factor")])
print(ci_gls_estw[c("parameter", "estimate", "lower", "upper", "cutoff")])
print(lrt_wls_estw[c("parameter", "target", "T_scaled", "p_value_scaled",
                     "scaling_factor")])
