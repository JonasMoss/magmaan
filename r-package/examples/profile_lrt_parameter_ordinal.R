library(magmaan)

core <- magmaan_core

set.seed(20260703)
n <- 360L
eta <- rnorm(n)
loading <- c(0.86, 0.78, 0.70, 0.62)
cuts <- c(-0.55, 0.45)
dat <- data.frame(row.names = seq_len(n))
for (j in seq_along(loading)) {
  y <- loading[j] * eta + sqrt(1 - loading[j]^2) * rnorm(n)
  dat[[paste0("x", j)]] <- ordered(
    cut(y, c(-Inf, cuts, Inf), labels = FALSE)
  )
}

ordered <- paste0("x", 1:4)
model <- "f =~ x1 + x2 + x3 + x4"
spec <- model_spec(model, ordered = ordered, parameterization = "delta")
stats <- core$data_ordinal_stats_from_df(dat, spec)
fit <- core$fit_dwls_ordinal(
  spec, stats,
  control = list(max_iter = 4000, ftol = 1e-12, gtol = 1e-8)
)

loading_row <- which(fit$partable$op == "=~" & fit$partable$rhs == "x2")
free_id <- fit$partable$free[loading_row]
stopifnot(length(free_id) == 1L, free_id > 0L)

target <- 0.95 * fit$theta[free_id]
lrt <- core$frontier_profile_lrt_parameter_ordinal(fit, free_id, target)
lrt_robust <- core$frontier_profile_lrt_parameter_ordinal(
  fit, free_id, target, robust = TRUE)
ci <- core$frontier_profile_lrt_ci_parameter_ordinal(
  fit, free_id,
  initial_step = 0.08 * abs(fit$theta[free_id]),
  root_tol = 1e-5,
  statistic_tol = 1e-5
)
ci_robust <- core$frontier_profile_lrt_ci_parameter_ordinal(
  fit, free_id,
  initial_step = 0.08 * abs(fit$theta[free_id]),
  root_tol = 1e-5,
  statistic_tol = 1e-5,
  robust = TRUE
)

block <- rep.int(1L, length(ordered))
omega_ci <- core$frontier_profile_lrt_ci_ordinal_polychoric_omega(
  fit, block = block,
  initial_step = 0.02,
  root_tol = 1e-5,
  statistic_tol = 1e-5
)
omega_target <- 0.98 * omega_ci$estimate
omega_lrt <- core$frontier_profile_lrt_ordinal_polychoric_omega(
  fit, block = block, omega0 = omega_target
)
omega_lrt_robust <- core$frontier_profile_lrt_ordinal_polychoric_omega(
  fit, block = block, omega0 = omega_target, robust = TRUE
)
omega_ci_robust <- core$frontier_profile_lrt_ci_ordinal_polychoric_omega(
  fit, block = block,
  initial_step = 0.02,
  root_tol = 1e-6,
  statistic_tol = 1e-5,
  robust = TRUE
)

stopifnot(
  isTRUE(lrt$constrained$ordinal),
  lrt$df == 1L,
  is.finite(lrt$T),
  abs(lrt$constrained_value - target) < 1e-5,
  abs(lrt$T - 2 * lrt$nobs *
        (lrt$fmin_constrained - lrt$fmin_unrestricted)) < 1e-8,
  is.finite(lrt_robust$scaling_factor),
  lrt_robust$scaling_factor > 0,
  abs(lrt_robust$T_scaled -
        lrt_robust$T / lrt_robust$scaling_factor) < 1e-8,
  ci$lower < fit$theta[free_id],
  ci$upper > fit$theta[free_id],
  abs(ci$lower_profile$T - ci$cutoff) < 1e-3,
  abs(ci$upper_profile$T - ci$cutoff) < 1e-3,
  ci_robust$lower < fit$theta[free_id],
  ci_robust$upper > fit$theta[free_id],
  abs(ci_robust$lower_profile$T_scaled - ci_robust$cutoff) < 1e-3,
  abs(ci_robust$upper_profile$T_scaled - ci_robust$cutoff) < 1e-3,
  omega_lrt$coefficient == "ordinal_polychoric_omega",
  omega_lrt$omega_target == "total",
  omega_lrt$df == 1L,
  abs(omega_lrt$constrained_value - omega_target) < 1e-5,
  abs(omega_lrt$T - 2 * omega_lrt$nobs *
        (omega_lrt$fmin_constrained - omega_lrt$fmin_unrestricted)) < 1e-8,
  is.finite(omega_lrt_robust$scaling_factor),
  omega_lrt_robust$scaling_factor > 0,
  abs(omega_lrt_robust$T_scaled -
        omega_lrt_robust$T / omega_lrt_robust$scaling_factor) < 1e-8,
  omega_ci$coefficient == "ordinal_polychoric_omega",
  omega_ci$lower < omega_ci$estimate,
  omega_ci$upper > omega_ci$estimate,
  abs(omega_ci$lower_profile$T - omega_ci$cutoff) < 1e-3,
  abs(omega_ci$upper_profile$T - omega_ci$cutoff) < 1e-3,
  omega_ci_robust$lower < omega_ci_robust$estimate,
  omega_ci_robust$upper > omega_ci_robust$estimate,
  abs(omega_ci_robust$lower_profile$T_scaled -
        omega_ci_robust$cutoff) < 1e-3,
  abs(omega_ci_robust$upper_profile$T_scaled -
        omega_ci_robust$cutoff) < 1e-3
)

print(lrt[c("parameter", "target", "T", "p_value")])
print(lrt_robust[c("parameter", "target", "T_scaled", "p_value_scaled",
                   "scaling_factor")])
print(ci[c("parameter", "estimate", "lower", "upper", "cutoff")])
print(ci_robust[c("parameter", "estimate", "lower", "upper", "cutoff")])
print(omega_lrt[c("coefficient", "omega_target", "target", "T", "p_value")])
print(omega_lrt_robust[c("coefficient", "omega_target", "target", "T_scaled",
                         "p_value_scaled", "scaling_factor")])
print(omega_ci[c("coefficient", "estimate", "lower", "upper", "cutoff")])
print(omega_ci_robust[c("coefficient", "estimate", "lower", "upper", "cutoff")])
