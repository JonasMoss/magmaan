library(magmaan)

core <- magmaan_core

loading_free_id <- function(fit, rhs) {
  row <- which(fit$partable$op == "=~" & fit$partable$rhs == rhs)
  free_id <- fit$partable$free[row]
  stopifnot(length(free_id) == 1L, free_id > 0L)
  free_id
}

check_profile <- function(lrt, target) {
  stopifnot(
    lrt$df == 1L,
    is.finite(lrt$T),
    abs(lrt$constrained_value - target) < 1e-5,
    abs(lrt$T - 2 * lrt$nobs *
          (lrt$fmin_constrained - lrt$fmin_unrestricted)) < 1e-8
  )
}

check_ci <- function(ci) {
  stopifnot(
    ci$lower < ci$estimate,
    ci$upper > ci$estimate,
    abs(ci$lower_profile$T - ci$cutoff) < 3e-3,
    abs(ci$upper_profile$T - ci$cutoff) < 3e-3
  )
}

set.seed(20260703)
n <- 240L
eta <- rnorm(n)
x1 <- eta + rnorm(n, sd = 0.60)
x2 <- 0.82 * eta + rnorm(n, sd = 0.70)
x3 <- 0.72 * eta + rnorm(n, sd = 0.78)
x4 <- 0.64 * eta + rnorm(n, sd = 0.86)
dat <- data.frame(x1, x2, x3, x4)
dat$x2[seq(7L, n, by = 17L)] <- NA_real_
dat$x3[seq(5L, n, by = 19L)] <- NA_real_

model <- "f =~ x1 + x2 + x3 + x4"
control <- list(max_iter = 2500, ftol = 1e-11, gtol = 1e-8)

fit_fiml <- magmaan(model, dat, estimator = "FIML", control = control)
free_fiml <- loading_free_id(fit_fiml, "x2")
target_fiml <- 0.97 * fit_fiml$theta[free_fiml]
lrt_fiml <- core$frontier_profile_lrt_parameter_fiml(
  fit_fiml, free_fiml, target_fiml)
ci_fiml <- core$frontier_profile_lrt_ci_parameter_fiml(
  fit_fiml, free_fiml,
  initial_step = 0.05 * abs(fit_fiml$theta[free_fiml]),
  root_tol = 1e-4,
  statistic_tol = 1e-4
)
check_profile(lrt_fiml, target_fiml)
check_ci(ci_fiml)
stopifnot(isTRUE(lrt_fiml$constrained$fiml))

fit_ml2s <- magmaan(model, dat, estimator = "ML2S", control = control)
free_ml2s <- loading_free_id(fit_ml2s, "x2")
target_ml2s <- 0.97 * fit_ml2s$theta[free_ml2s]
lrt_ml2s <- core$frontier_profile_lrt_parameter_ml2s_nt(
  fit_ml2s, free_ml2s, target_ml2s)
ci_ml2s <- core$frontier_profile_lrt_ci_parameter_ml2s_nt(
  fit_ml2s, free_ml2s,
  initial_step = 0.05 * abs(fit_ml2s$theta[free_ml2s]),
  root_tol = 1e-4,
  statistic_tol = 1e-4
)
check_profile(lrt_ml2s, target_ml2s)
check_ci(ci_ml2s)
stopifnot(identical(lrt_ml2s$constrained$stage2_weight, "nt"))

dat_mixed <- data.frame(
  x1 = ordered(cut(x1, c(-Inf, -0.55, 0.40, Inf), labels = FALSE)),
  x2 = ordered(cut(x2, c(-Inf, 0.10, Inf), labels = FALSE)),
  x3 = x3,
  x4 = x4
)
fit_mixed <- magmaan(
  model, dat_mixed, estimator = "DWLS", ordered = c("x1", "x2"),
  meanstructure = TRUE, control = control)
free_mixed <- loading_free_id(fit_mixed, "x2")
target_mixed <- 0.97 * fit_mixed$theta[free_mixed]
lrt_mixed <- core$frontier_profile_lrt_parameter_mixed_ordinal(
  fit_mixed, free_mixed, target_mixed)
ci_mixed <- core$frontier_profile_lrt_ci_parameter_mixed_ordinal(
  fit_mixed, free_mixed,
  initial_step = 0.05 * abs(fit_mixed$theta[free_mixed]),
  root_tol = 1e-4,
  statistic_tol = 1e-4
)
check_profile(lrt_mixed, target_mixed)
check_ci(ci_mixed)
stopifnot(isTRUE(lrt_mixed$constrained$mixed_ordinal))

print(lrt_fiml[c("parameter", "target", "T", "p_value")])
print(ci_fiml[c("parameter", "estimate", "lower", "upper", "cutoff")])
print(lrt_ml2s[c("parameter", "target", "T", "p_value")])
print(ci_ml2s[c("parameter", "estimate", "lower", "upper", "cutoff")])
print(lrt_mixed[c("parameter", "target", "T", "p_value")])
print(ci_mixed[c("parameter", "estimate", "lower", "upper", "cutoff")])
