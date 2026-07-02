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

fit <- magmaan("f =~ x1 + x2 + x3", dat, estimator = "ULS")
loading_row <- which(fit$partable$op == "=~" & fit$partable$rhs == "x2")
free_id <- fit$partable$free[loading_row]
stopifnot(length(free_id) == 1L, free_id > 0L)

target <- 0.95 * fit$theta[free_id]
lrt <- core$frontier_profile_lrt_parameter_gmm(fit, free_id, target)

stopifnot(
  is.finite(lrt$T),
  lrt$df == 1L,
  abs(lrt$constrained_value - target) < 1e-5,
  abs(lrt$T - 2 * lrt$nobs *
        (lrt$fmin_constrained - lrt$fmin_unrestricted)) < 1e-8
)

print(lrt[c("parameter", "target", "T", "p_value")])
