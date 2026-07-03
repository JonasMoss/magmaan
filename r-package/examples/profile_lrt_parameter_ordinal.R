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

stopifnot(
  isTRUE(lrt$constrained$ordinal),
  lrt$df == 1L,
  is.finite(lrt$T),
  abs(lrt$constrained_value - target) < 1e-5,
  abs(lrt$T - 2 * lrt$nobs *
        (lrt$fmin_constrained - lrt$fmin_unrestricted)) < 1e-8
)

print(lrt[c("parameter", "target", "T", "p_value")])
