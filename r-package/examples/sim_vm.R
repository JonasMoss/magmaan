core <- magmaan::magmaan_core

# Vale-Maurelli / Fleishman: non-normal continuous draws with a target
# correlation matrix and per-margin (skewness, excess kurtosis). The two-stage
# calibrate/draw contract mirrors sim_norta_* / sim_ig_*.

target_corr <- matrix(c(
  1.00, 0.30, 0.25, 0.20,
  0.30, 1.00, 0.35, 0.15,
  0.25, 0.35, 1.00, 0.30,
  0.20, 0.15, 0.30, 1.00
), 4, 4, byrow = TRUE)

skewness <- c(0.80, 0.80, -0.60, 0.00)
excess_kurtosis <- c(1.50, 1.50, 1.00, 0.80)

cal <- core$sim_vm_calibrate(target_corr, skewness, excess_kurtosis)
stopifnot(
  inherits(cal, "magmaan_vm_calibration"),
  nrow(cal$coefficients) == 4L,
  ncol(cal$coefficients) == 4L,
  identical(colnames(cal$coefficients), c("a", "b", "c", "d")),
  all(dim(cal$intermediate_corr) == c(4L, 4L))
)

# A single large draw recovers the target marginal moments and correlation.
N <- 200000L
draw <- core$sim_vm_draw(cal, n = N, reps = 1L, seed_base = 20260613)
X <- draw$draws[[1]]
stopifnot(nrow(X) == N, ncol(X) == 4L)

samp_skew <- function(x) {
  z <- x - mean(x)
  mean(z^3) / mean(z^2)^1.5
}
samp_exkurt <- function(x) {
  z <- x - mean(x)
  mean(z^4) / mean(z^2)^2 - 3
}

achieved_skew <- apply(X, 2, samp_skew)
achieved_exkurt <- apply(X, 2, samp_exkurt)
achieved_corr <- cor(X)

stopifnot(
  max(abs(achieved_skew - skewness)) < 0.05,
  max(abs(achieved_exkurt - excess_kurtosis)) < 0.20,
  max(abs(achieved_corr - target_corr)) < 0.02
)

# Calibrate-once / draw-many: two independent draws differ, but batch with the
# same seed_base reproduces the draw path exactly.
batch <- core$sim_vm_batch(
  target_corr, skewness, excess_kurtosis,
  n = 500L, reps = 2L, seed_base = 20260614)
draw2 <- core$sim_vm_draw(cal, n = 500L, reps = 2L, seed_base = 20260614)
stopifnot(
  length(batch$draws) == 2L,
  nrow(batch$draws[[1]]) == 500L,
  max(abs(batch$draws[[1]] - draw2$draws[[1]])) < 1e-9,
  max(abs(batch$draws[[2]] - draw2$draws[[2]])) < 1e-9
)

# Infeasible Fleishman moment pair (high skew with near-normal kurtosis) is
# surfaced as an error, not a silent bad fit.
infeasible <- tryCatch({
  core$sim_vm_calibrate(diag(2), c(5.0, 0.0), c(0.0, 0.0))
  FALSE
}, error = function(e) TRUE)
stopifnot(infeasible)

cat("sim_vm_* workflow: ok\n")
