# Non-regularity diagnostic.
#
# Direct evidence that omega_hat - alpha_hat is a second-order (non-regular)
# statistic at the tau-equivalent null, while each coefficient on its own is a
# regular asymptotically-normal estimate. Run at the tau population across a
# geometric ladder of N and read off how the Monte Carlo SD shrinks:
#
#   * a regular 1/sqrt(N) statistic halves when N quadruples (ratio ~0.50)
#   * the second-order 1/N difference quarters (ratio ~0.25)
#
# We also track corr(omega_hat, alpha_hat) -> 1: at the null the two estimators
# share one influence function, so their leading normal parts cancel in the
# difference. See notes/alpha_omega_second_order.qmd.

nonregularity_scaling <- function(p = 6L, N = c(250L, 1000L, 4000L),
                                  reps = 300L, seed_base = 4040L) {
  pop <- population("tau", p)
  one_N <- function(N, gi) {
    om <- al <- numeric(reps)
    for (r in seq_len(reps)) {
      set.seed(seed_base + gi * 100000L + r)
      d <- draw_sample(pop, N, "normal")
      cg <- tryCatch(fit_congeneric(d, "ML"), error = function(e) NULL)
      if (is.null(cg) || !cg$converged) { om[r] <- NA; al[r] <- NA; next }
      om[r] <- omega_from_lp(cg$lam, cg$psi)
      al[r] <- alpha_from_S(mle_cov(d))
    }
    diff <- om - al
    data.frame(N = N, mean_diff = mean(diff, na.rm = TRUE),
               sd_diff = stats::sd(diff, na.rm = TRUE),
               sd_omega = stats::sd(om, na.rm = TRUE),
               corr_omega_alpha = stats::cor(om, al, use = "complete.obs"),
               skew_diff = .skew(diff))
  }
  rows <- do.call(rbind, Map(one_N, N, seq_along(N)))
  # SD ratios when N grows by the ladder step (diagnostic of the scaling order).
  rows$sd_diff_ratio  <- c(NA, rows$sd_diff[-1]  / rows$sd_diff[-nrow(rows)])
  rows$sd_omega_ratio <- c(NA, rows$sd_omega[-1] / rows$sd_omega[-nrow(rows)])
  rows
}

.skew <- function(x) {
  x <- x[is.finite(x)]; m <- mean(x)
  mean((x - m)^3) / stats::sd(x)^3
}
