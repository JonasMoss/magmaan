# Population generators for the alpha-vs-omega simulation.
#
# Each population is a one-factor-style composition
#
#   X_j = lam_j * xi + gam_j * eta + eps_j,
#
# with xi (common factor), eta (optional minor factor), and eps_j all
# independent and standardized to unit variance. The population covariance is
# therefore exactly lam lam' + gam gam' + diag(psi), in closed form, whatever
# the component distributions are -- no moment-matching or numerical
# calibration. Non-normality is introduced by drawing the standardized
# components from a centred-scaled chi-square instead of a normal; the
# covariance is preserved and the marginals carry skew and excess kurtosis.
#
#   tau         : equal loadings, one factor          -> population alpha = omega
#   congeneric  : graded unequal loadings, one factor -> alpha < omega
#   misspecified: equal loadings + a minor 2nd factor -> one-factor model false
#
# psi_j is set so each X_j has unit total variance, keeping the scale of the
# coefficients comparable across conditions.

# Standardized chi-square component: mean 0, variance 1, excess kurtosis 12/df.
# df = Inf draws standard normal. df = 5 gives skew ~1.26, excess kurtosis 2.4.
.rcomp <- function(n, df = Inf) {
  if (is.infinite(df)) return(stats::rnorm(n))
  (stats::rchisq(n, df) - df) / sqrt(2 * df)
}

population <- function(name, p = 6L) {
  name <- match.arg(name, c("tau", "congeneric", "misspecified"))
  gam <- numeric(p)
  if (name == "tau") {
    lam <- rep(0.7, p)
  } else if (name == "congeneric") {
    lam <- seq(0.45, 0.85, length.out = p)
  } else {                                   # misspecified: equal loadings ...
    lam <- rep(0.7, p)
    half <- seq_len(p) <= ceiling(p / 2)     # ... plus a minor 2nd factor on
    gam[half] <- 0.35                        #     the first half of the items
  }
  psi <- pmax(1 - lam^2 - gam^2, 1e-3)       # unit total variance per item
  Sigma <- tcrossprod(lam) + tcrossprod(gam) + diag(psi)
  list(name = name, p = p, lam = lam, gam = gam, psi = psi, Sigma = Sigma)
}

# Population reliability targets, for interpreting the simulated rates.
#   alpha_pop : alpha of the true Sigma (model-free)
#   omega1f   : reliability of the best one-factor approximation is not closed
#               form here, so we report the "common-part" reliability that a
#               correctly specified model would recover; only meaningful when
#               the one-factor model holds (tau, congeneric).
population_targets <- function(pop) {
  Sigma <- pop$Sigma; p <- pop$p
  alpha_pop <- (p / (p - 1)) * (1 - sum(diag(Sigma)) / sum(Sigma))
  true_score <- sum(pop$lam)^2
  omega_1f <- true_score / sum(Sigma)        # = reliability when gam = 0
  c(alpha_pop = alpha_pop, omega_1f = omega_1f,
    gap = omega_1f - alpha_pop)
}

# Draw an n x p sample from a population under a distribution regime.
#   dist = "normal"     -> normal components
#   dist = "nonnormal"  -> chi-square(df) components (default df below)
draw_sample <- function(pop, n, dist = c("normal", "nonnormal"),
                        nonnormal_df = 5) {
  dist <- match.arg(dist)
  df <- if (dist == "normal") Inf else nonnormal_df
  p <- pop$p
  xi  <- .rcomp(n, df)
  eta <- .rcomp(n, df)
  X <- outer(xi, pop$lam) + outer(eta, pop$gam)
  for (j in seq_len(p)) X[, j] <- X[, j] + sqrt(pop$psi[j]) * .rcomp(n, df)
  colnames(X) <- paste0("x", seq_len(p))
  X
}
