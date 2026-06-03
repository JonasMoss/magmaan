# Population generators for the alpha-vs-omega simulation and proposed study.
#
# Each population is a one-factor-style composition
#
#   X_j = lam_j * xi + gam_j * eta + eps_j,
#
# with xi (common factor), eta (optional minor factor), and eps_j standardized
# to unit variance. In the default independent-source regimes all components
# are mutually independent, so the population covariance is exactly
# lam lam' + gam gam' + diag(psi), in closed form, whatever the component
# distributions are -- no moment-matching or numerical calibration. Additional
# stress regimes preserve the same covariance but change fourth moments, such
# as paired quadratic residuals with zero residual covariance and nonlinear
# within-pair dependence.
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

.tau_loading_from_reliability <- function(p, reliability) {
  q <- reliability / (p - reliability * (p - 1))
  sqrt(q)
}

.loading_pattern <- function(name, p, reliability = NULL) {
  if (name == "tau") {
    lam <- if (is.null(reliability)) rep(0.7, p)
    else rep(.tau_loading_from_reliability(p, reliability), p)
  } else if (name == "congeneric") {
    lam <- seq(0.45, 0.85, length.out = p)
  } else if (name == "congeneric_lit") {
    base <- seq(0.30, 0.80, length.out = 6L)
    lam <- rep(base, length.out = p)
  } else if (name == "congeneric_mild") {
    lam <- seq(0.55, 0.75, length.out = p)
  } else if (name == "congeneric_lopsided") {
    lam <- c(0.85, rep(0.45, p - 1L))
  } else {
    stop("unknown loading pattern: ", name, call. = FALSE)
  }
  lam
}

population <- function(name, p = 6L, reliability = NULL,
                       residual_cov = 0.20) {
  name <- match.arg(name, c("tau", "congeneric", "congeneric_mild",
                           "congeneric_lit", "congeneric_lopsided",
                           "misspecified", "minor_factor",
                           "correlated_residual"))
  gam <- numeric(p)
  resid <- matrix(0, p, p)
  if (name == "tau") {
    lam <- .loading_pattern("tau", p, reliability)
  } else if (name %in% c("congeneric", "congeneric_mild", "congeneric_lit",
                         "congeneric_lopsided")) {
    lam <- .loading_pattern(name, p, reliability)
  } else if (name %in% c("misspecified", "minor_factor")) {
    lam <- rep(0.7, p)
    half <- seq_len(p) <= ceiling(p / 2)     # ... plus a minor 2nd factor on
    gam[half] <- 0.35                        #     the first half of the items
  } else if (name == "correlated_residual") {
    lam <- rep(0.7, p)
    if (p < 2L) stop("correlated_residual requires p >= 2", call. = FALSE)
    resid[1L, 2L] <- residual_cov
    resid[2L, 1L] <- residual_cov
  }
  psi <- pmax(1 - lam^2 - gam^2, 1e-3)       # unit total variance per item
  Sigma <- tcrossprod(lam) + tcrossprod(gam) + diag(psi) + resid
  list(name = name, p = p, lam = lam, gam = gam, psi = psi, resid = resid,
       Sigma = Sigma)
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

# Draw paired quadratic residual components. Columns come in pairs:
# u, (u^2 - 1) / sqrt(2). This preserves zero residual covariance with perfect
# nonlinear dependence inside each pair.
.paired_quadratic_residuals <- function(n, p) {
  if (p %% 2L != 0L) {
    stop("paired_quadratic_residuals requires an even p", call. = FALSE)
  }
  E <- matrix(0, n, p)
  for (j in seq(1L, p, by = 2L)) {
    u <- stats::rnorm(n)
    E[, j] <- u
    E[, j + 1L] <- (u^2 - 1) / sqrt(2)
  }
  E
}

# Draw an n x p sample from a population under a distribution regime.
#   normal       : all independent normal sources
#   nonnormal    : all independent chi-square(df) sources (legacy alias)
#   latent_chisq : chi-square common/minor factors, normal residuals
#   error_chisq  : normal factors, independent chi-square residuals
#   all_chisq    : all independent chi-square sources
#   paired_quadratic_residuals:
#                  normal factors, zero-covariance dependent residual pairs
draw_sample <- function(pop, n, dist = c("normal", "nonnormal"),
                        nonnormal_df = 5) {
  dist <- match.arg(dist, c("normal", "nonnormal", "latent_chisq",
                            "error_chisq", "all_chisq",
                            "paired_quadratic_residuals"))
  if (dist == "nonnormal") dist <- "all_chisq"
  df_factor <- if (dist %in% c("latent_chisq", "all_chisq")) nonnormal_df else Inf
  df_error <- if (dist %in% c("error_chisq", "all_chisq")) nonnormal_df else Inf
  p <- pop$p
  xi  <- .rcomp(n, df_factor)
  eta <- .rcomp(n, df_factor)
  X <- outer(xi, pop$lam) + outer(eta, pop$gam)
  if (dist == "paired_quadratic_residuals") {
    E <- .paired_quadratic_residuals(n, p)
  } else {
    E <- replicate(p, .rcomp(n, df_error))
  }
  for (j in seq_len(p)) X[, j] <- X[, j] + sqrt(pop$psi[j]) * E[, j]
  if (any(pop$resid != 0)) {
    R <- diag(pop$psi) + pop$resid
    L <- chol(R)
    E_corr <- matrix(stats::rnorm(n * p), n, p) %*% L
    X <- outer(xi, pop$lam) + outer(eta, pop$gam) + E_corr
  }
  colnames(X) <- paste0("x", seq_len(p))
  X
}

alpha_gradient_matrix <- function(Sigma) {
  p <- ncol(Sigma)
  c_alpha <- p / (p - 1)
  T <- sum(diag(Sigma))
  V <- sum(Sigma)
  r <- T / V
  (c_alpha / V) * (r * matrix(1, p, p) - diag(p))
}

alpha_avar_components <- function(pop, latent_kurtosis = 0,
                                  residual_kurtosis = 0,
                                  paired_quadratic = FALSE) {
  p <- pop$p
  if (length(residual_kurtosis) == 1L) {
    residual_kurtosis <- rep(residual_kurtosis, p)
  }
  if (length(residual_kurtosis) != p) {
    stop("residual_kurtosis must be scalar or length p", call. = FALSE)
  }
  B <- alpha_gradient_matrix(pop$Sigma)
  normal <- 2 * sum(B * (pop$Sigma %*% B %*% pop$Sigma))
  latent <- latent_kurtosis * as.numeric(t(pop$lam) %*% B %*% pop$lam)^2
  residual <- sum(residual_kurtosis * (pop$psi * diag(B))^2)
  paired_dep <- 0
  if (paired_quadratic) {
    if (p %% 2L != 0L) {
      stop("paired_quadratic diagnostics require an even p", call. = FALSE)
    }
    delta_uv <- 4                         # u ~ N, v = (u^2 - 1) / sqrt(2)
    for (j in seq(1L, p, by = 2L)) {
      k <- j + 1L
      paired_dep <- paired_dep + delta_uv * pop$psi[j] * pop$psi[k] *
        (2 * B[j, j] * B[k, k] + 4 * B[j, k]^2)
    }
  }
  total <- normal + latent + residual + paired_dep
  data.frame(
    component = c("normal_theory", "latent_kurtosis",
                  "residual_kurtosis", "paired_dependence", "total"),
    avar = c(normal, latent, residual, paired_dep, total),
    proportion = c(normal, latent, residual, paired_dep, total) / total
  )
}

paired_quadratic_residual_kurtosis <- function(p) {
  if (p %% 2L != 0L) {
    stop("paired_quadratic_residual_kurtosis requires an even p", call. = FALSE)
  }
  rep(c(0, 12), length.out = p)
}

proposed_study_grid <- function() {
  p_grid <- c(4L, 6L, 12L)
  n_grid <- c(100L, 250L, 500L, 1000L)
  dist_core <- c("normal", "latent_chisq", "error_chisq", "all_chisq",
                 "paired_quadratic_residuals")
  tau <- expand.grid(
    population = "tau",
    dist = dist_core,
    p = p_grid,
    N = n_grid,
    reliability = c(0.55, 0.70, 0.85),
    stringsAsFactors = FALSE
  )
  congeneric <- expand.grid(
    population = c("congeneric_mild", "congeneric_lit",
                   "congeneric_lopsided"),
    dist = dist_core,
    p = p_grid,
    N = n_grid,
    reliability = NA_real_,
    stringsAsFactors = FALSE
  )
  minor <- expand.grid(
    population = "minor_factor",
    dist = c("normal", "all_chisq"),
    p = p_grid,
    N = n_grid,
    reliability = NA_real_,
    stringsAsFactors = FALSE
  )
  corr_resid <- expand.grid(
    population = "correlated_residual",
    dist = "normal",
    p = p_grid,
    N = n_grid,
    reliability = NA_real_,
    stringsAsFactors = FALSE
  )
  rbind(tau, congeneric, minor, corr_resid)
}
