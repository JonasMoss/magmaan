# Population, invariance-ladder syntax, and MCAR sampler for the FIML
# measurement-invariance FMG experiment. One latent factor, six indicators,
# two groups; the population is scalar-invariant (equal loadings AND equal
# intercepts across groups) with a group-2 latent-mean shift, so configural,
# metric, and scalar models are all correctly specified under H0.

build_invariance_population <- function() {
  list(
    ov      = paste0("x", 1:6),
    lambda  = c(1.00, 0.80, 0.90, 0.70, 1.10, 0.85),  # x1 is the marker
    nu      = c(0.50, 0.30, 0.40, 0.20, 0.60, 0.35),  # equal across groups
    theta   = c(0.50, 0.60, 0.55, 0.65, 0.50, 0.60),  # equal across groups
    psi     = c(1.00, 1.30),                          # per-group factor variance
    alpha   = c(0.00, 0.30)                           # per-group latent mean
  )
}

# Implied per-group mean/covariance under an optional H1 violation:
#   violate = "metric" shifts group-2 loading of x2 (loading non-invariance),
#   violate = "scalar" shifts group-2 intercept of x2 (intercept non-invariance).
group_moments <- function(pop, g, violate = NULL) {
  lambda <- pop$lambda
  nu     <- pop$nu
  if (identical(violate, "metric") && g == 2L) lambda[2] <- lambda[2] * 1.35
  if (identical(violate, "scalar") && g == 2L) nu[2]     <- nu[2] + 0.60
  Sigma <- tcrossprod(lambda) * pop$psi[g] + diag(pop$theta)
  mu    <- nu + lambda * pop$alpha[g]
  list(Sigma = Sigma, mu = mu)
}

# Draw one replicate's complete two-group data frame. dist in {"norm","t5"};
# the heavy-tailed arm scales standard-normal rows by a chi/df factor so the
# marginals are multivariate-t(df) with the same covariance.
# `n_per_group` is length 1 (equal groups) or length 2 (per-group sizes). Unequal
# group sizes are the case that exercises the pooled-weight projector — with
# equal groups the per-group weight w_b = n_b/N is a global scalar and the
# UGamma spectrum is insensitive to it.
draw_complete <- function(pop, n_per_group, dist = "norm", violate = NULL,
                          t_df = 5) {
  n_g <- if (length(n_per_group) == 1L) rep(n_per_group, 2L) else n_per_group
  blocks <- lapply(seq_len(2L), function(g) {
    n <- n_g[g]
    mom <- group_moments(pop, g, violate)
    L <- chol(mom$Sigma)               # upper; rows %*% L give the cov
    Z <- matrix(rnorm(n * length(pop$ov)), n)
    if (identical(dist, "t5")) {
      s <- sqrt(t_df / rchisq(n, t_df))   # per-row scale
      Z <- Z * s
      Z <- Z / sqrt(t_df / (t_df - 2))    # rescale to unit variance
    }
    X <- Z %*% L
    X <- sweep(X, 2L, mom$mu, `+`)
    colnames(X) <- pop$ov
    data.frame(X, school = if (g == 1L) "A" else "B",
               stringsAsFactors = FALSE)
  })
  do.call(rbind, blocks)
}

# Inject MCAR missingness (per-cell Bernoulli) into the observed columns only.
inject_mcar <- function(df, ov, rate) {
  if (rate <= 0) return(df)
  for (v in ov) {
    drop <- runif(nrow(df)) < rate
    df[[v]][drop] <- NA_real_
  }
  df
}

# Invariance-ladder syntax. Shared labels tie a parameter across groups (magmaan
# and lavaan agree on this), so the same string defines the same model in both.
invariance_syntax <- function(level) {
  ov <- paste0("x", 1:6)
  loadings_free   <- "f =~ x1 + x2 + x3 + x4 + x5 + x6"
  loadings_tied   <- "f =~ x1 + L2*x2 + L3*x3 + L4*x4 + L5*x5 + L6*x6"
  intercepts_tied <- paste(sprintf("x%d ~ t%d*1", 1:6, 1:6), collapse = "\n")
  switch(level,
    configural = loadings_free,
    metric     = loadings_tied,
    # scalar: loadings + intercepts tied, latent mean fixed in g1 / free in g2.
    scalar     = paste(loadings_tied, intercepts_tied, "f ~ c(0, NA)*1",
                       sep = "\n"),
    stop("unknown invariance level: ", level)
  )
}

# lavaan group.equal vector matching each level (for the lavaan oracle fit).
lavaan_group_equal <- function(level) {
  switch(level,
    configural = character(0),
    metric     = "loadings",
    scalar     = c("loadings", "intercepts"),
    stop("unknown invariance level: ", level)
  )
}

invariance_levels <- function() c("configural", "metric", "scalar")
# Adjacent nested pairs on the ladder (H0 = less restricted = H1 model).
nested_pairs <- function() {
  list(c(h1 = "configural", h0 = "metric"),
       c(h1 = "metric",     h0 = "scalar"))
}
