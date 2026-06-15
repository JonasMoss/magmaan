# Population, invariance-ladder syntax, and MCAR for the FIML FMG-vs-MLR study.
#
# One latent factor, six indicators, two groups; the population is
# scalar-invariant (equal loadings AND equal intercepts across groups) with a
# group-2 latent-mean shift, so the configural and metric models are both
# correctly specified under the invariant truth. Identical structural design to
# experiments/21-fiml-measurement-invariance-fmg (the validation probe); this
# experiment swaps the heavy-tailed t5 arm for the experiment-17 non-normal
# families (VM/IG/PL, severe skew+kurtosis) and adds the MLR baseline. Leaves are
# independent (no cross-experiment source()), so the structural code is carried
# locally.

build_invariance_population <- function() {
  list(
    ov      = paste0("x", 1:6),
    lambda  = c(1.00, 0.80, 0.90, 0.70, 1.10, 0.85),  # x1 is the marker
    nu      = c(0.50, 0.30, 0.40, 0.20, 0.60, 0.35),  # equal across groups
    theta   = c(0.50, 0.60, 0.55, 0.65, 0.50, 0.60),  # equal across groups
    psi     = c(1.00, 1.30),                          # per-group factor variance
    alpha   = c(0.00, 0.30),                          # per-group latent mean
    # Goodness-of-fit power lever: a residual covariance the 1-factor model omits.
    gof_pair = c(5L, 6L),
    gof_res_cov = 0.13,
    # Nested-test power lever: group-2 loading inflation for the violated indicator.
    nested_index = 2L,
    nested_scale = 1.35
  )
}

# Implied per-group mean/covariance under an optional truth violation:
#   violate = "nested" inflates the group-2 loading of the violated indicator
#             (loading non-invariance: powers the configural-vs-metric test),
#   violate = "gof"    adds a residual covariance the 1-factor model omits, in
#             both groups (powers the goodness-of-fit test of the metric model).
group_moments <- function(pop, g, violate = NULL) {
  lambda <- pop$lambda
  nu     <- pop$nu
  if (identical(violate, "nested") && g == 2L) {
    lambda[pop$nested_index] <- lambda[pop$nested_index] * pop$nested_scale
  }
  Theta <- diag(pop$theta)
  if (identical(violate, "gof")) {
    i <- pop$gof_pair[1L]; j <- pop$gof_pair[2L]
    Theta[i, j] <- Theta[j, i] <- pop$gof_res_cov
  }
  Sigma <- tcrossprod(lambda) * pop$psi[g] + Theta
  mu    <- nu + lambda * pop$alpha[g]
  list(Sigma = Sigma, mu = mu)
}

# Inject MCAR missingness (per-cell Bernoulli) into the observed columns only.
inject_mcar <- function(df, ov, rate) {
  if (rate <= 0) return(df)
  for (v in ov) {
    drop <- stats::runif(nrow(df)) < rate
    df[[v]][drop] <- NA_real_
  }
  df
}

# Invariance-ladder syntax. Shared labels tie a parameter across groups (magmaan
# and lavaan agree on this), so the same string defines the same model in both.
invariance_syntax <- function(level) {
  loadings_free   <- "f =~ x1 + x2 + x3 + x4 + x5 + x6"
  loadings_tied   <- "f =~ x1 + L2*x2 + L3*x3 + L4*x4 + L5*x5 + L6*x6"
  intercepts_tied <- paste(sprintf("x%d ~ t%d*1", 1:6, 1:6), collapse = "\n")
  switch(level,
    configural = loadings_free,
    metric     = loadings_tied,
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

invariance_levels <- function() c("configural", "metric")

# The single nested pair the study reports: configural (H1) vs metric (H0).
nested_pair <- function() c(h1 = "configural", h0 = "metric")

# Map a cell's truth axis to the population violation passed to the generator.
violate_for_truth <- function(truth) switch(truth,
  h0 = NULL, gof_power = "gof", nested_power = "nested",
  stop("unknown truth: ", truth))
