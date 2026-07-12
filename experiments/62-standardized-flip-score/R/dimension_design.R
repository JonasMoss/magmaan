# Variable-dimension design for the dimension-scaling probe. Holds the tested
# restriction count fixed (df = 8 loadings on the first two factors) while
# growing the ambient model from 2 to 6 factors (p = 10, 20, 30), so the
# nuisance dimension grows but the tested subspace does not. Reuses the severe
# copula generators from expansion_design.R (flip_expansion_draw_block is
# dimension-agnostic). Homogeneous populations isolate dimension from group
# heterogeneity.

dim_base_loading <- c(1, .9, .75, 1.1, .65)
dim_residual_cycle <- c(.50, .65, .80, .45, .90)

dim_ov <- function(n_factors) paste0("x", seq_len(5L * n_factors))

dim_model <- function(n_factors) {
  loads <- vapply(seq_len(n_factors), function(f)
    paste0("f", f, " =~ ", paste(paste0("x", (f - 1L) * 5L + 1:5), collapse = " + ")),
    character(1L))
  covs <- if (n_factors >= 2L) apply(utils::combn(n_factors, 2L), 2L,
    function(pair) paste0("f", pair[1L], " ~~ f", pair[2L])) else character(0L)
  paste(c(loads, covs), collapse = "\n")
}

dim_specs <- function(n_factors, df = 8L) {
  model <- dim_model(n_factors)
  tokens <- unlist(lapply(seq_len(n_factors), function(f)
    paste0("f", f, " =~ x", (f - 1L) * 5L + 2:5)))
  stopifnot(df <= length(tokens))
  tested <- tokens[seq_len(df)]
  partial <- setdiff(tokens, tested)
  list(
    configural = model_spec(model, meanstructure = TRUE, group = "school",
                            group_labels = c("A", "B")),
    restricted = model_spec(model, meanstructure = TRUE, group = "school",
                            group_labels = c("A", "B"), group_equal = "loadings",
                            group_partial = partial))
}

dim_population <- function(n_factors, heterogeneity = "homogeneous") {
  p <- 5L * n_factors
  loading <- matrix(0, p, n_factors)
  for (f in seq_len(n_factors))
    loading[((f - 1L) * 5L + 1L):(f * 5L), f] <- dim_base_loading
  theta1 <- diag(rep(dim_residual_cycle, n_factors))
  phi1 <- matrix(.3, n_factors, n_factors); diag(phi1) <- 1
  Sigma1 <- loading %*% phi1 %*% t(loading) + theta1
  if (heterogeneity == "homogeneous") {
    Sigma2 <- Sigma1
  } else if (heterogeneity == "geometry") {
    # Group 2: heterogeneous latent variances and residual scaling, so the
    # tested-vs-nuisance information geometry differs across groups. Loadings
    # stay invariant (the null holds).
    lv <- rep(c(3, .45), length.out = n_factors)
    phi2 <- outer(sqrt(lv), sqrt(lv)) * .30; diag(phi2) <- lv
    theta2 <- diag(diag(theta1) * rep(c(.40, 2.20), length.out = p))
    Sigma2 <- loading %*% phi2 %*% t(loading) + theta2
  } else stop("unknown heterogeneity: ", heterogeneity, call. = FALSE)
  intercept <- seq(.15, 1.5, length.out = p)
  alpha <- rep(c(.45, -.30), length.out = n_factors)
  list(mu = list(intercept, intercept + as.vector(loading %*% alpha)),
       Sigma = list(Sigma1, Sigma2))
}

# Total N chosen to hit a target per-group sample-to-free-parameter ratio.
# Free params per group are approximated as 15F + F(F-1)/2 (loadings, residual
# variances, factor variances and covariances, intercepts).
dim_free_params <- function(n_factors)
  15L * n_factors + n_factors * (n_factors - 1L) %/% 2L

dim_total_n <- function(n_factors, ratio) {
  per_group <- ceiling(ratio * dim_free_params(n_factors))
  2L * per_group
}
