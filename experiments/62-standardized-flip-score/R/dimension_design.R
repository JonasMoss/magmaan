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

dim_population <- function(n_factors) {
  p <- 5L * n_factors
  loading <- matrix(0, p, n_factors)
  for (f in seq_len(n_factors))
    loading[((f - 1L) * 5L + 1L):(f * 5L), f] <- dim_base_loading
  theta <- diag(rep(dim_residual_cycle, n_factors))
  phi <- matrix(.3, n_factors, n_factors); diag(phi) <- 1
  Sigma <- loading %*% phi %*% t(loading) + theta
  intercept <- seq(.15, 1.5, length.out = p)
  alpha <- rep(c(.45, -.30), length.out = n_factors)
  list(mu = list(intercept, intercept + as.vector(loading %*% alpha)),
       Sigma = list(Sigma, Sigma))   # homogeneous
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
