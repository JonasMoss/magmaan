fiml_flip_population <- function(truth = c("null", "power")) {
  truth <- match.arg(truth)
  lambda <- c(1.00, 0.80, 0.90, 0.70, 1.10, 0.85)
  loadings <- rbind(lambda, lambda)
  if (truth == "power") loadings[2L, 2L] <- 1.35 * loadings[2L, 2L]
  nu <- c(0.50, 0.30, 0.40, 0.20, 0.60, 0.35)
  theta <- c(0.50, 0.60, 0.55, 0.65, 0.50, 0.60)
  psi <- c(1.00, 1.30)
  alpha <- c(0.00, 0.30)
  Sigma <- lapply(seq_len(2L), function(g)
    psi[[g]] * tcrossprod(loadings[g, ]) + diag(theta))
  mu <- lapply(seq_len(2L), function(g) nu + loadings[g, ] * alpha[[g]])
  list(ov = paste0("x", 1:6), truth = truth, loadings = loadings,
       Sigma = Sigma, mu = mu)
}

fiml_flip_group_sizes <- function(n_group1) {
  c(as.integer(n_group1), as.integer(round(0.7 * n_group1)))
}

fiml_flip_syntax <- function(level = c("configural", "metric")) {
  level <- match.arg(level)
  if (level == "configural") return("f =~ x1 + x2 + x3 + x4 + x5 + x6")
  "f =~ x1 + L2*x2 + L3*x3 + L4*x4 + L5*x5 + L6*x6"
}

fiml_flip_specs <- function() {
  args <- list(group = "school", group_labels = c("A", "B"),
               meanstructure = TRUE)
  list(
    H1 = do.call(magmaan::model_spec,
                 c(list(syntax = fiml_flip_syntax("configural")), args)),
    H0 = do.call(magmaan::model_spec,
                 c(list(syntax = fiml_flip_syntax("metric")), args)))
}

fiml_flip_grid <- function(n_values = c(50L, 100L, 200L),
                           rates = c(0, 0.15, 0.30),
                           distributions = c("normal", "pl"),
                           truths = c("null", "power")) {
  out <- expand.grid(
    distribution = distributions, truth = truths,
    n_group1 = as.integer(n_values), missing_rate = as.numeric(rates),
    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  sizes <- lapply(out$n_group1, fiml_flip_group_sizes)
  out$n_group2 <- vapply(sizes, `[[`, integer(1L), 2L)
  out$n_total <- out$n_group1 + out$n_group2
  out$df <- 5L
  out$cell_id <- seq_len(nrow(out))
  out
}

fiml_flip_validate_design <- function() {
  for (truth in c("null", "power")) {
    pop <- fiml_flip_population(truth)
    stopifnot(length(pop$Sigma) == 2L, length(pop$mu) == 2L,
              all(vapply(pop$Sigma, function(S)
                min(eigen(S, symmetric = TRUE, only.values = TRUE)$values) > 0,
                logical(1L))))
  }
  grid <- fiml_flip_grid()
  stopifnot(nrow(grid) == 36L, all(grid$df == 5L),
            all(grid$n_group2 > 30L))
  invisible(TRUE)
}
