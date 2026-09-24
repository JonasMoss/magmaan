# Population designs. Every population is standardized (unit observed and
# latent variances), so parameters are comparable across designs.

ov_names <- function(p) paste0("x", seq_len(p))

lisrel_sigma <- function(Lambda, B, Psi) {
  A <- solve(diag(nrow(B)) - B)
  Phi <- A %*% Psi %*% t(A)
  Sigma <- Lambda %*% Phi %*% t(Lambda)
  diag(Sigma) <- 1
  nm <- ov_names(nrow(Lambda))
  dimnames(Sigma) <- list(nm, nm)
  Sigma
}

one_factor_design <- function(key, lambda) {
  p <- length(lambda)
  list(
    key = key,
    label = sprintf("One factor, loadings (%s)",
                    paste(format(lambda, nsmall = 1), collapse = ", ")),
    syntax = paste0("f =~ ", paste(ov_names(p), collapse = " + ")),
    Sigma = lisrel_sigma(matrix(lambda, p, 1), matrix(0, 1, 1), diag(1)),
    lv = "f",
    key_quantity = "min_theta",
    key_truth = min(1 - lambda^2)
  )
}

two_factor_design <- function(key, r, lambda = 0.7) {
  Lambda <- cbind(c(rep(lambda, 3), rep(0, 3)), c(rep(0, 3), rep(lambda, 3)))
  list(
    key = key,
    label = sprintf("Two factors, loadings %.1f, factor correlation %.2f",
                    lambda, r),
    syntax = "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6",
    Sigma = lisrel_sigma(Lambda, matrix(0, 2, 2), matrix(c(1, r, r, 1), 2)),
    lv = c("f1", "f2"),
    key_quantity = "factor_correlation",
    key_truth = r
  )
}

# f1 -> f2 -> f3 with f1 -> f3; R^2(f2) = .5, R^2(f3) = .9, loadings .7.
path_design <- function(key, r2_f3 = 0.9, lambda = 0.7) {
  b21 <- sqrt(0.5)
  b31 <- 0.3
  # Solve b32 from b31^2 + b32^2 + 2 b31 b32 b21 = r2_f3.
  b32 <- (-2 * b31 * b21 + sqrt((2 * b31 * b21)^2 - 4 * (b31^2 - r2_f3))) / 2
  B <- matrix(0, 3, 3)
  B[2, 1] <- b21
  B[3, 1] <- b31
  B[3, 2] <- b32
  Psi <- diag(c(1, 0.5, 1 - r2_f3))
  Lambda <- kronecker(diag(3), matrix(lambda, 3, 1))
  list(
    key = key,
    label = sprintf("Three-latent recursive path, R-squared(f3) = %.2f", r2_f3),
    syntax = paste(
      "f1 =~ x1 + x2 + x3",
      "f2 =~ x4 + x5 + x6",
      "f3 =~ x7 + x8 + x9",
      "f2 ~ f1",
      "f3 ~ f1 + f2",
      sep = "\n"),
    Sigma = lisrel_sigma(Lambda, B, Psi),
    lv = c("f1", "f2", "f3"),
    key_quantity = "r2_f3",
    key_truth = r2_f3
  )
}

all_designs <- function() {
  designs <- list(
    one_factor_design("f1_p3", c(0.9, 0.3, 0.9)),
    one_factor_design("f1_p5", c(0.9, 0.3, 0.9, 0.3, 0.9)),
    two_factor_design("f2_r90", 0.90),
    two_factor_design("f2_r97", 0.97),
    path_design("path_r2_90", 0.9)
  )
  names(designs) <- vapply(designs, `[[`, "", "key")
  for (i in seq_along(designs)) designs[[i]]$id <- i
  designs
}

simulate_data <- function(design, n, seed) {
  set.seed(seed)
  Sigma <- design$Sigma
  X <- matrix(stats::rnorm(n * ncol(Sigma)), n) %*% chol(Sigma)
  colnames(X) <- colnames(Sigma)
  as.data.frame(X)
}

simulation_seed <- function(seed_base, design_id, n, rep) {
  as.integer(seed_base + design_id * 1e7 + n * 1e4 + rep)
}
