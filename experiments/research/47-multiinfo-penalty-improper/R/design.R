# Population designs. Every population is standardized (unit observed and
# latent variances), so parameters are comparable across designs.
#
# kind: "interior" (truth inside the admissible set), "face" (truth on a face:
# an error-free indicator, a unit factor correlation, or a latent R-squared of
# one), "improper" (the covariance matrix equals Sigma(theta) for an improper
# theta, so only a test of the proper model can reject), and "misfit" (an
# omitted residual covariance makes the pseudo-true value improper and the
# model misfit).

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

# Key quantity: the residual variance of x1 (1 - lambda_1^2 in the
# population), a single named parameter.
one_factor_design <- function(key, lambda, kind = "interior") {
  p <- length(lambda)
  list(
    key = key,
    kind = kind,
    label = sprintf("One factor, loadings (%s)",
                    paste(format(lambda, nsmall = 1), collapse = ", ")),
    syntax = paste0("f =~ ", paste(ov_names(p), collapse = " + ")),
    Sigma = lisrel_sigma(matrix(lambda, p, 1), matrix(0, 1, 1), diag(1)),
    lv = "f",
    key_quantity = "theta_x1"
  )
}

# resid_cov > 0 adds the omitted residual covariances x1-x4, x2-x5, x3-x6.
two_factor_design <- function(key, r, lambda = 0.7, resid_cov = 0,
                              kind = "interior") {
  Lambda <- cbind(c(rep(lambda, 3), rep(0, 3)), c(rep(0, 3), rep(lambda, 3)))
  Sigma <- lisrel_sigma(Lambda, matrix(0, 2, 2), matrix(c(1, r, r, 1), 2))
  for (k in 1:3) {
    Sigma[k, k + 3] <- Sigma[k + 3, k] <- Sigma[k, k + 3] + resid_cov
  }
  list(
    key = key,
    kind = kind,
    label = sprintf("Two factors, loadings %.1f, factor correlation %.2f%s",
                    lambda, r,
                    if (resid_cov > 0) sprintf(", omitted residual covariances %.2f",
                                               resid_cov) else ""),
    syntax = "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6",
    Sigma = Sigma,
    lv = c("f1", "f2"),
    key_quantity = "factor_correlation"
  )
}

# f1 -> f2 -> f3 with f1 -> f3; R^2(f2) = .5, loadings .7.
path_design <- function(key, r2_f3 = 0.9, lambda = 0.7, kind = "interior") {
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
    kind = kind,
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
    key_quantity = "r2_f3"
  )
}

all_designs <- function() {
  designs <- list(
    one_factor_design("f1_p3", c(0.9, 0.3, 0.9)),
    one_factor_design("f1_p5", c(0.9, 0.3, 0.9, 0.3, 0.9)),
    two_factor_design("f2_r90", 0.90),
    two_factor_design("f2_r97", 0.97),
    path_design("path_r2_90", 0.9),
    # Appended so the first five keep their seeds.
    one_factor_design("f1_p3_face", c(1.0, 0.3, 0.9), kind = "face"),
    one_factor_design("f1_p5_face", c(1.0, 0.3, 0.9, 0.3, 0.9), kind = "face"),
    two_factor_design("f2_r99", 0.99),
    two_factor_design("f2_r100", 1.00, kind = "face"),
    path_design("path_r2_100", 1.0, kind = "face"),
    one_factor_design("imp_f1_p5", c(1.03, 0.3, 0.9, 0.3, 0.9), kind = "improper"),
    two_factor_design("imp_f2_r103", 1.03, kind = "improper"),
    two_factor_design("mis_f2_resid", 0.95, resid_cov = 0.10, kind = "misfit")
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
