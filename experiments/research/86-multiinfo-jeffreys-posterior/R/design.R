# Population designs. The five populations of the multi-information penalty
# study, with the same seed rule, so a given (design, N, replication) draws the
# same dataset. Every population is standardized (unit observed and latent
# variances).
#
# Each design also carries its structure in standardized-latent form, which the
# posterior sampler uses: latents are ordered so parents precede children,
# exogenous latents come first, every latent has unit variance, and structural
# coefficients are standardized.

ov_names <- function(p) paste0("x", seq_len(p))

structure_info <- function(Lambda, B, exo_cor, lv) {
  p <- nrow(Lambda)
  m <- ncol(Lambda)
  dimnames(Lambda) <- list(ov_names(p), lv)
  dimnames(B) <- list(lv, lv)
  load_pat <- Lambda != 0
  list(
    ov = ov_names(p), lv = lv, p = p, m = m,
    load_pat = load_pat,
    first_ind = apply(load_pat, 2, function(col) which(col)[[1L]]),
    B_pat = B != 0,
    exo = rowSums(B != 0) == 0,
    exo_cor = exo_cor)          # two-column matrix of free exogenous correlations
}

population_phi <- function(B, Psi) {
  A <- solve(diag(nrow(B)) - B)
  A %*% Psi %*% t(A)
}

make_design <- function(key, label, syntax, Lambda, B, Psi, lv, exo_cor,
                        key_params) {
  Phi <- population_phi(B, Psi)
  theta <- 1 - rowSums((Lambda %*% Phi) * Lambda)
  Sigma <- Lambda %*% Phi %*% t(Lambda)
  diag(Sigma) <- 1
  dimnames(Sigma) <- list(ov_names(nrow(Lambda)), ov_names(nrow(Lambda)))
  list(key = key, label = label, syntax = syntax, Sigma = Sigma, lv = lv,
       Lambda = Lambda, B = B, Phi = Phi, theta = theta,
       structure = structure_info(Lambda, B, exo_cor, lv),
       key_params = key_params)
}

one_factor_design <- function(key, lambda) {
  p <- length(lambda)
  strong <- ov_names(p)[lambda == max(lambda)]
  make_design(
    key,
    sprintf("One factor, loadings (%s)", paste(format(lambda, nsmall = 1), collapse = ", ")),
    paste0("f =~ ", paste(ov_names(p), collapse = " + ")),
    matrix(lambda, p, 1), matrix(0, 1, 1), diag(1), "f",
    exo_cor = matrix(integer(0), 0, 2),
    key_params = paste(strong, "~~", strong))
}

two_factor_design <- function(key, r, lambda = 0.7) {
  Lambda <- cbind(c(rep(lambda, 3), rep(0, 3)), c(rep(0, 3), rep(lambda, 3)))
  make_design(
    key,
    sprintf("Two factors, loadings %.1f, factor correlation %.2f", lambda, r),
    "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6",
    Lambda, matrix(0, 2, 2), matrix(c(1, r, r, 1), 2), c("f1", "f2"),
    exo_cor = matrix(c(1L, 2L), 1, 2),
    key_params = "f1 ~~ f2 (cor)")
}

# f1 -> f2 -> f3 with f1 -> f3; R^2(f2) = .5, R^2(f3) = r2_f3, loadings .7.
path_design <- function(key, r2_f3 = 0.9, lambda = 0.7) {
  b21 <- sqrt(0.5)
  b31 <- 0.3
  b32 <- (-2 * b31 * b21 + sqrt((2 * b31 * b21)^2 - 4 * (b31^2 - r2_f3))) / 2
  B <- matrix(0, 3, 3)
  B[2, 1] <- b21
  B[3, 1] <- b31
  B[3, 2] <- b32
  make_design(
    key,
    sprintf("Three-latent recursive path, R-squared(f3) = %.2f", r2_f3),
    paste("f1 =~ x1 + x2 + x3", "f2 =~ x4 + x5 + x6", "f3 =~ x7 + x8 + x9",
          "f2 ~ f1", "f3 ~ f1 + f2", sep = "\n"),
    kronecker(diag(3), matrix(lambda, 3, 1)), B, diag(c(1, 0.5, 1 - r2_f3)),
    c("f1", "f2", "f3"),
    exo_cor = matrix(integer(0), 0, 2),
    key_params = "R2 f3")
}

all_designs <- function() {
  designs <- list(
    one_factor_design("f1_p3", c(0.9, 0.3, 0.9)),
    one_factor_design("f1_p5", c(0.9, 0.3, 0.9, 0.3, 0.9)),
    two_factor_design("f2_r90", 0.90),
    two_factor_design("f2_r97", 0.97),
    path_design("path_r2_90", 0.9))
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
