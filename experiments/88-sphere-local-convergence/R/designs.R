# Populations and the fitted model. Two factors with three indicators each,
# Y regressed on X, unit residual variances, and var(X) = 1.

designs_all <- function() list(
  ernst = list(label = "Ernst et al. (loadings 1, .8, .6; beta .25)",
               lx = c(1, .8, .6), ly = c(1, .8, .6), beta = .25, psi = 1),
  weak_marker = list(label = "Weak marker (loadings .1, .8, .6 on both factors)",
                     lx = c(.1, .8, .6), ly = c(.1, .8, .6), beta = .25, psi = 1),
  high_r2 = list(label = "High R2 for Y (.98; disturbance variance .02)",
                 lx = c(1, .8, .6), ly = c(1, .8, .6), beta = sqrt(.98), psi = .02))

design_sigma <- function(d) {
  L <- matrix(0, 6, 2)
  L[1:3, 1] <- d$lx
  L[4:6, 2] <- d$ly
  Phi <- matrix(c(1, d$beta, d$beta, d$beta^2 + d$psi), 2)
  S <- L %*% Phi %*% t(L)
  diag(S) <- diag(S) + 1
  dimnames(S) <- list(ov_names, ov_names)
  S
}

ov_names <- c("x1", "x2", "x3", "y1", "y2", "y3")

draw_data <- function(sigma, n, seed) {
  set.seed(seed)
  z <- matrix(stats::rnorm(n * 6L), n, 6L) %*% chol(sigma)
  colnames(z) <- ov_names
  as.data.frame(z)
}

model_syntax <- "X =~ x1 + x2 + x3\nY =~ y1 + y2 + y3\nY ~ X"

identifications <- list(marker = list(), std_lv = list(std_lv = TRUE))
