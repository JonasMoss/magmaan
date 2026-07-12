library(magmaan)

set.seed(20260712)
lambda <- c(1, 0.8, 0.7, 0.9)
Sigma <- tcrossprod(lambda) + diag(c(0.6, 0.7, 0.8, 0.5))
X <- matrix(rnorm(240 * 4), 240, 4) %*% chol(Sigma)
dat <- as.data.frame(X)
names(dat) <- paste0("x", 1:4)

h1 <- magmaan("f =~ x1 + a*x2 + b*x3 + x4", dat,
              estimator = "ML", se = "none", test = "none")
h0 <- magmaan("f =~ x1 + a*x2 + b*x3 + x4\na == b", dat,
              estimator = "ML", se = "none", test = "none")

a <- score_flip_test(h1, h0, dat, n_flips = 63, seed = 17)
b <- score_flip_test(h1, h0, dat, n_flips = 63, seed = 17)
stopifnot(
  inherits(a, "magmaan_score_flip_test"),
  identical(a$p_basic, b$p_basic),
  identical(a$p_effective, b$p_effective),
  identical(a$p_standardized, b$p_standardized),
  identical(a$p_value, a$p_standardized),
  a$df == 1L,
  a$n_flips == 63L,
  all(c(a$p_basic, a$p_effective, a$p_standardized) >= 1 / 64),
  abs(a$statistic_effective - a$statistic_standardized) < 1e-8,
  a$p_effective == a$p_standardized,
  a$mean_variance_relative_shift == 0,
  a$max_variance_relative_shift == 0,
  a$total_seconds >= a$setup_seconds + a$resampling_score_seconds +
    a$resampling_standardization_seconds + a$asymptotic_seconds
)

print(a)
