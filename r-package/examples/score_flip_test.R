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
  isTRUE(a$sandwich_available),
  is.finite(a$sandwich_condition),
  abs(a$statistic_sandwich - a$statistic_mean_scaled) < 1e-8,
  abs(a$p_sandwich - a$p_mean_scaled) < 1e-12,
  abs(a$p_sandwich - a$p_mixture) < 1e-12,
  a$mean_variance_relative_shift == 0,
  a$max_variance_relative_shift == 0,
  a$total_seconds >= a$setup_seconds + a$resampling_score_seconds +
    a$resampling_standardization_seconds + a$asymptotic_seconds
)

print(a)

# Direct FIML uses the observed-data score and the pattern-conditional Fisher
# information. The fitted raw-data objects are the sample contract, so no data
# argument is passed to score_flip_test().
dat_mis <- dat
dat_mis$x3[seq(1, nrow(dat_mis), by = 4)] <- NA_real_
dat_mis$x4[seq(2, nrow(dat_mis), by = 5)] <- NA_real_
f1 <- magmaan("f =~ x1 + a*x2 + b*x3 + x4", dat_mis,
              estimator = "FIML", se = "none", test = "none")
f0 <- magmaan("f =~ x1 + a*x2 + b*x3 + x4\na == b", dat_mis,
              estimator = "FIML", se = "none", test = "none")
fa <- score_flip_test(f1, f0, n_flips = 63, seed = 19)
fb <- score_flip_test(f1, f0, n_flips = 63, seed = 19)
stopifnot(
  identical(fa$p_basic, fb$p_basic),
  identical(fa$p_effective, fb$p_effective),
  identical(fa$p_standardized, fb$p_standardized),
  fa$mean_variance_relative_shift > 0,
  abs(fa$statistic_effective - fa$statistic_standardized) < 1e-8,
  is.finite(fa$p_mixture),
  isTRUE(fa$sandwich_available)
)
