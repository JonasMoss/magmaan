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

global_a <- global_score_flip_test(h1, dat, n_flips = 63, seed = 23)
global_b <- global_score_flip_test(h1, dat, n_flips = 63, seed = 23)
stopifnot(
  inherits(global_a, "magmaan_global_score_flip_test"),
  global_a$df == 2L,
  global_a$saturated_moment_dim == 10L,
  global_a$tangent_rank == 8L,
  identical(global_a$p_effective, global_b$p_effective),
  identical(global_a$p_value, global_a$p_effective)
)

effective <- score_flip_test(
  h1, h0, dat, n_flips = 63, seed = 17, calibration = "effective")
score <- nested_score_test(h1, h0, dat)
h1_model <- model_spec("f =~ x1 + a*x2 + b*x3 + x4")
score_model <- nested_score_test(h1_model, h0, dat)
stopifnot(
  identical(effective$p_effective, a$p_effective),
  is.na(effective$p_basic),
  is.na(effective$p_standardized),
  effective$resampling_standardization_seconds == 0,
  inherits(score, "magmaan_nested_score_test"),
  score$n_flips == 0L,
  is.finite(score$p_peba4),
  identical(score$p_value, score$p_peba4),
  abs(score$statistic_effective - a$statistic_effective) < 1e-8,
  abs(score_model$statistic_effective - score$statistic_effective) < 1e-8,
  abs(score_model$p_peba4 - score$p_peba4) < 1e-12,
  score$resampling_score_seconds == 0,
  score$resampling_standardization_seconds == 0
)

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
fs <- nested_score_test(f1, f0)
f1_model <- model_spec(
  "f =~ x1 + a*x2 + b*x3 + x4", meanstructure = TRUE)
fs_model <- nested_score_test(f1_model, f0)
fg <- global_score_flip_test(f1, n_flips = 63, seed = 29)
stopifnot(
  identical(fa$p_basic, fb$p_basic),
  identical(fa$p_effective, fb$p_effective),
  identical(fa$p_standardized, fb$p_standardized),
  fa$mean_variance_relative_shift > 0,
  abs(fa$statistic_effective - fa$statistic_standardized) < 1e-8,
  is.finite(fa$p_mixture),
  isTRUE(fa$sandwich_available),
  is.finite(fs$p_peba4),
  abs(fs$statistic_effective - fa$statistic_effective) < 1e-8,
  abs(fs_model$statistic_effective - fs$statistic_effective) < 1e-8,
  abs(fs_model$p_peba4 - fs$p_peba4) < 1e-12,
  inherits(fg, "magmaan_global_score_flip_test"),
  fg$df == 2L,
  fg$saturated_moment_dim == 14L,
  fg$tangent_rank == 12L,
  is.finite(fg$p_effective)
)
