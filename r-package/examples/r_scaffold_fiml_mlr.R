## Exploratory R scaffold over C++ primitives: FIML MLR reporting plus
## post-fit inspection. Run after installing/loading magmaan from the repo.

set.seed(20260518)
n <- 140L
eta <- stats::rnorm(n)
df <- data.frame(
  x1 = 0.9 * eta + stats::rnorm(n, sd = 0.45),
  x2 = 0.8 * eta + stats::rnorm(n, sd = 0.55),
  x3 = 0.7 * eta + stats::rnorm(n, sd = 0.65),
  x4 = 0.6 * eta + stats::rnorm(n, sd = 0.75)
)
df$x2[seq(7L, n, by = 11L)] <- NA_real_
df$x4[seq(5L, n, by = 13L)] <- NA_real_

model <- "f =~ x1 + x2 + x3 + x4"
core <- magmaan::magmaan_core

fit <- magmaan::magmaan(
  model, df,
  estimator = "FIML",
  meanstructure = TRUE,
  se = "none",
  test = "none"
)

mlr <- core$estimate_fiml_robust_mlr(fit)
stopifnot(is.matrix(mlr$vcov), length(mlr$se) == length(fit$theta))
stopifnot(is.finite(mlr$chisq_scaled), mlr$df > 0L)

std_lv <- core$measures_standardize_lv(fit, mlr$vcov)
std_all <- core$measures_standardize_all(fit, mlr$vcov)
stopifnot(length(std_lv$theta) == length(fit$theta))
stopifnot(length(std_all$theta) == length(fit$theta))

mi <- core$inference_modification_indices(fit, candidates = "all")
scores <- core$inference_score_tests(fit)
stopifnot(is.data.frame(mi), is.data.frame(scores))

cat("R scaffold FIML MLR workflow: ok\n")
cat(sprintf("  robust scaled chi-square = %.4f (df %d)\n",
            mlr$chisq_scaled, mlr$df))
cat(sprintf("  robust SE range = [%.4f, %.4f]\n",
            min(mlr$se), max(mlr$se)))
cat(sprintf("  modification-index rows = %d, score-test rows = %d\n",
            nrow(mi), nrow(scores)))
