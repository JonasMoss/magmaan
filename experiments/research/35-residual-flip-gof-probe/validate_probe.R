#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "R", "design.R"))
source(file.path(script_dir, "R", "gof_flip.R"))

set.seed(6501)
pop <- gof_population(5L, "null")
X <- matrix(stats::rnorm(80L * 5L), 80L, 5L) %*% chol(pop$Sigma)
colnames(X) <- paste0("x", seq_len(5L))
fit <- magmaan(gof_model_spec(5L), as.data.frame(X), estimator = "ML",
               se = "none", test = "none")
stopifnot(isTRUE(fit$converged))

flip <- residual_flip_gof(fit, X, n_flips = 63L, seed = 6502L)
fmg <- fmg_tests(fit, tests = "std_rls", data = X)
stopifnot(
  flip$df == 5L,
  abs(flip$statistic_effective - fmg$base_statistic[[1L]]) < 1e-10,
  flip$p_effective >= 1 / 64, flip$p_effective <= 1,
  isTRUE(flip$standardization_available))

core <- magmaan::magmaan_core
uf <- core$infer_build_u_factor(fit, "expected", "structured")
zc <- core$infer_casewise_contributions(fit$partable, X)
resid <- uf$blocks[[1L]]$S - uf$blocks[[1L]]$Sigma_hat
rows <- sweep(zc, 2L, resid[lower.tri(resid, diag = TRUE)], "+") %*% uf$B
Q <- qr.Q(qr(matrix(stats::rnorm(25L), 5L, 5L)))
a <- .flip_projected_rows(rows, 63L, 6503L)
b <- .flip_projected_rows(rows %*% Q, 63L, 6503L)
stopifnot(
  abs(a$statistic_effective - b$statistic_effective) < 1e-10,
  abs(a$statistic_standardized - b$statistic_standardized) < 1e-9,
  identical(a$p_effective, b$p_effective),
  identical(a$p_standardized, b$p_standardized))

# At df >= n the unregularized empirical covariance cannot be inverted. The
# probe records that limitation rather than silently adding a ridge choice.
wide <- .flip_projected_rows(matrix(stats::rnorm(40L * 50L), 40L, 50L),
                             19L, 6504L)
stopifnot(!wide$standardization_available, is.na(wide$p_standardized))

cat("residual flip GOF validation passed\n")
