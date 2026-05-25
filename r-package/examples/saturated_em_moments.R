library(magmaan)
library(lavaan)

# Stage-1 of Savalei & Bentler (2009) "A two-stage approach to missing data":
# the saturated EM mean and covariance under MAR, together with the sandwich
# ACOV needed by a downstream second-stage SEM. magmaan exposes this as a
# methods-developer surface — no partable required, just raw data.

set.seed(20260525)
n <- 400
mu_true <- c(1.0, 2.0, 3.0, 4.0)
Sigma_true <- matrix(
  c(1.0, 0.4, 0.3, 0.2,
    0.4, 1.0, 0.5, 0.3,
    0.3, 0.5, 1.0, 0.4,
    0.2, 0.3, 0.4, 1.0),
  4, 4)
L <- chol(Sigma_true)
X <- matrix(rnorm(n * 4), n, 4) %*% L
X <- sweep(X, 2, mu_true, "+")
colnames(X) <- c("y1", "y2", "y3", "y4")

X[sample(n, 40), 1] <- NA
X[sample(n, 35), 2] <- NA
X[sample(n, 50), 3] <- NA
X[sample(n, 25), 4] <- NA
keep <- rowSums(!is.na(X)) > 0
X <- X[keep, , drop = FALSE]
n_eff <- nrow(X)

mask <- !is.na(X)
storage.mode(mask) <- "logical"

# Run magmaan's saturated EM.
mag <- magmaan_core$estimate_saturated_em_moments(list(X = X, mask = mask))
stopifnot(length(mag$mean) == 1L,
          length(mag$cov) == 1L,
          mag$n_obs == n_eff)

# Run lavaan's saturated EM at a tight tolerance for parity comparison.
mp <- lavaan:::lav_data_missing_patterns(X)
lav <- lavaan:::lav_mvnorm_missing_h1_estimate_moments(
  Y = X, Mp = mp, tol = 1e-12, max.iter = 10000L)

mean_diff <- max(abs(mag$mean[[1L]] - lav$Mu))
cov_diff  <- max(abs(mag$cov[[1L]] - lav$Sigma))
cat(sprintf("max abs diff (mean) vs lavaan: %.3e\n", mean_diff))
cat(sprintf("max abs diff (cov)  vs lavaan: %.3e\n", cov_diff))
stopifnot(mean_diff < 1e-6, cov_diff < 1e-6)

# Sandwich identity:  H · ACOV · H  ≡  J  (within floating-point round-off).
sandwich_resid <- max(abs(mag$H %*% mag$acov %*% mag$H - mag$J))
cat(sprintf("sandwich H·ACOV·H − J residual:  %.3e\n", sandwich_resid))
stopifnot(sandwich_resid < 1e-8)

# Symmetry / PSD invariants.
stopifnot(max(abs(mag$H    - t(mag$H)))    < 1e-8,
          max(abs(mag$J    - t(mag$J)))    < 1e-8,
          max(abs(mag$acov - t(mag$acov))) < 1e-10)
ev_H <- eigen(mag$H, symmetric = TRUE, only.values = TRUE)$values
ev_A <- eigen(mag$acov, symmetric = TRUE, only.values = TRUE)$values
stopifnot(min(ev_H) > 0, min(ev_A) > -1e-10)

cat("\nsaturated_em_moments example: PASS\n")
