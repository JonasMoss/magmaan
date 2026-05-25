library(magmaan)
library(lavaan)

# Van Praag pairwise covariance + casewise influence functions Ψ̂ for
# incomplete continuous data — the building blocks the ugamma-fast paper's
# section on pairwise incomplete data composes into a reduced robust spectrum
# through the same `reduced_gamma_sample` machinery as the complete-data case.

set.seed(20260525)
n <- 500
mu_true <- c(0, 0, 0, 0)
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

X[sample(n, 50), 1] <- NA
X[sample(n, 40), 2] <- NA
X[sample(n, 60), 3] <- NA
X[sample(n, 30), 4] <- NA
X <- X[rowSums(!is.na(X)) > 0, , drop = FALSE]
n_eff <- nrow(X)

mask <- !is.na(X)
storage.mode(mask) <- "logical"

pw  <- magmaan_core$data_pairwise_sample_stats(X, mask)
psi <- magmaan_core$robust_pairwise_casewise_contributions(X, mask)

# ---- Pairwise covariance parity vs `stats::cov(use="pairwise")` -----------
# magmaan uses an N-divisor pairwise covariance (matches lavaan's
# `likelihood = "normal"` convention everywhere else in the package). R's
# stats::cov (and lavaan::lavCor with missing="pairwise") use an N-1 divisor
# on the *overlap* counts. Convert before comparing.
ref_cov   <- stats::cov(X, use = "pairwise.complete.obs")
mag_cov   <- pw$S[[1L]]
rescaled  <- mag_cov * pw$n_pair[[1L]] / pmax(pw$n_pair[[1L]] - 1L, 1L)
cov_diff  <- max(abs(rescaled - ref_cov))
cat(sprintf("max abs diff vs stats::cov (after N→N−1 rescale): %.3e\n", cov_diff))
stopifnot(cov_diff < 1e-10)

# ---- Lavaan parity --------------------------------------------------------
lc <- lavaan::lavCor(as.data.frame(X), missing = "pairwise", output = "cov")
lav_diff <- max(abs(rescaled - as.matrix(lc)))
cat(sprintf("max abs diff vs lavaan::lavCor(missing='pairwise'): %.3e\n", lav_diff))
stopifnot(lav_diff < 1e-10)

# ---- Influence-function invariants ----------------------------------------
# colMeans of Ψ̂ are zero to floating-point — each component's pairwise
# residuals already sum to zero inside the overlap.
psi_means_max <- max(abs(colMeans(psi)))
cat(sprintf("max abs col-mean of Ψ̂: %.3e\n", psi_means_max))
stopifnot(psi_means_max < 1e-10)

# Empirical pairwise Γ̂ from Ψ̂ matches the textbook construction (column
# crossproduct / n).  Symmetric & PSD.
Gamma <- crossprod(psi) / n_eff
sym_resid <- max(abs(Gamma - t(Gamma)))
ev <- eigen(Gamma, symmetric = TRUE, only.values = TRUE)$values
cat(sprintf("Γ̂^pw symmetry residual: %.3e ; min eigenvalue: %.3e\n",
            sym_resid, min(ev)))
stopifnot(sym_resid < 1e-12, min(ev) > -1e-10)

# ---- G3b layout (μ + σ-vech) ---------------------------------------------
# include_means = TRUE adds the μ-segment as the first p columns per block,
# matching the layout `casewise_contributions(include_means = TRUE)` uses for
# the complete-data path. The μ scaling is the marginal counterpart of the
# σ-vech scaling: ψ̂_tj_μ = (R_tj / π̂_j) · (x_tj − m̄_j).
psi_g <- magmaan_core$robust_pairwise_casewise_contributions(
  X, mask, include_means = TRUE)
p <- ncol(X)
cat(sprintf("Ψ̂ (G3b) shape: %dx%d  (expected %dx%d)\n",
            nrow(psi_g), ncol(psi_g), n_eff, p + p * (p + 1) / 2))
stopifnot(ncol(psi_g) == p + p * (p + 1) / 2,
          nrow(psi_g) == n_eff)

# μ-segment column means are zero by construction (Σ_t R_tj (x_tj − m̄_j) = 0
# when m̄_j is the marginal mean; the marginal-availability rescaling
# preserves the zero sum).
mu_col_mean_max <- max(abs(colSums(psi_g[, seq_len(p), drop = FALSE])))
cat(sprintf("max abs μ column sum: %.3e\n", mu_col_mean_max))
stopifnot(mu_col_mean_max < 1e-10)

# μ-segment is zero where the corresponding variable is missing.
for (j in seq_len(p)) {
  unobs_rows <- which(!mask[, j])
  if (length(unobs_rows) > 0) {
    stopifnot(all(psi_g[unobs_rows, j] == 0))
  }
}

# Full empirical Γ̂^pw with both μ and σ blocks: symmetric PSD.
Gamma_g <- crossprod(psi_g) / n_eff
stopifnot(max(abs(Gamma_g - t(Gamma_g))) < 1e-12,
          min(eigen(Gamma_g, symmetric = TRUE, only.values = TRUE)$values)
            > -1e-10)
cat(sprintf("Γ̂^pw (G3b) min eigenvalue: %.3e\n",
            min(eigen(Gamma_g, symmetric = TRUE, only.values = TRUE)$values)))

# ---- Complete-data degeneracy ---------------------------------------------
# When `mask` says everything is observed, Ψ̂ collapses to the same casewise
# residual matrix the complete-data path produces and π̂ ≡ 1. The C++ unit
# tests already check exact equality with `robust::casewise_contributions`;
# here we just spot-check the all-observed limit.
set.seed(123)
X_full  <- matrix(rnorm(n * 4), n, 4) %*% L
pw_full <- magmaan_core$data_pairwise_sample_stats(X_full)
stopifnot(max(abs(pw_full$pi_hat[[1L]] - 1)) < 1e-12,
          all(pw_full$n_pair[[1L]] == n))

cat("\npairwise_casewise example: PASS\n")
