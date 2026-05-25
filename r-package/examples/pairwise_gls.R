library(magmaan)
library(lavaan)

# Pairwise GLS, two variants:
#   estimate_gls       (literature default; Σ-only weight Γ_NT(Ŝ^pw)⁻¹) —
#                      Savalei-Bentler 2005 / Gold-Bentler-Kim 2003. Trace
#                      identity, consistent, asymptotically suboptimal under
#                      MAR.
#   estimate_gls_pairwise (Γ_NT^pw weight) — breaks the trace identity,
#                      asymptotically efficient under MAR.
# Both reduce to the same fit on complete data; they diverge under
# missingness — the gap that motivates a downstream efficiency study.

set.seed(20260605)
n <- 600
# 5-indicator CFA so the model is *overidentified* — with a 3-indicator
# saturated CFA the weight makes no difference (F = 0 at the MLE for any W).
lam <- c(1.0, 0.8, 0.9, 0.7, 1.1)
psi <- 1.5
theta_res <- c(0.7, 0.6, 0.5, 0.8, 0.4)
p <- length(lam)
Sigma <- psi * tcrossprod(lam)
diag(Sigma) <- diag(Sigma) + theta_res
mu_true <- rep(0, p)

X_full <- t(t(matrix(rnorm(n * p), n, p) %*% chol(Sigma)) + mu_true)
colnames(X_full) <- paste0("x", seq_len(p))

model <- "f =~ x1 + x2 + x3 + x4 + x5"
partable <- lavaan::lavaanify(model, fixed.x = FALSE,
                              auto.var = TRUE, auto.fix.first = TRUE,
                              auto.cov.lv.x = TRUE)

# ---- Complete-data parity: both fits agree ---------------------------------
pw_full <- magmaan_core$data_pairwise_sample_stats(X_full)
sample_full <- list(
  S = pw_full$S,
  mean = pw_full$mean,
  nobs = pw_full$nobs
)
fit_a <- magmaan_core$estimate_gls(partable, sample_full)
fit_b <- magmaan_core$estimate_gls_pairwise(partable, X_full)
free_a <- fit_a$partable$est[fit_a$partable$free > 0L]
free_b <- fit_b$partable$est[fit_b$partable$free > 0L]
cmplt_diff <- max(abs(free_a - free_b))
cat(sprintf("complete-data θ̂ disagreement (Σ-only vs Γ_NT^pw): %.3e\n",
            cmplt_diff))
stopifnot(cmplt_diff < 1e-5)

# ---- Materialized Γ_NT^pw is symmetric PD ---------------------------------
G_full <- magmaan_core$data_gamma_nt_pairwise(X_full)[[1L]]
G_full_sym <- max(abs(G_full - t(G_full)))
G_full_min_eig <- min(eigen(G_full, symmetric = TRUE, only.values = TRUE)$values)
cat(sprintf("complete-data Γ_NT^pw: symmetry residual %.3e, min eig %.3f\n",
            G_full_sym, G_full_min_eig))
stopifnot(G_full_sym < 1e-12, G_full_min_eig > 0)

# ---- MAR data: the two variants diverge ------------------------------------
# Drop ~12% MCAR per column, never leaving a row fully blank.
X <- X_full
miss_rate <- 0.12
for (j in seq_len(ncol(X))) {
  X[runif(n) < miss_rate, j] <- NA
}
keep <- rowSums(!is.na(X)) > 0
X <- X[keep, , drop = FALSE]
n_eff <- nrow(X)
mask <- !is.na(X); storage.mode(mask) <- "logical"

pw <- magmaan_core$data_pairwise_sample_stats(X, mask)
sample_pw <- list(S = pw$S, mean = pw$mean, nobs = pw$nobs)

fit_a_miss <- magmaan_core$estimate_gls(partable, sample_pw)
fit_b_miss <- magmaan_core$estimate_gls_pairwise(partable, X, mask)
free_a_miss <- fit_a_miss$partable$est[fit_a_miss$partable$free > 0L]
free_b_miss <- fit_b_miss$partable$est[fit_b_miss$partable$free > 0L]
miss_diff <- max(abs(free_a_miss - free_b_miss))
cat(sprintf("MAR-data θ̂ disagreement: %.3e (non-trivial; both consistent, weights differ)\n",
            miss_diff))

# ---- Materialized Γ_NT^pw under missingness --------------------------------
G_miss <- magmaan_core$data_gamma_nt_pairwise(X, mask)[[1L]]
G_miss_sym <- max(abs(G_miss - t(G_miss)))
G_miss_min_eig <- min(eigen(G_miss, symmetric = TRUE, only.values = TRUE)$values)
cat(sprintf("MAR Γ_NT^pw: symmetry residual %.3e, min eig %.3f\n",
            G_miss_sym, G_miss_min_eig))
stopifnot(G_miss_sym < 1e-10, G_miss_min_eig > 0)

# Sanity: complete-data Γ_NT^pw matches the closed-form gamma_nt on Ŝ_pw.
G_close <- magmaan_core$robust_gamma_nt(pw_full$S[[1L]])
gnt_diff <- max(abs(G_full - G_close))
cat(sprintf("complete-data Γ_NT^pw vs gamma_nt(Ŝ): %.3e\n", gnt_diff))
stopifnot(gnt_diff < 1e-12)

cat("\npairwise_gls example: PASS\n")
