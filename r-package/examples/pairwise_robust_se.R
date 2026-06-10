library(magmaan)
library(lavaan)

# Demonstrates the 2x2 grid of pairwise-data sandwich SE configurations:
#
#   bread \  meat   |  ModelImplied (NT)            Empirical (Ψ̂'Ψ̂/n)
#   ----------------+------------------------------------------------
#   Structured/Σ-only |  naive complete-data NT SE  | lavaan robust.sem
#                     |  (wrong under MAR)          |  (complete-data robust)
#   Pairwise          |  naive pairwise SE          | the principled pairwise +
#                     |  (collapses to A⁻¹/N)       |  non-normal robust SE
#
# The bottom-right cell is the deliverable: it uses Γ_NT^pw as the bread
# (asymptotically efficient under MAR) and the empirical pairwise meat
# Ψ̂'Ψ̂/n (robust to non-normality). The bottom-left collapses because
# bread and meat match.

set.seed(20260626)
n <- 600
lam <- c(0.85, 0.7, 0.8, 0.65, 0.9)
psi <- 1.0
theta_res <- 1.0 - lam^2 * psi
p <- length(lam)
Sigma <- psi * tcrossprod(lam); diag(Sigma) <- diag(Sigma) + theta_res
X <- matrix(rnorm(n * p), n, p) %*% chol(Sigma)
colnames(X) <- paste0("x", seq_len(p))

# Self-contained MAR: keep x1, x2 intact and make x3..x5 missing with a
# probability that rises with the intact predictors (logistic in standardized
# x1 + x2), the intercept calibrated by bisection to ~25% per target column.
# (The pairwise-robust-sem paper uses a richer SB-2005 rule design; this demo
# only needs an MAR mechanism, so it stays self-contained.)
local({
  set.seed(42)
  z <- as.numeric(scale(X[, 1] + X[, 2]))
  beta <- 1.2; rate <- 0.25
  m <- matrix(TRUE, n, p)                       # TRUE = observed
  for (k in 3:5) {
    lo <- -12; hi <- 12
    for (.i in 1:40) {
      a <- (lo + hi) / 2
      if (mean(plogis(a + beta * z)) > rate) hi <- a else lo <- a
    }
    m[runif(n) < plogis((lo + hi) / 2 + beta * z), k] <- FALSE
  }
  Xna <- X; Xna[!m] <- NA
  keep <- rowSums(m) > 0
  Xm <<- Xna[keep, , drop = FALSE]
  mask <<- m[keep, , drop = FALSE]
})
n_eff <- nrow(Xm)
cat(sprintf("n_eff = %d (after dropping fully-missing rows)\n", n_eff))

partable <- lavaan::lavaanify("f =~ x1 + x2 + x3 + x4 + x5",
                              fixed.x = FALSE, auto.var = TRUE,
                              auto.fix.first = TRUE, auto.cov.lv.x = TRUE)

# Fit via Γ_NT^pw GLS — the bread we want the U-factor to use.
fit <- magmaan_core$estimate_gls_pairwise(partable, Xm, mask)
cat(sprintf("fit converged: estimator = %s; fmin = %.6f\n",
            fit$estimator, fit$fmin))

# Pairwise bread + ModelImplied meat: should collapse to the naive SE
# (eigenvalues all = 1).
uf_pw <- magmaan_core$robust_build_u_factor_pairwise(fit, Xm, mask)
M_collapse <- magmaan_core$robust_reduced_gamma_nt_pairwise(uf_pw, Xm, mask)
ev_collapse <- magmaan_core$robust_ugamma_eigenvalues(M_collapse)
collapse_max_dev <- max(abs(ev_collapse - 1))
cat(sprintf("Pairwise bread + Pairwise meat: max |eigenvalue - 1| = %.3e\n",
            collapse_max_dev))
stopifnot(collapse_max_dev < 1e-8)

# Pairwise bread + Empirical meat (Ψ̂'Ψ̂/n) — the principled pairwise +
# non-normal robust SE. Cov-only Ψ̂ (the partable here has no mean
# structure; the UFactor's has_means is false so the meat needs to be
# σ-only too).
psi <- magmaan_core$robust_pairwise_casewise_contributions(
  Xm, mask, include_means = FALSE)
M_emp_pw <- magmaan_core$robust_reduced_gamma_sample(uf_pw, psi, n_eff)
ev_emp_pw <- magmaan_core$robust_ugamma_eigenvalues(M_emp_pw)
cat(sprintf("Pairwise bread + Empirical meat: %d eigenvalues, range [%.3f, %.3f]\n",
            length(ev_emp_pw), min(ev_emp_pw), max(ev_emp_pw)))
stopifnot(all(is.finite(ev_emp_pw)), min(ev_emp_pw) > -1e-8)

# Structured bread + Empirical meat — lavaan-style "robust.sem" on the
# pairwise sample. Reference for comparison.
uf_str <- magmaan_core$robust_build_u_factor(fit, bread = "expected",
                                              moments = "structured")
M_emp_str <- magmaan_core$robust_reduced_gamma_sample(uf_str, psi, n_eff)
ev_emp_str <- magmaan_core$robust_ugamma_eigenvalues(M_emp_str)
cat(sprintf("Structured bread + Empirical meat: %d eigenvalues, range [%.3f, %.3f]\n",
            length(ev_emp_str), min(ev_emp_str), max(ev_emp_str)))

# Sketch the four-cell summary.
cell <- function(name, eig) {
  cat(sprintf("  %-44s  scale = %.4f   max-eig = %.4f\n",
              name, mean(eig), max(eig)))
}
cat("\n[four-cell summary] mean eigenvalue is the SB scaling factor:\n")
M_nt_str <- magmaan_core$robust_reduced_gamma_nt(uf_str)
ev_nt_str <- magmaan_core$robust_ugamma_eigenvalues(M_nt_str)
cell("Structured / ModelImplied (naive NT)", ev_nt_str)
cell("Structured / Empirical (robust.sem)", ev_emp_str)
cell("Pairwise   / ModelImplied (collapse)", ev_collapse)
cell("Pairwise   / Empirical (the deliverable)", ev_emp_pw)

cat("\npairwise_robust_se example: PASS\n")
