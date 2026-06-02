# Deng & Chan (2017) alpha-vs-omega machinery.
#
# Two reliability coefficients for a unidimensional scale of p items, both
# treated as smooth functions of the sample covariance s = vech(S):
#
#   alpha(S) = p/(p-1) * (1 - tr(S) / 1'S1)            -- model-free, closed form
#   omega    = (1'lam)^2 / ((1'lam)^2 + 1'Psi 1)        -- from a one-factor fit
#
# The difference omega - alpha has standard error tau/sqrt(n) where, following
# Deng & Chan (2017, Appendix A),
#
#   tau^2 = (omega_dot' P - alpha_dot') Gamma (P' omega_dot - alpha_dot),
#   P     = (sigma_dot' W sigma_dot)^{-1} sigma_dot' W,   W = Gamma_NT(Sigma_hat)^{-1}.
#
# Gamma is the asymptotic covariance of sqrt(n) s: normal-theory (Gamma_NT(S))
# for the "nm" SE, empirical fourth moments (ADF) for the sandwich "sw" SE.
#
# Everything below uses the column-major lower-triangular vech ordering, which
# is magmaan's Gamma ordering (verified to machine precision against the
# hand-built 2 D^+ (S (x) S) D^+').

# magmaan's Gamma builders are internal Rcpp bindings (not exported).
.gamma_nt  <- magmaan:::infer_gamma_nt        # normal-theory Gamma at a Sigma
.gamma_adf <- magmaan:::infer_empirical_gamma  # ADF Gamma from a raw data matrix

vech <- function(M) M[lower.tri(M, diag = TRUE)]

mle_cov <- function(d) {
  d <- as.matrix(d)
  stats::cov(d) * (nrow(d) - 1) / nrow(d)
}

alpha_from_S <- function(S) {
  p <- ncol(S)
  (p / (p - 1)) * (1 - sum(diag(S)) / sum(S))
}

# Gradient of alpha wrt vech(S). Uses Deng & Chan's a, b weight vectors:
# a marks diagonal positions (1), b marks them too but weights off-diagonals
# by 2, so a'sigma = tr(S) and b'sigma = 1'S1.
alpha_grad_s <- function(S) {
  p <- ncol(S)
  pstar <- p * (p + 1) / 2
  a <- numeric(pstar); b <- numeric(pstar); k <- 0L
  for (j in seq_len(p)) for (i in j:p) {
    k <- k + 1L
    if (i == j) { a[k] <- 1; b[k] <- 1 } else { b[k] <- 2 }
  }
  sig <- vech(S); as <- sum(a * sig); bs <- sum(b * sig)
  -(p / (p - 1)) * (a / bs - as * b / bs^2)
}

# Model Jacobian dvech(Sigma)/dtheta for the std.lv one-factor model
# Sigma = lam lam' + diag(psi), theta = (lam_1..lam_p, psi_1..psi_p). Built by
# forward difference on the closed-form Sigma(theta) so it shares the exact
# vech ordering used everywhere else (no analytic-vs-numeric ambiguity).
sigma_dot <- function(lam, psi, h = 1e-6) {
  p <- length(lam); pstar <- p * (p + 1) / 2
  Sig <- function(l, ps) tcrossprod(l) + diag(ps)
  base <- vech(Sig(lam, psi))
  J <- matrix(0, pstar, 2 * p)
  for (k in seq_len(p)) {
    l2 <- lam; l2[k] <- l2[k] + h
    J[, k] <- (vech(Sig(l2, psi)) - base) / h
  }
  for (k in seq_len(p)) {
    p2 <- psi; p2[k] <- p2[k] + h
    J[, p + k] <- (vech(Sig(lam, p2)) - base) / h
  }
  J
}

# Gradient of omega wrt theta = (lam, psi) (Deng & Chan 2017, Appendix A).
omega_grad_theta <- function(lam, psi) {
  A <- sum(lam)^2; B <- sum(psi); d <- (A + B)^2
  c(rep(2 * sum(lam) * B / d, length(lam)),
    rep(-A / d, length(psi)))
}

omega_from_lp <- function(lam, psi) sum(lam)^2 / (sum(lam)^2 + sum(psi))

# Fit the congeneric (free-loading) one-factor model with the factor variance
# fixed at 1 (std.lv), matching Deng & Chan's omega parameterization.
fit_congeneric <- function(d, estimator = "ML") {
  ov <- colnames(d)
  m <- paste0("f =~ NA*", ov[1L], " + ", paste(ov[-1L], collapse = " + "),
              "\n f ~~ 1*f")
  fit <- magmaan::magmaan(m, as.data.frame(d), estimator = estimator)
  .lp_from_fit(fit)
}

# Fit the essentially tau-equivalent (equal-loading) one-factor model, std.lv.
# auto_fix_first = FALSE is essential: otherwise the marker constraint pins the
# first loading to 1 and, through the shared label, forces every loading to 1.
fit_tau <- function(d, estimator = "ML") {
  ov <- colnames(d)
  m <- paste0("f =~ ", paste0("a*", ov, collapse = " + "), "\n f ~~ 1*f")
  fit <- magmaan::magmaan(m, as.data.frame(d), estimator = estimator,
                          auto_fix_first = FALSE)
  .lp_from_fit(fit)
}

.lp_from_fit <- function(fit) {
  pt <- fit$partable
  is_err <- pt$op == "~~" & pt$lhs == pt$rhs & pt$lhs != "f"
  list(fit = fit,
       lam = pt$est[pt$op == "=~"],
       psi = pt$est[is_err],
       converged = isTRUE(fit$converged))
}

# Deng & Chan reliability-difference test on a raw data frame. Returns the two
# coefficients, their difference, and the normal-theory ("nm") and sandwich
# ("sw") standard errors / z statistics / two-sided p-values.
deng_chan_test <- function(d) {
  d <- as.matrix(d)
  n <- nrow(d); S <- mle_cov(d)
  cg <- fit_congeneric(d, "ML")
  lam <- cg$lam; psi <- cg$psi
  Sig_hat <- tcrossprod(lam) + diag(psi)
  omega <- omega_from_lp(lam, psi)
  alpha <- alpha_from_S(S)

  W  <- solve(.gamma_nt(Sig_hat))
  sd <- sigma_dot(lam, psi)
  A  <- t(sd) %*% W %*% sd
  grad_omega_s <- W %*% sd %*% solve(A, omega_grad_theta(lam, psi))  # P' omega_dot
  g <- as.numeric(grad_omega_s) - alpha_grad_s(S)

  tau2_nm <- as.numeric(t(g) %*% .gamma_nt(S)   %*% g)
  tau2_sw <- as.numeric(t(g) %*% .gamma_adf(d)  %*% g)
  se_nm <- sqrt(tau2_nm / n); se_sw <- sqrt(tau2_sw / n)
  diff <- omega - alpha
  z_nm <- diff / se_nm; z_sw <- diff / se_sw
  list(omega = omega, alpha = alpha, diff = diff,
       se_nm = se_nm, se_sw = se_sw,
       z_nm = z_nm, z_sw = z_sw,
       p_nm = 2 * stats::pnorm(-abs(z_nm)),
       p_sw = 2 * stats::pnorm(-abs(z_sw)),
       converged = cg$converged)
}

# omega read off a ULS-fitted tau-equivalent model. This equals the textbook
# sample alpha exactly (ULS reproduces the diagonal and sets the common
# covariance to the off-diagonal mean), which is the experiment's identity check.
omega_uls_tau <- function(d) {
  lp <- fit_tau(d, "ULS")
  omega_from_lp(lp$lam, lp$psi)
}

# Nonparametric bootstrap SE of (omega_hat - alpha_hat), the gold-standard
# cross-check for the analytic sandwich SE.
bootstrap_diff_se <- function(d, B = 1000L, seed = 1L) {
  d <- as.matrix(d); n <- nrow(d)
  set.seed(seed)
  diffs <- numeric(B)
  for (b in seq_len(B)) {
    db <- d[sample.int(n, replace = TRUE), , drop = FALSE]
    cg <- tryCatch(fit_congeneric(db, "ML"), error = function(e) NULL)
    if (is.null(cg) || !cg$converged) { diffs[b] <- NA_real_; next }
    diffs[b] <- omega_from_lp(cg$lam, cg$psi) - alpha_from_S(mle_cov(db))
  }
  stats::sd(diffs, na.rm = TRUE)
}
