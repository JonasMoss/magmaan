# Self-contained bifactor Gaussian-ML machinery for the profile likelihood-ratio
# interval. The maximal-reliability constraint g(theta) = rho*_0 is nonlinear, so
# the constrained fit is done here with nloptr SLSQP (gradient-based, native
# equality constraint) rather than through magmaan. F_ML and its gradient are
# analytic (dF/dSigma = Sigma^-1 - Sigma^-1 S Sigma^-1, chained to Lambda, Psi);
# the constraint Jacobian is finite-differenced. Requires R/maxrel.R (rho_olc /
# rho_olsc) to be sourced first. make_lr_ml(pop) returns the closures for one
# population layout; the LR for testing g = rho0 is N (F_con - F_unc), ~ chi^2_1.

make_lr_ml <- function(pop) {
  p <- pop$p; k <- pop$k; items <- pop$items
  ng <- sum(lengths(items))
  unpack <- function(z) {
    Lam <- matrix(0, p, k + 1); Lam[, 1] <- z[1:p]
    pos <- p
    for (j in seq_len(k)) { ix <- items[[j]]
      Lam[ix, j + 1] <- z[(pos + 1):(pos + length(ix))]; pos <- pos + length(ix) }
    list(Lam = Lam, psi = exp(z[(p + ng + 1):(p + ng + p)]))
  }
  sigma_of <- function(z) { u <- unpack(z); tcrossprod(u$Lam) + diag(u$psi) }
  fml <- function(z, S, logdetS) {
    Sig <- sigma_of(z)
    ch <- tryCatch(chol(Sig), error = function(e) NULL)
    if (is.null(ch)) return(1e6)
    Si <- chol2inv(ch)
    2 * sum(log(diag(ch))) + sum(S * Si) - logdetS - p
  }
  grad_fml <- function(z, S) {
    u <- unpack(z); Lam <- u$Lam; psi <- u$psi
    Si <- solve(tcrossprod(Lam) + diag(psi))
    G <- Si - Si %*% S %*% Si
    GL <- 2 * (G %*% Lam)
    c(GL[, 1],
      unlist(lapply(seq_len(k), function(j) GL[items[[j]], j + 1])),
      diag(G) * psi)
  }
  g_of <- function(z, which) {
    u <- unpack(z); Sig <- tcrossprod(u$Lam) + diag(u$psi)
    if (which == "gen") rho_olc(Sig, u$Lam[, 1])
    else rho_olsc(Sig, u$Lam[, 2], items[[1]])
  }
  zeta_from <- function(Lam, psi)
    c(Lam[, 1], unlist(lapply(seq_len(k), function(j) Lam[items[[j]], j + 1])), log(psi))

  opts_unc <- list(algorithm = "NLOPT_LD_LBFGS", xtol_rel = 1e-8, maxeval = 500)
  opts_con <- list(algorithm = "NLOPT_LD_SLSQP", xtol_rel = 1e-8, maxeval = 500)

  fit_unc <- function(S, logdetS, z0) {
    r <- nloptr::nloptr(z0, eval_f = function(z)
      list(objective = fml(z, S, logdetS), gradient = grad_fml(z, S)), opts = opts_unc)
    list(F = r$objective, z = r$solution)
  }
  # constrained min of F_ML s.t. g = rho0; NULL if the constraint is not met.
  fit_con <- function(S, logdetS, z0, which, rho0) {
    heq <- function(z) g_of(z, which) - rho0
    jeq <- function(z) { e <- 1e-5
      matrix(vapply(seq_along(z), function(i) { zp <- z; zm <- z
        zp[i] <- zp[i] + e; zm[i] <- zm[i] - e
        (g_of(zp, which) - g_of(zm, which)) / (2 * e) }, numeric(1)), nrow = 1) }
    r <- tryCatch(nloptr::nloptr(z0,
      eval_f = function(z) list(objective = fml(z, S, logdetS), gradient = grad_fml(z, S)),
      eval_g_eq = function(z) list(constraints = heq(z), jacobian = jeq(z)),
      opts = opts_con), error = function(e) NULL)
    if (is.null(r) || abs(heq(r$solution)) > 1e-4) return(NULL)
    list(F = r$objective, z = r$solution)
  }
  list(unpack = unpack, sigma_of = sigma_of, fml = fml, g_of = g_of,
       zeta_from = zeta_from, fit_unc = fit_unc, fit_con = fit_con)
}
