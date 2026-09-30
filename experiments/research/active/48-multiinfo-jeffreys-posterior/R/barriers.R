# Barrier variants fitted directly in standardized-latent coordinates, where
# every latent has unit variance, so Var(eta) = Phi is already a correlation
# matrix and Theta is diagonal in these designs.
#
#   multiinfo:   log det Corr(eta, y) = log det Phi + sum_i log(theta_i / Sigma_ii)
#   determinacy: log det Var(eta | y) = log det (Phi - Phi L' Sigma^-1 L Phi)
#   two_weight:  w_y sum_i log(theta_i / Sigma_ii) + w_R log det Phi
#
# The objective matches magmaan's barrier scaling: maximize
# l(theta) + lambda * P(theta) with l = -(N/2)(log det Sigma + tr(S Sigma^-1)).

loglik_ml <- function(Sigma, S, n) {
  R <- tryCatch(chol(Sigma), error = function(e) NULL)
  if (is.null(R)) return(-Inf)
  Ri <- backsolve(R, diag(nrow(R)))
  -(n / 2) * (2 * sum(log(diag(R))) + sum((S %*% Ri) * Ri))
}

logdet_pd <- function(M) {
  R <- tryCatch(chol((M + t(M)) / 2), error = function(e) NULL)
  if (is.null(R)) -Inf else 2 * sum(log(diag(R)))
}

barrier_value <- function(kind, b, w) {
  if (any(b$theta <= 0)) return(-Inf)
  items <- sum(log(b$theta / diag(b$Sigma)))
  switch(kind,
    multiinfo = logdet_pd(b$Phi) + items,
    two_weight = w[["y"]] * items + w[["R"]] * logdet_pd(b$Phi),
    determinacy = {
      LP <- b$Lambda %*% b$Phi
      K <- tryCatch(chol2inv(chol(b$Sigma)), error = function(e) NULL)
      if (is.null(K)) -Inf else logdet_pd(b$Phi - t(LP) %*% K %*% LP)
    },
    stop("unknown barrier: ", kind))
}

barrier_configs <- function() {
  list(
    rfit_multi_l025 = list(kind = "multiinfo", lambda = 0.25),
    det_l025 = list(kind = "determinacy", lambda = 0.25),
    det_l050 = list(kind = "determinacy", lambda = 0.5),
    det_l100 = list(kind = "determinacy", lambda = 1),
    det_l200 = list(kind = "determinacy", lambda = 2),
    two_weight = list(kind = "two_weight", lambda = 1, w = c(y = 1, R = 0.5)))
}

# One penalized fit from an interior start phi0. Returns NULL on failure.
# Convergence is judged by the central-difference gradient of the scaled
# objective at the returned point, not by nlminb's code: PORT reports
# "singular convergence" when its relative tolerance outruns the objective's
# precision, even at a verified optimum.
fit_barrier <- function(st, S, n, phi0, cfg) {
  obj <- function(phi) {
    b <- build_std(st, phi)
    if (!b$ok) return(Inf)
    pen <- barrier_value(cfg$kind, b, cfg$w)
    if (!is.finite(pen)) return(Inf)
    ll <- loglik_ml(b$Sigma, S, n)
    if (!is.finite(ll)) return(Inf)
    -(ll + cfg$lambda * pen) / n
  }
  if (!is.finite(obj(phi0))) return(NULL)
  t0 <- proc.time()[["elapsed"]]
  opt <- tryCatch(
    stats::nlminb(phi0, obj, control = list(iter.max = 1000L, eval.max = 2000L,
                                            rel.tol = 1e-11)),
    error = function(e) NULL)
  if (is.null(opt) || !is.finite(opt$objective)) return(NULL)
  grad <- vapply(seq_along(opt$par), function(j) {
    h <- 1e-6 * max(1, abs(opt$par[[j]]))
    up <- opt$par
    dn <- opt$par
    up[[j]] <- up[[j]] + h
    dn[[j]] <- dn[[j]] - h
    (obj(up) - obj(dn)) / (2 * h)
  }, numeric(1))
  max_grad <- if (all(is.finite(grad))) max(abs(grad)) else Inf
  list(phi = opt$par, converged = max_grad < 1e-4, max_grad = max_grad,
       iterations = opt$iterations, seconds = proc.time()[["elapsed"]] - t0)
}
