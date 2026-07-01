# Generic normal-theory profile likelihood-ratio (test-inversion) confidence
# interval for a scalar functional g(theta) of an orthogonal std.lv CFA. This is
# the reusable harness the speculative profile-LR reliability-CI project is built
# on: the ONLY thing that varies across the reliability family (omega_total,
# omega_h, H, ...) is the functional closure passed in; the engine never knows
# which coefficient it is computing. Prototyped in R with zero magmaan core
# change, exactly as the design records (nloptr constrained fit + a g-closure).
#
# The interval is {g0 : T(g0) <= q}, T(g0) = N (F_con(g0) - F_unc), F the ML
# discrepancy. Under NT and H0, T ~ chi^2_1, so q = qchisq(1-alpha, 1). The
# `cscale` argument multiplies q: it is the single hook where every later tier
# plugs in (Bartlett E[LR] factor for small samples; Satorra-2000 c for robust;
# a sandwich scale for misspec-robust). NT is cscale = 1.
#
# Model parameterization (orthogonal std.lv, matching lavaan std.lv orthogonal):
# factor j (j = 1..m) loads on the items in cols[[j]]; all factor variances are
# fixed to 1 and all factor covariances to 0. The free vector is
#   z = [ loadings of factor 1, ..., loadings of factor m, log(psi_1..psi_p) ],
# residual variances carried on the log scale so the optimizer stays interior.

# Build the closures for a fixed loading pattern. `cols` is a list of integer
# item-index vectors, one per factor; `p` is the number of observed indicators.
make_engine <- function(cols, p) {
  m <- length(cols)
  nload <- sum(lengths(cols))
  # z -> (Lambda p x m, psi length p)
  unpack <- function(z) {
    Lam <- matrix(0, p, m)
    pos <- 0L
    for (j in seq_len(m)) {
      ix <- cols[[j]]
      Lam[ix, j] <- z[(pos + 1L):(pos + length(ix))]
      pos <- pos + length(ix)
    }
    list(Lam = Lam, psi = exp(z[(nload + 1L):(nload + p)]))
  }
  sigma_of <- function(z) { u <- unpack(z); tcrossprod(u$Lam) + diag(u$psi) }

  fml <- function(z, S, logdetS) {
    Sig <- sigma_of(z)
    ch <- tryCatch(chol(Sig), error = function(e) NULL)
    if (is.null(ch)) return(1e6)
    Si <- chol2inv(ch)
    2 * sum(log(diag(ch))) + sum(S * Si) - logdetS - p
  }
  # dF/dz analytic: G = Si - Si S Si; dF/dLambda = 2 G Lambda (free entries only),
  # dF/d log psi_i = G_ii psi_i (chain through the log parameterization).
  grad_fml <- function(z, S) {
    u <- unpack(z); Si <- solve(tcrossprod(u$Lam) + diag(u$psi))
    G <- Si - Si %*% S %*% Si
    GL <- 2 * (G %*% u$Lam)
    load_g <- unlist(lapply(seq_len(m), function(j) GL[cols[[j]], j]))
    c(load_g, diag(G) * u$psi)
  }
  # start vector from a (Lambda, psi) guess in the same free ordering
  zeta_from <- function(Lam, psi)
    c(unlist(lapply(seq_len(m), function(j) Lam[cols[[j]], j])), log(psi))

  opts_unc <- list(algorithm = "NLOPT_LD_LBFGS", xtol_rel = 1e-9, maxeval = 2000)
  opts_con <- list(algorithm = "NLOPT_LD_SLSQP", xtol_rel = 1e-9, maxeval = 2000)

  fit_unc <- function(S, logdetS, z0) {
    r <- nloptr::nloptr(z0, eval_f = function(z)
      list(objective = fml(z, S, logdetS), gradient = grad_fml(z, S)), opts = opts_unc)
    list(F = r$objective, z = r$solution, status = r$status)
  }

  # A functional is a closure gfun(z) returning either a scalar (engine will
  # finite-difference the constraint Jacobian) or list(value=, grad=) with grad
  # in z-space (used directly, no FD). This is the value-and-gradient interface
  # from the design; grad = NULL falls back to FD.
  gval <- function(gfun, z) { r <- gfun(z); if (is.list(r)) r$value else r }
  gjac <- function(gfun, z) {
    r <- gfun(z)
    if (is.list(r) && !is.null(r$grad)) return(matrix(r$grad, nrow = 1))
    e <- 1e-6
    matrix(vapply(seq_along(z), function(i) {
      zp <- z; zm <- z; zp[i] <- zp[i] + e; zm[i] <- zm[i] - e
      (gval(gfun, zp) - gval(gfun, zm)) / (2 * e)
    }, numeric(1)), nrow = 1)
  }

  # constrained min of F_ML s.t. g(z) = g0; NULL if the constraint is not met.
  fit_con <- function(S, logdetS, z0, gfun, g0) {
    r <- tryCatch(nloptr::nloptr(z0,
      eval_f = function(z) list(objective = fml(z, S, logdetS), gradient = grad_fml(z, S)),
      eval_g_eq = function(z) list(constraints = gval(gfun, z) - g0,
                                   jacobian = gjac(gfun, z)),
      opts = opts_con), error = function(e) NULL)
    if (is.null(r) || abs(gval(gfun, r$solution) - g0) > 1e-5) return(NULL)
    list(F = r$objective, z = r$solution)
  }

  list(cols = cols, p = p, m = m, unpack = unpack, sigma_of = sigma_of,
       fml = fml, grad_fml = grad_fml, zeta_from = zeta_from, gval = gval,
       fit_unc = fit_unc, fit_con = fit_con)
}

# The test statistic profile T(g0) = N (F_con(g0) - F_unc), or NA if the
# constrained fit fails. `unc` is the fit_unc result on (S, logdetS).
lr_profile <- function(eng, S, logdetS, n, unc, gfun, g0) {
  con <- eng$fit_con(S, logdetS, unc$z, gfun, g0)
  if (is.null(con)) return(NA_real_)
  max(0, n * (con$F - unc$F))
}

# Profile-LR CI by monotone bisection out from ghat on each side, on the natural
# [0,1] scale (reliability). Returns c(lo, hi, est). `cscale` multiplies the
# chi^2 threshold (the tier hook: 1 = NT; Bartlett / Satorra-2000 / sandwich
# scales plug in here later). `lo_lim`/`hi_lim` clamp the search to the
# admissible range; a bound that reaches the clamp before T hits q is reported at
# the clamp (an attained boundary, not a failure). Takes the already-computed
# unconstrained fit so a caller that fits once can reuse it for many functionals.
profile_lr_ci_fit <- function(eng, S, n, unc, gfun, alpha = 0.05, cscale = 1.0,
                              lo_lim = 1e-6, hi_lim = 1 - 1e-6, gtol = 1e-7) {
  logdetS <- as.numeric(determinant(S, logarithm = TRUE)$modulus)
  ghat <- eng$gval(gfun, unc$z)
  q <- qchisq(1 - alpha, 1) * cscale
  Tof <- function(g0) lr_profile(eng, S, logdetS, n, unc, gfun, g0)
  bound <- function(dir) {
    limit <- if (dir < 0) lo_lim else hi_lim
    step <- 0.5 * abs(limit - ghat)
    g_in <- ghat; g_out <- NA_real_
    for (k in 1:50) {
      g_try <- ghat + dir * step
      g_try <- if (dir < 0) max(g_try, limit) else min(g_try, limit)
      Tt <- Tof(g_try)
      if (is.na(Tt)) { step <- step * 0.5; next }
      if (Tt >= q) { g_out <- g_try; break }
      g_in <- g_try
      if (isTRUE(all.equal(g_try, limit))) return(limit)  # attained boundary
      step <- step * 1.5
    }
    if (is.na(g_out)) return(NA_real_)
    for (k in 1:80) {
      gm <- 0.5 * (g_in + g_out); Tm <- Tof(gm)
      if (is.na(Tm)) { g_out <- gm; next }
      if (Tm < q) g_in <- gm else g_out <- gm
      if (abs(g_out - g_in) < gtol) break
    }
    0.5 * (g_in + g_out)
  }
  c(lo = bound(-1), hi = bound(+1), est = ghat)
}
