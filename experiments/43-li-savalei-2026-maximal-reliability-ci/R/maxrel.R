# Maximal-reliability functionals and their delta-method standard errors for
# orthogonal bifactor models, the finite-sample inference Li & Savalei (2026,
# MBR 61:3, 469-490) derive the point formulas for but leave to future work.
#
# Everything here is pure R over a fitted magmaan object: the point coefficients
# are closed forms (their eqs. 14-20) on the model-implied Sigma = Lambda Lambda'
# + Psi, and the SE is the delta method grad(rho*)' V grad(rho*) with V either the
# normal-theory ("model") or sandwich ("robust") parameter covariance magmaan
# already returns from vcov(fit, regime=). No core/C++ changes; the maximal-
# reliability layer itself is the unbuilt measures::frontier surface (speculative).

# --- point coefficients (scale-invariant; Li & Savalei eqs. 14-20) ------------

# Maximal reliability of the optimal linear composite (OLC) for one target factor
# whose loading column is `lam_t`, treating ALL other variance as error:
#   rho* = 1 / (1 + 1 / (lam_t' (Sigma - lam_t lam_t')^-1 lam_t)).
rho_olc <- function(Sigma, lam_t) {
  E <- Sigma - tcrossprod(lam_t)              # error = Sigma minus the target factor
  q <- as.numeric(crossprod(lam_t, solve(E, lam_t)))
  1 / (1 + 1 / q)
}

# Maximal reliability of the optimal linear SUB-composite (OLSC) for group factor
# j: the SAME formula restricted to the sub-matrix of that factor's own items.
# This is the correct generalization of coefficient H (their eqs. 19-20).
rho_olsc <- function(Sigma, lam_grp_col, items) {
  rho_olc(Sigma[items, items, drop = FALSE], lam_grp_col[items])
}

# --- population design ---------------------------------------------------------

# An orthogonal bifactor population: `per` items per subscale, `k` subscales,
# general loading `gen`, per-subscale group loadings `grp` (length k), indicator
# variances standardized to 1. Returns Sigma, the general-factor loading vector,
# the group loading matrix, the per-subscale item index lists, and the syntax.
bifactor_population <- function(per, gen, grp) {
  k <- length(grp)
  p <- per * k
  ov <- paste0("x", seq_len(p))
  items <- split(seq_len(p), rep(seq_len(k), each = per))
  lamG <- rep(gen, p)
  LamGrp <- matrix(0, p, k)
  for (j in seq_len(k)) LamGrp[items[[j]], j] <- grp[j]
  psi <- 1 - lamG^2 - rowSums(LamGrp^2)
  if (any(psi <= 0)) stop("population residual variance <= 0; lower the loadings")
  Lam <- cbind(lamG, LamGrp)
  Sigma <- tcrossprod(Lam) + diag(psi)
  facs <- c("G", paste0("g", seq_len(k)))
  syntax <- paste0(
    "G =~ ", paste(ov, collapse = " + "), "\n",
    paste(vapply(seq_len(k), function(j)
      paste0("g", j, " =~ ", paste(ov[items[[j]]], collapse = " + ")),
      character(1)), collapse = "\n"))
  list(ov = ov, items = items, k = k, p = p, lamG = lamG, LamGrp = LamGrp,
       psi = psi, Lam = Lam, Sigma = Sigma, facs = facs, syntax = syntax,
       rho_gen = rho_olc(Sigma, lamG),
       rho_grp1 = rho_olsc(Sigma, LamGrp[, 1], items[[1]]))
}

# --- data generation -----------------------------------------------------------

# Draw n observations with covariance pop$Sigma. `dist = "normal"` is the paper's
# setting; `dist = "chisq"` keeps the SAME covariance (so rho*_pop is unchanged)
# but gives strongly skewed/leptokurtic indicators by drawing every independent
# common and unique factor as a standardized chi-square_1 ((X-1)/sqrt(2): skew
# ~2.8, excess kurtosis ~12). The fitted ML model stays correct; only normality
# breaks, which is exactly when the sandwich V must diverge from the model V.
gen_data <- function(pop, n, dist) {
  draw <- switch(dist,
    normal = function(m) matrix(stats::rnorm(n * m), n, m),
    chisq  = function(m) matrix((stats::rchisq(n * m, df = 1) - 1) / sqrt(2), n, m),
    stop("unknown dist: ", dist))
  f <- draw(pop$k + 1L)                         # G, g1..gk, each var 1
  e <- draw(pop$p) %*% diag(sqrt(pop$psi))      # uniqueness, var psi_i
  X <- f %*% t(pop$Lam) + e
  colnames(X) <- pop$ov
  as.data.frame(X)
}

# --- reconstruction of Lambda(theta), Psi(theta) from a magmaan partable -------

# A closure over the fitted partable that rebuilds (Lambda, Psi) at an arbitrary
# full parameter vector `est`, so rho*(theta) can be finite-differenced. Valid for
# the orthogonal std.lv bifactor whose only free parameters are loadings and
# residual variances (factor variances fixed at 1, covariances at 0).
make_rebuilder <- function(pt, pop) {
  ov <- pop$ov; facs <- pop$facs
  load_rows  <- which(pt$op == "=~" & pt$lhs %in% facs & pt$rhs %in% ov)
  resid_rows <- which(pt$op == "~~" & pt$lhs == pt$rhs & pt$lhs %in% ov)
  function(est) {
    L <- matrix(0, pop$p, length(facs), dimnames = list(ov, facs))
    P <- numeric(pop$p); names(P) <- ov
    L[cbind(match(pt$rhs[load_rows], ov), match(pt$lhs[load_rows], facs))] <-
      est[load_rows]
    P[match(pt$lhs[resid_rows], ov)] <- est[resid_rows]
    list(L = L, P = P, Sigma = tcrossprod(L) + diag(P))
  }
}

# rho* (gen OLC or group-1 OLSC) as a scalar function of the full theta vector.
rho_of_theta <- function(theta, rebuild, pop, which) {
  m <- rebuild(theta)
  if (which == "gen") rho_olc(m$Sigma, m$L[, "G"])
  else rho_olsc(m$Sigma, m$L[, "g1"], pop$items[[1]])
}

# Central finite-difference gradient of rho*(theta) in vcov (free-parameter)
# ordering: entry pt$free[r] gets d rho*/d theta_r.
grad_rho <- function(theta, pt, rebuild, pop, which) {
  free_rows <- which(pt$free > 0)
  g <- numeric(max(pt$free))
  for (r in free_rows) {
    h <- 1e-5 * max(1, abs(theta[r]))
    tp <- theta; tm <- theta; tp[r] <- tp[r] + h; tm[r] <- tm[r] - h
    g[pt$free[r]] <- (rho_of_theta(tp, rebuild, pop, which) -
                      rho_of_theta(tm, rebuild, pop, which)) / (2 * h)
  }
  g
}

# --- second-order (curvature) bias of the plug-in reliability ------------------
#
# rho* = g(theta) is a nonlinear functional, so even an unbiased theta-hat leaves
# E[g(theta-hat)] - g(theta) ~ (1/2) tr(g''(theta) V), the Jensen / curvature term
# (V = Cov(theta-hat)). We estimate it without forming the full Hessian: with the
# eigendecomposition V = sum_k lambda_k v_k v_k', tr(g'' V) = sum_k lambda_k
# v_k' g'' v_k, and each v_k' g'' v_k is a central second directional derivative.
# Cost is O(n_free) evaluations of g, not O(n_free^2). The propagated-parameter
# bias term g'(theta)' b_theta is handled separately, by evaluating g at a
# reduced-bias (RBM / Kosmidis) theta-tilde.

# Map free-parameter position (vcov ordering) -> partable row, so a direction in
# free space can perturb the full theta vector.
free_pos_to_row <- function(pt) {
  fr <- which(pt$free > 0)
  m <- integer(max(pt$free)); m[pt$free[fr]] <- fr
  m
}

# (1/2) tr(g'' V), or the same for the transformed functional logit(g) when
# link = "logit": there g lives on the unbounded scale where it is closer to
# linear, so the curvature term is smaller and the back-transformed correction
# stays inside (0, 1) by construction (no catastrophic over-shoot at small N).
curvature_bias <- function(theta, pt, rebuild, pop, which, V,
                           link = c("identity", "logit")) {
  link <- match.arg(link)
  lf <- switch(link,
    identity = function(x) x,
    logit    = function(x) stats::qlogis(pmin(1 - 1e-9, pmax(1e-9, x))))
  pos2row <- free_pos_to_row(pt)
  p <- length(pos2row)
  eg <- eigen((V + t(V)) / 2, symmetric = TRUE)
  g0 <- lf(rho_of_theta(theta, rebuild, pop, which))
  bump <- function(v, s) { th <- theta; th[pos2row] <- th[pos2row] + s * v; th }
  acc <- 0
  for (k in seq_len(p)) {
    lam <- eg$values[k]
    if (abs(lam) < 1e-10) next
    v <- eg$vectors[, k]
    h <- 1e-4
    d2 <- (lf(rho_of_theta(bump(v, h), rebuild, pop, which)) - 2 * g0 +
           lf(rho_of_theta(bump(v, -h), rebuild, pop, which))) / h^2
    acc <- acc + lam * d2
  }
  0.5 * acc
}

# --- one replicate: fit, both coefficients, model & robust delta SEs -----------

# Returns a one-row-per-coefficient data.frame with the point estimate and the
# normal-theory and sandwich delta-method SEs, or NULL on a non-converged /
# improper fit (negative residual variance = Heywood). `spec` is prebuilt once.
fit_coefficients <- function(spec, dat, pop) {
  fit <- tryCatch(magmaan::magmaan(spec, dat, estimator = "ML"),
                  error = function(e) NULL)
  if (is.null(fit) || !isTRUE(fit$converged)) return(NULL)
  pt <- fit$partable
  rebuild <- make_rebuilder(pt, pop)
  m <- rebuild(pt$est)
  if (any(m$P <= 0)) return(NULL)                       # improper (Heywood) drop
  Vm <- tryCatch(stats::vcov(fit, regime = "model",  data = dat),
                 error = function(e) NULL)
  Vr <- tryCatch(stats::vcov(fit, regime = "robust", data = dat),
                 error = function(e) NULL)
  if (!is.matrix(Vm) || !is.matrix(Vr)) return(NULL)
  out <- lapply(c(gen = "gen", grp = "grp"), function(w) {
    rho <- rho_of_theta(pt$est, rebuild, pop, w)
    g   <- grad_rho(pt$est, pt, rebuild, pop, w)
    se_m <- sqrt(max(0, as.numeric(crossprod(g, Vm %*% g))))
    se_r <- sqrt(max(0, as.numeric(crossprod(g, Vr %*% g))))
    data.frame(coef = w, rho = rho, se_model = se_m, se_robust = se_r)
  })
  do.call(rbind, out)
}

# --- interval constructors -----------------------------------------------------

z975 <- stats::qnorm(0.975)

# Wald CI on the raw rho* scale.
ci_wald <- function(rho, se) c(rho - z975 * se, rho + z975 * se)

# Wald CI on the logit scale, back-transformed: respects the [0,1] boundary and
# the right-skew of a reliability estimate near its floor. se on the logit scale
# is se(rho) / (rho (1-rho)) by the delta method.
ci_logit <- function(rho, se) {
  eta <- stats::qlogis(rho)
  se_eta <- se / (rho * (1 - rho))
  stats::plogis(c(eta - z975 * se_eta, eta + z975 * se_eta))
}

covers <- function(ci, target) is.finite(ci[1]) && ci[1] <= target && target <= ci[2]
width  <- function(ci) ci[2] - ci[1]
