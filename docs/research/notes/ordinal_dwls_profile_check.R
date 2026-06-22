#!/usr/bin/env Rscript

# Exploratory check for regular_sem_asymptotics_roadmap.tex.
#
# Binary ordinal DWLS, p = 4.  The first-stage object is
#
#   x = (u, gamma),
#
# where u stacks thresholds and tetrachoric correlations, and gamma is
# diag(Gamma_u), the diagonal NACOV used by DWLS.  The model comparison is
# congeneric one-factor (A) versus tau-equivalent one-factor (B).
#
# The script compares three spectra for the nested contrast:
#
#   U_gn:       Gauss--Newton fixed-weight projector, using only u.
#   Q_fixed:   fixed-weight profile Hessian in u, including model curvature.
#   Q_full:    full profile Hessian in x=(u,gamma), including DWLS weight input.
#
# This is not a finite-sample simulation.  It is a population/delta-method
# diagnostic to see where categorical DWLS can depart from the classical U law.

suppressPackageStartupMessages(library(mvtnorm))

options(digits = 5, width = 120)

p <- 4L
pairs <- combn(seq_len(p), 2L, simplify = FALSE)
n_pairs <- length(pairs)
n_u <- p + n_pairs
n_cell <- 2^p

patterns <- as.matrix(expand.grid(rep(list(0:1), p)))

pair_index <- function(i, j) {
  for (k in seq_along(pairs)) {
    if (identical(pairs[[k]], c(i, j))) return(k)
  }
  stop("bad pair")
}

invnorm <- function(x) qnorm(pmin(pmax(x, 1e-8), 1 - 1e-8))

biv_p00 <- function(t1, t2, rho) {
  R <- matrix(c(1, rho, rho, 1), 2, 2)
  as.numeric(pmvnorm(
    lower = c(-Inf, -Inf), upper = c(t1, t2),
    mean = c(0, 0), sigma = R, algorithm = Miwa()
  ))
}

tetra_from_p00 <- function(t1, t2, p00) {
  lo <- -0.999
  hi <- 0.999
  f <- function(r) biv_p00(t1, t2, r) - p00
  flo <- f(lo)
  fhi <- f(hi)
  if (flo > 0) return(lo)
  if (fhi < 0) return(hi)
  uniroot(f, c(lo, hi), tol = 1e-8)$root
}

cell_probs <- function(tau, R) {
  out <- numeric(n_cell)
  for (a in seq_len(n_cell)) {
    lower <- ifelse(patterns[a, ] == 0, -Inf, tau)
    upper <- ifelse(patterns[a, ] == 0, tau, Inf)
    out[a] <- as.numeric(pmvnorm(
      lower = lower, upper = upper, mean = rep(0, p), sigma = R,
      algorithm = Miwa()
    ))
  }
  out / sum(out)
}

moments_from_pi <- function(pi) {
  tau <- numeric(p)
  for (j in seq_len(p)) {
    tau[j] <- invnorm(sum(pi[patterns[, j] == 0]))
  }

  rho <- numeric(n_pairs)
  for (k in seq_along(pairs)) {
    i <- pairs[[k]][1]
    j <- pairs[[k]][2]
    p00 <- sum(pi[patterns[, i] == 0 & patterns[, j] == 0])
    rho[k] <- tetra_from_p00(tau[i], tau[j], p00)
  }
  c(tau, rho)
}

finite_jac_pi <- function(fun, pi, h = 1e-5) {
  d <- n_cell - 1L
  f0 <- fun(pi)
  J <- matrix(0, length(f0), d)
  for (k in seq_len(d)) {
    step <- min(h, pi[k] * 0.2, pi[n_cell] * 0.2)
    pp <- pi
    pm <- pi
    pp[k] <- pp[k] + step
    pp[n_cell] <- pp[n_cell] - step
    pm[k] <- pm[k] - step
    pm[n_cell] <- pm[n_cell] + step
    J[, k] <- (fun(pp) - fun(pm)) / (2 * step)
  }
  J
}

gamma_u_from_pi <- function(pi) {
  J <- finite_jac_pi(moments_from_pi, pi)
  pii <- pi[-n_cell]
  Sigma_pi <- diag(pii) - tcrossprod(pii)
  Gamma_u <- J %*% Sigma_pi %*% t(J)
  diag(Gamma_u)
}

x_from_pi <- function(pi) {
  u <- moments_from_pi(pi)
  gamma <- gamma_u_from_pi(pi)
  c(u, gamma)
}

gamma_x_from_pi <- function(pi) {
  J <- finite_jac_pi(x_from_pi, pi)
  pii <- pi[-n_cell]
  Sigma_pi <- diag(pii) - tcrossprod(pii)
  J %*% Sigma_pi %*% t(J)
}

corr_from_lambda <- function(lambda) {
  out <- numeric(n_pairs)
  for (k in seq_along(pairs)) {
    out[k] <- lambda[pairs[[k]][1]] * lambda[pairs[[k]][2]]
  }
  out
}

mu_A <- function(par) {
  tau <- par[seq_len(p)]
  lambda <- 0.95 * tanh(par[p + seq_len(p)])
  c(tau, corr_from_lambda(lambda))
}

mu_B <- function(par) {
  tau <- par[seq_len(p)]
  lambda <- rep(0.95 * tanh(par[p + 1L]), p)
  c(tau, corr_from_lambda(lambda))
}

start_A <- function(u) {
  R <- diag(1, p)
  R[lower.tri(R)] <- u[p + seq_len(n_pairs)]
  R <- R + t(R) - diag(1, p)
  ev <- eigen(R, symmetric = TRUE)
  lam <- sqrt(pmax(ev$values[1], 1e-6)) * ev$vectors[, 1]
  if (mean(lam) < 0) lam <- -lam
  lam <- pmin(pmax(lam, 0.05), 0.90)
  c(u[seq_len(p)], atanh(lam / 0.95))
}

start_B <- function(u) {
  rbar <- mean(u[p + seq_len(n_pairs)])
  lam <- sqrt(max(rbar, 1e-4))
  c(u[seq_len(p)], atanh(min(lam, 0.90) / 0.95))
}

fit_model <- function(kind, x) {
  u <- x[seq_len(n_u)]
  gamma <- pmax(x[n_u + seq_len(n_u)], 1e-8)
  if (kind == "A") {
    mu <- mu_A
    start <- start_A(u)
  } else {
    mu <- mu_B
    start <- start_B(u)
  }
  obj <- function(par) {
    r <- mu(par) - u
    0.5 * sum(r * r / gamma)
  }
  opt <- optim(start, obj, method = "BFGS",
               control = list(maxit = 1000, reltol = 1e-12))
  r <- mu(opt$par) - u
  list(par = opt$par, r = r, value = obj(opt$par), conv = opt$convergence)
}

profile_score_x <- function(kind, x) {
  fit <- fit_model(kind, x)
  gamma <- pmax(x[n_u + seq_len(n_u)], 1e-8)
  r <- fit$r
  c(-r / gamma, -0.5 * r * r / (gamma * gamma))
}

profile_score_u_fixed <- function(kind, u, gamma) {
  x <- c(u, gamma)
  fit <- fit_model(kind, x)
  -fit$r / gamma
}

jacobian_x <- function(fun, x, h = 2e-5) {
  f0 <- fun(x)
  J <- matrix(0, length(f0), length(x))
  for (k in seq_along(x)) {
    step <- if (k > n_u) max(h * abs(x[k]), 1e-6) else h
    xp <- x
    xm <- x
    xp[k] <- xp[k] + step
    xm[k] <- xm[k] - step
    if (k > n_u && xm[k] <= 0) {
      xm[k] <- x[k]
      xp[k] <- x[k] + 2 * step
      J[, k] <- (fun(xp) - fun(xm)) / (2 * step)
    } else {
      J[, k] <- (fun(xp) - fun(xm)) / (2 * step)
    }
  }
  0.5 * (J + t(J))
}

jacobian_u <- function(fun, u, h = 2e-5) {
  f0 <- fun(u)
  J <- matrix(0, length(f0), length(u))
  for (k in seq_along(u)) {
    up <- u
    um <- u
    up[k] <- up[k] + h
    um[k] <- um[k] - h
    J[, k] <- (fun(up) - fun(um)) / (2 * h)
  }
  0.5 * (J + t(J))
}

jac_mu <- function(kind, par, h = 1e-5) {
  mu <- if (kind == "A") mu_A else mu_B
  base <- mu(par)
  J <- matrix(0, length(base), length(par))
  for (k in seq_along(par)) {
    pp <- par
    pm <- par
    pp[k] <- pp[k] + h
    pm[k] <- pm[k] - h
    J[, k] <- (mu(pp) - mu(pm)) / (2 * h)
  }
  J
}

U_gn <- function(kind, x) {
  fit <- fit_model(kind, x)
  gamma <- pmax(x[n_u + seq_len(n_u)], 1e-8)
  W <- diag(1 / gamma)
  D <- jac_mu(kind, fit$par)
  W - W %*% D %*% solve(t(D) %*% W %*% D, t(D) %*% W)
}

spec <- function(Q, Gamma) {
  ev <- eigen(Q %*% Gamma, only.values = TRUE)$values
  ev <- Re(ev)
  sort(ev, decreasing = TRUE)
}

norm_score_diff <- function(x) {
  sb <- profile_score_x("B", x)
  sa <- profile_score_x("A", x)
  sqrt(sum((sb - sa)^2))
}

analyze <- function(label, tau, R) {
  pi <- cell_probs(tau, R)
  u <- moments_from_pi(pi)
  gamma <- gamma_u_from_pi(pi)
  x <- c(u, gamma)
  Gamma_u <- {
    J <- finite_jac_pi(moments_from_pi, pi)
    pii <- pi[-n_cell]
    S <- diag(pii) - tcrossprod(pii)
    J %*% S %*% t(J)
  }
  Gamma_x <- gamma_x_from_pi(pi)

  fitA <- fit_model("A", x)
  fitB <- fit_model("B", x)
  Q_full <- jacobian_x(function(xx) profile_score_x("B", xx), x) -
    jacobian_x(function(xx) profile_score_x("A", xx), x)
  Q_fixed <- jacobian_u(function(uu) profile_score_u_fixed("B", uu, gamma), u) -
    jacobian_u(function(uu) profile_score_u_fixed("A", uu, gamma), u)
  Q_gn <- U_gn("B", x) - U_gn("A", x)

  lam_full <- spec(Q_full, Gamma_x)
  lam_fixed <- spec(Q_fixed, Gamma_u)
  lam_gn <- spec(Q_gn, Gamma_u)

  cat("\n===", label, "===\n")
  cat("fit values: B-A =", sprintf("%.6g", fitB$value - fitA$value),
      "  ||score_B-score_A|| =", sprintf("%.4g", norm_score_diff(x)), "\n")
  cat("||r_A||_W =", sprintf("%.4f", sqrt(sum(fitA$r^2 / gamma))),
      " ||r_B||_W =", sprintf("%.4f", sqrt(sum(fitB$r^2 / gamma))), "\n")
  cat("GN fixed-U      eig:", paste(sprintf("%.4f", head(lam_gn, 6)), collapse = " "),
      " sum=", sprintf("%.4f", sum(lam_gn)), "\n")
  cat("fixed-weight Q  eig:", paste(sprintf("%.4f", head(lam_fixed, 6)), collapse = " "),
      " sum=", sprintf("%.4f", sum(lam_fixed)), "\n")
  cat("full DWLS Q     eig:", paste(sprintf("%.4f", head(lam_full, 6)), collapse = " "),
      " sum=", sprintf("%.4f", sum(lam_full)), "\n")
  cat("full/fixed trace ratio:", sprintf("%.4f", sum(lam_full) / sum(lam_fixed)), "\n")
}

scenario_stats <- function(tau, R) {
  pi <- cell_probs(tau, R)
  u <- moments_from_pi(pi)
  gamma <- gamma_u_from_pi(pi)
  x <- c(u, gamma)
  Gamma_u <- {
    J <- finite_jac_pi(moments_from_pi, pi)
    pii <- pi[-n_cell]
    S <- diag(pii) - tcrossprod(pii)
    J %*% S %*% t(J)
  }
  Gamma_x <- gamma_x_from_pi(pi)

  fitA <- fit_model("A", x)
  fitB <- fit_model("B", x)
  Q_full <- jacobian_x(function(xx) profile_score_x("B", xx), x) -
    jacobian_x(function(xx) profile_score_x("A", xx), x)
  Q_fixed <- jacobian_u(function(uu) profile_score_u_fixed("B", uu, gamma), u) -
    jacobian_u(function(uu) profile_score_u_fixed("A", uu, gamma), u)
  lam_full <- spec(Q_full, Gamma_x)
  lam_fixed <- spec(Q_fixed, Gamma_u)

  list(
    fit_diff = fitB$value - fitA$value,
    score_diff = norm_score_diff(x),
    sqrt2f = sqrt(sum(fitA$r^2 / gamma)),
    fixed = lam_fixed,
    full = lam_full,
    fixed_trace = sum(lam_fixed),
    full_trace = sum(lam_full)
  )
}

adjacent_cycle <- c(1, 0, 1, 1, 0, 1)
R_symmetric_cycle <- function(eps) {
  R <- R_eq
  for (k in seq_along(pairs)) {
    i <- pairs[[k]][1]
    j <- pairs[[k]][2]
    R[i, j] <- R[j, i] <- R[i, j] + eps * adjacent_cycle[k]
  }
  R
}

print_symmetric_cycle_scan <- function() {
  cat("\nSymmetry-protected C4 pseudo-null, equal binary thresholds.\n")
  cat("eps  sqrt2f  B-A        score      tr_fixed tr_full ratio  full eig top3\n")
  for (eps in c(0.02, 0.04, 0.06, 0.08, 0.10)) {
    out <- scenario_stats(rep(0, p), R_symmetric_cycle(eps))
    cat(sprintf(
      "%.2f  %.4f  %.2e  %.2e  %.4f  %.4f  %.4f   %s\n",
      eps, out$sqrt2f, out$fit_diff, out$score_diff,
      out$fixed_trace, out$full_trace, out$full_trace / out$fixed_trace,
      paste(sprintf("%.4f", head(out$full, 3)), collapse = " ")
    ))
  }
}

tau0 <- c(-0.6, -0.15, 0.25, 0.7)
lambda_eq <- rep(0.55, p)
R_eq <- diag(1, p)
R_eq[lower.tri(R_eq)] <- corr_from_lambda(lambda_eq)
R_eq <- R_eq + t(R_eq) - diag(1, p)

cycle <- c(1, -1, 0, 0, -1, 1)
R_cycle <- function(eps) {
  R <- R_eq
  for (k in seq_along(pairs)) {
    i <- pairs[[k]][1]
    j <- pairs[[k]][2]
    R[i, j] <- R[j, i] <- R[i, j] + eps * cycle[k]
  }
  R
}

lambda_uneq <- c(0.70, 0.60, 0.48, 0.36)
R_uneq <- diag(1, p)
R_uneq[lower.tri(R_uneq)] <- corr_from_lambda(lambda_uneq)
R_uneq <- R_uneq + t(R_uneq) - diag(1, p)

cat("Binary ordinal DWLS: A = congeneric one-factor, B = tau-equivalent.\n")
cat("u = thresholds + tetrachorics; gamma = diag(Gamma_u).\n")

analyze("exact nested null: tau-equivalent latent response", tau0, R_eq)
analyze("near-null misspec direction eps=0.04", tau0, R_cycle(0.04))
analyze("near-null misspec direction eps=0.08", tau0, R_cycle(0.08))
analyze("restriction false: congeneric latent response", tau0, R_uneq)
print_symmetric_cycle_scan()
