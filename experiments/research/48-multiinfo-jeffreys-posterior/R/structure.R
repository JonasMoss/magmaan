# Model algebra in two coordinate systems.
#
# Standardized-latent coordinates (phi) are what the sampler moves in: every
# latent has unit variance, structural coefficients are standardized, and each
# factor's first loading is positive. All improper faces sit at finite values
# (residual variance 0, latent R^2 = 1, |correlation| = 1), so there is no ridge
# to infinity. phi = [loadings, residual variances, structural coefficients,
# exogenous correlations].
#
# Marker coordinates are magmaan's own parameter vector, used for the point
# estimators and the truncated-normal approximation.
#
# Every estimator is summarized by the same identification-invariant quantities:
# observed residual variances, standardized loadings, latent correlations, and
# latent R^2.

phi_layout <- function(st) {
  list(n_load = sum(st$load_pat), n_theta = st$p, n_b = sum(st$B_pat),
       n_cor = nrow(st$exo_cor))
}

phi_labels <- function(st) {
  idx <- which(st$load_pat, arr.ind = TRUE)
  bidx <- which(st$B_pat, arr.ind = TRUE)
  c(sprintf("%s =~ %s", st$lv[idx[, 2]], st$ov[idx[, 1]]),
    sprintf("%s ~~ %s", st$ov, st$ov),
    if (nrow(bidx)) sprintf("%s ~ %s", st$lv[bidx[, 1]], st$lv[bidx[, 2]]),
    if (nrow(st$exo_cor)) sprintf("%s ~~ %s", st$lv[st$exo_cor[, 1]], st$lv[st$exo_cor[, 2]]))
}

# Latent correlation matrix and R^2 from standardized coefficients. Requires
# parents to precede children. Returns ok = FALSE outside the admissible set.
latent_phi <- function(st, B, cors) {
  m <- st$m
  Phi <- diag(m)
  if (!all(is.finite(B)) || !all(is.finite(cors))) {
    return(list(Phi = Phi, r2 = rep(0, m), ok = FALSE))
  }
  if (length(cors)) {
    Phi[st$exo_cor] <- cors
    Phi[st$exo_cor[, 2:1, drop = FALSE]] <- cors
  }
  r2 <- rep(0, m)
  for (j in which(!st$exo)) {
    pa <- which(st$B_pat[j, ])
    prev <- seq_len(j - 1L)
    cj <- drop(B[j, pa] %*% Phi[pa, prev, drop = FALSE])
    Phi[j, prev] <- cj
    Phi[prev, j] <- cj
    r2[j] <- sum(B[j, pa] * cj[pa])
  }
  ok <- all(r2[!st$exo] <= 1)
  if (ok && sum(st$exo) > 1L) {
    ok <- min(eigen(Phi[st$exo, st$exo], symmetric = TRUE, only.values = TRUE)$values) >= 0
  }
  list(Phi = Phi, r2 = r2, ok = ok)
}

unpack_phi <- function(st, phi) {
  lay <- phi_layout(st)
  i <- 0L
  take <- function(k) {
    out <- phi[i + seq_len(k)]
    i <<- i + k
    out
  }
  Lambda <- matrix(0, st$p, st$m)
  Lambda[st$load_pat] <- take(lay$n_load)
  theta <- take(lay$n_theta)
  B <- matrix(0, st$m, st$m)
  B[st$B_pat] <- take(lay$n_b)
  cors <- take(lay$n_cor)
  list(Lambda = Lambda, theta = theta, B = B, cors = cors)
}

# Sigma(phi) plus admissibility. Cheap: this is called once per MCMC step.
build_std <- function(st, phi) {
  if (!all(is.finite(phi))) {
    return(list(Sigma = NULL, ok = FALSE))
  }
  u <- unpack_phi(st, phi)
  lat <- latent_phi(st, u$B, u$cors)
  ok <- lat$ok && all(u$theta >= 0) &&
    all(u$Lambda[cbind(st$first_ind, seq_len(st$m))] > 0)
  LP <- u$Lambda %*% lat$Phi
  Sigma <- LP %*% t(u$Lambda)
  diag(Sigma) <- diag(Sigma) + u$theta
  list(Sigma = Sigma, Lambda = u$Lambda, Phi = lat$Phi, r2 = lat$r2,
       theta = u$theta, ok = ok, B = u$B, cors = u$cors)
}

# Normal log-likelihood with the mean integrated out under a flat prior, as a
# function of Sigma (S has divisor n).
loglik_sigma <- function(Sigma, S, n) {
  R <- tryCatch(chol(Sigma), error = function(e) NULL)
  if (is.null(R)) return(-Inf)
  Ri <- backsolve(R, diag(nrow(R)))
  -(n - 1) * sum(log(diag(R))) - (n / 2) * sum((S %*% Ri) * Ri)
}

# Per-observation expected information in phi coordinates:
# I_ab = 1/2 tr(K dSigma_a K dSigma_b), K = Sigma^-1. Loadings and residual
# variances are differentiated analytically, the few structural coordinates by
# central differences of the latent correlation matrix.
information_std <- function(st, phi, b = build_std(st, phi)) {
  lay <- phi_layout(st)
  p <- st$p
  K <- chol2inv(chol(b$Sigma))
  q <- length(phi)
  KD <- vector("list", q)          # K %*% dSigma_a
  idx <- which(st$load_pat, arr.ind = TRUE)
  G <- b$Lambda %*% b$Phi          # p x m
  KG <- K %*% G
  for (a in seq_len(lay$n_load)) {
    i <- idx[a, 1]
    j <- idx[a, 2]
    # dSigma = e_i g' + g e_i' with g = G[, j]
    M <- outer(K[, i], G[, j])
    M[, i] <- M[, i] + KG[, j]
    KD[[a]] <- M
  }
  for (i in seq_len(p)) {
    M <- matrix(0, p, p)
    M[, i] <- K[, i]
    KD[[lay$n_load + i]] <- M
  }
  n_lat <- lay$n_b + lay$n_cor
  if (n_lat) {
    base <- lay$n_load + lay$n_theta
    for (a in seq_len(n_lat)) {
      h <- 1e-6
      up <- phi
      dn <- phi
      up[base + a] <- up[base + a] + h
      dn[base + a] <- dn[base + a] - h
      bu <- unpack_phi(st, up)
      bd <- unpack_phi(st, dn)
      dPhi <- (latent_phi(st, bu$B, bu$cors)$Phi - latent_phi(st, bd$B, bd$cors)$Phi) / (2 * h)
      KD[[base + a]] <- K %*% (b$Lambda %*% dPhi %*% t(b$Lambda))
    }
  }
  P <- vapply(KD, as.vector, numeric(p * p))
  Q <- vapply(KD, function(M) as.vector(t(M)), numeric(p * p))
  0.5 * crossprod(P, Q)
}

# log of the Jeffreys density sqrt(det I(phi)); -Inf when I is singular.
log_jeffreys <- function(st, phi) {
  I <- information_std(st, phi)
  R <- tryCatch(chol((I + t(I)) / 2), error = function(e) NULL)
  if (is.null(R)) return(-Inf)
  sum(log(diag(R)))
}

# Identification-invariant summaries from standardized-latent quantities.
invariants_from <- function(st, Lambda, Phi, r2, theta, Sigma) {
  idx <- which(st$load_pat, arr.ind = TRUE)
  std_load <- Lambda[st$load_pat] / sqrt(diag(Sigma)[idx[, 1]])
  out <- c(theta, std_load)
  nm <- c(sprintf("%s ~~ %s", st$ov, st$ov),
          sprintf("%s =~ %s (std)", st$lv[idx[, 2]], st$ov[idx[, 1]]))
  if (st$m > 1L) {
    pairs <- which(upper.tri(Phi), arr.ind = TRUE)
    out <- c(out, Phi[pairs])
    nm <- c(nm, sprintf("%s ~~ %s (cor)", st$lv[pairs[, 1]], st$lv[pairs[, 2]]))
  }
  endo <- which(!st$exo)
  if (length(endo)) {
    out <- c(out, r2[endo])
    nm <- c(nm, sprintf("R2 %s", st$lv[endo]))
  }
  setNames(out, nm)
}

invariants_std <- function(st, phi, b = build_std(st, phi)) {
  invariants_from(st, b$Lambda, b$Phi, b$r2, b$theta, b$Sigma)
}

population_invariants <- function(design) {
  invariants_from(design$structure, design$Lambda, design$Phi,
                  1 - diag(population_psi(design)), design$theta, design$Sigma)
}

population_psi <- function(design) {
  B <- design$B
  A <- diag(nrow(B)) - B
  A %*% design$Phi %*% t(A)
}

# ---- Marker (magmaan) coordinates ------------------------------------------

free_order <- function(pt) {
  rows <- which(pt$free > 0L)
  rows[order(pt$free[rows])]
}

# theta (magmaan free-parameter order) -> LISREL matrices, using the fixed
# values of a fitted partable.
marker_builder <- function(pt, st) {
  ov <- st$ov
  lv <- st$lv
  L0 <- matrix(0, st$p, st$m, dimnames = list(ov, lv))
  B0 <- matrix(0, st$m, st$m, dimnames = list(lv, lv))
  P0 <- B0
  T0 <- setNames(rep(0, st$p), ov)
  kind <- ifelse(pt$op == "=~", "L",
          ifelse(pt$op == "~" & pt$lhs %in% lv, "B",
          ifelse(pt$op == "~~" & pt$lhs %in% lv, "P",
          ifelse(pt$op == "~~" & pt$lhs == pt$rhs & pt$lhs %in% ov, "T", NA))))
  for (i in which(pt$free == 0L & !is.na(kind))) {
    v <- pt$est[i]
    switch(kind[i],
           L = L0[pt$rhs[i], pt$lhs[i]] <- v,
           B = B0[pt$lhs[i], pt$rhs[i]] <- v,
           P = { P0[pt$lhs[i], pt$rhs[i]] <- v; P0[pt$rhs[i], pt$lhs[i]] <- v },
           T = T0[pt$lhs[i]] <- v)
  }
  rows <- free_order(pt)
  if (anyNA(kind[rows])) stop("unsupported free parameter in partable", call. = FALSE)
  li <- cbind(match(pt$rhs[rows], ov), match(pt$lhs[rows], lv))
  bi <- cbind(match(pt$lhs[rows], lv), match(pt$rhs[rows], lv))
  psi_i <- cbind(match(pt$lhs[rows], lv), match(pt$rhs[rows], lv))
  ti <- match(pt$lhs[rows], ov)
  kr <- kind[rows]
  function(theta) {
    L <- L0; B <- B0; P <- P0; Tt <- T0
    s <- kr == "L"; L[li[s, , drop = FALSE]] <- theta[s]
    s <- kr == "B"; B[bi[s, , drop = FALSE]] <- theta[s]
    s <- kr == "P"; P[psi_i[s, , drop = FALSE]] <- theta[s]; P[psi_i[s, 2:1, drop = FALSE]] <- theta[s]
    s <- kr == "T"; Tt[ti[s]] <- theta[s]
    list(Lambda = L, B = B, Psi = P, theta = unname(Tt))
  }
}

marker_admissible <- function(mats) {
  all(mats$theta >= 0) &&
    min(eigen(mats$Psi, symmetric = TRUE, only.values = TRUE)$values) >= 0
}

# Invariants from marker matrices; NA when a latent has non-positive variance.
invariants_marker <- function(st, mats) {
  m <- st$m
  A <- solve(diag(m) - mats$B)
  Phi_raw <- A %*% mats$Psi %*% t(A)
  d <- diag(Phi_raw)
  if (any(!is.finite(d)) || any(d <= 0)) return(NULL)
  D <- sqrt(d)
  Lambda <- sweep(mats$Lambda, 2, D, `*`)
  Phi <- Phi_raw / outer(D, D)
  r2 <- 1 - diag(mats$Psi) / d
  r2[st$exo] <- 0
  Sigma <- mats$Lambda %*% Phi_raw %*% t(mats$Lambda) + diag(mats$theta, st$p)
  invariants_from(st, Lambda, Phi, r2, mats$theta, Sigma)
}

# A fitted marker solution in standardized-latent coordinates (a sampler start).
marker_to_phi <- function(st, mats) {
  m <- st$m
  A <- solve(diag(m) - mats$B)
  d <- diag(A %*% mats$Psi %*% t(A))
  if (any(d <= 0)) return(NULL)
  D <- sqrt(d)
  Lambda <- sweep(mats$Lambda, 2, D, `*`)
  sgn <- sign(Lambda[cbind(st$first_ind, seq_len(m))])
  Lambda <- sweep(Lambda, 2, sgn, `*`)
  Bstd <- mats$B * outer(1 / D, D) * outer(sgn, sgn)
  Phi <- (A %*% mats$Psi %*% t(A)) / outer(D, D) * outer(sgn, sgn)
  cors <- if (nrow(st$exo_cor)) Phi[st$exo_cor] else numeric(0)
  c(Lambda[st$load_pat], mats$theta, Bstd[st$B_pat], cors)
}
