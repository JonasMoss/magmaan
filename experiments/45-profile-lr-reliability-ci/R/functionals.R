# The reliability family as functional closures over the engine's z-vector. Each
# builder returns a function g(z) -> list(value, grad) with grad the ANALYTIC
# gradient in z-space (free loadings in the engine's ordering, then d/d log psi);
# the engine uses it directly for the constraint Jacobian (no finite differencing),
# which is what makes the constrained fits cheap enough for nested bootstraps. The
# gradients are validated against central FD to ~1e-11. semlbci differentiates its
# := algebra numerically, so the NT parity check is unaffected (it compares bounds,
# not Jacobians).
#
# All coefficients are functions of the model-implied Sigma = Lambda Lambda' +
# diag(psi) with unit factor variances (std.lv), so they are scale-free in the
# usual reliability sense. `eng` supplies unpack(), cols, m, p. z-gradients: for a
# free loading z holds L_{ab} directly (dg/dz = dg/dL_{ab}); for a residual z holds
# log psi_i (dg/dz = psi_i * dg/dpsi_i).

# omega_total: proportion of unit-weighted total-score variance that is common,
#   1 - 1' Theta 1 / 1' Sigma 1.
# For a single common factor this equals (sum lambda)^2 / [(sum lambda)^2 + sum psi];
# the general form also covers the orthogonal bifactor (all common variance).
make_omega_total <- function(eng) {
  function(z) {
    u <- eng$unpack(z); Lam <- u$Lam; psi <- u$psi
    vt <- sum(tcrossprod(Lam)) + sum(psi); Spsi <- sum(psi); cmn <- vt - Spsi
    # dg/dL_{ij} = 2 Spsi colsum_j / vt^2 (const across i); dg/dpsi_i = -cmn/vt^2
    dL <- matrix(rep(2 * Spsi * colSums(Lam) / vt^2, each = eng$p), eng$p, eng$m)
    dlogpsi <- (-cmn / vt^2) * psi
    list(value = 1 - Spsi / vt,
         grad = c(unlist(lapply(seq_len(eng$m), function(j) dL[eng$cols[[j]], j])), dlogpsi))
  }
}

# omega_h (hierarchical): share of total-score variance due to the GENERAL factor
# alone (column `gcol`, default 1): (1' lambda_G)^2 / 1' Sigma 1.
make_omega_h <- function(eng, gcol = 1L) {
  function(z) {
    u <- eng$unpack(z); Lam <- u$Lam; psi <- u$psi
    vt <- sum(tcrossprod(Lam)) + sum(psi); sG <- sum(Lam[, gcol]); colsum <- colSums(Lam)
    # dg/dL_{i,gcol} = 2 sG (vt - sG^2)/vt^2; dg/dL_{ij, j!=gcol} = -2 sG^2 colsum_j/vt^2
    dL <- matrix(0, eng$p, eng$m)
    for (j in seq_len(eng$m))
      dL[, j] <- if (j == gcol) 2 * sG * (vt - sG^2) / vt^2 else -2 * sG^2 * colsum[j] / vt^2
    dlogpsi <- (-sG^2 / vt^2) * psi
    list(value = sG^2 / vt,
         grad = c(unlist(lapply(seq_len(eng$m), function(j) dL[eng$cols[[j]], j])), dlogpsi))
  }
}

# H / coefficient of construct reliability = maximal reliability of the optimal
# linear composite for a target factor (column `tcol`, default 1), treating all
# other variance as error:
#   H = 1 / (1 + 1 / (lambda_t' (Sigma - lambda_t lambda_t')^-1 lambda_t)).
# With a single factor and diagonal Theta this reduces to
#   q = sum(lambda_i^2 / psi_i),  H = q / (1 + q),
# which is what the semlbci := oracle uses for the 1-factor check.
make_H <- function(eng, tcol = 1L) {
  function(z) {
    u <- eng$unpack(z); Lam <- u$Lam; psi <- u$psi; lam <- Lam[, tcol]
    E <- tcrossprod(Lam) + diag(psi) - tcrossprod(lam)
    w <- solve(E, lam); q <- as.numeric(crossprod(lam, w)); dgdq <- 1 / (1 + q)^2
    # dq/dL_{.,tcol} = 2 w; dq/dL_{.,j!=tcol} = -2 w (L[,j]'w); dq/dpsi_i = -w_i^2
    cb <- as.numeric(crossprod(Lam, w))
    dL <- matrix(0, eng$p, eng$m)
    for (j in seq_len(eng$m)) dL[, j] <- if (j == tcol) 2 * w else -2 * w * cb[j]
    dlogpsi <- -w^2 * psi
    grad_q <- c(unlist(lapply(seq_len(eng$m), function(j) dL[eng$cols[[j]], j])), dlogpsi)
    list(value = q / (1 + q), grad = dgdq * grad_q)
  }
}

# --- lavaan := oracle syntax for each coefficient -----------------------------
# Return the model syntax plus a `:=` block defining the coefficient under the
# same std.lv orthogonal identification, so semlbci can serve as the NT oracle.
# `ov` are indicator names; `cols` the loading pattern (list of item-index
# vectors); loadings are labelled l<factor>_<item>, residuals e<item>.

lav_labels <- function(ov, cols) {
  p <- length(ov); m <- length(cols)
  loads <- vector("list", m)
  for (j in seq_len(m)) loads[[j]] <- paste0("l", j, "_", cols[[j]])
  list(loads = loads, resid = paste0("e", seq_len(p)))
}

lav_model_block <- function(ov, cols, facs = NULL) {
  m <- length(cols)
  if (is.null(facs)) facs <- if (m == 1) "f" else c("g", paste0("s", seq_len(m - 1L)))
  lab <- lav_labels(ov, cols)
  meas <- vapply(seq_len(m), function(j)
    paste0(facs[j], " =~ ",
           paste0(lab$loads[[j]], "*", ov[cols[[j]]], collapse = " + ")),
    character(1))
  res <- paste0(ov, " ~~ ", lab$resid, "*", ov)
  list(syntax = paste(c(meas, res), collapse = "\n"), facs = facs, lab = lab)
}

# omega_total := (common variance) / (total variance), common variance written
# as the sum of squared column loading-sums (orthogonal factors) so it is
# expressible in the := algebra without a matrix inverse.
lav_omega_total <- function(ov, cols) {
  mb <- lav_model_block(ov, cols); lab <- mb$lab
  colsum_sq <- vapply(seq_along(cols), function(j)
    paste0("(", paste(lab$loads[[j]], collapse = "+"), ")^2"), character(1))
  common <- paste(colsum_sq, collapse = " + ")
  err <- paste(lab$resid, collapse = "+")
  defs <- c(paste0("cmn := ", common),
            paste0("tot := cmn + ", err),
            "rel := cmn/tot")
  paste(c(mb$syntax, defs), collapse = "\n")
}

lav_omega_h <- function(ov, cols, gcol = 1L) {
  mb <- lav_model_block(ov, cols); lab <- mb$lab
  colsum_sq <- vapply(seq_along(cols), function(j)
    paste0("(", paste(lab$loads[[j]], collapse = "+"), ")^2"), character(1))
  tot <- paste0(paste(colsum_sq, collapse = " + "), " + ",
                paste(lab$resid, collapse = "+"))
  sG2 <- paste0("(", paste(lab$loads[[gcol]], collapse = "+"), ")^2")
  defs <- c(paste0("num := ", sG2), paste0("den := ", tot), "rel := num/den")
  paste(c(mb$syntax, defs), collapse = "\n")
}

# H for a single-factor target with diagonal residuals: q = sum l_i^2/e_i.
lav_H_diag <- function(ov, cols, tcol = 1L) {
  mb <- lav_model_block(ov, cols); lab <- mb$lab
  terms <- paste0(lab$loads[[tcol]], "^2/", lab$resid[cols[[tcol]]])
  defs <- c(paste0("q := ", paste(terms, collapse = " + ")), "rel := q/(1+q)")
  paste(c(mb$syntax, defs), collapse = "\n")
}
