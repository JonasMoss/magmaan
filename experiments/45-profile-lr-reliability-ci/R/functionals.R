# The reliability family as functional closures over the engine's z-vector. Each
# builder returns a function g(z) -> scalar; the engine finite-differences the
# constraint Jacobian (grad = NULL). Analytic gradients are the planned in-core
# optimization; the NT parity milestone does not need them, and semlbci itself
# differentiates the := algebra numerically, so FD keeps the two comparable.
#
# All coefficients are functions of the model-implied Sigma = Lambda Lambda' +
# diag(psi) with unit factor variances (std.lv), so they are scale-free in the
# usual reliability sense. `eng` supplies unpack(); `cols` names which loading
# column is the general/target factor.

# omega_total: proportion of unit-weighted total-score variance that is common,
#   1 - 1' Theta 1 / 1' Sigma 1.
# For a single common factor this equals (sum lambda)^2 / [(sum lambda)^2 + sum psi];
# the general form also covers the orthogonal bifactor (all common variance).
make_omega_total <- function(eng) {
  function(z) {
    u <- eng$unpack(z)
    Sig <- tcrossprod(u$Lam) + diag(u$psi)
    vt <- sum(Sig)
    1 - sum(u$psi) / vt
  }
}

# omega_h (hierarchical): share of total-score variance due to the GENERAL factor
# alone (column `gcol`, default 1): (1' lambda_G)^2 / 1' Sigma 1.
make_omega_h <- function(eng, gcol = 1L) {
  function(z) {
    u <- eng$unpack(z)
    Sig <- tcrossprod(u$Lam) + diag(u$psi)
    sG <- sum(u$Lam[, gcol])
    sG^2 / sum(Sig)
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
    u <- eng$unpack(z)
    lam <- u$Lam[, tcol]
    Sig <- tcrossprod(u$Lam) + diag(u$psi)
    E <- Sig - tcrossprod(lam)
    qv <- as.numeric(crossprod(lam, solve(E, lam)))
    1 / (1 + 1 / qv)
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
