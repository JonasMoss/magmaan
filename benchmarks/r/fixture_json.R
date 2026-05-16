# fixture_json.R --- shared lavaan-to-JSON fixture helpers.
#
# Side-effect-free function definitions sourced by both tools/regen_oracle.R
# (synthetic corpus fixtures) and tools/regen_parity_fixtures.R (real-data
# parity fixtures). Keeping one copy avoids the two oracle scripts drifting
# in how they serialize lavaan internals.
#
# Every function here is pure: it takes a fitted lavaan object (or plain data)
# and returns a list ready for jsonlite::write_json(). None reads script
# globals. Sourcing this file defines functions only -- it runs nothing.

# --- continuous LS helpers -------------------------------------------------

# A matrix or per-block list of matrices -> [{block, matrix}, ...].
matrix_blocks_json <- function(x) {
  if (is.list(x) && !is.data.frame(x)) {
    out <- vector("list", length(x))
    for (b in seq_along(x)) {
      out[[b]] <- list(block = as.integer(b - 1L),
                       matrix = unname(as.matrix(x[[b]])))
    }
    out
  } else {
    list(list(block = 0L, matrix = unname(as.matrix(x))))
  }
}

# Per-block sample covariance / mean / n from a fitted lavaan object.
sample_blocks_json <- function(fit) {
  sampstat <- lavInspect(fit, "sampstat")
  nobs     <- as.integer(lavInspect(fit, "nobs"))
  if (!is.list(sampstat[[1]])) {
    out <- list(sample_cov = list(list(block = 0L,
                                       matrix = unname(as.matrix(sampstat$cov)))),
                sample_mean = NULL,
                n_obs_per_block = nobs)
    if (!is.null(sampstat$mean)) {
      out$sample_mean <- list(list(block = 0L,
                                   vector = as.numeric(sampstat$mean)))
    }
    return(out)
  }

  covs <- vector("list", length(sampstat))
  means <- list()
  for (b in seq_along(sampstat)) {
    covs[[b]] <- list(block = as.integer(b - 1L),
                      matrix = unname(as.matrix(sampstat[[b]]$cov)))
    if (!is.null(sampstat[[b]]$mean)) {
      means[[b]] <- list(block = as.integer(b - 1L),
                         vector = as.numeric(sampstat[[b]]$mean))
    }
  }
  if (length(means) == 0) means <- NULL
  list(sample_cov = covs, sample_mean = means, n_obs_per_block = nobs)
}

# Point estimates, fit statistics, and the WLS weight matrix for an LS fit.
ls_fit_json <- function(fit) {
  pt_fitted <- parTable(fit)
  free_rows <- pt_fitted[pt_fitted$free > 0, ]
  free_rows <- free_rows[order(free_rows$free), ]
  fm <- fitMeasures(fit)
  samp <- sample_blocks_json(fit)
  wls_v <- lavInspect(fit, "WLS.V")

  list(theta_hat = as.numeric(free_rows$est),
       fmin      = as.numeric(fm["fmin"]),
       chisq     = as.numeric(fm["chisq"]),
       df        = as.integer(fm["df"]),
       npar      = as.integer(fm["npar"]),
       converged = isTRUE(lavInspect(fit, "converged")),
       WLS.V     = matrix_blocks_json(wls_v),
       sample_cov = samp$sample_cov,
       sample_mean = samp$sample_mean,
       n_obs = as.integer(lavInspect(fit, "ntotal")),
       n_obs_per_block = samp$n_obs_per_block)
}

# Robust SEs, Gamma, and Satorra-Bentler-family statistics for an LS fit.
ls_robust_json <- function(fit) {
  pt_fitted <- parTable(fit)
  free_rows <- pt_fitted[pt_fitted$free > 0, ]
  free_rows <- free_rows[order(free_rows$free), ]
  sb <- lavTest(fit, "satorra.bentler")
  mv <- lavTest(fit, "mean.var.adjusted")
  ss <- lavTest(fit, "scaled.shifted")
  G <- lavInspect(fit, "gamma")
  UG <- lavInspect(fit, "UGamma")
  ev <- Re(eigen(UG, only.values = TRUE)$values)
  ev <- sort(ev[is.finite(ev) & ev > 1e-8])
  list(se = as.numeric(free_rows$se),
       gamma = matrix_blocks_json(G),
       eigvals = I(as.numeric(ev)),
       chisq_standard = as.numeric(sb$scaled.test.stat),
       df = as.integer(sb$df),
       satorra_bentler = list(chisq = as.numeric(sb$stat),
                              scale = as.numeric(sb$scaling.factor),
                              df = as.integer(sb$df)),
       mean_var_adjusted = list(chisq = as.numeric(mv$stat),
                                df_adj = as.numeric(mv$df),
                                scale = as.numeric(mv$scaling.factor)),
       scaled_shifted = list(chisq = as.numeric(ss$stat),
                             scale = as.numeric(ss$scaling.factor),
                             shift = as.numeric(ss$shift.parameter),
                             df = as.integer(ss$df)))
}

# --- ordinal helpers -------------------------------------------------------

# A data frame of ordered factors -> integer matrix in observed-variable order.
ordered_to_int_matrix <- function(df, ov) {
  mat <- matrix(NA_integer_, nrow(df), length(ov), dimnames = list(NULL, ov))
  for (j in seq_along(ov)) mat[, j] <- as.integer(df[[ov[[j]]]])
  mat
}

# Strict lower triangle of M, column-major (matches the C++ vech order).
lower_tri_values <- function(M) {
  p <- nrow(M)
  out <- numeric(p * (p - 1) / 2)
  k <- 1L
  for (j in seq_len(p)) {
    if (j < p) {
      for (i in seq.int(j + 1L, p)) {
        out[k] <- M[i, j]
        k <- k + 1L
      }
    }
  }
  out
}

# A matrix or per-block list of matrices -> [{block, matrix}, ...].
matrix_list_json <- function(x) {
  if (is.list(x) && !is.data.frame(x)) {
    lapply(seq_along(x), function(b) list(block = as.integer(b - 1L),
                                          matrix = unname(as.matrix(x[[b]]))))
  } else {
    list(list(block = 0L, matrix = unname(as.matrix(x))))
  }
}

# Per-block ordinal sample statistics: thresholds, polychorics, NACOV, weights.
ordinal_samp_json <- function(fit) {
  ss <- lavInspect(fit, "sampstat")
  G  <- lavInspect(fit, "gamma")
  W  <- lavTech(fit, "WLS.V")
  if (!is.list(ss) || (!is.null(ss$cov) || !is.null(ss$th))) ss <- list(ss)
  if (!is.list(G)) G <- list(G)
  if (!is.list(W)) W <- list(W)
  lapply(seq_along(ss), function(b) {
    cov <- unname(as.matrix(ss[[b]]$cov))
    th <- as.numeric(ss[[b]]$th)
    list(block = as.integer(b - 1L),
         nobs = as.integer(lavInspect(fit, "nobs")[b]),
         thresholds = th,
         polychoric = cov,
         moments = c(th, lower_tri_values(cov)),
         NACOV = unname(as.matrix(G[[b]])),
         WLS.V = unname(as.matrix(W[[b]])),
         WLS.VD = diag(1 / diag(as.matrix(G[[b]])), nrow = nrow(as.matrix(G[[b]]))))
  })
}

# Point estimates and fit statistics for an ordinal DWLS/WLS fit.
ordinal_fit_json <- function(fit) {
  pt <- parTable(fit)
  free <- pt[pt$free > 0, ]
  free <- free[order(free$free), ]
  fm <- fitMeasures(fit)
  list(converged = isTRUE(lavInspect(fit, "converged")),
       theta_hat = as.numeric(free$est),
       free_rows = lapply(seq_len(nrow(free)), function(i) {
         list(lhs = as.character(free$lhs[i]),
              op = as.character(free$op[i]),
              rhs = as.character(free$rhs[i]),
              group = as.integer(if (!is.null(free$group)) free$group[i] else 1L),
              free = as.integer(free$free[i]),
              est = as.numeric(free$est[i]))
       }),
       chisq = as.numeric(fm["chisq"]),
       df = as.integer(fm["df"]),
       cfi = as.numeric(fm["cfi"]),
       tli = as.numeric(fm["tli"]),
       rmsea = as.numeric(fm["rmsea"]),
       srmr = as.numeric(fm["srmr"]))
}

# Robust SEs and Satorra-Bentler-family statistics for an ordinal fit.
ordinal_robust_json <- function(fit) {
  pt <- parTable(fit)
  free <- pt[pt$free > 0, ]
  free <- free[order(free$free), ]
  sb <- lavTest(fit, "satorra.bentler")
  mv <- lavTest(fit, "mean.var.adjusted")
  ss <- lavTest(fit, "scaled.shifted")
  UG <- lavInspect(fit, "UGamma")
  ev <- Re(eigen(UG, only.values = TRUE)$values)
  ev <- sort(ev[is.finite(ev) & ev > 1e-8])
  list(se = as.numeric(free$se),
       eigvals = as.numeric(ev),
       chisq_standard = as.numeric(sb$scaled.test.stat),
       df = as.integer(sb$df),
       satorra_bentler = list(chisq = as.numeric(sb$stat),
                              scale = as.numeric(sb$scaling.factor),
                              df = as.integer(sb$df)),
       mean_var_adjusted = list(chisq = as.numeric(mv$stat),
                                df_adj = as.numeric(mv$df),
                                scale = as.numeric(mv$scaling.factor)),
       scaled_shifted = list(chisq = as.numeric(ss$stat),
                             scale = as.numeric(ss$scaling.factor),
                             shift = as.numeric(ss$shift.parameter),
                             df = as.integer(ss$df)))
}

# --- raw-data helper -------------------------------------------------------

# A data frame -> {X, mask}: X is the raw matrix (NA preserved), mask is
# 1 = observed / 0 = missing. Used for FIML raw-data fixtures.
raw_block_json <- function(df, ov) {
  X <- as.matrix(df[, ov, drop = FALSE])
  M <- ifelse(is.na(X), 0L, 1L)
  list(X = unname(X), mask = unname(M))
}
