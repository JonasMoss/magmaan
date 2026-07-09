# Post-fit interface for the non-iterative (Guttman) closed-form CFA estimators.
#
# The C++ measure bodies (standardized solution, residuals/SRMR, factor scores,
# composite weights, defined parameters) are estimator-agnostic: they rebuild the
# model context from the fitted partable/S/theta and take an explicit vcov where
# standard errors are needed. So a non-iterative fit reuses the SAME post-fit
# calls as ML/LS once it (a) carries the "magmaan_fit" class for S3 dispatch and
# (b) can hand back the delta-method covariance through vcov(). This file adds
# that thin glue, a general parameter-table assembler, fit indices over the NT
# and ULS residual discrepancies (naive and robust/mixture-scaled), and clean
# not-implemented guards for the ML score / LRT machinery, which assumes a
# gradient-zero minimizer and does not transfer to a closed-form estimator.
# Derivations: papers/guttman-inference/dev/notes/noniterative_fit_indices.tex
# and noniterative_modification_indices.tex.

# ---- classing / predicates -------------------------------------------------

# Turn the raw fit_result list returned by the noniterative_cfa_*_impl calls
# into a first-class magmaan_fit. The closed-form map "converged" by
# construction (there is no optimizer), and `fmin` is a meaningless 0 for these
# estimators, so downstream code must route through the residual GOF, not
# `infer_chi2_stat(ss, fit$fmin)`.
.finalize_noniterative_fit <- function(out) {
  out$noniterative <- TRUE
  out$converged <- TRUE
  class(out) <- c("magmaan_noniterative_fit", "magmaan_fit", "list")
  out
}

.is_noniterative <- function(fit) {
  isTRUE(fit$noniterative) || inherits(fit, "magmaan_noniterative_fit")
}

.guard_noniterative <- function(fn) {
  stop(fn, " is not defined for non-iterative / Guttman closed-form CFA fits: ",
       "the ML score / LRT machinery assumes a gradient-zero minimizer, which a ",
       "closed-form map is not. Use the residual-based nested tests instead ",
       "(noniterative_cfa_wald / _difference_test / _pseudo_lrt / _constrained / ",
       "_scalar_invariance).", call. = FALSE)
}

# ---- general parameter table -----------------------------------------------

#' Assemble a lavaan-shaped parameter table (est / se / z / p / CI) from a fit
#' and an explicit parameter covariance.
#'
#' Estimator-agnostic: works for any magmaan fit given a `vcov` on the free
#' parameters. Rows are the free-parameter block of the fitted partable, in theta
#' order (by the `free` index), so `est`, `se`, `z`, and `pvalue` are aligned.
#'
#' @param fit A fitted magmaan model.
#' @param vcov Free-parameter covariance (`npar x npar`), e.g.
#'   `vcov(fit, regime = "model")` or `noniterative_cfa_se(fit)$vcov`.
#' @param level Confidence level for the Wald CI columns.
#' @return A data frame with `lhs`, `op`, `rhs`, `group`, `est`, `se`, `z`,
#'   `pvalue`, `ci.lower`, `ci.upper`.
#' @export
parameter_table <- function(fit, vcov, level = 0.95) {
  if (missing(vcov)) {
    stop("parameter_table(): `vcov` is required; compute it explicitly ",
         "(e.g. vcov(fit) or noniterative_cfa_se(fit)$vcov)", call. = FALSE)
  }
  pt <- fit$partable
  free <- as.integer(pt$free)
  keep <- which(free > 0L)
  ord <- keep[order(free[keep])]            # partable rows in theta (free) order
  se <- as.numeric(magmaan_core$infer_se(vcov))
  zt <- magmaan_core$infer_z_test(fit, se)
  est <- as.numeric(fit$theta)
  if (length(est) != length(ord)) {
    stop("parameter_table(): free-parameter count (", length(ord),
         ") does not match length(theta) (", length(est), ")", call. = FALSE)
  }
  zq <- stats::qnorm(1 - (1 - level) / 2)
  data.frame(
    lhs = pt$lhs[ord], op = pt$op[ord], rhs = pt$rhs[ord],
    group = if (!is.null(pt$group)) pt$group[ord] else 1L,
    est = est, se = se, z = as.numeric(zt$z), pvalue = as.numeric(zt$pvalue),
    ci.lower = est - zq * se, ci.upper = est + zq * se,
    stringsAsFactors = FALSE, row.names = NULL
  )
}

# ---- fit indices -----------------------------------------------------------

# Per-block covariances and per-block N for a fit (fit$S is always a per-group
# list; fit$nobs is a scalar or per-group vector).
.noniter_blocks <- function(fit) {
  Sl <- fit$S
  if (!is.list(Sl)) Sl <- list(Sl)
  nb <- as.numeric(fit$nobs)
  if (length(nb) == 1L && length(Sl) > 1L) nb <- rep(nb, length(Sl))
  list(S = Sl, n = nb)
}

.noniter_needs_grouped_inference <- function(fit) {
  length(.noniter_blocks(fit)$S) > 1L ||
    any(as.character(fit$partable$op %||% character()) == "~1")
}

.noniter_raw_data_arg <- function(fit, data) {
  if (is.null(data)) return(NULL)
  raw <- raw_data_arg(fit, data)
  if (is.list(raw) && !is.null(raw$X)) raw <- raw$X
  if (is.list(raw) && !is.data.frame(raw)) {
    X <- lapply(raw, as.matrix)
    if (length(X) == 1L) return(X[[1L]])
    return(list(X = X))
  }
  as.matrix(raw)
}

.noniter_raw_blocks <- function(fit, data) {
  raw <- .noniter_raw_data_arg(fit, data)
  if (is.null(raw)) return(NULL)
  if (is.list(raw) && !is.null(raw$X)) return(lapply(raw$X, as.matrix))
  if (is.list(raw) && !is.data.frame(raw)) return(lapply(raw, as.matrix))
  list(as.matrix(raw))
}

# Independence-model baseline statistic under the SAME residual discrepancy as
# the user statistic (so the incremental indices are internally coherent). The
# baseline residual is exactly the off-diagonals of S; the implied Sigma is
# diag(S). Matches the C++ convention T = N * r' V r locked in the derivation
# note: ULS r'Vr = 2 * sum_{i<j} S_ij^2; NTML (diagonal Sigma) r'Vr =
# sum_{i<j} S_ij^2 / (S_ii S_jj).
.noniter_baseline <- function(fit, discrepancy) {
  bl <- .noniter_blocks(fit)
  chi2 <- 0
  df <- 0L
  for (b in seq_along(bl$S)) {
    S <- bl$S[[b]]
    p <- ncol(S)
    if (p < 2L) next
    ut <- upper.tri(S)
    off <- S[ut]
    rVr <- if (identical(discrepancy, "uls")) {
      2 * sum(off^2)
    } else {
      d <- diag(S)
      sum(off^2 / outer(d, d)[ut])
    }
    chi2 <- chi2 + bl$n[b] * rVr
    df <- df + p * (p - 1L) / 2L
  }
  list(chi2 = chi2, df = as.integer(df))
}

# Robust (mixture) scaling factor c = sum(lambda)/df for the baseline reference
# spectrum eig(M_b' V_b M_b Gamma). For the independence model M_b keeps exactly
# the off-diagonal coordinates, so c reduces to a weighted trace over off-diagonal
# entries: c = [sum_{i<j} w_ij * Gamma_{(ij),(ij)}] / df, with the discrepancy
# weight w_ij (ULS: 2; NTML: 1/(S_ii S_jj)). Under gamma = "nt" the baseline
# Gamma is model-implied at diag(S), so Gamma_{(ij),(ij)} = S_ii S_jj and the
# NTML factor collapses to 1 exactly. Under gamma = "empirical" it is the
# raw-data ACOV of the cross-products. Derivation in the note.
.noniter_baseline_scale <- function(fit, discrepancy, gamma, data = NULL) {
  bl <- .noniter_blocks(fit)
  Xl <- NULL
  if (identical(gamma, "empirical")) {
    if (is.null(data)) {
      stop("fit_measures(gamma = 'empirical'): raw `data` is required for the ",
           "robust baseline scaling", call. = FALSE)
    }
    Xl <- .noniter_raw_blocks(fit, data)
  }
  num <- 0
  df <- 0L
  for (b in seq_along(bl$S)) {
    S <- bl$S[[b]]
    p <- ncol(S)
    if (p < 2L) next
    ut <- upper.tri(S)
    d <- diag(S)
    gdiag <- if (identical(gamma, "empirical")) {
      Xb <- as.matrix(Xl[[b]])
      w <- scale(Xb, center = TRUE, scale = FALSE)
      nb <- nrow(w)
      ii <- row(S)[ut]; jj <- col(S)[ut]
      vapply(seq_along(ii), function(k) {
        cp <- w[, ii[k]] * w[, jj[k]]
        mean((cp - mean(cp))^2)
      }, numeric(1))
    } else {
      (outer(d, d))[ut]            # model-implied NT: S_ii S_jj (off-diagonal)
    }
    w_ij <- if (identical(discrepancy, "uls")) {
      rep(2, sum(ut))
    } else {
      1 / (outer(d, d)[ut])
    }
    num <- num + sum(w_ij * gdiag)
    df <- df + p * (p - 1L) / 2L
  }
  if (df == 0L) return(NA_real_)
  num / df
}

#' Fit indices (CFI/TLI/RMSEA + SRMR) for a non-iterative CFA fit.
#'
#' The user statistic is the residual goodness-of-fit `T` on `df` degrees of
#' freedom under the chosen `discrepancy`, referred to an independence-model
#' baseline under the SAME discrepancy. Naive indices treat `T` as central
#' chi-square; robust indices use the Satorra-Bentler scaling `scale_c` (user) and
#' the baseline scaling (independence model) to correct the mixture reference.
#' `logl`/`unrestricted.logl`/`aic`/`bic`/`bic2` are reported as `NA`: these are
#' likelihood information criteria and are not defined for the closed-form
#' non-ML estimator.
#'
#' @param fit A fit from [fit_noniterative_cfa()].
#' @param discrepancy `"ntml"` (model-implied normal-theory) or `"uls"`.
#' @param gamma `"nt"` (normal-theory) or `"empirical"` (needs `data`).
#' @param data Raw data, required when `gamma = "empirical"`.
#' @param baseline Optional precomputed `list(chi2=, df=)`; default is the
#'   same-discrepancy independence model.
#' @param fmg Optional passthrough to [fmg_tests()].
#' @export
fit_measures_noniterative <- function(fit, discrepancy = c("ntml", "uls"),
                                      gamma = c("nt", "empirical"), data = NULL,
                                      baseline = NULL, fmg = NULL) {
  discrepancy <- match.arg(discrepancy)
  gamma <- match.arg(gamma)
  if (!is.null(fmg) && !identical(fmg, FALSE)) {
    stop("fit_measures(): fmg = ... tests are not available for non-iterative ",
         "fits; the residual GOF is already scaled. Use the noniterative_cfa_* ",
         "nested tests for difference testing.", call. = FALSE)
  }
  raw <- if (identical(gamma, "empirical")) .noniter_raw_data_arg(fit, data) else NULL
  inf <- if (.noniter_needs_grouped_inference(fit)) {
    noniterative_cfa_grouped_inference(fit, discrepancy = discrepancy,
                                       gamma = gamma, data = raw)
  } else {
    noniterative_cfa_inference(fit, discrepancy = discrepancy,
                               gamma = gamma, data = raw)
  }
  Tu <- inf$T
  dfu <- inf$df
  cu <- inf$scale_c
  if (is.null(baseline)) baseline <- .noniter_baseline(fit, discrepancy)
  cb <- .noniter_baseline_scale(fit, discrepancy, gamma, raw)
  n <- sum(as.numeric(fit$nobs))
  g <- as.integer(fit$ngroups %||% length(fit$nobs) %||% 1L)

  # Naive block + SRMR from the shared C++ closed form; likelihood criteria are
  # intentionally blanked below because the estimator is not an ML fit.
  fm <- magmaan_core$measures_fit(fit, Tu, dfu,
                                  list(chi2 = baseline$chi2, df = baseline$df))
  fm[c("logl", "unrestricted.logl", "aic", "bic", "bic2")] <- NA_real_

  # Robust / mixture-scaled additions.
  cfi_r <- .fit_cfi(Tu, dfu, baseline$chi2, baseline$df, c_hat = cu,
                    c_hat_null = cb, robust = TRUE)
  tli_r <- .fit_tli(Tu, dfu, baseline$chi2, baseline$df, c_hat = cu,
                    c_hat_null = cb, robust = TRUE)
  rm_r <- .fit_rmsea_family(Tu, dfu, n, g, c_hat = cu)

  c(list(
    chisq = Tu, df = dfu,
    pvalue = infer_chi2_pvalue(Tu, dfu),
    baseline.chisq = baseline$chi2, baseline.df = baseline$df,
    discrepancy = discrepancy, gamma = gamma,
    scale.c = cu, baseline.scale.c = cb
  ), fm, list(
    cfi.robust = cfi_r,
    tli.robust = tli_r,
    rmsea.robust = rm_r$rmsea,
    rmsea.ci.lower.robust = rm_r$rmsea.ci.lower,
    rmsea.ci.upper.robust = rm_r$rmsea.ci.upper,
    rmsea.pvalue.robust = rm_r$rmsea.pvalue
  ))
}
