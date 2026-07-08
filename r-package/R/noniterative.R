# Non-iterative CFA estimators: fit + residual-based inference.
#
# Closed-form CFA estimation as covariance maps, exposing a complete parameter
# vector so goodness-of-fit, nested tests, and delta-method standard errors
# follow as explicit post-fit calls. Thin wrappers over the C++
# estimate::frontier / robust::frontier entry points. See
# docs/research/notes/noniterative_cfa_tests.tex.

#' Estimate the diagonal of the Guttman H matrix.
#'
#' @param S Observed covariance or correlation matrix.
#' @param blocks One simple-structure factor/block label per observed variable.
#' @param method Communality rule: `"triad_ls"` (identity-weighted triad least
#'   squares), `"anchor_triad_ls"` (same with cross-block anchor rows), `"rs"`
#'   (ratio of sums), `"ar"` (average ratio), `"gmm_block"` (blockwise triad
#'   GMM), or `"gmm_full"` (joint selected triad GMM). Historical aliases
#'   `"ilm"` and `"anchor_ilm"` are accepted.
#' @return A list with correlation-scale communalities `h2`, covariance-scale
#'   diagonal `h_diag`, and `H`, equal to `S` with the diagonal replaced by
#'   `h_diag`.
#' @export
guttman_h <- function(S, blocks,
                      method = c("triad_ls", "anchor_triad_ls", "rs", "ar",
                                 "gmm_block", "gmm_full", "ilm",
                                 "anchor_ilm")) {
  method <- match.arg(method)
  if (method == "ilm") method <- "triad_ls"
  if (method == "anchor_ilm") method <- "anchor_triad_ls"
  S <- as.matrix(S)
  if (nrow(S) != ncol(S)) {
    stop("guttman_h(): `S` must be square", call. = FALSE)
  }
  if (length(blocks) != nrow(S)) {
    stop("guttman_h(): `blocks` length must match nrow(S)", call. = FALSE)
  }
  block_id <- as.integer(factor(blocks, levels = unique(blocks)))
  if (anyNA(block_id)) {
    stop("guttman_h(): `blocks` must not contain missing values", call. = FALSE)
  }

  out <- frontier_guttman_h_impl(S, block_id, method)
  nm <- rownames(S)
  if (is.null(nm)) nm <- colnames(S)
  if (!is.null(nm)) {
    names(out$h2) <- nm
    names(out$h_diag) <- nm
    dimnames(out$H) <- list(nm, nm)
  }
  out$blocks <- blocks
  out
}

#' Fit a non-iterative CFA estimator.
#'
#' @param partable A magmaan/lavaan partable (as consumed by [fit_fit()]).
#' @param sample_stats A magmaan sample-statistics list (S, nobs, ...).
#' @param estimator Non-iterative estimator. `"guttman"` keeps the legacy
#'   lavaan-like Spearman/incidence Guttman map; `"guttman_gls_aligned"` uses the
#'   block-GLS H diagonal and aligned score reconstruction.
#' @return A magmaan fit object (same shape as [fit_fit()]), usable by the
#'   inference helpers below and by partable inspection.
#' @export
fit_noniterative_cfa <- function(partable, sample_stats, estimator = "guttman") {
  noniterative_cfa_fit_impl(partable, sample_stats, estimator)
}

#' Fit the estimator-side metric-constrained non-iterative CFA map.
#'
#' This is not the inference-side Omega projection returned by
#' [noniterative_cfa_constrained()]. It imposes the metric loading-shape
#' constraint inside the Sigma/H reconstruction itself: a common standardized
#' loading shape is estimated across groups and then converted back to the
#' marker chart represented by `partable`.
#'
#' @inheritParams fit_noniterative_cfa
#' @export
fit_noniterative_cfa_metric <- function(partable, sample_stats,
                                        estimator = "guttman_gls_aligned") {
  noniterative_cfa_metric_fit_impl(partable, sample_stats, estimator)
}

#' Fit the residual/loading restricted non-iterative CFA map.
#'
#' Residual-variance equality constraints are imposed in the Guttman H
#' communality step before the loadings are reconstructed. Loading-only linear
#' equalities are then projected with a Sigma-only composite metric. Rows that
#' mix residual and loading parameters, or rows involving factor variances,
#' intercepts, or latent means, are rejected.
#'
#' @inheritParams fit_noniterative_cfa
#' @export
fit_noniterative_cfa_restricted <- function(partable, sample_stats,
                                            estimator = "guttman_gls_aligned") {
  noniterative_cfa_restricted_fit_impl(partable, sample_stats, estimator)
}

noniterative_estimator_arg <- function(estimator) {
  if (is.null(estimator)) "auto" else estimator
}

#' Residual-based inference for a non-iterative CFA fit.
#'
#' Returns delta-method standard errors (`se`, `vcov`), the goodness-of-fit
#' statistic `T` on `df` degrees of freedom, its weighted-chi-square p-value
#' family (`p_scaled` Satorra-Bentler, `p_meanvar`, `p_scaled_shifted`,
#' `p_mixture`), the reference eigenvalues, and the Satorra-Bentler `scale_c`.
#' For `discrepancy = "ntml"`, `rls_check` is the reweighted-least-squares
#' cross-check of `T`.
#'
#' @param fit A fit from [fit_noniterative_cfa()].
#' @param discrepancy `"uls"` (unweighted least squares) or `"ntml"`
#'   (model-implied normal-theory / reweighted least squares).
#' @param gamma Fourth-moment matrix: `"nt"` (normal-theory, from the fitted
#'   Sigma) or `"empirical"` (distribution-free; requires `data`).
#' @param data Raw data matrix (rows = observations), required when
#'   `gamma = "empirical"`.
#' @param estimator Non-iterative estimator; see [fit_noniterative_cfa()]. If
#'   `NULL`, infer the estimator map from `fit`.
#' @export
noniterative_cfa_inference <- function(fit, discrepancy = c("uls", "ntml"),
                                       gamma = c("nt", "empirical"),
                                       data = NULL, estimator = NULL) {
  discrepancy <- match.arg(discrepancy)
  gamma <- match.arg(gamma)
  estimator <- noniterative_estimator_arg(estimator)
  noniterative_cfa_inference_impl(fit, estimator, discrepancy, gamma, data)
}

#' Wald test of a linear restriction R theta = q for a non-iterative CFA fit.
#'
#' Exact chi-square on `nrow(R)` degrees of freedom, using the delta-method
#' covariance. The primary nested test for these estimators.
#'
#' @param fit A fit from [fit_noniterative_cfa()].
#' @param R Restriction matrix (`k x n_free`).
#' @param q Right-hand side (length `k`).
#' @inheritParams noniterative_cfa_inference
#' @export
noniterative_cfa_wald <- function(fit, R, q, discrepancy = c("uls", "ntml"),
                                  gamma = c("nt", "empirical"), data = NULL,
                                  estimator = NULL) {
  discrepancy <- match.arg(discrepancy)
  gamma <- match.arg(gamma)
  estimator <- noniterative_estimator_arg(estimator)
  R <- as.matrix(R)
  noniterative_cfa_wald_impl(fit, R, as.numeric(q), estimator, discrepancy, gamma, data)
}

#' Difference / pseudo-LRT test between two nested non-iterative CFA fits.
#'
#' `T_d = T(H0) - T(H1)` with a weighted-chi-square reference (H1-anchored).
#' Secondary to the Wald test: `T_d` can be negative for a non-minimizing
#' estimator (flagged in `warnings`).
#'
#' @param fit0 The more-restricted (H0) fit.
#' @param fit1 The less-restricted (H1) fit.
#' @param df_d Degrees of freedom of the restriction (q_H1 - q_H0).
#' @param data0,data1 Raw data for each fit when `gamma = "empirical"`.
#' @inheritParams noniterative_cfa_inference
#' @export
noniterative_cfa_difference_test <- function(fit0, fit1, df_d,
                                             discrepancy = c("uls", "ntml"),
                                             gamma = c("nt", "empirical"),
                                             data0 = NULL, data1 = NULL,
                                             estimator = NULL) {
  discrepancy <- match.arg(discrepancy)
  gamma <- match.arg(gamma)
  estimator <- noniterative_estimator_arg(estimator)
  noniterative_cfa_difference_impl(fit0, fit1, as.integer(df_d), estimator,
                                   discrepancy, gamma, data0, data1)
}

#' Pseudo-LRT between estimator-side nested non-iterative CFA fits.
#'
#' `fit_H1` is the less restricted model and `fit_H0` is the more restricted
#' estimator-side fit, typically from [fit_noniterative_cfa_restricted()]. The
#' statistic is `T_diff = T(H0) - T(H1)` with an H1-anchored weighted-chi-square
#' reference. This is an LRT-shaped discrepancy comparison; unlike ML, it can be
#' negative for a non-minimizing estimator.
#'
#' @param fit_H1 Less restricted non-iterative fit.
#' @param fit_H0 More restricted non-iterative fit.
#' @inheritParams noniterative_cfa_inference
#' @export
noniterative_cfa_pseudo_lrt <- function(fit_H1, fit_H0,
                                        discrepancy = c("uls", "ntml"),
                                        gamma = c("nt", "empirical"),
                                        data = NULL, estimator = NULL) {
  discrepancy <- match.arg(discrepancy)
  gamma <- match.arg(gamma)
  estimator <- noniterative_estimator_arg(estimator)
  noniterative_cfa_pseudo_lrt_impl(fit_H1, fit_H0, estimator,
                                   discrepancy, gamma, data)
}

#' Grouped (multi-group / mean-structure) residual-based inference.
#'
#' Fits every group's Guttman block independently and stacks them: a
#' block-diagonal delta-method covariance (`vcov`, `se`), a joint goodness-of-fit
#' `T` on `df` degrees of freedom with the union weighted-chi-square reference,
#' and `block_of_param` (0-based group index per free parameter). Reduces to
#' [noniterative_cfa_inference()] for a single group with no mean structure.
#' Required for mean-structure models (the single-group path omits the intercept
#' block).
#'
#' @inheritParams noniterative_cfa_inference
#' @export
noniterative_cfa_grouped_inference <- function(fit, discrepancy = c("uls", "ntml"),
                                               gamma = c("nt", "empirical"),
                                               data = NULL, estimator = NULL) {
  discrepancy <- match.arg(discrepancy)
  gamma <- match.arg(gamma)
  estimator <- noniterative_estimator_arg(estimator)
  noniterative_cfa_grouped_inference_impl(fit, estimator, discrepancy, gamma, data)
}

#' Linearly-constrained non-iterative CFA fit (measurement invariance).
#'
#' Minimum-distance projection of the configural (per-group) fit onto the linear
#' equality constraints `A theta = b` carried by the partable's `group.equal`
#' family (metric / tau-equivalence / strict invariance, fixed values, ...). The
#' Omega-metric projection is closed-form:
#'   theta_tilde = theta_hat - Omega A'(A Omega A')^-1 (A theta_hat - b),
#' and the constraint statistic `W` is an exact chi-square on `k` (Wald =
#' minimum-distance duality). Returns the projected estimate (`theta_tilde`), its
#' rank-deficient covariance (`vcov_constrained`, `se_constrained`), and the
#' constraint test (`W`, `k`, `p_wald`).
#'
#' @inheritParams noniterative_cfa_inference
#' @export
noniterative_cfa_constrained <- function(fit, discrepancy = c("uls", "ntml"),
                                         gamma = c("nt", "empirical"),
                                         data = NULL, estimator = NULL) {
  discrepancy <- match.arg(discrepancy)
  gamma <- match.arg(gamma)
  estimator <- noniterative_estimator_arg(estimator)
  noniterative_cfa_constrained_impl(fit, estimator, discrepancy, gamma, data)
}

#' True (lavaan-compatible) scalar-invariance test via the reference-group map.
#'
#' Frees the latent means. Reads the common metric off the reference group and
#' solves the latent means by least squares through the reference loadings:
#' `alpha_g = (Lr'Lr)^-1 Lr'(m_g - m_r)`, `nu = m_r`. Scalar invariance is tested
#' on the mean residual `d_g = (I - P_Lr)(m_g - m_r)` with a linearized
#' pseudo-inverse Wald `W` on `df = (G-1)(p - #factors)` degrees of freedom. The
#' fit must carry mean structure (fit a configural meanstructure model first) and
#' at least two groups.
#'
#' Returns the latent means `alpha` (a list, one per non-reference group) with
#' delta-method `alpha_se` / `alpha_cov`, the common intercept `nu`, the stacked
#' residual `d`, and the test (`W`, `df`, `rank`, `p_value`).
#'
#' @param ref_group Reference group (1-based; latent means are 0 there).
#' @inheritParams noniterative_cfa_inference
#' @export
noniterative_cfa_scalar_invariance <- function(fit, ref_group = 1,
                                               discrepancy = c("uls", "ntml"),
                                               gamma = c("nt", "empirical"),
                                               data = NULL, estimator = NULL) {
  discrepancy <- match.arg(discrepancy)
  gamma <- match.arg(gamma)
  estimator <- noniterative_estimator_arg(estimator)
  noniterative_cfa_scalar_impl(fit, as.integer(ref_group), estimator,
                               discrepancy, gamma, data)
}
