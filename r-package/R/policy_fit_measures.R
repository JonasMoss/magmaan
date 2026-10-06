#' Policy fit indices with misspecification-robust point corrections
#'
#' Computes indices on request, including an independence baseline under the
#' same estimator and data treatment. ML and FIML use the raw-likelihood
#' Takeuchi trace difference; ordinal and mixed DWLS use exact first-stage
#' influence with the estimated-weight channel. Continuous ULS uses its
#' profile trace plus covariance moment bias; ML2S-NT uses Stage-1 Gamma. No intervals are returned.
#' RMSEA includes the multigroup factor sqrt(G). TLI uses nominal degrees of
#' freedom and truncated corrected discrepancies; the lab misspecification
#' family's generalized-df TLI remains a separate comparator. Residual indices
#' pool squared residuals across groups and subtract their influence trace.
#' DWLS indices measure misfit in the DWLS metric; ML cutoffs do not transfer.
#' @param fit A fitted magmaanlab model retaining raw data or categorical stats.
#' @return A data frame with index, estimate and reason, and a details attribute
#'   carrying discrepancies, traces, corrected discrepancies and degrees of
#'   freedom. Unsupported components have NA estimates and explicit reasons.
#' @export
policy_fit_measures <- function(fit) {
  if (!inherits(fit, "magmaan_fit")) stop("policy_fit_measures(): supply a fitted magmaan model")
  policy_fit_measures_impl(fit, .policy_state(fit))
}
