.fit_measure_fit <- function(fit) {
  if (identical(toupper(fit$estimator), "FIML")) {
    context <- .policy_context(fit, "fit_measures", function()
      list(native = fit_measure_cache_impl()))
    attr(fit, "fit_measure_native") <- context$native
  }
  fit
}

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
  policy_fit_measures_impl(.fit_measure_fit(fit), .policy_state(fit))
}


#' Candidate continuous misspecification-robust fit-index intervals
#'
#' Lab-only Wald intervals for continuous complete-data ML and FIML with raw
#' observations, random x and affine equality constraints. The point estimates
#' are those of [policy_fit_measures()]. No ordinary-package default is changed.
#' `wald_f` forms a Wald interval for the corrected discrepancy and maps it to
#' RMSEA; `wald_sqrt` forms it in the square-root coordinate and requires a
#' positive corrected discrepancy. CFI and TLI use the joint user/baseline
#' first-order delta method plus the user discrepancy's second-order variance.
#' Their delta intervals are the same for both coordinate variants. CFI bounds
#' are clipped to [0,1]; TLI bounds are unbounded. RMSEA lower bounds are zero.
#' The second-order operator uses saturated observed information, the saturated
#' score outer product and the constraint-reduced model observed information;
#' covariance-only ML includes its profiled group-mean nuisance parameters.
#' This variance candidate has not been adopted or coverage-calibrated.
#' @param fit A fitted magmaanlab model retaining raw observations.
#' @param level Confidence level strictly between zero and one, default 0.90.
#' @return A data frame with index, variant, estimate, lower, upper, V1, V2 and
#'   typed reason. V1 and V2 always describe the user discrepancy variance,
#'   before mapping to any index. The details attribute includes casewise and
#'   fit discrepancies, the corrected discrepancy and both trace diagnostics.
#'   Unavailable intervals have NA bounds and explicit reasons.
#' @export
fit_index_intervals_misspec <- function(fit, level = 0.90) {
  if (!inherits(fit, "magmaan_fit")) stop("fit_index_intervals_misspec(): supply a fitted magmaan model")
  if (!is.numeric(level) || length(level) != 1L || is.na(level) ||
      !is.finite(level) || level <= 0 || level >= 1)
    stop("level must be strictly between zero and one")
  fit_index_intervals_misspec_impl(.fit_measure_fit(fit), level, .policy_state(fit))
}
