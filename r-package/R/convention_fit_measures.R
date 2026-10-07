#' Lavaan-compatible fit measures
#'
#' Composes standard, scaled and robust fit-index families in C++, including
#' an independence baseline under the selected convention. Model estimates
#' are retained. Compatibility intervals and close-fit tests use lavaan's
#' noncentral chi-square conventions; they are separate from policy indices.
#' @param fit A fitted magmaanlab model retaining its observations or ordinal stats.
#' @param convention A named lavaan inference bundle.
#' @return A data frame with index, estimate and reason columns and a
#'   lavaan_compat attribute naming the convention. Unsupported bundles error.
#' @export
convention_fit_measures <- function(fit, convention) {
  if (!inherits(fit, "magmaan_fit")) stop("convention_fit_measures(): supply a fitted model")
  convention <- match.arg(convention, c("ML", "MLM", "MLR", "DWLS", "WLSMV", "WLSM", "ULS", "ULSMV", "WLS"))
  convention_fit_measures_impl(fit, convention, .policy_state(fit))
}
