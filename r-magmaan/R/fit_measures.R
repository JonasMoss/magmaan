#' Policy fit measures
#'
#' Computes fit indices on request, fitting an independence baseline with the
#' same estimator and data treatment. ML/FIML target likelihood discrepancy;
#' continuous ULS targets squared covariance discrepancy; ML2S-NT uses Stage-1
#' Gamma; ordinal and mixed DWLS target discrepancy in the DWLS metric.
#' Misspecification-robust trace corrections subtract estimated sampling bias
#' before computing RMSEA, CFI and TLI. SRMR/CRMR subtract residual influence
#' traces. Multigroup RMSEA includes sqrt(G); residuals pool across groups.
#'
#' DWLS indices describe misfit in the DWLS metric: conventional ML cutoffs
#' do not transfer (Savalei, 2021; Xia and Yang, 2019). See Brosseau-Liard,
#' Savalei and Li (2012) and Savalei (2018, 2021) for robust corrections.
#' These component checks do not establish finite-sample bias or coverage.
#' Intervals and close-fit p-values await the registered evaluation study.
#'
#' Results have a fixed row set per estimator, including unavailable indices with NA
#' estimates and typed reasons. Likelihood criteria appear only for
#' likelihood estimators; CRMR appears for ordinal and mixed models. Computation includes a second fit and
#' is performed each time this function is called.
#' @param fit A [magmaan()] fit.
#' @param lavaan_compat Optional named lavaan inference bundle. Returns lavaan's
#'   standard, scaled and robust families, intervals and close-fit p-values.
#'   Unsupported estimator/bundle combinations error. FIML ML/MLR compatibility
#'   rows are unavailable pending validation against lavaan.
#' @return A data frame with columns index, estimate and reason. Stable index
#'   codes are rmsea, cfi, tli, srmr, crmr, logl, unrestricted_logl, aic and bic.
#'   The details attribute retains the lab composer's diagnostic ingredients.
#' @export
fit_measures <- function(fit, lavaan_compat = NULL) {
  if (!inherits(fit, "magmaan"))
    stop("fit_measures(): supply a magmaan() fit", call. = FALSE)
  if (!is.null(lavaan_compat))
    return(magmaanlab::convention_fit_measures(as_lab_fit(fit), lavaan_compat))
  out <- magmaanlab::policy_fit_measures(as_lab_fit(fit))
  out$index[out$index == "unrestricted.logl"] <- "unrestricted_logl"
  out
}
