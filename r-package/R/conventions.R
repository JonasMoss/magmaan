#' Named lavaan inference on retained estimates
#'
#' Thin adapters to C++ compatibility composers. A convention changes the
#' reported inference without changing fitting. Components outside the checked
#' model/data slice return an unavailable reason.
#' @param fit A fitted `magmaan_fit`.
#' @param convention A lavaan bundle: `"ML"`, `"MLM"`, `"MLR"`, `"DWLS"`,
#'   `"WLSMV"`, `"ULS"`, `"ULSMV"` or `"WLS"`.
#' @return A list with covariance and test results, component availability and
#'   convergence metadata. Nested inference returns one difference test.
#' @name convention_inference
#' @export
convention_inference <- function(fit, convention) {
  if (!inherits(fit, "magmaan_fit")) stop("convention_inference(): supply a fitted model")
  convention <- match.arg(convention, c("ML", "MLM", "MLR", "DWLS", "WLSMV", "ULS", "ULSMV", "WLS"))
  state <- .policy_state(fit)
  context <- NULL
  if (identical(fit$estimator, "ML") && !isTRUE(fit$ordinal) &&
      !isTRUE(fit$mixed_ordinal) && is.null(fit$nclusters) && state[[1]] && !state[[4]]) {
    context <- tryCatch(prepare_inference(fit), error = function(e) NULL)
  }
  convention_inference_impl(fit, context$native, convention, state)
}

#' @rdname convention_inference
#' @param fit_H1,fit_H0 Alternative and restricted fits to the same data.
#' @export
convention_nested <- function(fit_H1, fit_H0, convention) {
  if (!inherits(fit_H1, "magmaan_fit") || !inherits(fit_H0, "magmaan_fit"))
    stop("convention_nested(): supply two fitted models")
  convention <- match.arg(convention, c("ML", "MLM", "MLR", "DWLS", "WLSMV", "ULS", "ULSMV", "WLS"))
  if (!identical(fit_H1$raw_data, fit_H0$raw_data))
    stop("convention_nested(): the fits must use the same observations in the same order")
  states <- list(H0 = .policy_state(fit_H0), H1 = .policy_state(fit_H1))
  contexts <- list()
  if (all(vapply(list(fit_H0, fit_H1), function(f)
      identical(f$estimator, "ML") && !isTRUE(f$ordinal) && !isTRUE(f$mixed_ordinal) &&
        is.null(f$nclusters), logical(1))) && states$H0[[1]] && states$H1[[1]] &&
      !states$H0[[4]] && !states$H1[[4]]) {
    contexts <- tryCatch({
      data <- prepare_inference_data(fit_H1)
      list(H0 = prepare_inference(fit_H0, data), H1 = prepare_inference(fit_H1, data))
    }, error = function(e) list())
  }
  convention_nested_impl(contexts$H0$native, contexts$H1$native, convention,
                         states$H0, states$H1)
}
