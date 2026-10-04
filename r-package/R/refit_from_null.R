#' Refit a larger model from a restricted estimate
#'
#' Embeds the restricted estimate in the larger model's coordinates using the
#' native, moment-verified nesting map, then replays the larger fit's estimator
#' and fitting options with that explicit start. Neither input is modified.
#' The returned fit carries its own convergence diagnostics and native verdict.
#' @param fit_H1 The larger fitted model.
#' @param fit_H0 The restricted fitted model, using the same data and estimator.
#' @return A new estimate-only `magmaan_fit`.
#' @export
refit_from_null <- function(fit_H1, fit_H0) {
  if (!inherits(fit_H1, "magmaan_fit") || !inherits(fit_H0, "magmaan_fit"))
    stop("refit_from_null(): supply two fitted magmaan models")
  if (!identical(fit_H1$estimator, fit_H0$estimator))
    stop("refit_from_null(): fits must use the same estimator")
  ordinal <- isTRUE(fit_H1$ordinal)
  if (ordinal != isTRUE(fit_H0$ordinal) ||
      !identical(fit_H1$parameterization, fit_H0$parameterization))
    stop("refit_from_null(): model parameterizations differ")
  if (!toupper(fit_H1$estimator) %in% c("ML", "FIML", "DWLS", "ULS", "WLS") ||
      !is.null(fit_H1$nclusters) || !is.null(fit_H0$nclusters))
    stop("refit_from_null(): unsupported estimator or multilevel model")
  route <- fit_H1$options$route
  if (!identical(route$fitter, "fit_model"))
    stop("refit_from_null(): requires a fit_model() or estimate() fit")
  same <- if (ordinal) identical(fit_H1$ordinal_stats, fit_H0$ordinal_stats) else
    identical(fit_H1$raw_data, fit_H0$raw_data) &&
    identical(fit_H1$S, fit_H0$S) && identical(fit_H1$nobs, fit_H0$nobs) &&
    identical(fit_H1$sample_mean, fit_H0$sample_mean)
  if (!same) stop("refit_from_null(): fits must use the same observations in the same order")
  start <- nested_null_start_impl(fit_H1, fit_H0)
  args <- .refit_args(fit_H1, model_changed = FALSE)
  args$control$start <- start
  args$se <- "none"
  args$test <- "none"
  data <- if (ordinal) {
    stats <- fit_H1$ordinal_stats
    stats$ov_names <- model_matrix_rep(fit_H1$partable)$ov_names
    stats$ordered <- fit_H1$ordered
    stats$group_var <- fit_H1$group_var
    stats$group_labels <- fit_H1$group_labels
    structure(stats, class = c("magmaan_ordinal_data", "list"))
  } else if (toupper(fit_H1$estimator) == "FIML") {
    if (is.null(fit_H1$raw_data)) stop("refit_from_null(): FIML requires retained observations")
    fit_H1$raw_data
  } else list(S = fit_H1$S, mean = fit_H1$sample_mean, nobs = fit_H1$nobs)
  out <- do.call(fit_model, c(list(model = fit_H1$model, data = data), args))
  # Complete-data moment refitting uses exactly the retained sufficient
  # statistics; retain the same rows for subsequent casewise inference.
  if (!ordinal && toupper(fit_H1$estimator) != "FIML") out$raw_data <- fit_H1$raw_data
  if (ordinal) out$ordinal_stats <- fit_H1$ordinal_stats
  out
}
