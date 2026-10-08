#' Policy modification indices and equality releases
#'
#' Uses observed sensitivity and empirical scores for ML and FIML, and exact
#' first-stage sampling rows with estimated weights for ordinal or mixed DWLS.
#' Each robust score statistic has a chi-square(1) limiting reference law.
#' @param fit A fitted magmaanlab model.
#' @param data Optional raw observations; defaults to the retained fitting data.
#' @param releases Include univariate equality-release tests.
#' @return A data frame in candidate order, with typed unavailability reasons.
#' @export
policy_modification_indices <- function(fit, data = NULL, releases = TRUE) {
  if (!inherits(fit, "magmaan_fit")) stop("policy_modification_indices(): supply a fitted magmaan model")
  if (!is.logical(releases) || length(releases) != 1L || is.na(releases)) stop("releases must be TRUE or FALSE")
  state <- .policy_state(fit)
  if (state[1] && !state[4] && (length(fit$diagnostics$active_bounds_lower) || length(fit$diagnostics$active_bounds_upper))) {
    out <- data.frame(kind = "fixed", lhs = "", op = "", rhs = "", group = 0L,
      test = "score", statistic = NA_real_, df = 1L, pvalue = NA_real_, epc = NA_real_,
      sepc.lv = NA_real_, sepc.all = NA_real_, reason = "unsupported_model")
    attr(out, "detail") <- "active-bound inference is unsupported"
    return(out)
  }
  raw <- if (is.null(data)) NULL else raw_data_arg(fit, data)
  policy_modification_indices_impl(fit, state, raw, releases)
}

#' Refit one modification-index candidate from the embedded null
#'
#' Frees one candidate (or releases one equality) of a policy
#' modification-index table and refits from the fitted estimate embedded in
#' the augmented model, replaying the fit's options. The template is an
#' evaluation point for the native embedding map; it is never optimized.
#' @param fit The fitted (restricted) magmaanlab model.
#' @param row One row of [policy_modification_indices()] output.
#' @param candidate_row The matching element of that output's
#'   `candidate_row` attribute.
#' @return The fitted augmented model, for [policy_nested()].
#' @export
policy_mi_refit <- function(fit, row, candidate_row) {
  .marker_refusal(fit, caller = "policy_mi_refit()")
  pt <- policy_mi_alternative_impl(fit, row$kind, candidate_row,
      row$lhs, row$op, row$rhs, row$group)
  spec <- as_magmaan_model_spec(pt)
  spec$ordered <- fit$model$ordered
  spec$parameterization <- fit$parameterization %||% "delta"
  spec$options <- fit$model$options
  template <- fit
  template$partable <- pt
  template$theta <- c(fit$theta, if (row$kind == "fixed") 0)
  start <- nested_null_start_impl(template, fit)
  args <- .refit_args(fit)
  args$control$start <- start
  args$se <- "none"
  args$test <- "none"
  ordinal <- isTRUE(fit$ordinal) || isTRUE(fit$mixed_ordinal)
  data <- if (isTRUE(fit$mixed_ordinal)) {
    stats <- fit$mixed_ordinal_stats
    stats$ov_names <- model_matrix_rep(pt)$ov_names
    stats$ordered <- fit$ordered
    stats$group_var <- fit$group_var
    stats$group_labels <- fit$group_labels
    structure(stats, class = c("magmaan_mixed_ordinal_data", "list"))
  } else if (ordinal) {
    stats <- fit$ordinal_stats
    stats$ov_names <- model_matrix_rep(pt)$ov_names
    stats$ordered <- fit$ordered
    stats$group_var <- fit$group_var
    stats$group_labels <- fit$group_labels
    structure(stats, class = c("magmaan_ordinal_data", "list"))
  } else if (toupper(fit$estimator) == "FIML") fit$raw_data else
    list(S = fit$S, mean = fit$sample_mean, nobs = fit$nobs)
  out <- do.call(fit_model, c(list(model = spec, data = data), args))
  out$raw_data <- fit$raw_data
  if (isTRUE(fit$ordinal)) out$ordinal_stats <- fit$ordinal_stats
  if (isTRUE(fit$mixed_ordinal)) out$mixed_ordinal_stats <- fit$mixed_ordinal_stats
  out
}
