#' Construct a model from EQS equations
#'
#' Parses explicit single-group continuous covariance-model sections. Estimation
#' and inference use magmaan; this does not run EQS or import its job settings.
#'
#' @param syntax One string containing `/EQUATIONS`, `/VARIANCES` and optionally
#'   `/COVARIANCES`. Unambiguous section abbreviations with at least three
#'   letters, `!` comments, ranges and an optional `/END` are accepted.
#' @param observed_names Optional names in EQS data-column order: `V1` maps to
#'   the first name, `V2` to the second, and so on. Without names, data columns
#'   should be named `V1`, `V2`, etc.
#' @return A `magmaan_model_spec` usable in `fit_model()` and `prepare_model()`.
#'   Original EQS text is retained as `eqs_source`;
#'   `syntax` contains the equivalent explicit lavaan specification for refits.
#' @export
eqs_model <- function(syntax, observed_names = NULL) {
  if (!is.character(syntax) || length(syntax) != 1L || is.na(syntax)) {
    stop("eqs_model(): `syntax` must be one non-missing string", call. = FALSE)
  }
  if (!is.null(observed_names) &&
      (!is.character(observed_names) || anyNA(observed_names) ||
       any(!nzchar(observed_names)))) {
    stop("eqs_model(): `observed_names` must be non-missing column names", call. = FALSE)
  }
  parsed <- eqs_model_impl(syntax, observed_names)
  out <- as_magmaan_model_spec(parsed$partable)
  out$syntax <- parsed$syntax
  out$eqs_source <- syntax
  out$requested_meanstructure <- FALSE
  out$options <- list(
    auto_var = FALSE, auto_cov_lv_x = FALSE, auto_cov_y = FALSE,
    auto_fix_first = FALSE, auto_fix_single = FALSE, fixed_x = FALSE,
    meanstructure = FALSE
  )
  out
}
