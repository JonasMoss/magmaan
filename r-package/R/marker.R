# The rule and convergence verdict are native. These helpers only reconstruct
# syntax and replay the same fitter, including all of its retry attempts.
.marker_disabled <- function(options) {
  attr(options, "magmaan.marker_evaluated") <- TRUE
  options
}
.marker_requested <- function(options) {
  identical(options$marker %||% options$preset, "lavaan-0.7.2")
}
.marker_fit <- function(model, data, options, run) {
  if (is.null(options) || isTRUE(attr(options, "magmaan.marker_evaluated"))) return(NULL)
  options <- .fitting_control(options)$fitting_options
  result <- .marker_boundary(function() prepared_marker_adapt_impl(model$native, data$native, options,
    model$input_spec$syntax))
  if (!length(result$marker)) return(NULL)
  spec <- model$input_spec
  if (is.null(spec$syntax) || !is.null(spec$mplus_source) || !is.null(spec$eqs_source))
    stop(structure(list(message = paste0("unsupported_model: marker adaptation requires lavaan source syntax; ",
      'use options$marker = "default"'), call = NULL),
      class = c("magmaan_unsupported_model", "error", "condition")))
  switched <- .rebuild_model_spec(spec, overrides = list(marker = result$marker), caller = "marker adaptation")
  candidate <- .marker_run(function() run(.marker_prepare(switched, model), .marker_disabled(options)))
  selected <- .marker_select(candidate, function() .marker_run(function() run(model, .marker_disabled(options))), result$info)
  selected$requested_model <- spec
  selected$marker_spec <- selected$model %||%
    if (any(selected$fitting$marker_switch$reverted)) spec else switched
  selected$model <- selected$marker_spec
  selected$syntax <- selected$model$syntax
  # A refit must evaluate the rule again on its own sample.
  if (!is.null(selected$options$route$args$options))
    attr(selected$options$route$args$options, "magmaan.marker_evaluated") <- NULL
  if (!is.null(selected$options$route$args$control$fitting_options))
    attr(selected$options$route$args$control$fitting_options, "magmaan.marker_evaluated") <- NULL
  warnings <- attr(selected, "magmaan.marker_warnings")
  attr(selected, "magmaan.marker_warnings") <- NULL
  for (w in warnings) warning(w, call. = FALSE)
  selected
}
.marker_select <- function(candidate, original, info) {
  if (!isTRUE(candidate$converged)) {
    candidate <- original()
    info$reverted <- TRUE
  }
  candidate$fitting$marker_switch <- info
  candidate
}
.marker_fit_spec <- function(spec, data, options, run, kind = NULL) {
  if (is.null(options) || isTRUE(attr(options, "magmaan.marker_evaluated"))) return(NULL)
  if (!.marker_requested(options)) return(NULL)
  categorical <- inherits(data, "magmaan_ordinal_data") || inherits(data, "magmaan_mixed_ordinal_data")
  if (length(spec$ordered) && any(spec$partable$exo != 0L))
    .marker_boundary(function() stop("unsupported_model: marker adaptation lacks conditional.x residual H1 covariance"))
  if (categorical) {
    # The rule needs only the supplied H1 moments. The replay still uses the
    # original categorical specification and the same first-stage statistics.
    moments <- spec
    moments$ordered <- character()
    metadata <- as.list.environment(prepare_model(moments))
    metadata$input_spec <- spec
    metadata$categories <- setNames(lapply(spec$ordered, function(v) {
      j <- match(v, data$ov_names[[1L]])
      as.character(seq_len(data$n_levels[[1L]][j]))
    }), spec$ordered)
    model <- do.call(.prepared_object, c(metadata, list(class = "magmaan_prepared_model")))
    prepared <- prepare_data(model, list(S = data$R, nobs = data$nobs))
  } else if (identical(kind, "raw") && !is.data.frame(data) && !is.matrix(data)) {
    model <- prepare_model(spec)
    prepared <- .prepared_object(native = prepared_data_impl(model$native,
      fiml_data_arg(data), "raw", model$masks), class = "magmaan_prepared_data")
  } else {
    model <- prepare_model(spec, prototype = if (length(spec$ordered)) data else NULL)
    prepared <- prepare_data(model, if (is.data.frame(data)) data else if (identical(kind, "raw")) data else sample_stats_arg(data), kind = kind, missing = "listwise")
  }
  .marker_fit(model, prepared, options, function(m, o) run(m$input_spec, o))
}

.marker_refusal <- function(..., caller) {
  fits <- list(...)
  if (any(vapply(fits, function(f) {
    info <- f$fitting$marker_switch
    !is.null(info) && nrow(info) > 0L
  }, logical(1)))) stop(structure(list(message = paste0(caller,
    ": unsupported_model: contrasts across adapted marker coordinates are not yet supported"), call = NULL),
    class = c("magmaan_unsupported_model", "error", "condition")))
}

.marker_run <- function(run) {
  warnings <- character()
  result <- withCallingHandlers(run(), warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  attr(result, "magmaan.marker_warnings") <- warnings
  result
}

# Reconstruct categorical schema from the original handle, without borrowing
# empirical thresholds or starts.
.marker_prepare <- function(spec, original) {
  prototype <- if (length(original$categories)) as.data.frame(lapply(original$categories,
    function(levels) factor(character(), levels = levels, ordered = TRUE))) else NULL
  prepare_model(spec, prototype = prototype)
}

.marker_boundary <- function(run) {
  tryCatch(run(), error = function(e) {
    if (grepl("unsupported_model", conditionMessage(e), fixed = TRUE))
      stop(structure(list(message = conditionMessage(e), call = NULL),
        class = c("magmaan_unsupported_model", "error", "condition")))
    stop(e)
  })
}
