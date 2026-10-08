# Advanced fitting choices are resolved by the shared C++ engine. This helper
# only validates R shapes and reconciles the older lab argument locations.
.fitting_control <- function(options, control = NULL, optimizer = NULL) {
  if (is.null(options)) return(control)
  if (!is.list(options) || (length(options) && (is.null(names(options)) ||
      anyNA(names(options)) || any(!nzchar(names(options))) || anyDuplicated(names(options)))))
    stop("options must be a uniquely named list")
  allowed <- c("preset", "starts", "optimizer", "convergence", "marker")
  if (any(!names(options) %in% allowed)) stop("unknown fitting option: ",
      paste(setdiff(names(options), allowed), collapse = ", "))
  for (name in names(options)) {
    value <- options[[name]]
    if (!is.character(value) || length(value) != 1L || is.na(value) || !nzchar(value))
      stop("options$", name, " must be one nonmissing string")
  }
  if (!is.null(optimizer)) {
    if (!is.null(options$optimizer) || !is.null(options$preset))
      stop("optimizer conflicts with options; select it in options$optimizer")
    options$optimizer <- optimizer
  }
  if (!is.null(control) && (!is.list(control) ||
      (length(control) && (is.null(names(control)) || anyDuplicated(names(control))))))
    stop("control must be a named list")
  if (any(!names(control) %in% "start"))
    stop("versioned fitting options cannot be combined with solver controls")
  if (is.character(control$start)) {
    if (!is.null(options$starts) || !is.null(options$preset))
      stop("start constructor conflicts with options; select it in options$starts")
    options$starts <- control$start
    control$start <- NULL
  }
  control$fitting_options <- options
  control
}

# Arguments of fit_model() that describe the model rather than how it is fitted.
# A refit supplies its own model, so these are never replayed from a route.
.model_structure_args <- c("groups", "ordered", "parameterization")

# Arguments that refit a fit_model() fit. Every other recorded argument is
# replayed, so fitting options and arguments added later carry over without
# changes here; `overrides` replace recorded values. Fits without a
# fit_model() route refit with their estimator. A numeric start vector is
# positional and does not survive a model with different parameters.
.refit_args <- function(fit, overrides = list(), model_changed = TRUE) {
  route <- fit$options$route
  args <- if (identical(route$fitter, "fit_model")) route$args else
    c(list(estimator = toupper(fit$estimator %||% "ML")),
      if (isTRUE(fit$options$psd)) list(psd = TRUE))
  args <- args[setdiff(names(args), .model_structure_args)]
  if (model_changed && is.numeric(args$control$start)) {
    args$control$start <- NULL
    if (!length(args$control)) args$control <- NULL
  }
  args[names(overrides)] <- overrides
  args
}
