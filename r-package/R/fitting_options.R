# Advanced fitting choices are resolved by the shared C++ engine. This helper
# only validates R shapes and reconciles the older lab argument locations.
.fitting_control <- function(options, control = NULL, optimizer = NULL) {
  if (is.null(options)) return(control)
  if (!is.list(options) || (length(options) && (is.null(names(options)) ||
      anyNA(names(options)) || any(!nzchar(names(options))) || anyDuplicated(names(options)))))
    stop("options must be a uniquely named list")
  allowed <- c("preset", "starts", "optimizer", "convergence")
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
