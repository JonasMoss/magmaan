# The ordinary-user entry point. This file handles arguments only: estimates
# come from magmaanlab::fit_model() and inference from infer(). The option set
# and the policy are specified in project/design/r-interface-vision.md.

.continuous_estimators <- c("ML", "FIML", "ML2S", "GLS", "ULS", "WLS")
.ordered_estimators <- c("DWLS", "WLS", "ULS")

# lavaan names that bundle an estimator with a standard-error or test correction,
# mapped to the plain estimator magmaan() accepts instead.
.bundled_estimators <- c(
  MLM = "ML", MLMV = "ML", MLMVS = "ML", MLR = "ML", MLF = "ML",
  WLSM = "DWLS", WLSMV = "DWLS", WLSMVS = "DWLS",
  ULSM = "ULS", ULSMV = "ULS", ULSMVS = "ULS"
)

#' Fit a structural equation model with magmaan's default inference
#'
#' Estimates the model and, unless `inference = FALSE`, computes inference
#' under magmaan's single documented policy. Option names follow lavaan where
#' the concept is the same.
#'
#' @param model lavaan model syntax, one string.
#' @param data A data frame of raw observations.
#' @param estimator `"ML"`, `"FIML"`, `"ML2S"`, `"GLS"` or `"ULS"` for
#'   continuous variables; `"DWLS"`, `"WLS"` or `"ULS"` for variables declared
#'   in `ordered`.
#' @param ordered Names of the ordered (categorical) variables.
#' @param group Name of the grouping column.
#' @param group.equal,group.partial Cross-group equality constraints, as in
#'   lavaan.
#' @param cluster Name of the cluster column for two-level ML.
#' @param identification `"marker"` (first loading fixed to one) or `"std.lv"`
#'   (latent variances fixed to one).
#' @param parameterization `"delta"` or `"theta"`, for ordered variables.
#' @param meanstructure `"default"`, `TRUE` or `FALSE`.
#' @param fixed.x Treat exogenous observed covariates as fixed.
#' @param missing How rows with missing values are deleted: `"listwise"`
#'   (default) or `"pairwise"` (ordered variables only). Use `estimator =
#'   "FIML"` or `"ML2S"` to use incomplete rows.
#' @param psd Constrain the model-implied covariance matrices to be positive
#'   semidefinite.
#' @param inference Compute inference now. With `FALSE`, call [infer()] later.
#' @return An object of class `magmaan`.
#' @export
magmaan <- function(model, data,
                    estimator = "ML",
                    ordered = NULL,
                    group = NULL, group.equal = NULL, group.partial = NULL,
                    cluster = NULL,
                    identification = "marker",
                    parameterization = "delta",
                    meanstructure = "default", fixed.x = TRUE,
                    missing = "listwise",
                    psd = FALSE,
                    inference = TRUE) {
  if (!is.character(model) || length(model) != 1L || is.na(model)) {
    stop("magmaan(): `model` must be lavaan model syntax in one string", call. = FALSE)
  }
  if (!is.data.frame(data)) {
    stop("magmaan(): `data` must be a data frame of raw observations; ",
         "summary-statistic input is available in magmaanlab", call. = FALSE)
  }
  estimator <- .check_estimator(estimator)
  .check_flag(psd, "psd")
  .check_flag(inference, "inference")
  .check_flag(fixed.x, "fixed.x")
  identification <- .check_choice(identification, "identification",
                                  c("marker", "std.lv"), planned = "sphere")
  parameterization <- .check_choice(parameterization, "parameterization",
                                    c("delta", "theta"))
  missing <- .check_choice(missing, "missing", c("listwise", "pairwise"))
  ordered <- .check_ordered(ordered)
  .check_estimator_data(estimator, ordered, missing)
  group <- .check_column(group, "group", data)
  cluster <- .check_column(cluster, "cluster", data)
  if (!identical(meanstructure, "default") &&
      !(is.logical(meanstructure) && length(meanstructure) == 1L && !is.na(meanstructure))) {
    stop("magmaan(): `meanstructure` must be \"default\", TRUE or FALSE", call. = FALSE)
  }

  model_options <- list(fixed_x = fixed.x)
  if (!identical(meanstructure, "default")) model_options$meanstructure <- meanstructure
  if (identical(identification, "std.lv")) model_options$std_lv <- TRUE
  if (!is.null(group.equal)) model_options$group_equal <- group.equal
  if (!is.null(group.partial)) model_options$group_partial <- group.partial

  lab <- do.call(magmaanlab::fit_model, c(
    list(model = model, data = data, estimator = estimator, groups = group,
         cluster = cluster, ordered = ordered, parameterization = parameterization,
         missing = if (estimator %in% c("FIML", "ML2S")) "listwise" else missing,
         psd = psd),
    model_options))

  fit <- structure(
    list(lab = lab, call = match.call(), estimator = estimator, psd = psd,
         missing = if (estimator %in% c("FIML", "ML2S")) "fiml" else missing,
         rows = .row_accounting(data, group, lab),
         inference = NULL),
    class = "magmaan")
  if (inference) fit <- infer(fit)
  fit
}

#' The magmaanlab fit behind a magmaan fit
#'
#' Use it to run any alternative estimator option or inference method from
#' magmaanlab without refitting.
#'
#' @param fit A [magmaan()] fit.
#' @return The underlying `magmaan_fit` from magmaanlab.
#' @export
as_lab_fit <- function(fit) {
  if (!inherits(fit, "magmaan")) stop("as_lab_fit(): supply a magmaan() fit", call. = FALSE)
  fit$lab
}

.check_estimator <- function(estimator) {
  if (!is.character(estimator) || length(estimator) != 1L || is.na(estimator)) {
    stop("magmaan(): `estimator` must be one string", call. = FALSE)
  }
  est <- toupper(estimator)
  if (est %in% names(.bundled_estimators)) {
    stop(sprintf(paste0(
      "magmaan(): estimator = \"%s\" bundles an estimator with a correction. ",
      "Use estimator = \"%s\"; magmaan() computes inference automatically. ",
      "Other corrections are available in magmaanlab."),
      estimator, .bundled_estimators[[est]]), call. = FALSE)
  }
  known <- union(.continuous_estimators, .ordered_estimators)
  if (!est %in% known) {
    stop(sprintf("magmaan(): unknown estimator \"%s\"; use one of %s",
                 estimator, paste(known, collapse = ", ")), call. = FALSE)
  }
  est
}

.check_estimator_data <- function(estimator, ordered, missing) {
  has_ordered <- length(ordered) > 0L
  if (has_ordered && !estimator %in% .ordered_estimators) {
    stop(sprintf(paste0(
      "magmaan(): estimator = \"%s\" treats every variable as continuous. ",
      "Drop `ordered` to fit it to these items as continuous, or use DWLS, ",
      "WLS or ULS for ordered variables."), estimator), call. = FALSE)
  }
  if (!has_ordered && identical(estimator, "DWLS")) {
    stop("magmaan(): DWLS is for ordered variables; declare them with `ordered =`",
         call. = FALSE)
  }
  if (!has_ordered && identical(estimator, "WLS")) {
    stop("magmaan(): continuous WLS (ADF) is not available in magmaan() yet; use ",
         "magmaanlab::estimate() with prepare_weight(data, \"WLS\")", call. = FALSE)
  }
  if (identical(missing, "pairwise") && !has_ordered) {
    stop("magmaan(): missing = \"pairwise\" is for ordered variables; continuous ",
         "data are deleted listwise, or use estimator = \"FIML\"", call. = FALSE)
  }
  invisible(NULL)
}

.check_ordered <- function(ordered) {
  if (is.null(ordered) || identical(ordered, FALSE)) return(NULL)
  if (isTRUE(ordered)) {
    stop("magmaan(): name the ordered variables, e.g. ordered = c(\"y1\", \"y2\")",
         call. = FALSE)
  }
  if (!is.character(ordered) || anyNA(ordered) || !all(nzchar(ordered))) {
    stop("magmaan(): `ordered` must be a character vector of variable names", call. = FALSE)
  }
  unique(ordered)
}

.check_column <- function(x, arg, data) {
  if (is.null(x)) return(NULL)
  if (!is.character(x) || length(x) != 1L || is.na(x) || !nzchar(x)) {
    stop(sprintf("magmaan(): `%s` must be one column name", arg), call. = FALSE)
  }
  if (!x %in% names(data)) {
    stop(sprintf("magmaan(): `%s` column \"%s\" is not in `data`", arg, x), call. = FALSE)
  }
  x
}

.check_flag <- function(x, arg) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("magmaan(): `%s` must be TRUE or FALSE", arg), call. = FALSE)
  }
  invisible(x)
}

.check_choice <- function(x, arg, choices, planned = character()) {
  if (!is.character(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("magmaan(): `%s` must be one string", arg), call. = FALSE)
  }
  if (x %in% planned) {
    stop(sprintf("magmaan(): %s = \"%s\" is planned but not available yet", arg, x),
         call. = FALSE)
  }
  if (!x %in% choices) {
    stop(sprintf("magmaan(): `%s` must be one of %s", arg,
                 paste0("\"", choices, "\"", collapse = ", ")), call. = FALSE)
  }
  x
}

# Rows supplied, used and not used, per group. Estimators that are not designed
# for missing data delete listwise; FIML and ML2S use every row with data.
.row_accounting <- function(data, group, lab) {
  used <- as.integer(lab$nobs)
  labels <- as.character(lab$group_labels)
  if (is.null(group) || !length(labels)) {
    total <- nrow(data)
    labels <- "all"
  } else {
    total <- as.integer(table(factor(as.character(data[[group]]), levels = labels)))
  }
  if (length(used) != length(total)) used <- rep(NA_integer_, length(total))
  data.frame(group = labels, rows = total, used = used, deleted = total - used,
             stringsAsFactors = FALSE)
}
