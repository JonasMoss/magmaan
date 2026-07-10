#' Fit a continuous-data measurement-invariance ladder.
#'
#' The scalar rung explicitly frees every non-reference-group latent mean.
#' This is required when observed intercepts are constrained equal: the scalar
#' model is not an affine parameter subset of the metric model, so its robust
#' difference test uses the delta restriction map.
#'
#' @param model A lavaan-style syntax string or a `magmaan_model_spec` carrying
#'   its source syntax.
#' @param data A data frame containing the grouping column and continuous
#'   observed variables.
#' @param group Name of the grouping column. When `model` is a grouped model
#'   spec, its grouping column is used by default.
#' @param steps Requested rungs from `"configural"`, `"metric"`, and
#'   `"scalar"`. Requesting `"scalar"` also fits its metric prerequisite.
#' @param estimator `"ML"` for complete data or `"FIML"` when incomplete data
#'   require full-information estimation.
#' @param group_labels Optional group ordering. Defaults to data order (or the
#'   supplied model spec's ordering).
#' @param optimizer,control,bounds Fitting controls forwarded to [magmaan()].
#'
#' @return A `magmaan_continuous_invariance` list with fitted models, robust
#'   nested tests, and the constructed specifications.
#' @export
continuous_invariance <- function(model, data, group = NULL,
                                  steps = c("configural", "metric", "scalar"),
                                  estimator = c("ML", "FIML"),
                                  group_labels = NULL,
                                  optimizer = "nlopt-lbfgs",
                                  control = NULL,
                                  bounds = NULL) {
  estimator <- match.arg(estimator)
  steps <- unique(match.arg(steps, c("configural", "metric", "scalar"),
                            several.ok = TRUE))
  if (!length(steps)) {
    stop("continuous_invariance(): `steps` must include at least one rung",
         call. = FALSE)
  }
  if (!is.data.frame(data)) {
    stop("continuous_invariance(): `data` must be a data frame", call. = FALSE)
  }

  spec_options <- list()
  spec_group <- ""
  spec_labels <- NULL
  if (inherits(model, "magmaan_model_spec")) {
    if (is.null(model$syntax)) {
      stop("continuous_invariance(): model specs must carry source syntax",
           call. = FALSE)
    }
    if (length(model$ordered %||% character())) {
      stop("continuous_invariance(): categorical indicators belong in ",
           "mplus_wlsmv_invariance()", call. = FALSE)
    }
    syntax <- model$syntax
    spec_options <- model$options %||% list()
    spec_group <- model$group_var %||% ""
    spec_labels <- model$group_labels
  } else if (is.character(model) && length(model) == 1L) {
    syntax <- model
  } else {
    stop("continuous_invariance(): `model` must be a syntax string or ",
         "magmaan_model_spec", call. = FALSE)
  }

  if (is.null(group)) group <- spec_group
  group <- as.character(group %||% "")[1L]
  if (!nzchar(group)) {
    stop("continuous_invariance(): `group` must name the grouping column",
         call. = FALSE)
  }
  if (!group %in% names(data)) {
    stop("continuous_invariance(): grouping column not found: ", group,
         call. = FALSE)
  }
  g <- data[[group]]
  if (anyNA(g)) {
    stop("continuous_invariance(): grouping column contains missing values",
         call. = FALSE)
  }
  if (is.null(group_labels)) {
    group_labels <- spec_labels %||%
      if (is.factor(g)) levels(g) else unique(as.character(g))
  }
  group_labels <- as.character(group_labels)
  if (length(group_labels) < 2L) {
    stop("continuous_invariance(): at least two group labels are required",
         call. = FALSE)
  }
  if (!setequal(unique(as.character(g)), group_labels)) {
    stop("continuous_invariance(): `group_labels` must match the observed ",
         "group values", call. = FALSE)
  }

  if (length(spec_options$group_partial %||% character())) {
    stop("continuous_invariance(): partial-invariance specifications are not ",
         "yet supported", call. = FALSE)
  }
  option_names <- c("auto_var", "auto_cov_lv_x", "auto_cov_y", "orthogonal",
                    "auto_fix_first", "auto_fix_single", "std_lv",
                    "effect_coding", "fixed_x", "model_type")
  build_options <- spec_options[intersect(names(spec_options), option_names)]
  make_spec <- function(extra = "", group_equal = NULL) {
    txt <- syntax
    if (nzchar(extra)) txt <- paste(txt, extra, sep = "\n")
    do.call(
      model_spec,
      c(list(syntax = txt), build_options,
        list(meanstructure = TRUE, group = group, group_labels = group_labels,
             group_equal = group_equal)))
  }

  configural <- make_spec()
  rep <- model_matrix_rep(configural$partable)
  ov_by_group <- rep$ov_names
  if (!is.list(ov_by_group)) ov_by_group <- list(ov_by_group)
  observed <- unique(unlist(ov_by_group, use.names = FALSE))
  if (identical(estimator, "ML") && anyNA(data[, observed, drop = FALSE])) {
    stop("continuous_invariance(): estimator = \"ML\" requires complete ",
         "observed data; use estimator = \"FIML\" for incomplete data",
         call. = FALSE)
  }
  lv <- unique(as.character(configural$partable$lhs[
    configural$partable$op == "=~"]))
  if (!length(lv)) {
    stop("continuous_invariance(): could not find latent variables in `model`",
         call. = FALSE)
  }
  group_free <- paste(c("0", rep("NA", length(group_labels) - 1L)),
                      collapse = ", ")
  scalar_extra <- paste0(lv, " ~ c(", group_free, ")*1", collapse = "\n")
  specs <- list(
    configural = configural,
    metric = make_spec(group_equal = "loadings"),
    scalar = make_spec(extra = scalar_extra,
                       group_equal = c("loadings", "intercepts")))

  requested <- "configural"
  if (any(steps %in% c("metric", "scalar"))) requested <- c(requested, "metric")
  if ("scalar" %in% steps) requested <- c(requested, "scalar")
  fit_one <- function(spec) {
    magmaan(spec, data, estimator = estimator, optimizer = optimizer,
            control = control, bounds = bounds)
  }
  fits <- lapply(specs[requested], fit_one)

  tests <- list()
  add_test <- function(step, h1, h0, a_method) {
    raw <- if (identical(estimator, "ML")) raw_data_arg(fits[[h1]], data)$X else NULL
    nt <- nestedTest(fits[[h1]], fits[[h0]], data = raw,
                     method = "restriction_map", A.method = a_method)
    tests[[step]] <<- data.frame(
      step = step, h1 = h1, h0 = h0, A.method = a_method,
      T_diff = nt$T_diff %||% NA_real_,
      df_diff = nt$df_diff %||% NA_integer_,
      p_unscaled = nt$p_unscaled %||% NA_real_,
      scale_c = nt$scale_c %||% NA_real_,
      T_scaled = nt$T_scaled %||% NA_real_,
      p_scaled = nt$p_scaled %||% NA_real_,
      T_adjusted = nt$T_adjusted %||% NA_real_,
      p_adjusted = nt$p_adjusted %||% NA_real_,
      p_mixture = nt$p_mixture %||% NA_real_,
      stringsAsFactors = FALSE)
  }
  if ("metric" %in% requested) add_test("metric", "configural", "metric", "exact")
  if ("scalar" %in% requested) add_test("scalar", "metric", "scalar", "delta")
  test_table <- if (length(tests)) do.call(rbind, tests) else data.frame()
  rownames(test_table) <- NULL

  out <- list(
    fits = fits,
    tests = test_table,
    specs = specs[requested],
    syntax = lapply(specs[requested], `[[`, "syntax"),
    options = list(estimator = estimator, group = group,
                   group_labels = group_labels, steps = requested))
  class(out) <- c("magmaan_continuous_invariance", "list")
  out
}

#' @export
print.magmaan_continuous_invariance <- function(x, ...) {
  cat("magmaan continuous-data invariance\n")
  cat("  estimator: ", x$options$estimator, "\n", sep = "")
  cat("  groups: ", paste(x$options$group_labels, collapse = ", "), "\n",
      sep = "")
  if (nrow(x$tests)) print(x$tests)
  invisible(x)
}
