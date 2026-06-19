mplus_wlsmv_invariance <- function(model, data, ordered, group,
                                   steps = c("configural", "metric", "scalar"),
                                   estimator = c("WLSMV", "DWLS", "WLS"),
                                   missing = c("listwise", "pairwise", "error"),
                                   group_labels = NULL,
                                   optimizer = "nlopt-lbfgs",
                                   control = NULL,
                                   bounds = NULL,
                                   pd_gamma = c("overlap", "nominal"),
                                   validate = TRUE) {
  estimator <- match.arg(estimator)
  missing <- match.arg(missing)
  pd_gamma <- match.arg(pd_gamma)
  steps <- unique(match.arg(steps, c("configural", "metric", "scalar"),
                            several.ok = TRUE))
  if (!length(steps)) {
    stop("mplus_wlsmv_invariance(): `steps` must include at least one rung",
         call. = FALSE)
  }
  fit_estimator <- if (identical(estimator, "WLS")) "WLS" else "DWLS"
  if (missing(group) || is.null(group) || !nzchar(as.character(group)[1L])) {
    stop("mplus_wlsmv_invariance(): `group` must name the grouping column",
         call. = FALSE)
  }
  group <- as.character(group)[1L]

  if (inherits(model, "magmaan_model_spec")) {
    if (is.null(model$syntax)) {
      stop("mplus_wlsmv_invariance(): model specs must carry source syntax",
           call. = FALSE)
    }
    syntax <- model$syntax
    ordered <- if (missing(ordered) || is.null(ordered)) model$ordered else ordered
  } else if (is.character(model) && length(model) == 1L) {
    syntax <- model
  } else {
    stop("mplus_wlsmv_invariance(): `model` must be a syntax string or ",
         "magmaan_model_spec", call. = FALSE)
  }
  if (missing(ordered) || is.null(ordered)) ordered <- character()
  ordered <- as.character(ordered)
  if (!length(ordered)) {
    stop("mplus_wlsmv_invariance(): `ordered` must name categorical indicators",
         call. = FALSE)
  }

  if (is.null(group_labels) && is.data.frame(data)) {
    if (!group %in% names(data)) {
      stop("mplus_wlsmv_invariance(): grouping column not found: ", group,
           call. = FALSE)
    }
    g <- data[[group]]
    if (anyNA(g)) {
      stop("mplus_wlsmv_invariance(): grouping column contains missing values",
           call. = FALSE)
    }
    group_labels <- if (is.factor(g)) levels(g) else unique(as.character(g))
  }
  group_labels <- as.character(group_labels %||% character())
  if (length(group_labels) < 2L) {
    stop("mplus_wlsmv_invariance(): at least two group labels are required",
         call. = FALSE)
  }

  make_spec <- function(extra = "", group_equal = NULL) {
    txt <- syntax
    if (nzchar(extra)) txt <- paste(txt, extra, sep = "\n")
    model_spec(
      txt,
      meanstructure = TRUE,
      ordered = ordered,
      group = group,
      group_labels = group_labels,
      parameterization = "delta",
      group_equal = group_equal)
  }

  configural <- make_spec()
  rep <- model_matrix_rep(configural$partable)
  ov_by_group <- rep$ov_names
  if (!is.list(ov_by_group)) ov_by_group <- list(ov_by_group)
  all_ordinal <- all(vapply(ov_by_group, function(ov) {
    setequal(ordered, ov)
  }, logical(1)))
  if (!all_ordinal && identical(missing, "pairwise")) {
    stop("mplus_wlsmv_invariance(): missing = \"pairwise\" is currently ",
         "available for all-ordinal models only; mixed ordinal models use ",
         "listwise/error missing handling.", call. = FALSE)
  }

  data_obj <- data
  if (is.data.frame(data_obj)) {
    data_obj <- if (all_ordinal) {
      data_ordinal_stats_from_df(data_obj, configural, ordered = ordered,
                                 group = group, missing = missing,
                                 pd_gamma = pd_gamma)
    } else {
      data_mixed_ordinal_stats_from_df(data_obj, configural, ordered = ordered,
                                       group = group, missing = missing)
    }
  }
  if (all_ordinal && !inherits(data_obj, "magmaan_ordinal_data")) {
    stop("mplus_wlsmv_invariance(): all-ordinal fits require a data.frame or ",
         "magmaan_ordinal_data", call. = FALSE)
  }
  if (!all_ordinal && !inherits(data_obj, "magmaan_mixed_ordinal_data")) {
    stop("mplus_wlsmv_invariance(): mixed ordinal fits require a data.frame or ",
         "magmaan_mixed_ordinal_data", call. = FALSE)
  }

  lv <- unique(as.character(configural$partable$lhs[configural$partable$op == "=~"]))
  if (!length(lv)) {
    stop("mplus_wlsmv_invariance(): could not find latent variables in `model`",
         call. = FALSE)
  }
  group_free <- paste(c("0", rep("NA", length(group_labels) - 1L)),
                      collapse = ", ")
  group_zero <- paste(rep("0", length(group_labels)), collapse = ", ")
  latent_mean_rows <- paste0(lv, " ~ c(", group_free, ")*1")
  ordered_intercept_rows <- paste0(ordered, " ~ c(", group_zero, ")*1")
  scalar_extra <- paste(c(latent_mean_rows, ordered_intercept_rows),
                        collapse = "\n")
  continuous_ov <- setdiff(unique(unlist(ov_by_group, use.names = FALSE)),
                           ordered)
  scalar_equal <- c("loadings", "thresholds",
                    if (length(continuous_ov)) "intercepts")

  specs <- list(
    configural = configural,
    metric = make_spec(group_equal = "loadings"),
    scalar = make_spec(extra = scalar_extra, group_equal = scalar_equal)
  )
  requested <- unique(c("configural", steps))

  fit_one <- function(spec) {
    magmaan(spec, data_obj, estimator = fit_estimator, missing = missing,
            optimizer = optimizer, control = control, bounds = bounds)
  }
  fits <- lapply(specs[requested], fit_one)

  test_rows <- list()
  if (length(setdiff(requested, "configural"))) {
    for (nm in setdiff(requested, "configural")) {
      nt <- nestedTest(fits$configural, fits[[nm]], data = data_obj,
                       method = "restriction_map", A.method = "delta",
                       weight = fit_estimator)
      ss <- nt$scaled_shifted %||% list()
      test_rows[[nm]] <- data.frame(
        h1 = "configural",
        h0 = nm,
        T_diff = nt$T_diff %||% NA_real_,
        df_diff = nt$df_diff %||% NA_integer_,
        scale_c = nt$scale_c %||% NA_real_,
        T_scaled = nt$T_scaled %||% NA_real_,
        p_scaled = nt$p_scaled %||% NA_real_,
        T_scaled_shifted = ss$chi2_adj %||% NA_real_,
        df_scaled_shifted = ss$df %||% NA_real_,
        p_scaled_shifted = nt$p_scaled_shifted %||% NA_real_,
        stringsAsFactors = FALSE)
    }
  }
  tests <- if (length(test_rows)) {
    do.call(rbind, test_rows)
  } else {
    data.frame()
  }
  rownames(tests) <- NULL

  out <- list(
    fits = fits,
    tests = tests,
    data = data_obj,
    syntax = list(
      configural = specs$configural$syntax,
      metric = specs$metric$syntax,
      scalar = specs$scalar$syntax),
    options = list(
      estimator = estimator,
      fit_estimator = fit_estimator,
      parameterization = "delta",
      missing = missing,
      pd_gamma = pd_gamma,
      ordered = ordered,
      group = group,
      group_labels = group_labels,
      all_ordinal = all_ordinal,
      validate = isTRUE(validate)))
  class(out) <- c("magmaan_mplus_invariance", "list")
  out
}

print.magmaan_mplus_invariance <- function(x, ...) {
  cat("magmaan Mplus-style WLSMV invariance\n")
  cat("  estimator: ", x$options$estimator %||% "WLSMV", "\n", sep = "")
  cat("  parameterization: delta\n")
  cat("  groups: ", paste(x$options$group_labels, collapse = ", "), "\n", sep = "")
  cat("  ordered: ", paste(x$options$ordered, collapse = ", "), "\n", sep = "")
  if (nrow(x$tests)) print(x$tests)
  invisible(x)
}
