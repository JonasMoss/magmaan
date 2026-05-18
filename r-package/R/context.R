fit_sample_stats <- function(fit) {
  if (inherits(fit, "magmaan_data")) {
    return(list(S = fit$S, nobs = fit$nobs, mean = fit$mean))
  }
  list(S = fit$S, nobs = fit$nobs, mean = fit$sample_mean)
}

standardized <- function(fit, vcov, type = c("all", "lv")) {
  type <- match.arg(type)
  if (missing(vcov)) {
    stop("standardized(): `vcov` is required; compute it explicitly before calling")
  }
  if (identical(type, "all")) {
    return(magmaan_core$measures_standardize_all(fit, vcov))
  }
  magmaan_core$measures_standardize_lv(fit, vcov)
}

residuals.magmaan_fit <- function(object, standardized = FALSE, ...) {
  if (isTRUE(standardized)) {
    return(magmaan_core$measures_standardized_residuals(object))
  }
  magmaan_core$measures_residuals(object)
}

factor_scores <- function(fit, data, method = c("regression", "bartlett")) {
  method <- match.arg(method)
  if (missing(data)) {
    stop("factor_scores(): `data` is required; pass complete observed raw data")
  }
  magmaan_core$measures_factor_scores(fit, raw_data_arg(fit, data), method = method)
}

modification_indices <- function(fit, ..., candidates = "all") {
  magmaan_core$inference_modification_indices(fit, ..., candidates = candidates)
}

raw_data_arg <- function(fit, data) {
  if (!is.data.frame(data)) return(data)

  rep <- magmaan_core$model_matrix_rep(fit$partable)
  ov_by_group <- rep$ov_names
  if (!is.list(ov_by_group)) ov_by_group <- list(ov_by_group)
  group_var <- fit$group_var %||% ""
  group_labels <- fit$group_labels %||% character()

  make_block <- function(rows, ov) {
    missing <- setdiff(ov, names(data))
    if (length(missing)) {
      stop("factor_scores(): `data` is missing observed variables: ",
           paste(missing, collapse = ", "))
    }
    as.matrix(data[rows, ov, drop = FALSE])
  }

  if (length(ov_by_group) == 1L && !nzchar(group_var)) {
    return(make_block(rep(TRUE, nrow(data)), ov_by_group[[1L]]))
  }
  if (!nzchar(group_var) || !group_var %in% names(data)) {
    stop("factor_scores(): grouped fits require `data` with grouping column `",
         group_var, "` or an explicit list of raw matrices")
  }
  if (!length(group_labels)) {
    group_labels <- unique(as.character(data[[group_var]]))
  }
  if (length(group_labels) != length(ov_by_group)) {
    stop("factor_scores(): fit has ", length(ov_by_group),
         " group block(s), but ", length(group_labels), " group label(s)")
  }
  g <- as.character(data[[group_var]])
  X <- Map(function(label, ov) make_block(g == label, ov), group_labels, ov_by_group)
  names(X) <- group_labels
  list(X = X)
}
