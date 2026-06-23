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

# Parameter covariance under an explicit SE regime. `regime = "model"` uses the
# expected / Gauss-Newton bread (efficient when the model is correct; the
# robust.sem / WLSMV-style default); `regime = "robust"` uses the observed-Hessian
# bread. For ordinal DWLS this is the fixed-weight observed-bread sandwich; the
# estimated-weight infinitesimal jackknife is exposed explicitly as
# `magmaan_core$robust_ordinal_ij()` for all-ordinal fits. Feed the covariance
# into standardized(fit, vcov) to carry the regime choice into SE(beta).
vcov.magmaan_fit <- function(object, regime = c("model", "robust"),
                             data = NULL, ...) {
  regime <- match.arg(regime)
  bread <- if (identical(regime, "robust")) "observed" else "expected"
  fit <- object
  if (isTRUE(fit$ordinal)) {
    stats <- fit$ordinal_stats
    if (is.null(stats)) {
      stop("vcov(): ordinal fit does not carry $ordinal_stats")
    }
    return(magmaan_core$robust_ordinal(fit, stats, "", bread)$vcov)
  }
  if (isTRUE(fit$mixed_ordinal)) {
    stats <- fit$mixed_ordinal_stats
    if (is.null(stats)) {
      stop("vcov(): mixed fit does not carry $mixed_ordinal_stats")
    }
    return(magmaan_core$robust_mixed_ordinal(fit, stats, "", bread)$vcov)
  }
  if (is.null(data)) {
    stop("vcov(): continuous fits need `data` (raw observations) for the ",
         "empirical-meat sandwich")
  }
  magmaan_core$robust_se_raw_fit(fit, raw_data_arg(fit, data),
                                 bread = bread)$vcov
}

composite_weights <- function(fit, vcov) {
  if (missing(vcov)) {
    stop("composite_weights(): `vcov` is required; compute it explicitly before calling")
  }
  magmaan_core$measures_composite_weights(fit, vcov)
}

residuals.magmaan_fit <- function(object, standardized = FALSE, ...) {
  if (isTRUE(standardized)) {
    return(magmaan_core$measures_standardized_residuals(object))
  }
  magmaan_core$measures_residuals(object)
}

# lavResiduals() analogue: the standardized (cor.bentler) residual matrices, the
# residual SE/z-statistics, the SRMR, and the per-block `$summary` table
# (SRMR/USRMR with SE, exact-fit and close-fit z-tests, and a close-fit CI).
# Equivalent to residuals(fit, standardized = TRUE); named for familiarity with
# lavaan::lavResiduals(). `$summary` is a list of data frames, one per block.
lav_residuals <- function(fit) {
  magmaan_core$measures_standardized_residuals(fit)
}

factor_scores <- function(fit, data, method = NULL) {
  if (is.null(method)) {
    method <- if (isTRUE(fit$ordinal) || isTRUE(fit$mixed_ordinal)) {
      "EBM"
    } else {
      "regression"
    }
  } else {
    method <- match.arg(method, c("regression", "bartlett", "EBM", "ML", "EAP",
                                  "ebm", "ml", "eap"))
  }
  if (missing(data)) {
    stop("factor_scores(): `data` is required; pass complete observed raw data")
  }
  magmaan_core$measures_factor_scores(fit, raw_data_arg(fit, data), method = method)
}

factor_score_precision <- function(fit, data) {
  if (missing(data)) {
    stop("factor_score_precision(): `data` is required; pass complete observed raw data")
  }
  magmaan_core$measures_factor_score_precision(fit, raw_data_arg(fit, data))
}

modification_indices <- function(fit, data = NULL, ..., candidates = "all") {
  dots <- list(...)
  if (!is.null(data)) {
    if ("weight" %in% names(dots)) {
      stop("modification_indices(): pass only one of `data` or `weight`")
    }
    dots$weight <- data
  }
  dots$candidates <- candidates
  do.call(magmaan_core$inference_modification_indices, c(list(fit = fit), dots))
}

score_tests <- function(fit, data = NULL, ...) {
  dots <- list(...)
  if (!is.null(data)) {
    if ("weight" %in% names(dots)) {
      stop("score_tests(): pass only one of `data` or `weight`")
    }
    dots$weight <- data
  }
  do.call(magmaan_core$inference_score_tests, c(list(fit = fit), dots))
}

# Robust (generalized / Satorra-Bentler-scaled) modification indices and score
# tests: the `*_robust` frontier mirror of the two functions above. Each row
# keeps the ordinary `mi` and adds `mi_scaled = mi / scaling_factor`.
#
# Ordinal/mixed fits use the polychoric NACOV the fit already carries, so the
# scaling is intrinsic to the diagonal/identity weight (DWLS/ULS scale even on
# normal data) and `bread`/`moments`/`cov` are ignored. Continuous ML/ULS/GLS
# build the meat from `cov`: 'empirical'/'browne_unbiased' need the fitting
# `data` (raw observations); 'model_implied' uses Gamma_NT(S) and reduces to the
# ordinary statistic. WLS supplies its weight via `weight=` and always reduces.
# `estimated_weight = TRUE` routes the per-direction scaling through the complete
# (Hall-Inoue) sandwich, which carries the data-dependent-weight IF(W-hat) meat
# term beyond lavaan's global SB scalar. It applies to estimated second-stage
# weights (continuous GLS/WLS, ordinal/categorical DWLS/WLS), needs the fitting
# `data` for the continuous tier, and is not available for ML or mixed-ordinal.
modification_indices_robust <- function(fit, data = NULL, weight = NULL,
                                        bread = "expected",
                                        moments = "structured",
                                        cov = "empirical",
                                        candidates = "all",
                                        include_loadings = TRUE,
                                        include_covariances = TRUE,
                                        information = "expected",
                                        estimated_weight = FALSE) {
  is_ord <- isTRUE(fit$ordinal) || isTRUE(fit$mixed_ordinal)
  raw <- if (!is_ord && !is.null(data)) raw_data_arg(fit, data) else NULL
  magmaan_core$inference_modification_indices_robust(
    fit, raw = raw, weight = weight, bread = bread, moments = moments,
    cov = cov, information = information, candidates = candidates,
    include_loadings = include_loadings, include_covariances = include_covariances,
    estimated_weight = estimated_weight)
}

score_tests_robust <- function(fit, data = NULL, weight = NULL,
                               bread = "expected", moments = "structured",
                               cov = "empirical", estimated_weight = FALSE) {
  is_ord <- isTRUE(fit$ordinal) || isTRUE(fit$mixed_ordinal)
  raw <- if (!is_ord && !is.null(data)) raw_data_arg(fit, data) else NULL
  magmaan_core$inference_score_tests_robust(
    fit, raw = raw, weight = weight, bread = bread, moments = moments, cov = cov,
    estimated_weight = estimated_weight)
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
    block <- data[rows, ov, drop = FALSE]
    if (isTRUE(fit$ordinal) || isTRUE(fit$mixed_ordinal)) {
      for (nm in ov) {
        if (is.factor(block[[nm]])) {
          block[[nm]] <- as.integer(block[[nm]])
        }
      }
      return(data.matrix(block))
    }
    as.matrix(block)
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
