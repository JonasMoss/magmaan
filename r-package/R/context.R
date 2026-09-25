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

# Explicit names identify the formula, independently of the estimator.
# NULL preserves historical numerical defaults; model/robust are legacy aliases.
vcov.magmaan_fit <- function(object, regime = NULL, data = NULL, ...) {
  fit <- object
  sam <- inherits(fit, "magmaan_sam_fit")
  noniterative <- .is_noniterative(fit)
  estimator <- toupper(fit$estimator %||% "")
  fiml <- isTRUE(fit$fiml) || identical(estimator, "FIML")
  categorical <- isTRUE(fit$ordinal) || isTRUE(fit$mixed_ordinal)
  default <- if (sam) "stored" else if (noniterative) "delta_nt" else if (fiml) {
    "information_observed"
  } else "sandwich_expected"
  if (is.null(regime)) regime <- default
  regime <- match.arg(regime, c("information_expected", "information_observed",
    "sandwich_expected", "sandwich_observed", "delta_nt", "delta_empirical",
    "stored", "model", "robust"))
  if (identical(regime, "model")) regime <- default
  if (identical(regime, "robust")) {
    regime <- if (sam) "stored" else if (noniterative) "delta_empirical" else "sandwich_observed"
  }
  unsupported <- function() {
    stop("vcov(): regime = '", regime, "' is not supported for this fit", call. = FALSE)
  }
  if (sam) {
    if (!identical(regime, "stored")) unsupported()
    if (is.null(fit$vcov)) {
      stop("vcov(): SAM fit does not carry a covariance matrix; refit with se != 'none'")
    }
    return(fit$vcov)
  }
  if (noniterative) {
    if (!regime %in% c("delta_nt", "delta_empirical")) unsupported()
    data <- data %||% fit$raw_data
    gamma <- if (identical(regime, "delta_empirical")) "empirical" else "nt"
    if (identical(gamma, "empirical") && is.null(data)) {
      stop("vcov(): delta_empirical needs raw observations; supply `data` or refit with raw data",
           call. = FALSE)
    }
    return(noniterative_cfa_se(fit, gamma = gamma, data = data)$vcov)
  }
  sandwich <- regime %in% c("sandwich_expected", "sandwich_observed")
  information <- regime %in% c("information_expected", "information_observed")
  bread <- if (regime %in% c("sandwich_observed", "information_observed")) "observed" else "expected"
  if (categorical) {
    if (!sandwich) unsupported()
    if (isTRUE(fit$ordinal)) {
      if (is.null(fit$ordinal_stats)) stop("vcov(): ordinal fit does not carry $ordinal_stats")
      return(magmaan_core$robust_ordinal(fit, fit$ordinal_stats, "", bread)$vcov)
    }
    if (is.null(fit$mixed_ordinal_stats)) stop("vcov(): mixed fit does not carry $mixed_ordinal_stats")
    return(magmaan_core$robust_mixed_ordinal(fit, fit$mixed_ordinal_stats, "", bread)$vcov)
  }
  if (fiml) {
    if (!sandwich && !information) unsupported()
    if (!is.null(data)) stop("vcov(): FIML uses the fit's retained data; omit `data`", call. = FALSE)
    if (identical(regime, "sandwich_observed")) {
      return(magmaan_core$estimate_fiml_robust_mlr(fit)$vcov)
    }
    if (identical(regime, "information_observed")) {
      return(magmaan_core$fiml_observed_vcov(fit)$vcov)
    }
    expected <- magmaan_core$inference_fiml_information_vcov(fit)$expected
    if (!isTRUE(expected$ok)) stop("vcov(): ", expected$error, call. = FALSE)
    return(if (sandwich) expected$vcov_sandwich else expected$vcov_model)
  }
  if (information) {
    if (!identical(estimator, "ML")) unsupported()
    info <- if (identical(bread, "observed")) {
      magmaan_core$inference_information_observed_analytic(fit)
    } else magmaan_core$inference_information_expected(fit)
    return(magmaan_core$inference_vcov(info, fit))
  }
  if (!sandwich) unsupported()
  data <- data %||% fit$raw_data
  if (is.null(data)) {
    stop("vcov(): empirical-meat sandwiches need raw observations; supply `data` or refit with raw data",
         call. = FALSE)
  }
  raw <- raw_data_arg(fit, data, caller = "vcov")
  if (is.list(raw) && !is.null(raw$X)) raw <- raw$X
  magmaan_core$robust_se_raw_fit(fit, raw, bread = bread)$vcov
}

composite_weights <- function(fit, vcov) {
  if (missing(vcov)) {
    stop("composite_weights(): `vcov` is required; compute it explicitly before calling")
  }
  magmaan_core$measures_composite_weights(fit, vcov)
}

# `estimated_weight = TRUE` (continuous GLS/WLS/ULS only) routes the residual
# SE/z and `$summary` inference through the Hall-Inoue complete sandwich, which
# carries the data-dependent-weight influence IF(W-hat) beyond lavaan's NT
# projection; it needs the fitting `data` (raw observations).
residuals.magmaan_fit <- function(object, standardized = FALSE,
                                  estimated_weight = FALSE, data = NULL, ...) {
  if (isTRUE(standardized)) {
    if (isTRUE(estimated_weight)) {
      if (is.null(data)) {
        stop("residuals(estimated_weight = TRUE): `data` (raw observations) ",
             "is required")
      }
      return(magmaan_core$measures_standardized_residuals_estimated_weight(
        object, raw_data_arg(object, data)))
    }
    return(magmaan_core$measures_standardized_residuals(object))
  }
  magmaan_core$measures_residuals(object)
}

# lavResiduals() analogue: the standardized (cor.bentler) residual matrices, the
# residual SE/z-statistics, the SRMR, and the per-block `$summary` table
# (SRMR/USRMR with SE, exact-fit and close-fit z-tests, and a close-fit CI).
# Equivalent to residuals(fit, standardized = TRUE); named for familiarity with
# lavaan::lavResiduals(). `$summary` is a list of data frames, one per block.
# `estimated_weight = TRUE` (continuous GLS/WLS/ULS) uses the complete
# (Hall-Inoue) residual ACOV instead of the NT projection and needs `data`.
lav_residuals <- function(fit, estimated_weight = FALSE, data = NULL) {
  if (isTRUE(estimated_weight)) {
    if (is.null(data)) {
      stop("lav_residuals(estimated_weight = TRUE): `data` (raw observations) ",
           "is required")
    }
    return(magmaan_core$measures_standardized_residuals_estimated_weight(
      fit, raw_data_arg(fit, data)))
  }
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
  if (.is_noniterative(fit)) .guard_noniterative("modification_indices()")
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
  if (.is_noniterative(fit)) .guard_noniterative("score_tests()")
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
  if (.is_noniterative(fit)) .guard_noniterative("modification_indices_robust()")
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
  if (.is_noniterative(fit)) .guard_noniterative("score_tests_robust()")
  is_ord <- isTRUE(fit$ordinal) || isTRUE(fit$mixed_ordinal)
  raw <- if (!is_ord && !is.null(data)) raw_data_arg(fit, data) else NULL
  magmaan_core$inference_score_tests_robust(
    fit, raw = raw, weight = weight, bread = bread, moments = moments, cov = cov,
    estimated_weight = estimated_weight)
}

raw_data_arg <- function(fit, data, caller = "raw_data_arg") {
  if (!is.data.frame(data)) return(data)

  rep <- magmaan_core$model_matrix_rep(fit$partable)
  ov_by_group <- rep$ov_names
  if (!is.list(ov_by_group)) ov_by_group <- list(ov_by_group)
  group_var <- fit$group_var %||% ""
  group_labels <- fit$group_labels %||% character()

  make_block <- function(rows, ov) {
    missing <- setdiff(ov, names(data))
    if (length(missing)) {
      stop(caller, "(): `data` is missing observed variables: ",
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
    stop(caller, "(): grouped fits require `data` with grouping column `",
         group_var, "` or an explicit list of raw matrices")
  }
  if (!length(group_labels)) {
    group_labels <- unique(as.character(data[[group_var]]))
  }
  if (length(group_labels) != length(ov_by_group)) {
    stop(caller, "(): fit has ", length(ov_by_group),
         " group block(s), but ", length(group_labels), " group label(s)")
  }
  g <- as.character(data[[group_var]])
  X <- Map(function(label, ov) make_block(g == label, ov), group_labels, ov_by_group)
  names(X) <- group_labels
  list(X = X)
}
