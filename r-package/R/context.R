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
# NULL selects observed empirical covariance; model/robust are legacy aliases.
vcov.magmaan_fit <- function(object, regime = NULL, data = NULL, ...) {
  fit <- object
  if (identical(fit$penalty_inference, "not_validated") ||
      identical(fit$composition$moment_source, "pairwise_mcar")) {
    stop("vcov(): this fit requires its own sampling/inference contract", call. = FALSE)
  }
  sam <- inherits(fit, "magmaan_sam_fit")
  noniterative <- .is_noniterative(fit)
  estimator <- toupper(fit$estimator %||% "")
  fiml <- isTRUE(fit$fiml) || identical(estimator, "FIML")
  categorical <- isTRUE(fit$ordinal) || isTRUE(fit$mixed_ordinal)
  default <- if (sam) "stored" else if (noniterative) "delta_nt" else if (categorical || estimator %in% c("GLS", "WLS")) "sandwich_ij" else "sandwich_observed"
  if (is.null(regime)) regime <- default
  regime <- match.arg(regime, c("information_expected", "information_observed",
    "sandwich_expected", "sandwich_observed", "sandwich_ij", "delta_nt",
    "delta_empirical", "stored", "model", "robust"))
  if (identical(regime, "model")) regime <- if (sam) "stored" else if (noniterative) "delta_nt" else if (fiml) "information_observed" else "sandwich_expected"
  if (identical(regime, "robust")) {
    regime <- if (sam) "stored" else if (noniterative) "delta_empirical" else if (categorical || estimator %in% c("GLS", "WLS")) "sandwich_ij" else "sandwich_observed"
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
    # sandwich_ij is the estimated-weight (infinitesimal-jackknife) sandwich
    # the ordinary DWLS policy uses; the other sandwiches keep the weight fixed
    # (sandwich_expected is lavaan's robust.sem).
    if (identical(regime, "sandwich_ij")) {
      if (isTRUE(fit$mixed_ordinal)) return(infer_mixed_ordinal_robust_ij(fit, fit$mixed_ordinal_stats)$vcov)
      if (!isTRUE(fit$ordinal)) unsupported()
      if (is.null(fit$ordinal_stats)) stop("vcov(): ordinal fit does not carry $ordinal_stats")
      return(infer_ordinal_robust_ij(fit, fit$ordinal_stats)$vcov)
    }
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
  if (!sandwich && !identical(regime, "sandwich_ij")) unsupported()
  data <- data %||% fit$raw_data
  if (is.null(data)) {
    stop("vcov(): empirical-meat sandwiches need raw observations; supply `data` or refit with raw data",
         call. = FALSE)
  }
  raw <- raw_data_arg(fit, data, caller = "vcov")
  if (is.list(raw) && !is.null(raw$X)) raw <- raw$X
  if (estimator %in% c("ULS", "GLS", "WLS")) {
    if (identical(regime, "sandwich_ij")) {
      return(crossprod(infer_casewise_influence_ij_fit(fit, raw, weight = fit$W)$influence))
    }
    return(magmaan_core$infer_continuous_ls_robust(fit, raw, weight = fit$W,
      bread = bread, gamma = "empirical", fixed_weight = TRUE)$vcov)
  }
  magmaan_core$robust_se_raw_fit(fit, raw, bread = bread)$vcov
}

composite_weights <- function(fit, vcov) {
  if (missing(vcov)) {
    stop("composite_weights(): `vcov` is required; compute it explicitly before calling")
  }
  magmaan_core$measures_composite_weights(fit, vcov)
}

# `estimated_weight = TRUE` is the misspecification-robust default. For
# continuous GLS/WLS/ULS it routes residual SE/z and `$summary` inference through the Hall-Inoue complete sandwich, which
# carries the data-dependent-weight influence IF(W-hat) beyond lavaan's NT
# projection; it needs the fitting `data` (raw observations).
residuals.magmaan_fit <- function(object, standardized = FALSE,
                                  estimated_weight = TRUE, data = NULL, ...) {
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
# `estimated_weight = TRUE` is the misspecification-robust default. For
# continuous GLS/WLS/ULS it uses the complete (Hall-Inoue) residual ACOV instead of the NT projection and needs `data`.
lav_residuals <- function(fit, estimated_weight = TRUE, data = NULL) {
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
  if (is.null(dots$gamma) && identical(dots$cov, "model_implied") &&
      identical(dots$estimated_weight, FALSE)) {
    return(magmaan_core$inference_modification_indices(fit, weight = dots$weight,
      information = dots$information %||% dots$bread %||% "expected", candidates = candidates))
  }
  if (is.null(dots$estimated_weight) && toupper(fit$estimator) %in% c("ML", "FIML")) dots$estimated_weight <- FALSE
  do.call(modification_indices_robust, c(list(fit = fit, data = data, candidates = candidates), dots))
}

score_tests <- function(fit, data = NULL, ...) {
  if (.is_noniterative(fit)) .guard_noniterative("score_tests()")
  dots <- list(...)
  if (is.null(dots$gamma) && identical(dots$cov, "model_implied") &&
      identical(dots$estimated_weight, FALSE)) {
    return(magmaan_core$inference_score_tests(fit, weight = dots$weight))
  }
  if (is.null(dots$estimated_weight) && toupper(fit$estimator) %in% c("ML", "FIML")) dots$estimated_weight <- FALSE
  do.call(score_tests_robust, c(list(fit = fit, data = data), dots))
}

# Robust (generalized / Satorra-Bentler-scaled) modification indices and score
# tests: the `*_robust` frontier mirror of the two functions above. Each row
# reports the unscaled quadratic `mi` and adds `mi_scaled = mi / scaling_factor`.
#
# Ordinal/mixed fits use the polychoric NACOV the fit already carries, so the
# scaling is intrinsic to the diagonal/identity weight (DWLS/ULS scale even on
# normal data). All-ordinal `bread` selects observed or expected nuisance
# sensitivity; mixed fits retain their expected projection. Categorical
# `moments`/`cov` choices use the retained NACOV. Continuous ML/ULS/GLS/WLS
# build the meat from `cov`: 'empirical' needs the fitting `data` (raw
# observations); 'model_implied' uses Gamma_NT from the chosen moments.
# WLS-computed fits (WLS/ADF, DWLS, DLS, supplied W) use the fitting weight
# recorded in fit$W; `weight=` is needed only for fits without that record and
# must otherwise equal it. The ordinary statistic is recovered only when the
# weight is the inverse of the selected Gamma.
# Continuous and all-ordinal LS use observed-Hessian nuisance sensitivity by
# default, with the expected quadratic metric and matching projected meat.
# Expected sensitivity and fixed weights remain explicit comparators.
# Continuous LS does not implement 'browne_unbiased', and estimated-weight mode
# requires 'empirical'; unavailable covariance choices error explicitly.
# `estimated_weight = TRUE` (default) routes the scaling through the complete
# (Hall-Inoue) sandwich, which carries the data-dependent-weight IF(W-hat) meat
# term beyond lavaan's global SB scalar. The influence follows the fit's
# recorded weight recipe (NT, ADF, DWLS or DLS with its mixing weight; ULS has
# none). A supplied W has no recipe, and ordinal NT/DLS weights have no derived
# influence yet, so both are refused with UnsupportedInference. It needs the
# fitting `data` for the continuous tier and is not available for ML or
# mixed-ordinal.
# FIML uses observed information and observed-pattern casewise score meat;
# omitted bread/information select 'observed'. It uses retained raw observations
# unless `data` is supplied. Expected information, alternative covariance/moment
# recipes and second-stage weights are unsupported and rejected explicitly.
# Two-stage (ML2S) fits report `mi` as the naive Stage-2 statistic on the
# Stage-1 EM moments (attribute mi_type = "naive_stage2") and `mi.scaled` with
# the Stage-1 moment covariance as meat. They use the recorded Stage-2 weight
# and retained data; `data`, `weight`, observed information and non-default
# bread/moments/cov are refused. `estimated_weight = TRUE` adds the DWLS, ADF
# or DLS Stage-2 weight's data influence.
#' Robust modification indices and equality-release score tests
#'
#' @rdname robust_score_tests
#' @param gamma Optional caller NACOV: a symmetric positive semidefinite matrix
#'   for one group or a list in fitted group order. Continuous moments use means
#'   first (when fitted), then lower-triangle covariances by columns including
#'   the diagonal, in the model's observed-variable order. Ordinal moments use
#'   thresholds then lower-triangle polychorics; mixed moments use thresholds,
#'   negative continuous means, continuous variances, then associations.
#'   Blocks are unscaled per-group NACOV (N times the sampling covariance).
#'   Supplying gamma replaces the meat, preserves fitting weights, and requires
#'   estimated_weight = FALSE and cov = "empirical". FIML/ML2S refuse it.
modification_indices_robust <- function(fit, data = NULL, weight = NULL,
                                        bread = "observed",
                                        moments = "auto",
                                        cov = "empirical",
                                        candidates = "all",
                                        include_loadings = TRUE,
                                        include_covariances = TRUE,
                                        information = "expected",
                                        estimated_weight = TRUE, gamma = NULL) {
  if (.is_noniterative(fit)) .guard_noniterative("modification_indices_robust()")
  if (identical(fit$estimator, "FIML")) {
    if (missing(bread)) bread <- "observed"
    if (missing(information)) information <- "observed"
  }
  is_ord <- isTRUE(fit$ordinal) || isTRUE(fit$mixed_ordinal)
  raw <- if (!is_ord && !is.null(data)) raw_data_arg(fit, data) else NULL
  magmaan_core$inference_modification_indices_robust(
    fit, raw = raw, weight = weight, bread = bread, moments = moments,
    cov = cov, information = information, candidates = candidates,
    include_loadings = include_loadings, include_covariances = include_covariances,
    estimated_weight = estimated_weight, gamma = gamma)
}

#' Robust modification indices and equality-release score tests
#'
#' @rdname robust_score_tests
#' @param gamma Optional caller NACOV: a symmetric positive semidefinite matrix
#'   for one group or a list in fitted group order. Continuous moments use means
#'   first (when fitted), then lower-triangle covariances by columns including
#'   the diagonal, in the model's observed-variable order. Ordinal moments use
#'   thresholds then lower-triangle polychorics; mixed moments use thresholds,
#'   negative continuous means, continuous variances, then associations.
#'   Blocks are unscaled per-group NACOV (N times the sampling covariance).
#'   Supplying gamma replaces the meat, preserves fitting weights, and requires
#'   estimated_weight = FALSE and cov = "empirical". FIML/ML2S refuse it.
score_tests_robust <- function(fit, data = NULL, weight = NULL,
                               bread = "observed", moments = "auto",
                               cov = "empirical", estimated_weight = TRUE, gamma = NULL) {
  if (.is_noniterative(fit)) .guard_noniterative("score_tests_robust()")
  if (identical(fit$estimator, "FIML") && missing(bread)) bread <- "observed"
  is_ord <- isTRUE(fit$ordinal) || isTRUE(fit$mixed_ordinal)
  raw <- if (!is_ord && !is.null(data)) raw_data_arg(fit, data) else NULL
  magmaan_core$inference_score_tests_robust(
    fit, raw = raw, weight = weight, bread = bread, moments = moments, cov = cov,
    estimated_weight = estimated_weight, gamma = gamma)
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
