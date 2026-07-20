#' Multiplier score test for affine nested ML/FIML models.
#'
#' Calibrates the restrictions separating `fit_H0` from `fit_H1` by random
#' mean-zero, unit-variance multipliers on individual likelihood-score
#' contributions. Rademacher signs are the default and support the basic,
#' nuisance-effective, and flip-specifically standardized references. Mammen,
#' tunable two-point, Gaussian, and centered-exponential weights are
#' experimental diagnostics for the nuisance-effective reference.
#'
#' @param fit_H1 Less-restricted complete-data ML or direct-FIML fit, or a
#'   `magmaan_model_spec` for the less-restricted model. Supplying the model
#'   avoids fitting H1 because a score test uses only H1's tangent and H0's
#'   estimates.
#' @param fit_H0 More-restricted affine nested fit over the same parameter
#'   slots, estimator, and observations.
#' @param data Raw fitting data for complete-data ML. Direct FIML uses the
#'   observed-data sample stored on both fits, so this argument must be omitted.
#' @param n_flips Number of random multiplier transformations.
#' @param seed Non-negative deterministic integer seed.
#' @param multiplier Multiplier distribution. Non-Rademacher resampling
#'   requires `calibration = "effective"`.
#' @param two_point_skewness Non-negative third moment for
#'   `multiplier = "two-point"`. The corresponding fourth moment is the
#'   minimum possible value, `1 + two_point_skewness^2`; zero gives a
#'   Rademacher law and one gives Mammen's law.
#' @param center_multiplier_scores Whether to center effective-score
#'   contributions within information strata before applying multipliers.
#' @param multiplier_studentization Optional `"weighted-meat"` bootstrap-t
#'   diagnostic using the per-draw meat
#'   `sum(w_i^2 u_i u_i')`. `"none"` retains the expected-information metric.
#' @param calibration Which randomization references to compute. `"effective"`
#'   skips the basic and flip-specific covariance calculations;
#'   `"effective-standardized"` adds the standardized reference; `"all"`
#'   preserves the original three-reference result. `"asymptotic"` draws no
#'   multipliers and is used by [nested_score_test()].
#'
#' @return A list of class `magmaan_score_flip_test`. In addition to the three
#'   p-values it includes the mean-scaled and exact-mixture score references,
#'   the direct sandwich-studentized score statistic and p-value, flip-variance
#'   displacement diagnostics, optional multiplier-studentized output, and
#'   elapsed seconds for setup, resampling, flip standardization, and the
#'   asymptotic comparators.
#' @export
score_flip_test <- function(fit_H1, fit_H0, data = NULL,
                            n_flips = 999L, seed = 1,
                            calibration = c("all", "effective-standardized",
                                            "effective", "asymptotic"),
                            multiplier = c("rademacher", "mammen", "two-point",
                                           "gaussian", "centered-exponential"),
                            two_point_skewness = 1,
                            center_multiplier_scores = FALSE,
                            multiplier_studentization = c("none",
                                                          "weighted-meat")) {
  calibration <- match.arg(calibration)
  multiplier <- match.arg(multiplier)
  multiplier_studentization <- match.arg(multiplier_studentization)
  h1_is_model <- inherits(fit_H1, "magmaan_model_spec")
  estimator_H0 <- toupper(fit_H0$estimator %||% "ML")
  estimator_H1 <- if (h1_is_model) estimator_H0 else
    toupper(fit_H1$estimator %||% "ML")
  if (!identical(estimator_H1, estimator_H0) ||
      !estimator_H1 %in% c("ML", "FIML")) {
    stop("score_flip_test(): fits must use the same ML or FIML estimator",
         call. = FALSE)
  }
  n_flips <- as.integer(n_flips)[1L]
  if (is.na(n_flips) ||
      (calibration != "asymptotic" && n_flips < 1L) || n_flips < 0L) {
    stop("score_flip_test(): `n_flips` must be non-negative and positive when multipliers are drawn",
         call. = FALSE)
  }
  if (multiplier != "rademacher" &&
      !calibration %in% c("effective", "asymptotic")) {
    stop("score_flip_test(): non-Rademacher resampling requires `calibration = \"effective\"`",
         call. = FALSE)
  }
  two_point_skewness <- as.numeric(two_point_skewness)
  if (length(two_point_skewness) != 1L ||
      !is.finite(two_point_skewness) ||
      two_point_skewness < 0 || two_point_skewness > 1e6) {
    stop("score_flip_test(): `two_point_skewness` must be finite and in [0, 1e6]",
         call. = FALSE)
  }
  if (!is.logical(center_multiplier_scores) ||
      length(center_multiplier_scores) != 1L ||
      is.na(center_multiplier_scores)) {
    stop("score_flip_test(): `center_multiplier_scores` must be TRUE or FALSE",
         call. = FALSE)
  }
  center_multiplier_scores <- isTRUE(center_multiplier_scores)
  if ((center_multiplier_scores ||
       multiplier_studentization != "none") &&
      calibration != "effective") {
    stop("score_flip_test(): multiplier centering and studentization require `calibration = \"effective\"`",
         call. = FALSE)
  }
  seed <- as.numeric(seed)[1L]
  if (!is.finite(seed) || seed < 0 || seed != floor(seed)) {
    stop("score_flip_test(): `seed` must be a non-negative integer", call. = FALSE)
  }
  active_lower <- fit_H0$diagnostics$active_bounds_lower
  active_upper <- fit_H0$diagnostics$active_bounds_upper
  if (length(active_lower) > 0L || length(active_upper) > 0L) {
    stop("score_flip_test(): boundary null fits are not supported", call. = FALSE)
  }
  if (estimator_H1 == "FIML") {
    if (!missing(data) && !is.null(data)) {
      stop("score_flip_test(): FIML uses the sample stored on the fits; omit `data`",
           call. = FALSE)
    }
    if (is.null(fit_H0$raw_data) ||
        (!h1_is_model && is.null(fit_H1$raw_data))) {
      stop("score_flip_test(): FIML null fits must carry `raw_data`", call. = FALSE)
    }
    if (!h1_is_model && !identical(fit_H1$raw_data, fit_H0$raw_data)) {
      stop("score_flip_test(): FIML fits were not fitted to the same sample",
           call. = FALSE)
    }
    raw <- fit_H0$raw_data
  } else {
    if (missing(data) || is.null(data)) {
      stop("score_flip_test(): complete-data ML score flips require `data`",
           call. = FALSE)
    }
    raw <- raw_data_arg(fit_H0, data)
    if (is.list(raw) && !is.null(raw$X)) raw <- raw$X
  }
  out <- if (h1_is_model) {
    magmaan_core$inference_score_flip_test_model(
      fit_H1$partable, fit_H0, raw, n_flips, seed, calibration, multiplier,
      two_point_skewness, center_multiplier_scores,
      multiplier_studentization)
  } else {
    magmaan_core$inference_score_flip_test(
      fit_H1, fit_H0, raw, n_flips, seed, calibration, multiplier,
      two_point_skewness, center_multiplier_scores,
      multiplier_studentization)
  }
  out$calibration <- calibration
  out$multiplier <- multiplier
  out$two_point_skewness <- two_point_skewness
  out$center_multiplier_scores <- center_multiplier_scores
  out$multiplier_studentization <- multiplier_studentization
  class(out) <- c("magmaan_score_flip_test", "list")
  out
}

#' Nuisance-effective nested score test without random sign calibration.
#'
#' Computes the same nuisance-orthogonalized observed score and robust
#' restriction spectrum used by [score_flip_test()], but skips every random
#' sign draw and the flip-specific covariance calculations. The recommended
#' `p_value` is the pEBA4 mixture approximation. Scalar SB, exact-mixture, and
#' direct sandwich comparators remain available for diagnostics.
#'
#' @inheritParams score_flip_test
#'
#' @return A list of class `magmaan_nested_score_test` containing the effective
#'   score statistic, restriction eigenvalues, pEBA4/SB/exact-mixture/sandwich
#'   p-values, conditioning diagnostics, and setup/asymptotic timings.
#' @export
nested_score_test <- function(fit_H1, fit_H0, data = NULL) {
  out <- score_flip_test(
    fit_H1, fit_H0, data = data, n_flips = 0L, seed = 0,
    calibration = "asymptotic")
  peba4 <- magmaan_core$robust_fmg_test(
    out$statistic_effective, out$df, out$eigenvalues,
    method = "peba", param = 4, truncate_negative = TRUE)
  out$p_peba4 <- peba4$p_value
  out$p_sb <- out$p_mean_scaled
  out$p_all <- out$p_mixture
  out$p_value <- out$p_peba4
  class(out) <- c("magmaan_nested_score_test", "list")
  out
}

#' @export
print.magmaan_score_flip_test <- function(x, ...) {
  centered_label <- if (isTRUE(x$center_multiplier_scores)) "centered " else ""
  label <- if (!identical(x$multiplier_studentization %||% "none", "none")) {
    paste0(centered_label, x$multiplier %||% "rademacher",
           " weighted-meat multiplier score test")
  } else if (!identical(x$multiplier %||% "rademacher", "rademacher")) {
    if (identical(x$multiplier, "two-point")) {
      paste0(centered_label,
             sprintf("two-point (skewness %.3g) multiplier score test",
                     x$two_point_skewness))
    } else paste0(centered_label, x$multiplier, " multiplier score test")
  } else switch(x$calibration %||% "all",
    effective = "effective sign-flip score test",
    `effective-standardized` = "standardized sign-flip score test",
    asymptotic = "nested score test (no sign calibration)",
    "basic/effective/standardized sign-flip score test")
  statistic <- if (is.finite(x$statistic_standardized))
    x$statistic_standardized else x$statistic_effective
  cat(label, "\n")
  cat("  statistic:", format(statistic),
      " df:", x$df, "\n")
  cat("  p (basic/effective/standardized):",
      paste(format(c(x$p_basic, x$p_effective, x$p_standardized)),
            collapse = " / "), "\n")
  if (isTRUE(x$sandwich_available)) {
    cat("  direct sandwich score: statistic", format(x$statistic_sandwich),
        " p", format(x$p_sandwich), "\n")
  }
  cat("  random flips:", x$n_flips, " seed:", format(x$seed), "\n")
  invisible(x)
}

#' @export
print.magmaan_nested_score_test <- function(x, ...) {
  cat("nuisance-effective nested score test\n")
  cat("  statistic:", format(x$statistic_effective),
      " df:", x$df, "\n")
  cat("  p (pEBA4/SB/exact mixture):",
      paste(format(c(x$p_peba4, x$p_sb, x$p_all)), collapse = " / "),
      "\n")
  invisible(x)
}

#' Robust nested-model likelihood-ratio tests.
#'
#' Compares a restricted model `fit_H0` against the less-restricted superset
#' `fit_H1`. `method = "restriction_map"` uses Satorra's streaming low-rank
#' reduction of the asymptotic weighted-chi-square distribution. The
#' `"lavaan_sb2001"` and `"lavaan_sb2010"` options mirror lavaan's
#' compatibility difference-test approximations.
#'
#' @param fit_H1 Less-restricted fitted magmaan model, ordinarily returned by
#'   [magmaan()].
#' @param fit_H0 More-restricted fit (same lavaanified partable shape,
#'   differing only in constraint rows / shared labels).
#' @param data Raw complete data for complete-data fits: either a data.frame
#'   whose columns include the observed variables of `fit_H1` (single-group
#'   case), or a list of per-group matrices in the same block order the fit was
#'   built with. Ordinal DWLS/WLS pairs take the `magmaan_ordinal_data` object
#'   used for fitting. FIML pairs use `fit_H1$raw_data` and do not accept `data`.
#' @param gamma `"empirical"` (default, empirical Gamma-hat), `"unbiased"`
#'   (the Browne/Du-Bentler finite-sample correction for complete-data ML),
#'   `"both"` (both spectra in one shared streaming pass; the unbiased spectrum
#'   is returned as `eigenvalues_unbiased`; complete-data ML restriction-map
#'   pairs with `computation = "streaming"` only), or `"NT"` (normal-theory
#'   sanity-check path where all eigenvalues collapse to 1).
#' @param method `"restriction_map"` (default), `"lavaan_sb2001"`, or
#'   `"lavaan_sb2010"`.
#' @param A.method `"exact"` (default, exact parameter-nesting restriction)
#'   or `"delta"` (lavaan-style covariance/moment Jacobian column-space
#'   restriction). FIML pairs with nonlinear equality constraints use the local
#'   tangent restriction map even when `"exact"` is requested, because no global
#'   affine exact map exists. For lavaan's default FIML
#'   `lavTestLRT(method = "satorra.2000")` comparison, use `"delta"`.
#' @param computation `"streaming"` (default, projects casewise empirical
#'   Gamma contributions before crossproducts), `"materialized"` (forms the
#'   full empirical Gamma block before reducing), or `"dense"` (forms the full
#'   q-by-q Gamma and U matrices and eigendecomposes the q-by-q product, as
#'   standard SEM software does). FIML lavaan-convention tests use the same
#'   choices after projecting through lavaan-style `WLS.V`; `"materialized"`
#'   keeps the legacy sandwich path and `"dense"` is a diagnostic oracle.
#' @param convention `"magmaan"` (default) keeps magmaan's estimator-specific
#'   Satorra-2000 moment convention. `"lavaan"` uses lavaan's public
#'   `lavTestLRT(method = "satorra.2000")` convention for FIML/ML2S missing-data
#'   pairs: model-based raw-moment `Gamma`, lavaan-style per-group `WLS.V`
#'   weighting, and `"delta"` as the default `A.method` when the caller does not
#'   specify one.
#' @param ud_method Estimator of the U_D difference matrix whose `U_D * Gamma`
#'   eigenvalues drive the scaled statistic: `"2000"` (default, Satorra 2000
#'   restriction map from the H1 fit) or `"2001"` (Satorra-Bentler 2001
#'   `U_D = U0 - U1`, the difference of the two single-model projectors;
#'   `semTests::ugamma_nested(., "2001")`). `"2001"` is implemented for FIML and
#'   ML2S fits only and, unlike `"2000"`, accepts non-`==`-constrained nesting
#'   (different parameter counts); its spectrum can carry negative eigenvalues.
#' @param h1_reference_regularization FIML restriction-map tests only. `NULL`
#'   or `FALSE` keeps the raw saturated-H1 reference. `TRUE` applies the
#'   frontier condition-cap defaults. A list may set `condition_max`,
#'   `min_eigenvalue`, and optional `covariance` / `information` sublists.
#' @param weight For ordinal nested tests, `"DWLS"`, `"WLS"`, or `"ULS"`.
#'   For continuous WLS, the explicit fitting weight matrix (or per-group list
#'   of matrices). Continuous ULS/GLS rebuild their own weights and reject this
#'   argument.
#'
#' @return A list of class `magmaan_nested_test`. With `gamma = "both"`,
#'   `eigenvalues` and the scalar test summaries use empirical Gamma, while
#'   `eigenvalues_unbiased` contains the auxiliary unbiased spectrum.
#' @export
robust_nested_lrt <- function(fit_H1, fit_H0, data = NULL,
                              gamma = c("empirical", "unbiased", "NT", "both"),
                              method = c("restriction_map",
                                         "lavaan_sb2001",
                                         "lavaan_sb2010"),
                              A.method = c("exact", "delta"),
                              computation = c("streaming", "materialized",
                                              "dense"),
                              convention = c("magmaan", "lavaan"),
                              ud_method = c("2000", "2001"),
                              h1_reference_regularization = NULL,
                              weight = NULL) {
  gamma <- match.arg(gamma)
  method <- match.arg(method)
  convention <- match.arg(convention)
  if (missing(A.method) && identical(convention, "lavaan")) A.method <- "delta"
  A.method <- match.arg(A.method)
  computation <- match.arg(computation)
  ud_method <- match.arg(ud_method)
  h1_ref_requested <- !is.null(h1_reference_regularization) &&
    !identical(h1_reference_regularization, FALSE)

  fit_estimator <- function(fit) {
    toupper(as.character(fit$estimator %||% fit$options$estimator %||% ""))
  }
  is_ml2s <- function(fit) {
    grepl("^ML2S(_|$)", fit_estimator(fit))
  }
  is_fiml <- function(fit) {
    !is_ml2s(fit) &&
      (isTRUE(fit$fiml) || identical(fit_estimator(fit), "FIML") ||
         inherits(fit$raw_data, "magmaan_fiml_data"))
  }
  ml2s_stage2_weight <- function(fit) {
    w <- fit$stage2_weight
    if (is.null(w)) {
      est <- fit_estimator(fit)
      w <- if (identical(est, "ML2S")) "nt" else sub("^ML2S_", "", est)
    }
    w <- tolower(as.character(w)[1L])
    if (identical(w, "wls")) w <- "adf"
    if (!w %in% c("nt", "uls", "dwls", "adf", "dls")) {
      stop("robust_nested_lrt(): unknown ML2S stage2_weight '", w,
           "' (expected nt, uls, dwls, adf/wls, or dls).", call. = FALSE)
    }
    w
  }
  ml2s_H1 <- is_ml2s(fit_H1)
  ml2s_H0 <- is_ml2s(fit_H0)
  if (xor(ml2s_H1, ml2s_H0)) {
    stop("robust_nested_lrt(): mixed ML2S/non-ML2S model pairs are not ",
         "supported; fit both models with the same estimator.", call. = FALSE)
  }
  fiml_H1 <- is_fiml(fit_H1)
  fiml_H0 <- is_fiml(fit_H0)
  if (xor(fiml_H1, fiml_H0)) {
    stop("robust_nested_lrt(): mixed FIML/complete-data model pairs are not ",
         "supported; fit both models with the same estimator.", call. = FALSE)
  }
  ordinal_H1 <- isTRUE(fit_H1$ordinal)
  ordinal_H0 <- isTRUE(fit_H0$ordinal)
  mixed_ordinal_H1 <- isTRUE(fit_H1$mixed_ordinal)
  mixed_ordinal_H0 <- isTRUE(fit_H0$mixed_ordinal)
  if (xor(ordinal_H1, ordinal_H0) || xor(mixed_ordinal_H1, mixed_ordinal_H0)) {
    stop("robust_nested_lrt(): mixed ordinal/non-ordinal model pairs are not ",
         "supported; fit both models with the same estimator.", call. = FALSE)
  }
  estimator_H1 <- fit_estimator(fit_H1)
  estimator_H0 <- fit_estimator(fit_H0)
  # ML2S estimator labels encode the Stage-2 weight and are validated by the
  # dedicated branch below; keep its more specific same-weight diagnostic.
  if (!ml2s_H1 && !identical(estimator_H1, estimator_H0)) {
    stop("robust_nested_lrt(): fit_H1 and fit_H0 use different estimators ('",
         estimator_H1, "' versus '", estimator_H0, "').", call. = FALSE)
  }
  if (identical(gamma, "unbiased") &&
      (ml2s_H1 || fiml_H1 || ordinal_H1 || mixed_ordinal_H1 ||
       !identical(estimator_H1, "ML"))) {
    stop("robust_nested_lrt(): gamma = 'unbiased' is available only for ",
         "complete-data normal-theory ML pairs.", call. = FALSE)
  }
  if (identical(gamma, "both") &&
      (ml2s_H1 || fiml_H1 || ordinal_H1 || mixed_ordinal_H1 ||
       !identical(estimator_H1, "ML") ||
       !identical(method, "restriction_map") ||
       !identical(computation, "streaming"))) {
    stop("robust_nested_lrt(): gamma = 'both' is available only for ",
         "complete-data normal-theory ML restriction-map pairs with ",
         "computation = 'streaming'.", call. = FALSE)
  }
  if (ml2s_H1) {
    if (h1_ref_requested) {
      stop("robust_nested_lrt(): `h1_reference_regularization` is only ",
           "available for direct FIML nested tests; use ML2S ",
           "`stage1_regularization` at fit time.", call. = FALSE)
    }
    if (!is.null(data)) {
      stop("robust_nested_lrt(): ML2S nested tests use fit_H1$raw_data; ",
           "the `data` argument is not supported for ML2S fits.",
           call. = FALSE)
    }
    stage2_H1 <- ml2s_stage2_weight(fit_H1)
    stage2_H0 <- ml2s_stage2_weight(fit_H0)
    if (!identical(stage2_H1, stage2_H0)) {
      stop("robust_nested_lrt(): ML2S model pairs must use the same ",
           "stage2_weight (H1 = '", stage2_H1, "', H0 = '", stage2_H0, "').",
           call. = FALSE)
    }
    dls_H1 <- fit_H1$stage2_dls_a %||% fit_H1$dls_a %||% 0.5
    dls_H0 <- fit_H0$stage2_dls_a %||% fit_H0$dls_a %||% 0.5
    if (identical(stage2_H1, "dls") &&
        !isTRUE(all.equal(as.numeric(dls_H1), as.numeric(dls_H0),
                          tolerance = 1e-12))) {
      stop("robust_nested_lrt(): ML2S DLS model pairs must use the same ",
           "dls_a (H1 = ", dls_H1, ", H0 = ", dls_H0, ").",
           call. = FALSE)
    }
    if (!identical(method, "restriction_map") && !identical(stage2_H1, "nt")) {
      stop("robust_nested_lrt(): ML2S method = '", method, "' is currently ",
           "NT-only; use method = 'restriction_map' for stage2_weight = '",
           stage2_H1, "'.", call. = FALSE)
    }
    res <- switch(method,
      restriction_map = infer_ml2s_lr_test_satorra2000(
        fit_H1, fit_H0, gamma = gamma, a_method = A.method, ud_method = ud_method,
        stage2_weight = stage2_H1, dls_a = as.numeric(dls_H1),
        convention = convention),
      lavaan_sb2001 = infer_ml2s_lr_test_satorra_bentler2001(fit_H1, fit_H0),
      lavaan_sb2010 = infer_ml2s_lr_test_satorra_bentler2010(fit_H1, fit_H0),
      stop("robust_nested_lrt(): unsupported method '", method, "' for ML2S.",
           call. = FALSE))
    res$gamma <- gamma
    res$method <- method
    res$A.method <- A.method
    res$ud_method <- ud_method
    res$convention <- convention
    res$computation <- if (identical(method, "restriction_map") &&
                           identical(ud_method, "2001")) "ml2s_eta_2001"
                       else if (identical(convention, "lavaan")) "ml2s_lavaan"
                       else "ml2s_eta"
    res$stage2_weight <- stage2_H1
    if (identical(stage2_H1, "dls")) res$dls_a <- as.numeric(dls_H1)
    class(res) <- c("magmaan_nested_test", "list")
    return(res)
  }
  if (fiml_H1) {
    if (h1_ref_requested && !identical(method, "restriction_map")) {
      stop("robust_nested_lrt(): `h1_reference_regularization` is only ",
           "available for FIML method = 'restriction_map'.", call. = FALSE)
    }
    if (!is.null(data)) {
      stop("robust_nested_lrt(): FIML nested tests use fit_H1$raw_data; ",
           "the `data` argument is not supported for FIML fits.",
           call. = FALSE)
    }
    res <- switch(method,
      restriction_map = infer_fiml_lr_test_satorra2000(
        fit_H1, fit_H0, gamma = gamma, a_method = A.method, ud_method = ud_method,
        convention = convention, computation = computation,
        h1_reference_regularization = h1_reference_regularization),
      lavaan_sb2001 = infer_fiml_lr_test_satorra_bentler2001(fit_H1, fit_H0),
      lavaan_sb2010 = infer_fiml_lr_test_satorra_bentler2010(fit_H1, fit_H0),
      stop("robust_nested_lrt(): unsupported method '", method, "' for FIML.",
           call. = FALSE))
    res$gamma <- gamma
    res$method <- method
    res$A.method <- A.method
    res$ud_method <- ud_method
    res$convention <- convention
    res$computation <- if (identical(method, "restriction_map") &&
                           identical(ud_method, "2001")) "fiml_eta_2001"
                       else if (identical(convention, "lavaan")) {
                         paste0("fiml_lavaan_", computation)
                       }
                       else "fiml_eta"
    class(res) <- c("magmaan_nested_test", "list")
    return(res)
  }
  if (h1_ref_requested) {
    stop("robust_nested_lrt(): `h1_reference_regularization` is only ",
         "available for direct FIML nested tests.", call. = FALSE)
  }
  if (identical(gamma, "unbiased") &&
      !identical(method, "restriction_map")) {
    stop("robust_nested_lrt(): gamma = 'unbiased' is available only for ",
         "method = 'restriction_map'.", call. = FALSE)
  }
  if (identical(ud_method, "2001")) {
    stop("robust_nested_lrt(): ud_method = '2001' (the U0-U1 difference ",
         "spectrum) is currently implemented for FIML and ML2S fits only; ",
         "complete-data and ordinal pairs use the Satorra-2000 restriction map.",
         call. = FALSE)
  }
  if (mixed_ordinal_H1) {
    if (!identical(method, "restriction_map")) {
      stop("robust_nested_lrt(): mixed-ordinal nested tests support ",
           "method = 'restriction_map' only.", call. = FALSE)
    }
    if (!identical(gamma, "empirical")) {
      stop("robust_nested_lrt(): mixed-ordinal nested tests use the ",
           "polyserial/polychoric NACOV Gamma from `data`; gamma = 'NT' is ",
           "not defined.", call. = FALSE)
    }
    if (is.null(data)) {
      stop("robust_nested_lrt(): mixed-ordinal nested tests require `data` ",
           "to be the magmaan_mixed_ordinal_data object used for fitting.",
           call. = FALSE)
    }
    if (!inherits(data, "magmaan_mixed_ordinal_data")) {
      stop("robust_nested_lrt(): mixed-ordinal nested tests require `data` ",
           "to inherit from 'magmaan_mixed_ordinal_data'.", call. = FALSE)
    }
    T_H1 <- infer_chi2_stat(fit_sample_stats(fit_H1), fit_H1$fmin)
    T_H0 <- infer_chi2_stat(fit_sample_stats(fit_H0), fit_H0$fmin)
    resolved_weight <- weight %||% fit_H1$estimator %||% ""
    res <- infer_mixed_ordinal_lr_test_satorra2000(
      fit_H1, fit_H0, data,
      T_H1 = T_H1, df_H1 = 0L,
      T_H0 = T_H0, df_H0 = 0L,
      weight = resolved_weight,
      a_method = A.method)
    res$gamma <- gamma
    res$method <- method
    res$A.method <- A.method
    res$convention <- convention
    res$computation <- "mixed_ordinal_moment"
    res$weight <- resolved_weight
    class(res) <- c("magmaan_nested_test", "list")
    return(res)
  }
  if (ordinal_H1) {
    if (!identical(method, "restriction_map")) {
      stop("robust_nested_lrt(): ordinal nested tests support ",
           "method = 'restriction_map' only.", call. = FALSE)
    }
    if (!identical(gamma, "empirical")) {
      stop("robust_nested_lrt(): ordinal nested tests use the polychoric NACOV ",
           "Gamma from `data`; gamma = 'NT' is not defined.", call. = FALSE)
    }
    if (is.null(data)) {
      stop("robust_nested_lrt(): ordinal nested tests require `data` to be the ",
           "magmaan_ordinal_data object used for fitting.", call. = FALSE)
    }
    T_H1 <- infer_chi2_stat(fit_sample_stats(fit_H1), fit_H1$fmin)
    T_H0 <- infer_chi2_stat(fit_sample_stats(fit_H0), fit_H0$fmin)
    resolved_weight <- weight %||% fit_H1$estimator %||% ""
    res <- infer_ordinal_lr_test_satorra2000(
      fit_H1, fit_H0, data,
      T_H1 = T_H1, df_H1 = 0L,
      T_H0 = T_H0, df_H0 = 0L,
      weight = resolved_weight,
      a_method = A.method)
    res$gamma <- gamma
    res$method <- method
    res$A.method <- A.method
    res$convention <- convention
    res$computation <- "ordinal_moment"
    res$weight <- resolved_weight
    class(res) <- c("magmaan_nested_test", "list")
    return(res)
  }

  if (is.null(data)) {
    stop("robust_nested_lrt(): complete-data nested tests require `data`.",
         call. = FALSE)
  }

  ngroups <- fit_H1$ngroups
  if (is.null(ngroups) || ngroups < 1) ngroups <- 1L

  ov <- if (is.list(fit_H1$ov_names)) fit_H1$ov_names[[1]] else fit_H1$ov_names

  X_per_group <- if (is.list(data) && !is.data.frame(data)) {
    if (length(data) != ngroups) {
      stop("robust_nested_lrt(): `data` is a list of length ", length(data),
           " but fit_H1 has ", ngroups, " group(s); pass per-group raw ",
           "data in the same order as fit_H1$S.")
    }
    lapply(data, as.matrix)
  } else if (is.data.frame(data) || is.matrix(data)) {
    if (is.null(ov)) stop("robust_nested_lrt(): fit_H1 is missing $ov_names")
    miss <- setdiff(ov, colnames(data))
    if (length(miss)) {
      stop("robust_nested_lrt(): data is missing observed variables: ",
           paste(miss, collapse = ", "))
    }
    if (ngroups > 1L) {
      stop("robust_nested_lrt(): fit_H1 is multi-group (ngroups = ", ngroups,
           "); pass `data` as a list of per-group matrices.")
    }
    list(as.matrix(data[, ov, drop = FALSE]))
  } else {
    stop("robust_nested_lrt(): `data` must be a data.frame, matrix, or list ",
         "of per-group matrices")
  }

  T_H1 <- infer_chi2_stat(fit_sample_stats(fit_H1), fit_H1$fmin)
  df_H1 <- infer_df_stat(fit_H1$partable, fit_sample_stats(fit_H1))
  T_H0 <- infer_chi2_stat(fit_sample_stats(fit_H0), fit_H0$fmin)
  df_H0 <- infer_df_stat(fit_H0$partable, fit_sample_stats(fit_H0))

  if (estimator_H1 %in% c("ULS", "GLS", "WLS")) {
    if (!identical(method, "restriction_map")) {
      stop("robust_nested_lrt(): continuous ULS/GLS/WLS nested tests support ",
           "method = 'restriction_map' only.", call. = FALSE)
    }
    if (identical(gamma, "unbiased")) {
      stop("robust_nested_lrt(): gamma = 'unbiased' is available only for ",
           "complete-data normal-theory ML pairs.", call. = FALSE)
    }
    res <- infer_continuous_ls_lr_test_satorra2000(
      fit_H1, fit_H0, X_per_group,
      T_H1 = T_H1, df_H1 = df_H1,
      T_H0 = T_H0, df_H0 = df_H0,
      weight = weight, gamma = gamma, a_method = A.method
    )
    res$gamma <- gamma
    res$method <- method
    res$A.method <- A.method
    res$convention <- convention
    res$computation <- "continuous_ls_sandwich"
    res$weight <- estimator_H1
    class(res) <- c("magmaan_nested_test", "list")
    return(res)
  }

  res <- switch(method,
    restriction_map = infer_lr_test_satorra2000(
        fit_H1, fit_H0, X_per_group,
        T_H1 = T_H1, df_H1 = df_H1,
        T_H0 = T_H0, df_H0 = df_H0,
        gamma = gamma,
        a_method = A.method,
        computation = computation),
    lavaan_sb2001 = infer_lr_test_satorra_bentler2001(
        fit_H1, fit_H0, X_per_group,
        T_H1 = T_H1, df_H1 = df_H1,
        T_H0 = T_H0, df_H0 = df_H0,
        gamma = gamma),
    lavaan_sb2010 = infer_lr_test_satorra_bentler2010(
        fit_H1, fit_H0, X_per_group,
        T_H1 = T_H1, df_H1 = df_H1,
        T_H0 = T_H0, df_H0 = df_H0,
        gamma = gamma))
  res$gamma <- gamma
  res$method <- method
  res$A.method <- A.method
  res$convention <- convention
  res$computation <- computation
  class(res) <- c("magmaan_nested_test", "list")
  res
}

#' @rdname robust_nested_lrt
#' @param method Historical lavaan labels (`"satorra.2000"`,
#'   `"satorra.bentler.2001"`, `"satorra.bentler.2010"`) or the canonical
#'   robust nested-LRT method names.
#' @export
nestedTest <- function(fit_H1, fit_H0, data = NULL,
                       gamma = c("empirical", "unbiased", "NT", "both"),
                       method = c("satorra.2000",
                                  "satorra.bentler.2001",
                                  "satorra.bentler.2010",
                                  "restriction_map",
                                  "lavaan_sb2001",
                                  "lavaan_sb2010"),
                       A.method = c("exact", "delta"),
                       computation = c("streaming", "materialized",
                                       "dense"),
                       convention = c("magmaan", "lavaan"),
                       ud_method = c("2000", "2001"),
                       h1_reference_regularization = NULL,
                       weight = NULL) {
  method <- match.arg(method)
  computation <- match.arg(computation)
  convention <- match.arg(convention)
  if (missing(A.method) && identical(convention, "lavaan")) A.method <- "delta"
  ud_method <- match.arg(ud_method)
  if (.is_noniterative(fit_H1) || .is_noniterative(fit_H0)) {
    .guard_noniterative("nestedTest()")
  }
  canonical <- switch(method,
    satorra.2000 = "restriction_map",
    satorra.bentler.2001 = "lavaan_sb2001",
    satorra.bentler.2010 = "lavaan_sb2010",
    method)
  out <- robust_nested_lrt(fit_H1, fit_H0, data, gamma = gamma,
                           method = canonical, A.method = A.method,
                           computation = computation, convention = convention,
                           ud_method = ud_method,
                           h1_reference_regularization = h1_reference_regularization,
                           weight = weight)
  out$compat_method <- method
  out
}

#' @export
print.magmaan_nested_test <- function(x, digits = 4L, ...) {
  method <- x$method %||% "restriction_map"
  comp <- x$computation %||% "streaming"
  cat(sprintf("Scaled nested-model chi-square difference test (method = %s, Gamma = %s, computation = %s)\n\n",
              method, x$gamma, comp))
  if (identical(method, "restriction_map")) {
    rows <- data.frame(
        stat = c(x$T_diff, x$T_scaled, x$T_adjusted,
                 x$scaled_shifted$chi2_adj, x$T_diff),
        df = c(x$df_diff, x$df_diff, x$adjust_d0,
               x$scaled_shifted$df, x$df_diff),
        pval = c(x$p_unscaled, x$p_scaled, x$p_adjusted,
                 x$scaled_shifted$pvalue, x$p_mixture))
    rownames(rows) <- c("Unscaled chi-square",
                        "Scaled (Satorra-Bentler)",
                        "Mean+var adjusted",
                        "Scaled+shifted",
                        "Exact mixture (Imhof)")
  } else {
    rows <- data.frame(stat = x$T_scaled, df = x$df_diff, pval = x$p_value)
    rownames(rows) <- "Scaled difference"
  }
  print(format(rows, digits = digits))
  if (identical(method, "restriction_map")) {
    if (identical(x$gamma, "both")) {
      cat("\nEmpirical eigenvalues:",
          paste(format(x$eigenvalues, digits = digits), collapse = ", "),
          "\nUnbiased eigenvalues:",
          paste(format(x$eigenvalues_unbiased, digits = digits),
                collapse = ", "),
          "\nReported scalar statistics use empirical Gamma.")
    } else {
      cat("\nEigenvalues:",
          paste(format(x$eigenvalues, digits = digits), collapse = ", "))
    }
    cat("\nScale c:", format(x$scale_c, digits = digits),
        "  Adjust d0:", format(x$adjust_d0, digits = digits), "\n")
  } else {
    cat("\nScale c:", format(x$scale_c, digits = digits),
        "  c_H0:", format(x$c_H0, digits = digits),
        "  c_H1/M10:", format(x$c_H1, digits = digits), "\n")
  }
  if (length(x$warnings)) {
    cat("\nWarnings:\n  -", paste(x$warnings, collapse = "\n  - "), "\n")
  }
  invisible(x)
}
