#' Robust nested-model likelihood-ratio tests.
#'
#' Compares a restricted model `fit_H0` against the less-restricted superset
#' `fit_H1`. `method = "restriction_map"` uses Satorra's streaming low-rank
#' reduction of the asymptotic weighted-chi-square distribution. The
#' `"lavaan_sb2001"` and `"lavaan_sb2010"` options mirror lavaan's
#' compatibility difference-test approximations.
#'
#' @param fit_H1 Less-restricted fit from `magmaan::fit_fit()`.
#' @param fit_H0 More-restricted fit (same lavaanified partable shape,
#'   differing only in constraint rows / shared labels).
#' @param data Raw complete data for complete-data fits: either a data.frame
#'   whose columns include the observed variables of `fit_H1` (single-group
#'   case), or a list of per-group matrices in the same block order the fit was
#'   built with. Ordinal DWLS/WLS pairs take the `magmaan_ordinal_data` object
#'   used for fitting. FIML pairs use `fit_H1$raw_data` and do not accept `data`.
#' @param gamma `"empirical"` (default, empirical Gamma-hat) or `"NT"`
#'   (normal-theory sanity-check path where all eigenvalues collapse to 1).
#' @param method `"restriction_map"` (default), `"lavaan_sb2001"`, or
#'   `"lavaan_sb2010"`.
#' @param A.method `"exact"` (default, exact parameter-nesting restriction)
#'   or `"delta"` (lavaan-style covariance/moment Jacobian column-space
#'   restriction). FIML pairs with nonlinear equality constraints use the local
#'   tangent restriction map even when `"exact"` is requested, because no global
#'   affine exact map exists.
#' @param computation `"streaming"` (default, projects casewise empirical
#'   Gamma contributions before crossproducts), `"materialized"` (forms the
#'   full empirical Gamma block before reducing), or `"dense"` (forms the full
#'   q-by-q Gamma and U matrices and eigendecomposes the q-by-q product, as
#'   standard SEM software does). The latter two are diagnostic/reference paths
#'   for timing the algebraic reduction.
#' @param weight Ordinal nested tests only: `"DWLS"`, `"WLS"`, or `"ULS"`.
#'   Defaults to `fit_H1$estimator`.
#'
#' @return A list of class `magmaan_nested_test`.
#' @export
robust_nested_lrt <- function(fit_H1, fit_H0, data = NULL,
                              gamma = c("empirical", "NT"),
                              method = c("restriction_map",
                                         "lavaan_sb2001",
                                         "lavaan_sb2010"),
                              A.method = c("exact", "delta"),
                              computation = c("streaming", "materialized",
                                              "dense"),
                              weight = NULL) {
  gamma <- match.arg(gamma)
  method <- match.arg(method)
  A.method <- match.arg(A.method)
  computation <- match.arg(computation)

  fit_estimator <- function(fit) {
    toupper(as.character(fit$estimator %||% fit$options$estimator %||% ""))
  }
  is_fiml <- function(fit) {
    isTRUE(fit$fiml) || identical(fit_estimator(fit), "FIML") ||
      inherits(fit$raw_data, "magmaan_fiml_data")
  }
  is_ml2s <- function(fit) {
    identical(fit_estimator(fit), "ML2S")
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
  if (ml2s_H1) {
    if (!identical(method, "restriction_map")) {
      stop("robust_nested_lrt(): ML2S nested tests support ",
           "method = 'restriction_map' only; complete-data compatibility ",
           "methods are not defined for ML2S.", call. = FALSE)
    }
    if (!is.null(data)) {
      stop("robust_nested_lrt(): ML2S nested tests use fit_H1$raw_data; ",
           "the `data` argument is not supported for ML2S fits.",
           call. = FALSE)
    }
    res <- infer_ml2s_lr_test_satorra2000(
      fit_H1, fit_H0, gamma = gamma, a_method = A.method)
    res$gamma <- gamma
    res$method <- method
    res$A.method <- A.method
    res$computation <- "ml2s_eta"
    class(res) <- c("magmaan_nested_test", "list")
    return(res)
  }
  if (fiml_H1) {
    if (!identical(method, "restriction_map")) {
      stop("robust_nested_lrt(): FIML nested tests support ",
           "method = 'restriction_map' only; complete-data compatibility ",
           "methods are not defined for FIML.", call. = FALSE)
    }
    if (!is.null(data)) {
      stop("robust_nested_lrt(): FIML nested tests use fit_H1$raw_data; ",
           "the `data` argument is not supported for FIML fits.",
           call. = FALSE)
    }
    res <- infer_fiml_lr_test_satorra2000(
      fit_H1, fit_H0, gamma = gamma, a_method = A.method)
    res$gamma <- gamma
    res$method <- method
    res$A.method <- A.method
    res$computation <- "fiml_eta"
    class(res) <- c("magmaan_nested_test", "list")
    return(res)
  }
  if (mixed_ordinal_H1) {
    stop("robust_nested_lrt(): mixed-ordinal nested tests are not implemented; ",
         "use all-ordinal DWLS/WLS fits or complete-data/FIML fits.",
         call. = FALSE)
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
  res$computation <- computation
  class(res) <- c("magmaan_nested_test", "list")
  res
}

#' @rdname robust_nested_lrt
#' @param method Historical lavaan labels (`"satorra.2000"`,
#'   `"satorra.bentler.2001"`, `"satorra.bentler.2010"`) or the canonical
#'   robust nested-LRT method names.
#' @export
nestedTest <- function(fit_H1, fit_H0, data = NULL, gamma = c("empirical", "NT"),
                       method = c("satorra.2000",
                                  "satorra.bentler.2001",
                                  "satorra.bentler.2010",
                                  "restriction_map",
                                  "lavaan_sb2001",
                                  "lavaan_sb2010"),
                       A.method = c("exact", "delta"),
                       computation = c("streaming", "materialized",
                                       "dense"),
                       weight = NULL) {
  method <- match.arg(method)
  computation <- match.arg(computation)
  canonical <- switch(method,
    satorra.2000 = "restriction_map",
    satorra.bentler.2001 = "lavaan_sb2001",
    satorra.bentler.2010 = "lavaan_sb2010",
    method)
  out <- robust_nested_lrt(fit_H1, fit_H0, data, gamma = gamma,
                           method = canonical, A.method = A.method,
                           computation = computation, weight = weight)
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
    cat("\nEigenvalues:",
        paste(format(x$eigenvalues, digits = digits), collapse = ", "),
        "\nScale c:", format(x$scale_c, digits = digits),
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
