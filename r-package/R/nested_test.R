#' Scaled nested-model χ² difference tests.
#'
#' Compares a restricted model `fit_H0` against the less-restricted superset
#' `fit_H1`. `method = "satorra.2000"` uses Satorra's streaming low-rank
#' reduction of the asymptotic weighted-χ² distribution. The
#' `"satorra.bentler.2001"` and `"satorra.bentler.2010"` options mirror
#' lavaan's compatibility difference-test approximations.
#'
#' @param fit_H1   Less-restricted fit from `magmaan::fit_fit()`.
#' @param fit_H0   More-restricted fit (same lavaanified partable shape,
#'                 differing only in constraint rows / shared labels).
#' @param data     Raw data — either a data.frame whose columns include the
#'                 observed variables of `fit_H1` (single-group case), or a
#'                 *list* of per-group matrices (multi-group case), in the
#'                 same block order the fit was built with (i.e. the order
#'                 of `fit_H1$S`).  The list form matches how
#'                 `data_sample_stats_from_raw()` consumes data.
#' @param gamma    `"empirical"` (default — empirical Γ̂, matches lavaan's
#'                 `estimator = "MLR"` / `"MLM"`) or `"NT"` (normal-theory
#'                 sanity-check path where all eigenvalues collapse to 1).
#' @param method   `"satorra.2000"` (default), `"satorra.bentler.2001"`, or
#'                 `"satorra.bentler.2010"`.
#' @param A.method `"exact"` (default — exact parameter-nesting restriction)
#'                 or `"delta"` (lavaan-style covariance/moment Jacobian
#'                 column-space restriction).
#'
#' @return A list of class `magmaan_nested_test` with fields
#' \describe{
#'   \item{T_diff, df_diff, p_unscaled}{Naïve χ²(m) difference.}
#'   \item{eigenvalues}{The m generalised eigenvalues λⱼ.}
#'   \item{scale_c, T_scaled, p_scaled}{Satorra-Bentler scaled test.}
#'   \item{adjust_d0, T_adjusted, p_adjusted}{Mean-and-variance adjusted test.}
#'   \item{scaled_shifted}{Scaled-and-shifted test details.}
#'   \item{p_mixture}{Exact Pr(Σⱼ λⱼ χ²₁,ⱼ > T_diff) via Imhof.}
#'   \item{warnings}{Any numerical warnings from the C++ core.}
#' }
#' @export
nestedTest <- function(fit_H1, fit_H0, data, gamma = c("empirical", "NT"),
                       method = c("satorra.2000",
                                  "satorra.bentler.2001",
                                  "satorra.bentler.2010"),
                       A.method = c("exact", "delta")) {
  gamma <- match.arg(gamma)
  method <- match.arg(method)
  A.method <- match.arg(A.method)

  ngroups <- fit_H1$ngroups
  if (is.null(ngroups) || ngroups < 1) ngroups <- 1L

  ## Observed-variable order for the single-group data.frame path.
  ov <- if (is.list(fit_H1$ov_names)) fit_H1$ov_names[[1]] else fit_H1$ov_names

  X_per_group <- if (is.list(data) && !is.data.frame(data)) {
    ## Multi-group: list of per-group matrices in the fit's block order.
    if (length(data) != ngroups) {
      stop("nestedTest(): `data` is a list of length ", length(data),
           " but fit_H1 has ", ngroups, " group(s) — pass per-group raw ",
           "data in the same order as fit_H1$S (the same shape as ",
           "`data_sample_stats_from_raw()` consumes).")
    }
    lapply(data, as.matrix)
  } else if (is.data.frame(data) || is.matrix(data)) {
    ## Single-group: pick out the model variables.
    if (is.null(ov)) stop("nestedTest(): fit_H1 is missing $ov_names")
    miss <- setdiff(ov, colnames(data))
    if (length(miss)) {
      stop("nestedTest(): data is missing observed variables: ",
           paste(miss, collapse = ", "))
    }
    if (ngroups > 1L) {
      stop("nestedTest(): fit_H1 is multi-group (ngroups = ", ngroups,
           ") — pass `data` as a list of per-group matrices, not a ",
           "flat data.frame.")
    }
    list(as.matrix(data[, ov, drop = FALSE]))
  } else {
    stop("nestedTest(): `data` must be a data.frame, matrix, or list of ",
         "per-group matrices")
  }

  ## χ² and df are properties of the fit / model — no information matrix
  ## involved. Each is one function call away.
  T_H1  <- infer_chi2_stat(fit_sample_stats(fit_H1), fit_H1$fmin)
  df_H1 <- infer_df_stat(fit_H1$partable, fit_sample_stats(fit_H1))
  T_H0  <- infer_chi2_stat(fit_sample_stats(fit_H0), fit_H0$fmin)
  df_H0 <- infer_df_stat(fit_H0$partable, fit_sample_stats(fit_H0))

  res <- switch(method,
    satorra.2000 = infer_lr_test_satorra2000(
        fit_H1, fit_H0, X_per_group,
        T_H1 = T_H1, df_H1 = df_H1,
        T_H0 = T_H0, df_H0 = df_H0,
        gamma = gamma,
        a_method = A.method),
    satorra.bentler.2001 = infer_lr_test_satorra_bentler2001(
        fit_H1, fit_H0, X_per_group,
        T_H1 = T_H1, df_H1 = df_H1,
        T_H0 = T_H0, df_H0 = df_H0,
        gamma = gamma),
    satorra.bentler.2010 = infer_lr_test_satorra_bentler2010(
        fit_H1, fit_H0, X_per_group,
        T_H1 = T_H1, df_H1 = df_H1,
        T_H0 = T_H0, df_H0 = df_H0,
        gamma = gamma))
  res$gamma <- gamma
  res$method <- method
  res$A.method <- A.method
  class(res) <- c("magmaan_nested_test", "list")
  res
}

#' @export
print.magmaan_nested_test <- function(x, digits = 4L, ...) {
  method <- x$method %||% "satorra.2000"
  cat(sprintf("Scaled nested-model χ² difference test (method = %s, Γ = %s)\n\n",
              method, x$gamma))
  if (identical(method, "satorra.2000")) {
    rows <- data.frame(
        stat = c(x$T_diff, x$T_scaled, x$T_adjusted,
                 x$scaled_shifted$chi2_adj, x$T_diff),
        df   = c(x$df_diff, x$df_diff, x$adjust_d0,
                 x$scaled_shifted$df, x$df_diff),
        pval = c(x$p_unscaled, x$p_scaled, x$p_adjusted,
                 x$scaled_shifted$pvalue, x$p_mixture))
    rownames(rows) <- c("Unscaled χ²",
                        "Scaled (Satorra-Bentler)",
                        "Mean+var adjusted",
                        "Scaled+shifted",
                        "Exact mixture (Imhof)")
  } else {
    rows <- data.frame(stat = x$T_scaled, df = x$df_diff, pval = x$p_value)
    rownames(rows) <- "Scaled difference"
  }
  print(format(rows, digits = digits))
  if (identical(method, "satorra.2000")) {
    cat("\nEigenvalues λ:",
        paste(format(x$eigenvalues, digits = digits), collapse = ", "),
        "\nScale ĉ:", format(x$scale_c, digits = digits),
        "  Adjust d̂₀:", format(x$adjust_d0, digits = digits), "\n")
  } else {
    cat("\nScale ĉ:", format(x$scale_c, digits = digits),
        "  c_H0:", format(x$c_H0, digits = digits),
        "  c_H1/M10:", format(x$c_H1, digits = digits), "\n")
  }
  if (length(x$warnings)) {
    cat("\nWarnings:\n  -", paste(x$warnings, collapse = "\n  - "), "\n")
  }
  invisible(x)
}
