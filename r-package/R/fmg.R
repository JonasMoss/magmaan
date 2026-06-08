# Foldnes-Moss-Gronneberg single-model goodness-of-fit tests.
#
# The R side owns the semTests-style test-name parser and fit-object plumbing.
# C++ owns the UGamma spectra and eigenvalue-tail transforms. fmg_tests() is the
# diagnostic, first-class post-fit result; fmg_pvalues() is the historical named
# p-value vector view over the same implementation.

.fmg_default_tests <- function() {
  c("SB_UG_RLS", "pEBA2_UG_RLS", "pEBA4_RLS", "pEBA6_RLS", "pOLS_RLS")
}

# Parse one semTests-style test name into its components.
#   grammar: <type>[_ug][_ml|_rls]   (case-insensitive)
#     type : std | sb | ss | sf | all | pall | eba<j> | peba<j> | pols<gamma>
#     _ug  : unbiased Gamma-hat (Du-Bentler); absent => biased Gamma-hat
#     _ml | _rls : base statistic (default rls, as in semTests::split_input)
.fmg_parse_test <- function(name) {
  if (!is.character(name) || length(name) != 1L || is.na(name) || !nzchar(name)) {
    stop("fmg_tests(): test names must be non-empty strings.", call. = FALSE)
  }
  toks <- strsplit(tolower(name), "_", fixed = TRUE)[[1]]
  type <- toks[1]
  rest <- toks[-1]
  allowed_rest <- c("ug", "ml", "rls")
  bad <- setdiff(rest, allowed_rest)
  if (length(bad)) {
    stop("fmg_tests(): unrecognized FMG test suffix in '", name, "': ",
         paste(bad, collapse = ", "), call. = FALSE)
  }
  if (sum(rest == "ml") + sum(rest == "rls") > 1L) {
    stop("fmg_tests(): test name '", name, "' cannot request both ML and RLS.",
         call. = FALSE)
  }
  if (sum(rest == "ug") > 1L) {
    stop("fmg_tests(): repeated UG suffix in test name '", name, "'.",
         call. = FALSE)
  }
  ug <- "ug" %in% rest
  base <- if ("ml" %in% rest) "ml" else "rls"

  num <- function(s, default) {
    if (!nchar(s)) return(default)
    out <- suppressWarnings(as.numeric(s))
    if (!is.finite(out)) {
      stop("fmg_tests(): invalid numeric parameter in FMG test name '",
           name, "'.", call. = FALSE)
    }
    out
  }

  if (startsWith(type, "peba")) {
    method <- "peba"; param <- num(substring(type, 5L), 2)
    label  <- paste0("peba", param)
  } else if (startsWith(type, "pols")) {
    method <- "pols"; param <- num(substring(type, 5L), 2)
    label  <- paste0("pols", param)
  } else if (startsWith(type, "eba")) {
    method <- "eba"; param <- num(substring(type, 4L), 2)
    label  <- paste0("eba", param)
  } else if (type %in% c("std", "sb", "ss", "sf", "all", "pall")) {
    method <- c(std = "standard", sb = "sb", ss = "ss", sf = "scaled_f",
                all = "all", pall = "penalized_all")[[type]]
    param <- NA_real_; label <- type
  } else {
    stop("fmg_tests(): unrecognized FMG test name '", name, "'.", call. = FALSE)
  }

  canonical <- paste0(c(label, if (ug) "ug", base), collapse = "_")
  list(input = name, method = method, param = param, ug = ug, base = base,
       canonical = canonical)
}

.fmg_fit_estimator <- function(fit) {
  toupper(as.character(fit$estimator %||% fit$options$estimator %||% ""))
}

.fmg_raw_from_fit_or_data <- function(fit, data = NULL, caller = "fmg_tests") {
  if (!is.null(data)) {
    if (inherits(data, "magmaan_fiml_data")) {
      stop(caller, "(): FMG requires complete raw data; FIML/missing-data ",
           "raw objects are not supported.", call. = FALSE)
    }
    if (inherits(data, "magmaan_data")) {
      return(data$X)
    }
    if (is.data.frame(data) || is.matrix(data)) {
      ov <- if (is.list(fit$ov_names)) fit$ov_names[[1L]] else fit$ov_names
      if (is.null(ov)) {
        stop(caller, "(): fit is missing $ov_names.", call. = FALSE)
      }
      miss <- setdiff(ov, colnames(data))
      if (length(miss)) {
        stop(caller, "(): data is missing observed variables: ",
             paste(miss, collapse = ", "), call. = FALSE)
      }
      X <- as.matrix(data[, ov, drop = FALSE])
      if (anyNA(X)) {
        stop(caller, "(): missing observed values are not supported by FMG; ",
             "use complete data or listwise-complete df_to_data() input.",
             call. = FALSE)
      }
      if (any(!is.finite(X))) {
        stop(caller, "(): non-finite observed values are not supported by FMG.",
             call. = FALSE)
      }
      return(X)
    }
    if (is.list(data) && !is.null(data$X)) return(data$X)
    return(data)
  }

  raw <- fit$raw_data
  if (inherits(raw, "magmaan_fiml_data")) {
    stop(caller, "(): FMG is not available for FIML/missing-data fits; ",
         "complete-data UGamma spectra are required.", call. = FALSE)
  }
  if (is.list(raw) && !is.null(raw$X)) return(raw$X)
  if (!is.null(raw)) return(raw)

  stop(caller, "(): complete raw data are required. Refit from a data.frame ",
       "or df_to_data() object, or pass `data =` explicitly.", call. = FALSE)
}

.fmg_validate_complete_raw <- function(fit, X, caller = "fmg_tests") {
  n_blocks <- fit$ngroups %||% length(fit$nobs) %||% 1L
  blocks <- if (is.matrix(X) || is.data.frame(X)) list(as.matrix(X)) else X
  if (!is.list(blocks)) {
    stop(caller, "(): raw data must be a matrix or list of matrices.",
         call. = FALSE)
  }
  if (length(blocks) != n_blocks) {
    stop(caller, "(): raw data have ", length(blocks), " block(s), but fit has ",
         n_blocks, " group(s).", call. = FALSE)
  }
  nobs <- as.integer(fit$nobs)
  for (i in seq_along(blocks)) {
    Xi <- as.matrix(blocks[[i]])
    if (anyNA(Xi)) {
      stop(caller, "(): missing observed values are not supported by FMG; ",
           "use complete data or listwise-complete df_to_data() input.",
           call. = FALSE)
    }
    if (any(!is.finite(Xi))) {
      stop(caller, "(): non-finite observed values are not supported by FMG.",
           call. = FALSE)
    }
    if (length(nobs) >= i && !is.na(nobs[[i]]) && nrow(Xi) != nobs[[i]]) {
      stop(caller, "(): raw data block ", i, " has ", nrow(Xi),
           " row(s), but fit$nobs reports ", nobs[[i]], ".", call. = FALSE)
    }
  }
  invisible(TRUE)
}

.fmg_param_for_cpp <- function(param) {
  if (is.na(param)) 4.0 else param
}

.fmg_result_rows <- function(fit, X, specs) {
  ss <- fit_sample_stats(fit)
  df <- infer_df_stat(fit$partable, ss)
  implied <- model_implied(fit)
  base_stat <- c(
    ml = infer_chi2_stat(ss, fit$fmin),
    rls = infer_rls_chi2_fit(fit, implied)$statistic
  )
  need_ug <- any(vapply(specs, `[[`, logical(1), "ug"))
  ev <- infer_fmg_ugamma_spectra(fit, X, need_ug)

  rows <- lapply(specs, function(s) {
    eigvals <- if (s$ug) ev$unbiased else ev$biased
    res <- infer_fmg_test(base_stat[[s$base]], df, eigvals,
                          method = s$method,
                          param = .fmg_param_for_cpp(s$param))
    list(input = s$input,
         label = s$canonical,
         p_value = res$p_value,
         df = res$df,
         base = s$base,
         base_statistic = res$chi2_source,
         method = res$method,
         param = if (is.na(s$param)) NA_real_ else res$param,
         ug = s$ug,
         chi2_equiv = res$chi2_equiv,
         n_truncated = res$n_truncated,
         eigenvalues = eigvals,
         lambdas_raw = res$lambdas_raw,
         lambdas = res$lambdas,
         lambdas_reference = res$lambdas_reference)
  })

  scalar <- data.frame(
    input = vapply(rows, `[[`, character(1), "input"),
    label = vapply(rows, `[[`, character(1), "label"),
    p_value = vapply(rows, `[[`, numeric(1), "p_value"),
    df = vapply(rows, `[[`, integer(1), "df"),
    base = vapply(rows, `[[`, character(1), "base"),
    base_statistic = vapply(rows, `[[`, numeric(1), "base_statistic"),
    method = vapply(rows, `[[`, character(1), "method"),
    param = vapply(rows, `[[`, numeric(1), "param"),
    ug = vapply(rows, `[[`, logical(1), "ug"),
    chi2_equiv = vapply(rows, `[[`, numeric(1), "chi2_equiv"),
    n_truncated = vapply(rows, `[[`, integer(1), "n_truncated"),
    stringsAsFactors = FALSE
  )
  scalar$eigenvalues <- I(lapply(rows, `[[`, "eigenvalues"))
  scalar$lambdas_raw <- I(lapply(rows, `[[`, "lambdas_raw"))
  scalar$lambdas <- I(lapply(rows, `[[`, "lambdas"))
  scalar$lambdas_reference <- I(lapply(rows, `[[`, "lambdas_reference"))
  class(scalar) <- c("magmaan_fmg_tests", "data.frame")
  scalar
}

#' Foldnes-Moss-Gronneberg goodness-of-fit diagnostics.
#'
#' Computes FMG p-values and diagnostics for a complete-data single-group ML
#' fit. The fit must carry complete raw data, as fits produced from
#' `magmaan(..., data.frame, estimator = "ML")` or `fit_ml(model,
#' df_to_data(...))` do, or callers can pass complete raw `data` explicitly.
#' FIML/missing-data fits and multi-group fits are currently rejected because
#' the fused FMG UGamma spectra path is single-group complete-data only.
#'
#' @param fit A fitted magmaan complete-data ML model.
#' @param tests Character vector of semTests-style test names. Recognised types:
#'   `std`, `sb`, `ss`, `sf`, `all`, `pall`, `eba<j>`, `peba<j>`, `pols<gamma>`, each
#'   optionally suffixed `_ug` and `_ml` / `_rls`.
#' @param data Optional complete raw data. Usually unnecessary for new fits that
#'   retain `$raw_data`; kept for sample-stat-only and compatibility workflows.
#'
#' @return A data frame with scalar diagnostics plus list-columns for the raw
#'   UGamma spectrum and the method-specific lambda vectors.
#' @export
fmg_tests <- function(fit, tests = .fmg_default_tests(), data = NULL) {
  ngroups <- fit$ngroups %||% 1L
  if (ngroups > 1L) {
    stop("fmg_tests(): only complete-data single-group fits are supported ",
         "(got ngroups = ", ngroups, ").", call. = FALSE)
  }
  estimator <- .fmg_fit_estimator(fit)
  if (!identical(estimator, "ML")) {
    stop("fmg_tests(): FMG currently requires a complete-data ML fit ",
         "(got estimator = '", estimator, "').", call. = FALSE)
  }
  if (isTRUE(fit$fiml) || inherits(fit$raw_data, "magmaan_fiml_data")) {
    stop("fmg_tests(): FMG is not available for FIML/missing-data fits; ",
         "complete-data UGamma spectra are required.", call. = FALSE)
  }
  specs <- lapply(tests, .fmg_parse_test)
  X <- .fmg_raw_from_fit_or_data(fit, data, caller = "fmg_tests")
  .fmg_validate_complete_raw(fit, X, caller = "fmg_tests")
  .fmg_result_rows(fit, X, specs)
}

#' Foldnes-Moss-Gronneberg goodness-of-fit p-values for a fitted magmaan model.
#'
#' Back-compatible named-vector view over `fmg_tests()`. New code that needs
#' diagnostics should call `fmg_tests()` or `fit_measures(..., fmg = tests)`.
#'
#' @param fit A fitted magmaan model.
#' @param data Optional complete raw data. If omitted, `fit$raw_data` is used.
#' @param tests Character vector of semTests-style test names. Recognised types:
#'   `std`, `sb`, `ss`, `sf`, `all`, `pall`, `eba<j>`, `peba<j>`, `pols<gamma>`, each
#'   optionally suffixed `_ug` (unbiased Gamma-hat) and `_ml` / `_rls` (base
#'   statistic; default `rls`).
#'
#' @return A named numeric vector of p-values; names are the canonical test
#'   labels (e.g. `peba4_rls`, `sb_ug_rls`), matching `semTests::pvalues()`.
#' @export
fmg_pvalues <- function(fit, data = NULL, tests = .fmg_default_tests()) {
  tab <- fmg_tests(fit, tests = tests, data = data)
  out <- tab$p_value
  names(out) <- tab$label
  out
}

#' Normal-theory fit measures, optionally with FMG robust p-values.
#'
#' Convenience wrapper over the existing fit-measure primitives. By default it
#' reports the standard ML chi-square, df, p-value, baseline chi-square, and
#' ordinary fit measures. Passing `fmg = TRUE` or a character vector of FMG test
#' labels adds an `$fmg` diagnostic table from `fmg_tests()`.
#'
#' @param fit A fitted magmaan model.
#' @param baseline Optional baseline result from `magmaan_core$measures_baseline()`.
#' @param fmg `NULL`/`FALSE` for no FMG table, `TRUE` for default FMG tests, or
#'   a character vector of FMG test names.
#' @param data Optional complete raw data used only when FMG is requested and
#'   the fit does not carry `$raw_data`.
#' @export
fit_measures <- function(fit, baseline = NULL, fmg = NULL, data = NULL) {
  ss <- fit_sample_stats(fit)
  chi2 <- infer_chi2_stat(ss, fit$fmin)
  df <- infer_df_stat(fit$partable, ss)
  if (is.null(baseline)) baseline <- measures_baseline(ss)
  fm <- measures_fit(fit, chi2, df, baseline)
  out <- c(list(
    chisq = chi2,
    df = df,
    pvalue = infer_chi2_pvalue(chi2, df),
    baseline.chisq = baseline$chi2,
    baseline.df = baseline$df
  ), fm)
  if (!is.null(fmg) && !identical(fmg, FALSE)) {
    tests <- if (isTRUE(fmg)) .fmg_default_tests() else fmg
    out$fmg <- fmg_tests(fit, tests = tests, data = data)
  }
  out
}
