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
#     type : std | sb | ss | mv | sf | all | pall | eba<j> | peba<j> | pols<gamma>
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
  } else if (type %in% c("std", "sb", "ss", "mv", "sf", "all", "pall")) {
    method <- c(std = "standard", sb = "sb", ss = "ss", mv = "mean_var_adjusted",
                sf = "scaled_f", all = "all", pall = "penalized_all")[[type]]
    param <- NA_real_; label <- type
  } else {
    stop("fmg_tests(): unrecognized FMG test name '", name, "'.", call. = FALSE)
  }

  canonical <- paste0(c(label, if (ug) "ug", base), collapse = "_")
  list(input = name, method = method, param = param, ug = ug, base = base,
       label = label, base_explicit = ("ml" %in% rest) || ("rls" %in% rest),
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
      ngroups <- fit$ngroups %||% length(fit$nobs) %||% 1L
      ov_list <- if (is.list(fit$ov_names)) fit$ov_names else list(fit$ov_names)
      if (ngroups > 1L) {
        if (!is.data.frame(data)) {
          stop(caller, "(): multi-group raw data must be a list of per-group ",
               "matrices, a magmaan_data object, or a data.frame carrying the ",
               "fit's grouping column.", call. = FALSE)
        }
        group_var <- fit$group_var %||% ""
        group_labels <- fit$group_labels %||% character()
        if (!nzchar(group_var) || !group_var %in% names(data) ||
            length(group_labels) != ngroups) {
          stop(caller, "(): explicit multi-group data.frame input requires ",
               "`fit$group_var` and `fit$group_labels`; pass a list of ",
               "per-group matrices instead.", call. = FALSE)
        }
        return(lapply(seq_len(ngroups), function(g) {
          ov <- ov_list[[min(g, length(ov_list))]]
          if (is.null(ov)) stop(caller, "(): fit is missing $ov_names.",
                                call. = FALSE)
          rows <- as.character(data[[group_var]]) == group_labels[[g]]
          miss <- setdiff(ov, colnames(data))
          if (length(miss)) {
            stop(caller, "(): data is missing observed variables: ",
                 paste(miss, collapse = ", "), call. = FALSE)
          }
          X <- as.matrix(data[rows, ov, drop = FALSE])
          if (anyNA(X)) {
            stop(caller, "(): missing observed values are not supported by FMG; ",
                 "use complete data or listwise-complete df_to_data() input.",
                 call. = FALSE)
          }
          if (any(!is.finite(X))) {
            stop(caller, "(): non-finite observed values are not supported by FMG.",
                 call. = FALSE)
          }
          X
        }))
      } else {
        ov <- ov_list[[1L]]
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

  .fmg_rows_to_df(rows)
}

# Assemble the per-test row list into the `magmaan_fmg_tests` data.frame. Shared
# by the complete-data (`.fmg_result_rows`) and FIML (`.fmg_result_rows_fiml`)
# paths so the column layout stays identical.
.fmg_rows_to_df <- function(rows) {
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

# Is this a FIML / missing-data fit? FIML fits carry $fiml = TRUE, estimator
# "FIML", and a `magmaan_fiml_data` raw object.
.fmg_is_fiml <- function(fit) {
  isTRUE(fit$fiml) || identical(.fmg_fit_estimator(fit), "FIML") ||
    inherits(fit$raw_data, "magmaan_fiml_data")
}

.fmg_default_tests_fiml <- function() {
  c("pEBA4", "pEBA2", "pEBA6", "SB", "SS")
}

.fmg_resolve_default_tests <- function(fit, tests) {
  if (!is.null(tests)) return(tests)
  if (.fmg_is_ml2s(fit)) return(.fmg_default_tests_ml2s())
  if (.fmg_is_fiml(fit)) .fmg_default_tests_fiml() else .fmg_default_tests()
}

# Under FIML only the biased Gamma-hat and the ML (LRT) base statistic are
# defined; the Du-Bentler unbiased gamma and Browne's RLS statistic require the
# classical complete-data normal-theory ML case. An *explicit* `_rls`/`_ug` is an
# error; an unsuffixed (default) base resolves to ML, mirroring semTests' "auto".
.fmg_adjust_specs_fiml <- function(specs) {
  lapply(specs, function(s) {
    if (isTRUE(s$ug)) {
      stop("fmg_tests(): under FIML only the biased Gamma-hat is supported; the ",
           "unbiased Du-Bentler Gamma is undefined for missing data (test '",
           s$input, "').", call. = FALSE)
    }
    if (identical(s$base, "rls")) {
      if (isTRUE(s$base_explicit)) {
        stop("fmg_tests(): under FIML only the ML (LRT) base statistic is ",
             "supported; the RLS (browne.residual.nt.model) base requires the ",
             "classical complete-data normal-theory ML case (test '", s$input,
             "').", call. = FALSE)
      }
      s$base <- "ml"
      s$canonical <- sub("_rls$", "_ml", s$canonical)
    }
    s
  })
}

# FIML result rows: the spectrum comes from the first-principles missing-data
# UGamma path (`infer_fiml_fmg_spectrum`); the base statistic is the FIML LRT.
.fmg_result_rows_fiml <- function(fit, specs, h_step = 1e-4,
                                  h1_information = c("saturated",
                                                     "structured")) {
  h1_information <- match.arg(h1_information)
  sp <- infer_fiml_fmg_spectrum(fit, h_step, h1_information)
  df <- sp$df
  eigvals <- sp$biased
  rows <- lapply(specs, function(s) {
    res <- infer_fmg_test(sp$chi2_lrt, df, eigvals,
                          method = s$method,
                          param = .fmg_param_for_cpp(s$param))
    list(input = s$input,
         label = s$canonical,
         p_value = res$p_value,
         df = res$df,
         base = "ml",
         base_statistic = res$chi2_source,
         method = res$method,
         param = if (is.na(s$param)) NA_real_ else res$param,
         ug = FALSE,
         chi2_equiv = res$chi2_equiv,
         n_truncated = res$n_truncated,
         eigenvalues = eigvals,
         lambdas_raw = res$lambdas_raw,
         lambdas = res$lambdas,
         lambdas_reference = res$lambdas_reference)
  })
  out <- .fmg_rows_to_df(rows)
  attr(out, "trace_xcheck") <- sp$trace_xcheck
  attr(out, "h1_information") <- sp$h1_information
  out
}

# ── Two-stage ML (ML2S) ─────────────────────────────────────────────────────
# A two-stage missing-data fit estimates the saturated EM moments (Stage 1) and
# fits the structured model to them by complete-data ML (Stage 2). The reference
# law T_ML2 -> sum_j lambda_j chi^2_1 uses the same saturated-moment EM ACOV as
# the meat (Gamma_TS) but the complete-data normal-theory weight as the U-metric,
# with the Stage-2 ML chi-square as the base statistic. Both the df-dimensional
# UGamma spectrum and the base statistic are already attached to the fit as
# `fit$ml2s` (eigvals, chisq, df) by `fit_ml2s()` /
# `estimate_two_stage_em(kind = "ml")`, so an ML2S FMG test is just the
# estimator-agnostic eigenvalue-tail transform applied to that
# (chi-square, df, eigvals) triple --- exactly as on the ordinal and FIML paths.
#
# Like FIML, the EM ACOV is itself the asymptotic Gamma, so the Du-Bentler
# unbiased Gamma (`_ug`) is undefined; and the base is the Stage-2 ML statistic,
# so the complete-data RLS base (`_rls`) is rejected.

.fmg_is_ml2s <- function(fit) {
  identical(.fmg_fit_estimator(fit), "ML2S")
}

.fmg_default_tests_ml2s <- function() {
  c("pEBA4", "pEBA2", "pEBA6", "SB", "SS")
}

.fmg_adjust_specs_ml2s <- function(specs) {
  lapply(specs, function(s) {
    if (isTRUE(s$ug)) {
      stop("fmg_tests(): under ML2S only the biased Gamma-hat is supported; the ",
           "unbiased Du-Bentler Gamma is undefined for missing-data EM moments ",
           "(test '", s$input, "').", call. = FALSE)
    }
    if (identical(s$base, "rls")) {
      if (isTRUE(s$base_explicit)) {
        stop("fmg_tests(): under ML2S only the Stage-2 ML base statistic is ",
             "supported; the RLS (browne.residual.nt.model) base requires the ",
             "classical complete-data normal-theory ML case (test '", s$input,
             "').", call. = FALSE)
      }
      s$base <- "ml"
      s$canonical <- sub("_rls$", "_ml", s$canonical)
    }
    s
  })
}

# Pull the two-stage spectrum/base/df triple. Every `fit_ml2s()` fit carries it
# as `fit$ml2s`; recompute from the retained missing-data `$raw_data` only if it
# is absent.
.fmg_ml2s_spectrum <- function(fit, h_step = 1e-4) {
  sp <- fit$ml2s
  if (is.null(sp) || is.null(sp$eigvals) || is.null(sp$chisq) ||
      is.null(sp$df)) {
    raw <- fit$raw_data
    if (is.null(raw)) {
      stop("fmg_tests(): ML2S FMG requires the fit's two-stage inference ",
           "($ml2s) or its missing-data $raw_data to recompute it.",
           call. = FALSE)
    }
    sp <- estimate_two_stage_em_ml_inference(fit, raw, h_step)
  }
  sp
}

# ML2S result rows: the spectrum and the Stage-2 ML base chi-square come from the
# two-stage inference already attached to the fit. Mirrors `.fmg_result_rows_fiml`
# and `.fmg_result_rows_ordinal`.
.fmg_result_rows_ml2s <- function(fit, specs, h_step = 1e-4) {
  sp <- .fmg_ml2s_spectrum(fit, h_step)
  df <- sp$df
  eigvals <- sp$eigvals
  rows <- lapply(specs, function(s) {
    res <- infer_fmg_test(sp$chisq, df, eigvals,
                          method = s$method,
                          param = .fmg_param_for_cpp(s$param))
    list(input = s$input,
         label = s$canonical,
         p_value = res$p_value,
         df = res$df,
         base = "ml",
         base_statistic = res$chi2_source,
         method = res$method,
         param = if (is.na(s$param)) NA_real_ else res$param,
         ug = FALSE,
         chi2_equiv = res$chi2_equiv,
         n_truncated = res$n_truncated,
         eigenvalues = eigvals,
         lambdas_raw = res$lambdas_raw,
         lambdas = res$lambdas,
         lambdas_reference = res$lambdas_reference)
  })
  out <- .fmg_rows_to_df(rows)
  attr(out, "trace_xcheck") <- sp$trace_ugamma
  out
}

# ── Ordinal / mixed-ordinal (polychoric least-squares) FMG ──────────────────
# Ordinal DWLS/WLS/ULS fits are least-squares fits over polychoric/polyserial
# moments. They carry a single base statistic, the LS chi-square
# `chisq_standard = N * F_min`, and the UGamma eigenvalues come from the same
# polychoric-NACOV sandwich `robust_ordinal()`/`robust_mixed_ordinal()` already
# build and validate against lavaan. So an ordinal FMG test is just the
# estimator-agnostic eigenvalue-tail transform applied to that
# (chi-square, df, eigvals) triple — no new spectrum machinery.
#
# Unlike the complete-data ML path there is no ML/RLS base split (an LS fit has
# no likelihood) and no Du-Bentler unbiased Gamma (the polychoric NACOV is
# already the asymptotic Gamma), so `_ml` and `_ug` are rejected. The ordinal
# stats are passed explicitly, exactly like `robust_ordinal(fit, stats, weight)`.

.fmg_default_tests_ordinal <- function() {
  c("SB", "pEBA2", "pEBA4", "pEBA6", "pOLS")
}

.fmg_adjust_specs_ordinal <- function(specs, caller = "fmg_tests_ordinal") {
  lapply(specs, function(s) {
    if (isTRUE(s$ug)) {
      stop(caller, "(): the unbiased Du-Bentler Gamma is undefined for ",
           "polychoric least-squares fits; the ordinal NACOV is already the ",
           "asymptotic Gamma (test '", s$input, "').", call. = FALSE)
    }
    if (identical(s$base, "ml") && isTRUE(s$base_explicit)) {
      stop(caller, "(): an ordinal least-squares fit has no ML (LRT) base ",
           "statistic; drop the `_ml` suffix to use the LS base (test '",
           s$input, "').", call. = FALSE)
    }
    s$base <- "ls"
    s$canonical <- sub("_(rls|ml)$", "_ls", s$canonical)
    if (!grepl("_ls$", s$canonical)) s$canonical <- paste0(s$canonical, "_ls")
    s
  })
}

# Apply the eigenvalue-tail transforms to an ordinal robust spectrum (the list
# returned by `infer_ordinal_robust` / `infer_mixed_ordinal_robust`, carrying
# `eigvals`, `chisq_standard`, `df`). The source statistic is the LS chi-square.
.fmg_result_rows_ordinal <- function(spectrum, specs) {
  df <- spectrum$df
  eigvals <- spectrum$eigvals
  rows <- lapply(specs, function(s) {
    res <- infer_fmg_test(spectrum$chisq_standard, df, eigvals,
                          method = s$method,
                          param = .fmg_param_for_cpp(s$param))
    list(input = s$input,
         label = s$canonical,
         p_value = res$p_value,
         df = res$df,
         base = "ls",
         base_statistic = res$chi2_source,
         method = res$method,
         param = if (is.na(s$param)) NA_real_ else res$param,
         ug = FALSE,
         chi2_equiv = res$chi2_equiv,
         n_truncated = res$n_truncated,
         eigenvalues = eigvals,
         lambdas_raw = res$lambdas_raw,
         lambdas = res$lambdas,
         lambdas_reference = res$lambdas_reference)
  })
  .fmg_rows_to_df(rows)
}

.fmg_adjust_specs_nested <- function(specs, caller = "fmg_nested") {
  lapply(specs, function(s) {
    if (isTRUE(s$ug)) {
      stop(caller, "(): the unbiased Du-Bentler Gamma is undefined for ",
           "nested difference spectra (test '", s$input, "').",
           call. = FALSE)
    }
    if (identical(s$base, "rls") && isTRUE(s$base_explicit)) {
      stop(caller, "(): nested FMG uses the model-pair difference statistic; ",
           "the single-model RLS base is not defined for a difference spectrum ",
           "(test '", s$input, "').", call. = FALSE)
    }
    s$base <- "ml"
    s$canonical <- sub("_(rls|ml)$", "_ml", s$canonical)
    if (!grepl("_ml$", s$canonical)) s$canonical <- paste0(s$canonical, "_ml")
    s
  })
}

.fmg_result_rows_nested <- function(spectrum, specs) {
  df <- spectrum$df
  eigvals <- spectrum$eigvals
  rows <- lapply(specs, function(s) {
    res <- infer_fmg_test(spectrum$chisq_standard, df, eigvals,
                          method = s$method,
                          param = .fmg_param_for_cpp(s$param))
    list(input = s$input,
         label = s$canonical,
         p_value = res$p_value,
         df = res$df,
         base = "ml",
         base_statistic = res$chi2_source,
         method = res$method,
         param = if (is.na(s$param)) NA_real_ else res$param,
         ug = FALSE,
         chi2_equiv = res$chi2_equiv,
         n_truncated = res$n_truncated,
         eigenvalues = eigvals,
         lambdas_raw = res$lambdas_raw,
         lambdas = res$lambdas,
         lambdas_reference = res$lambdas_reference)
  })
  .fmg_rows_to_df(rows)
}

#' Foldnes-Moss-Gronneberg goodness-of-fit diagnostics for an ordinal fit.
#'
#' Applies the FMG eigenvalue-tail transforms to an ordinal (all-categorical) or
#' mixed continuous/ordinal least-squares fit (DWLS/WLS/ULS over polychoric and
#' polyserial moments). The UGamma spectrum, base chi-square, and df come from
#' the same polychoric-NACOV sandwich as `robust_ordinal()` /
#' `robust_mixed_ordinal()`, so the categorical stats are supplied explicitly the
#' same way. Single- and multi-group fits are supported.
#'
#' An ordinal LS fit has one base statistic, the LS chi-square
#' \eqn{N\,F_\mathrm{min}}; there is no ML/RLS split and no Du-Bentler unbiased
#' Gamma, so test names suffixed `_ml` or `_ug` are rejected. As on the
#' complete-data and FIML paths, the pEBA/pOLS/PALL transforms are magmaan
#' constructions with no external oracle.
#'
#' @param fit A fitted magmaan ordinal (or mixed-ordinal) least-squares model.
#' @param ordinal_stats,mixed_stats The categorical sample statistics built by
#'   `magmaan_core$data_ordinal_stats_from_df()` /
#'   `data_mixed_ordinal_stats_from_df()` (the same object passed to
#'   `robust_ordinal()`).
#' @param tests Character vector of semTests-style test names, or `NULL` for the
#'   ordinal defaults (`SB`, `pEBA2`, `pEBA4`, `pEBA6`, `pOLS`).
#' @param weight Estimation weight (`"DWLS"`, `"WLS"`, `"ULS"`); empty resolves
#'   from `fit$estimator`, matching `robust_ordinal()`.
#'
#' @return A `magmaan_fmg_tests` data frame, identical in shape to `fmg_tests()`.
#' @export
fmg_tests_ordinal <- function(fit, ordinal_stats, tests = NULL, weight = "") {
  tests <- tests %||% .fmg_default_tests_ordinal()
  specs <- .fmg_adjust_specs_ordinal(lapply(tests, .fmg_parse_test),
                                     caller = "fmg_tests_ordinal")
  spectrum <- infer_ordinal_robust(fit, ordinal_stats, weight)
  .fmg_result_rows_ordinal(spectrum, specs)
}

#' @rdname fmg_tests_ordinal
#' @export
fmg_tests_mixed_ordinal <- function(fit, mixed_stats, tests = NULL, weight = "") {
  tests <- tests %||% .fmg_default_tests_ordinal()
  specs <- .fmg_adjust_specs_ordinal(lapply(tests, .fmg_parse_test),
                                     caller = "fmg_tests_mixed_ordinal")
  spectrum <- infer_mixed_ordinal_robust(fit, mixed_stats, weight)
  .fmg_result_rows_ordinal(spectrum, specs)
}

#' Foldnes-Moss-Gronneberg eigenvalue-tail diagnostics for a nested ordinal pair.
#'
#' The nested-model analogue of [fmg_tests_ordinal()]. The Satorra-2000 reduction
#' of the two polychoric least-squares fits yields a difference triple --- the
#' unscaled LS difference statistic \eqn{T_{H0} - T_{H1}}, the restriction rank
#' (`df_diff`), and the difference spectrum (the nested UGamma eigenvalues) ---
#' and the same FMG eigenvalue-tail transforms are applied to it, exactly as
#' [fmg_tests_ordinal()] applies them to a single-model UGamma spectrum.
#'
#' `A.method` defaults to `"delta"`: the configural-to-metric ordinal invariance
#' step is non-nested (Wu-Estabrook metric frees the group-2+ scale/intercepts
#' that configural fixes), so the exact parameter-nesting restriction does not
#' apply --- the moment-Jacobian column-space (`delta`) restriction is what
#' lavaan's `lavTestLRT(method = "satorra.2000")` uses there. (`robust_nested_lrt()`
#' keeps the `"exact"` default for nested complete-data/FIML pairs.)
#'
#' @param fit_H1 Less-restricted (e.g. configural) ordinal least-squares fit.
#' @param fit_H0 More-restricted (e.g. metric / `group.equal`) ordinal fit.
#' @param ordinal_stats The categorical sample statistics used for both fits
#'   (the object from `data_ordinal_stats_from_df()`), as in `fmg_tests_ordinal()`.
#' @param tests,weight As in [fmg_tests_ordinal()].
#' @param A.method `"delta"` (default) or `"exact"`.
#'
#' @return A `magmaan_fmg_tests` data frame, identical in shape to
#'   [fmg_tests_ordinal()].
#' @export
fmg_nested_ordinal <- function(fit_H1, fit_H0, ordinal_stats, tests = NULL,
                               weight = "", A.method = c("delta", "exact")) {
  A.method <- match.arg(A.method)
  tests <- tests %||% .fmg_default_tests_ordinal()
  specs <- .fmg_adjust_specs_ordinal(lapply(tests, .fmg_parse_test),
                                     caller = "fmg_nested_ordinal")
  nested <- robust_nested_lrt(
    fit_H1, fit_H0, data = ordinal_stats, gamma = "empirical",
    method = "restriction_map", A.method = A.method,
    weight = if (nzchar(weight)) weight else NULL)
  spectrum <- list(chisq_standard = nested$T_diff,
                   df = nested$df_diff,
                   eigvals = nested$eigenvalues)
  .fmg_result_rows_ordinal(spectrum, specs)
}

#' Foldnes-Moss-Gronneberg diagnostics for a nested continuous/FIML/ML2S pair.
#'
#' Applies the FMG eigenvalue-tail transforms to the Satorra-2000
#' restriction-map difference spectrum returned by [robust_nested_lrt()]. This
#' is the non-ordinal analogue of [fmg_nested_ordinal()] for complete-data ML,
#' FIML, and two-stage ML (`ML2S`) nested pairs.
#'
#' @param fit_H1 Less-restricted fitted model.
#' @param fit_H0 More-restricted fitted model.
#' @param data Raw complete data for complete-data ML pairs. FIML and ML2S pairs
#'   use `fit_H1$raw_data` and reject `data`, matching [robust_nested_lrt()].
#' @param tests Character vector of semTests-style test names, or `NULL` for the
#'   nested defaults (`SB`, `pEBA2`, `pEBA4`, `pEBA6`, `pOLS`).
#' @param A.method `"exact"` (default) or `"delta"`.
#'
#' @return A `magmaan_fmg_tests` data frame.
#' @export
fmg_nested <- function(fit_H1, fit_H0, data = NULL, tests = NULL,
                       A.method = c("exact", "delta")) {
  A.method <- match.arg(A.method)
  tests <- tests %||% .fmg_default_tests_ordinal()
  specs <- .fmg_adjust_specs_nested(lapply(tests, .fmg_parse_test),
                                    caller = "fmg_nested")
  nested <- robust_nested_lrt(
    fit_H1, fit_H0, data = data, gamma = "empirical",
    method = "restriction_map", A.method = A.method)
  spectrum <- list(chisq_standard = nested$T_diff,
                   df = nested$df_diff,
                   eigvals = nested$eigenvalues)
  .fmg_result_rows_nested(spectrum, specs)
}

#' Foldnes-Moss-Gronneberg goodness-of-fit diagnostics.
#'
#' Computes FMG p-values and diagnostics for a complete-data ML fit or a FIML
#' (missing-data) fit. Complete-data fits must carry complete raw data, as fits
#' from `magmaan(..., data.frame, estimator = "ML")` or `fit_ml(model,
#' df_to_data(...))` do, or callers can pass complete raw `data` explicitly.
#'
#' FIML fits (`fit_fiml()` / `magmaan(..., estimator = "FIML")`, single- or
#' multi-group) are supported: the missing-data UGamma spectrum is computed
#' first-principles from the saturated-model EM ACOV and either saturated or
#' model-implied H1 information, with the FIML LRT as the base statistic. Under
#' FIML only the biased Gamma-hat and the ML base are defined, so `_ug` and
#' `_rls` are rejected (an unsuffixed base resolves to ML). This is a principled
#' construction, not a port of semTests' (unsound) FIML handling.
#'
#' Two-stage ML (ML2S) fits (`fit_ml2s()` / `magmaan(..., estimator = "ML2S")`)
#' are supported the same way: the df-dimensional UGamma spectrum and the
#' Stage-2 ML base chi-square are taken from the two-stage inference already
#' attached to the fit (`fit$ml2s`), and the eigenvalue-tail transforms are
#' applied to that triple. As under FIML, `_ug` and `_rls` are rejected and
#' `h1_information` is fixed at its `"saturated"` default.
#'
#' @param fit A fitted magmaan ML (complete-data), FIML, or ML2S model.
#' @param tests Character vector of semTests-style test names, or `NULL` for the
#'   recommended defaults (complete-data or FIML-appropriate). Recognised types:
#'   `std`, `sb`, `ss`, `mv`, `sf`, `all`, `pall`, `eba<j>`, `peba<j>`, `pols<gamma>`, each
#'   optionally suffixed `_ug` and `_ml` / `_rls` (complete-data only).
#' @param data Optional complete raw data (complete-data fits only). Usually
#'   unnecessary for new fits that retain `$raw_data`.
#' @param h1_information For FIML fits, `"saturated"` (default) uses the
#'   saturated normal-theory H1 information as the U projector metric; `"structured"`
#'   evaluates the same H1 curvature at the model-implied moments. Ignored only
#'   at its default for complete-data ML fits.
#'
#' @return A data frame with scalar diagnostics plus list-columns for the raw
#'   UGamma spectrum and the method-specific lambda vectors.
#' @export
fmg_tests <- function(fit, tests = NULL, data = NULL,
                      h1_information = c("saturated", "structured")) {
  h1_information <- match.arg(h1_information)
  tests <- .fmg_resolve_default_tests(fit, tests)
  specs <- lapply(tests, .fmg_parse_test)
  # ML2S must be checked before FIML: a two-stage fit also carries a
  # `magmaan_fiml_data` raw object, so `.fmg_is_fiml()` would otherwise claim it.
  if (.fmg_is_ml2s(fit)) {
    if (!is.null(data)) {
      stop("fmg_tests(): ML2S FMG uses the fit's own two-stage spectrum; the ",
           "`data` argument is not supported for ML2S fits.", call. = FALSE)
    }
    if (!identical(h1_information, "saturated")) {
      stop("fmg_tests(): h1_information is fixed for ML2S fits (the two-stage ",
           "convention uses the complete-data weight as the U-metric); only the ",
           "default 'saturated' is accepted.", call. = FALSE)
    }
    specs <- .fmg_adjust_specs_ml2s(specs)
    return(.fmg_result_rows_ml2s(fit, specs))
  }
  if (.fmg_is_fiml(fit)) {
    if (!is.null(data)) {
      stop("fmg_tests(): FIML FMG uses the fit's own missing-data raw blocks; ",
           "the `data` argument is not supported for FIML fits.", call. = FALSE)
    }
    specs <- .fmg_adjust_specs_fiml(specs)
    return(.fmg_result_rows_fiml(fit, specs,
                                 h1_information = h1_information))
  }
  if (!identical(h1_information, "saturated")) {
    stop("fmg_tests(): h1_information = 'structured' is only supported for ",
         "FIML fits.", call. = FALSE)
  }
  estimator <- .fmg_fit_estimator(fit)
  if (!identical(estimator, "ML")) {
    stop("fmg_tests(): FMG via this entry point requires a complete-data ML or ",
         "FIML fit (got estimator = '", estimator, "'). For ordinal/polychoric ",
         "least-squares fits use fmg_tests_ordinal(fit, ordinal_stats) or ",
         "fmg_tests_mixed_ordinal(fit, mixed_stats).", call. = FALSE)
  }
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
#'   `std`, `sb`, `ss`, `mv`, `sf`, `all`, `pall`, `eba<j>`, `peba<j>`, `pols<gamma>`, each
#'   optionally suffixed `_ug` (unbiased Gamma-hat) and `_ml` / `_rls` (base
#'   statistic; default `rls`).
#' @param h1_information Passed to [fmg_tests()] for FIML fits.
#'
#' @return A named numeric vector of p-values; names are the canonical test
#'   labels (e.g. `peba4_rls`, `sb_ug_rls`), matching `semTests::pvalues()`.
#' @export
fmg_pvalues <- function(fit, data = NULL, tests = NULL,
                        h1_information = c("saturated", "structured")) {
  tab <- fmg_tests(fit, tests = tests, data = data,
                   h1_information = h1_information)
  out <- tab$p_value
  names(out) <- tab$label
  out
}

.fit_rmsea_family <- function(x2, df, n, g = 1L, c_hat = 1,
                              close_h0 = 0.05, notclose_h0 = 0.08) {
  out <- list(
    rmsea = 0,
    rmsea.ci.lower = 0,
    rmsea.ci.upper = 0,
    rmsea.pvalue = NA_real_,
    rmsea.notclose.pvalue = NA_real_
  )
  if (!is.finite(x2) || !is.finite(df) || !is.finite(n) ||
      !is.finite(c_hat) || df <= 0 || n <= 0 || c_hat <= 0) {
    return(out)
  }
  g <- max(1L, as.integer(g))
  out$rmsea <- sqrt(max((x2 / n) / df - c_hat / n, 0)) * sqrt(g)
  lower_lambda <- function(lambda) pchisq(x2, df = df, ncp = lambda) - 0.95
  upper_lambda <- function(lambda) pchisq(x2, df = df, ncp = lambda) - 0.05
  if (df >= 1 && lower_lambda(0) >= 0) {
    lambda_l <- try(uniroot(lower_lambda, lower = 0, upper = x2)$root,
                    silent = TRUE)
    if (!inherits(lambda_l, "try-error") && is.finite(lambda_l)) {
      out$rmsea.ci.lower <- sqrt(c_hat * lambda_l / (n * df)) * sqrt(g)
    }
  }
  n_rmsea <- max(n, x2 * 4)
  if (df >= 1 && upper_lambda(n_rmsea) <= 0 && upper_lambda(0) >= 0) {
    lambda_u <- try(uniroot(upper_lambda, lower = 0, upper = n_rmsea)$root,
                    silent = TRUE)
    if (!inherits(lambda_u, "try-error") && is.finite(lambda_u)) {
      out$rmsea.ci.upper <- sqrt(c_hat * lambda_u / (n * df)) * sqrt(g)
    }
  }
  ncp_close <- n * df * close_h0^2 / (g * c_hat)
  ncp_notclose <- n * df * notclose_h0^2 / (g * c_hat)
  out$rmsea.pvalue <- 1 - pchisq(x2, df = df, ncp = ncp_close)
  out$rmsea.notclose.pvalue <- pchisq(x2, df = df, ncp = ncp_notclose)
  out
}

.fit_cfi <- function(x2, df, x2_null, df_null, c_hat = 1,
                     c_hat_null = 1, robust = FALSE) {
  adj <- if (robust) c_hat else 1
  adj_null <- if (robust) c_hat_null else 1
  t1 <- max(x2 - adj * df, 0)
  t2 <- max(x2 - adj * df, x2_null - adj_null * df_null, 0)
  if (isTRUE(all.equal(t1, 0)) && isTRUE(all.equal(t2, 0))) 1 else 1 - t1 / t2
}

.fit_tli <- function(x2, df, x2_null, df_null, c_hat = 1,
                     c_hat_null = 1, robust = FALSE) {
  if (df <= 0) return(1)
  adj <- if (robust) c_hat else 1
  adj_null <- if (robust) c_hat_null else 1
  t1 <- (x2 - adj * df) * df_null
  t2 <- (x2_null - adj_null * df_null) * df
  if (abs(t2) > 0) 1 - t1 / t2 else NA_real_
}

.robust_fit_measures_from_scalars <- function(x, n_total, n_groups = 1L,
                                              close_h0 = 0.05,
                                              notclose_h0 = 0.08) {
  need <- c("chisq", "df", "chisq.scaled", "chisq.scaling.factor",
            "baseline.chisq", "baseline.df", "baseline.chisq.scaled",
            "baseline.chisq.scaling.factor")
  miss <- setdiff(need, names(x))
  if (length(miss)) {
    stop("fit_measures(): robust scalar list is missing: ",
         paste(miss, collapse = ", "), call. = FALSE)
  }
  df <- as.numeric(x$df)
  c_hat <- as.numeric(x$chisq.scaling.factor)
  df_scaled_rmsea <- df * c_hat
  rmsea_scaled <- .fit_rmsea_family(as.numeric(x$chisq), df_scaled_rmsea,
                                    n_total, n_groups, 1,
                                    close_h0, notclose_h0)
  rmsea_robust_value <- .fit_rmsea_family(as.numeric(x$chisq), df,
                                          n_total, n_groups, c_hat,
                                          close_h0, notclose_h0)
  rmsea_robust_tail <- .fit_rmsea_family(as.numeric(x$chisq.scaled), df,
                                         n_total, n_groups, c_hat,
                                         close_h0, notclose_h0)
  list(
    chisq.scaled = as.numeric(x$chisq.scaled),
    df.scaled = as.integer(x$df),
    pvalue.scaled = pchisq(as.numeric(x$chisq.scaled), df = df,
                           lower.tail = FALSE),
    chisq.scaling.factor = c_hat,
    baseline.chisq.scaled = as.numeric(x$baseline.chisq.scaled),
    baseline.df.scaled = as.integer(x$baseline.df),
    baseline.pvalue.scaled =
      pchisq(as.numeric(x$baseline.chisq.scaled),
             df = as.numeric(x$baseline.df), lower.tail = FALSE),
    baseline.chisq.scaling.factor =
      as.numeric(x$baseline.chisq.scaling.factor),
    cfi.scaled = .fit_cfi(as.numeric(x$chisq.scaled), df,
                          as.numeric(x$baseline.chisq.scaled),
                          as.numeric(x$baseline.df)),
    tli.scaled = .fit_tli(as.numeric(x$chisq.scaled), df,
                          as.numeric(x$baseline.chisq.scaled),
                          as.numeric(x$baseline.df)),
    cfi.robust = .fit_cfi(as.numeric(x$chisq), df,
                          as.numeric(x$baseline.chisq),
                          as.numeric(x$baseline.df),
                          c_hat,
                          as.numeric(x$baseline.chisq.scaling.factor),
                          robust = TRUE),
    tli.robust = .fit_tli(as.numeric(x$chisq), df,
                          as.numeric(x$baseline.chisq),
                          as.numeric(x$baseline.df),
                          c_hat,
                          as.numeric(x$baseline.chisq.scaling.factor),
                          robust = TRUE),
    rmsea.scaled = rmsea_scaled$rmsea,
    rmsea.ci.lower.scaled = rmsea_scaled$rmsea.ci.lower,
    rmsea.ci.upper.scaled = rmsea_scaled$rmsea.ci.upper,
    rmsea.pvalue.scaled = rmsea_scaled$rmsea.pvalue,
    rmsea.notclose.pvalue.scaled = rmsea_scaled$rmsea.notclose.pvalue,
    rmsea.robust = rmsea_robust_value$rmsea,
    rmsea.ci.lower.robust = rmsea_robust_tail$rmsea.ci.lower,
    rmsea.ci.upper.robust = rmsea_robust_tail$rmsea.ci.upper,
    rmsea.pvalue.robust = rmsea_robust_tail$rmsea.pvalue,
    rmsea.notclose.pvalue.robust = rmsea_robust_tail$rmsea.notclose.pvalue
  )
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
#' @param robust Optional scalar list with lavaan-style scaled user and baseline
#'   test fields. Automatic robust estimator dispatch is not yet wired.
#' @param data Optional complete raw data used only when FMG is requested and
#'   the fit does not carry `$raw_data`.
#' @export
fit_measures <- function(fit, baseline = NULL, fmg = NULL, robust = NULL,
                         data = NULL) {
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
  if (!is.null(robust) && !identical(robust, FALSE)) {
    if (!is.list(robust)) {
      stop("fit_measures(): automatic robust = 'MLM'/'MLR' dispatch is not ",
           "wired yet; pass a scalar robust list.", call. = FALSE)
    }
    n_total <- sum(as.numeric(ss$nobs))
    n_groups <- length(ss$nobs)
    out <- c(out, .robust_fit_measures_from_scalars(robust, n_total, n_groups))
  }
  if (!is.null(fmg) && !identical(fmg, FALSE)) {
    tests <- if (isTRUE(fmg)) NULL else fmg
    out$fmg <- fmg_tests(fit, tests = tests, data = data)
  }
  out
}

#' Misspecification-robust ordinal fit measures (estimated-weight intervals)
#'
#' Estimated-weight (categorical DWLS) misspecification inference for the
#' absolute and incremental fit indices RMSEA, CRMR, SRMR, CFI and TLI, each with
#' a confidence interval that propagates the sampling variability of the
#' estimated polychoric weight (the gamma channel). The single consolidated
#' surface over the per-index entry points. Single-group all-ordinal DWLS fits
#' only.
#'
#' CFI carries a trustworthy interval and is largely robust to weight estimation;
#' TLI's point is calibrated but its interval is conservative and unreliable at
#' strong misfit (the generalized-df ratio is ill-conditioned). See
#' `docs/research/notes/cfi_tli_misspec_inference.tex`.
#'
#' @param fit A fitted all-ordinal DWLS magmaan object.
#' @param ordinal_stats The categorical sample statistics used for the fit, with
#'   `moment_influence` and `int_data` (e.g. from `data_ordinal_stats_from_df()`).
#'   Required and passed explicitly, like `robust_ordinal()` and the profile
#'   bindings: a fitted object does not retain the integer data the
#'   estimated-weight inference needs.
#' @param estimated_weight `TRUE` (default) propagates the polychoric-weight
#'   sampling variability; `FALSE` is the fixed-weight comparator.
#' @param conf_level Two-sided confidence level for the intervals.
#' @param eig_tol Eigenvalue tolerance for the profile-contrast spectrum.
#' @return A named list: `rmsea`/`crmr`/`srmr`/`cfi`/`tli` with `.ci.lower` and
#'   `.ci.upper`, the exact-fit `rmsea.pvalue` and `crmr.pvalue`, the user and
#'   baseline `chisq`/`df`, `conf.level`, `estimated.weight`, and `warnings`.
#' @export
fit_measures_misspec <- function(fit, ordinal_stats = NULL,
                                 estimated_weight = TRUE, conf_level = 0.90,
                                 eig_tol = 1e-10) {
  if (is.null(ordinal_stats)) {
    stop("fit_measures_misspec(): `ordinal_stats` is required (the categorical ",
         "sample statistics used for the fit, with moment_influence and ",
         "int_data, e.g. from data_ordinal_stats_from_df()). A fitted object ",
         "does not retain the integer data the estimated-weight inference needs.",
         call. = FALSE)
  }
  infer_ordinal_fit_measures_misspec(fit, ordinal_stats, estimated_weight,
                                     conf_level, eig_tol)
}
