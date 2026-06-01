# Foldnes-Moss-Gronneberg single-model goodness-of-fit p-values.
#
# A thin composition over the existing inference wrappers: it assembles the
# ingredients the C++ `infer_fmg_test()` needs (source chi-square, df, and the
# UGamma eigenvalues for biased / unbiased Gamma-hat) once, then dispatches each
# requested test. All numeric work lives in C++ (eigenvalues -> imhof_upper);
# this file only routes. The test-name grammar mirrors semTests so a magmaan
# call and a `semTests::pvalues()` call line up name-for-name.

# Parse one semTests-style test name into its components.
#   grammar: <type>[_ug][_ml|_rls]   (case-insensitive)
#     type : std | sb | ss | sf | all | pall | eba<j> | peba<j> | pols<gamma>
#     _ug  : unbiased Gamma-hat (Du-Bentler); absent => biased Gamma-hat
#     _ml | _rls : base statistic (default rls, as in semTests::split_input)
.fmg_parse_test <- function(name) {
  toks <- strsplit(tolower(name), "_", fixed = TRUE)[[1]]
  type <- toks[1]
  rest <- toks[-1]
  ug   <- "ug" %in% rest
  base <- if ("ml" %in% rest) "ml" else "rls"

  num <- function(s, default) if (nchar(s)) as.numeric(s) else default

  if (startsWith(type, "peba")) {
    method <- "peba"; param <- num(substring(type, 5L), 4)
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
    stop("fmg_pvalues(): unrecognized FMG test name '", name, "'.", call. = FALSE)
  }

  canonical <- paste0(c(label, if (ug) "ug", base), collapse = "_")
  list(method = method, param = param, ug = ug, base = base, canonical = canonical)
}

#' Foldnes-Moss-Gronneberg goodness-of-fit p-values for a fitted magmaan model.
#'
#' Computes the FMG family of robustified goodness-of-fit p-values (Foldnes,
#' Moss & Gronneberg, 2024) for a single fitted model, using magmaan's own
#' UGamma eigenvalues and Imhof tail (`infer_fmg_test()`). The `tests` grammar
#' mirrors `semTests::pvalues()`, so the two can be compared name-for-name.
#'
#' @param fit A fitted magmaan model (the list returned by `fit_fit()` /
#'   `magmaan()`), single-group.
#' @param data The raw data: a data.frame or matrix whose columns include the
#'   model's observed variables (reordered by name).
#' @param tests Character vector of semTests-style test names. Recognised types:
#'   `std`, `sb`, `ss`, `sf`, `all`, `pall`, `eba<j>`, `peba<j>`, `pols<gamma>`, each
#'   optionally suffixed `_ug` (unbiased Gamma-hat) and `_ml` / `_rls` (base
#'   statistic; default `rls`).
#'
#' @return A named numeric vector of p-values; names are the canonical test
#'   labels (e.g. `peba4_rls`, `sb_ug_rls`), matching `semTests::pvalues()`.
#' @export
fmg_pvalues <- function(fit, data,
                        tests = c("SB_UG_RLS", "pEBA2_UG_RLS", "pEBA4_RLS",
                                  "pEBA6_RLS", "pOLS_RLS")) {
  ngroups <- fit$ngroups %||% 1L
  if (ngroups > 1L) {
    stop("fmg_pvalues(): only single-group fits are supported (got ngroups = ",
         ngroups, ").", call. = FALSE)
  }
  specs <- lapply(tests, .fmg_parse_test)

  ov <- if (is.list(fit$ov_names)) fit$ov_names[[1]] else fit$ov_names
  if (is.null(ov)) stop("fmg_pvalues(): fit is missing $ov_names", call. = FALSE)
  if (is.data.frame(data) || is.matrix(data)) {
    miss <- setdiff(ov, colnames(data))
    if (length(miss)) {
      stop("fmg_pvalues(): data is missing observed variables: ",
           paste(miss, collapse = ", "), call. = FALSE)
    }
    X <- as.matrix(data[, ov, drop = FALSE])
  } else {
    stop("fmg_pvalues(): `data` must be a data.frame or matrix.", call. = FALSE)
  }

  pt <- fit$partable
  ss <- data_sample_stats_from_raw(X)

  # source statistics + df. The RLS base is the model-weighted Browne reweighted
  # least squares statistic (N/2) tr[((S - Sigma)Sigma^{-1})^2] -- lavaan's
  # `browne.residual.nt.model`, the FMG (2024) T_RLS -- NOT the sample-weighted
  # `browne.residual.nt` that `infer_browne_residual_nt()` returns.
  df   <- infer_df_stat(pt, ss)
  base_stat <- c(ml  = infer_chi2_stat(ss, fit$fmin),
                 rls = infer_rls_chi2_fit(fit, model_implied(fit))$statistic)

  # UGamma eigenvalues: biased (A) always; unbiased (U) only if requested.
  # Keep the hot spectra path in C++ so UFactor and its block factorizations
  # are not copied through R for every reducer.
  need_ug <- any(vapply(specs, `[[`, logical(1), "ug"))
  ev <- infer_fmg_ugamma_spectra(fit, X, need_ug)

  out <- vapply(specs, function(s) {
    eigvals <- if (s$ug) ev$unbiased else ev$biased
    param   <- if (is.na(s$param)) 4.0 else s$param
    infer_fmg_test(base_stat[[s$base]], df, eigvals,
                   method = s$method, param = param)$p_value
  }, numeric(1))
  names(out) <- vapply(specs, `[[`, "", "canonical")
  out
}
