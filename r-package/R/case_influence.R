# Case-level influence diagnostics (semfindr parity, exact leave-one-out engine).
#
# Reproduces the outputs and output format of the R package semfindr
# (Cheung & Lai, 2026; Pek & MacCallum, 2011 framework) for complete-data
# continuous ML/ULS/GLS, single group. The exact engine refits the model with
# one case removed and contrasts the estimates / fit measures against the full
# sample; the approximate one-step engine is added separately.
#
# References: Pek & MacCallum (2011) doi:10.1080/00273171.2011.561068;
# Cheung & Lai (2026) doi:10.1080/00273171.2026.2634293.

# Fitting helper for the supported estimators. Errors on anything that consumes
# raw rows / pairwise tables rather than the SampleStats-only objective.
.case_fit_fun <- function(estimator) {
  switch(estimator,
         ML  = fit_ml,
         ULS = fit_uls,
         GLS = fit_gls,
         stop("case_rerun(): leave-one-out is currently supported for ",
              "estimator ML/ULS/GLS only; got '", estimator, "'", call. = FALSE))
}

.case_estimator <- function(fit) {
  est <- fit$estimator %||% fit$options$estimator %||% NA_character_
  toupper(as.character(est)[1L])
}

# Rebuild a magmaan_data from a single-group raw block via the canonical
# df_to_data path, so the down-dated refit is configured exactly as the original
# fit was (same scaling, same ov order, same sample-stat reduction).
.case_loo_data <- function(X_block, spec, scaling) {
  df <- as.data.frame(X_block)
  df_to_data(df, spec, missing = "error", scaling = scaling)
}

# Warm-start: write the full-sample theta into the spec partable's `ustart` for
# each free row (free rows with a non-NaN ustart seed the optimizer). Returns a
# fresh spec; does not mutate the input.
.case_warm_spec <- function(fit, warm_start) {
  spec <- fit$model
  if (!warm_start) return(spec)
  ptab <- spec$partable
  theta <- fit$theta
  for (k in seq_along(theta)) {
    ptab$ustart[ptab$free == k] <- theta[k]
  }
  spec$partable <- ptab
  spec
}

# Free-parameter table for a fit: one row per free parameter, ordered by the
# free index k = 1..npar (matching the order of fit$theta and the vcov). Two
# name columns mirror semfindr's two conventions: `name` = lhs op rhs (e.g.
# "f1=~x2"; used by est_change_raw, which blanks user labels) and `label_name` =
# the user label if present else lhs op rhs (used by est_change).
.case_free_table <- function(fit) {
  ptab <- fit$partable
  fr <- which(ptab$free > 0L)
  if (!length(fr)) stop("case influence: fit has no free parameters")
  k <- as.integer(ptab$free[fr])
  ord <- order(k)
  fr <- fr[ord]
  k <- k[ord]
  name <- paste0(ptab$lhs[fr], ptab$op[fr], ptab$rhs[fr])
  lab <- if ("label" %in% names(ptab)) as.character(ptab$label[fr]) else rep("", length(fr))
  lab[is.na(lab)] <- ""
  label_name <- ifelse(nzchar(lab), lab, name)
  group <- if ("group" %in% names(ptab)) as.integer(ptab$group[fr]) else rep(1L, length(fr))
  # Multigroup column names collide (e.g. f1=~x2 in every group), so suffix them
  # with the group *label* (not index): magmaan and lavaan order groups
  # differently, so a label keeps the columns tool-independent. Single-group
  # output is unchanged.
  multigroup <- length(unique(group)) > 1L
  glabels <- fit$group_labels
  suffix <- if (!multigroup) {
    ""
  } else if (!is.null(glabels) && length(glabels) >= max(group)) {
    paste0(".", glabels[group])
  } else {
    paste0(".g", group)
  }
  data.frame(k = k, row = fr, group = group,
             lhs = ptab$lhs[fr], op = ptab$op[fr], rhs = ptab$rhs[fr],
             name = paste0(name, suffix),
             label_name = paste0(label_name, suffix),
             stringsAsFactors = FALSE)
}

# Resolve a `parameters` selector (lavaan-syntax names and/or bare operators)
# against the free-parameter table. Returns row indices into fp (a subset, in fp
# order). NULL selects all free parameters.
.case_pars_select <- function(fp, parameters) {
  if (is.null(parameters)) return(seq_len(nrow(fp)))
  ops <- c("=~", "~~", "~1", "~")
  hit <- logical(nrow(fp))
  for (tok in parameters) {
    tok <- as.character(tok)
    if (tok %in% ops) {
      hit <- hit | (fp$op == tok)
    } else {
      key <- gsub("\\s+", "", tok)
      hit <- hit | (fp$name == key)
    }
  }
  idx <- which(hit)
  if (!length(idx)) {
    stop("case influence: no free parameters matched `parameters`: ",
         paste(parameters, collapse = ", "), call. = FALSE)
  }
  idx
}

# Naive ML (lavaan se = "standard") parameter covariance: expected-information
# bread, no empirical meat. This is the matrix semfindr standardizes by under the
# default fit. Pass an explicit `vcov` to est_change() to use a robust regime.
.case_model_vcov <- function(fit) {
  info <- magmaan_core$inference_information_expected(fit)
  magmaan_core$inference_vcov(info, fit)
}

#' Refit a model repeatedly, each time with one case removed
#'
#' The exact leave-one-out engine: for each selected case, drop its row, rebuild
#' the sample statistics, and refit (warm-started from the full-sample solution).
#' Mirror of `semfindr::lavaan_rerun()`. Continuous ML/ULS/GLS.
#'
#' Single-group fits work directly from the raw data the fit carries. For
#' multiple-group fits, pass the original `data` frame (the per-group raw blocks
#' on the fit have lost the original row order needed to map a global case index
#' to its group); case ids are then the data-frame rows, as in semfindr.
#'
#' @param fit A fitted `magmaan_fit` (continuous ML/ULS/GLS).
#' @param data Optional original data frame. Required for multiple-group fits;
#'   for single-group fits, defaults to the raw data on `fit`.
#' @param to_rerun Integer case indices to refit; default all cases.
#' @param warm_start Seed each refit from the full-sample estimates (faster, and
#'   keeps refits on the same optimum). Default `TRUE`.
#' @return A `magmaan_case_rerun` list: `rerun` (per-case refit fits, `NULL` when
#'   a refit fails or is inadmissible), `fit`, `selected`, `case_id`, `converged`.
#' @export
case_rerun <- function(fit, data = NULL, to_rerun = NULL, warm_start = TRUE) {
  if (!inherits(fit, "magmaan_fit")) {
    stop("case_rerun(): `fit` must be a magmaan_fit", call. = FALSE)
  }
  estimator <- .case_estimator(fit)
  fit_fun <- .case_fit_fun(estimator)
  spec_warm <- .case_warm_spec(fit, warm_start)
  group_var <- fit$group_var %||% ""

  if (is.data.frame(data)) {
    # Data-frame path: works for any number of groups. Drop the global row and
    # rebuild per-group sample stats through the canonical df_to_data pipeline.
    n <- nrow(data)
    case_id <- rownames(data)
    grp <- if (nzchar(group_var)) group_var else NULL
    make_loo <- function(i) df_to_data(data[-i, , drop = FALSE], spec_warm,
                                       group = grp, missing = "error")
  } else {
    raw <- fit$raw_data
    if (is.null(raw) || is.null(raw$X)) {
      stop("case_rerun(): `fit` does not carry raw data; pass the original ",
           "`data` frame", call. = FALSE)
    }
    if (length(raw$X) != 1L) {
      stop("case_rerun(): multiple-group fits need the original `data` frame ",
           "(pass `data = <your data.frame>`)", call. = FALSE)
    }
    X <- raw$X[[1L]]
    n <- nrow(X)
    case_id <- rownames(X)
    scaling <- raw$scaling %||% "n"
    make_loo <- function(i) .case_loo_data(X[-i, , drop = FALSE], spec_warm, scaling)
  }
  if (is.null(case_id)) case_id <- as.character(seq_len(n))

  selected <- if (is.null(to_rerun)) seq_len(n) else as.integer(to_rerun)
  if (any(selected < 1L | selected > n)) {
    stop("case_rerun(): `to_rerun` out of range 1:", n, call. = FALSE)
  }

  reruns <- vector("list", length(selected))
  converged <- logical(length(selected))
  for (j in seq_along(selected)) {
    refit <- tryCatch(fit_fun(spec_warm, make_loo(selected[j])),
                      error = function(e) NULL)
    ok <- !is.null(refit) && isTRUE(refit$converged)
    converged[j] <- ok
    reruns[[j]] <- if (ok) refit else NULL
  }

  out <- list(rerun = reruns, fit = fit, selected = selected,
              case_id = case_id[selected], converged = converged,
              estimator = estimator, group_var = group_var)
  class(out) <- c("magmaan_case_rerun", "list")
  out
}

.case_check_rerun <- function(rerun, what) {
  if (!inherits(rerun, "magmaan_case_rerun")) {
    stop(what, ": `rerun_out` must come from case_rerun()", call. = FALSE)
  }
}

# Per-case raw or standardized-solution estimate vector, indexed by free k.
# Returns NULL for a failed refit.
.case_theta_minus <- function(refit, free_k) {
  if (is.null(refit)) return(NULL)
  refit$theta[free_k]
}

#' Case influence on raw parameter estimates (DFBETA / DFZTHETA)
#'
#' For each case, `(estimate with all cases) - (estimate without this case)`.
#' With `standardized = FALSE` (default) these are raw coefficient changes;
#' with `standardized = TRUE` they are changes in the standardized (std.all)
#' solution. The change is NOT divided by a standard error (see [est_change()]).
#' Mirror of `semfindr::est_change_raw()`.
#'
#' @param rerun_out Output of [case_rerun()].
#' @param parameters Optional selector (lavaan-syntax names and/or operators
#'   `=~`, `~`, `~~`, `~1`); default all free parameters.
#' @param standardized Use the standardized (std.all) solution. Default `FALSE`.
#' @return A matrix, one row per case, one column per selected parameter,
#'   rownames = case ids.
#' @export
est_change_raw <- function(rerun_out, parameters = NULL, standardized = FALSE) {
  .case_check_rerun(rerun_out, "est_change_raw()")
  fit <- rerun_out$fit
  fp <- .case_free_table(fit)
  sel <- .case_pars_select(fp, parameters)
  free_k <- fp$k[sel]
  col_names <- fp$name[sel]

  if (standardized) {
    return(.est_change_raw_std(rerun_out, fp, sel))
  }

  theta_full <- fit$theta[free_k]
  out <- matrix(NA_real_, nrow = length(rerun_out$selected), ncol = length(sel),
                dimnames = list(rerun_out$case_id, col_names))
  for (j in seq_along(rerun_out$rerun)) {
    tm <- .case_theta_minus(rerun_out$rerun[[j]], free_k)
    if (!is.null(tm)) out[j, ] <- theta_full - tm
  }
  out
}

# Standardized-solution (std.all) variant of est_change_raw. Diffs the std.all
# estimate of each selected free parameter, full vs leave-one-out.
.est_change_raw_std <- function(rerun_out, fp, sel) {
  fit <- rerun_out$fit
  std_full <- .case_std_all_by_free(fit)
  free_k <- fp$k[sel]
  col_names <- fp$name[sel]
  full_vals <- std_full[as.character(free_k)]
  out <- matrix(NA_real_, nrow = length(rerun_out$selected), ncol = length(sel),
                dimnames = list(rerun_out$case_id, col_names))
  for (j in seq_along(rerun_out$rerun)) {
    refit <- rerun_out$rerun[[j]]
    if (is.null(refit)) next
    std_i <- .case_std_all_by_free(refit)
    out[j, ] <- full_vals - std_i[as.character(free_k)]
  }
  out
}

# std.all estimates of the free parameters, named by free index k.
.case_std_all_by_free <- function(fit) {
  V <- .case_model_vcov(fit)
  std <- standardized(fit, V, type = "all")
  est <- std$est.std %||% std$est %||% std[["std.all"]]
  free <- fit$partable$free
  keep <- free > 0L
  vals <- est[keep]
  names(vals) <- as.character(free[keep])
  vals
}

#' Standardized case influence on parameter estimates (DFTHETAS) and gCD
#'
#' For each case, the raw change `(with all) - (without this case)` divided by
#' the standard error of the estimate *from the leave-one-out refit* (DFTHETAS;
#' Pek & MacCallum, 2011, Eq. 7), plus a final `gcd` column: the generalized
#' Cook's distance `Delta' V_(-i)^{-1} Delta` (Pek & MacCallum, 2011, Eq. 6)
#' over the selected parameters, with `V_(-i)` the leave-one-out parameter
#' covariance. Both the SE (Eq. 7) and the gCD covariance (Eq. 6) are the
#' reduced-sample versions, exactly as the paper specifies. Mirror of
#' `semfindr::est_change()` (which standardizes by, and forms gCD from, the
#' deleted-case refit's covariance).
#'
# Leave-one-out parameter covariance for a refit, under the chosen SE regime.
# "standard" = naive ML (lavaan se="standard"); "robust.sem" = expected-bread
# empirical-meat sandwich (Satorra-Bentler/MLM); "robust.huber.white" =
# observed-bread sandwich (Yuan-Bentler/MLR). The robust regimes need the
# refit's raw data (carried on every fit_* result).
.case_refit_vcov <- function(refit, se) {
  if (se == "standard") return(.case_model_vcov(refit))
  if (is.null(refit$raw_data) || is.null(refit$raw_data$X)) {
    stop("est_change(se = \"", se, "\"): the refit carries no raw data",
         call. = FALSE)
  }
  bread <- if (se == "robust.huber.white") "observed" else "expected"
  magmaan_core$robust_se_raw_fit(refit, refit$raw_data$X, bread = bread)$vcov
}

#' @param rerun_out Output of [case_rerun()].
#' @param parameters Optional parameter selector; default all free parameters.
#' @param se Standard-error regime for the leave-one-out covariance: `"standard"`
#'   (naive ML, the default), `"robust.sem"` (Satorra-Bentler sandwich), or
#'   `"robust.huber.white"` (Huber-White/MLR sandwich). Match the `se` of the
#'   reference fit when comparing to semfindr.
#' @return A matrix with one column per selected parameter plus a final `gcd`
#'   column, one row per case, rownames = case ids. Columns are named by user
#'   label when present, else `lhs op rhs`.
#' @export
est_change <- function(rerun_out, parameters = NULL,
                       se = c("standard", "robust.sem", "robust.huber.white")) {
  .case_check_rerun(rerun_out, "est_change()")
  se <- match.arg(se)
  fit <- rerun_out$fit
  fp <- .case_free_table(fit)
  sel <- .case_pars_select(fp, parameters)
  free_k <- fp$k[sel]
  col_names <- fp$label_name[sel]
  theta_full <- fit$theta

  ncase <- length(rerun_out$selected)
  dft <- matrix(NA_real_, ncase, length(sel),
                dimnames = list(rerun_out$case_id, col_names))
  gcd <- rep(NA_real_, ncase)
  for (j in seq_along(rerun_out$rerun)) {
    refit <- rerun_out$rerun[[j]]
    if (is.null(refit)) next
    d <- theta_full[free_k] - refit$theta[free_k]
    Vi <- .case_refit_vcov(refit, se)
    sei <- sqrt(diag(Vi))[free_k]
    dft[j, ] <- d / sei
    Vi_sel_inv <- solve(Vi[free_k, free_k, drop = FALSE])
    gcd[j] <- as.numeric(crossprod(d, Vi_sel_inv %*% d))
  }
  out <- cbind(dft, gcd = gcd)
  rownames(out) <- rerun_out$case_id
  out
}

.case_fm_values <- function(fm, names) {
  vapply(names, function(nm) {
    v <- fm[[nm]]
    if (is.null(v)) NA_real_ else as.numeric(v)[1L]
  }, numeric(1))
}

#' Case influence on fit measures
#'
#' For each case, `(fit measure with all cases) - (fit measure without this
#' case)` for each requested measure. Mirror of
#' `semfindr::fit_measures_change()`.
#'
#' @param rerun_out Output of [case_rerun()].
#' @param fit_measures Names of fit measures (as returned by [fit_measures()]);
#'   default `c("chisq", "cfi", "rmsea", "tli")`.
#' @return A matrix, one row per case, one column per measure, rownames = case
#'   ids.
#' @export
fit_measures_change <- function(rerun_out,
                                fit_measures = c("chisq", "cfi", "rmsea", "tli")) {
  .case_check_rerun(rerun_out, "fit_measures_change()")
  fit <- rerun_out$fit
  fm_full <- .case_fm_values(fit_measures(fit), fit_measures)
  out <- matrix(NA_real_, nrow = length(rerun_out$selected),
                ncol = length(fit_measures),
                dimnames = list(rerun_out$case_id, fit_measures))
  for (j in seq_along(rerun_out$rerun)) {
    refit <- rerun_out$rerun[[j]]
    if (is.null(refit)) next
    fm_i <- .case_fm_values(fit_measures(refit), fit_measures)
    out[j, ] <- fm_full - fm_i
  }
  out
}

#' Mahalanobis distance of each case on the observed variables
#'
#' Model-free leverage measure: `stats::mahalanobis(X, colMeans(X),
#' stats::cov(X))` (N-1 covariance), needing no refit. Mirror of
#' `semfindr::mahalanobis_rerun()`. For multiple-group fits the distance is
#' computed within each group (its own mean/cov); pass the original `data` frame
#' to place the per-group distances back at the original rows (semfindr order).
#'
#' @param fit A `magmaan_fit` or a `magmaan_case_rerun`.
#' @param data Optional original data frame (multiple-group fits, to recover the
#'   original row order). If omitted, multiple-group distances are returned in
#'   group-block order.
#' @return A one-column matrix of Mahalanobis distances, rownames = case ids.
#' @export
mahalanobis_rerun <- function(fit, data = NULL) {
  if (inherits(fit, "magmaan_case_rerun")) fit <- fit$fit
  if (!inherits(fit, "magmaan_fit")) {
    stop("mahalanobis_rerun(): `fit` must be a magmaan_fit or magmaan_case_rerun",
         call. = FALSE)
  }
  md_block <- function(Xg) stats::mahalanobis(Xg, colMeans(Xg), stats::cov(Xg))

  group_var <- fit$group_var %||% ""
  if (is.data.frame(data) && nzchar(group_var)) {
    ov <- colnames(fit$raw_data$X[[1L]])
    g <- as.character(data[[group_var]])
    n <- nrow(data)
    md <- rep(NA_real_, n)
    for (lab in unique(g)) {
      idx <- which(g == lab)
      md[idx] <- md_block(as.matrix(data[idx, ov, drop = FALSE]))
    }
    case_id <- rownames(data)
    if (is.null(case_id)) case_id <- as.character(seq_len(n))
    return(matrix(md, ncol = 1L, dimnames = list(case_id, "md")))
  }

  raw <- fit$raw_data
  if (is.null(raw) || is.null(raw$X)) {
    stop("mahalanobis_rerun(): `fit` does not carry raw data", call. = FALSE)
  }
  # One block (single group) or several (multi-group, group-block order).
  mds <- lapply(raw$X, md_block)
  ids <- lapply(seq_along(raw$X), function(b) {
    rn <- rownames(raw$X[[b]])
    if (is.null(rn)) rn <- as.character(seq_len(nrow(raw$X[[b]])))
    if (length(raw$X) > 1L) paste0("g", b, "_", rn) else rn
  })
  matrix(unlist(mds), ncol = 1L,
         dimnames = list(unlist(ids), "md"))
}

# ---------------------------------------------------------------------------
# Approximate one-step engine (no refit).
#
# The empirical-influence / one-step approximation of the leave-one-out change:
#   θ̂ − θ̂₍ᵢ₎ ≈ (N/(N−1))·V·s_i           (s_i = casewise score, V = full vcov)
# Mirror of semfindr::est_change_raw_approx / est_change_approx. Takes a fitted
# object (not a rerun); single-group continuous ML/ULS/GLS.
# ---------------------------------------------------------------------------

# Per-group raw blocks plus the block-stacked case ids the one-step bindings
# return. The casewise score / influence accessors loop over groups and stack
# their rows group-by-group (group b = `fit$raw_data$X[[b]]`), so the case ids
# follow that order. Single group: the bare rownames. Multiple groups: the
# `g{b}_{rowname}` convention used by `mahalanobis_rerun()` (the per-group raw
# blocks have lost the original global row order, so a within-block label is what
# keeps the rows identifiable without the original data frame). `arg` is what we
# hand the binding: the lone matrix for one group, the per-group list otherwise
# (both accepted by the C++ raw-data marshalling).
.case_raw_blocks <- function(fit) {
  raw <- fit$raw_data
  if (is.null(raw) || is.null(raw$X)) {
    stop("case influence: `fit` does not carry raw data (need fit$raw_data$X)",
         call. = FALSE)
  }
  X <- raw$X
  multigroup <- length(X) > 1L
  ids <- lapply(seq_along(X), function(b) {
    rn <- rownames(X[[b]])
    if (is.null(rn)) rn <- as.character(seq_len(nrow(X[[b]])))
    if (multigroup) paste0("g", b, "_", rn) else rn
  })
  list(arg = if (multigroup) X else X[[1L]],
       case_id = unlist(ids, use.names = FALSE))
}

# Shared one-step ingredients: the per-case raw influence `x0` (one row per
# case; the one-step raw change is (N/(N-1))·x0), the vcov `V` used to
# standardize it, and the case ids. Single- and multiple-group (the bindings
# block-stack the rows group by group; `V` is the full-θ block-diagonal vcov).
#
# `type = "standard"` (default, semfindr-style): x0 = scores · V with the ML
# casewise scores and the expected-information model vcov, so x0 row i = V·s_i.
#
# `type = "estimated.weight"` (frontier; the misspecification-robust dual): x0
# is the complete-sandwich per-case influence c_i for a continuous moment-
# quadratic fit, which carries the data-dependent-weight term; its column-Gram
# `crossprod(x0)` IS the estimated-weight ("complete-sandwich") IJ vcov, used as
# `V` here. `x0_naive` is the fixed-weight counterpart (weight treated as
# constant, as semfindr/Pek-MacCallum implicitly do), so `x0 - x0_naive` is the
# per-case data-dependent-weight diagnostic. Continuous GLS/WLS/ULS (and ordinal
# DWLS/WLSMV) only.
.case_approx_parts <- function(fit, type = c("standard", "estimated.weight")) {
  type <- match.arg(type)
  # Two-stage (ML2S) estimated-weight path: the missing-data member. The
  # influence rides the Stage-1 saturated-moment per-case influence plus the
  # Stage-2 data-dependent-weight term. The Stage-2 weight is encoded in the
  # estimator label ("ML2S" = NT, "ML2S_DWLS"/"_ADF"/"_DLS"/"_WLS" = non-NT);
  # NT treats the weight as fixed (correction zero). Case ids are sequential
  # (block-stacked per-case rows, no original-row map).
  if (type == "estimated.weight" &&
      grepl("^ML2S", toupper(fit$estimator %||% ""))) {
    raw <- fit$raw_data
    if (is.null(raw)) {
      stop("case influence: ML2S fit does not carry $raw_data", call. = FALSE)
    }
    sw <- sub("^ML2S_?", "", toupper(fit$estimator))
    stage2 <- if (nzchar(sw)) tolower(sw) else "nt"
    ij <- magmaan_core$infer_ml2s_casewise_influence_ij_fit(
      fit, raw, stage2_weight = stage2)
    case_id <- as.character(seq_len(nrow(ij$influence)))
    return(list(n = nrow(ij$influence), V = crossprod(ij$influence),
                x0 = ij$influence, x0_naive = ij$influence_naive,
                case_id = case_id, type = type))
  }
  # Ordinal (categorical DWLS/WLS/ULSMV) estimated-weight path: the influence
  # rides the ordinal IJ blocks (per-case polychoric/threshold scores + IF(Ŵ)),
  # not the continuous moment scores, so it routes through the ordinal accessor
  # using the stats the fit carries. Case ids are sequential (the ordinal stats
  # hold per-case influence rows, block-stacked, with no original-row map).
  if (type == "estimated.weight" && isTRUE(fit$ordinal)) {
    stats <- fit$ordinal_stats
    if (is.null(stats)) {
      stop("case influence: ordinal fit does not carry $ordinal_stats",
           call. = FALSE)
    }
    ij <- magmaan_core$infer_ordinal_casewise_influence_ij_fit(fit, stats)
    case_id <- as.character(seq_len(nrow(ij$influence)))
    return(list(n = nrow(ij$influence), V = crossprod(ij$influence),
                x0 = ij$influence, x0_naive = ij$influence_naive,
                case_id = case_id, type = type))
  }
  rb <- .case_raw_blocks(fit)
  if (type == "estimated.weight") {
    ij <- magmaan_core$infer_casewise_influence_ij_fit(fit, rb$arg)
    return(list(n = nrow(ij$influence), V = crossprod(ij$influence),
                x0 = ij$influence, x0_naive = ij$influence_naive,
                case_id = rb$case_id, type = type))
  }
  scores <- magmaan_core$infer_casewise_scores_fit(fit, rb$arg)
  V <- .case_model_vcov(fit)
  list(n = nrow(scores), V = V, x0 = scores %*% V,
       case_id = rb$case_id, type = type)
}

#' Approximate case influence on raw parameter estimates (one-step, no refit)
#'
#' `(N/(N-1))·V·s_i` per case, the one-step approximation of [est_change_raw()]
#' without refitting. Mirror of `semfindr::est_change_raw_approx()`.
#'
#' With `type = "estimated.weight"` this is the misspecification-robust
#' ("complete-sandwich") one-step change for a moment-quadratic fit whose weight
#' is estimated from the data (continuous GLS/WLS, ordinal DWLS/WLSMV; ULS is
#' fixed-weight and so unchanged): the per-case influence carries the
#' data-dependent-weight term that semfindr / Pek & MacCallum drop by treating
#' the estimator weight as fixed. That term is `O_p(N^{-1})` under a correct
#' model but `O_p(N^{-1/2})` under misspecification (Hall-Inoue order promotion),
#' so the two regimes agree at the null and diverge under misfit; it is the
#' casewise dual of the estimated-weight standard error. The result then carries
#' two attributes: `"naive"` (the fixed-weight one-step change) and
#' `"weight_diagnostic"` (their difference, the per-case `Δ'W'_d` term). There is
#' no semfindr counterpart; this is a frontier extension.
#'
#' Both regimes support single- and multiple-group fits; multiple-group rows are
#' block-stacked (`g{b}_{row}` ids, group-suffixed columns).
#'
#' @param fit A fitted `magmaan_fit` carrying raw data (`"standard"`: continuous
#'   ML/ULS/GLS; `"estimated.weight"`: continuous GLS/WLS/ULS or ordinal
#'   DWLS/WLSMV/ULSMV).
#' @param parameters Optional parameter selector; default all free parameters.
#' @param type `"standard"` (default; semfindr-style fixed-weight) or
#'   `"estimated.weight"` (misspecification-robust complete sandwich; estimators
#'   with a data-estimated weight, single- or multiple-group).
#' @return A matrix, one row per case, one column per selected parameter (plus
#'   the `"naive"` / `"weight_diagnostic"` attributes when
#'   `type = "estimated.weight"`).
#' @export
est_change_raw_approx <- function(fit, parameters = NULL,
                                  type = c("standard", "estimated.weight")) {
  type <- match.arg(type)
  fp <- .case_free_table(fit)
  sel <- .case_pars_select(fp, parameters)
  free_k <- fp$k[sel]
  P <- .case_approx_parts(fit, type)
  out <- (P$x0[, free_k, drop = FALSE]) * (P$n / (P$n - 1))
  dimnames(out) <- list(P$case_id, fp$name[sel])
  if (type == "estimated.weight") {
    naive <- (P$x0_naive[, free_k, drop = FALSE]) * (P$n / (P$n - 1))
    dimnames(naive) <- list(P$case_id, fp$name[sel])
    attr(out, "naive") <- naive
    attr(out, "weight_diagnostic") <- out - naive
  }
  out
}

#' Approximate standardized case influence (DFTHETAS) and gCD (one-step)
#'
#' The one-step approximation of [est_change()] without refitting: the
#' approximate raw change `Δ ≈ (N/(N-1))·V·s_i` standardized by the full-sample
#' SE, plus an approximate generalized Cook's distance
#' `gcd_approx = Δ' V_sel^{-1} Δ`.
#'
#' This deliberately uses the *correct* finite-sample scaling, which differs
#' from `semfindr::est_change_approx()` (v0.2.0) by two known constant factors:
#' semfindr applies `N/(N-1)` twice to DFTHETAS (one too many — the factor is
#' already in the raw one-step change) and applies it only once inside gCD (one
#' too few). With matched columns, `est_change_approx(magmaan)` relates to
#' semfindr by `dftheta_magmaan = dftheta_semfindr · (N-1)/N` and
#' `gcd_magmaan = gcd_semfindr · N/(N-1)`. Both factors are O(1/N) and immaterial
#' relative to the one-step error, but they have no first-principles basis (see
#' the influence-function derivation; tracked as an upstream-PR item). The raw
#' [est_change_raw_approx()] is unaffected (semfindr is correct there).
#'
#' @param fit A fitted `magmaan_fit` (continuous ML/ULS/GLS) carrying raw data.
#' @param parameters Optional parameter selector; default all free parameters.
#' @param type `"standard"` (default; semfindr-style fixed-weight) or
#'   `"estimated.weight"` (misspecification-robust complete sandwich; estimators
#'   with a data-estimated weight: continuous GLS/WLS, ordinal DWLS/WLSMV,
#'   single- or multiple-group). Under `"estimated.weight"` both the one-step
#'   change and the SE / `gcd` metric use the complete-sandwich IJ covariance,
#'   the casewise dual of the estimated-weight standard error; see
#'   [est_change_raw_approx()].
#' @return A matrix with one column per selected parameter plus `gcd_approx`,
#'   one row per case.
#' @export
est_change_approx <- function(fit, parameters = NULL,
                              type = c("standard", "estimated.weight")) {
  type <- match.arg(type)
  fp <- .case_free_table(fit)
  sel <- .case_pars_select(fp, parameters)
  free_k <- fp$k[sel]
  P <- .case_approx_parts(fit, type)
  n <- P$n
  se <- sqrt(diag(P$V))
  # xr = one-step raw change ≈ θ̂ − θ̂₍ᵢ₎ = (N/(N-1))·V·s_i, selected columns.
  xr <- (P$x0[, free_k, drop = FALSE]) * (n / (n - 1))
  dft <- sweep(xr, 2L, se[free_k], "/")                 # DFTHETAS = Δ / SE
  # gcd_approx = Δ' V_sel^{-1} Δ — the one-step form of est_change()'s gcd, with
  # the full-sample vcov submatrix inverted (matching the exact engine).
  Vsel_inv <- solve(P$V[free_k, free_k, drop = FALSE])
  gcd_approx <- rowSums((xr %*% Vsel_inv) * xr)
  out <- cbind(dft, gcd_approx = gcd_approx)
  dimnames(out) <- list(P$case_id, c(fp$name[sel], "gcd_approx"))
  out
}
