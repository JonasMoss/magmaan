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
  data.frame(k = k, row = fr,
             lhs = ptab$lhs[fr], op = ptab$op[fr], rhs = ptab$rhs[fr],
             name = name, label_name = label_name, stringsAsFactors = FALSE)
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
#' Mirror of `semfindr::lavaan_rerun()`. Single-group continuous ML/ULS/GLS only.
#'
#' @param fit A fitted `magmaan_fit` (continuous ML/ULS/GLS) carrying raw data.
#' @param to_rerun Integer case indices to refit; default all cases.
#' @param warm_start Seed each refit from the full-sample estimates (faster, and
#'   keeps refits on the same optimum). Default `TRUE`.
#' @return A `magmaan_case_rerun` list: `rerun` (per-case refit fits, `NULL` when
#'   a refit fails or is inadmissible), `fit`, `selected`, `case_id`, `converged`.
#' @export
case_rerun <- function(fit, to_rerun = NULL, warm_start = TRUE) {
  if (!inherits(fit, "magmaan_fit")) {
    stop("case_rerun(): `fit` must be a magmaan_fit", call. = FALSE)
  }
  estimator <- .case_estimator(fit)
  fit_fun <- .case_fit_fun(estimator)
  raw <- fit$raw_data
  if (is.null(raw) || is.null(raw$X)) {
    stop("case_rerun(): `fit` does not carry raw data (need fit$raw_data$X)",
         call. = FALSE)
  }
  if (length(raw$X) != 1L) {
    stop("case_rerun(): multiple-group leave-one-out is not supported yet; ",
         "single-group fits only", call. = FALSE)
  }
  X <- raw$X[[1L]]
  n <- nrow(X)
  selected <- if (is.null(to_rerun)) seq_len(n) else as.integer(to_rerun)
  if (any(selected < 1L | selected > n)) {
    stop("case_rerun(): `to_rerun` out of range 1:", n, call. = FALSE)
  }
  case_id <- rownames(X)
  if (is.null(case_id)) case_id <- as.character(seq_len(n))
  scaling <- fit$raw_data$scaling %||% "n"
  spec_warm <- .case_warm_spec(fit, warm_start)

  reruns <- vector("list", length(selected))
  converged <- logical(length(selected))
  for (j in seq_along(selected)) {
    i <- selected[j]
    refit <- tryCatch({
      data_i <- .case_loo_data(X[-i, , drop = FALSE], spec_warm, scaling)
      fit_fun(spec_warm, data_i)
    }, error = function(e) NULL)
    ok <- !is.null(refit) && isTRUE(refit$converged)
    converged[j] <- ok
    reruns[[j]] <- if (ok) refit else NULL
  }

  out <- list(rerun = reruns, fit = fit, selected = selected,
              case_id = case_id[selected], converged = converged,
              estimator = estimator)
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
#' @param rerun_out Output of [case_rerun()].
#' @param parameters Optional parameter selector; default all free parameters.
#' @return A matrix with one column per selected parameter plus a final `gcd`
#'   column, one row per case, rownames = case ids. Columns are named by user
#'   label when present, else `lhs op rhs`.
#' @export
est_change <- function(rerun_out, parameters = NULL) {
  .case_check_rerun(rerun_out, "est_change()")
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
    Vi <- .case_model_vcov(refit)
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
#' `semfindr::mahalanobis_rerun()`. Single-group continuous fits.
#'
#' @param fit A `magmaan_fit` or a `magmaan_case_rerun`.
#' @return A one-column matrix of Mahalanobis distances, rownames = case ids.
#' @export
mahalanobis_rerun <- function(fit) {
  if (inherits(fit, "magmaan_case_rerun")) fit <- fit$fit
  if (!inherits(fit, "magmaan_fit")) {
    stop("mahalanobis_rerun(): `fit` must be a magmaan_fit or magmaan_case_rerun",
         call. = FALSE)
  }
  raw <- fit$raw_data
  if (is.null(raw) || is.null(raw$X)) {
    stop("mahalanobis_rerun(): `fit` does not carry raw data", call. = FALSE)
  }
  if (length(raw$X) != 1L) {
    stop("mahalanobis_rerun(): multiple-group fits not supported yet", call. = FALSE)
  }
  X <- raw$X[[1L]]
  case_id <- rownames(X)
  if (is.null(case_id)) case_id <- as.character(seq_len(nrow(X)))
  md <- stats::mahalanobis(X, colMeans(X), stats::cov(X))
  matrix(md, ncol = 1L, dimnames = list(case_id, "md"))
}

# ---------------------------------------------------------------------------
# Approximate one-step engine (no refit).
#
# The empirical-influence / one-step approximation of the leave-one-out change:
#   θ̂ − θ̂₍ᵢ₎ ≈ (N/(N−1))·V·s_i           (s_i = casewise score, V = full vcov)
# Mirror of semfindr::est_change_raw_approx / est_change_approx. Takes a fitted
# object (not a rerun); single-group continuous ML/ULS/GLS.
# ---------------------------------------------------------------------------

.case_raw_matrix <- function(fit) {
  raw <- fit$raw_data
  if (is.null(raw) || is.null(raw$X)) {
    stop("case influence: `fit` does not carry raw data (need fit$raw_data$X)",
         call. = FALSE)
  }
  if (length(raw$X) != 1L) {
    stop("case influence: multiple-group approximation not supported yet",
         call. = FALSE)
  }
  raw$X[[1L]]
}

# Shared one-step ingredients: per-case scores (N×npar), the full-sample vcov,
# and the approximate raw change x0 = scores · V (one row per case).
.case_approx_parts <- function(fit) {
  X <- .case_raw_matrix(fit)
  scores <- magmaan_core$infer_casewise_scores_fit(fit, X)
  V <- .case_model_vcov(fit)
  case_id <- rownames(X)
  if (is.null(case_id)) case_id <- as.character(seq_len(nrow(X)))
  list(n = nrow(X), scores = scores, V = V, x0 = scores %*% V, case_id = case_id)
}

#' Approximate case influence on raw parameter estimates (one-step, no refit)
#'
#' `(N/(N-1))·V·s_i` per case, the one-step approximation of [est_change_raw()]
#' without refitting. Mirror of `semfindr::est_change_raw_approx()`.
#'
#' @param fit A fitted `magmaan_fit` (continuous ML/ULS/GLS) carrying raw data.
#' @param parameters Optional parameter selector; default all free parameters.
#' @return A matrix, one row per case, one column per selected parameter.
#' @export
est_change_raw_approx <- function(fit, parameters = NULL) {
  fp <- .case_free_table(fit)
  sel <- .case_pars_select(fp, parameters)
  free_k <- fp$k[sel]
  P <- .case_approx_parts(fit)
  out <- (P$x0[, free_k, drop = FALSE]) * (P$n / (P$n - 1))
  dimnames(out) <- list(P$case_id, fp$name[sel])
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
#' @return A matrix with one column per selected parameter plus `gcd_approx`,
#'   one row per case.
#' @export
est_change_approx <- function(fit, parameters = NULL) {
  fp <- .case_free_table(fit)
  sel <- .case_pars_select(fp, parameters)
  free_k <- fp$k[sel]
  P <- .case_approx_parts(fit)
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
