# LRT (refit-based) modification indices and equality-release score tests.
#
# Two refit counterparts to the one-step post-fit sweeps, mirroring the
# established pair:
#   * modification_indices_lrt() -- the LRT of modification_indices() /
#     lavaan modindices(): for each parameter ABSENT from the model (cross
#     loadings, residual covariances) refit with it freed and report the nested
#     chi-square difference plus the EXACT refitted parameter change.
#   * score_tests_lrt() -- the LRT of score_tests() / lavaan lavTestScore():
#     for each cross-group EQUALITY tie, refit the released model and run the
#     nested Satorra-2000 difference test (with the robust reference-law family).
#
# Both are thin compositions of magmaan() and the existing nested tests; they
# carry no SEM logic of their own. The refit isolates each release exactly even
# under structural misspecification elsewhere, where a one-step score statistic
# leaks. See docs/research/notes/lrt_modification_indices.tex.

# ---- fixed/absent releases: the lavaan modindices() table, by LRT -----------

modification_indices_lrt <- function(fit, data,
                                     candidates = c("all", "loadings",
                                                    "covariances"),
                                     robust = TRUE) {
  candidates <- match.arg(candidates)
  if (!inherits(fit, "magmaan_fit")) {
    stop("modification_indices_lrt(): `fit` must be a fitted magmaan model.",
         call. = FALSE)
  }
  if (is.null(fit$model) || is.null(fit$model$syntax)) {
    stop("modification_indices_lrt(): `fit` is missing its model spec; fit with ",
         "magmaan() so the augmented models can be reconstructed.", call. = FALSE)
  }
  if (toupper(fit$estimator %||% "ML") == "FIML") {
    stop("modification_indices_lrt(): FIML is not yet supported here (the ",
         "incomplete-data chi-square baseline differs); for missing data use ",
         "the one-step modification_indices().", call. = FALSE)
  }

  ngroups <- fit$ngroups %||% 1L
  ops <- switch(candidates,
    all         = c("=~", "~~"),
    loadings    = "=~",
    covariances = "~~")

  # Reuse the one-step sweep for both the candidate enumeration and the
  # comparison column (lavaan-style mi / epc).
  one_step <- modification_indices(fit)
  cand <- one_step[one_step$op %in% ops, , drop = FALSE]
  if (!nrow(cand)) {
    stop("modification_indices_lrt(): modification_indices() found no absent ",
         "'", candidates, "' parameters to add.", call. = FALSE)
  }

  T0 <- infer_chi2_stat(fit_sample_stats(fit), fit$fmin)
  # Per-group raw data (ov order) for the observed-Hessian profile difference
  # test, the misspecification-robust reference law (ml_profile_lrt).
  data_list <- if (isTRUE(robust)) .lrt_group_data(fit, data) else NULL

  rows <- lapply(seq_len(nrow(cand)), function(i) {
    row <- cand[i, ]
    line <- .lrt_aug_line(row, ngroups)
    rel <- tryCatch(.lrt_refit(fit, data, extra_syntax = line),
                    error = function(e) e)
    if (inherits(rel, "error") || !isTRUE(rel$converged)) {
      lrt <- NA_real_; lrt_p <- NA_real_; lrt_p_obs <- NA_real_
      epc_lrt <- NA_real_
    } else {
      T1 <- infer_chi2_stat(fit_sample_stats(rel), rel$fmin)
      lrt <- T0 - T1
      lrt_p <- stats::pchisq(lrt, df = 1L, lower.tail = FALSE)
      epc_lrt <- .lrt_added_est(rel$partable, row)
      # Observed-bread (model-misspecification-robust) reference law: the exact
      # eigenvalue-mixture tail of the profile-Hessian difference spectrum. We
      # use p_mixture, not the mean-scaled p_scaled, because the profile contrast
      # spreads over the moment space (spectrum_size != df_diff for a 1-df add).
      lrt_p_obs <- if (isTRUE(robust)) {
        pr <- tryCatch(infer_ml_profile_lrt(rel, fit, data_list),
                       error = function(e) NULL)
        if (is.null(pr)) NA_real_ else pr$p_mixture
      } else NA_real_
    }
    data.frame(
      lhs = row$lhs, op = row$op, rhs = row$rhs, group = row$group, df = 1L,
      mi = row$mi, mi_p = stats::pchisq(row$mi, df = 1L, lower.tail = FALSE),
      lrt = lrt, lrt_p = lrt_p, lrt_p_obs = lrt_p_obs,
      epc = row$epc, epc_lrt = epc_lrt,
      stringsAsFactors = FALSE)
  })

  out <- do.call(rbind, rows)
  out <- out[order(-out$lrt), , drop = FALSE]
  rownames(out) <- NULL
  attr(out, "robust") <- isTRUE(robust)
  class(out) <- c("magmaan_mi_lrt", "data.frame")
  out
}

# Augmented-model syntax line that frees one absent parameter. Single group: a
# plain "lhs op rhs" row. Multi-group: free it in the candidate's group only,
# via a per-group "c(0,..,NA,..,0)*rhs" modifier (the others stay fixed at 0),
# matching lavaan's per-group modindices.
.lrt_aug_line <- function(row, ngroups) {
  if (ngroups <= 1L) return(paste(row$lhs, row$op, row$rhs))
  vec <- rep("0", ngroups)
  vec[row$group] <- "NA"
  paste0(row$lhs, " ", row$op, " c(", paste(vec, collapse = ","), ")*", row$rhs)
}

# Exact EPC: the refitted estimate of the added parameter, in its group.
.lrt_added_est <- function(pt_rel, row) {
  sel <- pt_rel$op == row$op & pt_rel$lhs == row$lhs & pt_rel$rhs == row$rhs &
    pt_rel$group == row$group & (pt_rel$free %||% 0L) > 0L
  est <- pt_rel$est[sel]
  if (length(est)) est[1] else NA_real_
}

print.magmaan_mi_lrt <- function(x, ...) {
  cat("LRT modification indices (nested chi-square difference vs one-step mi)\n")
  print(as.data.frame(x), ...)
  invisible(x)
}

# ---- equality releases: the score_tests() sweep, by LRT ---------------------

score_tests_lrt <- function(fit, data,
                            candidates = c("loadings", "intercepts", "all"),
                            method = c("restriction_map", "lavaan_sb2001",
                                       "lavaan_sb2010"),
                            gamma = c("empirical", "NT")) {
  candidates <- match.arg(candidates)
  method <- match.arg(method)
  gamma <- match.arg(gamma)

  if (!inherits(fit, "magmaan_fit")) {
    stop("score_tests_lrt(): `fit` must be a fitted magmaan model.", call. = FALSE)
  }
  model <- fit$model
  if (is.null(model) || is.null(model$syntax)) {
    stop("score_tests_lrt(): `fit` is missing its model spec; fit the anchor ",
         "with magmaan() so the released models can be reconstructed.",
         call. = FALSE)
  }
  group_var   <- fit$group_var
  group_equal <- model$group_equal
  if (is.null(group_var) || !nzchar(group_var) || (fit$ngroups %||% 1L) < 2L) {
    stop("score_tests_lrt(): the anchor must be a multi-group fit with ",
         "cross-group equality constraints.", call. = FALSE)
  }
  if (is.null(group_equal) || !length(group_equal)) {
    stop("score_tests_lrt(): the anchor has no `group_equal` constraints to ",
         "release.", call. = FALSE)
  }

  fam <- .lrt_eq_candidates(fit$partable, candidates)
  if (!nrow(fam)) {
    stop("score_tests_lrt(): found no releasable cross-group equalities of ",
         "kind '", candidates, "'.", call. = FALSE)
  }

  is_fiml <- toupper(fit$estimator %||% "ML") == "FIML"
  data_list <- if (is_fiml) NULL else .lrt_group_data(fit, data)

  released <- vector("list", nrow(fam))
  rows <- lapply(seq_len(nrow(fam)), function(i) {
    rel <- .lrt_refit(fit, data, extra_partial = fam$token[i])
    nt <- if (is_fiml) {
      robust_nested_lrt(rel, fit, method = method, gamma = gamma)
    } else {
      robust_nested_lrt(rel, fit, data = data_list, method = method,
                        gamma = gamma)
    }
    est <- .lrt_eq_released_est(rel$partable, fam[i, ])
    released[[i]] <<- est
    data.frame(
      lhs = fam$lhs[i], op = fam$op[i], rhs = fam$rhs[i],
      df = nt$df_diff %||% nt$df %||% NA_integer_,
      lrt = nt$T_diff %||% NA_real_,
      p_unscaled = nt$p_unscaled %||% NA_real_,
      p_scaled = nt$p_scaled %||% NA_real_,
      p_mixture = nt$p_mixture %||% NA_real_,
      epc_range = if (length(est$est) > 1L) diff(range(est$est)) else NA_real_,
      stringsAsFactors = FALSE)
  })

  out <- do.call(rbind, rows)
  ord <- order(-out$lrt)
  out <- out[ord, , drop = FALSE]
  rownames(out) <- NULL
  attr(out, "released_estimates") <- released[ord]
  attr(out, "method") <- method
  attr(out, "gamma") <- gamma
  attr(out, "conditioned_on") <- model$group_partial
  class(out) <- c("magmaan_score_lrt", "data.frame")
  out
}

.lrt_eq_candidates <- function(pt, which) {
  ops <- switch(which, loadings = "=~", intercepts = "~1", all = c("=~", "~1"))
  lab <- pt$label %||% rep("", nrow(pt))
  keep <- pt$op %in% ops & (pt$free %||% 0L) > 0L & nzchar(lab)
  pt2 <- pt[keep, , drop = FALSE]
  empty <- data.frame(lhs = character(), op = character(), rhs = character(),
                      label = character(), token = character(),
                      stringsAsFactors = FALSE)
  if (!nrow(pt2)) return(empty)
  n_groups <- tapply(pt2$group, pt2$label, function(g) length(unique(g)))
  tied <- names(n_groups)[n_groups >= 2L]
  pt2 <- pt2[pt2$label %in% tied, , drop = FALSE]
  if (!nrow(pt2)) return(empty)
  fam <- pt2[!duplicated(pt2$label), c("lhs", "op", "rhs", "label"),
             drop = FALSE]
  fam$token <- ifelse(fam$op == "~1",
                      paste0(fam$lhs, "~1"),
                      paste0(fam$lhs, fam$op, fam$rhs))
  fam
}

.lrt_eq_released_est <- function(pt_rel, fam_row) {
  sel <- pt_rel$op == fam_row$op & pt_rel$lhs == fam_row$lhs &
    (fam_row$op == "~1" | pt_rel$rhs == fam_row$rhs) & (pt_rel$free %||% 0L) > 0L
  sub <- pt_rel[sel, , drop = FALSE]
  list(group = sub$group, est = sub$est)
}

print.magmaan_score_lrt <- function(x, ...) {
  method <- attr(x, "method"); gamma <- attr(x, "gamma")
  if (!is.null(method)) {
    cat("LRT equality-release score tests (", method, ", gamma=", gamma, ")\n",
        sep = "")
  }
  cond <- attr(x, "conditioned_on")
  if (length(cond)) {
    cat("conditioned on (group_partial baseline): ",
        paste(cond, collapse = ", "), "\n", sep = "")
  }
  print(as.data.frame(x), ...)
  invisible(x)
}

# ---- shared helpers ---------------------------------------------------------

# Refit a model derived from `fit`'s anchor: append `extra_syntax` rows and/or
# add `extra_partial` group_partial tokens, preserving the anchor's estimator,
# grouping, group_equal, existing group_partial, and model options.
.lrt_refit <- function(fit, data, extra_syntax = "", extra_partial = NULL) {
  model <- fit$model
  syntax <- model$syntax
  if (nzchar(extra_syntax)) syntax <- paste0(syntax, "\n", extra_syntax)
  valid <- c("auto_var", "auto_cov_lv_x", "auto_cov_y", "orthogonal",
             "auto_fix_first", "auto_fix_single", "std_lv", "effect_coding",
             "fixed_x", "meanstructure", "model_type", "parameterization")
  mo <- fit$options$model_options %||% list()
  mo <- mo[intersect(names(mo), valid)]
  partial <- unique(c(model$group_partial, extra_partial))
  do.call(magmaan, c(
    list(model = syntax, data = data,
         estimator = toupper(fit$estimator %||% "ML"),
         groups = fit$group_var,
         group_equal = model$group_equal,
         group_partial = if (length(partial)) partial else NULL),
    mo))
}

.lrt_group_data <- function(fit, data) {
  ov <- if (is.list(fit$ov_names)) fit$ov_names[[1]] else fit$ov_names
  if (is.list(data) && !is.data.frame(data)) {
    return(lapply(data, function(d) as.matrix(d[, ov, drop = FALSE])))
  }
  gv <- fit$group_var
  if (is.null(gv) || !nzchar(gv) || (fit$ngroups %||% 1L) < 2L) {
    return(list(as.matrix(data[, ov, drop = FALSE])))
  }
  if (!gv %in% names(data)) {
    stop("modification_indices_lrt(): `data` must contain the grouping column '",
         gv, "', or be a list of per-group matrices.", call. = FALSE)
  }
  lapply(fit$group_labels, function(g)
    as.matrix(data[as.character(data[[gv]]) == g, ov, drop = FALSE]))
}
