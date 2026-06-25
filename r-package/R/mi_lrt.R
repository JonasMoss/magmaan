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
                                     robust = TRUE, weight = NULL) {
  candidates <- match.arg(candidates)
  if (!inherits(fit, "magmaan_fit")) {
    stop("modification_indices_lrt(): `fit` must be a fitted magmaan model.",
         call. = FALSE)
  }
  if (is.null(fit$model) || is.null(fit$model$syntax)) {
    stop("modification_indices_lrt(): `fit` is missing its model spec; fit with ",
         "magmaan() so the augmented models can be reconstructed.", call. = FALSE)
  }
  # FIML / ML2S use an incomplete-data baseline: the model-vs-saturated chi-square
  # is not 2N*fmin, so the plain lrt/lrt_p come from the profile-LRT's own T_diff /
  # p_unscaled (the saturated term cancels in the difference), not T0 - T1.
  est <- toupper(fit$estimator %||% "ML")
  use_profile_diff <- isTRUE(fit$fiml) || est %in% c("FIML", "ML2S")
  # Continuous WLS keeps no weight on the fit, so the same fitting `weight=` (W)
  # must be supplied to refit the augmented models and build the profile-LRT.
  if (est == "WLS" && is.null(weight)) {
    stop("modification_indices_lrt(): continuous WLS requires the fitting ",
         "`weight=` (the W matrix); it is not retained on the fit.",
         call. = FALSE)
  }

  ngroups <- fit$ngroups %||% 1L
  ops <- switch(candidates,
    all         = c("=~", "~~"),
    loadings    = "=~",
    covariances = "~~")

  # Reuse the one-step sweep for both the candidate enumeration and the
  # comparison column (lavaan-style mi / epc). WLS needs the weight here too.
  one_step <- if (is.null(weight)) modification_indices(fit) else
    modification_indices(fit, weight = weight)
  cand <- one_step[one_step$op %in% ops, , drop = FALSE]
  if (!nrow(cand)) {
    stop("modification_indices_lrt(): modification_indices() found no absent ",
         "'", candidates, "' parameters to add.", call. = FALSE)
  }

  # Complete-data chi-square baseline (2N*fmin); skipped for FIML/ML2S.
  T0 <- if (use_profile_diff) NA_real_ else
    infer_chi2_stat(fit_sample_stats(fit), fit$fmin)
  # Per-group raw data (ov order) for the ML/continuous-LS observed-Hessian
  # profile difference test (the empirical Gamma is built from it). The
  # categorical branches of .lrt_profile_obj() share the anchor's polychoric stage
  # via fit$*_stats instead, and FIML/ML2S read fit$raw_data, so only the
  # complete-data ML/ULS/GLS/WLS branches consume this list.
  need_raw_data <- isTRUE(robust) && est %in% c("ML", "ULS", "GLS", "WLS")
  data_list <- if (need_raw_data) .lrt_group_data(fit, data) else NULL

  rows <- lapply(seq_len(nrow(cand)), function(i) {
    row <- cand[i, ]
    line <- .lrt_aug_line(row, ngroups)
    rel <- tryCatch(.lrt_refit(fit, data, extra_syntax = line, weight = weight),
                    error = function(e) e)
    if (inherits(rel, "error") || !isTRUE(rel$converged)) {
      lrt <- NA_real_; lrt_p <- NA_real_; lrt_p_obs <- NA_real_
      epc_lrt <- NA_real_; sepc_lrt <- NA_real_
    } else {
      added <- .lrt_added_est(rel$partable, row)
      epc_lrt <- added$est
      sepc_lrt <- .lrt_std_epc(rel, added$free)
      if (use_profile_diff) {
        # FIML/ML2S: one profile-LRT call yields the plain difference (T_diff /
        # p_unscaled; the incomplete-data saturated term cancels) and the robust
        # mixture tail (p_mixture).
        pr <- tryCatch(.lrt_profile_obj(rel, fit, data_list),
                       error = function(e) NULL)
        lrt   <- if (is.null(pr)) NA_real_ else pr$T_diff
        lrt_p <- if (is.null(pr)) NA_real_ else pr$p_unscaled
        lrt_p_obs <- if (isTRUE(robust) && !is.null(pr)) pr$p_mixture else NA_real_
      } else {
        T1 <- infer_chi2_stat(fit_sample_stats(rel), rel$fmin)
        lrt <- T0 - T1
        lrt_p <- stats::pchisq(lrt, df = 1L, lower.tail = FALSE)
        # Observed-bread (model-misspecification-robust) reference law (p_mixture),
        # dispatched on the anchor's estimator/data type (see .lrt_profile_obj).
        lrt_p_obs <- if (isTRUE(robust)) {
          pr <- tryCatch(.lrt_profile_obj(rel, fit, data_list, weight = weight),
                         error = function(e) NULL)
          if (is.null(pr)) NA_real_ else pr$p_mixture
        } else NA_real_
      }
    }
    data.frame(
      lhs = row$lhs, op = row$op, rhs = row$rhs, group = row$group, df = 1L,
      mi = row$mi, mi_p = stats::pchisq(row$mi, df = 1L, lower.tail = FALSE),
      lrt = lrt, lrt_p = lrt_p, lrt_p_obs = lrt_p_obs,
      epc = row$epc, epc_lrt = epc_lrt, sepc_lrt = sepc_lrt,
      stringsAsFactors = FALSE)
  })

  out <- do.call(rbind, rows)
  out <- out[order(-out$lrt), , drop = FALSE]
  rownames(out) <- NULL
  attr(out, "robust") <- isTRUE(robust)
  class(out) <- c("magmaan_mi_lrt", "data.frame")
  out
}

# Observed-bread (model-misspecification-robust) profile-LRT object for one freed
# parameter, dispatched on the anchor's estimator/data type. Returns the full
# WeightedProfileLRTResult list (or NULL if the estimator has no wired path), so
# callers can read p_mixture (the robust reference law for lrt_p_obs) and, for the
# missing-data estimators, T_diff / p_unscaled (the plain difference, whose
# incomplete-data saturated term cancels). `rel` is the released (H1) fit, `fit`
# the anchor (H0); the categorical branches share the anchor's polychoric stage
# via fit$ordinal_stats / fit$mixed_ordinal_stats, the ML/continuous branches the
# per-group raw data, and FIML/ML2S read fit$raw_data. p_mixture (not the
# mean-scaled p_scaled) because the profile contrast spreads over the moment space
# (spectrum_size != df_diff for a 1-df add). The ordinal/mixed bindings are
# DWLS-only, so a non-DWLS ordinal anchor returns NULL here; the plain lrt/lrt_p
# columns still report for any complete-data estimator. Continuous WLS needs the
# caller's `weight` (the W matrix) since it is not retained on the fit.
.lrt_profile_obj <- function(rel, fit, data_list, weight = NULL,
                             eig_tol = 1e-10) {
  est <- toupper(fit$estimator %||% "ML")
  if (isTRUE(fit$ordinal) && est == "DWLS") {
    infer_ordinal_profile_lrt(rel, fit, fit$ordinal_stats, eig_tol)
  } else if (isTRUE(fit$mixed_ordinal) && est == "DWLS") {
    infer_mixed_ordinal_profile_lrt(rel, fit, fit$mixed_ordinal_stats, eig_tol)
  } else if (est %in% c("ULS", "GLS", "WLS")) {
    infer_continuous_ls_profile_lrt(rel, fit, data_list, weight = weight,
                                    eig_tol = eig_tol)
  } else if (isTRUE(fit$fiml) || est == "FIML") {
    infer_fiml_profile_lrt(rel, fit, eig_tol)
  } else if (est == "ML2S") {
    infer_two_stage_nt_profile_lrt(rel, fit, eig_tol)
  } else if (est == "ML") {
    infer_ml_profile_lrt(rel, fit, data_list)
  } else {
    NULL
  }
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

# Exact EPC: the refitted estimate of the added parameter (and its free index),
# in its group.
.lrt_added_est <- function(pt_rel, row) {
  sel <- pt_rel$op == row$op & pt_rel$lhs == row$lhs & pt_rel$rhs == row$rhs &
    pt_rel$group == row$group & (pt_rel$free %||% 0L) > 0L
  i <- which(sel)
  if (!length(i)) return(list(est = NA_real_, free = NA_integer_))
  list(est = pt_rel$est[i[1]], free = pt_rel$free[i[1]])
}

# Standardized exact EPC: the std.all value of the freed parameter in the
# released fit (the exact analog of lavaan's sepc.all). The std *point* values do
# not use the vcov (only the SEs do), so a correctly-shaped zero matrix satisfies
# standardized()'s shape check at no inference cost.
.lrt_std_epc <- function(rel, free_index) {
  if (is.na(free_index)) return(NA_real_)
  np <- max(rel$partable$free)
  std <- tryCatch(standardized(rel, matrix(0, np, np), type = "all"),
                  error = function(e) NULL)
  if (is.null(std) || is.null(std$theta) || free_index > length(std$theta)) {
    return(NA_real_)
  }
  std$theta[free_index]
}

print.magmaan_mi_lrt <- function(x, ...) {
  cat("LRT modification indices (nested chi-square difference vs one-step mi)\n")
  sc <- attr(x, "mi_screen")
  if (!is.null(sc) && "verdict" %in% names(x)) {
    cat(sprintf("  verdict = %s-adjusted %s (alpha=%.3g) x |%s| >= %.2f\n",
                sc$adjust, sc$p, sc$alpha, sc$effect, sc$effect_min))
  }
  print(as.data.frame(x), ...)
  invisible(x)
}

# ---- decision layer: multiplicity + significant x substantial verdict -------

# Add a multiplicity-adjusted p-value and a verdict to an MI table
# (modification_indices_lrt or score_tests_lrt). Keys off named columns, so the
# same helper screens either table. The verdict is the Saris-Satorra-Sorbom 2x2
# on (adjusted p <= alpha) x (|effect| >= effect_min):
#   free          significant and substantively large -> a real release
#   trivial       significant but a negligible effect  -> large-N artifact
#   underpowered  large effect but not significant      -> flag, do not free
#   ok            neither                               -> leave fixed
# Rows with a missing p or effect get verdict NA. With p/effect = NULL the robust
# columns are auto-resolved: lrt_p_obs/sepc_lrt for modification_indices_lrt,
# p_mixture/epc_range for score_tests_lrt; BH is the default (Thissen-style FDR).
mi_screen <- function(x, p = NULL, effect = NULL,
                      alpha = 0.05, effect_min = 0.10, adjust = "BH") {
  if (!is.data.frame(x)) {
    stop("mi_screen(): `x` must be a modification-index table (data.frame).",
         call. = FALSE)
  }
  # Resolve the robust p and effect columns: lrt_p_obs/sepc_lrt for
  # modification_indices_lrt, p_mixture/epc_range for score_tests_lrt.
  if (is.null(p)) {
    p <- intersect(c("lrt_p_obs", "p_mixture"), names(x))[1]
    if (is.na(p)) {
      stop("mi_screen(): no robust p-value column found; pass `p=`.",
           call. = FALSE)
    }
  } else if (!p %in% names(x)) {
    stop("mi_screen(): p-value column '", p, "' not found in `x`.", call. = FALSE)
  }
  if (is.null(effect)) {
    effect <- intersect(c("sepc_lrt", "epc_range"), names(x))[1]
    if (is.na(effect)) {
      stop("mi_screen(): no effect column found; pass `effect=`.", call. = FALSE)
    }
  } else if (!effect %in% names(x)) {
    stop("mi_screen(): effect column '", effect, "' not found in `x`.",
         call. = FALSE)
  }
  padj <- stats::p.adjust(x[[p]], method = adjust)
  eff  <- x[[effect]]
  ok   <- !is.na(padj) & !is.na(eff)
  sig  <- padj <= alpha
  big  <- abs(eff) >= effect_min
  verdict <- rep(NA_character_, nrow(x))
  verdict[ok] <- ifelse(sig[ok] & big[ok], "free",
                 ifelse(sig[ok] & !big[ok], "trivial",
                 ifelse(!sig[ok] & big[ok], "underpowered", "ok")))
  x[[paste0(p, "_", tolower(adjust))]] <- padj
  x$verdict <- verdict
  attr(x, "mi_screen") <- list(p = p, effect = effect, alpha = alpha,
                               effect_min = effect_min, adjust = adjust)
  x
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
.lrt_refit <- function(fit, data, extra_syntax = "", extra_partial = NULL,
                       weight = NULL) {
  model <- fit$model
  syntax <- model$syntax
  if (nzchar(extra_syntax)) syntax <- paste0(syntax, "\n", extra_syntax)
  # "ordered" is essential: without it an ordinal anchor refits as continuous and
  # the DWLS/WLS/ULS path errors ("requires ordered variables"); "parameterization"
  # keeps the released fit on the anchor's theta/delta convention (the profile-LRT
  # binding requires H1/H0 to match).
  valid <- c("auto_var", "auto_cov_lv_x", "auto_cov_y", "orthogonal",
             "auto_fix_first", "auto_fix_single", "std_lv", "effect_coding",
             "fixed_x", "meanstructure", "model_type", "ordered",
             "parameterization")
  mo <- fit$options$model_options %||% list()
  mo <- mo[intersect(names(mo), valid)]
  partial <- unique(c(model$group_partial, extra_partial))
  do.call(magmaan, c(
    list(model = syntax, data = data,
         estimator = toupper(fit$estimator %||% "ML"),
         groups = fit$group_var,
         group_equal = model$group_equal,
         group_partial = if (length(partial)) partial else NULL),
    if (!is.null(weight)) list(W = weight),  # continuous WLS augmented refit
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
