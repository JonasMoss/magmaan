# LRT-based (refit) modification indices for cross-group measurement-invariance
# equality releases.
#
# For each parameter family tied across groups in a constrained `group_equal`
# anchor fit, refit the released model (that one family freed via `group_partial`)
# and run the nested Satorra-2000 difference test, returning the full robust
# reference-law family (naive / Satorra-Bentler / mean-var / scaled-shifted /
# exact mixture) plus the EXACT refitted parameter change. This is the refit
# counterpart to the one-step `score_tests()` modification index: it isolates
# each release exactly even when the model is structurally misspecified elsewhere
# (see docs/research/notes/lrt_modification_indices.tex).
#
# It is a thin composition of the existing one-call estimator (`magmaan`) and
# nested test (`robust_nested_lrt`); it carries no SEM logic of its own. If the
# anchor already has a `group_partial` baseline, that baseline is preserved in
# every released refit, which gives the conditional (structural-robust)
# localization for free.

modification_indices_lrt <- function(fit, data,
                                     candidates = c("loadings", "intercepts",
                                                    "all"),
                                     method = c("restriction_map",
                                                "lavaan_sb2001", "lavaan_sb2010"),
                                     gamma = c("empirical", "NT")) {
  candidates <- match.arg(candidates)
  method <- match.arg(method)
  gamma <- match.arg(gamma)

  if (!inherits(fit, "magmaan_fit")) {
    stop("modification_indices_lrt(): `fit` must be a fitted magmaan model.",
         call. = FALSE)
  }
  model <- fit$model
  if (is.null(model) || is.null(model$syntax)) {
    stop("modification_indices_lrt(): `fit` is missing its model spec; fit the ",
         "anchor with magmaan() so the released models can be reconstructed.",
         call. = FALSE)
  }

  syntax       <- model$syntax
  estimator    <- toupper(fit$estimator %||% "ML")
  group_var    <- fit$group_var
  group_equal  <- model$group_equal
  base_partial <- model$group_partial   # preserved -> conditional localization

  if (is.null(group_var) || !nzchar(group_var) || (fit$ngroups %||% 1L) < 2L) {
    stop("modification_indices_lrt(): the anchor must be a multi-group fit ",
         "with cross-group equality constraints.", call. = FALSE)
  }
  if (is.null(group_equal) || !length(group_equal)) {
    stop("modification_indices_lrt(): the anchor has no `group_equal` ",
         "constraints to release.", call. = FALSE)
  }

  fam <- .lrt_mi_candidates(fit$partable, candidates)
  if (!nrow(fam)) {
    stop("modification_indices_lrt(): found no releasable cross-group ",
         "equalities of kind '", candidates, "'.", call. = FALSE)
  }

  is_fiml <- estimator == "FIML"
  data_list <- if (is_fiml) NULL else .lrt_mi_group_data(fit, data)

  released <- vector("list", nrow(fam))
  rows <- lapply(seq_len(nrow(fam)), function(i) {
    tok <- fam$token[i]
    rel <- magmaan(syntax, data, estimator = estimator, groups = group_var,
                   group_equal = group_equal,
                   group_partial = unique(c(base_partial, tok)))
    nt <- if (is_fiml) {
      robust_nested_lrt(rel, fit, method = method, gamma = gamma)
    } else {
      robust_nested_lrt(rel, fit, data = data_list, method = method,
                        gamma = gamma)
    }
    est <- .lrt_mi_released_est(rel$partable, fam[i, ])
    released[[i]] <<- est
    data.frame(
      lhs              = fam$lhs[i],
      op               = fam$op[i],
      rhs              = fam$rhs[i],
      df               = nt$df_diff %||% nt$df %||% NA_integer_,
      lrt              = nt$T_diff %||% NA_real_,
      p_unscaled       = nt$p_unscaled %||% NA_real_,
      p_scaled         = nt$p_scaled %||% NA_real_,
      p_adjusted       = nt$p_adjusted %||% NA_real_,
      p_scaled_shifted = nt$p_scaled_shifted %||% NA_real_,
      p_mixture        = nt$p_mixture %||% NA_real_,
      epc_range        = if (length(est$est) > 1L) diff(range(est$est)) else NA_real_,
      stringsAsFactors = FALSE)
  })

  out <- do.call(rbind, rows)
  ord <- order(-out$lrt)
  out <- out[ord, , drop = FALSE]
  rownames(out) <- NULL
  attr(out, "released_estimates") <- released[ord]
  attr(out, "method") <- method
  attr(out, "gamma") <- gamma
  attr(out, "conditioned_on") <- base_partial
  class(out) <- c("magmaan_mi_lrt", "data.frame")
  out
}

# ---- helpers (not exported) -------------------------------------------------

# Enumerate tied parameter families from a fitted partable. A release candidate
# is a label shared by free rows of the requested op across >= 2 groups; one row
# per family, with a group_partial release token.
.lrt_mi_candidates <- function(pt, which) {
  ops <- switch(which,
    loadings   = "=~",
    intercepts = "~1",
    all        = c("=~", "~1"))
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

# Per-group observed-variable matrices in fit's group order, for the
# complete-data nested test.
.lrt_mi_group_data <- function(fit, data) {
  ov <- if (is.list(fit$ov_names)) fit$ov_names[[1]] else fit$ov_names
  if (is.list(data) && !is.data.frame(data)) {
    return(lapply(data, function(d) as.matrix(d[, ov, drop = FALSE])))
  }
  gv <- fit$group_var
  if (is.null(gv) || !gv %in% names(data)) {
    stop("modification_indices_lrt(): `data` must contain the grouping column '",
         gv, "', or be a list of per-group matrices.", call. = FALSE)
  }
  lapply(fit$group_labels, function(g)
    as.matrix(data[as.character(data[[gv]]) == g, ov, drop = FALSE]))
}

# The exact released estimates (the LRT EPC): per-group values of the freed
# parameter in the released fit.
.lrt_mi_released_est <- function(pt_rel, fam_row) {
  sel <- pt_rel$op == fam_row$op & pt_rel$lhs == fam_row$lhs &
    (fam_row$op == "~1" | pt_rel$rhs == fam_row$rhs) & (pt_rel$free %||% 0L) > 0L
  sub <- pt_rel[sel, , drop = FALSE]
  list(group = sub$group, est = sub$est)
}

print.magmaan_mi_lrt <- function(x, ...) {
  method <- attr(x, "method"); gamma <- attr(x, "gamma")
  if (!is.null(method)) {
    cat("LRT-based modification indices (", method, ", gamma=", gamma, ")\n",
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
