# Accessors and presentation. Everything here reads the magmaanlab fit or the
# retained inference results; nothing recomputes the model.

.constraint_ops <- c("==", "<", ">")

.user_label <- function(label) {
  label <- as.character(label)
  label[is.na(label) | startsWith(label, ".")] <- ""
  label
}

.param_names <- function(pt, ngroups) {
  base <- paste0(pt$lhs, pt$op, pt$rhs)
  if (ngroups > 1L) {
    base <- ifelse(pt$group > 1L, paste0(base, ".g", pt$group), base)
  }
  label <- .user_label(pt$label)
  ifelse(nzchar(label), label, base)
}

.free_rows <- function(lab) {
  pt <- lab$partable
  pt <- pt[pt$free > 0L, , drop = FALSE]
  pt <- pt[!duplicated(pt$free), , drop = FALSE]
  pt[order(pt$free), , drop = FALSE]
}

#' Inspect a magmaan fit
#'
#' `coef(fit)` returns free estimates in the same order as the rows and columns
#' of `vcov(fit)`. `coef(summary(fit))` returns the full parameter table.
#' `vcov()` and `confint()` raise a `magmaan_inference_unavailable` condition
#' when covariance inference is unavailable; its `component` and `reason`
#' fields identify the missing result. `nobs()` counts observations used,
#' summed over groups.
#'
#' @param object,x A magmaan fit or its summary, as appropriate.
#' @param parm Parameter names or positive integer indices; omitted for all
#'   free parameters. Indices distinguish parameters with repeated labels.
#' @param level One confidence level strictly between zero and one.
#' @param test Interval test, currently only `"wald"`.
#' @param lavaan_compat Lavaan compatibility bundle: `NULL` (default policy), `"ML"`,
#'   `"MLM"`, `"MLR"`, `"DWLS"`, `"WLSMV"`, `"ULS"`, `"ULSMV"` or
#'   `"WLS"`, compatible with the fitted estimator. Lavaan bundles compute
#'   on demand without refitting. For parity, compare lavaan fits with the same
#'   model and estimation settings, including `meanstructure = TRUE` and
#'   `fixed.x = FALSE`. FIML supports `"ML"` and `"MLR"` with lavaan's
#'   `missing = "ml"` recipes. Unchecked components remain unavailable.
#' @param ... Unused.
#' @return `coef()` returns a named numeric vector for fits and a data frame
#'   for summaries. `vcov()` returns the parameter covariance matrix;
#'   `confint()` a matrix with two columns; `nobs()` the total count.
#'   Printing returns its input invisibly.
#' @name magmaan_methods
#' @export
coef.magmaan <- function(object, ...) {
  pt <- .free_rows(object$lab)
  stats::setNames(pt$est, .param_names(pt, length(object$lab$nobs)))
}

#' @rdname magmaan_methods
#' @export
vcov.magmaan <- function(object, lavaan_compat = NULL, ...) {
  object <- .with_lavaan_compat(object, lavaan_compat, "vcov()")
  .inference_result(object, "covariance", "vcov()")
}

#' @rdname magmaan_methods
#' @export
confint.magmaan <- function(object, parm, level = 0.95, test = "wald", lavaan_compat = NULL, ...) {
  .check_test(test, "confint()")
  .check_level(level, "confint()")
  object <- .with_lavaan_compat(object, lavaan_compat, "confint()")
  V <- .inference_result(object, "covariance", "confint()")
  est <- coef(object)
  index <- seq_along(est)
  if (!missing(parm)) {
    if (is.character(parm)) {
      index <- match(parm, names(est))
      if (anyNA(index)) stop("confint(): unknown parameter name", call. = FALSE)
    } else if (is.numeric(parm) && !is.complex(parm) && !anyNA(parm) &&
               all(is.finite(parm) & parm == floor(parm) & parm >= 1 & parm <= length(est))) {
      index <- as.integer(parm)
    } else {
      stop("confint(): `parm` must contain parameter names or positive integer indices within coef(fit)", call. = FALSE)
    }
  }
  est <- est[index]
  half <- stats::qnorm(1 - (1 - level) / 2) * sqrt(diag(V)[index])
  out <- cbind(est - half, est + half)
  pct <- paste0(format(100 * c((1 - level) / 2, 1 - (1 - level) / 2),
                       trim = TRUE, scientific = FALSE, digits = 3), " %")
  dimnames(out) <- list(names(est), pct)
  if (!is.null(object$inference$lavaan_compat)) attr(out, "lavaan_compat") <- object$inference$lavaan_compat
  out
}

#' Model-implied moments of a magmaan fit
#'
#' The covariance matrix and, with a mean structure, the mean vector that the
#' fitted model implies, as lavaan's `fitted()` gives them: one list for a
#' single group, a list per group otherwise. A model whose parameters are all
#' fixed is evaluated at those values, so this also gives the moments of a
#' population written in model syntax.
#'
#' @param object A [magmaan()] fit.
#' @param ... Unused.
#' @return `list(cov, mean)`, or a named list of them per group.
#' @export
fitted.magmaan <- function(object, ...) {
  lab <- object$lab
  implied <- magmaanlab::magmaan_core$model_implied(lab)
  ov <- lab$ov_names
  one <- function(g) {
    cov <- implied$sigma[[g]]
    dimnames(cov) <- list(ov, ov)
    out <- list(cov = cov)
    mean <- implied$mu[[g]]
    if (length(mean)) out$mean <- stats::setNames(as.numeric(mean), ov)
    out
  }
  groups <- seq_along(implied$sigma)
  if (length(groups) == 1L) return(one(1L))
  stats::setNames(lapply(groups, one), lab$group_labels)
}

#' @rdname magmaan_methods
#' @export
nobs.magmaan <- function(object, ...) {
  sum(object$lab$nobs)
}

# The parameter table: one row per model parameter, fixed and free, with the
# estimate and, when inference is available, its standard error, z-statistic,
# p-value and Wald interval. summary() stores it; coef(summary(fit)) returns it.
.parameter_table <- function(fit, level = 0.95) {
  pt <- fit$lab$partable
  pt <- pt[!pt$op %in% .constraint_ops, , drop = FALSE]
  ngroups <- length(fit$lab$nobs)
  out <- data.frame(lhs = pt$lhs, op = pt$op, rhs = pt$rhs, stringsAsFactors = FALSE)
  if (ngroups > 1L) out$group <- pt$group
  out$label <- .user_label(pt$label)
  out$free <- pt$free > 0L
  out$est <- pt$est
  out$se <- NA_real_
  V <- tryCatch(.inference_result(fit, "covariance", "summary()"),
                magmaan_inference_unavailable = function(e) NULL)
  if (!is.null(V)) {
    se_free <- sqrt(pmax(diag(V), 0))
    out$se[out$free] <- se_free[pt$free[out$free]]
  }
  defined <- fit$inference$defined
  is_def <- out$op == ":="
  if (!is.null(defined) && any(is_def)) {
    idx <- match(out$lhs[is_def], defined$lhs)
    out$est[is_def] <- defined$est[idx]
    out$se[is_def] <- defined$se[idx]
  }
  z <- stats::qnorm(1 - (1 - level) / 2)
  out$z <- out$est / out$se
  out$pvalue <- 2 * stats::pnorm(-abs(out$z))
  out$ci.lower <- out$est - z * out$se
  out$ci.upper <- out$est + z * out$se
  rownames(out) <- NULL
  out
}

.estimator_label <- function(fit) {
  paste0(fit$estimator, .covariance_label(fit$covariance))
}

.covariance_label <- function(covariance) {
  switch(covariance$policy %||% "unrestricted",
         psd = " (PSD-constrained)",
         barrier = if (covariance$lambda == 0) ", barrier(0): the unrestricted fit" else
           sprintf(", barrier(lambda = %s), experimental", format(covariance$lambda)),
         "")
}

.converged_label <- function(lab) {
  rule <- lab$fitting$effective$convergence
  if (!is.null(rule) && !identical(rule, "newton")) {
    return(sprintf("%s, by the %s rule (magmaan's check: %s)",
                   if (isTRUE(lab$converged)) "yes" else "no", rule,
                   lab$diagnostics$verdict$status %||% "unchecked"))
  }
  if (isTRUE(lab$converged)) return("yes")
  status <- lab$verdict$status %||% lab$optimizer_status
  state <- if (identical(lab$converged, FALSE)) "no" else "unchecked"
  paste0(state, if (!is.null(status)) paste0(" (", status, ")") else "")
}

.rows_label <- function(fit) {
  r <- fit$rows
  used <- sum(r$used)
  total <- sum(r$rows)
  deleted <- sum(r$deleted)
  if (is.na(used)) return(sprintf("%d rows supplied", total))
  how <- if (fit$estimator %in% c("FIML", "ML2S")) "not used (no observed values)" else
    "deleted listwise"
  sprintf("%d used of %d rows; %d %s", used, total, deleted, how)
}

#' @rdname magmaan_methods
#' @export
print.magmaan <- function(x, ...) {
  cat("magmaan fit\n")
  cat("  estimator:       ", .estimator_label(x), "\n", sep = "")
  cat("  converged:       ", .converged_label(x$lab), "\n", sep = "")
  cat("  observations:    ", .rows_label(x), "\n", sep = "")
  cat("  free parameters: ", length(coef(x)), "\n", sep = "")
  cat("  inference:       ", .inference_label(x), "\n", sep = "")
  t <- x$inference$global_score
  if (identical(t$reference, "all"))
    cat("  global p-value:  ", format(t$p_all),
        " (exact spectrum (All); decisions/05-dwls-policy-calibration)\n", sep = "")
  invisible(x)
}

#' Summary of a magmaan fit
#'
#' The fit, its parameter table and its global tests. `coef()` on the summary
#' returns the parameter table: one row per model parameter, fixed and free,
#' including defined (`:=`) parameters, with the estimate and, when inference
#' is available, its selected standard error, z-statistic, p-value and Wald
#' interval. `coef(fit)` stays the vector of free estimates that matches
#' `vcov(fit)`.
#'
#' @param object A [magmaan()] fit.
#' @param level Confidence level of the intervals.
#' @param lavaan_compat Inference bundle, as in [vcov.magmaan()].
#' @param ... Unused.
#' @return An object of class `summary.magmaan`.
#' @export
summary.magmaan <- function(object, level = 0.95, lavaan_compat = NULL, ...) {
  .check_level(level, "summary()")
  view <- .with_lavaan_compat(object, lavaan_compat, "summary()")
  structure(list(fit = object, inference = view$inference,
                 coefficients = .parameter_table(view, level = level),
                 tests = .global_tests(view), level = level),
            class = "summary.magmaan")
}

#' @rdname magmaan_methods
#' @export
coef.summary.magmaan <- function(object, ...) {
  object$coefficients
}

#' @rdname magmaan_methods
#' @param digits Number of digits printed after the decimal point.
#' @export
print.summary.magmaan <- function(x, digits = 3, ...) {
  fit <- x$fit
  fit$inference <- x$inference %||% fit$inference
  print(fit)
  if (nrow(fit$rows) > 1L) {
    cat("\nObservations by group\n")
    print(fit$rows, row.names = FALSE)
  }
  cat("\nParameters\n")
  p <- x$coefficients
  num <- vapply(p, is.numeric, logical(1))
  p[num] <- lapply(p[num], function(v) round(v, digits))
  print(p, row.names = FALSE)
  if (!is.null(x$tests)) {
    cat("\nGlobal tests against the saturated model\n")
    t <- x$tests
    num <- vapply(t, is.numeric, logical(1))
    t[num] <- lapply(t[num], function(v) round(v, digits))
    print(t, row.names = FALSE)
    if (identical(fit$inference$global_score$reference, "all"))
      cat("Reference: exact spectrum (All); decisions/05-dwls-policy-calibration.\n")
    .peba_note(x$tests)
    .lr_note(t)
  }
  inf <- fit$inference
  if (isTRUE(inf$psd_boundary)) cat("\n", .boundary_note, "\n", sep = "")
  if (isTRUE(inf$convergence$disagree)) cat("\n", .verdict_note(inf$convergence), "\n", sep = "")
  if (!is.null(inf)) {
    # Components that do not exist for the estimator are notes, not gaps.
    off <- inf$status[!inf$status$available, , drop = FALSE]
    inapplicable <- off[off$reason == "inapplicable", , drop = FALSE]
    missing <- off[off$reason != "inapplicable", , drop = FALSE]
    if (nrow(missing)) {
      cat("\nUnavailable inference\n")
      for (i in seq_len(nrow(missing))) {
        cat("  ", missing$component[i], ": ", missing$reason[i],
            if (nzchar(missing$detail[i])) paste0(" (", missing$detail[i], ")"), "\n", sep = "")
      }
    }
    if (nrow(inapplicable)) {
      cat("\n")
      for (i in seq_len(nrow(inapplicable))) cat("Note: ", inapplicable$detail[i], ".\n", sep = "")
    }
  }
  invisible(x)
}

# Printed under policy output that shows a likelihood-ratio test.
.lr_note <- function(t) {
  if (any(t$test == "likelihood ratio" & is.finite(t$statistic)))
    cat("The score test is primary; the likelihood-ratio test tends to over-reject",
        "when N is small relative to its df.\n")
  invisible(NULL)
}

.boundary_note <- paste(
  "The PSD estimate lies on the boundary of the covariance space. The inference",
  "assumes the population is interior (every covariance matrix positive definite).")

# A compatibility acceptance rule and magmaan's own convergence check disagree.
.verdict_note <- function(convergence) {
  rule <- convergence$rule
  if (isTRUE(convergence$converged)) {
    paste0("Note: the ", rule, " rule accepted this fit, but magmaan's convergence\n",
           "check rejects it (as_lab_fit(fit)$diagnostics$verdict). Estimates and\n",
           "inference follow the ", rule, " rule. Refit with magmaan's default\n",
           "fitting options to use its own search and check.")
  } else {
    paste0("Note: the ", rule, " rule rejected this fit, but magmaan's convergence\n",
           "check accepts it. Inference is unavailable, following the ", rule, " rule.")
  }
}

.nested_verdict_note <- paste(
  "Note: for at least one fit, a compatibility convergence rule and magmaan's",
  "convergence check disagree; the tests follow the selected rule.")

#' Compare two nested magmaan fits
#'
#' Score and likelihood-ratio tests of the restricted fit against the other,
#' each calibrated with SB and PEBA4, as the global tests are. The score test
#' is primary and comes first: it calibrates better, especially when N is
#' small relative to the df, where the likelihood-ratio test tends to
#' over-reject. Select rows by `test` rather than position. The restricted
#' model may constrain, fix or drop paths from the other model (for example,
#' a shared label, `b == 0`, or a loading fixed to zero), fitted to the same
#' observations with the same estimator and covariance policy. For a single
#' restriction, a Wald test
#' is the z-statistic of a defined parameter such as `d := a - b` in
#' `coef(summary(fit))`.
#'
#' @param object,... Two [magmaan()] fits, in either order.
#' @param lavaan_compat `NULL` (default) reports the policy's score and LR
#'   tests. `"ML"`, `"MLM"` and `"MLR"` report lavaan's default difference
#'   test for complete-data ML. `"WLSMV"` and `"ULSMV"` report the ordinal
#'   Satorra-2000 scaled-shifted difference; plain `"DWLS"`/`"ULS"` retain
#'   the statistic without a p-value, and `"WLS"` reports the standard difference.
#'   FIML supports `"ML"` (standard) and `"MLR"` (SB2001 using Yuan-Bentler
#'   Mplus scales); `"MLM"` is inapplicable.
#' @return A data frame with one row per test, of class `magmaan_anova`.
#' @export
anova.magmaan <- function(object, ..., lavaan_compat = NULL) {
  fits <- c(list(object), list(...))
  labels <- vapply(as.list(substitute(list(object, ...)))[-1L],
                   function(e) paste(deparse(e), collapse = ""), character(1))
  if (length(fits) != 2L || !all(vapply(fits, inherits, logical(1), "magmaan"))) {
    stop("anova(): compare exactly two magmaan() fits", call. = FALSE)
  }
  a <- fits[[1L]]$lab
  b <- fits[[2L]]$lab
  if (!identical(fits[[1L]]$estimator, fits[[2L]]$estimator) ||
      !identical(unclass(fits[[1L]]$covariance), unclass(fits[[2L]]$covariance))) {
    stop("anova(): the fits must use the same estimator and covariance policy", call. = FALSE)
  }
  if (!identical(a$raw_data, b$raw_data)) {
    stop("anova(): the fits must use the same observations in the same order", call. = FALSE)
  }
  lavaan_compat <- .check_lavaan_compat(fits[[1L]], lavaan_compat, "anova()")
  .check_lavaan_compat(fits[[2L]], lavaan_compat, "anova()")
  if (!is.null(lavaan_compat)) {
    null <- 2L
    res <- magmaanlab::convention_nested(a, b, lavaan_compat)
    if (identical(res$test$reason, "not_nested")) {
      res <- magmaanlab::convention_nested(b, a, lavaan_compat)
      null <- 1L
      if (identical(res$test$reason, "not_nested")) stop("anova(): the models are not nested", call. = FALSE)
    }
    t <- res$test
    reasons <- if (isTRUE(t$available)) character() else
      c(lr = paste0(t$reason, ": ", t$detail))
    return(structure(.lavaan_compat_test_row(t), class = c("magmaan_anova", "data.frame"),
      lavaan_compat = lavaan_compat, restricted = labels[[null]], alternative = labels[[3L - null]],
      unavailable = reasons, psd_boundary = isTRUE(res$psd_boundary),
      verdict_disagreement = isTRUE(res$verdict_disagreement)))
  }
  null <- 2L
  res <- magmaanlab::policy_nested(a, b)
  if (identical(res$lr$reason, "not_nested")) {
    swapped <- magmaanlab::policy_nested(b, a)
    if (identical(swapped$lr$reason, "not_nested")) {
      stop("anova(): the models are not nested: the restricted model must be the ",
           "other model with paths dropped, fixed or constrained (",
           res$lr$detail, ")", call. = FALSE)
    }
    res <- swapped
    null <- 1L
  }
  # The score test leads: it calibrates better than the likelihood ratio,
  # especially at small N and high df (decided 2026-10-02, todo.md).
  rows <- lapply(c("score", "lr"), function(component) {
    t <- res[[component]]
    label <- if (component == "score") "score" else
      if (identical(t$label, "fit_function_difference")) "fit-function difference" else "likelihood ratio"
    data.frame(test = label,
               statistic = t$statistic, df = t$df, p.sb = t$p_sb,
               p.peba4 = t$p_peba4, sb.scale = t$sb_scale,
               stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows)
  reasons <- vapply(res[c("score", "lr")], function(t)
    if (isTRUE(t$available)) "" else paste0(t$reason, if (nzchar(t$detail)) paste0(": ", t$detail)),
    character(1))
  structure(out, class = c("magmaan_anova", "data.frame"),
            restricted = labels[[null]], alternative = labels[[3L - null]],
            peba_blocks = vapply(res[c("score", "lr")],
              function(t) as.integer(t$peba_blocks %||% 0L), integer(1)),
            unavailable = reasons[nzchar(reasons)],
            psd_boundary = isTRUE(res$psd_boundary),
            verdict_disagreement = isTRUE(res$verdict_disagreement))
}

#' @rdname anova.magmaan
#' @param x A nested-test result.
#' @param digits Number of digits printed after the decimal point.
#' @export
print.magmaan_anova <- function(x, digits = 3, ...) {
  cat("Nested tests of ", attr(x, "restricted"), " (restricted) against ",
      attr(x, "alternative"), "\n", sep = "")
  if (!is.null(attr(x, "lavaan_compat"))) cat("lavaan compatibility: ", attr(x, "lavaan_compat"), "\n", sep = "")
  t <- as.data.frame(unclass(x), stringsAsFactors = FALSE)
  num <- vapply(t, is.numeric, logical(1))
  t[num] <- lapply(t[num], function(v) round(v, digits))
  print(t, row.names = FALSE)
  .peba_note(x)
  if (is.null(attr(x, "lavaan_compat"))) .lr_note(t)
  u <- attr(x, "unavailable")
  for (i in seq_along(u)) cat("  ", names(u)[i], " unavailable: ", u[[i]], "\n", sep = "")
  if (isTRUE(attr(x, "psd_boundary"))) cat(.boundary_note, "\n")
  if (isTRUE(attr(x, "verdict_disagreement"))) cat(.nested_verdict_note, "\n")
  invisible(x)
}

# The test an interval inverts. Only Wald exists; `confint()` keeps room for
# likelihood-ratio inversion without changing calls.
.check_test <- function(test, caller) {
  .check_choice(test, "test", "wald", planned = "lr", caller = caller)
}

.check_level <- function(level, caller) {
  if (!is.numeric(level) || is.complex(level) || length(level) != 1L || is.na(level) ||
      !is.finite(level) || level <= 0 || level >= 1) {
    stop(paste0(caller, ": `level` must be one number strictly between zero and one"),
         call. = FALSE)
  }
  invisible(level)
}
