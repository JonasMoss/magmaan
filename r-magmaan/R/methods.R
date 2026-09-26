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

#' @export
coef.magmaan <- function(object, ...) {
  pt <- .free_rows(object$lab)
  stats::setNames(pt$est, .param_names(pt, length(object$lab$nobs)))
}

#' @export
vcov.magmaan <- function(object, ...) {
  .inference_result(object, "covariance", "vcov()")
}

#' @export
confint.magmaan <- function(object, parm, level = 0.95, test = "wald", ...) {
  .check_test(test, "confint()")
  V <- .inference_result(object, "covariance", "confint()")
  est <- coef(object)
  if (!missing(parm)) est <- est[parm]
  half <- stats::qnorm(1 - (1 - level) / 2) * sqrt(diag(V)[names(est)])
  out <- cbind(est - half, est + half)
  pct <- paste0(format(100 * c((1 - level) / 2, 1 - (1 - level) / 2),
                       trim = TRUE, scientific = FALSE, digits = 3), " %")
  dimnames(out) <- list(names(est), pct)
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
  paste0(fit$estimator, if (isTRUE(fit$psd)) " (PSD-constrained)" else "")
}

.converged_label <- function(lab) {
  if (isTRUE(lab$converged)) return("yes")
  status <- lab$verdict$status %||% lab$optimizer_status
  paste0("no", if (!is.null(status)) paste0(" (", status, ")") else "")
}

.rows_label <- function(fit) {
  r <- fit$rows
  used <- sum(r$used)
  total <- sum(r$rows)
  deleted <- sum(r$deleted)
  if (is.na(used)) return(sprintf("%d rows supplied", total))
  how <- switch(fit$missing,
                fiml = "not used (no observed values)",
                pairwise = "fully deleted",
                "deleted listwise")
  sprintf("%d used of %d rows; %d %s", used, total, deleted, how)
}

#' @export
print.magmaan <- function(x, ...) {
  cat("magmaan fit\n")
  cat("  estimator:       ", .estimator_label(x), "\n", sep = "")
  cat("  converged:       ", .converged_label(x$lab), "\n", sep = "")
  cat("  observations:    ", .rows_label(x), "\n", sep = "")
  cat("  free parameters: ", length(coef(x)), "\n", sep = "")
  cat("  inference:       ", .inference_label(x), "\n", sep = "")
  invisible(x)
}

#' Summary of a magmaan fit
#'
#' The fit, its parameter table and its global tests. `coef()` on the summary
#' returns the parameter table: one row per model parameter, fixed and free,
#' including defined (`:=`) parameters, with the estimate and, when inference
#' is available, its robust standard error, z-statistic, p-value and Wald
#' interval. `coef(fit)` stays the vector of free estimates that matches
#' `vcov(fit)`.
#'
#' @param object A [magmaan()] fit.
#' @param level Confidence level of the intervals.
#' @param ... Unused.
#' @return An object of class `summary.magmaan`.
#' @export
summary.magmaan <- function(object, level = 0.95, ...) {
  structure(list(fit = object, coefficients = .parameter_table(object, level = level),
                 tests = .global_tests(object), level = level),
            class = "summary.magmaan")
}

#' @export
coef.summary.magmaan <- function(object, ...) {
  object$coefficients
}

#' @export
print.summary.magmaan <- function(x, digits = 3, ...) {
  fit <- x$fit
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
  }
  inf <- fit$inference
  if (isTRUE(inf$psd_boundary)) cat("\n", .boundary_note, "\n", sep = "")
  if (!is.null(inf) && !all(inf$status$available)) {
    cat("\nUnavailable inference\n")
    s <- inf$status[!inf$status$available, , drop = FALSE]
    for (i in seq_len(nrow(s))) {
      cat("  ", s$component[i], ": ", s$reason[i],
          if (nzchar(s$detail[i])) paste0(" (", s$detail[i], ")"), "\n", sep = "")
    }
  }
  invisible(x)
}

.boundary_note <- paste(
  "The PSD estimate lies on the boundary of the covariance space. The inference",
  "assumes the population is interior (every covariance matrix positive definite).")

#' Compare two nested magmaan fits
#'
#' Likelihood-ratio and score tests of the restricted fit against the other,
#' each calibrated with SB and PEBA4, as the global tests are. The restricted
#' model must be the other model plus equality constraints on its parameters,
#' such as a shared label or `b == 0`, fitted to the same observations with
#' the same estimator and `psd` setting. For a single restriction, a Wald test
#' is the z-statistic of a defined parameter such as `d := a - b` in
#' `coef(summary(fit))`.
#'
#' @param object,... Two [magmaan()] fits, in either order.
#' @return A data frame with one row per test, of class `magmaan_anova`.
#' @export
anova.magmaan <- function(object, ...) {
  fits <- c(list(object), list(...))
  labels <- vapply(as.list(substitute(list(object, ...)))[-1L],
                   function(e) paste(deparse(e), collapse = ""), character(1))
  if (length(fits) != 2L || !all(vapply(fits, inherits, logical(1), "magmaan"))) {
    stop("anova(): compare exactly two magmaan() fits", call. = FALSE)
  }
  a <- fits[[1L]]$lab
  b <- fits[[2L]]$lab
  if (!identical(fits[[1L]]$estimator, fits[[2L]]$estimator) ||
      !identical(fits[[1L]]$psd, fits[[2L]]$psd)) {
    stop("anova(): the fits must use the same estimator and psd setting", call. = FALSE)
  }
  if (!identical(a$raw_data, b$raw_data)) {
    stop("anova(): the fits must use the same observations in the same order", call. = FALSE)
  }
  null <- 2L
  res <- magmaanlab::policy_nested(a, b)
  if (identical(res$lr$reason, "not_nested")) {
    swapped <- magmaanlab::policy_nested(b, a)
    if (identical(swapped$lr$reason, "not_nested")) {
      stop("anova(): the models are not nested: the restricted model must be the ",
           "other model plus equality constraints on its parameters (",
           res$lr$detail, ")", call. = FALSE)
    }
    res <- swapped
    null <- 1L
  }
  rows <- lapply(c("lr", "score"), function(component) {
    t <- res[[component]]
    data.frame(test = if (component == "lr") "likelihood ratio" else "score",
               statistic = t$statistic, df = t$df, p.sb = t$p_sb,
               p.peba4 = t$p_peba4, sb.scale = t$sb_scale,
               stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows)
  reasons <- vapply(res[c("lr", "score")], function(t)
    if (isTRUE(t$available)) "" else paste0(t$reason, if (nzchar(t$detail)) paste0(": ", t$detail)),
    character(1))
  structure(out, class = c("magmaan_anova", "data.frame"),
            restricted = labels[[null]], alternative = labels[[3L - null]],
            unavailable = reasons[nzchar(reasons)],
            psd_boundary = isTRUE(res$psd_boundary))
}

#' @export
print.magmaan_anova <- function(x, digits = 3, ...) {
  cat("Nested tests of ", attr(x, "restricted"), " (restricted) against ",
      attr(x, "alternative"), "\n", sep = "")
  t <- as.data.frame(unclass(x), stringsAsFactors = FALSE)
  num <- vapply(t, is.numeric, logical(1))
  t[num] <- lapply(t[num], function(v) round(v, digits))
  print(t, row.names = FALSE)
  u <- attr(x, "unavailable")
  for (i in seq_along(u)) cat("  ", names(u)[i], " unavailable: ", u[[i]], "\n", sep = "")
  if (isTRUE(attr(x, "psd_boundary"))) cat(.boundary_note, "\n")
  invisible(x)
}

# The test an interval inverts. Only Wald exists; `confint()` keeps room for
# likelihood-ratio inversion without changing calls.
.check_test <- function(test, caller) {
  .check_choice(test, "test", "wald", planned = "lr", caller = caller)
}
