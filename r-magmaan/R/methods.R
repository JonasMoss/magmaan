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
confint.magmaan <- function(object, parm, level = 0.95, ...) {
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

#' @export
nobs.magmaan <- function(object, ...) {
  sum(object$lab$nobs)
}

#' Parameter table of a magmaan fit
#'
#' One row per model parameter, fixed and free, with the estimate and, when
#' inference is available, its standard error, z-statistic, p-value and Wald
#' interval.
#'
#' @param fit A [magmaan()] fit.
#' @param level Confidence level of the intervals.
#' @return A data frame.
#' @export
parameters <- function(fit, level = 0.95) {
  if (!inherits(fit, "magmaan")) stop("parameters(): supply a magmaan() fit", call. = FALSE)
  pt <- fit$lab$partable
  pt <- pt[!pt$op %in% .constraint_ops, , drop = FALSE]
  ngroups <- length(fit$lab$nobs)
  out <- data.frame(lhs = pt$lhs, op = pt$op, rhs = pt$rhs, stringsAsFactors = FALSE)
  if (ngroups > 1L) out$group <- pt$group
  out$label <- .user_label(pt$label)
  out$free <- pt$free > 0L
  out$est <- pt$est
  out$se <- NA_real_
  V <- tryCatch(.inference_result(fit, "covariance", "parameters()"),
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

#' @export
summary.magmaan <- function(object, level = 0.95, ...) {
  structure(list(fit = object, parameters = parameters(object, level = level),
                 tests = .global_tests(object), level = level),
            class = "summary.magmaan")
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
  p <- x$parameters
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
