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
#'   `missing = "ml"` recipes. Complete mixed DWLS supports `"WLSMV"`,
#'   with NACOV covariance and scaled-shifted global/Satorra-2000 nested tests.
#'   Mixed delta/theta reporting is validated at identical parameter points;
#'   grouped/theta retained endpoints have limited validation. Other mixed
#'   bundles and missing mixed observations remain unavailable.
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

.standardized_report <- function(fit) {
  reason <- NULL
  V <- tryCatch(.inference_result(fit, "covariance", "summary()"),
    magmaan_inference_unavailable = function(e) {
      reason <<- e$reason
      NULL
    })
  available <- !is.null(V)
  if (!available) V <- matrix(0, length(fit$lab$theta), length(fit$lab$theta))
  std <- magmaanlab::standardized_rows(fit$lab, V)
  pt <- fit$lab$partable
  keep <- !pt$op %in% .constraint_ops
  columns <- as.data.frame(lapply(std, function(v) as.numeric(v)[keep]))
  if (!available) {
    columns$std.lv.se <- NA_real_
    columns$std.all.se <- NA_real_
    attr(columns, "inference_reason") <- reason
  }
  if (any(pt$op[keep] == ":="))
    attr(columns, "defined_reason") <- "unsupported_defined_scale"
  # Every measurement outcome and regression outcome is endogenous. The
  # diagonal std.all residual row is the unexplained variance fraction.
  endogenous <- unique(paste(pt$group[pt$op == "=~"], pt$rhs[pt$op == "=~"]))
  endogenous <- union(endogenous, paste(pt$group[pt$op == "~"], pt$lhs[pt$op == "~"]))
  rows <- which(pt$op == "~~" & pt$lhs == pt$rhs &
                  paste(pt$group, pt$lhs) %in% endogenous)
  r2 <- data.frame(variable = pt$lhs[rows], r2 = 1 - as.numeric(std$std.all)[rows],
                   r2.se = as.numeric(std$std.all.se)[rows], stringsAsFactors = FALSE)
  if (length(fit$lab$nobs) > 1L)
    r2 <- cbind(group = pt$group[rows], r2)
  if (!available) {
    r2$r2.se <- NA_real_
    attr(r2, "inference_reason") <- reason
  }
  list(coefficients = columns, r2 = r2)
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
  # A structurally unidentified model fails under every rule (TASK-33.3).
  if (identical(lab$diagnostics$identification$status, "unidentified")) {
    return(paste0("no (structurally unidentified; see ",
                  "as_lab_fit(fit)$diagnostics$identification)"))
  }
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
  if (isTRUE(t$available) && identical(t$reference, "peba4")) {
    cat("  global p-value:  ", format(t$p_peba4), " (score/PEBA4)\n", sep = "")
  }
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
#' @param standardized Add std.lv, std.lv.se, std.all and std.all.se columns
#'   and an R-squared table. Delta-method SEs use the selected covariance.
#'   Without covariance, estimates remain available and SEs are NA with the
#'   inference reason attached to the tables. Defined parameters have no
#'   declared standardized scale and are NA (unsupported_defined_scale).
#' @param fit_measures Logical; compute and attach [fit_measures()] on request.
#'   With `lavaan_compat`, returns the selected bundle's standard, scaled
#'   and robust measures, intervals and close-fit p-values, including FIML
#'   ML/MLR where supported.
#' @param references Character vector of reference laws, or `NULL` for the policy defaults.
#' @param ... Unused.
#' @return An object of class `summary.magmaan`. With `standardized = TRUE`,
#'   `r2` is a data frame with variable, r2 and r2.se columns (and group for
#'   grouped models), covering endogenous observed and latent variables.
#' @section Simulation studies:
#' `references` accepts case-insensitive `std`, `sb`, `ss`, `mv`, `scaled_f`,
#' `all`, `pall`, `peba<k>` and `eba<k>` (integer `k >= 1`, up to the C++
#' integer limit). Output names are lower case. Each available test has one
#' row per requested reference, using its policy statistic and stored spectrum;
#' covariance and fitting choices stay fixed. Defaults are PEBA4 for ML/FIML
#' and All for DWLS global and nested tests. Use `references = c("sb", "peba4")`
#' to include the SB comparator. The base columns are `test`, `statistic`, `df`,
#' `reference`, `pvalue`, `recommended`, `reason`. `test` uses stable codes
#' `score`, `lr`, `fit_function`, `fit_function_difference`. Unavailable tests retain one
#' row with missing reference and p-value and a typed reason. Select rows by
#' `test` and `reference`; migrate `p.sb`/`p.peba4` to `pvalue` on the corresponding
#' rows. `recommended` is TRUE only for the policy's recommended p-values: score
#' with PEBA4 for ML/FIML, or fit_function with All for DWLS global tests.
#' The likelihood-ratio test is reported because it is the standard statistic,
#' not recommended; `references = "std"` gives its plain chi-square p-value.
#' All rows are printed, including LR rows with the small-sample caveat.
#' EBA/pEBA partition the retained spectrum (including zero eigenvalues) into
#' blocks of size `ceiling(df / k)`; when `k >= df`, blocks are singletons.
#' EBA then equals All, and pEBA equals pAll. References cannot be combined
#' with `lavaan_compat`; compatibility rows append `unscaled.statistic`, `scale`
#' and `shift` and are never marked recommended.
#' @export
summary.magmaan <- function(object, level = 0.95, lavaan_compat = NULL, references = NULL,
                            standardized = FALSE, fit_measures = FALSE, ...) {
  if (!is.logical(standardized) || length(standardized) != 1L || is.na(standardized))
    stop("summary(): standardized must be TRUE or FALSE", call. = FALSE)
  if (!is.logical(fit_measures) || length(fit_measures) != 1L || is.na(fit_measures))
    stop("summary(): fit_measures must be TRUE or FALSE", call. = FALSE)
  measures <- if (fit_measures) magmaan::fit_measures(object, lavaan_compat) else NULL
  .check_level(level, "summary()")
  references <- .check_references(references, lavaan_compat, "summary()")
  view <- .with_lavaan_compat(object, lavaan_compat, "summary()")
  coefficients <- .parameter_table(view, level = level)
  r2 <- NULL
  if (standardized) {
    report <- .standardized_report(view)
    coefficients <- cbind(coefficients, report$coefficients)
    attr(coefficients, "inference_reason") <- attr(report$coefficients, "inference_reason")
    attr(coefficients, "defined_reason") <- attr(report$coefficients, "defined_reason")
    r2 <- report$r2
  }
  structure(list(fit = object, inference = view$inference, r2 = r2,
                 coefficients = coefficients,
                 tests = .global_tests(view, references), fit_measures = measures,
                 level = level, references = references),
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
  if (!is.null(x$r2)) {
    cat("\nR-squared\n")
    r <- x$r2
    r[c("r2", "r2.se")] <- lapply(r[c("r2", "r2.se")], round, digits = digits)
    print(r, row.names = FALSE)
  }
  if (!is.null(x$tests)) {
    cat("\nGlobal tests against the saturated model\n")
    t <- x$tests
    num <- vapply(t, is.numeric, logical(1))
    t[num] <- lapply(t[num], function(v) round(v, digits))
    .print_test_table(t)
    if (identical(fit$inference$global_score$reference, "all"))
      cat("Reference: exact spectrum (All); decisions/05-dwls-policy-calibration.\n")
    .peba_note(x$tests)
    .lr_note(t)
  }
  if (!is.null(x$fit_measures)) {
    cat("\nPolicy fit measures\n")
    measures <- x$fit_measures
    measures$estimate <- round(measures$estimate, digits)
    print(measures, row.names = FALSE)
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

.print_test_table <- function(t) {
  labels <- c(score = "score", lr = "likelihood ratio", fit_function = "fit function",
              fit_function_difference = "fit-function difference")
  code <- t$test %in% names(labels)
  t$test[code] <- labels[t$test[code]]
  print(t, row.names = FALSE)
}

# Printed under policy output that shows a likelihood-ratio test.
.lr_note <- function(t) {
  if (any(t$test == "lr" & is.finite(t$statistic)))
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
#' each calibrated with PEBA4, as the global tests are. The score test
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
#' If the larger model fits worse than the restricted model, it is refitted
#' from the embedded restricted estimate with its own fitting options. A better
#' endpoint is used only when its native convergence verdict passes. The input
#' fits are unchanged; `attr(result, "refit")` records the old and new objective
#' and printing reports the recovery. Internal refit warnings are captured in
#' `attr(result, "reseed")$warnings` and printed as notes, including when a
#' retry is unsuccessful. An unsuccessful retry retains the failure.
#'
#' @param object,... Two [magmaan()] fits, in either order.
#' @param lavaan_compat `NULL` (default) reports the policy's score and LR
#'   tests. `"ML"`, `"MLM"` and `"MLR"` report lavaan's default difference
#'   test for complete-data ML. `"WLSMV"` and `"ULSMV"` report the ordinal
#'   Satorra-2000 scaled-shifted difference; plain `"DWLS"`/`"ULS"` retain
#'   the statistic without a p-value, and `"WLS"` reports the standard difference.
#'   FIML supports `"ML"` (standard) and `"MLR"` (SB2001 using Yuan-Bentler
#'   Mplus scales); `"MLM"` is inapplicable.
#' @param references Character vector of reference laws, or `NULL` for the policy defaults.
#' @section Simulation studies:
#' See [summary.magmaan()] for the reference grammar and uniform base columns.
#' Each available nested test has one row per reference; an unavailable test
#' retains one row with its typed `reason` and missing `reference`/`pvalue`.
#' Defaults are PEBA4 for ML/FIML and All for DWLS. `recommended` is TRUE only for the policy's
#' recommended p-values: score with PEBA4 for ML/FIML, or
#' fit_function_difference with All for DWLS. The likelihood-ratio test
#' is reported because it is the standard statistic, not recommended;
#' `references = "std"` gives its plain chi-square p-value. Alternatives
#' use the same policy statistic and retained nested spectrum (`spectra` attribute),
#' including after recovery. Select by `test` and `reference` rather than position.
#' References cannot be combined with `lavaan_compat`.
#' @return A data frame of class `magmaan_anova`.
#' @export
anova.magmaan <- function(object, ..., lavaan_compat = NULL, references = NULL) {
  references <- .check_references(references, lavaan_compat, "anova()")
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
    recovery <- .nested_recovery(res, fits[[3L - null]]$lab, fits[[null]]$lab,
      function(h1, h0) magmaanlab::convention_nested(h1, h0, lavaan_compat), "test")
    res <- recovery$result
    t <- res$test
    reasons <- if (isTRUE(t$available)) character() else
      c(lr = paste0(t$reason, ": ", t$detail))
    return(structure(.lavaan_compat_test_row(t,
      if (fits[[1L]]$estimator %in% c("ML", "FIML")) "lr" else "fit_function_difference"), class = c("magmaan_anova", "data.frame"),
      lavaan_compat = lavaan_compat, restricted = labels[[null]], alternative = labels[[3L - null]],
      unavailable = reasons, psd_boundary = isTRUE(res$psd_boundary),
      verdict_disagreement = isTRUE(res$verdict_disagreement), refit = recovery$refit, reseed = recovery$reseed))
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
  recovery <- .nested_recovery(res, fits[[3L - null]]$lab, fits[[null]]$lab,
    magmaanlab::policy_nested, "lr")
  res <- recovery$result
  # The score test leads: it calibrates better than the likelihood ratio,
  # especially at small N and high df (decided 2026-10-02, todo.md).
  rows <- lapply(c("score", "lr"), function(component) {
    t <- res[[component]]
    label <- if (component == "score") "score" else
      if (identical(t$label, "fit_function_difference") || fits[[1L]]$estimator == "DWLS")
        "fit_function_difference" else "lr"
    .policy_test_rows(t, label, references)
  })
  out <- .bind_test_rows(rows)
  reasons <- vapply(res[c("score", "lr")], function(t)
    if (isTRUE(t$available)) "" else paste0(t$reason, if (nzchar(t$detail)) paste0(": ", t$detail)),
    character(1))
  structure(out, class = c("magmaan_anova", "data.frame"),
            restricted = labels[[null]], alternative = labels[[3L - null]],
            references = references, spectra = lapply(res[c("score", "lr")], function(t) t$eigenvalues),
            peba_blocks = attr(out, "peba_blocks"),
            unavailable = reasons[nzchar(reasons)],
            psd_boundary = isTRUE(res$psd_boundary),
            verdict_disagreement = isTRUE(res$verdict_disagreement), refit = recovery$refit, reseed = recovery$reseed)
}

.nested_recovery <- function(result, alternative, null, compare, component) {
  out <- list(result = result, refit = NULL, reseed = NULL)
  test <- result[[component]]
  if (!identical(test$reason, "not_converged") ||
      !grepl("the alternative fits worse than the null", test$detail, fixed = TRUE)) return(out)
  warnings <- character()
  if (identical(alternative$diagnostics$identification$status, "unidentified")) {
    .check_identification(list(identification_report = alternative$diagnostics$identification))
  }
  retry <- tryCatch(withCallingHandlers(
    magmaanlab::refit_from_null(alternative, null),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }), error = function(e) NULL)
  out$reseed <- list(warnings = warnings)
  if (is.null(retry) || !isTRUE(retry$converged) ||
      !identical(retry$diagnostics$verdict$status, "passed") ||
      !is.finite(retry$fmin) || retry$fmin > null$fmin ||
      retry$fmin >= alternative$fmin) return(out)
  updated <- tryCatch(compare(retry, null), error = function(e) NULL)
  if (is.null(updated) || !isTRUE(updated[[component]]$available)) return(out)
  out$result <- updated
  out$refit <- list(objective_before = alternative$fmin,
                   objective_after = retry$fmin,
                   verdict = retry$diagnostics$verdict)
  out
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
  .print_test_table(t)
  refit <- attr(x, "refit")
  if (!is.null(refit)) cat("larger model refit from the restricted estimate; its objective improved from ",
    format(refit$objective_before, digits = digits), " to ",
    format(refit$objective_after, digits = digits), "\n", sep = "")
  reseed <- attr(x, "reseed")
  if (length(reseed$warnings)) {
    cat("larger model refit warnings:\n")
    cat(paste0("  ", reseed$warnings, collapse = "\n"), "\n", sep = "")
  }
  if (any(x$test == "fit_function_difference" & x$reference %in% "all"))
    cat("Reference: exact spectrum (All); decisions/05-dwls-policy-calibration and decisions/06-ordinal-threshold-invariance.\n")
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
