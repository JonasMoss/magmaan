#' Modification indices and equality releases
#'
#' The default reports misspecification-robust one-df score statistics with
#' chi-square(1) p-values. Equality releases test each native affine restriction
#' separately, including shared labels, explicit equalities and group.equal.
#' Candidate order is preserved; results are never sorted automatically.
#'
#' LR tests refit one augmented model per candidate from the embedded fitted
#' estimate, and are consequently more costly. They use the same policy nested
#' test as anova(): LR for ML/FIML and fit-function differences for DWLS, with
#' the All reference law at one df. EPC columns retain the one-step score EPC.
#' @param fit A magmaan fit.
#' @param test `"score"` (default) or `"lr"`.
#' @param candidates NULL for all candidates, a character vector of operators
#'   (`"=~"`, `"~~"`, `"~"`, `"=="`), or a data frame with lhs, op, rhs
#'   and optional group and kind columns, selecting rows from the full table.
#' @param releases Include equality releases (default TRUE).
#' @return A data frame with kind, lhs, op, rhs, group, test, statistic, df,
#'   pvalue, epc, sepc.lv, sepc.all and reason. Unavailable statistics are NA.
#' @section Candidate scope:
#' The current generator proposes loadings and residual/latent covariances.
#' Absent structural regressions require additional variable-table support;
#' requests for `"~"` carry an unsupported_model row.
#' @export
modindices <- function(fit, test = c("score", "lr"), candidates = NULL, releases = TRUE) {
  if (!inherits(fit, "magmaan")) stop("modindices(): supply a magmaan() fit", call. = FALSE)
  test <- match.arg(test)
  out <- magmaanlab::policy_modification_indices(fit$lab, releases = releases)
  rows <- attr(out, "candidate_row")
  if (!is.null(candidates)) {
    if (is.character(candidates)) {
      if (anyNA(candidates) || any(!candidates %in% c("=~", "~~", "~", "==")))
        stop("candidates must contain supported operator codes", call. = FALSE)
      keep <- out$op %in% candidates | (out$op == "" & out$reason != "available")
    } else if (is.data.frame(candidates)) {
      required <- c("lhs", "op", "rhs")
      if (!all(required %in% names(candidates))) stop("candidate rows need lhs, op and rhs", call. = FALSE)
      keys <- intersect(c("kind", required, "group"), names(candidates))
      keep <- vapply(seq_len(nrow(out)), function(i) any(vapply(seq_len(nrow(candidates)),
        function(j) all(vapply(keys, function(k) identical(as.character(out[[k]][i]),
          as.character(candidates[[k]][j])), logical(1))), logical(1))), logical(1))
    } else stop("candidates must be NULL, operator codes or a data frame", call. = FALSE)
    rows <- rows[keep]; out <- out[keep, , drop = FALSE]
  }
  if (test == "lr" && nrow(out)) {
    out$test <- if (fit$estimator == "DWLS") "fit_function_difference" else "lr"
    for (i in seq_len(nrow(out))) {
      if (!out$reason[i] %in% c("available", "numeric_failure") || out$group[i] == 0L && out$kind[i] == "fixed") next
      augmented <- tryCatch(suppressWarnings(getFromNamespace(".policy_mi_refit", "magmaanlab")(
          fit$lab, out[i, , drop = FALSE], rows[i])), error = function(e) e)
      out$statistic[i] <- out$pvalue[i] <- NA_real_
      if (inherits(augmented, "error")) {
        out$reason[i] <- "refit_failed"
      } else if (!isTRUE(augmented$converged)) {
        out$reason[i] <- "not_converged"
      } else {
        nested <- magmaanlab::policy_nested(augmented, fit$lab)$lr
        out$reason[i] <- nested$reason
        out$df[i] <- nested$df
        if (isTRUE(nested$available)) {
          out$statistic[i] <- nested$statistic
          out$pvalue[i] <- if (fit$estimator == "DWLS") nested$p_all else nested$p_peba4
        }
      }
    }
  }
  request_regressions <- is.character(candidates) && "~" %in% candidates ||
    is.data.frame(candidates) && any(candidates$op == "~")
  if (request_regressions && !any(out$op == "~")) {
    missing <- data.frame(kind = "fixed", lhs = "", op = "~", rhs = "", group = 0L,
      test = if (test == "score") "score" else if (fit$estimator == "DWLS") "fit_function_difference" else "lr",
      statistic = NA_real_, df = 1L, pvalue = NA_real_, epc = NA_real_, sepc.lv = NA_real_,
      sepc.all = NA_real_, reason = "unsupported_model")
    out <- rbind(out, missing)
  }
  attr(out, "candidate_row") <- NULL
  rownames(out) <- NULL
  out
}
