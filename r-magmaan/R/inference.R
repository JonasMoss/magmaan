# Inference under magmaan's policy (project/design/r-interface-vision.md):
# the observed-information sandwich for parameter uncertainty, and global
# score and likelihood-ratio tests, each calibrated with SB and PEBA4. The
# policy is composed in C++ (magmaanlab::policy_inference()); a component it
# cannot compute carries a reason instead of a substitute result.

.inference_components <- c("covariance", "global_score", "global_lr")

#' Compute inference for a magmaan fit
#'
#' Runs magmaan's inference policy on an existing fit without refitting. A
#' component that cannot be computed is recorded with a reason; the estimates
#' remain usable.
#'
#' @param fit A [magmaan()] fit.
#' @return The fit with its inference results attached.
#' @export
infer <- function(fit) {
  if (!inherits(fit, "magmaan")) stop("infer(): supply a magmaan() fit", call. = FALSE)
  fit$inference <- .policy_inference(fit)
  fit
}

.policy_inference <- function(fit) {
  lab <- fit$lab
  res <- magmaanlab::policy_inference(lab)
  status <- data.frame(
    component = .inference_components,
    available = c(res$covariance_available, res$score$available, res$lr$available),
    reason = c(res$covariance_reason, res$score$reason, res$lr$reason),
    detail = c(res$covariance_detail, res$score$detail, res$lr$detail),
    stringsAsFactors = FALSE
  )
  out <- list(status = status)
  if (isTRUE(res$covariance_available)) {
    V <- res$covariance
    nm <- names(coef(fit))
    dimnames(V) <- list(nm, nm)
    out$covariance <- V
    if (any(lab$partable$op == ":=")) {
      out$defined <- magmaanlab::compute_defined(lab$syntax, lab, res$covariance)
    }
  }
  if (isTRUE(res$score$available)) out$global_score <- res$score
  if (isTRUE(res$lr$available)) out$global_lr <- res$lr
  out
}

# Condition raised when a caller asks for an inference result that does not
# exist; `reason` is machine-readable.
.inference_condition <- function(caller, component, reason, detail) {
  structure(
    class = c("magmaan_inference_unavailable", "error", "condition"),
    list(message = sprintf("%s: %s is unavailable (%s): %s", caller, component,
                           reason, detail),
         call = NULL, component = component, reason = reason)
  )
}

.inference_result <- function(fit, component, caller) {
  inf <- fit$inference
  if (is.null(inf)) {
    stop(.inference_condition(caller, component, "not_computed",
                              "the fit was made with inference = FALSE; call infer(fit)"))
  }
  row <- inf$status[inf$status$component == component, , drop = FALSE]
  if (!nrow(row) || !isTRUE(row$available)) {
    stop(.inference_condition(caller, component, row$reason %||% "unknown",
                              row$detail %||% ""))
  }
  inf[[component]]
}

.inference_label <- function(fit) {
  inf <- fit$inference
  if (is.null(inf)) return("not computed; call infer(fit)")
  s <- inf$status
  if (all(s$available)) return("computed")
  if (!any(s$available)) {
    reasons <- unique(s$reason)
    if (length(reasons) == 1L) return(paste0("unavailable (", reasons, ")"))
    return("unavailable")
  }
  paste0("partly available; missing ",
         paste0(s$component[!s$available], " (", s$reason[!s$available], ")",
                collapse = ", "))
}

.global_tests <- function(fit) {
  inf <- fit$inference
  if (is.null(inf)) return(NULL)
  rows <- lapply(c("global_score", "global_lr"), function(component) {
    t <- inf[[component]]
    if (is.null(t)) return(NULL)
    data.frame(test = if (component == "global_score") "score" else "likelihood ratio",
               statistic = t$statistic, df = t$df, p.sb = t$p_sb,
               p.peba4 = t$p_peba4, sb.scale = t$sb_scale,
               stringsAsFactors = FALSE)
  })
  rows <- Filter(Negate(is.null), rows)
  if (!length(rows)) return(NULL)
  do.call(rbind, rows)
}

`%||%` <- function(x, y) if (is.null(x)) y else x
