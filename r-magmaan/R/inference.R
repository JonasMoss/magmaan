# Inference under magmaan's policy (project/design/r-interface-vision.md):
# the observed-information sandwich for parameter uncertainty, and global
# score and likelihood-ratio tests, each calibrated with SB and PEBA4. The
# composer lives in C++; until it lands every component reports a typed
# reason instead of a substitute result.

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
  status <- data.frame(
    component = .inference_components,
    available = FALSE,
    reason = "not_implemented",
    stringsAsFactors = FALSE
  )
  list(status = status,
       detail = "the inference policy composer is not implemented yet")
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
    stop(.inference_condition(caller, component, row$reason %||% "unknown", inf$detail))
  }
  inf[[component]]
}

.inference_label <- function(fit) {
  inf <- fit$inference
  if (is.null(inf)) return("not computed; call infer(fit)")
  if (all(inf$status$available)) return("computed")
  if (!any(inf$status$available)) return(paste("unavailable:", inf$detail))
  paste("partly available:", paste(inf$status$component[!inf$status$available],
                                   collapse = ", "), "missing")
}

`%||%` <- function(x, y) if (is.null(x)) y else x
