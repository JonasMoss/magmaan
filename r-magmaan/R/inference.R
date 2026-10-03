# Inference under magmaan's policy (project/design/r-interface-vision.md):
# the observed-information sandwich for parameter uncertainty, and global
# ML/FIML score and likelihood-ratio tests calibrated with SB and PEBA4,
# and the DWLS global fit-function test with the exact spectrum All tail. The
# policy is composed in C++ (magmaanlab::policy_inference()); a component it
# cannot compute carries a reason instead of a substitute result.

.inference_components <- c("covariance", "global_score", "global_lr")

#' Compute inference for a magmaan fit
#'
#' Runs the selected inference bundle on an existing fit without refitting. A
#' component that cannot be computed is recorded with a reason; the estimates
#' remain usable.
#'
#' @param fit A [magmaan()] fit.
#' @param lavaan_compat `NULL` computes the default policy. A lavaan bundle
#'   such as `"MLM"`, `"MLR"` or `"WLSMV"` stores an additional compatibility bundle
#'   for reuse by the reporting methods; it leaves the policy intact.
#' @return The fit with its inference results attached.
#' @export
infer <- function(fit, lavaan_compat = NULL) {
  if (!inherits(fit, "magmaan")) stop("infer(): supply a magmaan() fit", call. = FALSE)
  lavaan_compat <- .check_lavaan_compat(fit, lavaan_compat, "infer()")
  if (is.null(lavaan_compat)) fit$inference <- .policy_inference(fit)
  else fit$lavaan_compat[[lavaan_compat]] <- .lavaan_compat_inference(fit, lavaan_compat)
  fit
}

.check_lavaan_compat <- function(fit, lavaan_compat, caller) {
  choices <- c("ML", "MLM", "MLR", "DWLS", "WLSMV", "ULS", "ULSMV", "WLS")
  if (is.null(lavaan_compat)) return(NULL)
  lavaan_compat <- .check_choice(lavaan_compat, "lavaan_compat", choices, caller = caller)
  compatible <- if (isTRUE(fit$lab$ordinal)) {
    switch(fit$estimator, DWLS = c("DWLS", "WLSMV"), ULS = c("ULS", "ULSMV"), WLS = "WLS", character())
  } else switch(fit$estimator, ML = c("ML", "MLM", "MLR"), FIML = c("ML", "MLR"),
               ULS = "ULS", WLS = "WLS", character())
  if (!lavaan_compat %in% compatible) {
    stop(sprintf("%s: lavaan_compat = \"%s\" is incompatible with this %s fit; compatibility bundles change inference, so fit the required estimator first",
                 caller, lavaan_compat, fit$estimator), call. = FALSE)
  }
  lavaan_compat
}

.lavaan_compat_inference <- function(fit, lavaan_compat) {
  res <- magmaanlab::convention_inference(fit$lab, lavaan_compat)
  out <- list(lavaan_compat = lavaan_compat,
    status = data.frame(component = c("covariance", "global_lr"),
      available = c(res$covariance_available, res$test$available),
      reason = c(res$covariance_reason, res$test$reason),
      detail = c(res$covariance_detail, res$test$detail), stringsAsFactors = FALSE),
    psd_boundary = isTRUE(res$psd_boundary), convergence = .convergence_record(fit$lab, res))
  if (isTRUE(res$covariance_available)) {
    out$covariance <- res$covariance
    nm <- names(coef(fit))
    dimnames(out$covariance) <- list(nm, nm)
    attr(out$covariance, "lavaan_compat") <- lavaan_compat
    if (any(fit$lab$partable$op == ":="))
      out$defined <- magmaanlab::compute_defined(fit$lab$syntax, fit$lab, res$covariance)
  }
  if (isTRUE(res$test$available)) out$global_lr <- res$test
  out
}

# Selecting a compatibility bundle creates a local reporting view; the fit's policy and
# any other cached compatibility bundle remain intact under R's value semantics.
.with_lavaan_compat <- function(fit, lavaan_compat, caller) {
  lavaan_compat <- .check_lavaan_compat(fit, lavaan_compat, caller)
  if (!is.null(lavaan_compat))
    fit$inference <- fit$lavaan_compat[[lavaan_compat]] %||% .lavaan_compat_inference(fit, lavaan_compat)
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
  # A PSD estimate on the cone boundary: computed, valid for an interior
  # population (project/design/r-interface-vision.md, Availability).
  out <- list(status = status, psd_boundary = isTRUE(res$psd_boundary),
              convergence = .convergence_record(lab, res))
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

# The rule that decided convergence, magmaan's own check and whether they
# disagree. They can disagree only under a compatibility rule such as the
# lavaan-0.7.2 preset; inference then follows the selected rule.
.convergence_record <- function(lab, res) {
  list(rule = lab$fitting$effective$convergence %||% "newton",
       converged = isTRUE(lab$converged),
       magmaan = lab$diagnostics$verdict$status %||% NA_character_,
       disagree = isTRUE(res$verdict_disagreement))
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
  if (!nrow(row)) {
    stop(.inference_condition(caller, component, "unknown", "no component status was retained"))
  }
  if (!isTRUE(row$available)) {
    stop(.inference_condition(caller, component, row$reason %||% "unknown",
                              row$detail %||% ""))
  }
  inf[[component]]
}

.inference_label <- function(fit) {
  label <- .inference_status_label(fit)
  if (!is.null(fit$inference$lavaan_compat))
    label <- paste0("lavaan compatibility: ", fit$inference$lavaan_compat, "; ", label)
  if (isTRUE(fit$inference$convergence$disagree)) paste0(label, "; see the convergence note") else label
}

.inference_status_label <- function(fit) {
  inf <- fit$inference
  if (is.null(inf)) return("not computed; call infer(fit)")
  s <- inf$status
  # A component that does not exist for the estimator (an LR test without a
  # likelihood) is not missing.
  s <- s[s$available | s$reason != "inapplicable", , drop = FALSE]
  if (all(s$available))
    return(if (isTRUE(inf$psd_boundary)) "computed, assuming an interior population" else "computed")
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
  if (!is.null(inf$lavaan_compat)) {
    t <- inf$global_lr
    if (is.null(t)) return(NULL)
    return(.lavaan_compat_test_row(t))
  }
  rows <- lapply(c("global_score", "global_lr"), function(component) {
    t <- inf[[component]]
    if (is.null(t)) return(NULL)
    label <- if (component == "global_lr") "likelihood ratio" else
      if (identical(t$label, "fit_function")) "fit function" else "score"
    if (identical(t$reference, "all"))
      return(data.frame(test = label, statistic = t$statistic, df = t$df,
                        pvalue = t$p_all, reference = "exact spectrum (All)",
                        stringsAsFactors = FALSE))
    data.frame(test = label,
               statistic = t$statistic, df = t$df, p.sb = t$p_sb,
               p.peba4 = t$p_peba4, sb.scale = t$sb_scale,
               stringsAsFactors = FALSE)
  })
  rows <- Filter(Negate(is.null), rows)
  if (!length(rows)) return(NULL)
  out <- do.call(rbind, rows)
  tests <- Filter(Negate(is.null), inf[c("global_score", "global_lr")])
  attr(out, "peba_blocks") <- vapply(tests,
    function(t) as.integer(t$peba_blocks %||% 0L), integer(1))
  out
}

.lavaan_compat_test_row <- function(t) {
  data.frame(test = t$method, statistic = t$statistic, df = t$df,
             pvalue = t$pvalue, unscaled.statistic = t$unscaled_statistic,
             scale = t$scale, shift = t$shift, stringsAsFactors = FALSE)
}

`%||%` <- function(x, y) if (is.null(x)) y else x

.peba_note <- function(t) {
  blocks <- attr(t, "peba_blocks")
  if (is.null(blocks)) return(invisible(NULL))
  reduced <- which(blocks > 0L & blocks < 4L & is.finite(t$p.peba4))
  if (length(reduced)) {
    notes <- unique(paste0(blocks[reduced], " eigenvalue blocks (df = ", t$df[reduced], ")"))
    cat("PEBA4 formed ", paste(notes, collapse = "; "), ".\n", sep = "")
  }
  invisible(NULL)
}
