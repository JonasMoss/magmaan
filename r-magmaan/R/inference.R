# Inference under magmaan's policy (project/design/r-interface-vision.md):
# the observed-information sandwich for parameter uncertainty, and global
# score and likelihood-ratio tests, each calibrated with SB and PEBA4. The
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
#' @param convention `"magmaan"` computes the default policy. A lavaan bundle
#'   such as `"MLM"`, `"MLR"` or `"WLSMV"` stores an additional convention
#'   for reuse by the reporting methods; it leaves the policy intact.
#' @return The fit with its inference results attached.
#' @export
infer <- function(fit, convention = "magmaan") {
  if (!inherits(fit, "magmaan")) stop("infer(): supply a magmaan() fit", call. = FALSE)
  convention <- .check_convention(fit, convention, "infer()")
  if (convention == "magmaan") fit$inference <- .policy_inference(fit)
  else fit$conventions[[convention]] <- .convention_inference(fit, convention)
  fit
}

.check_convention <- function(fit, convention, caller) {
  choices <- c("magmaan", "ML", "MLM", "MLR", "DWLS", "WLSMV", "ULS", "ULSMV", "WLS")
  convention <- .check_choice(convention, "convention", choices, caller = caller)
  if (convention == "magmaan") return(convention)
  compatible <- if (isTRUE(fit$lab$ordinal)) {
    switch(fit$estimator, DWLS = c("DWLS", "WLSMV"), ULS = c("ULS", "ULSMV"), WLS = "WLS", character())
  } else switch(fit$estimator, ML = c("ML", "MLM", "MLR"), FIML = c("ML", "MLR"),
               ULS = "ULS", WLS = "WLS", character())
  if (!convention %in% compatible) {
    stop(sprintf("%s: convention = \"%s\" is incompatible with this %s fit; conventions change inference, so fit the required estimator first",
                 caller, convention, fit$estimator), call. = FALSE)
  }
  convention
}

.convention_inference <- function(fit, convention) {
  res <- magmaanlab::convention_inference(fit$lab, convention)
  out <- list(convention = convention,
    status = data.frame(component = c("covariance", "global_lr"),
      available = c(res$covariance_available, res$test$available),
      reason = c(res$covariance_reason, res$test$reason),
      detail = c(res$covariance_detail, res$test$detail), stringsAsFactors = FALSE),
    psd_boundary = isTRUE(res$psd_boundary), convergence = .convergence_record(fit$lab, res))
  if (isTRUE(res$covariance_available)) {
    out$covariance <- res$covariance
    nm <- names(coef(fit))
    dimnames(out$covariance) <- list(nm, nm)
    attr(out$covariance, "convention") <- convention
    if (any(fit$lab$partable$op == ":="))
      out$defined <- magmaanlab::compute_defined(fit$lab$syntax, fit$lab, res$covariance)
  }
  if (isTRUE(res$test$available)) out$global_lr <- res$test
  out
}

# Selecting a convention creates a local reporting view; the fit's policy and
# any other cached convention remain intact under R's value semantics.
.with_convention <- function(fit, convention, caller) {
  convention <- .check_convention(fit, convention, caller)
  if (convention != "magmaan")
    fit$inference <- fit$conventions[[convention]] %||% .convention_inference(fit, convention)
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
  if (!is.null(fit$inference$convention))
    label <- paste0("lavaan ", fit$inference$convention, "; ", label)
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
  if (!is.null(inf$convention)) {
    t <- inf$global_lr
    if (is.null(t)) return(NULL)
    return(.convention_test_row(t))
  }
  rows <- lapply(c("global_score", "global_lr"), function(component) {
    t <- inf[[component]]
    if (is.null(t)) return(NULL)
    label <- if (component == "global_lr") "likelihood ratio" else
      if (identical(t$label, "fit_function")) "fit function" else "score"
    data.frame(test = label,
               statistic = t$statistic, df = t$df, p.sb = t$p_sb,
               p.peba4 = t$p_peba4, sb.scale = t$sb_scale,
               stringsAsFactors = FALSE)
  })
  rows <- Filter(Negate(is.null), rows)
  if (!length(rows)) return(NULL)
  do.call(rbind, rows)
}

.convention_test_row <- function(t) {
  data.frame(test = t$method, statistic = t$statistic, df = t$df,
             pvalue = t$pvalue, unscaled.statistic = t$unscaled_statistic,
             scale = t$scale, shift = t$shift, stringsAsFactors = FALSE)
}

`%||%` <- function(x, y) if (is.null(x)) y else x
