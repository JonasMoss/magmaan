# Frontier / experimental: marker <-> std_lv identification swap.
#
# All three primitives delegate to C++ implementations in
# src/model/auto_identification.cpp via Rcpp shims defined in
# r-package/src/fit.cpp (`frontier_*_impl`).
#
# Exposed to R only via the `frontier_*` aliases in r-package/R/zzz_core.R.
# Not part of the core stable API. See
# experiments/_archive/02-latent-metric-identification/ for the validation that
# established the predicate set (0 false positives on the textbook corpus).
#
# Surface:
#   is_std_lv_admissible_impl(marker_spec, std_lv_spec?)
#     Structural check + npar safety net. The R wrapper handles the option
#     gates (std_lv flag, effect_coding, growth, auto_fix_first) that aren't
#     visible from the partable alone, then delegates row-wise predicates to
#     C++.
#
#   backconvert_std_lv_to_marker_impl(std_lv_fit, marker_partable)
#     Per-factor rescaling of the std_lv fit's `est` vector back to marker
#     coords. Returns a numeric vector aligned with marker_partable rows.
#     Pure C++ call.
#
#   fit_ml_auto_identification_impl(model, data, optimizer, control)
#     Build marker spec, run check, fit in whichever parameterization is
#     faster, return a marker-shaped fit. On the std_lv_swap path the std_lv
#     partable comes from partable-surgery on the marker (NO second lavaanify
#     call). Downstream inference_* recomputes the Hessian from
#     (partable, theta, data), so SE/vcov come out in marker coords naturally.

is_std_lv_admissible_impl <- function(marker_spec, std_lv_spec = NULL) {
  opts <- marker_spec$options %||% list()

  if (isTRUE(opts$std_lv))
    return(list(admissible = FALSE, reason = "marker side already in std_lv"))
  if (!isTRUE(opts$auto_fix_first))
    return(list(admissible = FALSE, reason = "auto_fix_first off"))
  if (isTRUE(opts$effect_coding))
    return(list(admissible = FALSE, reason = "effect_coding"))
  if (identical(opts$model_type, "growth"))
    return(list(admissible = FALSE, reason = "growth model"))

  std_lv_pt <- if (is.null(std_lv_spec)) NULL else std_lv_spec$partable
  frontier_is_std_lv_admissible_impl(marker_spec$partable, std_lv_pt)
}

backconvert_std_lv_to_marker_impl <- function(std_lv_fit, marker_partable) {
  frontier_backconvert_std_lv_to_marker_impl(marker_partable,
                                             std_lv_fit$partable$est)
}

fit_ml_auto_identification_impl <- function(model, data,
                                            optimizer = "nlopt-lbfgs",
                                            control = NULL) {
  # Resolve `model` to a marker spec. We need the partable + options to do
  # admissibility check + partable surgery; spec/data.frame inputs that
  # don't expose those fall back to direct fitting.
  marker_spec <- NULL
  if (inherits(model, "magmaan_model_spec")) {
    if (isTRUE(model$options$std_lv)) {
      fit <- magmaan_core$fit_ml(model, data, optimizer = optimizer,
                                 control = control)
      fit$auto_identification <- list(path = "direct",
                                      reason = "input spec is already std_lv")
      return(fit)
    }
    marker_spec <- model
  } else if (is.character(model) && length(model) == 1L) {
    marker_spec <- model_spec(model, std_lv = FALSE, auto_fix_first = TRUE)
  } else {
    fit <- magmaan_core$fit_ml(model, data, optimizer = optimizer,
                               control = control)
    fit$auto_identification <- list(
      path = "direct",
      reason = "input is a partable/data.frame; cannot synthesize std_lv form"
    )
    return(fit)
  }

  # Partable surgery (C++): produces the std_lv partable without a second
  # lavaanify call. This is the biggest single overhead saver vs. the
  # `model_spec(..., std_lv = TRUE)` approach.
  std_lv_partable <- frontier_partable_marker_to_std_lv_impl(
    marker_spec$partable
  )

  check <- is_std_lv_admissible_impl(
    marker_spec,
    list(partable = std_lv_partable)
  )
  if (!isTRUE(check$admissible)) {
    fit <- magmaan_core$fit_ml(marker_spec, data, optimizer = optimizer,
                               control = control)
    fit$auto_identification <- list(path = "marker_direct",
                                    admissible = FALSE,
                                    reason = check$reason)
    return(fit)
  }

  std_fit <- magmaan_core$fit_ml(std_lv_partable, data,
                                 optimizer = optimizer, control = control)
  if (!isTRUE(std_fit$converged)) {
    fit <- magmaan_core$fit_ml(marker_spec, data, optimizer = optimizer,
                               control = control)
    fit$auto_identification <- list(path = "marker_direct",
                                    admissible = TRUE,
                                    reason = "std_lv fit did not converge")
    return(fit)
  }

  marker_est <- backconvert_std_lv_to_marker_impl(std_fit,
                                                   marker_spec$partable)

  out <- std_fit
  out$partable <- marker_spec$partable
  out$partable$est <- marker_est
  free <- marker_spec$partable$free
  npar <- max(free, 0L)
  theta <- numeric(npar)
  free_mask <- free > 0L
  theta[free[free_mask]] <- marker_est[free_mask]
  out$theta <- theta
  out$auto_identification <- list(path = "std_lv_swap",
                                  admissible = TRUE,
                                  reason = "")
  out
}
