# Frontier / experimental: marker <-> std_lv identification swap.
#
# When a model's marker partable satisfies a small set of structural
# predicates, the per-factor rescaling (lambda' = lambda / c_f,
# psi' = c_f^2 * psi) is a bijection between the marker and std_lv
# parameterizations. Fitting in std_lv form is typically faster (median
# fit-only saving of ~18% on textbook-corpus models) and can be backconverted
# at near-zero cost. See experiments/02-latent_metric_identification/ for the
# validation that established the predicate set (0 false positives on the
# corpus).
#
# Three primitives:
#   is_std_lv_admissible_impl()           - structural check + npar safety net
#   backconvert_std_lv_to_marker_impl()   - vectorized parameter rescaling
#   fit_ml_auto_identification_impl()     - end-to-end wrapper that fits in
#                                           std_lv when admissible, backconverts,
#                                           and returns a marker-shaped fit
#                                           (downstream inference_* recomputes
#                                           Hessian from partable+theta+data,
#                                           so SE/vcov come out in marker coords
#                                           naturally — no delta-method needed).

is_std_lv_admissible_impl <- function(marker_spec, std_lv_spec = NULL) {
  pt <- marker_spec$partable
  opts <- marker_spec$options %||% list()

  if (isTRUE(opts$std_lv))
    return(list(admissible = FALSE, reason = "marker side already in std_lv"))
  if (!isTRUE(opts$auto_fix_first))
    return(list(admissible = FALSE, reason = "auto_fix_first off"))
  if (isTRUE(opts$effect_coding))
    return(list(admissible = FALSE, reason = "effect_coding"))
  if (identical(opts$model_type, "growth"))
    return(list(admissible = FALSE, reason = "growth model"))

  if (any(pt$op %in% c("==", ":=", "<", ">")))
    return(list(admissible = FALSE,
                reason = "partable contains ==/:=/</> constraint rows"))

  labs <- pt$label[nzchar(pt$label)]
  dups <- unique(labs[duplicated(labs)])
  if (length(dups))
    return(list(admissible = FALSE,
                reason = sprintf("duplicated label(s): %s",
                                 paste(dups, collapse = ", "))))

  latents <- unique(pt$lhs[pt$op == "=~"])

  load_rows <- pt[pt$op == "=~", , drop = FALSE]
  if (any(load_rows$rhs %in% latents))
    return(list(admissible = FALSE,
                reason = "higher-order factor (latent appears as rhs of =~)"))

  for (f in latents) {
    f_loads <- load_rows[load_rows$lhs == f, , drop = FALSE]
    n_fixed <- sum(f_loads$free == 0L)
    if (n_fixed != 1L)
      return(list(admissible = FALSE,
                  reason = sprintf("latent %s has %d fixed loadings (need exactly 1)",
                                   f, n_fixed)))
  }

  fixed <- which(pt$free == 0L)
  for (i in fixed) {
    op <- pt$op[[i]]; lhs <- pt$lhs[[i]]; rhs <- pt$rhs[[i]]
    v <- pt$ustart[[i]]
    if (op == "=~") {
      if (is.na(v) || !isTRUE(all.equal(v, 1)))
        return(list(admissible = FALSE,
                    reason = sprintf("fixed loading %s=~%s at %s (only 1 is safe)",
                                     lhs, rhs, format(v))))
    } else if (op == "~~") {
      if (lhs == rhs && (lhs %in% latents)) {
        return(list(admissible = FALSE,
                    reason = sprintf("latent variance %s~~%s fixed at %s (must be free in marker)",
                                     lhs, rhs, format(v))))
      }
      if ((lhs %in% latents) || (rhs %in% latents)) {
        if (is.na(v) || !isTRUE(all.equal(v, 0)))
          return(list(admissible = FALSE,
                      reason = sprintf("fixed nonzero latent covariance %s~~%s at %s",
                                       lhs, rhs, format(v))))
      }
    } else if (op == "~") {
      if ((lhs %in% latents) || (rhs %in% latents)) {
        if (is.na(v) || !isTRUE(all.equal(v, 0)))
          return(list(admissible = FALSE,
                      reason = sprintf("fixed nonzero regression %s~%s at %s (touches a latent)",
                                       lhs, rhs, format(v))))
      }
    } else if (op == "~1") {
      if (lhs %in% latents) {
        if (is.na(v) || !isTRUE(all.equal(v, 0)))
          return(list(admissible = FALSE,
                      reason = sprintf("fixed nonzero latent intercept %s~1 at %s",
                                       lhs, format(v))))
      }
    }
  }

  # npar safety net: catches lavaan-specific quirks where the user supplied
  # `1*indicator` modifiers that survive std.lv=TRUE and create extra
  # constraints not visible from the marker partable alone.
  if (!is.null(std_lv_spec)) {
    n_marker <- sum(pt$free > 0L, na.rm = TRUE)
    n_stdlv <- sum(std_lv_spec$partable$free > 0L, na.rm = TRUE)
    if (n_marker != n_stdlv)
      return(list(admissible = FALSE,
                  reason = sprintf("npar mismatch after swap (marker=%d, std_lv=%d)",
                                   n_marker, n_stdlv)))
  }

  list(admissible = TRUE, reason = "")
}

backconvert_std_lv_to_marker_impl <- function(std_lv_fit, marker_partable) {
  std_pt <- std_lv_fit$partable
  est <- std_pt$est
  m_pt <- marker_partable
  lhs <- std_pt$lhs; rhs <- std_pt$rhs; op <- std_pt$op
  is_load <- op == "=~"
  latents <- unique(lhs[is_load])

  marker_anchors <- m_pt$op == "=~" & m_pt$free == 0 &
                    !is.na(m_pt$ustart) & m_pt$ustart == 1
  anchor_key <- paste0(m_pt$lhs[marker_anchors], "\r",
                       m_pt$rhs[marker_anchors])
  std_key <- paste0(lhs, "\r", rhs)
  anchor_idx <- match(anchor_key, std_key)
  c_per <- setNames(rep(1, length(latents)), latents)
  ok <- !is.na(anchor_idx)
  if (any(ok)) c_per[m_pt$lhs[marker_anchors][ok]] <- est[anchor_idx[ok]]

  lhs_c <- c_per[lhs]; lhs_c[is.na(lhs_c)] <- 1
  rhs_c <- c_per[rhs]; rhs_c[is.na(rhs_c)] <- 1
  out <- est

  # Loadings: factor on lhs, indicator on rhs. For higher-order loadings the
  # rhs is itself a latent, so we multiply by c[rhs] as well.
  out[is_load] <- est[is_load] * rhs_c[is_load] / lhs_c[is_load]

  # Latent variances: multiply by c[lhs]^2 (lhs == rhs by construction).
  is_lv_var <- op == "~~" & lhs == rhs & lhs %in% latents
  out[is_lv_var] <- est[is_lv_var] * lhs_c[is_lv_var]^2

  # Latent covariances: multiply by c[lhs] * c[rhs].
  is_lv_cov <- op == "~~" & lhs != rhs & lhs %in% latents & rhs %in% latents
  out[is_lv_cov] <- est[is_lv_cov] * lhs_c[is_lv_cov] * rhs_c[is_lv_cov]

  # Regressions: multiply by c[lhs] / c[rhs]; observed sides have c == 1.
  is_reg <- op == "~"
  out[is_reg] <- est[is_reg] * lhs_c[is_reg] / rhs_c[is_reg]

  # Latent intercepts: multiply by c[lhs] (latent mean scales linearly with
  # the latent: f_marker = c_f * f_std_lv => mean(f_marker) = c_f * mean).
  is_int <- op == "~1" & lhs %in% latents
  out[is_int] <- est[is_int] * lhs_c[is_int]

  out
}

# Convenience wrapper: build marker + std_lv specs from `model`, run the
# admissibility check, fit in whichever parameterization is faster, and return
# a marker-shaped fit. Downstream inference (`inference_se`, fit measures,
# etc.) operates on (partable, theta, data) and so produces marker-coord
# results naturally — no delta-method machinery needed.
#
# The result has an `auto_identification` slot recording which path was taken:
#   path = "std_lv_swap"  : checker admitted, fit in std_lv, backconverted
#   path = "marker_direct": checker rejected, fit in marker directly
#   reason                : the checker's reason when path == "marker_direct"
#
# Internal optimization-side diagnostics (`grad_norm`, `audit`, `iterations`,
# `f_evals`, `g_evals`) reflect the parameterization actually fitted; on the
# "std_lv_swap" path they are std_lv-coord and labeled as such.

fit_ml_auto_identification_impl <- function(model, data,
                                            optimizer = "nlopt-lbfgs",
                                            control = NULL) {
  # Resolve `model` to a marker spec. We need syntax + options to build the
  # std_lv counterpart, so spec/data.frame inputs that lack syntax fall back
  # to direct fitting.
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
    # Partable or otherwise; we cannot rebuild std_lv from it, so fit direct.
    fit <- magmaan_core$fit_ml(model, data, optimizer = optimizer,
                               control = control)
    fit$auto_identification <- list(
      path = "direct",
      reason = "input is a partable/data.frame; cannot synthesize std_lv form"
    )
    return(fit)
  }

  std_lv_opts <- marker_spec$options
  std_lv_opts$std_lv <- TRUE
  std_lv_opts$auto_fix_first <- FALSE
  std_lv_spec <- do.call(
    model_spec,
    c(list(marker_spec$syntax), std_lv_opts)
  )

  check <- is_std_lv_admissible_impl(marker_spec, std_lv_spec)
  if (!isTRUE(check$admissible)) {
    fit <- magmaan_core$fit_ml(marker_spec, data, optimizer = optimizer,
                               control = control)
    fit$auto_identification <- list(path = "marker_direct",
                                    admissible = FALSE,
                                    reason = check$reason)
    return(fit)
  }

  std_fit <- magmaan_core$fit_ml(std_lv_spec, data, optimizer = optimizer,
                                 control = control)
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
