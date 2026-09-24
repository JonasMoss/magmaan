# Structural admissibility check for the marker -> std_lv -> backconvert swap.
#
# Returns list(admissible, reason). When admissible == TRUE, the per-factor
# rescaling that defines backconvert is a bijection on the parameter space and
# preserves every constraint in the partable.

is_std_lv_admissible <- function(marker_spec, std_lv_spec = NULL) {
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

  # Higher-order factors (a latent appears as rhs of =~): conservatively
  # reject. Algebraically the per-factor rescaling can still be defined, but
  # the std_lv fit on a hierarchy can land at a different local optimum and
  # the backconvert pass needs the rhs-rescaling branch.
  load_rows <- pt[pt$op == "=~", , drop = FALSE]
  if (any(load_rows$rhs %in% latents))
    return(list(admissible = FALSE,
                reason = "higher-order factor (latent appears as rhs of =~)"))

  # Exactly one fixed loading per latent (the marker fix). More than one is a
  # tau-equivalence-style constraint in marker that std_lv does not mirror.
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
        # Latent VARIANCE in marker form must be free — std_lv will fix it,
        # and any user-supplied fix here would create an over-identified swap.
        return(list(admissible = FALSE,
                    reason = sprintf("latent variance %s~~%s fixed at %s (must be free in marker)",
                                     lhs, rhs, format(v))))
      }
      if ((lhs %in% latents) || (rhs %in% latents)) {
        # Latent COVARIANCE: only fixed-at-0 is safe.
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

  # Final safety net: build std_lv partable and require matching npar. Catches
  # lavaan-specific quirks (user-supplied `1*indicator` modifiers that survive
  # std.lv=TRUE and create extra constraints).
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
