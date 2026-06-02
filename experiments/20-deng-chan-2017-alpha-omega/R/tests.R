# The two competing tests of H0: alpha = omega (equivalently, tau-equivalence
# within the one-factor model), plus the per-replication driver.
#
#   Deng-Chan Wald : z-test of (omega_hat - alpha_hat), normal-theory (nm) and
#                    sandwich (sw) standard errors. See R/alpha_omega.R.
#   Satorra-2000   : scaled likelihood-ratio test of the equal-loading
#                    restriction (H1 congeneric one-factor vs H0 tau), using
#                    magmaan's empirical-Gamma restriction-map machinery.

# Satorra-2000 scaled LRT of equal loadings. Fits H1 (free loadings, std.lv)
# and H0 (equal loadings, std.lv) by ML and runs magmaan::nestedTest. Returns
# the unscaled chi-square-difference p-value and the Satorra-Bentler-scaled
# p-value, or NA on any failure.
satorra_tau_test <- function(d) {
  d <- as.data.frame(d); ov <- colnames(d)
  m_h1 <- paste0("f =~ ", paste0("L", seq_along(ov), "*", ov, collapse = " + "),
                 "\n f ~~ 1*f")
  m_h0 <- paste0("f =~ ", paste0("a*", ov, collapse = " + "), "\n f ~~ 1*f")
  out <- tryCatch({
    fit_h1 <- magmaan::magmaan(m_h1, d, estimator = "ML", auto_fix_first = FALSE)
    fit_h0 <- magmaan::magmaan(m_h0, d, estimator = "ML", auto_fix_first = FALSE)
    nt <- magmaan::nestedTest(fit_h1, fit_h0, d,
                              method = "satorra.2000", gamma = "empirical")
    list(p_unscaled = nt$p_unscaled, p_scaled = nt$p_scaled,
         df = nt$df_diff, converged = isTRUE(fit_h1$converged) &&
           isTRUE(fit_h0$converged))
  }, error = function(e) NULL)
  if (is.null(out)) {
    return(list(p_unscaled = NA_real_, p_scaled = NA_real_,
                df = NA_integer_, converged = FALSE))
  }
  out
}

# One replication: both tests on the same sample. Returns a one-row data frame
# of p-values (and convergence flags) for the four reported variants.
run_one_rep <- function(d) {
  dc <- tryCatch(deng_chan_test(d), error = function(e) NULL)
  st <- satorra_tau_test(d)
  if (is.null(dc)) {
    dc <- list(omega = NA, alpha = NA, diff = NA,
               p_nm = NA_real_, p_sw = NA_real_, converged = FALSE)
  }
  data.frame(
    omega = dc$omega, alpha = dc$alpha, diff = dc$diff,
    p_wald_nm = dc$p_nm, p_wald_sw = dc$p_sw,
    p_satorra_unscaled = st$p_unscaled, p_satorra_scaled = st$p_scaled,
    dc_converged = isTRUE(dc$converged),
    st_converged = isTRUE(st$converged)
  )
}

# Reject-at-level helper that ignores NA (non-converged) replications.
rejection_rate <- function(pvals, level = 0.05) {
  ok <- pvals[is.finite(pvals)]
  if (!length(ok)) return(NA_real_)
  mean(ok < level)
}
