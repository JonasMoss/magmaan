# The competing tests of H0: alpha = omega (equivalently tau-equivalence /
# equal loadings within a one-factor model), plus the per-replication driver.
# Estimation is traditional normal-theory ML throughout; the only axis is how
# the hypothesis is tested.
#
#   Wald (reliability) : Deng & Chan z-test on (omega_hat - alpha_hat). A
#                        1-df contrast that is NON-REGULAR at the null (its
#                        gradient vanishes there) -- the baseline to beat.
#   Imhof (reliability): same statistic, referenced against its true
#                        quadratic-form (sum of chi-square_1) law. The fix.
#                        See R/alpha_omega.R.
#   Wald (structural)  : regular Wald test of the equal-loading restriction in
#                        the free-loading one-factor model. (p-1)-df,
#                        NORMAL-THEORY ONLY.
#   Satorra-2000 (LR)  : scaled likelihood-ratio test of the equal-loading
#                        restriction; the Imhof "mixture" p-value is its exact
#                        weighted-chi-square tail. (p-1)-df, regular.
#   Score / LM         : lavaan's standard score test of releasing the
#                        equal-loading constraint, from the restricted fit
#                        alone. (p-1)-df, regular, NORMAL-THEORY ONLY (lavaan
#                        has no robust release-score test).

# Standard (normal-theory) Wald test of tau-equivalence: fit the unrestricted
# equal-loading model with labeled loadings and test L2 = L1, ..., Lp = L1.
wald_tau_test <- function(d) {
  d <- as.data.frame(d); ov <- colnames(d)
  labs <- paste0("L", seq_along(ov))
  m1 <- paste0("f =~ ", paste0(labs, "*", ov, collapse = " + "),
               "\n f ~~ 1*f")
  constraints <- paste(paste0(labs[-1L], " == ", labs[1L]), collapse = "; ")
  out <- tryCatch({
    fit1 <- lavaan::cfa(m1, d, std.lv = FALSE, meanstructure = FALSE,
                        auto.fix.first = FALSE)
    wt <- lavaan::lavTestWald(fit1, constraints = constraints)
    list(p_wald_equal = wt$p.value, stat = wt$stat, df = wt$df,
         converged = isTRUE(lavaan::lavInspect(fit1, "converged")))
  }, error = function(e) NULL)
  if (is.null(out)) {
    return(list(p_wald_equal = NA_real_, stat = NA_real_, df = NA_integer_,
                converged = FALSE))
  }
  out
}

# Satorra-2000 scaled LRT of equal loadings (ML H1 vs ML H0), via magmaan's
# empirical-Gamma restriction map. Returns the scaled and exact-Imhof-mixture
# p-values, or NA on failure.
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
    list(p_scaled = nt$p_scaled, p_mixture = nt$p_mixture, df = nt$df_diff,
         converged = isTRUE(fit_h1$converged) && isTRUE(fit_h0$converged))
  }, error = function(e) NULL)
  if (is.null(out)) {
    return(list(p_scaled = NA_real_, p_mixture = NA_real_,
                df = NA_integer_, converged = FALSE))
  }
  out
}

# Standard (normal-theory) score test of tau-equivalence: fit the restricted
# equal-loading model and test releasing the equalities. Uses lavaan, whose
# robust release-score test is not implemented (it silently falls back to the
# ordinary score test), so this is the NT-only trinity member.
score_tau_test <- function(d) {
  d <- as.data.frame(d); ov <- colnames(d)
  m0 <- paste0("f =~ ", paste0("a*", ov, collapse = " + "))
  out <- tryCatch({
    fit0 <- lavaan::cfa(m0, d, std.lv = TRUE, meanstructure = FALSE)
    sc <- lavaan::lavTestScore(fit0)
    list(p_score = sc$test$p.value[1L], df = sc$test$df[1L])
  }, error = function(e) NULL)
  if (is.null(out)) return(list(p_score = NA_real_, df = NA_integer_))
  out
}

# One replication: every test on the same sample. One row of p-values + flags.
run_one_rep <- function(d) {
  rt <- tryCatch(reliability_tests(d), error = function(e) NULL)
  wt <- wald_tau_test(d)
  st <- satorra_tau_test(d)
  sc <- score_tau_test(d)
  if (is.null(rt)) {
    rt <- list(omega = NA, alpha = NA, diff = NA,
               p_wald_nm = NA_real_, p_wald_sw = NA_real_,
               p_imhof_nm = NA_real_, p_imhof_sw = NA_real_, converged = FALSE)
  }
  data.frame(
    omega = rt$omega, alpha = rt$alpha, diff = rt$diff,
    p_wald_nm = rt$p_wald_nm, p_wald_sw = rt$p_wald_sw,
    p_imhof_nm = rt$p_imhof_nm, p_imhof_sw = rt$p_imhof_sw,
    p_wald_equal = wt$p_wald_equal,
    p_satorra_scaled = st$p_scaled, p_satorra_mixture = st$p_mixture,
    p_score = sc$p_score,
    rel_converged = isTRUE(rt$converged),
    wald_converged = isTRUE(wt$converged),
    st_converged = isTRUE(st$converged)
  )
}

# Reject-at-level helper that ignores NA (non-converged) replications.
rejection_rate <- function(pvals, level = 0.05) {
  ok <- pvals[is.finite(pvals)]
  if (!length(ok)) return(NA_real_)
  mean(ok < level)
}
