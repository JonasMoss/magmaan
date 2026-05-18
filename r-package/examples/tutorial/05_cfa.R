## lavaan tutorial — A CFA example   https://lavaan.ugent.be/tutorial/cfa.html
##
## The 3-factor Holzinger-Swineford 1939 CFA. lavaan does it with cfa() +
## summary(fit, fit.measures = TRUE); magmaan estimates with magmaan() and
## then asks for SEs and fit measures explicitly. Everything is cross-checked
## against lavaan::cfa().

suppressMessages({ library(magmaan); library(lavaan) })

near <- function(a, b, tol = 1e-3)
  isTRUE(all.equal(unname(as.numeric(a)), unname(as.numeric(b)), tolerance = tol))
ok <- function(cond, what) {
  cat(sprintf("  %-44s %s\n", what, if (isTRUE(cond)) "ok" else "MISMATCH"))
  stopifnot(isTRUE(cond))
}
est_match <- function(fit, lav, tol = 1e-3) {
  mp <- fit$partable[fit$partable$free > 0, , drop = FALSE]
  lp <- lavaan::parameterEstimates(lav)
  all(vapply(seq_len(nrow(mp)), function(i) {
    r <- lp[lp$lhs == mp$lhs[i] & lp$op == mp$op[i] & lp$rhs == mp$rhs[i], ]
    nrow(r) >= 1 && near(mp$est[i], r$est[1], tol)
  }, logical(1)))
}
## magmaan SE vector is free-index ordered; match it to lavaan by partable row.
se_match <- function(fit, se, lav, tol = 1e-2) {
  mp <- fit$partable[fit$partable$free > 0, , drop = FALSE]
  mp <- mp[order(mp$free), ]
  lp <- lavaan::parameterEstimates(lav)
  all(vapply(seq_len(nrow(mp)), function(i) {
    r <- lp[lp$lhs == mp$lhs[i] & lp$op == mp$op[i] & lp$rhs == mp$rhs[i], ]
    nrow(r) >= 1 && near(se[i], r$se[1], tol)
  }, logical(1)))
}

model <- "
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"
hs  <- HolzingerSwineford1939
fit <- magmaan(model, hs, estimator = "ML", se = "none", test = "none")
lav <- cfa(model, data = hs)

## SEs from the expected information; fit measures from the baseline model.
info <- magmaan_core$infer_information_expected(fit)
vc   <- magmaan_core$infer_vcov_partable(info, fit$partable)
se   <- magmaan_core$infer_se(vc)
ss   <- magmaan_core$fit_sample_stats(fit)
chi2 <- magmaan_core$infer_chi2_stat(ss, fit$fmin)
dfm  <- magmaan_core$infer_df_stat(fit$partable, ss)
fm   <- magmaan_core$measures_fit(fit, chi2, dfm, magmaan_core$infer_baseline(ss))
lfm  <- fitMeasures(lav, c("chisq", "df", "cfi", "tli", "rmsea", "srmr"))

cat("=== 3-factor Holzinger-Swineford CFA ===\n")
ok(fit$converged,                       "magmaan converged")
ok(est_match(fit, lav),                 "point estimates vs lavaan")
ok(se_match(fit, se, lav),              "standard errors vs lavaan")
ok(near(chi2, lfm[["chisq"]]),          "chi-square vs lavaan")
ok(dfm == lfm[["df"]],                  "degrees of freedom vs lavaan")
ok(near(c(fm$cfi, fm$tli, fm$rmsea, fm$srmr),
        lfm[c("cfi","tli","rmsea","srmr")]), "CFI / TLI / RMSEA / SRMR vs lavaan")
cat(sprintf("  chisq=%.3f df=%d  CFI=%.3f RMSEA=%.3f SRMR=%.3f\n",
            chi2, dfm, fm$cfi, fm$rmsea, fm$srmr))
cat("CFA example: ok\n")
