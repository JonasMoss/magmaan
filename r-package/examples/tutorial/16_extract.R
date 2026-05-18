## lavaan tutorial — Extracting information   https://lavaan.ugent.be/tutorial/inspect.html
##
## lavaan offers coef(), vcov(), fitMeasures(), fitted(), parameterEstimates(),
## etc. magmaan exposes the same quantities through the explicit `magmaan_core`
## primitives. This pulls each one and cross-checks it against lavaan; the few
## extractors the R package does not yet bind are noted at the end.

suppressMessages({ library(magmaan); library(lavaan) })

near <- function(a, b, tol = 1e-3)
  isTRUE(all.equal(unname(as.numeric(a)), unname(as.numeric(b)), tolerance = tol))
ok <- function(cond, what) {
  cat(sprintf("  %-44s %s\n", what, if (isTRUE(cond)) "ok" else "MISMATCH"))
  stopifnot(isTRUE(cond))
}
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

## coef()  — the free parameter estimates
free <- fit$partable[fit$partable$free > 0, ]
free <- free[order(free$free), ]
## vcov() / SE  — explicit information -> vcov -> se chain
info <- magmaan_core$infer_information_expected(fit)
vc   <- magmaan_core$infer_vcov_partable(info, fit$partable)
se   <- magmaan_core$infer_se(vc)
## fitMeasures()
ss   <- magmaan_core$fit_sample_stats(fit)
chi2 <- magmaan_core$infer_chi2_stat(ss, fit$fmin)
dfm  <- magmaan_core$infer_df_stat(fit$partable, ss)
fm   <- magmaan_core$measures_fit(fit, chi2, dfm, magmaan_core$infer_baseline(ss))
## fitted()  — the model-implied covariance matrix
implied <- magmaan_core$model_implied(fit)

cat("=== extracting information ===\n")
ok(near(free$est, coef(lav)),                  "coef() — point estimates vs lavaan")
ok(se_match(fit, se, lav),                     "vcov()/SE vs lavaan")
ok(near(c(fm$aic, fm$bic), fitMeasures(lav, c("aic","bic"))),
                                               "AIC / BIC vs lavaan")
ok(near(fm$cfi, fitMeasures(lav, "cfi")),      "fitMeasures() CFI vs lavaan")
impl_cov <- implied$sigma[[1]]      # model_implied() returns list(sigma=, mu=)
ok(near(as.numeric(impl_cov), as.numeric(fitted(lav)$cov)),
                                               "fitted() model-implied covariance vs lavaan")

cat("  not yet bound in the R package (magmaan::api only):",
    "standardizedSolution, modindices;\n",
    "  not implemented anywhere: lavResiduals, lavPredict",
    "(see docs/lavaan_tutorial_parity.md section 16).\n")
cat("extracting information: ok\n")
