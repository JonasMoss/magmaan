## lavaan tutorial — A SEM example   https://lavaan.ugent.be/tutorial/sem.html
##
## Bollen's industrialization / political-democracy model: three latent
## variables, latent regressions, and a set of correlated residuals. lavaan
## fits it with sem(); magmaan with magmaan(estimator = "ML").

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

model <- "
  # measurement model
  ind60 =~ x1 + x2 + x3
  dem60 =~ y1 + y2 + y3 + y4
  dem65 =~ y5 + y6 + y7 + y8
  # latent regressions
  dem60 ~ ind60
  dem65 ~ ind60 + dem60
  # correlated residuals
  y1 ~~ y5
  y2 ~~ y4 + y6
  y3 ~~ y7
  y4 ~~ y8
  y6 ~~ y8
"
fit <- magmaan(model, PoliticalDemocracy, estimator = "ML",
               se = "none", test = "none")
lav <- sem(model, data = PoliticalDemocracy)

ss   <- magmaan_core$fit_sample_stats(fit)
chi2 <- magmaan_core$infer_chi2_stat(ss, fit$fmin)
dfm  <- magmaan_core$infer_df_stat(fit$partable, ss)
lfm  <- fitMeasures(lav, c("chisq", "df"))

cat("=== Bollen political-democracy SEM ===\n")
ok(fit$converged,                  "magmaan converged")
ok(fit$npar == length(coef(lav)),  "free-parameter count vs lavaan")
ok(est_match(fit, lav),            "point estimates vs lavaan")
ok(near(chi2, lfm[["chisq"]]),     "chi-square vs lavaan")
ok(dfm == lfm[["df"]],             "degrees of freedom vs lavaan")
cat("SEM example: ok\n")
