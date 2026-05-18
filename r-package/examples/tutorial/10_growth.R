## lavaan tutorial — Growth curves   https://lavaan.ugent.be/tutorial/growth.html
##
## A linear latent-growth model: a random intercept `i` and slope `s` over
## four time points. lavaan fits it with growth(); magmaan applies the same
## growth defaults via model_spec(model_type = "growth") — observed-variable
## intercepts fixed at 0, latent growth-factor means freely estimated.

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
  i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4
  s =~ 0*t1 + 1*t2 + 2*t3 + 3*t4
"
spec <- model_spec(model, model_type = "growth")
fit  <- magmaan(spec, Demo.growth, estimator = "ML", se = "none", test = "none")
lav  <- growth(model, data = Demo.growth)

ss   <- magmaan_core$fit_sample_stats(fit)
chi2 <- magmaan_core$infer_chi2_stat(ss, fit$fmin)
dfm  <- magmaan_core$infer_df_stat(fit$partable, ss)
lfm  <- fitMeasures(lav, c("chisq", "df"))

cat("=== linear latent growth curve ===\n")
ok(fit$converged,                  "magmaan converged")
ok(fit$npar == length(coef(lav)),  "free-parameter count vs lavaan")
ok(est_match(fit, lav),            "estimates (growth-factor means/vars) vs lavaan")
ok(near(chi2, lfm[["chisq"]]),     "chi-square vs lavaan")
ok(dfm == lfm[["df"]],             "degrees of freedom vs lavaan")
cat("growth curves: ok\n")
