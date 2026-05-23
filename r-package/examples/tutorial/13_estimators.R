## lavaan tutorial — Estimators and more   https://lavaan.ugent.be/tutorial/est.html
##
## magmaan exposes ML / ULS / GLS / WLS / DWLS as `estimator =`. Robust
## (Satorra-Bentler-style) test statistics are an *explicit post-fit* step
## rather than a fit-time `estimator = "MLM"` string. This fits the 3-factor
## CFA with ML, ULS and GLS, and reproduces the scaled chi-square that lavaan
## reports under estimator = "MLM".

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
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"
hs  <- HolzingerSwineford1939
opt <- list(max_iter = 3000, ftol = 1e-12, gtol = 1e-9)

cat("=== estimators ===\n")
for (est in c("ML", "ULS", "GLS")) {
  fit <- magmaan(model, hs, estimator = est, control = opt,
                 se = "none", test = "none")
  lav <- cfa(model, data = hs, estimator = est)
  ok(fit$converged && est_match(fit, lav, tol = 1e-2),
     sprintf("estimator = \"%s\" point estimates vs lavaan", est))
}

## robust scaled chi-square: estimate with ML, then the explicit
## Satorra-Bentler post-fit chain (U-factor -> reduced Gamma -> eigenvalues).
fit  <- magmaan(model, hs, estimator = "ML", se = "none", test = "none")
X    <- as.matrix(hs[paste0("x", 1:9)])
ss   <- magmaan_core$fit_sample_stats(fit)
chi2 <- magmaan_core$infer_chi2_stat(ss, fit$fmin)
dfm  <- magmaan_core$infer_df_stat(fit$partable, ss)
uf   <- magmaan_core$infer_build_u_factor_fit(fit)
Zc   <- magmaan_core$infer_casewise_contributions(fit$partable, X)
ev   <- magmaan_core$infer_ugamma_eigenvalues(
          magmaan_core$infer_reduced_gamma_sample(uf, Zc, fit$nobs))
sb   <- magmaan_core$infer_satorra_bentler(chi2, dfm, ev)
lav_mlm <- cfa(model, data = hs, estimator = "MLM")

ok(near(sb$chi2_scaled, fitMeasures(lav_mlm, "chisq.scaled"), 1e-3),
   "Satorra-Bentler scaled chi-square vs lavaan MLM")
cat(sprintf("  T_ML = %.3f   T_SB = %.3f (scaling %.4f)\n",
            chi2, sb$chi2_scaled, sb$scale_c))
cat("estimators: ok\n")
