## lavaan tutorial — Extracting information   https://lavaan.ugent.be/tutorial/inspect.html
##
## lavaan offers coef(), vcov(), fitMeasures(), fitted(), parameterEstimates(),
## etc. magmaan exposes the same quantities through the explicit `magmaan_core`
## primitives. This pulls each one and cross-checks it against lavaan; the few
## accessors still missing from the R package are noted at the end.

suppressMessages(requireNamespace("lavaan"))
core <- magmaan::magmaan_core

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
utils::data("HolzingerSwineford1939", package = "lavaan")
hs <- HolzingerSwineford1939
fit <- magmaan::magmaan(model, hs, estimator = "ML", se = "none", test = "none")
lav <- lavaan::cfa(model, data = hs)

## coef()  — the free parameter estimates
free <- fit$partable[fit$partable$free > 0, ]
free <- free[order(free$free), ]
## vcov() / SE  — explicit information -> vcov -> se chain
info <- core$inference_information_expected(fit)
vc   <- core$inference_vcov_partable(info, fit$partable)
se   <- core$inference_se(vc)
## fitMeasures()
ss   <- core$fit_sample_stats(fit)
chi2 <- core$inference_chi2_stat(ss, fit$fmin)
dfm  <- core$inference_df_stat(fit$partable, ss)
fm   <- core$measures_fit(fit, chi2, dfm, core$measures_baseline(ss))
## fitted()  — the model-implied covariance matrix
implied <- core$model_implied(fit)
res <- core$measures_residuals(fit)
std <- core$measures_standardize_all(fit, vc)

cat("=== extracting information ===\n")
ok(near(free$est, lavaan::coef(lav)),          "coef() — point estimates vs lavaan")
ok(se_match(fit, se, lav),                     "vcov()/SE vs lavaan")
ok(near(c(fm$aic, fm$bic), lavaan::fitMeasures(lav, c("aic","bic"))),
                                               "AIC / BIC vs lavaan")
ok(near(fm$cfi, lavaan::fitMeasures(lav, "cfi")),
                                               "fitMeasures() CFI vs lavaan")
impl_cov <- implied$sigma[[1]]      # model_implied() returns list(sigma=, mu=)
ok(near(as.numeric(impl_cov), as.numeric(lavaan::fitted(lav)$cov)),
                                               "fitted() model-implied covariance vs lavaan")
ok(is.matrix(res$cov[[1]]),                    "residuals() covariance residuals")
ok(length(std$theta) == length(fit$theta),      "standardizedSolution() primitive")

cat("  not yet implemented: residual z-statistics and bootstrap extraction workflows",
    "(see docs/lavaan_tutorial_parity.md section 16).\n")
cat("extracting information: ok\n")
