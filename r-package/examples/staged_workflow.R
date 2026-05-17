## magmaan R bindings — the intended *staged* post-fit workflow, end to end.
##
## Run from the repo root (after `R CMD INSTALL r-package`, or
## `devtools::load_all("r-package")`):
##     Rscript r-package/examples/staged_workflow.R
##
## magmaan deliberately keeps every statistical choice separate. `magmaan()` is
## estimate-only: `se = "none"` and `test = "none"` are the only values it
## accepts, and it never folds inference into the fit. *Every* post-fit quantity
## — standard errors, robust test statistics, fit measures, defined parameters,
## nested-model tests — is its own explicit call, and each call takes only the
## pieces it needs (a partable, an information matrix, a vcov, raw data), never
## a fitted-object God-method.
##
## This example walks that whole staircase on one model — the 3-factor
## Holzinger-Swineford 1939 CFA — each step cross-checked against lavaan:
##
##   1. estimate            magmaan(..., se = "none", test = "none")
##   2. SEs + z tests       infer_information_expected -> infer_vcov -> infer_se
##   3. robust test         the UГ-eigenvalue / Satorra-Bentler scaled chi-square
##   4. fit measures        infer_baseline -> measures_fit (CFI/TLI/RMSEA/SRMR)
##   5. defined parameters  compute_defined() — a `:=` expression over labels
##   6. nested-model test   the likelihood-ratio chi-square difference test
##
## The textual loadings carry labels `L5`, `L6` so step 5 can name them in a
## `:=` row; step 6 then tests the same restriction (L5 == L6) by LR.

suppressMessages({ library(magmaan); library(lavaan) })

mok <- function(a, b, tol = 1e-4)
  if (isTRUE(all.equal(unname(as.numeric(a)), unname(as.numeric(b)),
                       tolerance = tol))) "ok" else "MISMATCH"

hs <- lavaan::HolzingerSwineford1939
df <- as.data.frame(hs[paste0("x", 1:9)])      # 301 x 9 raw data, columns x1..x9
X  <- as.matrix(df)                            # raw matrix for the robust chain

model <- "
  visual  =~ x1 + x2 + x3
  textual =~ x4 + L5*x5 + L6*x6
  speed   =~ x7 + x8 + x9
  ld_gap := L5 - L6
"

lav <- cfa(model, data = df)                   # the oracle, fitted once

## ===========================================================================
## 1. Estimate — magmaan() is estimate-only
## ===========================================================================
fit <- magmaan(model, df, estimator = "ML", se = "none", test = "none")

cat("=== 1. estimate (magmaan, estimate-only) ===\n")
cat(sprintf("  estimator = %s, npar = %d, converged = %s\n",
            fit$estimator, fit$npar, fit$converged))
cat(sprintf("  the fit object carries no SEs / vcov / tests: %s\n",
            !any(c("se", "vcov", "test") %in% names(fit))))
cat(sprintf("  point estimates vs lavaan: %s\n\n", mok(fit$theta, coef(lav))))
stopifnot(!any(c("se", "vcov", "test") %in% names(fit)),
          mok(fit$theta, coef(lav)) == "ok")

## ===========================================================================
## 2. Standard errors + z tests — explicit, from the expected information
## ===========================================================================
info  <- magmaan_core$infer_information_expected(fit)        # n_free x n_free
vcov  <- magmaan_core$infer_vcov_partable(info, fit$partable)# invert (+constraints)
se    <- magmaan_core$infer_se(vcov)                         # sqrt(diag), NaN on Heywood
ztest <- magmaan_core$infer_z_test_fit(fit, se)              # z = est/se, two-sided p

free <- fit$partable[fit$partable$free > 0, ]
free <- free[order(free$free), ]
tab  <- data.frame(param = paste(free$lhs, free$op, free$rhs),
                   est = free$est, se = se, z = ztest$z, p = ztest$pvalue)
cat("=== 2. standard errors + z tests (expected information) ===\n")
print(head(tab, 6), row.names = FALSE, digits = 4)
cat(sprintf("  ... %d free parameters in total\n", nrow(tab)))
cat(sprintf("  SEs vs lavaan: %s\n\n", mok(se, sqrt(diag(lavaan::vcov(lav))))))
stopifnot(mok(se, sqrt(diag(lavaan::vcov(lav)))) == "ok")

## ===========================================================================
## 3. Robust test — Satorra-Bentler scaled chi-square (the UГ-eigenvalue chain)
## ===========================================================================
chi2 <- magmaan_core$infer_chi2_stat(magmaan_core$fit_sample_stats(fit), fit$fmin)
dfm  <- magmaan_core$infer_df_stat(fit$partable, magmaan_core$fit_sample_stats(fit))
uf   <- magmaan_core$infer_build_u_factor_fit(fit)                  # U-factor at theta-hat
Zc   <- magmaan_core$infer_casewise_contributions(fit$partable, X)  # casewise vech rows
ev   <- magmaan_core$infer_ugamma_eigenvalues(
          magmaan_core$infer_reduced_gamma_sample(uf, Zc, fit$nobs))
sb   <- magmaan_core$infer_satorra_bentler(chi2, dfm, ev)

cat("=== 3. robust test — Satorra-Bentler scaled chi-square ===\n")
cat(sprintf("  T_ML = %.3f  (df %d, p %.3g)\n", chi2, dfm,
            magmaan_core$infer_chi2_pvalue(chi2, dfm)))
cat(sprintf("  T_SB = %.3f  (scaling c = %.4f, df %d, p %.3g)\n",
            sb$chi2_scaled, sb$scale_c, sb$df,
            magmaan_core$infer_chi2_pvalue(sb$chi2_scaled, sb$df)))
lav_mlm <- cfa(model, data = df, estimator = "MLM")
cat(sprintf("  lavaan MLM: T_SB = %.3f, c = %.4f  ->  %s\n\n",
            fitMeasures(lav_mlm, "chisq.scaled"),
            fitMeasures(lav_mlm, "chisq.scaling.factor"),
            mok(sb$chi2_scaled, fitMeasures(lav_mlm, "chisq.scaled"), 1e-3)))
stopifnot(mok(sb$chi2_scaled, fitMeasures(lav_mlm, "chisq.scaled"), 1e-3) == "ok")

## ===========================================================================
## 4. Fit measures — incremental + absolute indices
## ===========================================================================
baseline <- magmaan_core$infer_baseline(magmaan_core$fit_sample_stats(fit))
fm <- magmaan_core$measures_fit(fit, chi2, dfm, baseline)

cat("=== 4. fit measures ===\n")
cat(sprintf("  CFI = %.4f   TLI = %.4f\n", fm$cfi, fm$tli))
cat(sprintf("  RMSEA = %.4f  [90%% CI %.4f, %.4f]\n",
            fm$rmsea, fm$rmsea.ci.lower, fm$rmsea.ci.upper))
cat(sprintf("  RMSEA close-fit test: p(RMSEA <= %.2f) = %.4g\n",
            fm$rmsea.close.h0, fm$rmsea.pvalue))
cat(sprintf("  SRMR = %.4f   AIC = %.1f   BIC = %.1f\n", fm$srmr, fm$aic, fm$bic))
lav_fm <- fitMeasures(lav, c("cfi", "tli", "rmsea", "srmr", "aic", "bic"))
cat(sprintf("  vs lavaan (cfi/tli/rmsea/srmr/aic/bic): %s\n\n",
            mok(c(fm$cfi, fm$tli, fm$rmsea, fm$srmr, fm$aic, fm$bic),
                lav_fm, 1e-3)))
stopifnot(mok(c(fm$cfi, fm$tli, fm$rmsea, fm$srmr, fm$aic, fm$bic),
              lav_fm, 1e-3) == "ok")

## ===========================================================================
## 5. Defined parameters — `:=` expressions, delta-method SEs
## ===========================================================================
## `ld_gap := L5 - L6` is the difference between the two free textual loadings;
## compute_defined() evaluates it and propagates a delta-method SE from `vcov`.
defs <- compute_defined(model, fit, vcov)

cat("=== 5. defined parameters (:=) ===\n")
print(defs, row.names = FALSE, digits = 5)
lav_def <- parameterEstimates(lav)
lav_def <- lav_def[lav_def$op == ":=" & lav_def$lhs == "ld_gap", ]
cat(sprintf("  ld_gap vs lavaan: est %s, se %s\n\n",
            mok(defs$est[defs$lhs == "ld_gap"], lav_def$est),
            mok(defs$se[defs$lhs == "ld_gap"], lav_def$se)))
stopifnot(mok(defs$est[defs$lhs == "ld_gap"], lav_def$est) == "ok",
          mok(defs$se[defs$lhs == "ld_gap"], lav_def$se) == "ok")

## ===========================================================================
## 6. Nested-model test — likelihood-ratio chi-square difference
## ===========================================================================
## H0 nests in the model above by tying the two textual loadings (`Lt*x5 +
## Lt*x6`, one equality constraint -> +1 df). The LR test is a pure-R
## subtraction of the two fits' chi-square statistics.
model_h0 <- "
  visual  =~ x1 + x2 + x3
  textual =~ x4 + Lt*x5 + Lt*x6
  speed   =~ x7 + x8 + x9
"
fit_h0 <- magmaan(model_h0, df, estimator = "ML", se = "none", test = "none")
chi2_h0 <- magmaan_core$infer_chi2_stat(
  magmaan_core$fit_sample_stats(fit_h0), fit_h0$fmin)
df_h0   <- magmaan_core$infer_df_stat(
  fit_h0$partable, magmaan_core$fit_sample_stats(fit_h0))

lr_stat <- chi2_h0 - chi2
lr_df   <- df_h0 - dfm
cat("=== 6. nested-model test — likelihood-ratio chi-square difference ===\n")
cat(sprintf("  H1 (loadings free):  chi2 = %.3f, df = %d\n", chi2, dfm))
cat(sprintf("  H0 (L5 == L6 tied):  chi2 = %.3f, df = %d\n", chi2_h0, df_h0))
cat(sprintf("  LR test: delta-chi2 = %.4f, delta-df = %d, p = %.4g\n",
            lr_stat, lr_df, magmaan_core$infer_chi2_pvalue(lr_stat, lr_df)))
lav_h0 <- cfa(model_h0, data = df)
lr_lav <- lavTestLRT(lav, lav_h0)
cat(sprintf("  vs lavaan lavTestLRT: delta-chi2 %s, delta-df %s\n",
            mok(lr_stat, lr_lav[2, "Chisq diff"]),
            mok(lr_df, lr_lav[2, "Df diff"], 0)))
cat("  (for the robust, scaled nested test under non-normality use\n",
    "  nestedTest() — see nested_test_satorra2000.R.)\n\n", sep = "")
stopifnot(mok(lr_stat, lr_lav[2, "Chisq diff"]) == "ok",
          mok(lr_df, lr_lav[2, "Df diff"], 0) == "ok")

cat("staged post-fit workflow: ok\n")
