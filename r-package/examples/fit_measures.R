## magmaan R bindings — fit measures: incremental indices, RMSEA with its
## close-fit hypothesis test, SRMR, and the likelihood-based information
## criteria, cross-checked against lavaan.
##
## Run from the repo root (after `R CMD INSTALL r-package`, or
## `devtools::load_all("r-package")`):
##     Rscript r-package/examples/fit_measures.R
##
## Fit measures are post-fit, not part of estimation. The recipe is two explicit
## calls:
##   infer_baseline(sample_stats)         -> the independence-model chi-square
##   measures_fit(fit, chi2, df, baseline)-> CFI/TLI, RMSEA (+ 90% CI and the
##                                           close-fit p-values), SRMR, AIC/BIC
##
## To show the measures actually discriminating, this fits the same Holzinger-
## Swineford 1939 data twice: the well-specified 3-factor CFA, and a misspecified
## single-factor CFA. Watch CFI/TLI fall and RMSEA rise between them.

suppressMessages({ library(magmaan); library(lavaan) })

mok <- function(a, b, tol = 1e-3)
  if (isTRUE(all.equal(unname(as.numeric(a)), unname(as.numeric(b)),
                       tolerance = tol))) "ok" else "MISMATCH"

hs <- lavaan::HolzingerSwineford1939
df <- as.data.frame(hs[paste0("x", 1:9)])

model_3f <- "visual  =~ x1 + x2 + x3
             textual =~ x4 + x5 + x6
             speed   =~ x7 + x8 + x9"
model_1f <- "g =~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9"

## measures_fit() needs the model chi-square, its df, and the baseline fit.
## Each is one explicit primitive away from the estimate-only fit.
measures_of <- function(model) {
  fit  <- magmaan(model, df, estimator = "ML", se = "none", test = "none")
  ss   <- magmaan_core$fit_sample_stats(fit)
  chi2 <- magmaan_core$infer_chi2_stat(ss, fit$fmin)
  dfm  <- magmaan_core$infer_df_stat(fit$partable, ss)
  base <- magmaan_core$infer_baseline(ss)              # independence-model chi2
  fm   <- magmaan_core$measures_fit(fit, chi2, dfm, base)
  c(list(chi2 = chi2, df = dfm, baseline.chi2 = base$chi2), fm)
}

m3 <- measures_of(model_3f)
m1 <- measures_of(model_1f)

show <- function(field, fmt = "%.4f")
  sprintf(paste0("  %-26s 3-factor: ", fmt, "   1-factor: ", fmt, "\n"),
          field, m3[[field]], m1[[field]])

cat("=================================================================\n")
cat(" Fit measures — 3-factor CFA (well-specified) vs 1-factor (not)\n")
cat("=================================================================\n\n")

cat("--- model chi-square ---\n")
cat(show("chi2"), show("df", "%d"), sep = "")
cat(sprintf("  baseline (independence) chi2 = %.1f / %.1f\n\n",
            m3$baseline.chi2, m1$baseline.chi2))

cat("--- incremental fit indices (1.0 = baseline-to-saturated span) ---\n")
cat(show("cfi"), show("tli"), sep = "")
cat("\n")

cat("--- RMSEA and its hypothesis tests ---\n")
cat(show("rmsea"))
cat(sprintf("  %-26s 3-factor: [%.4f, %.4f]   1-factor: [%.4f, %.4f]\n",
            "rmsea 90% CI",
            m3$rmsea.ci.lower, m3$rmsea.ci.upper,
            m1$rmsea.ci.lower, m1$rmsea.ci.upper))
## rmsea.pvalue tests close fit, H0: RMSEA <= rmsea.close.h0 (0.05): a *small*
## p rejects close fit. rmsea.notclose.pvalue tests H0: RMSEA >= 0.08: a small
## p there is the favourable outcome.
cat(sprintf("  %-26s 3-factor: %.4g   1-factor: %.4g\n",
            sprintf("p(RMSEA <= %.2f)", m3$rmsea.close.h0),
            m3$rmsea.pvalue, m1$rmsea.pvalue))
cat(sprintf("  %-26s 3-factor: %.4g   1-factor: %.4g\n\n",
            sprintf("p(RMSEA >= %.2f)", m3$rmsea.notclose.h0),
            m3$rmsea.notclose.pvalue, m1$rmsea.notclose.pvalue))

cat("--- absolute residual + information criteria ---\n")
cat(show("srmr"), show("aic", "%.1f"), show("bic", "%.1f"), sep = "")
cat("\n")

## ---- cross-check both models against lavaan::fitMeasures() ----------------
keys <- c("cfi", "tli", "rmsea", "rmsea.ci.lower", "rmsea.ci.upper",
          "rmsea.pvalue", "srmr", "aic", "bic")
lav3 <- fitMeasures(cfa(model_3f, data = df), keys)
lav1 <- fitMeasures(cfa(model_1f, data = df), keys)
ck3 <- mok(unlist(m3[keys]), lav3)
ck1 <- mok(unlist(m1[keys]), lav1)
cat("--- cross-check vs lavaan::fitMeasures() ---\n")
cat(sprintf("  3-factor: %s   |   1-factor: %s\n\n", ck3, ck1))
stopifnot(ck3 == "ok", ck1 == "ok")

## The misspecified model should fail close fit decisively while the 3-factor
## model fits far better — a sanity check that the measures discriminate.
stopifnot(m1$cfi < m3$cfi, m1$rmsea > m3$rmsea, m1$rmsea.pvalue < 1e-6)

cat("fit measures workflow: ok\n")
