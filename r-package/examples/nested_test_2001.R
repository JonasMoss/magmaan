## magmaan R bindings — Satorra-Bentler "method 2001" difference spectrum
## (U_D = U0 - U1) for FIML nested tests, plus the scalar SB2001/SB2010
## baselines, on a 2-group configural-vs-metric invariance pair.
##
## Run from the repo root (after `R CMD INSTALL r-package`):
##     Rscript r-package/examples/nested_test_2001.R
##
## The method-2001 U_D estimator differences the two single-model residual
## projectors (Satorra & Bentler 2001, p.510; semTests::ugamma_nested(., "2001")),
## as opposed to the Satorra-2000 restriction map (`ud_method = "2000"`). Its
## top df_H0 - df_H1 eigenvalues of (U0 - U1)·Γ feed the same scaled / mixture
## readouts (and the FMG/pEBA tail transforms). Validated here against semTests
## on complete data, where lavaan's lavInspect("U")/("gamma") coincide with the
## official UGamma convention. (Under missing data, lavInspect's UGamma is NOT
## the official convention; magmaan matches lavaan's official scaling factor.)

suppressMessages({ library(magmaan); library(lavaan) })
ok <- function(cond) if (isTRUE(cond)) "ok" else "MISMATCH"

set.seed(11); n_g <- 300
pop <- "f =~ 0.8*x1 + 0.75*x2 + 0.7*x3 + 0.65*x4 + 0.6*x5 + 0.7*x6
        x1~~1*x1;x2~~1*x2;x3~~1*x3;x4~~1*x4;x5~~1*x5;x6~~1*x6; f~~1*f"
dA <- lavaan::simulateData(pop, sample.nobs = n_g, meanstructure = TRUE); dA$school <- "A"
dB <- lavaan::simulateData(pop, sample.nobs = n_g, meanstructure = TRUE); dB$school <- "B"
df <- rbind(dA, dB)

cfg_syntax <- "f =~ x1 + x2 + x3 + x4 + x5 + x6"
met_syntax <- "f =~ x1 + L2*x2 + L3*x3 + L4*x4 + L5*x5 + L6*x6"

## ---- magmaan FIML fits + nested tests --------------------------------------
s_cfg <- magmaan::model_spec(cfg_syntax, group = "school", group_labels = c("A","B"), meanstructure = TRUE)
s_met <- magmaan::model_spec(met_syntax, group = "school", group_labels = c("A","B"), meanstructure = TRUE)
cfg <- magmaan::magmaan_core$fit_fiml(s_cfg, magmaan::df_to_fiml_data(df, s_cfg, group = "school"))
met <- magmaan::magmaan_core$fit_fiml(s_met, magmaan::df_to_fiml_data(df, s_met, group = "school"))

r2000 <- magmaan::nestedTest(cfg, met, method = "satorra.2000", ud_method = "2000")
r2001 <- magmaan::nestedTest(cfg, met, method = "satorra.2000", ud_method = "2001")
sb01  <- magmaan::nestedTest(cfg, met, method = "satorra.bentler.2001")
sb10  <- magmaan::nestedTest(cfg, met, method = "satorra.bentler.2010")

cat("\n=== FIML configural vs metric (2 groups), T_diff =",
    sprintf("%.3f", r2001$T_diff), "df =", r2001$df_diff, "===\n")
cat(sprintf("  Satorra-2000 (restriction map)  p_scaled = %.4f\n", r2000$p_scaled))
cat(sprintf("  Satorra-2001 (U0 - U1 spectrum) p_scaled = %.4f\n", r2001$p_scaled))
cat(sprintf("  scalar SB2001 (trace baseline)  p        = %.4f\n", sb01$p_value))
cat(sprintf("  scalar SB2010 (M10 positivity)  p        = %.4f\n", sb10$p_value))

## ---- semTests oracle for the 2001 spectrum (complete data) -----------------
if (requireNamespace("semTests", quietly = TRUE)) {
  lcfg <- lavaan::cfa(cfg_syntax, df, group = "school", missing = "ml", estimator = "ML",
                      test = "yuan.bentler", meanstructure = TRUE, h1.information = "unstructured")
  lmet <- lavaan::cfa(met_syntax, df, group = "school", missing = "ml", estimator = "ML",
                      test = "yuan.bentler", meanstructure = TRUE, h1.information = "unstructured")
  ug <- semTests:::ugamma_nested(lmet, lcfg, method = "2001")[[1]]
  ev_st  <- sort(Re(eigen(ug, only.values = TRUE)$values), decreasing = TRUE)[1:r2001$df_diff]
  ev_mag <- sort(as.numeric(r2001$eigenvalues), decreasing = TRUE)
  cat("\n--- 2001 difference spectrum: magmaan vs semTests ---\n")
  cat("  magmaan :", sprintf("%.5f", ev_mag), "\n")
  cat("  semTests:", sprintf("%.5f", ev_st), "\n")
  cat("  max abs diff:", format(max(abs(ev_mag - ev_st)), digits = 3), "  ",
      ok(max(abs(ev_mag - ev_st)) < 1e-3), "\n")
} else {
  cat("\n(semTests not installed; skipping the 2001-spectrum parity check.)\n")
}
