## magmaan R bindings — Satorra-Bentler "method 2001" difference spectrum
## (U_D = U0 - U1) for FIML nested tests, plus the scalar SB2001/SB2010
## baselines, on a 2-group configural-vs-metric invariance pair.
##
## Run from the repo root (after `R CMD INSTALL r-package`):
##     Rscript r-package/examples/nested_test_2001.R
##
## The method-2001 U_D estimator differences the two single-model residual
## projectors (Satorra & Bentler 2001, p.510), as opposed to the Satorra-2000
## restriction map (`ud_method = "2000"`). Its top df_H0 - df_H1 eigenvalues of
## (U0 - U1)·Γ feed the same scaled / mixture readouts (and the FMG/pEBA tail
## transforms).
##
## There is no longer an external oracle for this spectrum. semTests 1.0.0
## withdrew it outright ("Nested method 2001 is withdrawn because of its poor
## performance", NEWS.md), so `semTests:::ugamma_nested(., method = "2001")` is
## gone and both survivors (`lav_ugamma_nested_2000`, `ugamma_nested_reference`)
## are method-2000 constructions. Nothing to repoint at, so the former parity
## block is dropped rather than aimed at a differently-defined internal.
##
## The 2001 path is gated transitively instead, per the oracle policy in
## AGENTS.md: U0 and U1 come from the single-model spectrum machinery that IS
## lavaan-gated (`ugamma_eigvals_nt` in the inference goldens,
## `mlr_trace_ugamma{,_h0,_h1}` in the FIML goldens), and the difference plus
## eigen-solve is checked against an independent dense `Eigen::EigenSolver`
## oracle in cpp/tests/unit/satorra2000_test.cpp (including the indefinite case and
## the negative-eigenvalue warning). U_D = U0 - U1 introduces no unvalidated
## quantity of its own.
##
## magmaan keeps `ud_method = "2001"` because it is the documented fallback when
## SB2010 cannot run: method 2000 needs a restriction map between same-parameter
## nested models, while 2001 needs only the two fitted models. See the error
## text in cpp/src/robust/lr_test_satorra.cpp. Upstream's "poor performance" verdict
## is about power, not correctness, so prefer `ud_method = "2000"` when both
## apply.

suppressMessages({ library(magmaanlab); library(lavaan) })
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
s_cfg <- magmaanlab::model_spec(cfg_syntax, group = "school", group_labels = c("A","B"), meanstructure = TRUE)
s_met <- magmaanlab::model_spec(met_syntax, group = "school", group_labels = c("A","B"), meanstructure = TRUE)
cfg <- magmaanlab::magmaan_core$fit_fiml(s_cfg, magmaanlab::df_to_fiml_data(df, s_cfg, group = "school"))
met <- magmaanlab::magmaan_core$fit_fiml(s_met, magmaanlab::df_to_fiml_data(df, s_met, group = "school"))

r2000 <- magmaanlab::nestedTest(cfg, met, method = "satorra.2000", ud_method = "2000")
r2001 <- magmaanlab::nestedTest(cfg, met, method = "satorra.2000", ud_method = "2001")
sb01  <- magmaanlab::nestedTest(cfg, met, method = "satorra.bentler.2001")
sb10  <- magmaanlab::nestedTest(cfg, met, method = "satorra.bentler.2010")

cat("\n=== FIML configural vs metric (2 groups), T_diff =",
    sprintf("%.3f", r2001$T_diff), "df =", r2001$df_diff, "===\n")
cat(sprintf("  Satorra-2000 (restriction map)  p_scaled = %.4f\n", r2000$p_scaled))
cat(sprintf("  Satorra-2001 (U0 - U1 spectrum) p_scaled = %.4f\n", r2001$p_scaled))
cat(sprintf("  scalar SB2001 (trace baseline)  p        = %.4f\n", sb01$p_value))
cat(sprintf("  scalar SB2010 (M10 positivity)  p        = %.4f\n", sb10$p_value))

## ---- self-consistency checks on the 2001 spectrum ---------------------------
## With no external oracle left, pin the properties the construction must have.
ev_mag <- sort(as.numeric(r2001$eigenvalues), decreasing = TRUE)
cat("\n--- 2001 difference spectrum (magmaan) ---\n")
cat("  eigenvalues:", sprintf("%.5f", ev_mag), "\n")
stopifnot(length(ev_mag) == r2001$df_diff)
stopifnot(all(is.finite(ev_mag)))
## T_diff and df_diff are spectrum-independent, so 2000 and 2001 must agree on
## them exactly and differ only in the scaled readout.
stopifnot(identical(r2001$df_diff, r2000$df_diff))
stopifnot(abs(r2001$T_diff - r2000$T_diff) < 1e-10)
cat("  T_diff/df_diff agree with method 2000:", ok(TRUE), "\n")
## The scaled p-value is a monotone transform of a positive spectrum, so it must
## be a usable probability whenever no negative-eigenvalue warning fired.
stopifnot(is.finite(r2001$p_scaled), r2001$p_scaled >= 0, r2001$p_scaled <= 1)
cat("  p_scaled in [0, 1]:", ok(TRUE), "\n")
