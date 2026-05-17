## magmaan R bindings — Satorra (2000) scaled nested-model χ² test, via
## `nestedTest()`, on Holzinger-Swineford × school (configural vs metric
## invariance).
##
## Run from the repo root (after `R CMD INSTALL r-package`):
##     Rscript r-package/examples/nested_test_satorra2000.R
##
## `nestedTest()` reports four p-values for H1 ⊃ H0: the naïve χ²(m), the
## Satorra-Bentler scaled correction, the mean-and-variance adjustment, and the
## exact Imhof mixture tail.
##
## NOTE ON LAVAAN PARITY. magmaan's scaled difference statistic is the
## *H1-anchored reduced-form* Satorra-2000 test (everything evaluated at the H1
## fit). lavaan's `lavTestLRT(method = "satorra.2000")` computes a different
## variant; the two agree on the unscaled difference and df exactly, but the
## *scaled* statistic diverges for strongly-binding restrictions. On this
## example metric invariance nearly holds on the HS data, so the restriction
## barely binds and the two happen to be close — but that is not a parity
## guarantee. See docs/satorra2000_parity.md. This example therefore checks
## only the unscaled difference and df against lavaan, and shows lavaan's scaled
## statistic for reference, not as a parity check.

suppressMessages({ library(magmaan); library(lavaan) })

ok <- function(cond) if (isTRUE(cond)) "ok" else "MISMATCH"

hs    <- HolzingerSwineford1939
vars  <- paste0("x", 1:9)
df_hs <- as.data.frame(hs[c("school", vars)])
Xg    <- lapply(split(hs[vars], hs$school), as.matrix)
ssg   <- magmaan_core$data_sample_stats_from_raw(Xg)

m_cfg <- "visual  =~ x1 + x2 + x3
          textual =~ x4 + x5 + x6
          speed   =~ x7 + x8 + x9"
m_met <- "visual  =~ x1 + L1*x2 + L2*x3
          textual =~ x4 + L3*x5 + L4*x6
          speed   =~ x7 + L5*x8 + L6*x9"

## ---- fit configural (H1, less restricted) and metric (H0) -----------------

pt_cfg <- magmaan_core$lavaan_lavaanify(m_cfg, n_groups = 2L, group_var = "school")
pt_met <- magmaan_core$lavaan_lavaanify(m_met, n_groups = 2L, group_var = "school")
fit_cfg <- magmaan_core$fit_fit(pt_cfg, ssg)     # H1
fit_met <- magmaan_core$fit_fit(pt_met, ssg)     # H0  (additional cross-group `==` rows)

## ---- run the Satorra-2000 nested test --------------------------------------

## Multi-group: pass per-group raw data as a list (same block order as
## fit_cfg$S — which is the order `magmaan_core$data_sample_stats_from_raw()` consumed
## above).
res <- nestedTest(fit_H1 = fit_cfg, fit_H0 = fit_met, data = Xg)
cat("\n=== magmaan: nestedTest(fit_cfg, fit_met) ===\n")
print(res)

## ---- lavaan, for reference -------------------------------------------------

lav_cfg <- lavaan::cfa(m_cfg, data = df_hs, group = "school",
                       estimator = "MLM")
lav_met <- lavaan::cfa(m_cfg, data = df_hs, group = "school",
                       group.equal = "loadings", estimator = "MLM")
lav_lr  <- lavaan::lavTestLRT(lav_cfg, lav_met, method = "satorra.2000")

cat("\n=== lavaan::lavTestLRT(., method = 'satorra.2000') ===\n")
print(lav_lr)

## Layout: rows 1 = H1 (NA Δχ²), 2 = H0 (Δχ², Δdf, p).
lav_df       <- as.numeric(lav_lr[2, "Df diff"])
lav_unscaled <- as.numeric(fitMeasures(lav_met, "chisq") -
                           fitMeasures(lav_cfg, "chisq"))
lav_scaled   <- as.numeric(lav_lr[2, "Chisq diff"])

## ---- checks: unscaled difference + df only ---------------------------------
## These are exact across implementations. The *scaled* statistic is variant-
## dependent (see the header note / docs/satorra2000_parity.md) and is shown
## for reference only.
cat("\n--- magmaan vs lavaan -------------------------------------------\n")
cat(sprintf("  Δdf:              magmaan = %d        lavaan = %d         %s\n",
            res$df_diff, as.integer(lav_df),
            ok(identical(as.integer(res$df_diff), as.integer(lav_df)))))
cat(sprintf("  Δχ² (unscaled):   magmaan = %.4f   lavaan = %.4f   %s\n",
            res$T_diff, lav_unscaled,
            ok(isTRUE(all.equal(res$T_diff, lav_unscaled, tolerance = 1e-3)))))
cat(sprintf("  Δχ² (scaled):     magmaan = %.4f   lavaan = %.4f   (different variants — reference only)\n",
            res$T_scaled, lav_scaled))
cat(sprintf("  ĉ (magmaan scale factor): %.6f\n", res$scale_c))
cat(sprintf("  d̂₀ (adj. df):             %.6f\n", res$adjust_d0))
cat(sprintf("  Imhof mixture p:          %.6g\n", res$p_mixture))
cat("\nEigenvalues:\n")
print(res$eigenvalues)

stopifnot(identical(as.integer(res$df_diff), as.integer(lav_df)))
stopifnot(isTRUE(all.equal(res$T_diff, lav_unscaled, tolerance = 1e-3)))

cat("\nnestedTest() Satorra-2000 workflow: ok\n")
