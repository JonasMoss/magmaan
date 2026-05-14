## magmaan R bindings — Satorra (2000) scaled nested-model χ² test on
## Holzinger-Swineford × school (configural vs metric invariance), cross-checked
## against lavaan's `lavTestLRT(method = "satorra.2000")`.
##
## Run from the repo root (after `R CMD INSTALL r-package`):
##     Rscript r-package/examples/nested_test_satorra2000.R

suppressMessages({ library(magmaan); library(lavaan) })

ok  <- function(cond) if (isTRUE(cond)) "ok" else "MISMATCH"
mok <- function(a, b, tol = 1e-2)
  if (isTRUE(all.equal(unname(as.numeric(a)),
                       unname(as.numeric(b)), tolerance = tol))) "ok" else "MISMATCH"

hs    <- HolzingerSwineford1939
vars  <- paste0("x", 1:9)
df_hs <- as.data.frame(hs[c("school", vars)])
Xg    <- lapply(split(hs[vars], hs$school), as.matrix)
ssg   <- data_sample_stats_from_raw(Xg)

m_cfg <- "visual  =~ x1 + x2 + x3
          textual =~ x4 + x5 + x6
          speed   =~ x7 + x8 + x9"
m_met <- "visual  =~ x1 + L1*x2 + L2*x3
          textual =~ x4 + L3*x5 + L4*x6
          speed   =~ x7 + L5*x8 + L6*x9"

## ---- fit configural (H1, less restricted) and metric (H0) -----------------

pt_cfg <- lavaan_lavaanify(m_cfg, n_groups = 2L, group_var = "school")
pt_met <- lavaan_lavaanify(m_met, n_groups = 2L, group_var = "school")
fit_cfg <- fit_fit(pt_cfg, ssg)     # H1
fit_met <- fit_fit(pt_met, ssg)     # H0  (additional cross-group `==` rows)

## ---- run the Satorra-2000 nested test --------------------------------------

## Multi-group: pass per-group raw data as a list (same block order as
## fit_cfg$S — which is the order `data_sample_stats_from_raw()` consumed
## above).
res <- nestedTest(fit_H1 = fit_cfg, fit_H0 = fit_met, data = Xg)
cat("\n=== magmaan: nestedTest(fit_cfg, fit_met) ===\n")
print(res)

## ---- cross-check vs lavaan -------------------------------------------------

lav_cfg <- lavaan::cfa(m_cfg, data = df_hs, group = "school",
                       estimator = "MLM")
lav_met <- lavaan::cfa(m_cfg, data = df_hs, group = "school",
                       group.equal = "loadings", estimator = "MLM")
lav_lr  <- lavaan::lavTestLRT(lav_cfg, lav_met, method = "satorra.2000")

cat("\n=== lavaan::lavTestLRT(., method = 'satorra.2000') ===\n")
print(lav_lr)

## The "scaled" row in lavaan's table is the SB scaled test on the difference;
## extract its statistic and df.  Layout: rows 1 = H1 (NA Δχ²), 2 = H0 (Δχ², Δdf, p).
lav_T   <- as.numeric(lav_lr[2, "Chisq diff"])
lav_df  <- as.numeric(lav_lr[2, "Df diff"])
lav_p   <- as.numeric(lav_lr[2, "Pr(>Chisq)"])

cat("\n--- cross-check: magmaan vs lavaan -------------------------------\n")
## NB: lavaan's `method = "satorra.2000"` and our `compute_satorra2000` can
## differ in the n vs (n−1) divisor on Γ̂, so we cross-check to a relative
## tolerance of 1% on the scaled stat and 5% on the p-value.  The eigenvalue
## spectrum itself should be within those bounds too.
cat(sprintf("  Δχ² (scaled):    magmaan = %.4f   lavaan = %.4f   %s\n",
            res$T_scaled, lav_T, mok(res$T_scaled, lav_T, tol = 1e-2)))
cat(sprintf("  Δdf:             magmaan = %d       lavaan = %d        %s\n",
            res$df_diff, as.integer(lav_df),
            ok(identical(as.integer(res$df_diff), as.integer(lav_df)))))
cat(sprintf("  p (scaled):      magmaan = %.4g   lavaan = %.4g   %s\n",
            res$p_scaled, lav_p, mok(res$p_scaled, lav_p, tol = 5e-2)))
cat(sprintf("  ĉ (scale factor): %.6f\n", res$scale_c))
cat(sprintf("  d̂₀ (adj. df):     %.6f\n", res$adjust_d0))
cat(sprintf("  Imhof mixture p:  %.6g\n", res$p_mixture))
cat("\nEigenvalues:\n")
print(res$eigenvalues)
