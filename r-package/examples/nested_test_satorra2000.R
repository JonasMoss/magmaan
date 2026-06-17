## magmaan R bindings — robust nested-model likelihood-ratio test, via
## `robust_nested_lrt()`, on Holzinger-Swineford × school (configural vs metric
## invariance).
##
## Run from the repo root (after `R CMD INSTALL r-package`):
##     Rscript r-package/examples/nested_test_satorra2000.R
##
## `robust_nested_lrt()` reports five p-values for H1 ⊃ H0: the naïve χ²(m), the
## Satorra-Bentler scaled correction, the mean-and-variance adjustment, the
## scaled-and-shifted correction, and the exact Imhof mixture tail.
##
## NOTE ON LAVAAN PARITY. magmaan's scaled difference statistic is the exact
## parameter-nesting Satorra-2000 test. The matching lavaan oracle is
## `lavTestLRT(..., method = "satorra.2000", A.method = "exact",
## scaled.shifted = FALSE)`. lavaan's defaults use a covariance-nesting
## moment-Jacobian construction for the restriction matrix and the
## scaled-shifted statistic; both can differ from this mean-scaled
## parameter-nesting check. Use `A.method = "delta"` explicitly to request the
## lavaan-style moment-Jacobian construction. See docs/validation/satorra2000_parity.md.

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
res <- magmaan::robust_nested_lrt(fit_H1 = fit_cfg, fit_H0 = fit_met,
                                  data = Xg, A.method = "exact")
cat("\n=== magmaan::robust_nested_lrt(fit_cfg, fit_met, A.method = 'exact') ===\n")
print(res)

## ---- lavaan, for reference -------------------------------------------------

lav_cfg <- lavaan::cfa(m_cfg, data = df_hs, group = "school",
                       estimator = "MLM")
lav_met <- lavaan::cfa(m_cfg, data = df_hs, group = "school",
                       group.equal = "loadings", estimator = "MLM")
lav_lr  <- lavaan::lavTestLRT(lav_cfg, lav_met, method = "satorra.2000",
                              A.method = "exact", scaled.shifted = FALSE)
lav_lr_delta <- lavaan::lavTestLRT(lav_cfg, lav_met, method = "satorra.2000",
                                   A.method = "delta")

cat("\n=== lavaan::lavTestLRT(., method = 'satorra.2000', A.method = 'exact', scaled.shifted = FALSE) ===\n")
print(lav_lr)

## Layout: rows 1 = H1 (NA Δχ²), 2 = H0 (Δχ², Δdf, p).
lav_df       <- as.numeric(lav_lr[2, "Df diff"])
lav_unscaled <- as.numeric(fitMeasures(lav_met, "chisq") -
                           fitMeasures(lav_cfg, "chisq"))
lav_scaled   <- as.numeric(lav_lr[2, "Chisq diff"])

## ---- checks: exact parameter-nesting parity --------------------------------
cat("\n--- magmaan vs lavaan -------------------------------------------\n")
cat(sprintf("  Δdf:              magmaan = %d        lavaan = %d         %s\n",
            res$df_diff, as.integer(lav_df),
            ok(identical(as.integer(res$df_diff), as.integer(lav_df)))))
cat(sprintf("  Δχ² (unscaled):   magmaan = %.4f   lavaan = %.4f   %s\n",
            res$T_diff, lav_unscaled,
            ok(isTRUE(all.equal(res$T_diff, lav_unscaled, tolerance = 1e-3)))))
cat(sprintf("  Δχ² (scaled):     magmaan = %.4f   lavaan = %.4f   %s\n",
            res$T_scaled, lav_scaled,
            ok(isTRUE(all.equal(res$T_scaled, lav_scaled, tolerance = 1e-3)))))
cat(sprintf("  Δχ² (lavaan delta A.method, reference only): %.4f\n",
            as.numeric(lav_lr_delta[2, "Chisq diff"])))
cat(sprintf("  ĉ (magmaan scale factor): %.6f\n", res$scale_c))
cat(sprintf("  d̂₀ (adj. df):             %.6f\n", res$adjust_d0))
cat(sprintf("  Imhof mixture p:          %.6g\n", res$p_mixture))
cat("\nEigenvalues:\n")
print(res$eigenvalues)

stopifnot(identical(as.integer(res$df_diff), as.integer(lav_df)))
stopifnot(isTRUE(all.equal(res$T_diff, lav_unscaled, tolerance = 1e-3)))
stopifnot(isTRUE(all.equal(res$T_scaled, lav_scaled, tolerance = 1e-3)))

## ===========================================================================
## MEAN STRUCTURE — the same configural vs metric pair, fit with
## `meanstructure = TRUE`.  The free intercepts populate the augmented
## [μ; vech(Σ)] moment vector; before mean-structure support landed this raised
## "pooled expected info P is rank-deficient".  The saturated means do not
## change the covariance fit, so the scaled difference must still match lavaan's
## mean-structured satorra.2000.  This exercises the augmented P / V / casewise
## meat end-to-end through lavaanify → fit → robust_nested_lrt.
## ===========================================================================

pt_cfg_m <- magmaan_core$lavaan_lavaanify(m_cfg, n_groups = 2L,
                                          group_var = "school",
                                          meanstructure = TRUE)
pt_met_m <- magmaan_core$lavaan_lavaanify(m_met, n_groups = 2L,
                                          group_var = "school",
                                          meanstructure = TRUE)
fit_cfg_m <- magmaan_core$fit_fit(pt_cfg_m, ssg)     # H1
fit_met_m <- magmaan_core$fit_fit(pt_met_m, ssg)     # H0

res_m <- magmaan::robust_nested_lrt(fit_H1 = fit_cfg_m, fit_H0 = fit_met_m,
                                    data = Xg, A.method = "exact")
cat("\n=== meanstructure=TRUE: robust_nested_lrt(cfg, met, A.method='exact') ===\n")
print(res_m)

lav_cfg_m <- lavaan::cfa(m_cfg, data = df_hs, group = "school",
                         estimator = "MLM", meanstructure = TRUE)
lav_met_m <- lavaan::cfa(m_cfg, data = df_hs, group = "school",
                         group.equal = "loadings", estimator = "MLM",
                         meanstructure = TRUE)
lav_lr_m  <- lavaan::lavTestLRT(lav_cfg_m, lav_met_m, method = "satorra.2000",
                                A.method = "exact", scaled.shifted = FALSE)

lav_df_m       <- as.numeric(lav_lr_m[2, "Df diff"])
lav_unscaled_m <- as.numeric(fitMeasures(lav_met_m, "chisq") -
                             fitMeasures(lav_cfg_m, "chisq"))
lav_scaled_m   <- as.numeric(lav_lr_m[2, "Chisq diff"])

cat("\n--- meanstructure: magmaan vs lavaan ----------------------------\n")
cat(sprintf("  Δdf:              magmaan = %d        lavaan = %d         %s\n",
            res_m$df_diff, as.integer(lav_df_m),
            ok(identical(as.integer(res_m$df_diff), as.integer(lav_df_m)))))
cat(sprintf("  Δχ² (unscaled):   magmaan = %.4f   lavaan = %.4f   %s\n",
            res_m$T_diff, lav_unscaled_m,
            ok(isTRUE(all.equal(res_m$T_diff, lav_unscaled_m, tolerance = 1e-3)))))
cat(sprintf("  Δχ² (scaled):     magmaan = %.4f   lavaan = %.4f   %s\n",
            res_m$T_scaled, lav_scaled_m,
            ok(isTRUE(all.equal(res_m$T_scaled, lav_scaled_m, tolerance = 1e-3)))))

stopifnot(identical(as.integer(res_m$df_diff), as.integer(lav_df_m)))
stopifnot(isTRUE(all.equal(res_m$T_diff, lav_unscaled_m, tolerance = 1e-3)))
stopifnot(isTRUE(all.equal(res_m$T_scaled, lav_scaled_m, tolerance = 1e-3)))

## ===========================================================================
## MEAN-PARAMETER RESTRICTION — intercept invariance.  Now the restriction
## itself lands on the mean block: H1 frees the 9 indicator intercepts per
## group; H0 ties them across groups (latent means fixed at 0 in both groups,
## so the pair is a pure parameter-nesting the exact restriction map accepts).
## A_α loads the μ-rows, so this exercises the augmented mean machinery in the
## restriction, not just the moment space.  Matches lavaan's satorra.2000.
## ===========================================================================

lmean      <- "visual ~ c(0,0)*1\n  textual ~ c(0,0)*1\n  speed ~ c(0,0)*1"
ints_free  <- paste(sprintf("x%d ~ 1", 1:9), collapse = "\n  ")
ints_tied  <- paste(sprintf("x%d ~ c(t%d,t%d)*1", 1:9, 1:9, 1:9), collapse = "\n  ")
m_int_free <- paste0(m_met, "\n  ", ints_free, "\n  ", lmean)   # H1
m_int_tied <- paste0(m_met, "\n  ", ints_tied, "\n  ", lmean)   # H0

pt_if  <- magmaan_core$lavaan_lavaanify(m_int_free, n_groups = 2L,
                                        group_var = "school", meanstructure = TRUE)
pt_it  <- magmaan_core$lavaan_lavaanify(m_int_tied, n_groups = 2L,
                                        group_var = "school", meanstructure = TRUE)
fit_if <- magmaan_core$fit_fit(pt_if, ssg)   # H1
fit_it <- magmaan_core$fit_fit(pt_it, ssg)   # H0

res_i <- magmaan::robust_nested_lrt(fit_H1 = fit_if, fit_H0 = fit_it,
                                    data = Xg, A.method = "exact")
cat("\n=== intercept invariance: robust_nested_lrt(free, tied, A.method='exact') ===\n")
print(res_i)

lm_obs <- "visual =~ x1 + x2 + x3\n  textual =~ x4 + x5 + x6\n  speed =~ x7 + x8 + x9"
lav_if <- lavaan::cfa(lm_obs, data = df_hs, group = "school",
                      group.equal = "loadings", estimator = "MLM",
                      meanstructure = TRUE)
lav_it <- lavaan::cfa(paste0(lm_obs, "\n  ", lmean), data = df_hs, group = "school",
                      group.equal = c("loadings", "intercepts"),
                      estimator = "MLM", meanstructure = TRUE)
lav_lr_i <- lavaan::lavTestLRT(lav_if, lav_it, method = "satorra.2000",
                               A.method = "exact", scaled.shifted = FALSE)

lav_df_i       <- as.numeric(lav_lr_i[2, "Df diff"])
lav_unscaled_i <- as.numeric(fitMeasures(lav_it, "chisq") - fitMeasures(lav_if, "chisq"))
lav_scaled_i   <- as.numeric(lav_lr_i[2, "Chisq diff"])

cat("\n--- intercept invariance: magmaan vs lavaan ---------------------\n")
cat(sprintf("  Δdf:              magmaan = %d        lavaan = %d         %s\n",
            res_i$df_diff, as.integer(lav_df_i),
            ok(identical(as.integer(res_i$df_diff), as.integer(lav_df_i)))))
cat(sprintf("  Δχ² (unscaled):   magmaan = %.4f  lavaan = %.4f  %s\n",
            res_i$T_diff, lav_unscaled_i,
            ok(isTRUE(all.equal(res_i$T_diff, lav_unscaled_i, tolerance = 1e-3)))))
cat(sprintf("  Δχ² (scaled):     magmaan = %.4f  lavaan = %.4f  %s\n",
            res_i$T_scaled, lav_scaled_i,
            ok(isTRUE(all.equal(res_i$T_scaled, lav_scaled_i, tolerance = 1e-3)))))

stopifnot(identical(as.integer(res_i$df_diff), as.integer(lav_df_i)))
stopifnot(isTRUE(all.equal(res_i$T_diff, lav_unscaled_i, tolerance = 1e-3)))
stopifnot(isTRUE(all.equal(res_i$T_scaled, lav_scaled_i, tolerance = 1e-3)))

cat("\nrobust_nested_lrt() restriction-map workflow: ok\n")
