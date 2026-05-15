## magmaan R bindings — measurement invariance on Holzinger-Swineford × school
## (configural → metric), the invariance LR test, and the robust (Satorra-
## Bentler) χ² for the multi-group models — all cross-checked against lavaan.
##
## Run from the repo root (after `R CMD INSTALL r-package`, or
## `devtools::load_all("r-package")`):
##     Rscript r-package/examples/holzinger_invariance.R
##
## Multi-group is one uniform path: `lavaan_lavaanify(m, n_groups = 2L,
## group_var = "school")` builds the per-group partable (shared loading labels
## auto-synthesize the cross-group `==` rows ≙ `group.equal = "loadings"`), and
## `fit_fit(pt, data_sample_stats_from_raw(Xg))` takes the {S, nobs, mean}
## bundle (Xg a list of per-group raw matrices) straight in. See
## holzinger_2group_satorra_bentler.R for more on the SB pipeline.

suppressMessages({ library(magmaan); library(lavaan) })

ok <- function(cond) if (isTRUE(cond)) "ok" else "MISMATCH"
mok <- function(a, b, tol = 1e-2) if (isTRUE(all.equal(unname(a), unname(as.numeric(b)), tolerance = tol))) "ok" else "MISMATCH"

hs   <- HolzingerSwineford1939
vars <- paste0("x", 1:9)
Xg   <- lapply(split(hs[vars], hs$school), as.matrix)   # list of 2 raw-data matrices
gnm  <- names(Xg)
ssg  <- data_sample_stats_from_raw(Xg)                 # list(S = list(S1,S2), mean = list(m1,m2), nobs = c(n1,n2))
ng   <- ssg$nobs
df_hs <- as.data.frame(hs[c("school", vars)])
cat(sprintf("Groups: %s   (n = %s)\n\n", paste(gnm, collapse = ", "), paste(ng, collapse = ", ")))

m_cfg <- "visual  =~ x1 + x2 + x3
          textual =~ x4 + x5 + x6
          speed   =~ x7 + x8 + x9"
# metric invariance: bare loading labels are shared across groups by magmaan's
# (and lavaan's) multi-group machinery — exactly `group.equal = "loadings"`.
m_met <- "visual  =~ x1 + L1*x2 + L2*x3
          textual =~ x4 + L3*x5 + L4*x6
          speed   =~ x7 + L5*x8 + L6*x9"

## ---- fit both 2-group models ----------------------------------------------
pt_cfg <- lavaan_lavaanify(m_cfg, n_groups = 2L, group_var = "school")
pt_met <- lavaan_lavaanify(m_met, n_groups = 2L, group_var = "school")

fit_cfg <- fit_fit(pt_cfg, ssg)
fit_met <- fit_fit(pt_met, ssg)
T_cfg   <- infer_chi2_stat(fit_sample_stats(fit_cfg), fit_cfg$fmin);  df_cfg <- infer_df_stat(fit_cfg$partable, fit_sample_stats(fit_cfg))
T_met   <- infer_chi2_stat(fit_sample_stats(fit_met), fit_met$fmin);  df_met <- infer_df_stat(fit_met$partable, fit_sample_stats(fit_met))
n_eqcon <- sum(fit_met$partable$op == "==")

cat("--- configural (loadings free in both groups) ---\n")
cat(sprintf("  ngroups = %d, npar = %d, df = %d, χ² = %.4f (p %.4g)\n\n",
            fit_cfg$ngroups, fit_cfg$npar, df_cfg, T_cfg,
            infer_chi2_pvalue(T_cfg, df_cfg)))

cat("--- metric (loadings constrained equal across groups) ---\n")
cat(sprintf("  npar = %d free indices, %d cross-group equality constraints → %d effective, df = %d, χ² = %.4f (p %.4g)\n",
            fit_met$npar, n_eqcon, fit_met$npar - n_eqcon,
            df_met, T_met, infer_chi2_pvalue(T_met, df_met)))
ld <- fit_met$partable$est[fit_met$partable$op == "=~" & fit_met$partable$rhs == "x2"]
cat(sprintf("  e.g. λ(visual=~x2): group 1 = %.6f, group 2 = %.6f   (equal: %s)\n\n",
            ld[1], ld[2], isTRUE(all.equal(ld[1], ld[2]))))

cat("--- metric-vs-configural invariance test (LR, in pure R via magmaan) ---\n")
lr_stat <- T_met - T_cfg
lr_df   <- df_met - df_cfg
cat(sprintf("  Δχ² = %.4f,  Δdf = %d,  p = %.4g\n\n", lr_stat, lr_df, infer_chi2_pvalue(lr_stat, lr_df)))

## ---- cross-check vs lavaan ------------------------------------------------
lav_cfg <- lavaan::cfa(m_cfg, data = df_hs, group = "school")
lav_met <- lavaan::cfa(m_cfg, data = df_hs, group = "school", group.equal = "loadings")
cat("--- cross-check vs lavaan (group = \"school\") ---\n")
cat(sprintf("  lavaan: configural df=%g χ²=%.4f | metric df=%g χ²=%.4f | ΔLR=%.4f Δdf=%g\n",
            fitMeasures(lav_cfg, "df"), fitMeasures(lav_cfg, "chisq"),
            fitMeasures(lav_met, "df"), fitMeasures(lav_met, "chisq"),
            as.numeric(fitMeasures(lav_met, "chisq")) - as.numeric(fitMeasures(lav_cfg, "chisq")),
            fitMeasures(lav_met, "df") - fitMeasures(lav_cfg, "df")))
cat(sprintf("  match: configural χ² %s | metric χ² %s | configural df %s | metric df %s\n\n",
            mok(T_cfg, fitMeasures(lav_cfg, "chisq")),
            mok(T_met, fitMeasures(lav_met, "chisq")),
            ok(identical(as.integer(df_cfg), as.integer(as.numeric(fitMeasures(lav_cfg, "df"))))),
            ok(identical(as.integer(df_met), as.integer(as.numeric(fitMeasures(lav_met, "df")))))))

## ---- robust (Satorra-Bentler) χ² for both multi-group models --------------
##   the same 5-step chain on each fit; for the metric model `infer_build_u_factor`
##   reparameterizes Δ → Δ·K so the spectrum has df = p* − n_alpha eigenvalues.
sb_of <- function(pt, fit, T_ml, df_ml) {
  uf <- infer_build_u_factor_parts(fit$partable, fit_sample_stats(fit), fit$theta)
  ev <- infer_ugamma_eigenvalues(infer_reduced_gamma_sample(uf, infer_casewise_contributions(pt, Xg), fit$nobs))
  infer_satorra_bentler(T_ml, df_ml, ev)
}
sb_cfg <- sb_of(pt_cfg, fit_cfg, T_cfg, df_cfg)
sb_met <- sb_of(pt_met, fit_met, T_met, df_met)
lav_cfg_m <- lavaan::cfa(m_cfg, data = df_hs, group = "school", estimator = "MLM")
lav_met_m <- lavaan::cfa(m_cfg, data = df_hs, group = "school", group.equal = "loadings", estimator = "MLM")
cat("--- robust (Satorra-Bentler) scaled χ², both multi-group models ---\n")
cat(sprintf("  configural: T_ML=%.4f → c=%.4f, T_SB=%.4f   (lavaan c=%.4f, T_SB=%.4f)  | c %s, T_SB %s\n",
            T_cfg, sb_cfg$scale_c, sb_cfg$chi2_scaled,
            fitMeasures(lav_cfg_m, "chisq.scaling.factor"), fitMeasures(lav_cfg_m, "chisq.scaled"),
            mok(sb_cfg$scale_c, fitMeasures(lav_cfg_m, "chisq.scaling.factor")),
            mok(sb_cfg$chi2_scaled, fitMeasures(lav_cfg_m, "chisq.scaled"))))
cat(sprintf("  metric:     T_ML=%.4f → c=%.4f, T_SB=%.4f   (lavaan c=%.4f, T_SB=%.4f)  | c %s, T_SB %s\n",
            T_met, sb_met$scale_c, sb_met$chi2_scaled,
            fitMeasures(lav_met_m, "chisq.scaling.factor"), fitMeasures(lav_met_m, "chisq.scaled"),
            mok(sb_met$scale_c, fitMeasures(lav_met_m, "chisq.scaling.factor")),
            mok(sb_met$chi2_scaled, fitMeasures(lav_met_m, "chisq.scaled"))))
