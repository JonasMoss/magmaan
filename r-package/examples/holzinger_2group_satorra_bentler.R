## latva R bindings — two-group Holzinger-Swineford CFA with the
## Satorra-Bentler scaled χ², built up from the thin wrappers and
## cross-checked against lavaan's `estimator = "MLM"`.
##
## Run from the repo root (after `R CMD INSTALL r-package`, or
## `devtools::load_all("r-package")`):
##     Rscript r-package/examples/holzinger_2group_satorra_bentler.R
##
## What it shows, end to end:
##   1. fit the 3-factor CFA on both `school` groups at once — one uniform path,
##      `latva_lavaanify(m, n_groups = 2L, group_var = "school")` →
##      `latva_fit(pt, latva_sample_stats_from_raw(Xg))` (Xg a list of two raw
##      matrices; the {S, nobs, mean} bundle goes straight in);
##   2. the *pooled* (multi-group) Satorra-Bentler scaled χ² — the canonical
##      `latva_build_u_factor → latva_casewise_contributions →
##       latva_reduced_gamma_sample(uf, Zc, fit$nobs) → latva_ugamma_eigenvalues →
##       latva_satorra_bentler` pipeline (`fit$nobs` is the per-block divisor
##      vector: the reduced meat is Σ_g B_gᵀΓ̂_gB_g with each Γ̂_g over n_g);
##   3. the same for the *metric-invariance* model (shared loading labels →
##      cross-group `==` rows; `latva_build_u_factor` reparameterizes Δ → Δ·K,
##      so its UΓ̂ spectrum is df = p* − n_alpha eigenvalues);
##   4. the robust ("sandwich") two-group SEs — `latva_robust_se_raw` (≙ lavaan
##      `se = "robust.sem"`).
##
## Everything is checked against `lavaan::cfa(..., estimator = "MLM")` /
## `cfa(..., se = "robust.sem")`. χ² / df / the SB scaling factor are group-order
## invariant, so the `split(...)`-vs-lavaan group ordering is irrelevant for them.

suppressMessages({ library(latva); library(lavaan) })

ok <- function(a, b, tol = 1e-2) if (isTRUE(all.equal(unname(a), unname(as.numeric(b)), tolerance = tol))) "ok" else "MISMATCH"

hs    <- HolzingerSwineford1939
vars  <- paste0("x", 1:9)
Xg    <- lapply(split(hs[vars], hs$school), as.matrix)   # list of 2 raw-data matrices
gnm   <- names(Xg)
ssg   <- latva_sample_stats_from_raw(Xg)                 # list(S = list(S1,S2), mean = list(m1,m2), nobs = c(n1,n2))
ng    <- ssg$nobs
df_hs <- as.data.frame(hs[c("school", vars)])

m <- "visual  =~ x1 + x2 + x3
      textual =~ x4 + x5 + x6
      speed   =~ x7 + x8 + x9"

cat(sprintf("Holzinger-Swineford, grouped by school: %s  (n = %s, N = %d)\n\n",
            paste(gnm, collapse = ", "), paste(ng, collapse = ", "), sum(ng)))

## ===========================================================================
## 1. two-group CFA fit
## ===========================================================================
pt   <- latva_lavaanify(m, n_groups = 2L, group_var = "school")
fit  <- latva_fit(pt, ssg)                  # the {S, nobs, mean} bundle, straight in
se   <- latva_se_expected(fit)

cat("--- two-group configural CFA (latva) ---\n")
cat(sprintf("  ngroups = %d (%s), npar = %d, df = %d, T_ML = χ² = %.4f  (p %.4g)\n\n",
            fit$ngroups, paste(fit$group_labels, collapse = "/"), fit$npar, se$df, se$chi2,
            latva_chi2_pvalue(se$chi2, se$df)))

lav  <- lavaan::cfa(m, data = df_hs, group = "school", estimator = "MLM")
cat("--- cross-check vs lavaan::cfa(group = \"school\") ---\n")
cat(sprintf("  lavaan: df = %g, χ² = %.4f\n", fitMeasures(lav, "df"), fitMeasures(lav, "chisq")))
cat(sprintf("  match: df %s | T_ML %s\n\n",
            ok(as.integer(se$df), as.integer(fitMeasures(lav, "df")), 0),
            ok(se$chi2, fitMeasures(lav, "chisq"))))

## ===========================================================================
## 2. pooled (multi-group) Satorra-Bentler scaled χ²
## ===========================================================================
uf <- latva_build_u_factor(fit)                          # blocks = 2; B is (p*_1 + p*_2) × df
Zc <- latva_casewise_contributions(pt, Xg)               # N × (p*_1 + p*_2), block-diagonal layout
M  <- latva_reduced_gamma_sample(uf, Zc, fit$nobs)       # Σ_g B_gᵀΓ̂_gB_g, each Γ̂_g divided by n_g
ev <- latva_ugamma_eigenvalues(M)
sb <- latva_satorra_bentler(se$chi2, se$df, ev)

cat("--- pooled Satorra-Bentler scaled χ² (both groups) ---\n")
cat(sprintf("  UΓ̂ spectrum: %d eigenvalues, mean %.4f (= scaling c), range [%.4f, %.4f]\n",
            length(ev), mean(ev), min(ev), max(ev)))
cat(sprintf("  T_ML = %.4f (df %d)  →  c = %.4f,  T_SB = %.4f  (p %.4g)\n",
            se$chi2, se$df, sb$scale_c, sb$chi2_scaled, latva_chi2_pvalue(sb$chi2_scaled, sb$df)))
cat(sprintf("  lavaan MLM (group = \"school\"): c = %.4f, T_SB = %.4f\n",
            fitMeasures(lav, "chisq.scaling.factor"), fitMeasures(lav, "chisq.scaled")))
cat(sprintf("  match: c %s | T_SB %s\n\n",
            ok(sb$scale_c,     fitMeasures(lav, "chisq.scaling.factor")),
            ok(sb$chi2_scaled, fitMeasures(lav, "chisq.scaled"))))

## ===========================================================================
## 3. metric-invariance model (loadings tied across groups) — SB still runs
## ===========================================================================
m_met <- "visual  =~ x1 + L1*x2 + L2*x3
          textual =~ x4 + L3*x5 + L4*x6
          speed   =~ x7 + L5*x8 + L6*x9"   # bare loading labels ⇒ cross-group ==  ≙ group.equal="loadings"
pt_met <- latva_lavaanify(m_met, n_groups = 2L, group_var = "school")
fit_met <- latva_fit(pt_met, ssg)
se_met  <- latva_se_expected(fit_met)
uf_met  <- latva_build_u_factor(fit_met)                 # Δ → Δ·K internally; df = p* − n_alpha
ev_met  <- latva_ugamma_eigenvalues(
             latva_reduced_gamma_sample(uf_met, latva_casewise_contributions(pt_met, Xg), fit_met$nobs))
sb_met  <- latva_satorra_bentler(se_met$chi2, se_met$df, ev_met)
lr_stat <- se_met$chi2 - se$chi2;  lr_df <- se_met$df - se$df

cat("--- metric invariance (loadings equal across groups) ---\n")
cat(sprintf("  df = %d, T_ML = %.4f  →  c = %.4f,  T_SB = %.4f   |  invariance LR: Δχ² = %.4f, Δdf = %d, p = %.4g\n",
            se_met$df, se_met$chi2, sb_met$scale_c, sb_met$chi2_scaled, lr_stat, lr_df,
            latva_chi2_pvalue(lr_stat, lr_df)))
lav_met <- lavaan::cfa(m, data = df_hs, group = "school", group.equal = "loadings", estimator = "MLM")
cat(sprintf("  lavaan group.equal=\"loadings\" MLM: df = %g, c = %.4f, T_SB = %.4f\n",
            fitMeasures(lav_met, "df"), fitMeasures(lav_met, "chisq.scaling.factor"), fitMeasures(lav_met, "chisq.scaled")))
cat(sprintf("  match: df %s | c %s | T_SB %s\n\n",
            ok(as.integer(se_met$df), as.integer(fitMeasures(lav_met, "df")), 0),
            ok(sb_met$scale_c,     fitMeasures(lav_met, "chisq.scaling.factor")),
            ok(sb_met$chi2_scaled, fitMeasures(lav_met, "chisq.scaled"))))

## ===========================================================================
## 4. robust ("sandwich") two-group SEs  (≙ lavaan se = "robust.sem")
## ===========================================================================
rse <- latva_robust_se_raw(fit, Xg)                      # multi-block Expected bread
## same model on lavaan's side (cov-only — `cfa(group=)` would auto-add a
## saturated mean structure, which doesn't change χ²/df/SB but does change the
## parameter list), and match SEs by school *name* (latva's group g ↔ Xg[[g]];
## lavaan orders groups by appearance in the data — `lavInspect(., "group.label")`):
lav_rsem <- lavaan::cfa(m, data = df_hs, group = "school", se = "robust.sem", meanstructure = FALSE)
ptl <- fit$partable; ptl <- ptl[ptl$free > 0, ]
ptl$key <- paste(ptl$lhs, ptl$op, ptl$rhs, names(Xg)[ptl$group], sep = "|"); ptl$se_lat <- rse$se[ptl$free]
pel <- parameterEstimates(lav_rsem); pel <- pel[pel$op %in% c("=~", "~~", "~1") & pel$se > 0, ]
pel$key <- paste(pel$lhs, pel$op, pel$rhs, lavInspect(lav_rsem, "group.label")[pel$group], sep = "|")
mm  <- merge(ptl[, c("key", "se_lat")], pel[, c("key", "se")], by = "key")
cat("--- robust.sem (sandwich) SEs, two groups ---\n")
cat(sprintf("  %d free params; matched %d to lavaan se=\"robust.sem\"; max |Δ se| = %.2g  →  %s\n",
            nrow(ptl), nrow(mm), max(abs(mm$se_lat - mm$se)), ok(max(abs(mm$se_lat - mm$se)), 0, 1e-3)))
