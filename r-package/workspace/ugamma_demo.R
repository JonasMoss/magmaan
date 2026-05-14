suppressMessages({ library(magmaan); library(lavaan) })

# §0  setup — Holzinger-Swineford 3-factor CFA, single- and two-group
m <- "visual  =~ x1 + x2 + x3
      textual =~ x4 + x5 + x6
      speed   =~ x7 + x8 + x9"
hs   <- HolzingerSwineford1939
vars <- paste0("x", 1:9)
X    <- as.matrix(hs[vars])
Xg   <- lapply(split(hs[vars], hs$school), as.matrix)
ss   <- data_sample_stats_from_raw(X)
ssg  <- data_sample_stats_from_raw(Xg)
pt   <- lavaan_lavaanify(m)
fit  <- fit_fit(pt, ss)
se   <- infer_information_expected(fit)
c(npar = fit$npar, df = se$df, chi2 = se$chi2)

# §1  UΓ_NT sanity — eigvals collapse to 1 under MVN
uf    <- infer_build_u_factor(fit)         # expected bread, structured moments
M_nt  <- infer_reduced_gamma_nt(uf)
ev_nt <- infer_ugamma_eigenvalues(M_nt)
ev_nt
range(ev_nt - 1)

# §2  empirical Γ̂ → Satorra-Bentler scaled χ²
Zc <- infer_casewise_contributions(pt, X)  # N × p* centred contributions
M  <- infer_reduced_gamma_sample(uf, Zc, ss$nobs)
ev <- infer_ugamma_eigenvalues(M)
ev
infer_satorra_bentler(se$chi2, se$df, ev)

# §3  three corrections from the same eigenvalues
infer_satorra_bentler(se$chi2, se$df, ev)      # mean (SB)
infer_mean_var_adjusted(se$chi2, se$df, ev)    # Satterthwaite, real df
infer_scaled_shifted(se$chi2, se$df, ev)       # SB-2010, integer df

# §4  Browne unbiased Γ_u (single-block)
M_u  <- infer_reduced_gamma_unbiased(uf, ss$nobs, M, M_nt)
ev_u <- infer_ugamma_eigenvalues(M_u)
ev_u
infer_satorra_bentler(se$chi2, se$df, ev_u)

# §5  observed bread variant
uf_obs <- infer_build_u_factor(fit, bread = "observed")
ev_obs <- infer_ugamma_eigenvalues(
            infer_reduced_gamma_sample(uf_obs, Zc, ss$nobs))
infer_satorra_bentler(se$chi2, se$df, ev_obs)

lav_model <- lavaan::cfa(model, data = HolzingerSwineford1939, estimator = "MLM")


# §6  lavaan MLM single-group — bare side-by-side
lav <- lavaan::cfa(m, data = hs[vars], estimator = "MLM")
fitMeasures(lav, c("chisq", "df", "chisq.scaled", "chisq.scaling.factor"))

# §7  two-group multi-block UΓ̂ (configural)
pt2  <- lavaan_lavaanify(m, n_groups = 2L, group_var = "school")
fit2 <- fit_fit(pt2, ssg)
se2  <- infer_information_expected(fit2)
uf2  <- infer_build_u_factor(fit2)         # B is (p*_1 + p*_2) × df
Zc2  <- infer_casewise_contributions(pt2, Xg)
M2   <- infer_reduced_gamma_sample(uf2, Zc2, fit2$nobs)  # Σ_g B_gᵀΓ̂_g B_g
ev2  <- infer_ugamma_eigenvalues(M2)
ev2
infer_satorra_bentler(se2$chi2, se2$df, ev2)
lav2 <- lavaan::cfa(m, data = hs, group = "school", estimator = "MLM")
fitMeasures(lav2, c("chisq", "df", "chisq.scaled", "chisq.scaling.factor"))

# §8  metric invariance — tied loadings reparameterize Δ → Δ·K
m_met <- "visual  =~ x1 + L1*x2 + L2*x3
          textual =~ x4 + L3*x5 + L4*x6
          speed   =~ x7 + L5*x8 + L6*x9"
pt_met  <- lavaan_lavaanify(m_met, n_groups = 2L, group_var = "school")
fit_met <- fit_fit(pt_met, ssg)
se_met  <- infer_information_expected(fit_met)
uf_met  <- infer_build_u_factor(fit_met)   # df shrinks by # of cross-group equalities
ev_met  <- infer_ugamma_eigenvalues(
             infer_reduced_gamma_sample(uf_met,
               infer_casewise_contributions(pt_met, Xg),
               fit_met$nobs))
ev_met
infer_satorra_bentler(se_met$chi2, se_met$df, ev_met)

# §9  sandwich SEs (same UΓ machinery, expected bread)
rse  <- infer_robust_se_raw(fit,  X)
rse2 <- infer_robust_se_raw(fit2, Xg)
head(rse$se)
head(rse2$se)

# §10 Satorra-2000 nested LR — configural (H1) vs metric (H0)
res <- infer_lr_test_satorra2000(
          fit2, fit_met, Xg,
          T_H1 = se2$chi2,    df_H1 = se2$df,
          T_H0 = se_met$chi2, df_H0 = se_met$df,
          gamma = "empirical")
res[c("T_diff", "df_diff", "p_unscaled",
      "scale_c", "T_scaled", "p_scaled",
      "adjust_d0", "T_adjusted", "p_adjusted", "p_mixture")]
res$eigenvalues

# §11 NT sanity for the nested test — eigenvalues collapse to 1
infer_lr_test_satorra2000(
    fit2, fit_met, Xg,
    T_H1 = se2$chi2,    df_H1 = se2$df,
    T_H0 = se_met$chi2, df_H0 = se_met$df,
    gamma = "NT")$eigenvalues
