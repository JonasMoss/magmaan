## magmaan R bindings — equality constraints + the robust (Satorra-Bentler) χ²,
## on the Holzinger-Swineford 1939 data, cross-checked against lavaan.
##
## Run from the repo root (after `R CMD INSTALL r-package`, or
## `devtools::load_all("r-package")`):
##     Rscript r-package/examples/constraints_and_satorra_bentler.R
##
## What it shows:
##   1. Equality constraints — `f =~ x1 + a*x2 + a*x3` ties the two loadings;
##      `fit_fit()` enforces it (reparam θ = K·α), `infer_df_stat()` accounts for
##      the rank gain and `infer_vcov_partable()`/`infer_se()` give the tied params
##      equal SEs. Cross-checked vs lavaan.
##   2. The robust UΓ-eigenvalue / Satorra-Bentler chain on the 3-factor HS CFA,
##      composed from the thin wrappers, cross-checked vs lavaan estimator="MLM".
##
## All `magmaan_*` calls take the {S, nobs, mean} sample-stats bundle that
## `data_sample_stats_from_raw()` returns (or a hand-built list(S=, nobs=)).
## For per-group / measurement-invariance + robust, see holzinger_invariance.R
## and holzinger_2group_satorra_bentler.R.

suppressMessages({ library(magmaan); library(lavaan) })

ok <- function(cond) if (isTRUE(cond)) "ok" else "MISMATCH"

hs    <- HolzingerSwineford1939
Xfull <- as.matrix(hs[paste0("x", 1:9)])      # 301 × 9 raw data, columns x1..x9
n     <- nrow(Xfull)

cat("=================================================================\n")
cat("1. Equality constraints: tie the x2 and x3 loadings on a 1-factor CFA\n")
cat("=================================================================\n\n")

X3  <- Xfull[, c("x1", "x2", "x3")]           # all 301 cases, 3 indicators
ss3 <- data_sample_stats_from_raw(X3)        # list(S = list(<3x3>), mean = list(<3>), nobs = 301)

pt_uncon <- lavaan_lavaanify("visual =~ x1 + x2 + x3")
pt_con   <- lavaan_lavaanify("visual =~ x1 + a*x2 + a*x3")   # synthesizes the `==` row

fit_u <- fit_fit(pt_uncon, ss3)
fit_c <- fit_fit(pt_con,   ss3)
T_u   <- infer_chi2_stat(fit_sample_stats(fit_u), fit_u$fmin);   df_u  <- infer_df_stat(fit_u$partable, fit_sample_stats(fit_u))
T_c   <- infer_chi2_stat(fit_sample_stats(fit_c), fit_c$fmin);   df_c  <- infer_df_stat(fit_c$partable, fit_sample_stats(fit_c))
se_c_vec <- infer_se(infer_vcov_partable(infer_information_expected(fit_c), fit_c$partable))

# SE indexed by the free-parameter ordinal in partable$free.
se_of <- function(se_vec, partable, lhs_, rhs_) {
  k <- partable$free[partable$lhs == lhs_ & partable$rhs == rhs_]
  if (length(k) != 1L || k <= 0L) NA_real_ else se_vec[k]
}

cat("--- unconstrained ---\n")
print(fit_u$partable[, c("lhs", "op", "rhs", "free", "est")])
cat(sprintf("  npar = %d, df = %d, chi2 = %.5g\n\n", fit_u$npar, df_u, T_u))

cat("--- with `a*x2 + a*x3` (loadings on x2 and x3 tied) ---\n")
print(fit_c$partable[, c("lhs", "op", "rhs", "free", "label", "est")])
ld <- fit_c$partable$est[fit_c$partable$lhs == "visual" & fit_c$partable$rhs %in% c("x2", "x3")]
cat(sprintf("  effective npar = %d (%d free indices − 1 == constraint), df = %d, chi2 = %.5g\n",
            fit_c$npar - 1L, fit_c$npar, df_c, T_c))
cat(sprintf("  tied loadings = %s   (equal: %s)\n",
            paste(format(ld, digits = 8), collapse = " "), isTRUE(all.equal(ld[1], ld[2]))))
cat(sprintf("  SE(visual=~x2) = %.6f,  SE(visual=~x3) = %.6f   (equal: %s)\n",
            se_of(se_c_vec, fit_c$partable, "visual", "x2"), se_of(se_c_vec, fit_c$partable, "visual", "x3"),
            isTRUE(all.equal(se_of(se_c_vec, fit_c$partable, "visual", "x2"),
                             se_of(se_c_vec, fit_c$partable, "visual", "x3")))))

cat("\n--- LR test for the equality constraint (pure R, via magmaan) ---\n")
lr_stat <- T_c - T_u
lr_df   <- df_c - df_u
cat(sprintf("  Δχ2 = %.5g,  Δdf = %d,  p = %.4g\n", lr_stat, lr_df, infer_chi2_pvalue(lr_stat, lr_df)))

cat("\n--- cross-check vs lavaan (same S, no rescaling) ---\n")
S3 <- ss3$S[[1]]; dimnames(S3) <- list(colnames(X3), colnames(X3))
lav_u <- lavaan::cfa("visual =~ x1+x2+x3",     sample.cov = S3, sample.nobs = n, sample.cov.rescale = FALSE)
lav_c <- lavaan::cfa("visual =~ x1+a*x2+a*x3", sample.cov = S3, sample.nobs = n, sample.cov.rescale = FALSE)
cat(sprintf("  lavaan: unconstrained df=%g chi2=%.5g | constrained df=%g chi2=%.5g  a=%.6f\n",
            fitMeasures(lav_u, "df"), fitMeasures(lav_u, "chisq"),
            fitMeasures(lav_c, "df"), fitMeasures(lav_c, "chisq"), as.numeric(coef(lav_c)["a"])))
cat(sprintf("  match:  df %s   tied loading %s   constrained chi2 %s\n",
            ok(identical(as.integer(df_c), as.integer(as.numeric(fitMeasures(lav_c, "df"))))),
            ok(all.equal(unname(ld[1]), as.numeric(coef(lav_c)["a"]), tolerance = 1e-5)),
            ok(all.equal(T_c, as.numeric(fitMeasures(lav_c, "chisq")), tolerance = 1e-3))))


cat("\n=================================================================\n")
cat("2. Robust UΓ eigenvalues / Satorra-Bentler χ² — 3-factor HS CFA\n")
cat("=================================================================\n\n")

m_hs <- "visual  =~ x1 + x2 + x3
         textual =~ x4 + x5 + x6
         speed   =~ x7 + x8 + x9"

pt_hs  <- lavaan_lavaanify(m_hs)
fit_hs <- fit_fit(pt_hs, data_sample_stats_from_raw(Xfull))
T_hs   <- infer_chi2_stat(fit_sample_stats(fit_hs), fit_hs$fmin);  df_hs <- infer_df_stat(fit_hs$partable, fit_sample_stats(fit_hs))

# the robust chain, composed from the thin wrappers:
uf  <- infer_build_u_factor_parts(fit_hs$partable, fit_sample_stats(fit_hs), fit_hs$theta)  # U-factor at θ̂
Zc  <- infer_casewise_contributions(pt_hs, Xfull)           # casewise vech contributions (raw data)
M   <- infer_reduced_gamma_sample(uf, Zc, fit_hs$nobs)      # BᵀΓ̂B  (df × df); per-block divisor = nobs
ev  <- infer_ugamma_eigenvalues(M)                          # eigenvalues of UΓ̂  ← the deliverable
sb  <- infer_satorra_bentler(T_hs, df_hs, ev)
mva <- infer_mean_var_adjusted(T_hs, df_hs, ev)
ss2 <- infer_scaled_shifted(T_hs, df_hs, ev)

cat(sprintf("  T_ML = %.4f  (df %d, p %.4g)\n", T_hs, df_hs, infer_chi2_pvalue(T_hs, df_hs)))
cat(sprintf("  UΓ̂ eigenvalues (%d of them): min %.4f, mean %.4f, max %.4f\n", length(ev), min(ev), mean(ev), max(ev)))
cat(sprintf("  Satorra-Bentler:        T_SB = %.4f  (c = %.4f, df %d, p %.4g)\n",
            sb$chi2_scaled, sb$scale_c, sb$df, infer_chi2_pvalue(sb$chi2_scaled, sb$df)))
cat(sprintf("  mean-and-var adjusted:  T    = %.4f  (df_adj = %.4f)\n", mva$chi2_adj, mva$df_adj))
cat(sprintf("  scaled-and-shifted:     T    = %.4f  (a = %.4f, b = %.4f, df %d)\n",
            ss2$chi2_adj, ss2$scale_a, ss2$shift_b, ss2$df))
# robust ("sandwich") SEs — se = "robust.sem":
rse <- infer_robust_se_raw_parts(fit_hs$partable, fit_sample_stats(fit_hs), fit_hs$theta, Xfull)

cat("\n--- cross-check vs lavaan ---\n")
lav_mlm <- lavaan::cfa(m_hs, data = as.data.frame(Xfull), estimator = "MLM")
cat(sprintf("  lavaan MLM: T_SB = %.4f  (c = %.4f)\n",
            fitMeasures(lav_mlm, "chisq.scaled"), fitMeasures(lav_mlm, "chisq.scaling.factor")))
cat(sprintf("  match:  T_SB %s   c %s   robust.sem SE %s\n",
            ok(all.equal(sb$chi2_scaled, as.numeric(fitMeasures(lav_mlm, "chisq.scaled")), tolerance = 1e-3)),
            ok(all.equal(sb$scale_c,     as.numeric(fitMeasures(lav_mlm, "chisq.scaling.factor")), tolerance = 1e-3)),
            ok(all.equal(unname(rse$se), unname(sqrt(diag(lavaan::vcov(lav_mlm)))), tolerance = 1e-4))))
