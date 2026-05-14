## magmaan R bindings — general *linear* equality constraints (P9 phase 2):
## a `==` row whose sides are affine in the parameters (`b2 + b3 == 1.5`),
## plus the `effect.coding` identification convention (`Σλ == #indicators`).
## Both are enforced by `fit_fit()` via an affine reparam θ = θ₀ + Kα — the
## solver never sees the constraint. Cross-checked against lavaan.
##
## Run from the repo root (after `R CMD INSTALL r-package`):
##     Rscript r-package/examples/linear_equality_constraint.R

suppressMessages({ library(magmaan); library(lavaan) })

ok  <- function(cond) if (isTRUE(cond)) "ok" else "MISMATCH"
mok <- function(a, b, tol = 1e-3)
  if (isTRUE(all.equal(unname(as.numeric(a)), unname(as.numeric(b)), tolerance = tol))) "ok" else "MISMATCH"

hs <- HolzingerSwineford1939
S  <- data_sample_stats_from_raw(list(as.matrix(hs[paste0("x", 1:3)])))

## ---- 1) a linear `==` constraint: λ_x2 + λ_x3 == 1.5 ----------------------
m  <- "visual =~ x1 + b2*x2 + b3*x3\nb2 + b3 == 1.5"
pt <- lavaan_lavaanify(m)                     # the `==` row rides as op == "=="
fit <- fit_fit(pt, S)                      # from_lavaan_partable re-parses it
T_ml <- infer_chi2_stat(fit_sample_stats(fit), fit$fmin);  df_ml <- infer_df_stat(fit$partable, fit_sample_stats(fit))
se   <- infer_se(infer_vcov(infer_information_expected(fit), fit))
lav <- cfa(m, hs)
fr  <- { p <- parTable(lav)[parTable(lav)$free > 0, ]; p[order(p$free), ] }

cat("== linear == constraint  (visual =~ x1 + b2*x2 + b3*x3 ; b2 + b3 == 1.5) ==\n")
cat(sprintf("  converged                : %s\n", ok(isTRUE(fit$converged))))
cat(sprintf("  b2 + b3 == 1.5           : %s   (got %.6f)\n",
            ok(abs(fit$theta[1] + fit$theta[2] - 1.5) < 1e-6), fit$theta[1] + fit$theta[2]))
cat(sprintf("  theta_hat vs lavaan      : %s\n", mok(fit$theta, fr$est, 1e-4)))
cat(sprintf("  se vs lavaan             : %s\n", mok(se, fr$se, 1e-3)))
cat(sprintf("  chi2 / df vs lavaan      : %s / %s   (%.4f vs %.4f ; %d vs %d)\n",
            mok(T_ml, fitMeasures(lav)["chisq"], 1e-3),
            ok(df_ml == as.integer(fitMeasures(lav)["df"])),
            T_ml, fitMeasures(lav)["chisq"], df_ml, as.integer(fitMeasures(lav)["df"])))

## ---- 2) effect coding: Σλ == #indicators ---------------------------------
m2  <- "visual =~ x1 + x2 + x3"
pt2 <- lavaan_lavaanify(m2, effect_coding = TRUE)   # synthesizes `.p1.+.p2.+.p3. == 3`
fit2 <- fit_fit(pt2, S)
T2   <- infer_chi2_stat(fit_sample_stats(fit2), fit2$fmin);  df2 <- infer_df_stat(fit2$partable, fit_sample_stats(fit2))
lav2 <- cfa(m2, hs, effect.coding = "loadings")

# loadings are the first three free params under effect coding (all free).
lsum <- fit2$theta[1] + fit2$theta[2] + fit2$theta[3]
cat("\n== effect coding  (visual =~ x1 + x2 + x3 , effect_coding = TRUE) ==\n")
cat(sprintf("  converged                : %s\n", ok(isTRUE(fit2$converged))))
cat(sprintf("  Σλ == 3                  : %s   (got %.6f)\n", ok(abs(lsum - 3) < 1e-5), lsum))
cat(sprintf("  chi2 / df vs lavaan      : %s / %s   (%.6f vs %.6f ; %d vs %d)\n",
            mok(T2, fitMeasures(lav2)["chisq"], 1e-3),
            ok(df2 == as.integer(fitMeasures(lav2)["df"])),
            T2, fitMeasures(lav2)["chisq"], df2, as.integer(fitMeasures(lav2)["df"])))
# bijectivity: same model under the marker convention has the same χ² / df.
m_mk <- "visual =~ x1 + x2 + x3"
fit_mk <- fit_fit(lavaan_lavaanify(m_mk), S)
T_mk   <- infer_chi2_stat(fit_sample_stats(fit_mk), fit_mk$fmin);  df_mk <- infer_df_stat(fit_mk$partable, fit_sample_stats(fit_mk))
cat(sprintf("  chi2 / df vs marker fit  : %s / %s\n",
            mok(T2, T_mk, 1e-6), ok(df2 == df_mk)))
