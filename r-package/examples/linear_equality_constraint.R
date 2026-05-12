## latva R bindings — general *linear* equality constraints (P9 phase 2):
## a `==` row whose sides are affine in the parameters (`b2 + b3 == 1.5`),
## plus the `effect.coding` identification convention (`Σλ == #indicators`).
## Both are enforced by `latva_fit()` via an affine reparam θ = θ₀ + Kα — the
## solver never sees the constraint. Cross-checked against lavaan.
##
## Run from the repo root (after `R CMD INSTALL r-package`):
##     Rscript r-package/examples/linear_equality_constraint.R

suppressMessages({ library(latva); library(lavaan) })

ok  <- function(cond) if (isTRUE(cond)) "ok" else "MISMATCH"
mok <- function(a, b, tol = 1e-3)
  if (isTRUE(all.equal(unname(as.numeric(a)), unname(as.numeric(b)), tolerance = tol))) "ok" else "MISMATCH"

hs <- HolzingerSwineford1939
S  <- latva_sample_stats_from_raw(list(as.matrix(hs[paste0("x", 1:3)])))

## ---- 1) a linear `==` constraint: λ_x2 + λ_x3 == 1.5 ----------------------
m  <- "visual =~ x1 + b2*x2 + b3*x3\nb2 + b3 == 1.5"
pt <- latva_lavaanify(m)                     # the `==` row rides as op == "=="
fit <- latva_fit(pt, S)                      # from_lavaan_partable re-parses it
se  <- latva_se_expected(fit)
lav <- cfa(m, hs)
fr  <- { p <- parTable(lav)[parTable(lav)$free > 0, ]; p[order(p$free), ] }

cat("== linear == constraint  (visual =~ x1 + b2*x2 + b3*x3 ; b2 + b3 == 1.5) ==\n")
cat(sprintf("  converged                : %s\n", ok(isTRUE(fit$converged))))
cat(sprintf("  b2 + b3 == 1.5           : %s   (got %.6f)\n",
            ok(abs(fit$theta[1] + fit$theta[2] - 1.5) < 1e-6), fit$theta[1] + fit$theta[2]))
cat(sprintf("  theta_hat vs lavaan      : %s\n", mok(fit$theta, fr$est, 1e-4)))
cat(sprintf("  se vs lavaan             : %s\n", mok(se$se, fr$se, 1e-3)))
cat(sprintf("  chi2 / df vs lavaan      : %s / %s   (%.4f vs %.4f ; %d vs %d)\n",
            mok(se$chi2, fitMeasures(lav)["chisq"], 1e-3),
            ok(se$df == as.integer(fitMeasures(lav)["df"])),
            se$chi2, fitMeasures(lav)["chisq"], se$df, as.integer(fitMeasures(lav)["df"])))

## ---- 2) effect coding: Σλ == #indicators ---------------------------------
m2  <- "visual =~ x1 + x2 + x3"
pt2 <- latva_lavaanify(m2, effect_coding = TRUE)   # synthesizes `.p1.+.p2.+.p3. == 3`
fit2 <- latva_fit(pt2, S)
se2  <- latva_se_expected(fit2)
lav2 <- cfa(m2, hs, effect.coding = "loadings")

# loadings are the first three free params under effect coding (all free).
lsum <- fit2$theta[1] + fit2$theta[2] + fit2$theta[3]
cat("\n== effect coding  (visual =~ x1 + x2 + x3 , effect_coding = TRUE) ==\n")
cat(sprintf("  converged                : %s\n", ok(isTRUE(fit2$converged))))
cat(sprintf("  Σλ == 3                  : %s   (got %.6f)\n", ok(abs(lsum - 3) < 1e-5), lsum))
cat(sprintf("  chi2 / df vs lavaan      : %s / %s   (%.6f vs %.6f ; %d vs %d)\n",
            mok(se2$chi2, fitMeasures(lav2)["chisq"], 1e-3),
            ok(se2$df == as.integer(fitMeasures(lav2)["df"])),
            se2$chi2, fitMeasures(lav2)["chisq"], se2$df, as.integer(fitMeasures(lav2)["df"])))
# bijectivity: same model under the marker convention has the same χ² / df.
m_mk <- "visual =~ x1 + x2 + x3"
fit_mk <- latva_fit(latva_lavaanify(m_mk), S)
se_mk  <- latva_se_expected(fit_mk)
cat(sprintf("  chi2 / df vs marker fit  : %s / %s\n",
            mok(se2$chi2, se_mk$chi2, 1e-6), ok(se2$df == se_mk$df)))
