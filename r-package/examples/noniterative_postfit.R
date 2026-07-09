## magmaan R bindings -- post-fit interface for the non-iterative (Guttman)
## closed-form CFA estimators. A Guttman fit is a first-class magmaan_fit, so the
## ordinary post-fit measures apply: vcov(), standardized(), residuals(),
## factor_scores(), composite_weights(), parameter_table(), fit_measures()
## (CFI/TLI/RMSEA + SRMR, naive and robust-scaled, over the NT / ULS
## discrepancies), and residual-based modification-index diagnostics. The ML
## score / LRT machinery is guarded off, since a closed-form map is not a
## gradient-zero minimizer.
##
## Run from the repo root (after `R CMD INSTALL r-package` / `just r-dev`):
##     Rscript r-package/examples/noniterative_postfit.R

suppressMessages({ library(magmaan); library(lavaan) })
core <- magmaan::magmaan_core

hs <- lavaan::HolzingerSwineford1939
X  <- as.matrix(hs[paste0("x", 1:9)])
model_3f <- "visual  =~ x1 + x2 + x3
             textual =~ x4 + x5 + x6
             speed   =~ x7 + x8 + x9"
model_1f <- "g =~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9"

guttman <- function(model) {
  pt <- core$lavaan_lavaanify(model)
  ss <- core$data_sample_stats_from_raw(X)
  fit_noniterative_cfa(pt, ss, estimator = "guttman_gls_aligned",
                       composite = "standardized")
}

m_g  <- guttman(model_3f)
m_g1 <- guttman(model_1f)          # misspecified, for the discrimination check

## 1. classed as a first-class fit -----------------------------------------
stopifnot(inherits(m_g, "magmaan_fit"), isTRUE(m_g$noniterative))
print(m_g)

## 2. vcov() dispatches: model (NT) and robust (empirical, needs data) -------
V  <- vcov(m_g, regime = "model")
Vr <- vcov(m_g, regime = "robust", data = X)
stopifnot(is.matrix(V), nrow(V) == m_g$npar, ncol(V) == m_g$npar,
          isTRUE(all.equal(V, t(V))), all(is.finite(diag(V))),
          nrow(Vr) == m_g$npar, all(is.finite(diag(Vr))))
## regime = "robust" without data must error clearly.
stopifnot(inherits(try(vcov(m_g, regime = "robust"), silent = TRUE), "try-error"))

## 3. standardized solution --------------------------------------------------
## std.all puts loadings/correlations in [-1,1] and standardized residual
## variances in [0,1]; a blanket bound is order-independent.
sa <- standardized(m_g, V, type = "all")
stopifnot(all(is.finite(sa$se)), all(abs(sa$theta) <= 1.05))

## 4. residuals + SRMR -------------------------------------------------------
rr <- residuals(m_g)
sr <- lav_residuals(m_g)
stopifnot(is.finite(sr$srmr), sr$srmr >= 0, is.list(sr$summary))

## 5. fit measures: naive + robust-scaled, NT and ULS discrepancies ----------
fm_nt  <- fit_measures(m_g)                                   # default: ntml/nt
fm_uls <- fit_measures_noniterative(m_g, discrepancy = "uls")
S <- m_g$S[[1]]
for (fm in list(fm_nt, fm_uls)) {
  stopifnot(fm$chisq != 0,                                    # not the fmin=0 path
            fm$cfi >= 0, fm$cfi <= 1, fm$tli <= 1.0001, fm$rmsea >= 0,
            fm$srmr >= 0, fm$df > 0,
            is.finite(fm$cfi.robust), fm$cfi.robust >= 0, fm$cfi.robust <= 1,
            is.finite(fm$rmsea.robust), fm$rmsea.robust >= 0)
}
## NTML baseline scaling collapses to 1 under NT Gamma (efficient weight).
stopifnot(isTRUE(all.equal(fm_nt$baseline.scale.c, 1)))
## Likelihood criteria are intentionally undefined for closed-form non-ML fits.
stopifnot(all(is.na(unlist(
  fm_nt[c("logl", "unrestricted.logl", "aic", "bic", "bic2")]))))
fm_emp <- fit_measures_noniterative(m_g, discrepancy = "uls",
                                    gamma = "empirical", data = X)
Xc <- scale(X, center = TRUE, scale = FALSE)
ii <- row(S)[upper.tri(S)]; jj <- col(S)[upper.tri(S)]
gdiag <- vapply(seq_along(ii), function(k) {
  cp <- Xc[, ii[k]] * Xc[, jj[k]]
  mean((cp - mean(cp))^2)
}, numeric(1))
stopifnot(isTRUE(all.equal(fm_emp$baseline.scale.c, sum(2 * gdiag) / length(gdiag))))
## discrimination: the 1-factor model must fit worse.
fm_bad <- fit_measures(m_g1)
stopifnot(fm_bad$cfi < fm_nt$cfi, fm_bad$rmsea > fm_nt$rmsea)

## 6. parameter table: est/se/z/p/CI, aligned to theta -----------------------
ptab <- parameter_table(m_g, V)
stopifnot(all(c("lhs", "op", "rhs", "est", "se", "z", "pvalue",
                "ci.lower", "ci.upper") %in% names(ptab)),
          nrow(ptab) == m_g$npar, all(is.finite(ptab$se)),
          isTRUE(all.equal(ptab$est, as.numeric(m_g$theta))))

## 7. residual-based modification-index diagnostics --------------------------
mi <- noniterative_cfa_modification_indices(m_g, discrepancy = "uls",
                                            gamma = "empirical", data = X)
stopifnot(nrow(mi) > 0,
          all(c("mi.raw", "mi.resid", "drop.resid", "epc.resid") %in% names(mi)),
          any(mi$kind == "fixed"),
          any(is.finite(mi$mi.resid)),
          any(is.finite(mi$drop.resid)))

## 8. composite weights + factor scores --------------------------------------
cw <- composite_weights(m_g, V)
fs <- factor_scores(m_g, hs)
stopifnot(all(is.finite(unlist(fs$scores))))

## 9. guarded (ML score / LRT machinery) -------------------------------------
guarded <- list(
  function() modification_indices(m_g),
  function() score_tests(m_g),
  function() modification_indices_robust(m_g),
  function() case_rerun(m_g, data = hs),
  function() nestedTest(m_g, m_g1)
)
for (g in guarded) stopifnot(inherits(try(g(), silent = TRUE), "try-error"))

## ---- baseline convention cross-check (dev) --------------------------------
## The closed-form ULS baseline T should equal 2 N sum_{i<j} S_ij^2.
off <- S[upper.tri(S)]
Tb_uls_hand <- m_g$nobs * 2 * sum(off^2)
stopifnot(isTRUE(all.equal(fm_uls$baseline.chisq, Tb_uls_hand)))

cat("\n--- non-iterative fit measures (3-factor Guttman) ---\n")
cat(sprintf("  discrepancy=ntml  chisq=%.2f/%d  cfi=%.3f (robust %.3f)  rmsea=%.3f (robust %.3f)  srmr=%.3f\n",
            fm_nt$chisq, fm_nt$df, fm_nt$cfi, fm_nt$cfi.robust,
            fm_nt$rmsea, fm_nt$rmsea.robust, fm_nt$srmr))
cat(sprintf("  discrepancy=uls   chisq=%.2f/%d  cfi=%.3f (robust %.3f)  rmsea=%.3f (robust %.3f)\n",
            fm_uls$chisq, fm_uls$df, fm_uls$cfi, fm_uls$cfi.robust,
            fm_uls$rmsea, fm_uls$rmsea.robust))
cat("\nnoniterative post-fit interface: ok\n")
