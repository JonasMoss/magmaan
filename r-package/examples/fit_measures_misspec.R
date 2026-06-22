# Misspecification-robust ordinal fit measures with estimated-weight confidence
# intervals: fit_measures_misspec() is the single consolidated surface over the
# per-index estimated-weight inference (RMSEA, CRMR, SRMR, CFI, TLI), each with a
# CI that propagates the sampling variability of the estimated polychoric weight.
#
# There is no external oracle for these intervals (they are magmaan
# constructions; see docs/research/notes/cfi_tli_misspec_inference.tex and the
# RMSEA/CRMR notes). This example pins the wiring: the bundle's RMSEA equals the
# standalone profile-RMSEA binding, SRMR is the CRMR statistic rescaled to the
# vech denominator, the intervals are ordered and in range, and CFI is nearly
# invariant to whether the weight is treated as estimated or fixed (the
# Monte-Carlo finding), unlike RMSEA.

core <- magmaan::magmaan_core

## --- a clearly misspecified all-ordinal DWLS fit ----------------------------
# Congeneric one-factor population (unequal loadings) fit with a tau-equivalent
# model (loadings tied), so the model is misspecified and every index is
# non-trivial.
set.seed(7L)
n <- 1000L
lambda <- c(0.85, 0.65, 0.50, 0.40)
p <- length(lambda)
f <- rnorm(n)
z <- sapply(seq_len(p), function(j) lambda[j] * f + sqrt(1 - lambda[j]^2) * rnorm(n))
df <- data.frame(lapply(seq_len(p), function(j)
  ordered(cut(z[, j], c(-Inf, -0.4, 0.7, Inf), labels = FALSE))))
names(df) <- paste0("x", seq_len(p))

m <- magmaan::model_spec("f =~ x1 + 1*x2 + 1*x3 + 1*x4",
                         ordered = paste0("x", seq_len(p)),
                         parameterization = "delta")
d <- core$data_ordinal_stats_from_df(df, m)
fit <- core$fit_dwls_ordinal(
  m, d, control = list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8))
stopifnot(isTRUE(fit$ordinal), identical(fit$estimator, "DWLS"))

## --- the consolidated table runs end to end ---------------------------------
fm <- magmaan::fit_measures_misspec(fit, d)
expected <- c("rmsea", "rmsea.ci.lower", "rmsea.ci.upper", "rmsea.pvalue",
              "crmr", "crmr.ci.lower", "crmr.ci.upper", "crmr.pvalue",
              "srmr", "srmr.ci.lower", "srmr.ci.upper",
              "cfi", "cfi.ci.lower", "cfi.ci.upper",
              "tli", "tli.ci.lower", "tli.ci.upper",
              "chisq", "df", "baseline.chisq", "baseline.df",
              "conf.level", "estimated.weight", "warnings")
stopifnot(all(expected %in% names(fm)))
stopifnot(isTRUE(fm$estimated.weight), fm$conf.level == 0.90)
stopifnot(fm$df == 5L, fm$baseline.df == 6L)              # p=4 congeneric vs indep
stopifnot(fm$baseline.chisq > fm$chisq)                   # baseline misfits grossly

## --- intervals are ordered and in range -------------------------------------
stopifnot(fm$rmsea >= 0, fm$rmsea.ci.lower <= fm$rmsea, fm$rmsea <= fm$rmsea.ci.upper)
stopifnot(fm$crmr >= 0, fm$crmr.ci.lower <= fm$crmr.ci.upper)
stopifnot(fm$srmr >= 0, fm$srmr.ci.lower <= fm$srmr.ci.upper)
stopifnot(fm$cfi >= 0, fm$cfi <= 1, fm$cfi.ci.lower <= fm$cfi, fm$cfi <= fm$cfi.ci.upper)
stopifnot(fm$cfi.ci.upper <= 1)
stopifnot(fm$tli.ci.lower <= fm$tli, fm$tli <= fm$tli.ci.upper)
stopifnot(all(vapply(list(fm$rmsea.pvalue, fm$crmr.pvalue), function(q)
  is.finite(q) && q >= 0 && q <= 1, logical(1))))

## --- the bundle equals the standalone profile-RMSEA binding -----------------
pr <- core$ordinal_profile_rmsea(fit, d)
stopifnot(isTRUE(all.equal(pr$rmsea, fm$rmsea, tolerance = 1e-9)))

## --- SRMR is the CRMR statistic rescaled to the vech denominator -------------
# p = 4: off-diagonal count 6, vech length 10, scale = sqrt(6 / 10).
scale <- sqrt(6 / 10)
stopifnot(isTRUE(all.equal(fm$srmr, fm$crmr * scale, tolerance = 1e-9)))
stopifnot(isTRUE(all.equal(fm$srmr.ci.lower, fm$crmr.ci.lower * scale, tolerance = 1e-9)))

## --- estimated vs fixed weight: both run; CFI is nearly weight-invariant -----
# The gamma channel is large for RMSEA but small for CFI (it sits at the CRMR end
# of the spectrum), so the estimated- and fixed-weight CFI intervals nearly
# coincide while RMSEA's move more.
fx <- magmaan::fit_measures_misspec(fit, d, estimated_weight = FALSE)
stopifnot(isFALSE(fx$estimated.weight))
stopifnot(abs(fx$cfi - fm$cfi) < 0.01)                    # CFI ~ weight-invariant
stopifnot(is.finite(fx$rmsea), is.finite(fx$tli))

## --- ordinal_stats must be supplied explicitly ------------------------------
# A fitted object does not retain the integer data the estimated-weight
# inference needs, so the stats are passed explicitly (like robust_ordinal()).
err <- tryCatch(magmaan::fit_measures_misspec(fit),
                error = function(e) conditionMessage(e))
stopifnot(grepl("ordinal_stats", err, fixed = TRUE))

cat(sprintf(
  "fit_measures_misspec: rmsea=%.3f [%.3f, %.3f]  cfi=%.3f [%.3f, %.3f]  tli=%.3f [%.3f, %.3f]\n",
  fm$rmsea, fm$rmsea.ci.lower, fm$rmsea.ci.upper,
  fm$cfi, fm$cfi.ci.lower, fm$cfi.ci.upper,
  fm$tli, fm$tli.ci.lower, fm$tli.ci.upper))
cat("fit_measures_misspec example: ok\n")
