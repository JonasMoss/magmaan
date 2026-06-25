## LRT (refit) modification indices: the lavaan modindices() table, by refit.
##
## modification_indices_lrt(fit, data) is the refit counterpart of
## modification_indices(): for each parameter ABSENT from the model (cross
## loadings, residual covariances) it refits the augmented model and reports the
## nested chi-square difference (lrt) and the EXACT refitted parameter change
## (epc_lrt), alongside the one-step mi/epc for comparison. See
## docs/research/notes/lrt_modification_indices.tex.

suppressMessages(library(magmaan))

## Single-group two-factor CFA where x3 truly cross-loads on f2 (0.4); the fitted
## model omits that cross-loading, so the modification index has a real target.
set.seed(3)
n <- 600
f1 <- rnorm(n); f2 <- 0.4 * f1 + sqrt(1 - 0.16) * rnorm(n)
mk <- function(a, b = 0)
  a * f1 + b * f2 + rnorm(n, 0, sqrt(pmax(1 - a^2 - b^2, 1e-6)))
dat <- data.frame(x1 = mk(.7), x2 = mk(.7), x3 = mk(.7, .4),
                  x4 = mk(.7), x5 = mk(.7), x6 = mk(.7))
syntax <- "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6"
fit <- magmaan(syntax, dat, estimator = "ML")

mi <- modification_indices_lrt(fit, dat)
stopifnot(inherits(mi, "magmaan_mi_lrt"),
          all(c("mi", "lrt", "epc", "epc_lrt") %in% names(mi)),
          nrow(mi) > 1L)

## The refit chi-square difference of the top candidate reproduces a manual
## augmented refit exactly.
top <- mi[1, ]
rel <- magmaan(paste0(syntax, "\n", top$lhs, " ", top$op, " ", top$rhs),
               dat, estimator = "ML")
chi2 <- function(f) magmaan::magmaan_core$inference_chi2_stat(
  list(S = f$S, nobs = f$nobs, mean = f$sample_mean), f$fmin)
stopifnot(abs(top$lrt - (chi2(fit) - chi2(rel))) < 1e-6)

## The one-step EPC can blow up where the refit (exact) EPC is well behaved: the
## linearization is what the refit avoids.
cat("LRT vs one-step modification indices (top 5):\n")
print(head(mi[, c("lhs", "op", "rhs", "mi", "lrt", "epc", "epc_lrt")], 5),
      row.names = FALSE)
cat("top refit chi-square difference matches a manual augmented refit: ok\n")
