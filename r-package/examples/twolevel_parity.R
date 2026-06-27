# Two-level ML: magmaan vs lavaan parity smoke (R end-to-end).
# Exercises the exported two-level surface -- fit_twolevel() and the
# magmaan(cluster = ) one-call entry -- against lavaan::sem(..., cluster = ),
# comparing theta-hat, SE, chi-square, and df. The exhaustive checks (including
# the non-saturated and multi-group cases) live in tests/testthat/test-twolevel.R.
suppressMessages({
  library(magmaan)
  library(lavaan)
})

data(Demo.twolevel)
d <- Demo.twolevel

model <- '
level: 1
  fw =~ y1 + y2 + y3
level: 2
  fb =~ y1 + y2 + y3
'

fit_lav <- sem(model, data = d, cluster = "cluster")
pt_l    <- parTable(fit_lav)
lav_chisq <- unname(fitMeasures(fit_lav, "chisq"))
lav_df    <- as.integer(unname(fitMeasures(fit_lav, "df")))

# Exported wrapper: model syntax + data.frame + cluster column name.
fm <- fit_twolevel(model, d, cluster = "cluster")

# One-call entry: magmaan(..., cluster = ) must reproduce the same fit.
fm1 <- magmaan(model, d, estimator = "ML", cluster = "cluster")
stopifnot(isTRUE(all.equal(fm$theta, fm1$theta, tolerance = 1e-8)),
          isTRUE(all.equal(fm$chisq, fm1$chisq, tolerance = 1e-8)))

cat(sprintf("magmaan: chisq=%.5f  df=%d  npar=%d  conv=%s  iters=%d\n",
            fm$chisq, fm$df, fm$npar, fm$converged, fm$iterations))
cat(sprintf("lavaan : chisq=%.5f  df=%d\n", lav_chisq, lav_df))

# Match free parameters by (lhs, op, rhs, block).
key   <- function(t) paste(t$lhs, t$op, t$rhs, t$block)
mfree <- fm$partable[fm$partable$free > 0, ]
mfree$k       <- key(mfree)
mfree$mag_est <- mfree$est
mfree$mag_se  <- fm$se[mfree$free]
lfree <- pt_l[pt_l$free > 0, ]
lfree$k <- key(lfree)

cmp <- merge(mfree[, c("k", "mag_est", "mag_se")],
             lfree[, c("k", "est", "se")], by = "k")
cmp$d_est <- abs(cmp$mag_est - cmp$est)
cmp$d_se  <- abs(cmp$mag_se  - cmp$se)
print(cmp[, c("k", "mag_est", "est", "mag_se", "se", "d_est", "d_se")],
      digits = 5, row.names = FALSE)

cat(sprintf("\nmatched %d/%d free params | max|d_est|=%.2e  max|d_se|=%.2e  |d_chisq|=%.2e\n",
            nrow(cmp), sum(fm$partable$free > 0),
            max(cmp$d_est), max(cmp$d_se), abs(fm$chisq - lav_chisq)))

ok <- nrow(cmp) == sum(fm$partable$free > 0) &&
      max(cmp$d_est) < 1e-3 && max(cmp$d_se) < 5e-3 &&
      abs(fm$chisq - lav_chisq) < 1e-3 && fm$df == lav_df
cat(if (ok) "\nPARITY OK\n" else "\nPARITY MISMATCH\n")
if (!ok) stop("twolevel_parity.R: PARITY MISMATCH")
