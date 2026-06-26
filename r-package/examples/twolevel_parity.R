# Two-level ML: magmaan vs lavaan parity smoke (R end-to-end).
# Fits a two-level CFA on lavaan's Demo.twolevel and compares theta-hat, SE,
# chi-square, and df between magmaan's R binding (fit_twolevel_impl) and
# lavaan::sem(..., cluster=).
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

# magmaan model (model_spec carries the `group` column + block=level encoding).
ms   <- model_spec(model)
pt_m <- ms$partable

ov  <- c("y1", "y2", "y3")
X   <- as.matrix(d[, ov]); storage.mode(X) <- "double"
cid <- as.integer(factor(d$cluster)) - 1L

fm <- magmaan:::fit_twolevel_impl(pt_m, X, cid)

lav_chisq <- unname(fitMeasures(fit_lav, "chisq"))
lav_df    <- as.integer(unname(fitMeasures(fit_lav, "df")))
cat(sprintf("magmaan: chisq=%.5f  df=%d  npar=%d  conv=%s  iters=%d\n",
            fm$chisq, fm$df, fm$npar, fm$converged, fm$iterations))
cat(sprintf("lavaan : chisq=%.5f  df=%d\n", lav_chisq, lav_df))

# Match free parameters by (lhs, op, rhs, block).
key   <- function(t) paste(t$lhs, t$op, t$rhs, t$block)
mfree <- pt_m[pt_m$free > 0, ]
mfree$k       <- key(mfree)
mfree$mag_est <- fm$theta[mfree$free]
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
            nrow(cmp), sum(pt_m$free > 0),
            max(cmp$d_est), max(cmp$d_se), abs(fm$chisq - lav_chisq)))

ok <- nrow(cmp) == sum(pt_m$free > 0) &&
      max(cmp$d_est) < 1e-3 && max(cmp$d_se) < 5e-3 &&
      abs(fm$chisq - lav_chisq) < 1e-3 && fm$df == lav_df
cat(if (ok) "\nPARITY OK\n" else "\nPARITY MISMATCH\n")
