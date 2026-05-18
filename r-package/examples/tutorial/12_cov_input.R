## lavaan tutorial — Covariance-matrix input   https://lavaan.ugent.be/tutorial/cov.html
##
## A model can be fit from a sample covariance matrix + sample size, with no
## raw data. lavaan takes sample.cov / sample.nobs; magmaan accepts a
## list(S = , nobs = ) in place of a data frame. The fit must match the
## raw-data fit of the same model.

suppressMessages({ library(magmaan); library(lavaan) })

near <- function(a, b, tol = 1e-3)
  isTRUE(all.equal(unname(as.numeric(a)), unname(as.numeric(b)), tolerance = tol))
ok <- function(cond, what) {
  cat(sprintf("  %-44s %s\n", what, if (isTRUE(cond)) "ok" else "MISMATCH"))
  stopifnot(isTRUE(cond))
}
est_match <- function(fit, lav, tol = 1e-3) {
  mp <- fit$partable[fit$partable$free > 0, , drop = FALSE]
  lp <- lavaan::parameterEstimates(lav)
  all(vapply(seq_len(nrow(mp)), function(i) {
    r <- lp[lp$lhs == mp$lhs[i] & lp$op == mp$op[i] & lp$rhs == mp$rhs[i], ]
    nrow(r) >= 1 && near(mp$est[i], r$est[1], tol)
  }, logical(1)))
}

model <- "
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"
hs   <- HolzingerSwineford1939[, paste0("x", 1:9)]
nobs <- nrow(hs)
## magmaan's covariance path takes the N-divisor (maximum-likelihood) sample
## covariance directly; lavaan's sample.cov rescaling is turned off so both
## fit exactly the same moments.
S <- cov(hs) * (nobs - 1) / nobs

fit <- magmaan(model, list(S = S, nobs = nobs), estimator = "ML",
               se = "none", test = "none")
lav <- cfa(model, sample.cov = S, sample.nobs = nobs,
           sample.cov.rescale = FALSE)

cat("=== covariance-matrix input ===\n")
ok(fit$converged,                  "magmaan converged from a covariance matrix")
ok(fit$npar == length(coef(lav)),  "free-parameter count vs lavaan")
ok(est_match(fit, lav),            "estimates vs lavaan (sample.cov / sample.nobs)")
cat("covariance matrix input: ok\n")
