## magmaan R bindings — standard errors from the *observed* information matrix,
## next to the expected-information SEs, cross-checked against lavaan.
##
## Run from the repo root (after `R CMD INSTALL r-package`, or
## `devtools::load_all("r-package")`):
##     Rscript r-package/examples/observed_information_se.R
##
## SEs come from inverting an information matrix, and *which* information matrix
## is an explicit choice — lavaan's `information=` argument. magmaan exposes it
## as three separate primitives, all taking just the fit:
##
##   infer_information_expected(fit)            expected (Fisher) information:
##                                              E[-Hessian], a function of the
##                                              model and the implied moments.
##   infer_information_observed_fd(fit)         observed information: the actual
##                                              Hessian at theta-hat, by a
##                                              central-difference of the
##                                              analytic ML gradient.
##   infer_information_observed_analytic(fit)   observed information in closed
##                                              form.
##
## At the optimum the expected and observed matrices are asymptotically equal,
## so the SEs agree as N grows but differ in any finite sample. The two observed
## variants (finite-difference vs closed-form) should agree to numerical
## precision. This fits the 3-factor Holzinger-Swineford 1939 CFA (covariance
## structure only) and lays the three SE vectors side by side.

suppressMessages({ library(magmaan); library(lavaan) })

mok <- function(a, b, tol = 1e-5)
  if (isTRUE(all.equal(unname(as.numeric(a)), unname(as.numeric(b)),
                       tolerance = tol))) "ok" else "MISMATCH"

hs <- lavaan::HolzingerSwineford1939
df <- as.data.frame(hs[paste0("x", 1:9)])

model <- "visual  =~ x1 + x2 + x3
          textual =~ x4 + x5 + x6
          speed   =~ x7 + x8 + x9"

fit <- magmaan(model, df, estimator = "ML", se = "none", test = "none")

## SE = sqrt(diag(vcov)); vcov = inverse information (with constraint projection
## applied from the partable). Same pipeline three times — only the information
## matrix changes.
se_from <- function(info) magmaan_core$infer_se(
  magmaan_core$infer_vcov_partable(info, fit$partable))

se_exp <- se_from(magmaan_core$infer_information_expected(fit))
se_ofd <- se_from(magmaan_core$infer_information_observed_fd(fit))
se_oan <- se_from(magmaan_core$infer_information_observed_analytic(fit))

free <- fit$partable[fit$partable$free > 0, ]
free <- free[order(free$free), ]
tab <- data.frame(param = paste(free$lhs, free$op, free$rhs),
                  est = free$est,
                  se.expected = se_exp,
                  se.obs.fd = se_ofd,
                  se.obs.analytic = se_oan)

cat("=================================================================\n")
cat(" Standard errors under expected vs observed information\n")
cat("=================================================================\n\n")
print(head(tab, 9), row.names = FALSE, digits = 4)
cat(sprintf("  ... %d free parameters in total\n\n", nrow(tab)))

cat(sprintf("  max |se(observed, fd) - se(observed, analytic)| = %.2e\n",
            max(abs(se_ofd - se_oan))))
cat(sprintf("  max |se(expected)     - se(observed)|          = %.2e\n",
            max(abs(se_exp - se_ofd))))
cat("  (the first is numerical noise; the second is the expected-vs-observed\n")
cat("   gap that vanishes asymptotically.)\n\n")

## ---- cross-check against lavaan -------------------------------------------
## lavaan's `information=` argument selects the same choice; magmaan's free
## parameters are in lavaan's coefficient order, so the SE vectors line up.
lav_exp <- cfa(model, data = df, information = "expected")
lav_obs <- cfa(model, data = df, information = "observed")
lse_exp <- sqrt(diag(lavaan::vcov(lav_exp)))
lse_obs <- sqrt(diag(lavaan::vcov(lav_obs)))

cat("--- cross-check vs lavaan ---\n")
cat(sprintf("  expected info SEs vs lavaan information=\"expected\": %s\n",
            mok(se_exp, lse_exp)))
cat(sprintf("  observed info SEs vs lavaan information=\"observed\": %s (fd), %s (analytic)\n",
            mok(se_ofd, lse_obs), mok(se_oan, lse_obs)))
cat(sprintf("  finite-difference vs analytic observed information: %s\n\n",
            mok(se_ofd, se_oan)))
stopifnot(mok(se_exp, lse_exp) == "ok",
          mok(se_ofd, lse_obs) == "ok",
          mok(se_oan, lse_obs) == "ok",
          mok(se_ofd, se_oan) == "ok")

cat("observed-information SE workflow: ok\n")
