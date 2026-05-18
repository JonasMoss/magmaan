## lavaan tutorial — Meanstructures   https://lavaan.ugent.be/tutorial/means.html
##
## With meanstructure = TRUE the model also fits observed-variable intercepts
## (and latent means where free). magmaan adds the ν / α rows just as lavaan
## does; here the 3-factor CFA is fit with a mean structure and cross-checked.

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
hs  <- HolzingerSwineford1939
fit <- magmaan(model, hs, estimator = "ML", meanstructure = TRUE,
               se = "none", test = "none")
lav <- cfa(model, data = hs, meanstructure = TRUE)

n_int <- sum(fit$partable$op == "~1" & fit$partable$free > 0)

cat("=== meanstructure: 3-factor CFA with intercepts ===\n")
ok(fit$converged,                  "magmaan converged")
ok(n_int == 9,                     "9 observed-variable intercepts (~1) are free")
ok(fit$npar == length(coef(lav)),  "free-parameter count vs lavaan")
ok(est_match(fit, lav),            "estimates (loadings, variances, intercepts) vs lavaan")
cat("meanstructures: ok\n")
