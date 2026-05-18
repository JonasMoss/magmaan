## lavaan tutorial — Model syntax 1   https://lavaan.ugent.be/tutorial/syntax1.html
##
## The four core operators: =~ (latent definition), ~ (regression),
## ~~ ((co)variance), ~1 (intercept). magmaan's parser accepts the same
## lavaan model syntax; this fits one model that exercises all four and
## cross-checks the estimates against lavaan.

suppressMessages({ library(magmaan); library(lavaan) })

near <- function(a, b, tol = 1e-3)
  isTRUE(all.equal(unname(as.numeric(a)), unname(as.numeric(b)), tolerance = tol))
ok <- function(cond, what) {
  cat(sprintf("  %-44s %s\n", what, if (isTRUE(cond)) "ok" else "MISMATCH"))
  stopifnot(isTRUE(cond))
}
## match every magmaan free estimate to lavaan, keyed by lhs/op/rhs
est_match <- function(fit, lav, tol = 1e-3) {
  mp <- fit$partable[fit$partable$free > 0, , drop = FALSE]
  lp <- lavaan::parameterEstimates(lav)
  all(vapply(seq_len(nrow(mp)), function(i) {
    r <- lp[lp$lhs == mp$lhs[i] & lp$op == mp$op[i] & lp$rhs == mp$rhs[i], ]
    nrow(r) >= 1 && near(mp$est[i], r$est[1], tol)
  }, logical(1)))
}

model <- "
  # =~  latent variable definition
  ind60 =~ x1 + x2 + x3
  dem60 =~ y1 + y2 + y3 + y4
  # ~   regression
  dem60 ~ ind60
  # ~~  residual covariance
  y1 ~~ y2
"

fit <- magmaan(model, PoliticalDemocracy, estimator = "ML",
               se = "none", test = "none")
lav <- sem(model, data = PoliticalDemocracy)

cat("=== Model syntax 1: =~  ~  ~~ ===\n")
ok(fit$converged,                              "magmaan converged")
ok(fit$npar == length(coef(lav)),              "free-parameter count vs lavaan")
ok(est_match(fit, lav),                        "all point estimates vs lavaan")
cat("model syntax 1: ok\n")
