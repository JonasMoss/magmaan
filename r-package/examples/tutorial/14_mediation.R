## lavaan tutorial — Mediation   https://lavaan.ugent.be/tutorial/mediation.html
##
## The classic X -> M -> Y mediation model. The indirect effect `ab := a*b`
## and total effect `total := c + a*b` are user-defined (`:=`) parameters;
## magmaan evaluates them with compute_defined(), propagating a delta-method
## standard error. Cross-checked against lavaan::sem().

suppressMessages({ library(magmaan); library(lavaan) })

near <- function(a, b, tol = 1e-4)
  isTRUE(all.equal(unname(as.numeric(a)), unname(as.numeric(b)), tolerance = tol))
ok <- function(cond, what) {
  cat(sprintf("  %-44s %s\n", what, if (isTRUE(cond)) "ok" else "MISMATCH"))
  stopifnot(isTRUE(cond))
}

set.seed(1234)
X <- rnorm(100)
M <- 0.5 * X + rnorm(100)
Y <- 0.7 * M + rnorm(100)
Data <- data.frame(X = X, Y = Y, M = M)

model <- "
  Y ~ c*X            # direct effect
  M ~ a*X            # X -> M
  Y ~ b*M            # M -> Y
  ab    := a*b       # indirect effect
  total := c + a*b   # total effect
"
fit <- magmaan(model, Data, estimator = "ML", se = "none", test = "none")
lav <- sem(model, data = Data)

## defined parameters need the parameter covariance for the delta-method SE.
info <- magmaan_core$infer_information_expected(fit)
vc   <- magmaan_core$infer_vcov_partable(info, fit$partable)
defs <- compute_defined(model, fit, vc)

lp  <- parameterEstimates(lav)
lab <- function(nm, col) lp[[col]][lp$op == ":=" & lp$lhs == nm][1]
get <- function(nm, col) defs[[col]][defs$lhs == nm][1]

cat("=== mediation: indirect and total effects via := ===\n")
ok(fit$converged,                              "magmaan converged")
ok(near(get("ab", "est"),    lab("ab", "est")),    "indirect effect ab := a*b vs lavaan")
ok(near(get("ab", "se"),     lab("ab", "se")),     "  delta-method SE of ab vs lavaan")
ok(near(get("total", "est"), lab("total", "est")), "total effect := c + a*b vs lavaan")
ok(near(get("total", "se"),  lab("total", "se")),  "  delta-method SE of total vs lavaan")
cat(sprintf("  ab = %.4f (se %.4f)   total = %.4f (se %.4f)\n",
            get("ab","est"), get("ab","se"), get("total","est"), get("total","se")))
cat("mediation: ok\n")
