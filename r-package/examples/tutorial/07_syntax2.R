## lavaan tutorial — Model syntax 2   https://lavaan.ugent.be/tutorial/syntax2.html
##
## Fixing parameters, freeing them (NA*), starting values, parameter labels,
## simple equality constraints, the equal() modifier, orthogonal = TRUE, and
## nonlinear equality constraints. Each is fit and cross-checked against lavaan.

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
hs <- HolzingerSwineford1939

cat("=== Model syntax 2 ===\n")

## --- fixing a loading, freeing the marker (NA*), starting values -----------
m_fix <- "visual =~ NA*x1 + 0.7*x2 + x3
          visual ~~ 1*visual"
f_fix <- magmaan(m_fix, hs, estimator = "ML", se = "none", test = "none")
l_fix <- cfa(m_fix, data = hs)
ok(f_fix$converged && est_match(f_fix, l_fix), "fixed loading + NA* free marker")

m_start <- "visual =~ x1 + start(0.7)*x2 + start(1.2)*x3"
f_start <- magmaan(m_start, hs, estimator = "ML", se = "none", test = "none")
l_start <- cfa(m_start, data = hs)
ok(f_start$converged && est_match(f_start, l_start), "start() starting values")

## --- simple equality via a shared label, and the equal() modifier ----------
m_lab <- "visual =~ x1 + v*x2 + v*x3"
f_lab <- magmaan(m_lab, hs, estimator = "ML", se = "none", test = "none")
l_lab <- cfa(m_lab, data = hs)
ok(f_lab$converged && est_match(f_lab, l_lab), "shared-label equality (v*x2 + v*x3)")

m_eq <- "visual =~ x1 + x2 + equal(\"visual=~x2\")*x3"
f_eq <- magmaan(m_eq, hs, estimator = "ML", se = "none", test = "none")
l_eq <- cfa(m_eq, data = hs)
ok(f_eq$converged && near(magmaan_core$infer_chi2_stat(
       magmaan_core$fit_sample_stats(f_eq), f_eq$fmin),
       fitMeasures(l_eq, "chisq")), "equal(\"...\") modifier vs lavaan")

## --- orthogonal = TRUE: latent covariances fixed at 0 ----------------------
m_orth <- "visual =~ x1+x2+x3
           textual =~ x4+x5+x6
           speed =~ x7+x8+x9"
f_orth <- magmaan(model_spec(m_orth, orthogonal = TRUE), hs,
                  estimator = "ML", se = "none", test = "none")
l_orth <- cfa(m_orth, data = hs, orthogonal = TRUE)
ok(f_orth$converged && est_match(f_orth, l_orth), "orthogonal = TRUE")

## --- nonlinear equality constraint: a == b^2 -------------------------------
m_nl <- "visual =~ x1 + a*x2 + b*x3
         a == b^2"
f_nl <- magmaan(m_nl, hs[, c("x1","x2","x3")], estimator = "ML",
                se = "none", test = "none")
l_nl <- cfa(m_nl, data = hs[, c("x1","x2","x3")])
ok(f_nl$converged && est_match(f_nl, l_nl), "nonlinear constraint a == b^2")
## the augmented-Lagrangian fit satisfies the constraint at the solution
a_hat <- f_nl$partable$est[f_nl$partable$lhs == "visual" & f_nl$partable$rhs == "x2"][1]
b_hat <- f_nl$partable$est[f_nl$partable$lhs == "visual" & f_nl$partable$rhs == "x3"][1]
ok(near(a_hat, b_hat^2, 1e-5),                 "  constraint h(theta) = a - b^2 ~ 0")

cat("model syntax 2: ok\n")
