## lavaan tutorial — Multiple groups   https://lavaan.ugent.be/tutorial/groups.html
##
## A 3-factor CFA fit separately in each level of `school`. magmaan replicates
## the model per group; cross-group equality (measurement invariance) is
## expressed with shared parameter labels. Here: the configural model (all
## parameters group-specific) and a metric-invariance model (loadings tied
## across groups by label).

suppressMessages({ library(magmaan); library(lavaan) })

near <- function(a, b, tol = 1e-3)
  isTRUE(all.equal(unname(as.numeric(a)), unname(as.numeric(b)), tolerance = tol))
ok <- function(cond, what) {
  cat(sprintf("  %-44s %s\n", what, if (isTRUE(cond)) "ok" else "MISMATCH"))
  stopifnot(isTRUE(cond))
}
## Group order (which school is group 1) need not agree between magmaan and
## lavaan, so compare the full estimate set as an order-invariant multiset.
est_match_set <- function(fit, lav, tol = 1e-3)
  near(sort(fit$partable$est[fit$partable$free > 0]),
       sort(unname(coef(lav))), tol)

model <- "
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"
hs <- HolzingerSwineford1939

## --- configural: every parameter free in each group -----------------------
f_cfg <- magmaan(model, hs, estimator = "ML", groups = "school",
                 se = "none", test = "none")
## lavaan auto-enables a meanstructure for multigroup fits; magmaan does not
## unless asked, so compare the covariance-only configural model.
l_cfg <- cfa(model, data = hs, group = "school", meanstructure = FALSE)
cat("=== multiple groups (school): configural ===\n")
ok(f_cfg$converged,                          "magmaan converged")
ok(f_cfg$ngroups == 2L,                      "two groups")
ok(f_cfg$npar == length(coef(l_cfg)),        "free-parameter count vs lavaan")
ok(est_match_set(f_cfg, l_cfg),                "per-group estimates vs lavaan")

## --- metric invariance: loadings tied across groups by shared labels ------
model_metric <- "
  visual  =~ x1 + L2*x2 + L3*x3
  textual =~ x4 + L5*x5 + L6*x6
  speed   =~ x7 + L8*x8 + L9*x9
"
f_met <- magmaan(model_metric, hs, estimator = "ML", groups = "school",
                 se = "none", test = "none")
## bare shared labels tie loadings across groups in both tools (lavaan just
## says so out loud) — that is exactly the metric-invariance intent.
l_met <- suppressWarnings(
  cfa(model_metric, data = hs, group = "school", meanstructure = FALSE))
cat("=== metric invariance (loadings equal across groups) ===\n")
ok(f_met$converged,                          "magmaan converged")
ok(f_met$npar == length(coef(l_met)),        "free-parameter count vs lavaan")
ok(est_match_set(f_met, l_met),                "per-group estimates vs lavaan")
cat("multiple groups: ok\n")
