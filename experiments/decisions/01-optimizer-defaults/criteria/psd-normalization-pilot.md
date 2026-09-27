# Unconstrained PSD normalization pilot — 2026-09-27

Scope: try full sample normalization before fitting, then backtransform estimates.
No production default change and no equality-constrained models in this pilot.
Fixed zero coefficients, the chosen marker loading, and std.lv's fixed unit
latent variances remain supported identification restrictions. Reject other
nonzero fixed values, repeated free-parameter labels and explicit constraints.

Use the existing test populations outside the constrained family, three draws
per population/N, seed base 202609291. Run each unconstrained model in marker
and std.lv identification, independently. All four existing observed-unit
transforms apply. Add a correlated-residual version of the residual-misfit
population, with no equality constraints. Smoke: first population per family,
smallest N, one draw, seed base + 1.

Within each route compare the current fitter to the same fitter applied to
S / (sd sd'). All starts are constructed in the space where fitting happens:
FABIN3-auto, ordinary L-BFGS with information coordinates, constrained SLSQP
with diagonal preconditioning, 5000 evaluations, ftol_rel=1e-12 and
xtol_rel=1e-10, start floor and feasibility tolerance 1e-6. Both direct PSD and
ordinary-then-PSD are evaluated. Recompute normalization separately for every
unit transform; do not reuse a fitted endpoint or round sample correlations.

For marker identification, each latent's backtransformation unit is the sample
SD of its existing marker. For std.lv it is 1. Transport loadings, latent
regressions, residual/latent covariances and means by their corresponding units.
Verify implied covariance congruence and objective equality with independent
library evaluation in the original units. Retain PSD admissibility and the
library's PSD-aware accuracy result in original units separately from the
normalized fitter's verdict. Never substitute the unconstrained ML accuracy
verdict for the PSD-domain check.

Report paired gains/losses, best observed objective matches, and discrepancies
between unit transforms within each identification. Covariance backtransformation
error is relative to sample SDs; objective agreement threshold 1e-8*(1+abs(f)).
For unit comparisons use 1e-6*(1+abs(f)) and normalized covariance difference
1e-5. Retain all failures. The candidate need not match a different
identification's basin to pass a unit-transport check. This is exploratory;
normalization invariance alone does not establish optimality or justify adoption.
