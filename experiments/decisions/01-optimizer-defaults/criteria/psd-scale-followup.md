# PSD coordinate-scale follow-up — 2026-09-27

Written before the follow-up runs. This is a diagnostic and route comparison,
not authorization to promote a route or a new start policy. Previous PSD lane
results and the sphere study are development evidence.

First test the identical model and transported start under changed observed
units. The current absolute information-scale bounds are the control. The
candidate keeps bounds 1e-4 and 1e4 but applies them to the ratio of the
information scale to a sample-derived coordinate unit; zero information uses
that coordinate unit. Original parameters use the shared parameter/equality
coordinate units. Every Cholesky entry uses its row variable's unit, since
L transforms as D L when covariance transforms as D C D. Do not change starts,
feasibility tolerances, the PSD boundary, or optimizer budgets in this check.
This isolates relative clamping from a separate choice of refinement bounds.

Use all existing test populations, three replications at each sample size,
seed base 202609290. Compare direct FABIN3-auto/SLSQP/diagonal PSD with
FABIN3-auto/L-BFGS ordinary followed, when rejected or inadmissible, by
SLSQP/diagonal PSD. ML controls: 5000 evaluations, ftol_rel 1e-12,
xtol_rel 1e-10; ordinary information coordinates. Keep the same paired samples
for absolute and relative scales, all four existing unit transforms when
model-compatible. Smoke uses one replication, first population per family,
and seed base + 1. This small run estimates effects and regressions; it is
not a powered replacement for the earlier defaults lane.

Report library certification plus PSD admissibility, objective agreement among
certified fits, time and paired gains/losses, split by family and units. Keep
failed or worse cases visible. If relative scaling fixes the coordinate test
but does not improve fitting, report that distinction and investigate constraint
or start scaling rather than attributing every units failure to this clamp.
No automatic route promotion follows from this diagnostic.
