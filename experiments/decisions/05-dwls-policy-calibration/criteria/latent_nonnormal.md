# Latent non-normality: exact versus OPG (lane B)

Registered before smoke/pilot on 2026-10-05. No production or Modal launch;
merger review precedes production. Exact remains the policy regardless of
comparative results. Library convergence is the fitting judge.

## Populations and estimands

One factor/six indicators and two factors/twelve indicators, loadings
(.65,.70,.75,.80,.70,.75), unit response variances and factor correlation .3.
Independent standardized centered chi-square factors and residuals use df 8
or 2; Gaussian innovations form the development control. Correlated factors
are formed with the Cholesky factor of the target factor covariance. Responses
are linear factor plus residual combinations, then thresholded directly.
No discretized Vale–Maurelli or optional VITA arm is used.
Binary symmetric (.5) and five-category symmetric (.2,.4,.6,.8) or skewed
(.45,.70,.85,.95) normal-quantile cuts; N=300,1000,4000 **per group**.
Coverage and descriptive global cells are single-group delta. Nested cells
use two independent identically distributed groups, configural versus equal
loadings in delta; a theta subset at N=1000 uses equal thresholds at H1 and
equal thresholds plus loadings at H0 to identify the metric comparison.
All cells have cross-loading zero, CFA model and 2000 production replicates.
Binary items are the pairwise-saturated first-stage control: exact and OPG
should agree asymptotically, even with non-Gaussian responses.

Targets are loading x2, threshold x2|t1 and (two-factor only) factor correlation.
For each population use a fixed draw of 1000000 observations, streamed in
10000-row chunks into pair tables. Reconstruct each pair's integer rows from
counts and call the library Stage-1 constructor; assemble its thresholds,
polychorics and diagonal OPG fitting weights into a population-only Stage-2
input. No first-stage estimator is reimplemented. The DWLS pseudo-true point
minimizes this weighted population discrepancy, not the generating parameters.
Repeat with an independent million-row draw; report target differences and both
F and RMSEA=sqrt(F/df). Smoke/pilot use 20000 rows per draw, explicitly a
coarse engineering approximation, unsuitable for coverage conclusions.
Nested restrictions hold at the limiting pseudo-true point by group symmetry;
the larger factor model need not hold exactly at the polychoric level.

## Arms and decision rules

Same draws and fits: exact policy IJ versus robust_ordinal_ij(first_stage='opg')
coverage; global policy All versus OPG spectrum All; nested SB/PEBA4/All
on exact versus OPG IJ parameter-space spectra with common observed Hessian
and restriction map. Global rejection is descriptive only, paired with
population misfit, and never gates. No policy fitting options are changed.
Report per-cell Wilson intervals, successful and attempted denominators,
paired exact-minus-OPG bootstrap intervals and failures. Coverage below 93%
at N>=300 or nested size outside [3%,7%] at N>=500 flags finite-sample work,
never a return to OPG. Gaussian cells are non-gating development controls.
Report the expected drift of OPG and consistency of exact honestly, including
an absence of that pattern. Tiny pilot rates are mechanics, not calibration.

## Seeds, compute and provenance

Population bases 217000001 and 227000001 (independent MC approximation draws);
smoke 717000001, pilot 1917000001, production 2017000001. Primary bases are
at least 100000000 from every earlier decisions/05 draw base and TASK-79's;
the second MC draw is paired population uncertainty, not a separate lane base.
Population seed adds 10000 times the full-grid population key index; study
seed adds 10000*cell_id+replicate. Bootstrap seeds do not generate study draws.
Smoke one replicate on representative coverage/global/nested cells; pilot two
replicates on a bounded selection covering all generators and both dimensions,
about ten minutes, <=4 cores, nice and one math thread each. Freeze summaries,
metadata, both population approximations, target differences, timing, failures,
paired intervals and partial pricing; unpriced cells remain explicit. Cell
fan-out uses the same seeds and strict source/package provenance; verify two
cells against the serial summaries. Raw rows and generated reports stay ignored.
Production uses single-threaded cells and bounded population memory; simbox
is unavailable until explicitly re-enabled by the user.
