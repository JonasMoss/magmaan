# Association-ML policy calibration

Registered before any study smoke or pilot, 2026-10-05 (TASK-32.5).
Question: does the lab exact Stage-1 IJ/observed Stage-2 recipe calibrate
well enough for ordinary exposure? Ordinary inference remains unavailable.
No production is authorized by this card. The approximately ten-minute pilot
is authorized, with at most two workers, nice priority and one math thread.

## Design and targets

One factor with six indicators and two factors with twelve indicators;
primary loadings (.65,.70,.75,.80,.70,.75), factor correlation .3, unit
response variances. Independent Gaussian or standardized centered chi-square
(df 2) factors and errors, followed by factor-covariance Cholesky transport,
then direct thresholding at normal-quantile cuts. Binary cuts .5 or .85;
five-category cuts (.2,.4,.6,.8) or (.45,.70,.85,.95). Total N=300,1000,4000.
Correct generating CFA and an omitted .3 cross-loading on x7 from f1
(two-factor only). One-factor misspecification instead adds a .3 residual
correlation between x5 and x6; this retains both dimensions in the robustness
panel and is declared before seeing results. Residual variances maintain unit
response variances, including the cross-loading. Identical independent groups
split total N equally for configural versus equal-loadings nested tests.
All cells use delta, fixed response scales and saturated thresholds.

Each population has two independent 1,000,000-row draws (two million total),
streamed in 10,000-row chunks into pair tables. Reconstruct integer rows from
pair counts and call library Stage 1, then fit population association ML and
DWLS separately. No polychoric estimator is reimplemented. Record both
pseudo-true loading x2, threshold x2|t1, and factor correlation where present,
population discrepancies, Stage-1 values, timings and target differences.
Smoke alone uses two 20,000-row engineering approximations. Pilot/production
use full population draws. The first draw fixes coverage and MI targets;
the second measures approximation uncertainty, never selects a target.

Every single-group cell provides covariance coverage, global tests and MI;
every two-group cell provides the metric nested test. MI releases one fixed
loading f1=~x2: fix at the unrestricted population loading (size), then at
that loading plus .15 (power). This replaces the proposed zero residual-row
null because an omitted cross-loading or non-Gaussian polychoric target need
not leave any chosen residual row zero at its pseudo-true point. The fixed
loading null holds at the unrestricted pseudo-true value under all generators.
Group symmetry supplies the nested null under misspecification. Population
approximation error must be assessed before interpreting production MI size.

## Arms, judge and rules

Association ML: exact joint IJ covariance with observed sensitivity; global
All with SB/PEBA4 comparators; nested parameter-space All/SB/PEBA4; robust
one-df MI. Same-draw DWLS ordinary policy covariance, global All and nested
references are separate estimand comparators; no paired estimand contrasts.
Use default fitting options and library fit$converged. No truth starts or
study convergence tolerance. Exceptions, nonconvergence and unavailable
inference are recorded with seeds and remain in attempted denominators.
Report successful denominators too, plus Wilson 95% intervals and failures.

Coverage below 93% at N>=300 flags finite-sample work. Size outside [3%,7%]
at N>=1000 flags global tests only for Gaussian correctly specified models,
and nested/MI whenever the pseudo-true null holds. Non-Gaussian or misspecified
global rejection is descriptive, alongside population misfit. Power is
reported only. Report families and all comparator losses separately, never
pool away a flag. Flags never select OPG, expected bread or another inconsistent
recipe. Pilot flags are mechanical annotations, not default decisions.
Production confirmation uses 2,000 replicates per cell and fresh seeds; no
ordinary exposure decision before production and review of MC uncertainty,
availability and population approximation errors.

## Seeds, artifacts and provenance

Population bases 617000001 and 1117000001; smoke 1317000001, pilot 1717000001,
production 2117000001. Each is at least 100000000 from all study-draw bases
registered in decisions/05-07 (bootstrap-only bases are not study draws).
Population seed adds 10000 times the stable full-grid population index;
chunk adds its zero-based index; replicate seed adds 10000*cell_id+replicate.
Distinct populations, modes and cells have disjoint draw ranges. Stable full
grid IDs survive selections and fan-out. Frozen CSV evidence has source,
criteria, package binary and R-wrapper fingerprints, command and versions.
Combine checks COMPLETE, exact provenance, population targets, seed/count
and unique cell/replicate keys. Incomplete attempts never resume silently.

Smoke: one draw on a one-factor Gaussian symmetric binary single-group cell
and its nested cell. Pilot: two draws on two-factor skewed five-category
chi-square2 misspecified single-group and nested cells. This bounds setup
while pricing a difficult family; all unpriced cells are listed, with no
full-grid extrapolation from unrelated cells. Verify two-cell fan-out against
serial smoke summaries. Freeze summary, failures, timing, cost, cells,
metadata, population targets/Stage 1/uncertainty and population timing.
Raw draws and serialized checkpoints remain ignored. Production is deferred;
simbox remains unavailable until the user explicitly enables it.

## Registration (merger, 2026-10-05, after the pilot, before production)

Registered as written above. The pilot priced two difficult cells (0.36 and
0.54 s per draw plus about 55 s of population setup per cell); the full grid is
planned at roughly 50-60 CPU-hours, a planning figure rather than a registered
quantity. Production runs with seed base 2117000001 and 2,000 replicates per
cell on the maintainer's simbox workstation after the queued decisions/05 and
decisions/07 runs, with single-threaded cells, the study's per-cell runner and
its provenance-checked combine; the executor is recorded in the metadata.

## Post-hoc amendment, 2026-10-08: MI population approximation

Written after seeing the production results; approved by the user on
2026-10-08 (TASK-128.3). The registered rules above remain unchanged.
For every single-group MI size cell, use the ML loading targets from population
draws 1 and 2 and set Delta = |lambda1 - lambda2|, treated as a conservative
bound on draw-1 approximation error. Divide Delta by the median loading SE
among successful association_ij loading rows in that cell. With
ncp = (Delta / SE)^2, compute the upper-tail probability above the nominal
5% central chi-square threshold for a one-df noncentral chi-square.
The approximation is negligible for this check if the maximum implied size
across these cells is at most 5.5%. Ordinary association-ML MIs may be exposed
only after this check passes. This approximation check does not replace the
registered empirical size, availability or Monte Carlo uncertainty review.
