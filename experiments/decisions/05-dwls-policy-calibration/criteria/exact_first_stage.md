# Exact first-stage all-ordinal reconfirmation (lane A)

Registered 2026-10-05 before smoke/pilot. Merger registration is required before
production. TASK-84 owns latent non-normality; this lane uses Gaussian copulas.
Earlier lanes, seeds and evidence remain unchanged. The current policy uses
exact first-stage rows, estimated-weight IJ covariance, global All and nested
SB/PEBA4. Flags call for finite-sample work, never a return to OPG.

## Design and arms

Reuse the amended dwls_policy.md populations and stable cell IDs: 52 global
null, 12 global power, 32 nested null, 8 nested power and 42 coverage cells.
Use 2000 replicates per null/coverage cell, 1000 per power cell. N is total,
split equally across groups. The corrected theta pair has equal thresholds
at H1 and equal thresholds plus loadings at H0. Coverage targets are the
registered loading, threshold, correlation and structural path, with omitted
cross-loading 0, .2 or .4. Population targets remain Monte Carlo approximations
from 100000 rows per group, using a fresh stable population-design seed.

On each identical draw and fitted point, compare policy exact with OPG:
- Global: policy All versus All on robust_ordinal's OPG NACOV spectrum.
- Coverage: policy covariance versus robust_ordinal_ij(first_stage='opg').
- Nested: policy SB/PEBA4 versus SB/PEBA4 on the parameter-space spectrum
  reconstructed from the same observed H, exact restriction A and OPG IJ
  covariance. Include All on each spectrum as a non-gating TASK-81 comparator.
No fitting options override the baseline policy. Library fit$converged is the
judge. Record errors, nonconvergence and unavailable arms independently.

## Seeds

Bases: population 117000001, smoke 317000001, pilot 517000001,
production 1817000001. These are mutually at least 100000000 apart and at
least that far from all earlier 817xxxxxx and mixed 1017/1217/1417/1617xxxxxx
bases. Draw seed = base + 10000*cell_id + replicate. Population-design key
IDs come from the full coverage grid, independent of selected cells.
Bootstrap seed = 617000001 + 10000*cell_id + pair index; 2000 paired
replicate resamples, percentile 95% intervals. Bootstrap seeds do not generate
study draws. Replicate IDs are stable under fan-out.

## Reporting and rules

Report each cell/target/arm separately: attempted-denominator rejection or
coverage with Wilson intervals, successful-arm rates, failures and unavailable
arms. Flag policy null rejection outside [3%,7%] for N >= 500 and policy IJ
coverage below 93% for N >= 300. Report paired exact-minus-OPG differences
and bootstrap intervals among jointly available draws, with paired and excluded
counts. Also report attempted-denominator differences (failure counts remain
visible). Power is descriptive, with arm-specific matched-null size adjustment.
Never pool away a regression; pilot flags are engineering diagnostics only.

## Compute boundary and artifacts

Smoke: one replicate in a representative subset covering all three families.
Pilot: two replicates per selected pricing cell (document selection), about ten
minutes, at most four cores with one math thread each and nice priority.
No production or Modal launch. Freeze pilot summaries, metadata, population
targets, failures, paired intervals, timing and partial production pricing.
List unpriced cells rather than extrapolating an unmeasured full-grid cost.
Preemption-safe cell attempts publish COMPLETE only after draws and gates;
resume and combine must reject changed source/package/seed/count provenance.
Local two-cell fan-out must reproduce summaries (excluding timing fields).
