# Threshold-invariance calibration — registered criteria

Status: registered by the merger on 2026-10-05, before production. Smoke and pilot
are development timing/availability controls, never confirming evidence.
No default change is authorized by this study.

Decision: does the current DWLS moment-nested threshold-step law have acceptable
null rejection for configural H1 versus thresholds-equal H0? Baseline fits use
`fit_model(..., estimator="DWLS")` without optimizer overrides and the actual
`fit$converged` verdict. The policy is `policy_nested()` (the ordinary `anova()`
composer). Report SB, PEBA4 and All from the same retained spectrum. Comparator:
delta-coordinate `convention_nested(...,"WLSMV")`, Satorra-2000 scaled-shifted;
retain typed unavailable reasons rather than substituting another test.

Grid: two independent groups; one factor/six indicators or two factors/twelve
indicators; Gaussian factors with correlation .3, loadings .7, unit response
variance; categories 4, 5, 7; symmetric normal-quantile cuts or cuts shifted .65;
N per group 250, 500, 1000; theta and delta parameterizations; correct model or
omitted cross-loading .3 on item 2 onto factor 2, equal in both groups. For the
one-factor misspecified family (where a cross-loading is undefined), use an
omitted residual correlation .3 between items 1 and 2. The merger accepts this
extension (2026-10-05). Identical distributions across groups define the
pseudo-null even under misspecification. Three-category controls record the
zero-df `equivalent_models` reason only; they never gate calibration or power.

Power changes item 2's second threshold by +.3 in group 2. Each power cell has
an otherwise identical null cell. Size-adjusted power uses the matched null's
empirical fifth percentile of p-values (type-8 quantile); report matched counts.
Never pool rates across families or hide unavailable/nonconverged draws.

Production: 2,000 null and 1,000 power draws per cell. Seed bases: smoke
818130001 (2 draws), pilot 818140001 (20 draws), production 818150001.
Seed = base + 10000*cell_id + replicate. Full Cartesian cell IDs remain stable
when selecting cells or distributing them to Modal. Pilot may use a subset if
full-grid timing exceeds ten minutes. Production starts only after the merger
commits the criteria, with fresh output. Post-pilot design audit (2026-10-05; arithmetic correction): the requested
bases are 10,000 apart, so smoke cell 3 shares seeds with production cell 1
(and pilot cell 2 also shares those seeds). This is a registration blocker. The merger must amend the phase bases to disjoint ranges before
production; the current requested bases are retained for the development runs.

Registered rule (board decision, 2026-10-05): for each supported null cell with
N >= 500 per group, flag each policy arm whose rejection at alpha .05 is outside
[.03,.07]. Report Wilson 95% intervals, attempted/successful counts, all failures
and spectra. A flag requests investigation and fresh confirmation; it never
switches the inference recipe. Unsupported slices remain open; pilot flags
are diagnostics, not decisions. All and WLSMV are comparators, not rule-gating
arms. The default remains provisional until the registered production evidence.

Cost uses pilot CPU seconds by factor/parameterization family extrapolated over
the full grid. It excludes container build/startup and is a planning estimate.


## Registration amendment (merger, 2026-10-05, before production)

The production seed base is 830150001 (seed = base + 10000 * cell_id +
replicate). With at most 384 cells its range, 830160001 to 834,0xx,xxx, is
disjoint from the smoke (818130001) and pilot (818140001) ranges, which keep
their development values. The one-factor omitted residual correlation .3
between items 1 and 2 is accepted as the misspecified family for one factor.
The WLSMV comparator returns a typed not_nested reason for this moment-nested
step and is retained as such. Everything else is registered as written above.


## Amendment (merger, 2026-10-05, after production, before confirmation)

Written after seeing the production results
(`results/production-2026-10-05`). Production flagged 22 policy arm-cells,
all with seven categories: SB 7.05-7.95% and PEBA4 7.05-7.5% at N >= 500 per
group. SB and PEBA4 rise with the number of categories (at N >= 500, medians
about 5.0% with four, 5.5% with five and 6.5-7.1% with seven categories),
while the registered All comparator stayed within 3.5-6.0% in every null cell.
Size-adjusted power was equal across arms. As registered, the flags call for
investigation and fresh confirmation and do not switch the recipe. Adopting
All for the nested DWLS reference, as the user chose for the global DWLS test
(decisions/05, 2026-10-03), is a user decision recorded on the board.

To give that decision fresh evidence, a confirmation reruns all 144 null cells
with four, five or seven categories on fresh draws: mode `confirm`, seed base
840150001 (seeds 840160001 to about 843,850,000, disjoint from every earlier
range), 2,000 draws per cell, the same arms and summaries. Criterion fixed now:
All rejection within [3%, 7%] in every null cell with N >= 500 per group. SB and
PEBA4 are reported under the original rule. Choosing All from the production
rates is post hoc; the confirmation does not tune anything.
