# Threshold-invariance calibration — registration draft

Status: draft for merger review and commit before production. Smoke and pilot
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
omitted residual correlation .3 between items 1 and 2. This explicit design
extension needs merger review. Identical distributions across groups define the
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
