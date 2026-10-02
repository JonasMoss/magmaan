# Inference capability inventory

This is the single inventory for 0.2.0 exit criterion 6. The initial rows cover
named reporting conventions; the full policy, estimation/domain, penalty and
verdict inventory remains in the active backlog. Evidence is component-specific:
an existing primitive is not automatically a checked reporting bundle.
The 2026-10-02 reporting audit below distinguishes C++, lab adapters and the
ordinary interface; it does not complete the wider exit-criterion inventory.

States: **validated** within the named checked slice; **limited validation**
when components exist but the required composition or regime is not fully
gated; **unsupported** when the reporting route returns a typed reason;
**inapplicable** when the convention does not define the component. Parity
establishes numerical compatibility, not calibration or policy adoption.
Unsupported describes the named API route, not the absence of a C++ algorithm
or a limitation of lavaan. In the next table it means **not yet supported in
ordinary magmaan compatibility reporting**.

## Ordinary reporting conventions

All routes use retained fits and do not optimize again. `convention = "magmaan"`
remains the default. Positive barrier penalties return `penalized`; failed
selected convergence verdicts return `not_converged`. Numerical failures retain
their reason per component. Score tests are inapplicable to lavaan reporting
bundles and are not replaced by policy score tests.

| Fitted setup / convention | Covariance | Global test | Nested test | Evidence and limits |
| --- | --- | --- | --- | --- |
| Complete continuous ML / ML | Validated | Validated: standard | Validated: standard difference | Single-group CFA; grouped loading/intercept invariance; random-X regression; nested fixed-zero covariance; matched means/group order |
| Complete continuous ML / MLM | Validated: expected empirical sandwich | Validated: Satorra-Bentler | Validated: SB2001 | Same global/covariance slices; single-group nested difference |
| Complete continuous ML / MLR | Validated: observed exact-score sandwich | Validated: YB-Mplus H1-minus-H0 trace | Validated: SB2001 with MLR scales | Same global/covariance slices; single-group nested difference; independent C++ saturated-distance/H0 numerical-score trace |
| All-ordinal DWLS / DWLS | Validated: NACOV sandwich | Validated: unscaled statistic, p-value unavailable | Unsupported | Delta/theta, single group and two unequal groups with loading equality; per-group n minus one |
| All-ordinal DWLS / WLSMV | Validated: NACOV sandwich | Validated: scaled-shifted | Unsupported | Same slices; the C++ delta engine and scaled-shifted reducer exist, but the default nested reporting bundle needs wiring and gates |
| All-ordinal ULS / ULS | Validated: NACOV sandwich | Validated: unscaled statistic, p-value unavailable | Unsupported | Single-group delta/theta; grouped reporting remains limited validation |
| All-ordinal ULS / ULSMV | Validated: NACOV sandwich | Validated: scaled-shifted | Unsupported | Single-group delta/theta; grouped reporting remains limited validation |
| All-ordinal WLS / WLS | Validated: standard covariance | Validated: standard | Unsupported | Single-group delta/theta; grouped reporting remains limited validation |
| FIML / ML or MLR | Unsupported | Unsupported | Unsupported | Covariance/global and nested engines already exist in C++ and the lab; ordinary bundles await composition and convention-matched gates |
| FIML / MLM | Inapplicable | Inapplicable | Inapplicable | Rejected: MLM's missing-data handling would change estimation |
| Continuous GLS, ULS, WLS; ML2S; mixed ordinal | Unsupported | Unsupported | Unsupported | No checked ordinary compatibility composition; see lab inventory below |

The whole-bundle installed-lavaan gates are
[`test_conventions.R`](../../r-magmaan/tests/testthat/test_conventions.R).
They compare full covariance matrices by parameter/group keys, global statistics,
df, scaling, shift and p-values, and nested defaults in either model order.
Intervals, defined parameters, deferred computation, serialization, caching,
incompatible-regime rejection and preservation of the default policy are checked.
The covariance and independent MLR algebra gates are in
[`policy_test.cpp`](../../cpp/tests/unit/policy_test.cpp).

The checked classical fitting domain is unrestricted covariance with affine
equalities and no active bounds. Interior PSD fits can use the same retained-fit
recipes; broader PSD/boundary compatibility is limited validation, and the
interior-population assumption is printed. This table makes no claim that a
constrained estimate equals lavaan's unconstrained optimum.

Compare matching structural and estimation settings in lavaan, including
`meanstructure = TRUE` and `fixed.x = FALSE` for ordinary continuous fits.
Missing-data deletions, estimation weights, parameterization, group order and
convergence rules remain fitting choices. Reported bundles target lavaan 0.7.2.

## Existing lab components and remaining composition gates

| Setup | Existing covariance/global primitives and gates | Nested primitives and remaining gates |
| --- | --- | --- |
| Complete ML | `ntml_covariance`, `ntml_score_sandwich`, expected UGamma, `fiml_robust_mlr` complete-data trace reduction. SE/SB fixtures: `inference_golden_test.cpp`, `multigroup_inference_golden_test.cpp`; complete bundles above | `lr_test_satorra_bentler2001`/2010 and restriction-map Satorra-2000 exist. SB2001 bundles above are checked; SB2010 is not an ordinary bundle |
| FIML | `fiml_observed_information`, `fiml_robust_mlr` and spectrum primitives exist; `fiml_golden_test.cpp` checks robust SEs, MLR statistics, scales and H1/H0 traces. Lab `vcov()` exposes observed-information and observed-sandwich routes. Full ordinary missing-data bundles remain unsupported | Restriction-map and scalar SB2001/2010 engines and lab adapters exist. The scalar engines derive their single-model scales from a residual-projector spectrum, so their names alone do not establish agreement with lavaan's MLR trace recipe. A frozen grouped scalar-invariance fixture gates the lavaan-convention Satorra-2000 mean-scaled route, not the ordinary default MLR difference bundle |
| All-ordinal DWLS/ULS/WLS | `robust_ordinal`; `ordinal_golden_test.cpp` gates component SEs/tests. Ordinary composer adds lavaan reporting normalization and bundled method selection | `lr_test_satorra2000_ordinal` supports exact/delta and returns scaled-shifted results; lab `robust_nested_lrt()` already dispatches to it. Existing grouped DWLS/theta golden comparisons gate the delta mean-scaled result with lavaan normalization. Default WLSMV/ULSMV scaled-shifted reporting, its normalization and whole-bundle gates remain to do |
| Continuous GLS/WLS/ULS | `robust_continuous_ls` with explicit weight/Gamma; `ls_golden_test.cpp` and weighted-inference tests. The ULS/DWLS Browne-residual NT reporting recipe is not composed here | Continuous weighted Satorra-2000 primitives exist; no checked ordinary lavaan-default bundle |
| ML2S | `two_stage_em_ml_inference` and weighted Stage-2 inference exist | Restriction-map and scalar SB2001/2010 engines and lab dispatch exist; scalar methods are NT-only. Naive unstructured-information compatibility and a checked default two-stage reporting bundle remain to do |
| Mixed ordinal | `robust_mixed_ordinal` and mixed component gates exist | Mixed Satorra-2000 primitives exist; ordinary compatibility remains unsupported |

No policy covariance is described merely as "lavaan with the weight correction
turned on": policy and compatibility can also differ in bread, meat, sensitivity
and calibration. Isolating weight influence requires matched lab recipes.

## Reporting gap audit (2026-10-02)

This is a static audit of implementations, dispatch and existing tests. It adds
no numerical validation or policy decision. The existing component gates should
be reused when filling an interface gap; missing bundle tests do not justify
reimplementing their algorithms.

| Surface | Existing implementation or evidence | Remaining work and owner |
| --- | --- | --- |
| Ordinal compatibility `anova()` | C++ ordinal Satorra-2000, exact/delta restriction maps, scaled-shifted reducer and lab adapter | Compose the retained-fit ordinary route; match per-group reporting normalization; gate WLSMV/ULSMV defaults and probe plain DWLS/ULS/WLS nested reporting separately. Backlog: extend checked reporting conventions |
| FIML compatibility `vcov()`, `confint()`, `summary()`, `anova()` | Standard/robust covariance, global MLR and several nested engines, with component and selected fixture gates | Bind ML/MLR to the exact covariance/global/default nested recipes, retain H1/fit context, and add complete bundles for grouped and incomplete data. Backlog: extend reporting conventions; pin FIML robust conventions |
| Existing complete-ML compatibility tests | Single-group nested defaults; grouped covariance/global tests | Grouped nested defaults with loading/intercept equalities and released means are not gated through the new reporting bundle. Backlog: close reporting-bundle validation gaps |
| Existing ordinal compatibility tests | Single-group delta/theta for DWLS/ULS/WLS; unequal-group loading equality for DWLS/WLSMV | Grouped ULS/ULSMV/WLS, additional threshold/mean restrictions and the new nested routes need their own whole-bundle gates. Backlog: close reporting-bundle validation gaps |
| Covariance domains and failures | Classical affine/no-active-bound slice; convergence and positive-penalty refusals; PSD metadata | Interior PSD, boundary endpoints, active bounds, degenerate scales and group-count normalization need API-specific evidence or explicit exclusions. Backlog: reporting validation and wider capability inventory; PSD hardening remains 0.3.0 |
| Ordinary FIML policy | C++ likelihood-score, covariance and LR-spectrum primitives | `policy_inference_fiml`, `policy_nested_fiml`, reusable contexts and R dispatch are missing. This is separate from lavaan compatibility. Backlog: compose the FIML policy |
| Ordinary all-ordinal DWLS policy | IJ covariance, fixed-weight global/nested spectra and estimated-weight profile-LR primitives | Compose the adopted policy and settle its nested recipe/calibration; a joint nested score primitive is absent and remains typed unavailable for 0.2.0. Backlog: all-ordinal DWLS policy tasks |
| Parameter confidence intervals | Ordinary Wald intervals and defined-parameter delta SEs; C++ and lab profile-test/CI engines for ML, FIML, ordinal and other routes | Ordinary `confint(test = "lr")` is planned but rejected. Its refit, inversion, calibration and compatibility interaction need an explicit contract and adapter. Backlog: planned LR interval interface, unscheduled |
| Lab estimated-weight comparison | Fixed/estimated-weight primitives; explicit switches on several score, residual, profile and fit-measure routes | `frontier_rbm()` has no off switch; defaults vary between routes. Backlog: lab correction switches/defaults, unscheduled |
| Compatibility naming | Current ordinary selector is `convention`; historical bundle names are accepted only for their compatible fits | `lavaan_compat = NULL` and explicit compatibility output are proposed, not adopted. Decide the name before changing arguments, attributes, caches and docs. Backlog: compatibility selector decision |
| Additional lavaan variants and secondary setups | Mixture reducers, continuous-LS, ML2S and mixed-ordinal components exist | MLMV/MLMVS, WLSM/WLSMVS, ULSM and further LS/two-stage/mixed bundles have no ordinary selector/composer gates. Inventory the exact recipes if a consumer needs them; their existence does not add a 0.2.0 requirement |

The missing parts above must be separated from deliberate interface limits:
ordinary inference uses random covariates; a report cannot change `fixed.x`,
deletion, weights or parameterization. Score tests are inapplicable to these
lavaan bundles, and plain ordinal DWLS/ULS global p-values are absent by recipe.
Standalone lab fit measures, residual diagnostics and standardized estimates
are not implicitly promised by `summary()`. Existing two-level, SAM, composite
and noniterative interfaces retain their gates without an expansion programme.

Source checks: [`conventions.cpp`](../../cpp/src/api/conventions.cpp),
[`conventions.R`](../../r-package/R/conventions.R),
[`nested_test.R`](../../r-package/R/nested_test.R),
[`context.R`](../../r-package/R/context.R),
[`scores.R`](../../r-package/R/scores.R),
[`methods.R`](../../r-magmaan/R/methods.R),
[`ordinal_golden_test.cpp`](../../cpp/tests/golden/ordinal_golden_test.cpp),
[`fiml_golden_test.cpp`](../../cpp/tests/golden/fiml_golden_test.cpp), and profile
tests in [`constraints_test.cpp`](../../cpp/tests/unit/constraints_test.cpp),
[`fiml_test.cpp`](../../cpp/tests/unit/fiml_test.cpp) and
[`ordinal_test.cpp`](../../cpp/tests/unit/ordinal_test.cpp).
