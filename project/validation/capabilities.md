# Inference capability inventory

This is the single inventory for 0.2.0 exit criterion 6. The current rows cover
named reporting conventions and lab/C++ MI and equality-release components;
the full policy, estimation/domain, penalty and verdict inventory remains in
the active backlog. Evidence is component-specific:
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
| Complete continuous ML / MLM | Validated: expected empirical sandwich | Validated: Satorra-Bentler | Validated: SB2001 | Same global/covariance slices; single-group and grouped loading/intercept/mean nested differences |
| Complete continuous ML / MLR | Validated: observed exact-score sandwich | Validated: YB-Mplus H1-minus-H0 trace | Validated: SB2001 with MLR scales | Same global/covariance slices; single-group and grouped loading/intercept/mean nested differences; independent C++ saturated-distance/H0 numerical-score trace |
| All-ordinal DWLS / DWLS | Validated: NACOV sandwich | Validated: unscaled statistic, p-value unavailable | Unsupported | Delta/theta, single group and two unequal groups with loading equality; per-group n minus one |
| All-ordinal DWLS / WLSMV | Validated: NACOV sandwich | Validated: scaled-shifted | Unsupported | Same slices; the C++ delta engine and scaled-shifted reducer exist, but the default nested reporting bundle needs wiring and gates |
| All-ordinal ULS / ULS | Validated: NACOV sandwich | Validated: unscaled statistic, p-value unavailable | Unsupported | Delta/theta, single group and two unequal groups with loading equality; per-group n minus one |
| All-ordinal ULS / ULSMV | Validated: NACOV sandwich | Validated: scaled-shifted | Unsupported | Delta/theta, single group and two unequal groups with loading equality; per-group n minus one |
| All-ordinal WLS / WLS | Validated: standard covariance | Validated: standard | Unsupported | Delta/theta, single group and two unequal groups with loading equality; per-group n minus one |
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
equalities and no active bounds. This table makes no claim that a constrained
estimate equals lavaan's unconstrained optimum.

| Retained-fit domain slice | Reporting validation and limits |
| --- | --- |
| Interior PSD, zero penalty | **Limited validation** for ordinary `vcov`, `confint`, `summary` and `anova` compatibility bundles. The retained-fit recipes exist, but there is no domain-specific whole-bundle lavaan gate; unrestricted parity above does not validate PSD optimization or its verdict. |
| PSD boundary endpoint | **Limited validation** for those APIs: no boundary-specific covariance/global/nested compatibility gate or boundary reference law. Classical interior-population inference must not be described as boundary-calibrated. PSD hardening remains 0.3.0. |
| Active parameter bound | **Limited validation** for those APIs: the checked affine slice excludes active bounds; no active-set covariance or nested reference-law gate. A finite classical result is not evidence of bound-adjusted inference. |
| Degenerate robust scale | **Limited validation** outside the exact saturated gate. `scaled saturated tests and penalized fits keep typed unavailability` gates `summary(..., convention = "MLR")`: covariance remains finite while the global test returns `saturated`; other zero/nonfinite scale endpoints have no whole-bundle gate and are not promoted to validated. |
| Positive barrier penalty | **Unsupported** compatibility covariance/nested reporting, gated by the same test through `vcov(..., convention = "MLM")` and `anova(..., convention = "MLR")`: typed `penalized`, no fallback. |

The grouped complete-ML nested gate calls `lavaan::lavTestLRT` defaults and
checks explicit `standard` (ML) or `satorra.bentler.2001` (MLM/MLR), in both
model orders. Loading, intercept and latent-mean equalities are released in
successive comparisons. Grouped ordinal gates call `lavaan::lavInspect("vcov")`
and `lavaan::lavInspect("test")`: `robust.sem` covariance with `scaled.shifted`
tests for WLSMV/ULSMV, `robust.sem` with unscaled `standard` tests for DWLS/ULS
(no p-value), and `standard` covariance/test for WLS. Both delta and theta
parameterizations use unequal groups and each group's n_g minus one reporting
normalization, with loading equality. Grouped loading/threshold equality is **limited validation** for covariance,
intervals and global reporting across DWLS/ULS/WLS and MV bundles: a live
lavaan 0.7.2 delta probe finds a different free-row contract (magmaan frees
second-group residual variances; lavaan frees scaling factors and intercepts).
Thus no like-for-like full covariance gate is retained for that slice; theta
threshold equality also remains ungated. This requires a model-contract fix,
not a covariance-key formatting workaround. Ordinal nested bundles remain
unsupported, including threshold-restriction comparisons.

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
| Existing complete-ML compatibility tests | Single-group and grouped loading/intercept/mean nested defaults; grouped covariance/global tests | Existing complete-data bundle gates closed by task-7.1; additional regimes require separate evidence |
| Existing ordinal compatibility tests | Single/grouped delta/theta for DWLS/ULS/WLS and MV reporting, loading equality | Threshold equality is explicitly limited above; nested routes and further mean restrictions still need their own whole-bundle gates. Backlog: extend checked reporting conventions |
| Covariance domains and failures | Classical affine/no-active-bound slice; convergence and positive-penalty refusals; PSD metadata | Domain-specific limits are inventoried above; unequal-group normalization is gated for existing ordinal bundles. Wider capability inventory and PSD hardening remain separate work (hardening: 0.3.0) |
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

## MI and equality-release score components

The following 23 estimator/weight rows cover 92 component cells. A named gate
validates its stated numerical slice; explicit refusals are unsupported, and
estimated quadratic-weight channels for likelihood fits are inapplicable.
Mixed ordinary MI remains **limited validation** because its oracle scale
discrepancy is unresolved. These states describe the lab/C++ component routes;
ordinary magmaan inference and lavaan reporting remain governed by their own
composition gates above.

This inventory covers the lab/C++ one-parameter score surface for 0.2.0.
An ordinary MI here is the unscaled score statistic, not the ordinary R
package's inference policy. Each robust cell covers both MI and one-at-a-time
equality releases unless its gate explicitly says MI only. Fixed weight uses
the fitted W; estimated weight adds the influence of that same recorded recipe.
This is component evidence, not calibration or an ordinary-user recommendation.

### Gates

The labels below give a file and the exact test name (or a stated group of
names); a cell lists its C++ gate first and its R gate second.

- **C-ML**: `cpp/tests/unit/score_test.cpp`, `modification_indices: complete ML reports finite fixed-row tests`, `score_tests: equality releases are reported in constrained ML models`, and `modification_indices: absent-row MI / EPC match lavaan modindices`.
- **C-GOLD**: `cpp/tests/golden/score_golden_test.cpp`, `score/modification-index goldens match lavaan fixed-row and equality-release targets` (ML, observed FIML, ULS, ordinal/mixed DWLS, GLS and ADF/WLS).
- **C-ROB-ML**: `cpp/tests/unit/score_robust_test.cpp`, `frontier robust MI: model-implied Expected bread reduces to NT`, `frontier robust MI: observed bread is finite and positive`, `frontier robust score test: equality release reduces to NT`, `frontier robust MI multi-group: model-implied Γ_NT meat reduces to NT`, and `frontier robust MI multi-group: empirical raw-data path scales, finite`. `cpp/tests/golden/score_robust_golden_test.cpp`, `robust release-score matches the lavaan-internals oracle (c != 1)`, pins the robust MLM release.
- **C-LS**: `cpp/tests/unit/score_robust_test.cpp`, `frontier robust LS MI: GLS weight + Gamma_NT(S) meat reduces to NT`, `frontier robust LS score test: GLS equality release reduces to NT`, `frontier robust LS MI: DWLS raw path scales; sandwich matches primitives`, and `frontier robust LS score test: raw and supplied Gamma agree`.
- **C-RECIPE**: `cpp/tests/unit/score_robust_test.cpp`, `frontier robust LS MI: an estimated-weight recipe must reproduce the fitting weight` and `estimated-weight IJ mode follows the recorded weight recipe`.
- **C-ORD**: `cpp/tests/unit/score_robust_test.cpp`, `frontier robust ordinal MI: WLS + NACOV meat reduces to ordinary`, `frontier robust ordinal score test: WLS equality release reduces`, `frontier robust ordinal MI: DWLS scales against the NACOV meat`, `frontier robust ordinal MI: estimated-weight DWLS shifts the scaling`, and `ordinal score rank: latent units and nearby points preserve candidates`. The primary ordinal rank and weight-scale R gates are R-ORD.
- **C-ORD-REFUSE**: `cpp/tests/unit/score_robust_test.cpp`, `frontier robust ordinal inference: estimated weight refuses NT, DLS and supplied weights`.
- **C-MIX**: `cpp/tests/unit/score_robust_test.cpp`, `frontier robust mixed ordinal: WLS reduces, DWLS finite, ULS rejected` and `frontier robust mixed ordinal multi-group: WLS reductions cover MI and score`.
- **C-2S**: `cpp/tests/unit/score_robust_test.cpp`, `frontier ML2S MI and releases reduce to complete-data robust tests` and `frontier ML2S MI: naive statistic plus Stage-1 scaling under missing data`.
- **R-ML**: `r-package/tests/testthat/test-score-rank.R`, `MI tables exclude identification releases across indicator units`; `test_wls_robust_covariance.R` supplies related weight and covariance guards. `test_weight_recipe_inference.R`, `complete ML MI and equality releases share ordinary and robust coordinates` gates complete-data ordinary/fixed-robust releases and the estimated-weight R refusal.
- **C-FIML-ROB**: `cpp/tests/unit/score_robust_test.cpp`, `frontier FIML robust MI: unscaled mi matches the non-robust FIML MI`, `frontier FIML robust score test: equality release runs and is finite`, and `frontier FIML robust MI and score tests support multi-group raw blocks`; `cpp/tests/golden/score_robust_golden_test.cpp`, `robust FIML release-score matches the lavaan-internals oracle (MLR)`.
- **R-SUPPLIED**: `r-package/tests/testthat/test_wls_robust_covariance.R`, `WLS robust MI and releases preserve empirical covariance and fitting W` and `WLS covariance choices fail explicitly when unavailable` (single/unequal groups, means, shared labels and supplied diagonal/full W).
- **R-FIML**: `r-package/tests/testthat/test_fiml_robust_score.R`, `FIML robust MI and releases agree across retained and explicit data` and `FIML robust score wrappers reject incompatible conventions`.
- **R-RECIPE**: `r-package/tests/testthat/test_weight_recipe_inference.R`, `estimated-weight inference reads the recorded continuous recipe`, `release score tests read the recorded recipe`, and `supplied weights are used as fitted but have no estimated-weight recipe`.
- **R-LS-MEAN**: same file, `continuous LS MI and releases preserve means and equality constraints`.
- **R-ORD**: `r-package/tests/testthat/test_ordinal_score_rank.R`, `ordinal ordinary and robust ranks agree across weight scales` and `ordinal fixed rows and equality releases use the same rank and moment scale`.
- **R-ORD-RECIPE**: `r-package/tests/testthat/test_weight_recipe_inference.R`, `ordinal NT, DLS and supplied weights refuse the weight influence`.
- **R-2S**: same file, `two-stage MI and releases reduce to complete-data robust tests`, `two-stage MI under missing data uses the recorded Stage-2 weight`, `two-stage unequal groups reduce to complete-data robust tests`, and `two-stage MAR MI and releases retain positive Stage-1 scaling`.

- **C-LS-MATRIX**: `cpp/tests/unit/score_robust_test.cpp`, `continuous LS recipe matrix gates means constraints and ordinary score reductions` (ULS, NT, DWLS, ADF and DLS; ordinary and fixed/estimated robust MI and releases).
- **C-ORD-MATRIX**: same file, `ordinal MI recipe matrix gates ordinary releases and typed refusals` (ULS, DWLS, WLS, NT, DLS and supplied W; both ordinary statistics and both robust channels). The association-tagged estimates assert `UnsupportedInference` in all four workers; the genuine association fit is also gated by `cpp/tests/unit/api_sem_test.cpp`, `api ML dispatches ordinal associations without granting Gaussian inference`.
- **C-MIX-MATRIX**: same file, `mixed ordinal MI recipe matrix gates ordinary cells and typed robust refusals` (ULS, DWLS and WLS; ordinary MI/releases, fixed robust DWLS/WLS, ULS and estimated robust refusals).
- **R-ORD-MATRIX**: `r-package/tests/testthat/test_weight_recipe_inference.R`, `ordinal recipe matrix gates releases and estimated-weight refusals` (all six recipes, both MI and releases; association-ML refusals).
- **R-MIX-MATRIX**: same file, `mixed ordinal MI matrix gates fixed weights and explicit refusals` (DWLS/WLS MI and releases; R ULS fitting refusal).

### Cells

`UnsupportedInference` names the core error kind; the R binding includes it in
its error message where available. ML has no second-stage estimated weight.
Ordinal ULS has no weight influence, so its estimated channel reduces to fixed.
Each cell names an exercised component gate or the actual refusal. ML/FIML
have no estimated quadratic-weight core operation; their R-level argument
guards reject attempts to request one. Mixed ordinal currently reports
`PostError::NumericIssue` for unavailable robust choices, not
`UnsupportedInference`; the matrix records that behavior without relabeling it.

| Fitting estimator / weight | Ordinary MI | Ordinary release | Robust fixed weight | Robust estimated weight |
| --- | --- | --- | --- | --- |
| Complete ML / likelihood | C-ML, C-GOLD / R-ML | C-ML, C-GOLD / R-ML | C-ROB-ML / R-ML | Inapplicable: no Stage-2 recipe; R-ML rejects estimated weight |
| Direct FIML / likelihood | C-GOLD / R-FIML | C-GOLD / R-FIML | observed statistic/bread and pattern-score meat: C-FIML-ROB / R-FIML | Inapplicable: no second-stage weight; R-FIML rejects it |
| ML2S / NT-ML | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S; equals fixed (no quadratic-weight influence) |
| ML2S / ULS | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S; equals fixed (identity) |
| ML2S / DWLS | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S |
| ML2S / ADF | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S |
| ML2S / DLS(a) | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S |
| Continuous ULS / identity | C-GOLD, C-LS-MATRIX / R-LS-MEAN | C-GOLD, C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-LS-MEAN | Identity has zero influence: C-LS-MATRIX / R-LS-MEAN |
| Continuous GLS / NT(S) | C-GOLD (transported), C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-RECIPE, R-LS-MEAN | C-LS-MATRIX / R-RECIPE, R-LS-MEAN | C-LS-MATRIX, C-RECIPE / R-RECIPE, R-LS-MEAN |
| Continuous WLS / ADF | C-GOLD (transported), C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-RECIPE, R-LS-MEAN | C-LS-MATRIX / R-RECIPE, R-LS-MEAN | C-LS-MATRIX, C-RECIPE / R-RECIPE, R-LS-MEAN |
| Continuous DWLS / diag ADF | C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX, C-RECIPE / R-LS-MEAN |
| Continuous DLS(a) / NT-ADF mixture | C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX, C-RECIPE / R-LS-MEAN |
| Continuous LS / supplied W | C-LS / R-RECIPE | C-LS / R-SUPPLIED | C-LS / R-SUPPLIED | UnsupportedInference: unknown influence of supplied W; C-RECIPE / R-RECIPE |
| All-ordinal ULS / identity | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | Zero influence: C-ORD-MATRIX / R-ORD-MATRIX |
| All-ordinal DWLS / diag NACOV | C-GOLD, C-ORD-MATRIX / R-ORD-MATRIX | C-GOLD, C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX |
| All-ordinal WLS / inverse NACOV | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX |
| All-ordinal GLS / NT | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | UnsupportedInference: NT weight influence not derived; C-ORD-MATRIX / R-ORD-MATRIX |
| All-ordinal DLS(a) | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | UnsupportedInference: DLS weight influence not derived; C-ORD-MATRIX / R-ORD-MATRIX |
| All-ordinal LS / supplied W | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | UnsupportedInference: unknown influence of supplied W; C-ORD-MATRIX / R-ORD-MATRIX |
| Mixed ordinal / DWLS | C-MIX-MATRIX / R-MIX-MATRIX | C-MIX-MATRIX / R-MIX-MATRIX | C-MIX-MATRIX / R-MIX-MATRIX | NumericIssue: mixed weight influence not implemented; C-MIX-MATRIX / R-MIX-MATRIX |
| Mixed ordinal / WLS | C-MIX-MATRIX / R-MIX-MATRIX | C-MIX-MATRIX / R-MIX-MATRIX | C-MIX-MATRIX / R-MIX-MATRIX | NumericIssue: mixed weight influence not implemented; C-MIX-MATRIX / R-MIX-MATRIX |
| Mixed ordinal / ULS | C-MIX-MATRIX / R-MIX-MATRIX rejects fitting ULS | C-MIX-MATRIX / R-MIX-MATRIX rejects fitting ULS | NumericIssue: mixed robust score supports DWLS/WLS only; C-MIX-MATRIX / R fitting refusal | Same core NumericIssue / R fitting refusal |
| Prepared ordinal association ML | UnsupportedInference: LS score is not the ML-target score; C-ORD-MATRIX / R-ORD-MATRIX | Same rejection and gates | Same rejection and gates | Same rejection and gates |

### Conventions and limits

FIML uses analytic observed information, with expected information only for
identification rank. Complete ML exposes expected and observed geometry.
The complete ML R adapter refuses model-implied robust equality releases
(the core accepts caller Gamma_NT); empirical robust releases are gated in
R-ML. R mixed ULS is refused at fit_model(), while its C++ ordinary score
workers exist. These adapter guards currently provide explicit R error text,
not a PostError enum.
Continuous LS uses the expected residual-Jacobian geometry and refuses observed
bread; estimated weights require empirical moments. Caller-Gamma R adapters and
a separately derived observed/estimated-weight score projection remain task-4.
Pairwise moment sources remain MCAR; no MAR law is inferred for those routes.
Mixed ordinal expansion and association-ML score contracts remain 0.3.0 work.

ML2S distinguishes sampling groups from missingness patterns. Its ordinary MI
is explicitly `naive_stage2`, evaluated on Stage-1 EM moments with the fitted
Stage-2 objective. Robust scaling uses Stage-1 moment covariance. lavaan's
ordinary two-stage `modindices()` uses unstructured information; it is not a
structured naive-Stage-2 reference fixture.

The grouped reductions check the same raw-data covariance construction on both
sides, with unequal n/N weights. They do not establish reference-law calibration
under a misspecified multi-group mean restriction. Group-allocation and score
centering follow the [sampling-law scope](../scope.md#group-allocation-and-likelihood-score-covariance);
random missingness patterns are not sampling groups. A separate fixed-allocation
extension or a choice to center between-group score means is not validated by
these algebraic gates.

GLS/WLS golden parity is **transported**, not raw. For the fitted covariance
residual r, nuisance moment Jacobian J and candidate derivative d, define
v = d - J(J'WJ)^(-1)J'Wd. The core fixed-W statistic is
MI = N(v'Wr)^2/(v'Wv), and EPC = (v'Wr)/(v'Wv).
The new generator independently constructs the one-factor covariance and its
analytic derivatives, using column-major lower-triangle moments including the
diagonal. Both fits set `sample.cov.rescale=FALSE`; the generator verifies that
the frozen sample covariance equals the unbiased N-1 covariance. Installed lavaan 0.7-2's continuous GLS/WLS output uses
k = (N-1)/N in the candidate score, while its expected information remains
J'WJ: MI_lavaan = N(k d'Wr)^2/(v'Wv), EPC_lavaan = k d'Wr/(v'Wv).
The inspected pinned `lav_model_grad` sets GLS/WLS group score weights to
(nobs-1)/ntotal, while `modindices` combines that candidate gradient with N and
the expected-information Schur complement. The generator checks both latter formulas to 1e-8 and preserves raw oracle
MI/EPC in JSON, alongside k and the independent efficient-score result. The
C++ consumer explicitly multiplies its MI by k^2 and EPC by k for the oracle
comparison. This scalar score transport leaves no hidden change of fitted W,
sample covariance, n or oracle values. Small remaining differences reflect the
nonzero fitted nuisance gradient: the core retains J'Wr in the Schur score,
whereas the stationary oracle form assumes it is zero. This is a stated
comparison convention, not an oracle-defect exemption.

### Measurements

Measured on the pinned oracle theta, with no refitting in the C++ consumer.
The table lists maximum absolute differences over the checked target rows.
Raw comparisons preserve the existing fixture's sample-size conventions;
GLS/WLS explicitly use the recorded score transport described above.

| Fixture | MI error | EPC error | MI / EPC gate | Release MI / p error | Release MI / p gate |
| --- | ---: | ---: | --- | --- | --- |
| ML | 3.450044e-5 | 5.133190e-7 | 1e-4 / 2e-6 | 5.231415e-11 / 4.194478e-12 | 1e-7 / 1e-8 |
| Observed FIML | 7.532795e-6 | 1.116287e-7 | 2e-5 / 5e-7 | 4.804921e-10 / 1.864403e-11 | 1e-7 / 1e-8 |
| Raw ULS | 0.059270984 | 0.000940361 | 0.065 / 0.0012 | 0.023441073 / 0.000856438 | 0.03 / 0.0012 |
| Raw ordinal DWLS | 0.013473879 | 0.000601743 | 0.02 / 0.001 | 0.002135256 / 0.001134891 | 0.003 / 0.002 |
| Raw mixed DWLS | 0.829736594 | 0.000195575 | 1.1 retained / 0.0005 | 0.005409003 / 0.001345888 | 0.01 / 0.002 |
| Transported GLS | 3.013589e-6 | 4.491212e-8 | 1e-5 / 5e-7 | No equality targets | — |
| Transported ADF/WLS | 7.960787e-6 | 1.534900e-7 | 1e-5 / 5e-7 | No equality targets | — |

The FIML MI limit is tightened from 0.2 to 2e-5; ordinal DWLS from 0.1
to 0.02; ULS MI from 0.07 to 0.065 and release from 0.05 to 0.03.
ULS and ordinal DWLS cannot use machine-size **raw** tolerances: their leading
MI ratio is (N/(N-1))^2 and EPC ratio N/(N-1), with N=301 and N=360,
respectively. The small residual after that transport is optimizer/polychoric
precision. Their frozen raw oracle values have not been rewritten.

Mixed ordinary MI is **limited validation**, not raw lavaan parity. Its current
worker uses score=2N J'Wr and information=2N J'WJ, while its equality-release
worker uses N. The measured MI/oracle ratio is approximately
2(N/(N-1))^2. The existing 1.1 absolute MI tolerance is retained solely as the
pre-existing regression gate; it is not a justified precision tolerance and
must not be used to claim oracle agreement. Resolving the mixed MI target/scale
is `TASK-33.4` in the local work board (0.3.0); this lane changes no numerical convention or default.
The new mixed matrix gates establish the retained ordinary/robust reduction
and precise refusals, not a solved oracle discrepancy.

The GLS/WLS generator checks the stationary candidate-score oracle form to
1e-8, then checks the independently reconstructed efficient MI/EPC to 1e-5.
The C++ consumer compares the independent results to 1e-7 relative tolerance
and the standardized EPCs to 1e-6. The remaining 3e-6/8e-6 transported MI
errors are explained by recorded nonzero nuisance scores, not finite-difference
steps (the generator uses analytic derivatives). No tolerance was widened.

### Verification

On 2026-10-02, the full `fast` C++ suite passed all 1,430 tests with
`just jobs=1 test`. The full installed `magmaanlab` testthat suite completed
without test failures or errors; its existing two-level tests emitted two
admissibility warnings and skipped the unavailable multi-group lavaan oracle.
The lane package was installed by `just jobs=1 r-dev fast` into
`/tmp/rlib-task-3`, reusing the fast core and its normal O1 development glue.
The pinned generator's independent guards, focused recipe gates, layering
guard and tracked-file checks also passed.
