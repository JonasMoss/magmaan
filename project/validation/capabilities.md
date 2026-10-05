# Inference capability inventory

This is the single inventory for 0.2.0 exit criterion 6 and defines
"supported" for 0.2.0. It covers the ordinary inference policy, estimation and
verdicts, covariance domains and penalties, named lavaan reporting conventions,
and lab/C++ MI and equality-release components. Evidence is
component-specific: an existing primitive is not automatically a checked
reporting bundle.

States: **validated** within the named checked slice; **limited validation**
when components exist but the required composition, regime or calibration is
not fully gated; **unsupported** when the route returns a typed reason (or, for
estimation, refuses the call); **inapplicable** when the recipe or convention
does not define the component. Parity establishes numerical compatibility, not
calibration or policy adoption. Unsupported describes the named API route, not
the absence of a C++ algorithm or a limitation of lavaan.

## Primary inventory (0.2.0)

Ordinary behaviour in these tables is gated by
[`test_capability_inventory.R`](../../r-magmaan/tests/testthat/test_capability_inventory.R),
which fits every estimator and covariance policy and checks each component's
state and typed reason. The ordinary package calls the lab's
`policy_inference()`/`policy_nested()`, which call the C++ composers
`api::policy_inference_ml`, `policy_inference_dwls`, `policy_nested_ml` and
`policy_nested_dwls` ([`policy.cpp`](../../cpp/src/api/policy.cpp)); the three
tiers therefore share one dispatch. Recipes are in the
[inference policy](../design/r-interface-vision.md#inference-policy).

### Ordinary policy inference

Slice: raw data, one or more groups, affine equality constraints (shared
labels, `group.equal`, `==`), unrestricted covariance, converged fit.
"Gate" is algebra and composition evidence; "calibration" is target-regime
evidence for the adopted recipe (exit criterion 3).

| Setup | Component | State | Gate | Calibration | Open work |
| --- | --- | --- | --- | --- | --- |
| Complete continuous ML (incomplete rows deleted listwise) | Covariance (observed-bread sandwich) | Validated | `policy covariance: observed-bread sandwich of casewise scores`, `... exact scores under a misspecified structured mean` (C++); ordinary `ML covariance is lavaan's observed-information sandwich` (lavaan `robust.huber.white`) | [decisions/03](../../experiments/decisions/03-score-centering/report.qmd): N=80 Wald undercoverage (90–93%) is sandwich-variance error, not point bias; near nominal by N=300 | Documented (task-19); small-sample correction banked |
| | Global score and LR (expected geometry, SB and PEBA4) | Validated | `policy global tests: shared geometry with SB and PEBA4`; ordinary `the likelihood-ratio test's SB calibration is lavaan's Satorra-Bentler` | [complete-ml-global-test-geometry](../../experiments/_archive/complete-ml-global-test-geometry): PEBA4 score 2.0 to 7.2% in 32 cells | None |
| | Nested LR (Satorra-2000, observed information) | Validated | `policy nested tests: observed geometry under a misspecified larger model`, `... coincide at exact fit`, `frozen lavaan dropped-loading score and exact LR`, `ML nested LR: restricted misspecified means use exact casewise scores` (direct complete-data FIML and finite-difference scores, casewise/tiled); ordinary invariance nesting matches lavaan's LR | [decisions/04](../../experiments/decisions/04-nested-ml-geometry/report.qmd): near nominal by N = 300 per group under strong misspecification, where expected drifts to 10–16%; with unmodeled group mean differences, never worse than the pre-66 meat (structured-mean production); over-rejects with skewed data at N = 100 | Reported second, after the score test, with a small-sample caveat |
| | Nested score (observed projection, expected metric) | Validated (conservative at small N) | `policy nested tests: the hypothesis quadratics with SB and PEBA4`; `... complete-data FIML nested score equals the ML policy score`; first-principles score gate | [decisions/04](../../experiments/decisions/04-nested-ml-geometry/report.qmd): 2–3% at N = 100 per group, 3.6–5.7% at N = 300; near 5% under strong misspecification | Finite-sample correction (backlog); primary nested test |
| | Wald intervals, defined parameters | Validated (with the covariance) | Ordinary `defined parameters use the policy covariance`, `confint() intervals are Wald ...` | As covariance | None |
| All-ordinal DWLS (delta or theta) | Covariance (estimated-weight IJ sandwich) | Validated (OPG first stage); exact-first-stage reconfirmation pending (TASK-79) | `DWLS policy IJ covariance agrees with the delete-one jackknife`; `at exact fit the IJ covariance is the fixed-weight sandwich` (C++); ordinary DWLS policy test | [decisions/05](../../experiments/decisions/05-dwls-policy-calibration/report.qmd): median coverage about 95% for every target in 42 cells (misspecification to .4), minimum 92.3%; three two-group correlation cells at N = 300 below 93% | None |
| | Global fit-function statistic n F (score slot; exact spectrum All reference on every positive `robust_ordinal` eigenvalue) | Validated (OPG first stage); exact-first-stage reconfirmation pending (TASK-79) | `DWLS policy: IJ covariance and one fit-function global test`, saturated reduction | [decisions/05](../../experiments/decisions/05-dwls-policy-calibration/report.qmd): All chosen from 13 references (SB and PEBA4 over-reject at df >= 53); fresh-seed confirmation 2.9-6.8%, none outside [3%, 7%] at N >= 500 | None |
| | Global LR | Inapplicable | Typed `inapplicable`: DWLS has no likelihood | | |
| | Nested fit-function difference (r-term parameter-space estimated-weight IJ law, SB and PEBA4) | Validated (OPG first stage); exact-first-stage reconfirmation pending (TASK-79) | `DWLS nested policy: fit-function difference ...`, `delta and theta match their common-point laws; two groups compose`, two-group theta invariance and lab diagnostic-IJ gates; `... IJ law approaches Satorra-2000` | [decisions/05](../../experiments/decisions/05-dwls-policy-calibration/report.qmd): separate-point profile law diagnosed and replaced; fresh-seed confirmation 4.3-7.8%, none outside [3%, 7%] at N = 1000 (cells above 7% at N = 400) | None |
| | Moment-nested threshold fit-function difference (q1 − q0 tangent/IJ law) | Limited validation; calibration pending | `DWLS moment nesting: Wu-Estabrook threshold steps`; parameter-law equivalence <= 1e-10; lab policy and ordinary anova gates; 3-category typed `equivalent_models`; loadings-only comparison stays `not_nested` | 100-replicate correct-model mean/variance diagnostic, 1000/group; agrees within Monte Carlo uncertainty | Fresh-seed size calibration pending |
| | Nested score | Unsupported (intentional for 0.2.0) | Typed `unsupported_model` | | Joint nested score primitive absent (task-18, decision D4) |
| | Wald intervals, defined parameters | Validated (OPG first stage); exact-first-stage reconfirmation pending (TASK-79) | As covariance | As covariance | None |
| FIML (complete or incomplete continuous data) | Covariance; global and nested score/LR with SB and PEBA4 | Limited validation | `policy_fiml_test.cpp`: complete-data reductions, MCAR/MAR lavaan-gated sandwich, grouped components, direct-score finite differences under larger-model misspecification, typed reasons | research/44 global; decisions/03 nested observed score (one df-1 normal family) | No MAR latent sensitivity panel or frozen confirmatory run; growing score conservatism with df; FIML compatibility has separate ML/MLR reporting gates below |
| Mixed DWLS (complete data, delta/theta) | Covariance (exact first-stage estimated-weight IJ) | Limited validation | TASK-67 stratified jackknife; `Mixed DWLS policy: exact sampling global law and IJ nested law`; ordinary inventory and mixed policy gates | Calibration pending | Empirical sampling rows differ from fitting OPG NACOV |
| | Global n F, All spectrum from exact sampling influence | Limited validation | Explicit exact-row cross-product construction; OPG-spectrum difference; one/two groups delta/theta | Calibration pending | OPG NACOV fitting weights retained |
| | Nested fit-function difference, observed-Hessian/IJ SB/PEBA4 law; moment-tangent nesting | Limited validation | Common-point profile and direct parameter-space laws agree <= 1e-10; lab and ordinary nested reporting | Calibration pending | Local moment embedding witness |
| | Global LR; nested score | Inapplicable; unsupported | Typed `inapplicable`; `unsupported_model` | | DWLS has no likelihood; nested score is not derived |
| All-ordinal association ML | Every ordinary component | Unsupported | Typed association-ML refusal; lab covariance, spectral and score tests below | | Policy calibration remains TASK-32 subcard 5 |
| ML2S, GLS, continuous ULS, all-ordinal ULS/WLS, mixed WLS | Every component | Unsupported | Typed `unsupported_model`; lab components below | | Secondary breadth, consumer-gated; no 0.2.0 requirement |
| Continuous WLS (ADF), mixed ULS | Estimation | Unsupported in `magmaan()` | The call errors and names `magmaanlab::estimate()` / DWLS or WLS | | Consumer-gated |
| Two-level, SAM, composites, closed-form estimators | Every component | Not offered in `magmaan()` | Lab interfaces keep their own gates (area files) | | No 0.2.0 requirement |

### Domains, penalties and fit states

These rules apply to every setup above; the reason is typed per component.

| Domain or state | Ordinary policy behaviour | State | Evidence |
| --- | --- | --- | --- |
| PSD covariance, interior estimate | Same recipes; the PSD and unrestricted estimators coincide | Validated (by reduction) | Probe in the inventory test; `covariance = "psd"` ordinary tests |
| PSD covariance, boundary estimate | Computed for an interior population; the output says so | Limited validation | Ordinary `PSD fits on the cone boundary get inference for an interior population`; the covariance-honest interior study used normal-theory tests, and its rerun with policy components is planned |
| Population on the PSD boundary | Chi-bar-square limits | Inapplicable (outside scope) | [Availability](../design/r-interface-vision.md#availability) |
| `barrier(lambda)`, lambda > 0 | Estimates only; every component `penalized` | Unsupported | `policy: a penalized estimate gets no inference, before any other gate`; ordinary `barrier fits are experimental, penalized and recorded`, `barrier fits cover the primary estimators` |
| `barrier(0)` | The unrestricted fit | As unrestricted | `magmaan()` maps lambda = 0 to unrestricted |
| Failed convergence verdict | Every component `not_converged` | Validated typing | `policy: the fit state gates every component`; ordinary `unchecked convergence and retained fits remain inspectable` |
| Saturated model | Covariance computed; global tests `saturated` | Validated | `policy: saturated models get the covariance but no global test`; ordinary `saturated models have a covariance but no global test` |
| Inadmissible unrestricted estimate (improper solution) | Computed under the interior-population assumption, with an admissibility warning | Limited validation | [Availability](../design/r-interface-vision.md#availability) |
| Active parameter bounds, nonlinear constraints | Not expressible in `magmaan()` | Inapplicable to the ordinary API | Lab only; compatibility limits below |

### Estimation, verdicts and lavaan-compatible fitting

| Estimator (ordinary) | Native estimation | Verdict | `preset = "lavaan-0.7.2"` |
| --- | --- | --- | --- |
| ML | Validated: lavaan parity goldens and ordinary `ML estimates come from magmaanlab and match lavaan` | Exact-Hessian Newton check | Validated: unrestricted, linear equalities (task-9) |
| FIML | Validated: `fiml_golden_test.cpp` | Newton check | Validated for unrestricted and supported linear equalities: pinned MCAR/MAR fixtures, single and grouped (task-10) |
| All-ordinal DWLS/ULS/WLS | Validated: `ordinal_golden_test.cpp`, grouped and threshold-equality gates | Newton check | Unsupported: errors (task-11) |
| ML2S, GLS, continuous ULS, mixed DWLS/WLS | Validated by component goldens (area files) | Newton check | Unsupported: errors |
| PSD and barrier fits | PSD: validated on the classical slice, hardening 0.3.0; barrier: experimental | Face-restricted Newton check | Unsupported: errors |

Rescaled retry endpoints and their verdicts can depend on floating-point search
paths; their gates check each endpoint under its declared acceptance rule
(exit criterion 2). Under a compatibility rule inference follows the rule, and
`fit$inference$convergence$disagree` records any disagreement with magmaan's
own check.

### Cross-cutting ownership

- **Groups.** The policy gates above include two-group cases (grouped ML
  covariance against lavaan's MLR sandwich, grouped global geometry, invariance
  nesting, two-group DWLS composition); lavaan ordinal conventions use each
  group's n minus one.
- **Missing data.** ML, GLS, ULS and the ordinal estimators delete incomplete
  rows listwise (reported in `fit$rows`); FIML and ML2S use them. The ordinary
  policy uses observed-data FIML inference for incomplete continuous data.
- **Constraints.** Affine equalities are validated; defined parameters use the
  delta method with the active covariance. Fixed, omitted and equality-written
  nested nulls share one null geometry (`policy nested embedding: ...`).
- **Likelihood-score contracts.** `policy_score_contracts_test.cpp` gates
  `policy score contracts: stationarity raw meat and projection` for complete,
  MCAR and MAR rows in one/two groups, including a single-observed-variable
  pattern. It checks finite-difference casewise scores and observed bread,
  free-direction stationarity (half the terminal 1e-3 deviance-gradient
  tolerance), raw sandwich covariance, nuisance orthogonality and projected
  raw cross-products. `policy score contracts: equivalent constraints at a
  common null point` compares shared labels, `==` rows and two-group
  `group.equal` in common larger-model coordinates (relative 1e-8), including
  global/nested statistics and spectra. `policy score contracts: two-group
  complete-data ML reduction` gates nested observed scores and covariance.
  `policy score contracts: actual Heywood PSD boundary has typed inference`
  uses a fitted PSD boundary, checking its flag and finite available values
  or typed unavailability. These are component identities, not calibration.
- **Retained data and refits.** Inference reuses the fit's retained data and
  geometry; `infer()` after `inference = FALSE` equals the default call;
  `anova()` requires the same observations in the same order and both fits'
  estimator and covariance policy; refits keep the anchor's group order.
  `api::refit_from_null` and lab `refit_from_null(fit_H1, fit_H0)` explicitly
  start H1 from the verified embedded H0 estimate, replaying H1's fitting
  options. Ordinary `anova()` retries only an "alternative fits worse"
  failure, accepts an improved endpoint no worse than H0 with a passing native
  verdict, and records/prints the refit without replacing input fits. ML/FIML
  use retained observations or moments; ordinal DWLS reuses retained stage-1
  statistics. Gates: `test_nested_reseed.R`, `test_refit_from_null.R`, and
  `api refit_from_null` C++ cases. Unsuccessful retries keep the typed failure.
- **Intervals.** Wald only. `confint(test = "lr")` is rejected with a message;
  the LR interval interface is planned and unscheduled (task-36).

In the next table, unsupported means **not yet supported in ordinary magmaan
compatibility reporting**.

## Ordinary reporting conventions

Reporting uses retained fits; nested comparisons have the conditional refit
recovery described above. `lavaan_compat = NULL`
remains the default. Positive barrier penalties return `penalized`; failed
selected convergence verdicts return `not_converged`. Numerical failures retain
their reason per component. Score tests are inapplicable to lavaan reporting
bundles and are not replaced by policy score tests.

| Fitted setup / convention | Covariance | Global test | Nested test | Evidence and limits |
| --- | --- | --- | --- | --- |
| Complete continuous ML / ML | Validated | Validated: standard | Validated: standard difference | Single-group CFA; grouped loading/intercept invariance; random-X regression; nested fixed-zero covariance; matched means/group order |
| Complete continuous ML / MLM | Validated: expected empirical sandwich | Validated: Satorra-Bentler | Validated: SB2001 | Same global/covariance slices; single-group and grouped loading/intercept/mean nested differences |
| Complete continuous ML / MLR | Validated: observed exact-score sandwich | Validated: YB-Mplus H1-minus-H0 trace | Validated: SB2001 with MLR scales | Same global/covariance slices; single-group and grouped loading/intercept/mean nested differences; independent C++ saturated-distance/H0 numerical-score trace |
| All-ordinal DWLS / DWLS | Validated: NACOV sandwich | Validated: unscaled statistic, p-value unavailable | Validated: standard difference, p-value unavailable | Delta/theta, single group and two unequal groups with loading equality; theta threshold equalities; per-group n minus one |
| All-ordinal DWLS / WLSMV | Validated: NACOV sandwich | Validated: scaled-shifted | Validated: Satorra-2000 delta, scaled-shifted | Single-group delta/theta covariance/loading restrictions; two-group theta configural → loadings and Wu-Estabrook thresholds → thresholds+loadings; saturated alternative; both ordinary argument orders |
| All-ordinal ULS / ULS | Validated: NACOV sandwich | Validated: unscaled statistic, p-value unavailable | Validated: standard difference, p-value unavailable | Delta/theta, single group and two unequal groups with loading equality; theta threshold equalities; per-group n minus one |
| All-ordinal ULS / ULSMV | Validated: NACOV sandwich | Validated: scaled-shifted | Validated: Satorra-2000 delta, scaled-shifted | Same nested slices as WLSMV; per-group n minus one for the objective, original n_g/N fractions for the nested sandwich |
| All-ordinal WLS / WLS | Validated: standard covariance | Validated: standard | Validated: standard difference | Delta/theta, single group and two unequal groups with loading equality; theta threshold equalities; per-group n minus one |
| FIML / ML or MLR | Validated: observed Hessian / Huber–White sandwich | Validated: standard / Yuan–Bentler Mplus | Validated: standard / SB2001 with YB-Mplus scales | HS MCAR/MAR CFA; school loading/intercept invariance; missing random covariate; both argument orders and saturated alternative; retained estimates/policy and cached/deferred/serialized reporting |
| FIML / MLM | Inapplicable | Inapplicable | Inapplicable | Rejected: MLM's missing-data handling would change estimation |
| Complete mixed DWLS / WLSMV | Validated at identical parameter points: NACOV sandwich | Validated at identical points: scaled-shifted | Validated at identical points: Satorra-2000 delta, scaled-shifted | Single/two groups delta/theta; Stage-1 parity; strict single-group delta retained gates; grouped/theta endpoints have limited validation due to stopping differences (oracle observations ledger); missing and other mixed bundles typed unsupported |
| Continuous GLS, ULS, WLS; ML2S; other mixed bundles | Unsupported | Unsupported | Unsupported | No checked ordinary compatibility composition; see lab inventory below |

The whole-bundle installed-lavaan gates are
[`test_lavaan_compat.R`](../../r-magmaan/tests/testthat/test_lavaan_compat.R).
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
| Degenerate robust scale | **Limited validation** outside the exact saturated gate. `scaled saturated tests and penalized fits keep typed unavailability` gates `summary(..., lavaan_compat = "MLR")`: covariance remains finite while the global test returns `saturated`; other zero/nonfinite scale endpoints have no whole-bundle gate and are not promoted to validated. |
| Positive barrier penalty | **Unsupported** compatibility covariance/nested reporting, gated by the same test through `vcov(..., lavaan_compat = "MLM")` and `anova(..., lavaan_compat = "MLR")`: typed `penalized`, no fallback. |

The grouped complete-ML nested gate calls `lavaan::lavTestLRT` defaults and
checks explicit `standard` (ML) or `satorra.bentler.2001` (MLM/MLR), in both
model orders. Loading, intercept and latent-mean equalities are released in
successive comparisons. Grouped ordinal gates call `lavaan::lavInspect("vcov")`
and `lavaan::lavInspect("test")`: `robust.sem` covariance with `scaled.shifted`
tests for WLSMV/ULSMV, `robust.sem` with unscaled `standard` tests for DWLS/ULS
(no p-value), and `standard` covariance/test for WLS. Both delta and theta
parameterizations use unequal groups and each group's n_g minus one reporting
normalization, with loading equality. Theta threshold equalities (thresholds;
loadings and thresholds; loadings, thresholds and intercepts) are gated for the
same covariance/global bundles. Delta threshold equality remains **limited
validation**: lavaan's released `~*~` scale is not identified under delta (it
stays at 1 with a singular covariance), while magmaan releases the second-group
scale, so there is no like-for-like lavaan reference; see
[ordinal and mixed capabilities](../architecture/capabilities/ordinal_and_mixed.md).
Ordinal nested bundles remain unsupported, including threshold-restriction
comparisons.

Compare matching structural and estimation settings in lavaan, including
`meanstructure = TRUE` and `fixed.x = FALSE` for ordinary continuous fits.
FIML recipes pinned against installed lavaan 0.7.2: `missing="ml"`, random X,
mean structure, `information="observed"`, `h1.information="structured"`,
`observed.information="hessian"`; ML standard SE/test, MLR
`robust.huber.white` SE and `yuan.bentler.mplus` test. The Mplus test overrides
H1 information to unstructured EM moments internally and divides the
H1-minus-H0 information/score trace by df. Default nested ML is standard;
MLR uses SB2001 with those YB-Mplus scales (zero weighted contribution from
a saturated alternative). Compatibility does not assert calibrated policy.

Missing-data deletions, estimation weights, parameterization, group order and
convergence rules remain fitting choices. Reported bundles target lavaan 0.7.2.

## Existing lab components and remaining composition gates

Association-ML covariance is a lab-only component in
`estimate::frontier::association_ml_ij` / `magmaanlab::association_ml_ij()`.
It uses observed sensitivity and exact empirical Stage-1 influence, including
threshold cross-covariance; independent numerical gates pass. Ordinary
association-ML inference remains unsupported pending
calibration (TASK-32 subcard 5). Lab global/nested tests now report All,
SB and PEBA4 from local correlation ML geometry/exact Stage-1 Gamma and
observed H/IJ meat/exact parameter restrictions, respectively. Independent
matrix reconstruction, normalization, exact-fit, saturated/identical zero-df,
grouped and misspecified pseudo-true restriction gates pass in `ordinal_ij_test.cpp`.
Moment-only nesting has a typed refusal. Lab R spectrum/reference reconstruction,
zero-df, common-Stage-1 and unsupported-channel gates are in
`r-package/tests/testthat/test_association_ml_tests.R`. Lab-only
`association_ml_modification_indices` and `association_ml_score_tests` add
one-direction observed-Schur EPC and exact projected-meat chi-square tests,
with per-candidate typed refusals and `test_association_ml_scores.R` gates.

The complete all-ordinal lab IJ has an explicit exact empirical-Jacobian
first-stage comparator (`first_stage = "exact"`), returning sampling rows and
Gamma alongside covariance. OPG remains the lab default; the ordinary
all-ordinal DWLS policy uses the exact empirical influence (TASK-69), with
reconfirmation pending (TASK-79). TASK-74 gates case-weight derivatives,
one/two-group delta/theta composition and Gaussian-copula convergence.


| Setup | Existing covariance/global primitives and gates | Nested primitives and remaining gates |
| --- | --- | --- |
| Complete ML | `ntml_covariance`, `ntml_score_sandwich`, expected UGamma, `fiml_robust_mlr` complete-data trace reduction. Lab empirical SE/score/MI meats default to exact casewise likelihood projections (TASK-77), with structured/unstructured moments as explicit comparators. SE/SB fixtures: `inference_golden_test.cpp`, `multigroup_inference_golden_test.cpp`; complete bundles above | `lr_test_satorra_bentler2001`/2010 and restriction-map Satorra-2000 exist. SB2001 bundles above are checked; SB2010 is not an ordinary bundle |
| FIML | `fiml_observed_information`, `fiml_robust_mlr` and spectrum primitives exist; `fiml_golden_test.cpp` checks robust SEs, MLR statistics, scales and H1/H0 traces. Lab `vcov()` exposes observed-information and observed-sandwich routes. Ordinary ML/MLR bundles compose these primitives; live `test_lavaan_compat.R` gates covariance, intervals, global and default nested reports | Restriction-map and scalar SB2001/2010 engines and lab adapters exist. The scalar engines derive their single-model scales from a residual-projector spectrum, so their names alone do not establish agreement with lavaan's MLR trace recipe. The ordinary default MLR bundle uses the scalar SB2001 reducer with explicit `fiml_robust_mlr` YB-Mplus scales, gated on single/grouped missing-data pairs and a saturated alternative; the frozen Satorra-2000 fixture remains a separate gate |
| All-ordinal association ML | Lab-only `association_ml_ij`: observed H, exact empirical Stage-1 meat, active/full covariance and joint thresholds. Independent q/score/H/D, nonnormal case-weight, stratified delete-one (N=250/1000/4000), exact-fit and chart gates in `ordinal_ij_test.cpp`; lab covariance reconstruction and threshold cross-covariance | Lab global/nested All/SB/PEBA4 with independent spectra, normalization, exact-fit, zero-df and grouped pseudo-true restriction gates; moment nesting refused. Lab fixed/absent MI and affine releases use observed Schur sensitivity/EPC and exact projected meat; independent Schur, gradient, finite-difference/refit, grouped and nested-spectrum gates. Ordinary MI/releases await subcard-5 policy calibration |
| All-ordinal DWLS/ULS/WLS | `robust_ordinal`; `ordinal_golden_test.cpp` gates component SEs/tests. Ordinary composer adds lavaan reporting normalization and bundled method selection | `lr_test_satorra2000_ordinal` supports exact/delta and returns scaled-shifted results; lab `robust_nested_lrt()` already dispatches to it. Existing grouped DWLS/theta golden comparisons gate the delta mean-scaled result with lavaan normalization. Retained-fit WLSMV/ULSMV scaled-shifted reporting is gated by `test_convention_nested_ordinal.R` and `test_lavaan_compat.R`; unsupported slices stay explicit |
| Continuous GLS/WLS/ULS | `robust_continuous_ls` with explicit weight/Gamma; `ls_golden_test.cpp` and weighted-inference tests. The ULS/DWLS Browne-residual NT reporting recipe is not composed here | Continuous weighted Satorra-2000 primitives exist; no checked ordinary lavaan-default bundle |
| ML2S | `two_stage_em_ml_inference` and weighted Stage-2 inference exist | Restriction-map and scalar SB2001/2010 engines and lab dispatch exist; scalar methods are NT-only. Naive unstructured-information compatibility and a checked default two-stage reporting bundle remain to do |
| Mixed ordinal | `robust_mixed_ordinal` and mixed component gates exist; complete DWLS `robust_mixed_ordinal_ij` empirical sampling and weight channels pass case-weight, exact-fit and stratified jackknife gates in `mixed_ij_test.cpp`; other routes retain their existing contracts; see [mixed limitations](../architecture/capabilities/ordinal_and_mixed.md) | Mixed WLSMV compatibility composes NACOV covariance, scaled-shifted global and default Satorra-2000 nested tests; point-gated one/two-group delta/theta, limited retained grouped/theta validation; other bundles unsupported |

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
| Ordinal compatibility `anova()` | Retained-fit C++ composer; installed lavaan 0.7.2 default `satorra.2000`, `A.method="delta"`, `scaled.shifted=TRUE`, larger-model information/Jacobian; n_g−1 objective counts and original n_g/N sandwich fractions | Whole-bundle statistic/df/p/divisor/shift gates at doctest-relative 1e-5 in `test_convention_nested_ordinal.R`; plain DWLS/ULS retain statistic/df without p, WLS uses standard chi-square. Actual nesting, parameterization, convergence and positive-penalty refusals are explicit. Additional ordinal slices require separate gates |
| FIML compatibility `vcov()`, `confint()`, `summary()`, `anova()` | ML/MLR bundles compose observed-Hessian/Huber–White covariance, standard/YB-Mplus global tests and standard/SB2001 nested tests with explicit trace-based scales; installed-lavaan MCAR/MAR, school invariance, missing random-X and saturated gates | Task-14 complete. Raw/pattern context supports deferred, cached and serialized reporting; policy is separate. Additional regimes need their own evidence |
| Existing complete-ML compatibility tests | Single-group and grouped loading/intercept/mean nested defaults; grouped covariance/global tests | Existing complete-data bundle gates closed by task-7.1; additional regimes require separate evidence |
| Existing ordinal compatibility tests | Single/grouped delta/theta for DWLS/ULS/WLS and MV reporting, loading equality | Theta threshold equality is gated and delta threshold equality is limited above; nested covariance/loading restrictions and theta invariance now have whole-bundle gates; further mean restrictions require separate evidence. Backlog: extend checked reporting conventions |
| Covariance domains and failures | Classical affine/no-active-bound slice; convergence and positive-penalty refusals; PSD metadata | Domain-specific limits are inventoried above, policy domains in the primary inventory; unequal-group normalization is gated for existing ordinal bundles. PSD hardening is 0.3.0 |
| Ordinary FIML policy | C++ likelihood-score, covariance and LR-spectrum primitives | Composed through `policy_inference_fiml`, `policy_nested_fiml` and the retained raw/pattern R context. Numerical gates exist; calibration remains limited (research/44 global, decisions/03 nested). This is separate from lavaan compatibility |
| Ordinary all-ordinal DWLS policy | IJ covariance, fixed-weight global/nested spectra and estimated-weight profile-LR primitives | Compose the adopted policy and settle its nested recipe/calibration; a joint nested score primitive is absent and remains typed unavailable for 0.2.0. Backlog: all-ordinal DWLS policy tasks |
| Parameter confidence intervals | Ordinary Wald intervals and defined-parameter delta SEs; C++ and lab profile-test/CI engines for ML, FIML, ordinal and other routes | Ordinary `confint(test = "lr")` is planned but rejected. Its refit, inversion, calibration and compatibility interaction need an explicit contract and adapter. Backlog: planned LR interval interface, unscheduled |
| Lab estimated-weight comparison | Fixed/estimated-weight primitives; explicit switches on several score, residual, profile and fit-measure routes | `frontier_rbm()` has no off switch; defaults vary between routes. Backlog: lab correction switches/defaults, unscheduled |
| Compatibility naming | Ordinary selector is `lavaan_compat = NULL`; historical bundle names are accepted only for their compatible fits | Adopted 2026-10-02 with explicit lavaan compatibility output, matching attributes and caches; NULL retains the policy default |
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
Mixed ordinary MI remains **limited validation**: TASK-33.4 fixed its
factor-of-two scale defect and ordinary/fixed robust MI has independent
equality-release reconstruction gates, but estimated mixed-weight MI remains
unsupported. These states describe the lab/C++ component routes;
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
- **R-GAMMA**: `r-package/tests/testthat/test_caller_gamma.R`, `caller Gamma reproduces continuous ML and LS MI and releases`, `caller NACOV preserves ordinal fitting weights`, `caller Gamma validates shape, symmetry, PSD and influence provenance`, `caller Gamma reaches continuous covariance and profile LRT adapters`, `explicit Gamma_NT is available for complete ML releases`, and `caller NACOV reaches supported mixed ordinal LS score routes` (single/unequal groups, means, ML/GLS/DWLS/WLS/ULS/DLS/supplied W, all-ordinal and mixed DWLS/WLS, changed meat with unchanged fitting W).
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

Complete mixed DWLS estimated-weight IJ is independently gated by
`mixed_ij_test.cpp`: empirical-Jacobian sampling rows and fitting-weight rows
agree with replicated case-weight derivatives under misspecification; stratified
single/two-group delta/theta delete-one diagonal errors shrink from N = 600 to
1200 and stay below 2.5% at N = 1200, beating the fixed-weight OPG sandwich.
The saturated mixed gate confirms vanishing weight influence at exact fit.
Lavaan NACOV and fitting weights remain unchanged. The separate empirical
sampling channel is scoped to complete ordinary mixed DWLS IJ, not the ULS/WLS,
missing-data, robust-builder or RBM routes. Pure endpoint fits are rejected with
`NumericIssue`; continuous marginal rows have independent analytic controls,
but continuous empirical-Gamma fitting weights differ from mixed OPG NACOV.
TASK-71 composes the ordinary mixed DWLS policy; these gates do not establish estimated-weight
MI/release or sampling calibration.

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
| Complete ML / likelihood | C-ML, C-GOLD / R-ML | C-ML, C-GOLD / R-ML | C-ROB-ML / R-ML; R-GAMMA | Inapplicable: no Stage-2 recipe; R-ML rejects estimated weight |
| Direct FIML / likelihood | C-GOLD / R-FIML | C-GOLD / R-FIML | observed statistic/bread and pattern-score meat: C-FIML-ROB / R-FIML | Inapplicable: no second-stage weight; R-FIML rejects it |
| ML2S / NT-ML | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S; equals fixed (no quadratic-weight influence) |
| ML2S / ULS | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S; equals fixed (identity) |
| ML2S / DWLS | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S |
| ML2S / ADF | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S |
| ML2S / DLS(a) | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S |
| Continuous ULS / identity | C-GOLD, C-LS-MATRIX / R-LS-MEAN | C-GOLD, C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-LS-MEAN; R-GAMMA; C-OBS-LS / R-OBS-LS | Identity has zero influence: C-LS-MATRIX / R-LS-MEAN; C-OBS-LS / R-OBS-LS |
| Continuous GLS / NT(S) | C-GOLD (transported), C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-RECIPE, R-LS-MEAN | C-LS-MATRIX / R-RECIPE, R-LS-MEAN; R-GAMMA; C-OBS-LS / R-OBS-LS | C-LS-MATRIX, C-RECIPE / R-RECIPE, R-LS-MEAN; C-OBS-LS / R-OBS-LS |
| Continuous WLS / ADF | C-GOLD (transported), C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-RECIPE, R-LS-MEAN | C-LS-MATRIX / R-RECIPE, R-LS-MEAN; R-GAMMA; C-OBS-LS / R-OBS-LS | C-LS-MATRIX, C-RECIPE / R-RECIPE, R-LS-MEAN; C-OBS-LS / R-OBS-LS |
| Continuous DWLS / diag ADF | C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-LS-MEAN; R-GAMMA; C-OBS-LS / R-OBS-LS | C-LS-MATRIX, C-RECIPE / R-LS-MEAN; C-OBS-LS / R-OBS-LS |
| Continuous DLS(a) / NT-ADF mixture | C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-LS-MEAN; R-GAMMA; C-OBS-LS / R-OBS-LS | C-LS-MATRIX, C-RECIPE / R-LS-MEAN; C-OBS-LS / R-OBS-LS |
| Continuous LS / supplied W | C-LS / R-RECIPE | C-LS / R-SUPPLIED | C-LS / R-SUPPLIED; R-GAMMA; C-OBS-LS / R-OBS-LS | UnsupportedInference: unknown influence of supplied W; C-RECIPE / R-RECIPE |
| All-ordinal ULS / identity | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX; C-OBS-LS / R-OBS-LS | Zero influence: C-ORD-MATRIX / R-ORD-MATRIX; C-OBS-LS / R-OBS-LS |
| All-ordinal DWLS / diag NACOV | C-GOLD, C-ORD-MATRIX / R-ORD-MATRIX | C-GOLD, C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX; R-GAMMA; C-OBS-LS / R-OBS-LS | C-ORD-MATRIX / R-ORD-MATRIX; C-OBS-LS / R-OBS-LS |
| All-ordinal WLS / inverse NACOV | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX; R-GAMMA; C-OBS-LS / R-OBS-LS | C-ORD-MATRIX / R-ORD-MATRIX; C-OBS-LS / R-OBS-LS |
| All-ordinal GLS / NT | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | UnsupportedInference: NT weight influence not derived; C-ORD-MATRIX / R-ORD-MATRIX |
| All-ordinal DLS(a) | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | UnsupportedInference: DLS weight influence not derived; C-ORD-MATRIX / R-ORD-MATRIX |
| All-ordinal LS / supplied W | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | UnsupportedInference: unknown influence of supplied W; C-ORD-MATRIX / R-ORD-MATRIX |
| Mixed ordinal / DWLS | C-MIX-MATRIX / R-MIX-MATRIX | C-MIX-MATRIX / R-MIX-MATRIX | C-MIX-MATRIX / R-MIX-MATRIX; R-GAMMA | NumericIssue: mixed weight influence not implemented; C-MIX-MATRIX / R-MIX-MATRIX |
| Mixed ordinal / WLS | C-MIX-MATRIX / R-MIX-MATRIX | C-MIX-MATRIX / R-MIX-MATRIX | C-MIX-MATRIX / R-MIX-MATRIX; R-GAMMA | NumericIssue: mixed weight influence not implemented; C-MIX-MATRIX / R-MIX-MATRIX |
| Mixed ordinal / ULS | C-MIX-MATRIX / R-MIX-MATRIX rejects fitting ULS | C-MIX-MATRIX / R-MIX-MATRIX rejects fitting ULS | NumericIssue: mixed robust score supports DWLS/WLS only; C-MIX-MATRIX / R fitting refusal | Same core NumericIssue / R fitting refusal |
| Prepared ordinal association ML ([contract plan](../design/association-ml-inference.md)) | UnsupportedInference: LS score is not the ML-target score; C-ORD-MATRIX / R-ORD-MATRIX | Same rejection and gates | Same rejection and gates | Same rejection and gates |

### Conventions and limits

FIML uses analytic observed information, with expected information only for
identification rank. Complete ML exposes expected and observed geometry.
The complete ML R adapter refuses model-implied robust equality releases
(the core accepts explicitly supplied Gamma_NT through the new gamma adapter);
empirical robust releases are gated in R-ML and caller Gamma in R-GAMMA.
R mixed ULS is refused at fit_model(), while its C++ ordinary score
workers exist. These adapter guards currently provide explicit R error text,
not a PostError enum.
Continuous LS and all-ordinal ULS/DWLS/WLS support exact observed-Hessian
nuisance sensitivity with the expected residual-Jacobian metric; estimated
weights require empirical moments. The lab defaults select observed sensitivity
and estimated weights. Expected sensitivity and fixed weights remain explicit
comparators. C-OBS-LS (`score_robust_test.cpp`, tests prefixed `observed LS score:`)
gates both Hessians, unequal-group case-weight and projected influences, exact-fit
reduction, normal GLS, and scalar/multi-df spectrum calibration. R-OBS-LS
(`test_lab_inference_defaults.R`, `LS score defaults use observed sensitivity
and estimated weights`) gates lab routing and comparators. Caller-Gamma R adapters
are gated by R-GAMMA for fixed-weight ML, continuous LS and all-ordinal LS;
categorical workers replace NACOV while preserving fitting W. Supplied Gamma
requires explicit estimated_weight=FALSE, validates dimension/symmetry/PSD and
refuses FIML/ML2S scores and casewise weight influence. Mixed LS uses the same
NACOV adapter, with its existing unsupported weight cells unchanged.
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
| Raw mixed DWLS | 0.004577846 | 0.000195575 | 0.005 / 0.0005 | 0.005409003 / 0.001345888 | 0.01 / 0.002 |
| Transported GLS | 3.013589e-6 | 4.491212e-8 | 1e-5 / 5e-7 | No equality targets | — |
| Transported ADF/WLS | 7.960787e-6 | 1.534900e-7 | 1e-5 / 5e-7 | No equality targets | — |

The FIML MI limit is tightened from 0.2 to 2e-5; ordinal DWLS from 0.1
to 0.02; ULS MI from 0.07 to 0.065 and release from 0.05 to 0.03.
ULS and ordinal DWLS cannot use machine-size **raw** tolerances: their leading
MI ratio is (N/(N-1))^2 and EPC ratio N/(N-1), with N=301 and N=360,
respectively. The small residual after that transport is optimizer/polychoric
precision. Their frozen raw oracle values have not been rewritten.

Mixed ordinary and fixed-weight robust MI now use the same criterion units as
equality releases: the fitter minimizes F/2, with F = (s-sigma)'W(s-sigma)
(block fractions included). The score is -N J'r and the Gauss–Newton metric
is N J'J for the whitened residual r; no extra factor two belongs to mixed
moments. The previous doubling was a magmaan defect (TASK-33.4), fixed in
both ordinary and robust workers without changing EPC or fitting weights.

The independent C++ test `mixed frozen moments: independent df=1 Schur MI
and equality reconstruction` uses fixture 0005's frozen first-stage moments,
W, Gamma, theta and N. It projects the added column j off the nuisance
columns JK, giving v; ordinary MI is N(v'r)^2/(v'v), EPC is -(v'r)/(v'v),
and fixed-weight robust MI is N(v'r)^2/(v' sqrt(W) Gamma sqrt(W) v).
It also differentiates F/2 numerically and reconstructs the single equality
release using its constraint normal. This gates criterion units independently
of the production score/Schur routines. The frozen oracle comparison separately
transports the score by (N-1)/N, hence MI by its square and EPC by that factor;
the MI/EPC/release comparisons are gated at 1e-8 relative tolerance (MI error
4.3e-9 and release error 1.8e-12). This is a comparison convention, not an
oracle-defect exemption. The raw-data
gate retains the measured finite-divisor/first-stage floor above. Estimated
mixed-weight influence remains refused; this fix establishes no new sampling
calibration or ordinary-user mixed inference policy.

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
