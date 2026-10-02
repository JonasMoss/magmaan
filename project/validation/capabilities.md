# Inference capability inventory

This is the single inventory for 0.2.0 exit criterion 6. The initial rows cover
named reporting conventions; the full policy, estimation/domain, penalty and
verdict inventory remains in the active backlog. Evidence is component-specific:
an existing primitive is not automatically a checked reporting bundle.

States: **validated** within the named checked slice; **limited validation**
when components exist but the required composition or regime is not fully
gated; **unsupported** when the reporting route returns a typed reason;
**inapplicable** when the convention does not define the component. Parity
establishes numerical compatibility, not calibration or policy adoption.

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
| All-ordinal DWLS / WLSMV | Validated: NACOV sandwich | Validated: scaled-shifted | Unsupported | Same slices; exact Satorra-2000 delta/shifted nested bundle still needs composition gates |
| All-ordinal ULS / ULS | Validated: NACOV sandwich | Validated: unscaled statistic, p-value unavailable | Unsupported | Single-group delta/theta; grouped reporting remains limited validation |
| All-ordinal ULS / ULSMV | Validated: NACOV sandwich | Validated: scaled-shifted | Unsupported | Single-group delta/theta; grouped reporting remains limited validation |
| All-ordinal WLS / WLS | Validated: standard covariance | Validated: standard | Unsupported | Single-group delta/theta; grouped reporting remains limited validation |
| FIML / ML or MLR | Unsupported | Unsupported | Unsupported | Observed-information and robust-MLR lab primitives exist; full bundles await missing-data convention gates |
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
| FIML | `fiml_observed_information`, `fiml_robust_mlr` and spectrum primitives exist. Full missing-data recipe naming/composition is limited validation; backlog "pin FIML robust conventions" owns it | Restriction-map route exists; lab R boundary rejects SB2001/2010 for FIML. Ordinary lavaan nested bundle unsupported |
| All-ordinal DWLS/ULS/WLS | `robust_ordinal`; `ordinal_golden_test.cpp` gates component SEs/tests. Ordinary composer adds lavaan reporting normalization and bundled method selection | `lr_test_satorra2000_ordinal` supports delta; ordinal golden invariance gates exist. Default shifted bundle and ordinary adapter remain to do |
| Continuous GLS/WLS/ULS | `robust_continuous_ls` with explicit weight/Gamma; `ls_golden_test.cpp` and weighted-inference tests. The ULS/DWLS Browne-residual NT reporting recipe is not composed here | Continuous weighted Satorra-2000 primitives exist; no checked ordinary lavaan-default bundle |
| ML2S | `two_stage_em_ml_inference` and weighted Stage-2 inference exist | Naive unstructured-information compatibility and default two-stage difference bundle remain to do |
| Mixed ordinal | `robust_mixed_ordinal` and mixed component gates exist | Mixed Satorra-2000 primitives exist; ordinary compatibility remains unsupported |

No policy covariance is described merely as "lavaan with the weight correction
turned on": policy and compatibility can also differ in bread, meat, sensitivity
and calibration. Isolating weight influence requires matched lab recipes.
