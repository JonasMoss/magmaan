# Shared fitting composition: implementation merge update

Completion recorded 2026-10-01. The initial decisions and dependency order
have been merged into the
[roadmap](../architecture/roadmap.md#shared-fitting-composition) and
[active TODO](todo.md#shared-fitting-composition). This file is a focused
completion record for the maintainer editing those documents; fold it into
them and remove it after the implementation and validation below are recorded
there.

## Foundation implementation

`model::MomentTarget` distinguishes covariance and correlation targets.
`model::correlation_evaluation()` projects grouped covariance values and their
optional vech Jacobians, drops mean moments, and fixes diagonal correlations
and their derivatives to one and zero respectively. Positive finite variances
are required; the consuming discrepancy owns the definiteness requirement.

`estimate::ml_objective()` accepts either target while retaining its existing
two-argument covariance entry point and half-discrepancy scale. Correlation
inputs must already have unit diagonals and no mean moments; input moments are
not repaired. Ordinal association ML uses the same projection and ML kernel as
the covariance route. The first foundation commit retained its old catML
fitting wrappers; the subsequent dispatch slice below replaces them.

The existing determinacy penalty composes with the correlation objective at
the scalar-problem level. This establishes the foundation for shared penalties;
it does not yet add fitting dispatch, penalty finalization or an inference
contract for every discrepancy and moment source.

## Validation and maintained-document update

Validated in a frozen copy of committed base `979d5d86` plus this slice, keeping
concurrent parser edits out of the build:

- Optimized C++ build and all 1,339 tests excluding the heavy `parity` label
  passed. This includes independent finite-difference derivatives, grouped
  projection, observation-unit invariance, invalid inputs, barrier composition
  and existing ordinary/PSD catML checks.
- Isolated development installation of `magmaanlab` and the full
  `test-admissibility.R` suite passed, including the PSD catML wrapper.
- Layering, tracked-file and whitespace checks passed. All five refreshed
  vendored files match their canonical C++ sources exactly.

Close **complete and validate the moment-target foundation** in the active
TODO. Replace the roadmap's foundation-in-progress statement with the
implemented contract above and these validation gates. The foundation commit
alone did not establish ordinal fitting or inference contracts; the following
slice completes the supported ordinal association contract and its ML dispatch.

## Ordinal association contract and ML dispatch

The methods-development fitting surface now calls this capability **ML**.
C++ `api::fit(model, ordinal_data, api::ml())` dispatches to ordinal association
ML. Low-level `estimate::frontier::fit_ml()` / `fit_ml_psd()` overloads consume
`OrdinalStats`; the separate `fit_catml()`, `fit_catml_psd()` and
`catml_objective()` fitting names are removed. Historical diagnostic formulas
and Newton audit identifiers retain their categorical-ML names.

In `magmaanlab`, use `fit_model(spec, data, estimator = "ML")` with all variables
ordered, or pass prepared ordinal stats. `psd = TRUE` uses the same discrepancy
over PSD primitive covariance blocks. Direct
`magmaan_core$fit_ml(spec, ordinal_stats)` and
prepared `estimate(model, data, estimator = "ML")` share the ordinary route;
`frontier_fit_ml_psd()` also accepts ordinal stats. The separate
`frontier_fit_catml_psd()` wrapper/export/registration is removed. The active
PSD stress-study adapters use the replacement while retaining their historical
arm IDs. Ordinary-user `magmaan()` defaults and estimator policy are unchanged.

The model triple retains the prepared full parameter vector and its threshold
rows for reporting/reconstruction. Stage-1 thresholds are imposed as affine
restrictions at their actual fitted values, independent of starting hints;
they are excluded from the association search. Fixed unit indicator residual
variances supply the association gauge. Delta/theta conventions remain reporting
choices over the same correlation target. The active correlation-Jacobian rank
sets df; an unidentified association map is an explicit error. Linear loading
equalities, including grouped models, are supported. Threshold constraints,
mean/intercept requests, released response scales, non-unit response variance
constraints and nonlinear equalities are rejected before preparation can hide
them. Mixed/polyserial ML remains unsupported in this slice.

Fit provenance records the moment source/target, discrepancy, covariance domain,
algorithm, saturated-threshold policy and unvalidated inference state. Full
`npar` still includes reported threshold coordinates; `npar_active` and
`association` record the actual search dimension, rank and df. Stage-1 R,
thresholds and available sampling objects remain unchanged. Model-implied values
are the fitted correlations. Gaussian likelihood/AIC/BIC and post-fit sampling
inference are explicitly unavailable for these fits pending their own contract.

The shared penalty fitting/finalization work and barriers across retained moment
sources/discrepancies remain open, including the separate direct-FIML barrier.
The new metadata describes this ordinal ML route; it does not finish general
dispatcher metadata consolidation across every retained route.

## Ordinal dispatch validation and merge actions

- Full optimized C++ build and all 1,358 tests excluding the heavy `parity`
  label passed, including the ordinary/PSD derivative gates, fixed threshold
  starts, identification charts, overidentified/grouped/loading-constrained
  models, unsupported constraints and API post-fit rejection.
- Isolated development installation of the compiled package passed its full
  testthat suite, including staged/convenience/direct/refit equivalence,
  delta/theta reporting, unchanged Stage-1 objects and post-fit error gates.
  The suite retains its documented multi-group two-level oracle skip and
  covariance-admissibility warnings on two existing two-level controls.
- The ordinary R package installed against that compiled dependency and passed
  its full testthat suite, including live lavaan checks.
- Layering, tracked-file and whitespace guards passed; generated exports and
  vendored C++ sources are synchronized.

Close **define the ordinal association-model contract** for the supported
all-ordinal fixed-gauge/linear-equality slice. Record ML-on-ordinal ordinary/PSD
dispatch and CatML fitting-name retirement under **unify fitting dispatch and
composition metadata**; keep that task open for cross-route consolidation.
Preserve the mixed/polyserial and sampling/inference gates and the shared
barrier/penalty priorities. Replace roadmap claims that the old CatML fitting
entry points remain available with the replacement API and limitations above.

## Next mapping slice: saturated-FIML moments

Decision recorded 2026-10-01: consolidate the existing continuous two-stage
FIML/EM route next; defer mixed/polyserial ML. This is a composition migration,
not a new estimator or a change to the existing two-stage numeric/inference
conventions.

| Existing route | Moment source | Model target | Fitting discrepancy/weight |
| --- | --- | --- | --- |
| ML2S, `stage2_weight = "nt"` | Saturated continuous FIML/EM | Covariance and means | Shared ML discrepancy |
| ML2S, `"uls"` | Same | Same | Moment quadratic, identity weight |
| ML2S, `"dwls"` | Same | Same | Moment quadratic, inverse diagonal Stage-1 Gamma |
| ML2S, `"adf"` / `"wls"` | Same | Same | Moment quadratic, inverse Stage-1 Gamma |
| ML2S, `"dls"` | Same | Same | Moment quadratic, inverse fixed-a NT/Stage-1 Gamma mixture |

Keep the full Stage-1 object (means, covariances, sample sizes, ACOV/influence
information and existing optional regularization provenance), not just S.
Covariance domain is a separate unrestricted/PSD choice; shared model penalties
must apply to the final SEM while leaving Stage 1 unchanged unless a distinct
Stage-1 transformation is requested. Preserve ordinary two-stage sandwich/test
dispatch and existing PSD inference exclusions. Direct FIML remains the
observed-pattern likelihood route. Gate this migration on unchanged ordinary
and PSD estimates, unchanged Stage 1, refit reconstruction and inference parity.

Weight names describe estimation as well as potential inputs to inference.
Fixed WLS minimizes a quadratic moment discrepancy with W held constant during
optimization; W can have been estimated from the data. Changing W generally
changes the point estimate in an overidentified model. Its sampling covariance
and any estimated-weight contribution belong to the separate inference
contract. A fixed-a DLS NT endpoint remains a quadratic fit; it is not an alias
for the nonlinear ML discrepancy.

## Non-mixed covariance policies and shared barriers

Implementation recorded 2026-10-01. `magmaanlab::fit_model()` and prepared
`estimate()` now accept `covariance = "unrestricted" | "psd" | "barrier"`.
`psd = TRUE` remains a compatible spelling. Barrier options select
`target = "joint" | "determinacy"` and finite non-negative `weight` (default
0.25). Explicit contradictory options and additional barrier bounds are
rejected. Defaults for ordinary fitting and ordinary-user inference are
unchanged.

| Moment source | Shared PSD / barrier fitting coverage |
| --- | --- |
| Complete continuous and pairwise MCAR moments | ML, ULS, GLS, explicit fixed-weight WLS |
| All-ordinal polychorics | Association ML, ULS, DWLS, WLS |
| Saturated continuous FIML/EM | ML2S with NT-ML or fixed ULS/DWLS/ADF/DLS Stage 2 |
| Direct continuous FIML | Observed-pattern likelihood, separate raw-data route |

In C++, existing ML/FIML penalty entry points and the new
`estimate::frontier::fit_gmm_multiinfo()` and `fit_ordinal_multiinfo()` compose
the same scalar penalty wrapper. GLS constructs its normal-theory metric once;
general WLS retains the supplied metric. ML2S composes the existing Stage-1
moment and Stage-2 weight builders with these fitting primitives. This does
not introduce a covariance option into the C++ `api::EstimatorSpec` facade.
Retain facade consolidation as a separate follow-up if wanted.

The shared objective is `f - weight / N * P`. Fits preserve unpenalized `fmin`
and report `penalized_fmin` separately; stationarity diagnostics use the actual
penalized objective. Available ML/FIML/GMM Newton audits include analytic
penalty curvature. Ordinal penalty routes currently provide geometric
stationarity, without a new general analytic penalized-ordinal Newton audit.
Observed Sigma must be PD when a positive-weight penalty conditions on it,
including LS routes whose base discrepancy does not impose that requirement.
Zero weight retains the base discrepancy's domain; an undefined penalty report
at an improper solution is represented by NaN rather than rejecting the fit.

Continuous ML, fixed-weight quadratics and direct FIML barriers use the
existing unit-normalization contract. Parameter values, bounds and affine
constraints are transported. Quadratic weights are transported with the
moment units so internal scaling does not change the fitting metric. Direct
FIML reports the caller-unit likelihood, including its missingness-pattern
Jacobian constant. This is not a new normalization contract for unsupported
nonlinear constraints.

Prepared ordinal fit-only ULS needs no sampling Gamma; fit-only DWLS needs
only its fitting diagonal. The transformed LS builder now validates those
inputs without pretending that an unmaterialized Gamma exists. Post-fit
inference retains its stricter sampling-input validation. Threshold parameter
locations are excluded from model-penalty derivatives. Ordinal ML continues
to hold Stage-1 thresholds fixed; ordinal LS fits its usual threshold moments.

Continuous pairwise fitting is a moment source for the shared discrepancies,
selected through `missing = "pairwise"`, rather than a separately named GLS
estimator. Fits retain `pairwise_stats`, overlap counts/probabilities and raw
missingness provenance. Supplied pairwise summaries keep their source identity
even when the caller omits the missing-data option. ML2S retains the complete
Stage-1 object, including ACOV/influence and regularization provenance.
PSD/barriers change final SEM fitting only; input moments are not repaired.
Refit routes and common composition metadata retain these choices.

General penalized and pairwise sampling inference remains explicitly
unvalidated. New compositions reject generic SE/test and likelihood-based
fit-measure helpers rather than returning complete-data formulas. This does
not change the earlier experimental penalty wrappers' research post-fit
behavior. Mixed/polyserial ML/barriers and two-level policies remain deferred.
Near-pole optimizer/default/fallback research and general penalty-inference
calibration remain open; fitting coverage alone does not close those studies.

For the maintained documents, close the non-mixed **shared penalty fitting and
finalization** slice and the **saturated-FIML composition migration** fitting
slice. Record common R covariance dispatch, unchanged Stage 1, separate direct
FIML and these explicit limits. Keep mixed work, inference calibration, the
C++ friendly facade decision and any remaining numerical research gates open.

Validation: the optimized shared working-tree build passed all 1,363 tests
excluding the heavy `parity` label (concurrent policy work was present).
New independent gates check fixed-weight criterion preservation, zero-weight
reductions including improper solutions, affine-constraint transport,
threshold-free penalty derivatives and fit-only ordinal ULS/DWLS without
sampling Gamma. An isolated development installation passed the full compiled
R suite and the full ordinary-user R suite, including live lavaan comparisons.
R gates cover all tabled routes, both barriers, prepared/convenience/refit
equivalence, unchanged Stage 1, provenance and inference exclusions. A strict
L-BFGS unit gate checks ML and direct-FIML fits under mixed observation units
and the FIML missingness-pattern likelihood constant. The compiled suite
retains its documented multi-group two-level oracle skip and two existing
covariance-admissibility warnings. Vendored core files and generated exports
are synchronized; layering, tracked-file and whitespace guards pass.
