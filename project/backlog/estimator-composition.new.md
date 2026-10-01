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
