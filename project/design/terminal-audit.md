# Terminal Audit (L1) and Fit Diagnostics (L2)

## Authoritative fit verdict (2026-09-12)

The full-model audit is the authority for numerical acceptance throughout
magmaan. Backend termination explains how the search stopped; it is neither
proof of convergence nor an additional veto on a passing common verdict.
The driven-coordinate L1 record remains available for debugging and
lavaan-compatible inspection. The backend adapter policies described below
are implementation details, not the fit-level acceptance contract.

`estimate::fit_verdict(estimates)` is the C++ entry point. Its status and
component checks are `Passed`, `Failed`, or `Unchecked`. A failed component
makes the overall verdict failed; all required components must pass to make
it passed. An incomplete audit with no demonstrated failure is unchecked.
No unchecked component falls back to a backend status or L1 boolean. A
normal-cone projection that did not converge leaves stationarity unchecked;
it is not evidence that the fitted point is nonstationary.

The required checks are:

1. Recompute the original objective at the returned full parameter vector,
   on the common per-observation half-discrepancy scale. Require finite parameters/objective
   and agreement with the reported objective within
   `1e-6 * (1 + abs(reported))`; expose the normalized values and threshold.
   Record `objective_multiplier`, applied to both value and gradient. It is
   one for ordinary ML/LS/FIML and `1/N` for two-level ML, whose native `fmin`
   is a total deviance. This conversion does not change fitting or inference.
2. Recompute the full-model objective gradient, including eliminated linear
   and threshold coordinates. Check the declared constraints and the existing
   model-Frobenius metric-dual L2 stationarity residual at tolerance `1e-3`.
   This change does not recalibrate tolerances or assert global optimality.
3. For complete-data ML (2026-09-24), stationarity at a regular interior
   point of the fitting domain is decided by the Newton accuracy check
   instead: `d = sqrt(G' I^{-1} G) <= .01` from the total score and observed
   information after linear-equality reduction, with the conditioning and
   solve guards of `newton_accuracy_ml`. Every ambient fit is interior to its
   domain, improper estimates included. A PSD fit is interior when no
   primitive covariance block is singular (`covariance_nullity == 0`).
   Nonpositive curvature, ill conditioning or an unreliable solve fail the
   check, because no accuracy statement is available there. PSD ML fits use
   the covariance-domain version at every point, boundaries included
   (2026-09-24): for each singular connected component of a primitive block,
   with null basis U and multipliers from least squares on the face
   constraints, null directions with positive multipliers are held
   (U_c' dC U_c = 0), the reduced Hessian adds the rank-constrained set's
   curvature 2 tr(M dC C^+ dC) (Shapiro's sigma term; the blocks are linear
   in theta), and d is computed on the tangent space. Directions with zero
   or negative multipliers stay free. Fixed-zero variances hold their rows.
   Active box bounds and nonlinear equalities keep the first-order check of
   item 2, and so does every path without the Newton diagnostic (LS, FIML,
   ordinal, two-level, penalized). The interior first-order
   residual is retained as telemetry, not as an extra veto. The verdict's
   `criterion` records which check decided.
   Decision 2026-09-25: every iterative estimator gets this check at fit
   time, with the Newton step measured in standard-error units
   (rollout evidence: engineering/active/19-newton-verdict-migration). Landed the same day
   for FIML (ordinary, PSD; observed-information metric) and
   every moment-quadratic fit (GLS, ULS, WLS, DWLS, GMM,
   SNLLS, their PSD versions; exact Hessian, normal-theory sandwich metric
   `d^2 = (H s)' Omega^{-1} (H s)`), then for the ordinal and mixed
   least-squares fits (exact Hessian, Gauss-Newton sandwich metric) and the
   barrier fitters (penalized Hessian as metric). The diagnostic records its
   objective, curvature source and metric. A positive-definite implied Sigma
   is required only for likelihood objectives. CatML stays on the first-order
   check by decision, two-level until it has an analytic Hessian. Once `checked` is true, an `Unavailable` curvature
   fails the verdict, like nonpositive curvature; a free parameter outside
   the closed-form second derivatives reports `Unsupported` and leaves the
   first-order check deciding.

The fit entry point declares `Ambient` or `Psd` before the verdict is selected.
Ordinary fits use equality/bound normals; explicitly PSD-constrained fits
also use primitive covariance-cone normals and require PSD feasibility.
Never select the domain according to which residual passes. Both residuals
remain visible and coincide in the positive-definite interior.

Covariance admissibility is a separate diagnostic for ordinary fits: an
unconstrained optimum may be numerically converged but inadmissible. Numerical
convergence also does not establish identification, a local minimum, or the
validity of a particular standard-error or test formula. Consumers may require
those properties in addition, with separately reported reasons.

R exposes `fit$verdict` and `fit$diagnostics$verdict`; `fit$converged` is the
logical projection: TRUE for passed, FALSE for failed, NA for unchecked.
`optimizer_status` and `fit$audit` remain diagnostic. Consumers should use
`isTRUE(fit$converged)` when they require a certified numerical fit and must
not turn missing/NA verdicts into success. `evaluate_at(ML)` now supplies the same Newton evidence as ML fit finalization,
and accepts an explicit ambient/PSD domain. Its `audit_options` still govern
only the legacy L1 record; explicit common-policy controls are provided by
`frontier::assess_convergence`. Closed-form estimators that are not
objective minimizers report numerical convergence as not applicable rather
than inventing a stationary objective.

Failed or unchecked fits remain inspectable return values. A future adapter
migration must also preserve evaluable terminal candidates across soft backend
exits so the common audit can assess them; invalid inputs and absent/nonfinite
candidates remain errors. Different stopping algorithms are allowed, but
candidate retention and final assessment must follow one policy.

### Standardization rollout

Implemented first: one verdict function independent of optimizer status,
original-objective verification at the existing continuous ML/LS, FIML,
ML2S, Fisher/IRLS, two-level ML and ordinal finalization seams, explicit PSD
domain selection, continuous closed-form SNLLS coverage, full-coordinate audits for profiled ordinal paths,
and Estimates-based R result serialization. Uncovered specialized or extra-
callback-constraint paths report unchecked; they are not silently certified.
The existing frozen SNLLS handoff remains evidence of its pinned revision and
retains its original screening policy, not a template for new studies.

Remaining migration work belongs in `project/backlog/todo.md`: backend candidate
retention, complete specialized/extra-constraint coverage, and migration of
research consumers that still read L1/backend flags. This section supersedes
older statements below that treat the full-model audit as merely additive or
an optimizer status as the authoritative fit verdict.

## Reusable Newton computations (2026-09-24)

The Newton calculation has an explicit owning C++ interface in
`estimate/frontier/newton_accuracy.hpp`. Fitting retains the small diagnostic
record as before; callers needing the underlying computation use
`frontier::audit_newton_ml(pt, rep, sample, theta, domain, options)`. Its
`NewtonAudit` owns derivatives, geometry, numerical solve preparation, the
solution, effective options and diagnostics. No optimizer runs in this call.

Each stage is independently callable:

1. `evaluate_newton_ml` computes the per-observation objective plus the full
   gradient and analytic observed Hessian on the total negative-log-likelihood
   scale. It retains theta and the total sample size.
2. `prepare_newton_geometry` consumes those derivatives and builds the equality
   basis K and, for the PSD domain, tangent basis Z and full-coordinate
   curvature correction Q. The reduced system is
   `g = (K Z)' G`, `H = (K Z)' (I + Q) (K Z)`. I remains unchanged and available.
3. `prepare_newton_system(H)` equilibrates and factorizes H. The owning result
   can be reused for additional right-hand sides. `solve_newton_system` returns
   the signed step `-H^{-1}g`, Newton distance, predicted gain and solve residual.
4. `assess_newton_accuracy` applies the budget, condition-number guard and solve
   residual guard. These decisions can be changed without evaluating the model
   or factorizing again. Its `NewtonAudit` overload preserves domain and boundary
   metadata. The geometry tolerance is a preparation input; changing it requires
   rebuilding geometry and its dependent solve, not just reassessing.

Low-level matrix stages also accept caller-supplied curvature. The caller must
establish its objective, normalization and coordinates; accepting a matrix does
not grant it the complete-data ML statistical interpretation. Reusing artifacts
requires the same model, sample, parameter ordering and point. The full observed
information can be passed to existing inference primitives; the PSD-adjusted
matrix must not be substituted for it. There is no hidden cache or borrowed
lifetime in the retained artifacts.

For example, callers can retain the complete calculation and reuse the full
information for a separately requested inference calculation:

```cpp
using namespace magmaan::estimate::frontier;
auto audit = audit_newton_ml(pt, rep, sample, fit.theta, domain, options);
// Check the artifact statuses before consuming numerical values.
auto covariance = magmaan::inference::vcov(audit.derivatives.hessian, pt);
auto stricter = audit.options;
stricter.budget = 0.001;
auto reassessed = assess_newton_accuracy(audit, stricter);
```

Existing `newton_accuracy_from`, `newton_accuracy_ml` and
`newton_accuracy_ml_psd` remain summary convenience functions over these stages.
This separates computation from assessment without changing the prescribed
fit-level convergence criteria. General convergence-report composition remains backlog work; the estimator
adapters described below reuse these stages.

## Box-constrained Newton corrections

Explicit Newton audits now respect finite box bounds through a convex quadratic
correction in equality-reduced coordinates. `solve_newton_box` is independently
callable with an already prepared `NewtonSystem`; `audit_newton_derivatives`
composes it automatically when bounds are supplied. It minimizes
`g' s + s' H s / 2` subject to `lower - theta <= B s <= upper - theta`, where
`B` maps reduced increments into full coordinates. Equal lower/upper bounds are
held as fixed coordinates before preparing the Hessian. The original full
Hessian remains unchanged and retained.

The reduced Hessian must be positive definite. A feasible active-set solve
reuses its equilibrated factorization; blocking bounds can enter and bounds
with wrong-sign multipliers can leave. Merely sitting at a bound does not freeze
that coordinate: an inward improving direction remains available. Weakly active
bounds require no arbitrary positive-multiplier cutoff. Bounds that the
unconstrained correction would cross also participate, even if initially inactive.
The zero correction must be feasible; out-of-box inputs are not silently snapped.

The retained `NewtonAudit::box` contains the inequality normals/offsets,
multipliers, final working set, iteration count and normalized primal, dual and
complementarity residuals. `box_max_iter` and `box_tolerance` are runtime
preparation controls in `NewtonAccuracyOptions`; changing them requires solving
again. Iteration exhaustion or a singular working set returns unavailable
evidence, never a weaker acceptance test. The usual condition and solve-residual
guards apply during assessment. The standard quadratic-program stationarity,
feasibility and complementarity conditions are described in the
[CVXOPT mathematical documentation](https://cvxopt.org/userguide/coneprog.html#quadratic-programming);
this implementation uses its own feasible active-set solver.

For constrained corrections, `predicted_gain = -g's - s'Hs/2` and
`distance = sqrt(2*predicted_gain)`. This agrees with the existing Newton distance
when the unconstrained minimizer is feasible, but at a binding inequality it is
an objective-gain budget, not an interior sampling-standard-error interpretation.
A boundary optimum can have a nonzero raw gradient and zero feasible gain.
Nonpositive reduced curvature fails this convex-quadratic criterion even when a
more general constrained second-order test might establish local optimality.

PSD geometry remains separate. In the PSD interior the box solve is available;
the existing local PSD geometry is not a guarantee that a full finite correction
remains covariance-admissible. No globalization or line search is performed.
At singular PSD faces, lower bounds at or below zero on primitive covariance
diagonal parameters are redundant with PSD and can use the existing cone
geometry. Other inactive bounds are supported if the existing PSD correction
stays within them. A nonredundant active box or a PSD correction that crosses a
box boundary remains unsupported: treating these requires joint box/cone
multipliers and curvature. Nonlinear constraints likewise remain unsupported.

Explicit convergence policies recognize `box_constrained` evidence at active
bounds. The compatibility policy and existing fit-time routing are unchanged:
they continue to use first-order stationarity at active bounds. Regression tests
in `newton_box_test.cpp` cover inward/outward and weakly active boundaries,
coupled corrections, equality and fixed-coordinate reduction, redundant and
interacting PSD constraints, numerical exhaustion, and exhaustive two-dimensional
box minima under parameter rescaling.

## Explicit evidence and acceptance contract

`estimate/frontier/convergence.hpp` collects owning `ConvergenceReport` objects;
`convergence_policy.hpp` assesses them or existing `FitDiagnostics` summaries.
No optimizer status is an acceptance input. Collection and assessment are
independent calls, with explicit runtime options and no global option lookup.

The contract is: the same objective, point, model representation, data, weights,
penalty, fitting domain and collection settings produce the same evidence;
the same evidence and policy produce the same decision. `common_fit_verdict`
now delegates to the named compatibility policy, preserving existing fit-time
acceptance. `evaluate_at(ML)` supplies the previously missing Newton evidence.
Its historical default variance bounds remain; pass actual bounds and domain
when comparing it with a fit. Empty bounds in the new report API mean unbounded.

Explicit policies require a finite objective and feasibility in the declared
domain, plus first-order stationarity, Newton accuracy, or both. First-order
uses the retained metric-dual residual and completed normal-cone projection;
Newton uses the declared-domain distance and curvature/solve guards. Ambient
feasibility does not require covariance admissibility. PSD feasibility does.
Neither implies global optimality or validates statistical inference.

Every check returns its status, whether it was required, and a reason.
A known required failure makes the assessment `Failed`; otherwise missing or
unsupported required evidence leaves it `Unchecked`; all required checks must
pass for `Passed`. A missing Newton computation never selects first-order
instead. Nonpositive curvature and failed numerical solve/condition guards are
failures of the Newton policy; unavailable derivative probes or unsupported
constraint geometry are unresolved. Optional checks remain visible without
vetoing acceptance. Compatibility is explicitly the old evidence-dependent
selection rule and does not acquire the semantics of an explicit policy.

A supplied reported objective is checked independently from the recomputed
value. Omitting it leaves consistency unchecked; it is not manufactured by
comparing the recomputed value with itself. Policies can require consistency.
The compatibility policy requires a reported value. All report objectives and
first-order gradients use total/N units; generic scalar inputs declare the
native-to-total multiplier, and their reported value is supplied in native units.

```cpp
namespace audit = magmaan::estimate::frontier;
audit::ConvergenceRequest request;
request.newton = true;
request.bounds = actual_fit_bounds;
request.domain = actual_fit_domain;
auto report = audit::audit_convergence_ml(
    pt, rep, sample, fit.theta, request, fit.fmin);
if (!report) return std::unexpected(report.error());
auto policy = audit::newton_convergence_policy();
policy.require_objective_consistency = true;
auto verdict = audit::assess_convergence(*report, policy);
policy.newton.budget = 0.005;
auto stricter = audit::assess_convergence(*report, policy);
```

Reassessment performs no objective evaluations or Hessian factorizations.
Numeric Newton distance/step/gain survive guard rejection in both the full
artifact and its summary, so changing acceptance guards gives matching results.
Geometry tolerances, active sets, covariance-face preparation and differentiation
controls belong to collection; changing them requires recollection. The
`interior_eigen_tol` member of policy Newton options is not an assessment control.

The generic scalar collector optionally differentiates the original gradient;
the ML collector uses analytic observed information. Any existing retained
Newton adapter can be composed with `audit_convergence(pt, rep, std::move(audit))`
without reevaluating its objective or Hessian. This requires the exact prepared
model used by that adapter, including fixed.x and ordinal parameterization.
The adapter's explicitly held coordinates also constrain the first-order audit.
Reports own their computations and retain the effective request; they do not
own the source data or provide automatic model/data fingerprints. Do not attach
independently computed summaries from other points or objectives.

Regression tests in `convergence_policy_test.cpp` cover missing evidence,
required versus optional checks, objective mismatch, feasibility, unsupported
bounds, nonpositive curvature, guard reassessment without callback evaluations,
ambient versus PSD domains, LS artifact composition, and fit/post-fit ML parity.

## Sphere-native endpoint audit

Sphere frontier fits additionally retain an audit of the driven endpoint before
translation or polish (`gauge$native_audit`, C++ `SphereAudit`). Its objective
excludes the radial pin, its analytic chain rule includes loading-normalization
curvature, and its Newton solve removes radial directions. ML/FIML use observed
information, continuous moment-quadratic fits use their sandwich metric, and
PSD interiors use the sphere tangent geometry. Singular PSD faces, active boxes
and additional nonlinear equalities report unchecked until their joint geometry
is supported. The requested-chart fit keeps its own authoritative verdict:
polish can move the point, so that verdict and the driven-endpoint assessment
must remain separate. A failed or unchecked native audit cannot support a
`magmaan_user_chart_singular` optimum claim. See the
[sphere contract](../architecture/roadmap.md).

## Two-stage convergence composition

`estimate/frontier/ml2s_audit.hpp` provides `audit_saturated_endpoint`,
`audit_ml2s`, and separate assessment overloads. The owning two-stage report
contains the saturated endpoint audit, original Stage-1 inference ingredients,
the optional transformation and its diagnostics, the actual supplied Stage-2
input, an optional recorded fitting-input snapshot, and the Stage-2 report.
No optimizer or EM iteration runs during auditing.

`FIMLH1` and `SaturatedMoments` now retain effective EM options and structured
per-block stopping evidence: direct solution versus parameter-tolerance stop
versus iteration limit, iteration count, final parameter/objective changes,
and covariance-repair count/magnitude. Complete-data direct solutions have zero
EM iterations. Existing `error_on_nonconvergence` behavior is unchanged; set it
false when an early-stopped endpoint should be returned for inspection.

The saturated endpoint audit checks covariance positive definiteness, the
observed-pattern objective, the total negative-log-likelihood gradient and,
when requested, the unmodified analytic Hessian. Its open positive-definite
covariance domain has no active cone-face calculation. Non-PD endpoints fail
feasibility; a covariance floor inside EM does not become an implicitly declared
constraint in the audit. First-order stationarity uses total/N gradients in the
product Frobenius metric (off-diagonal covariance coordinates have weight two).
Stage-1 Newton evidence is independent of the EM stopping flag.

`SaturatedMoments::raw_H` and `raw_gradient` preserve derivatives at the original
endpoint in blockwise `[mean; vech(cov)]` order. `H` retains its existing
inference behavior and may be repaired before inversion; the repair flag,
minimum eigenvalue and ridge are retained separately. The endpoint audit uses
only raw curvature. Its coordinate order is all group covariance vechs followed
by all group means; retained derivatives are explicitly permuted. Analytic raw
curvature can be reused after exact endpoint/count checks. Legacy objects and
diagnostic finite-difference Hessians use the public analytic evaluator instead.
The caller must supply the same raw data/pack that produced retained derivatives;
there is no automatic data fingerprint.

`audit_ml2s` constructs Stage-1 inference ingredients once, or reuses a supplied
`retained_stage1` object; it then audits Stage 2 under NT/ULS/DWLS/ADF/DLS using
the existing adapters. Raw moments stay in `source`. An explicitly requested
`regularize_saturated_stage1` transformation supplies separate Stage-2 moments
and transformed ACOV. Transformed objects clear raw derivative slots, since those
derivatives do not describe the transformed endpoint. Original `H`/`J` and
solver telemetry still describe the source calculation, not a new likelihood
fit to the transformed moments.

Handoff evidence compares means, covariances, counts, weight kind, DLS mixing
value when relevant, and ACOV for DWLS/ADF/DLS against an optional
`Ml2sStage2Input` snapshot retained by the caller when fitting. Equality is exact;
serialized/rounded inputs are not silently treated as identical. A match means
“matches supplied fit-input record”, not proof of historical causality. Without
a record the handoff is unchecked. Existing fit entry points do not automatically
retain this new snapshot. Stage-2 auditing alone establishes convergence against
the supplied inputs, independently of historical provenance.

Assessment accepts separate stage policies, defaults to requiring the handoff,
and optionally requires the Stage-1 solver stopping condition. A known required
failure fails the composition; otherwise missing required evidence leaves it
unchecked. All required pieces must pass. Solver-history absence does not block
an independent endpoint audit unless explicitly required. Newton distances are
not summed, and this is not a bound on propagated structural-parameter error.
Invalid inputs or inability to construct Stage-2 inference ingredients return
expected errors; the standalone endpoint audit remains independently callable.

```cpp
namespace audit = magmaan::estimate::frontier;
// Retain alongside the Stage-2 fit made with these exact inputs:
audit::Ml2sStage2Input fit_input{stage1_moments, weight_kind, dls};
auto report = audit::audit_ml2s(
    pt, rep, raw, pack, h1, fit.theta, weight_kind, dls, {},
    fit_input, fit.fmin, &stage1_moments);
if (!report) return std::unexpected(report.error());
audit::Ml2sConvergencePolicy policy;
policy.stage1 = audit::newton_convergence_policy();
policy.stage2 = audit::newton_convergence_policy();
policy.stage1.require_objective_consistency = true;
policy.stage2.require_objective_consistency = true;
auto verdict = audit::assess_convergence(*report, policy);
```

`ml2s_audit_test.cpp` covers complete data, unequal groups with missingness,
analytic/retained derivative agreement and normalization, early EM termination,
raw negative curvature despite repaired inference information, all Stage-2
weight policies, missing/mismatched handoff records, separate solver-stop policy,
and transformed inputs. R bindings, automatically captured fit-input records,
propagated numerical accuracy  remain follow-ups.

## Coverage by applicable check

This matrix distinguishes existing fit finalization from explicit, reusable
post-fit Newton artifacts. “Fit checks” means the original-objective,
feasibility and first-order checks at the documented common finalization seam;
the explicit report composes these checks independently of fit finalization.

| Fit path | Common fit checks | Explicit retained Newton | Remaining coverage |
| --- | --- | --- | --- |
| Complete-data ML, Fisher/IRLS | Yes | Analytic | Fit/post-fit ML parity tested |
| Fixed LS/GMM, GLS, expanded SNLLS | Yes, Newton at fit time (2026-09-25) | Analytic exact, sandwich metric; requested GN | Broader constraint/domain regression combinations |
| FIML | Yes, Newton at fit time (2026-09-25) | Analytic | Broader missingness/group combinations |
| ML2S | Yes, Stage 2; explicit Stage-1 endpoint and composed report | Analytic Stage 1; all five Stage-2 policies | Automatic fit-input capture and R reports |
| Ordinal/mixed LS, including profiled fits | Yes, full coordinates; Newton at fit time (2026-09-25) | Delta/theta; analytic exact, Gauss-Newton sandwich metric; requested GN | Broader group/profile combinations |
| CatML | Yes, correlation objective | Numerical; thresholds held | Stage-1 threshold estimation is separate |
| Two-level ML | Yes | Numerical | Broader between/within constraint combinations |
| Multi-information penalized ML/FIML | Yes, Newton at fit time (2026-09-25) | Analytic, includes the penalty Hessian | None |
| Native FCSEM | Specialized path | No dedicated adapter | Native parameter/geometry integration |
| Implicit RBM | Specialized path | No dedicated adapter | Actual penalized-objective integration |
| Additional callback constraints / specialized chart objectives | Not uniformly covered | No general adapter | Objective lifting and constraint geometry |

For supported model representations, linear equalities and PSD geometry are
shared by the Newton adapters. Box-constrained quadratic corrections are
available with positive-definite reduced curvature. Nonlinear equalities and
genuinely interacting boxes on singular PSD faces still retain derivatives but
report Newton unsupported; their first-order checks remain applicable. A failed numerical curvature probe reports unavailable,
not mathematical inapplicability. Closed-form non-optimization estimators do not
need an invented optimization convergence test.

Validation anchors are `cpp/tests/unit/newton_accuracy_test.cpp` (quadratic
identities, equality reduction, PSD face curvature, multi-group ML, retained
factorization) and `cpp/tests/unit/newton_adapters_test.cpp` (each listed adapter,
all five ML2S policies, fixed weights away from an optimum, unequal-group
LS normalization, delta/theta mixed LS, and unsupported/unavailable cases).
These are representative checks, not an exhaustive Cartesian product of model,
constraint, domain and estimator choices. Thin R access to the full retained
artifacts and report remains backlog work.

## Newton adapter coverage and curvature provenance

`estimate/frontier/newton_adapters.hpp` provides explicit post-fit adapters;
none runs an optimizer or changes a stored fit verdict. Each returns
`fit_expected<NewtonAudit>`: invalid problem construction is an error, while
unavailable numerical curvature remains an inspectable result with a status
and detail. These are local curvature audits; objective consistency, primal
feasibility, admissibility and first-order evidence remain separate checks.

| Objective | Entry point | Retained curvature |
| --- | --- | --- |
| Complete-data ML | `audit_newton_ml` | Analytic observed information |
| ML2S Stage 2 | `audit_newton_ml2s` | NT uses analytic ML; ULS/DWLS/ADF/DLS use fixed-weight LS |
| ULS | `audit_newton_uls` | Exact analytic (`gmm::moment_quadratic_hessian`), sandwich metric; optional Gauss-Newton |
| GLS | `audit_newton_gls` | Same, using the sample-based NT weight |
| WLS/DWLS/GMM | `audit_newton_wls` / `audit_newton_gmm` | Same, with the supplied fixed weight |
| Ordinary LS-SNLLS | `audit_newton_snlls` | Full expanded LS objective, including eliminated coordinates |
| GLS-SNLLS | `audit_newton_gls` at expanded theta | Full GLS objective |
| FIML | `audit_newton_fiml` | Analytic observed information, using the supplied FIMLPack |
| Ordinal / mixed LS | `audit_newton_ordinal` / `audit_newton_mixed_ordinal` | Full moment objective, delta or theta; gradient differences or explicit GN |
| CatML | `audit_newton_catml` | Correlation-objective gradient differences; Stage-1 thresholds held fixed |
| Two-level ML | `audit_newton_twolevel` | Total-likelihood gradient differences |
| Multi-information penalized ML / FIML | `audit_newton_penalized_ml` / `audit_newton_penalized_fiml` | Gradient differences including the actual penalty |
| Other supplied smooth objectives | `audit_newton_objective` | Gradient differences of the supplied original full-theta objective |



Derivatives retain the objective kind, curvature source, native-to-total
multiplier, N, theta and full Hessian. LS also retains native whitened residuals
and Jacobians. Numerical curvature uses central differences at h and h/2,
checks their discrepancy and pre-symmetrization asymmetry, and retains the fine
steps and effective difference controls. Probes can shrink to stay evaluable;
failed probes or failed reliability checks report unavailable and never switch
to one-sided differences or Gauss-Newton. Callers can instead provide their own
Hessian through `audit_newton_derivatives` and retain its independent provenance.

The total-objective convention is N times the native per-observation objective
for LS, ordinal, CatML, FIML and penalized fits. Two-level's native scalar
objective already equals total negative log likelihood, so its multiplier is
one, not N or one half. The recorded `objective` is total/N in every adapter.
A numerical Hessian costs up to four gradient probes per parameter, plus the
point evaluation (more when steps shrink). It is requested explicitly after
fitting, not imposed on every optimization run.

The returned distance and predicted gain use this declared objective curvature.
For LS, limited-information CatML and penalized fits they are not automatically
sampling-standard-error or likelihood-ratio accuracy claims. Gauss-Newton is
labelled as an approximation and may differ from the actual Hessian away from
zero residuals. The inherited .01 budget is configurable; its ML interpretation
does not establish a cross-estimator calibration. Existing non-ML fit-time
first-order verdicts and R's summary-only Newton helper are unchanged.

```cpp
namespace nf = magmaan::estimate::frontier;
nf::NewtonAdapterOptions request;
request.accuracy.budget = 0.005;
request.differences.relative_step = 1e-4;
request.bounds = actual_bounds;
auto audit = nf::audit_newton_gmm(pt, rep, sample, fit.theta, weight, request);
if (audit && audit->derivatives.status == nf::NewtonAccuracyStatus::Available) {
  const auto& H = audit->derivatives.hessian; // original full-coordinate matrix
  auto policy = audit->options;
  policy.budget = 0.001;
  auto revised = nf::assess_newton_accuracy(*audit, policy); // no recomputation
}
```

## Context

magmaan's Newsom corpus speed survey turned up cases where the L-BFGS Full
GLS fit reaches the *objective value* of the optimum (`ex5_4`:
`f = 0.00301331`, matching SNLLS to 6 digits and lavaan to 5) but is then
discarded as a `LineSearchFailed`. NLopt's `nlopt_optimize` returns
`NLOPT_FAILURE` because its line search cannot find a measurable decrease
step at the floating-point noise floor of a near-perfect-fit GLS objective;
`cpp/src/optim/nlopt_optimizer.cpp` mapped that straight to
`FitError::LineSearchFailed` and discarded the iterate **without checking
whether it was a stationary point**. The gradient-norm computation at the
bottom of the function sat on the success branch only, so the wrapper never
examined the geometry of the point it just rejected.

This is structurally the wrong shape. The principle is well-established in
optimization engineering (Nocedal & Wright; Ceres' explicit projected-gradient
KKT termination; NLopt's own acknowledgement that `ROUNDOFF_LIMITED` may
still leave a useful minimum; lme4's convergence guidance treating optimizer
status as diagnostic, not verdict): **the optimizer's return code is a hint;
first-order stationarity at the returned point is the fact**.

This change introduces a terminal audit as two cooperating layers, both as
plain structs carried on the fit record, with R-side surfacing as nested
`fit$audit` and `fit$diagnostics` sub-lists.

## Principle

**Optimizers propose; the audit disposes.** Classification of the returned
iterate is by *geometry* (first-order stationarity via a projected gradient
or equality-constrained KKT residual), not by the backend's return code. The
two layers separate concerns that the old code conflated:

- **L1 — Optimizer Terminal Audit** lives in `cpp/src/optim/`. Operates in
  **driven coordinates** — the reduced/profiled/lifted space the optimizer
  actually minimized over. Answers "is this point primal-feasible and
  first-order stationary for the problem the backend solved?"
- **L2 — Fit Finalization Audit** lives in `cpp/src/estimate/`. Operates on
  **expanded full θ**. It records admissibility and feasibility, and—when the
  fit path supplies the ordinary full-`theta` objective gradient—audits the
  same terminal differential against the model's covariance-cone geometry.

Two coordinate systems, two questions, one fit record. L1's stationarity
verdict is what `fit$audit$stationary` reports; L2 records what
SE/χ²/robust-correction code needs and exposes the cross-method stationarity
comparison at `fit$diagnostics$geometric_stationarity`. The latter is
additive: it never overwrites the lavaan-compatible L1 result.

## L1: optimizer terminal audit

**Files:**
- `cpp/include/magmaan/optim/terminal_audit.hpp` — declarations of the
  `audit_terminal_iterate` and
  `audit_equality_constrained_terminal_iterate` free functions.
- `cpp/include/magmaan/optim/problem.hpp` — `TerminalAudit` and
  `TerminalAuditOptions` structs (placed alongside `OptimResult` because they
  are part of the optimizer-output contract).
- `cpp/src/optim/terminal_audit.cpp` — implementation.

### Signature

```cpp
TerminalAudit audit_terminal_iterate(
    const ObjectiveFn&     f,
    const Eigen::VectorXd& x,
    double                 reported_f,
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper,
    TerminalAuditOptions   opts = {});

TerminalAudit audit_equality_constrained_terminal_iterate(
    const ConstrainedScalarProblem& prob,
    const Eigen::VectorXd&          x,
    double                          reported_f,
    const Eigen::VectorXd&          lower,
    const Eigen::VectorXd&          upper,
    TerminalAuditOptions            opts = {});
```

The audit recomputes `f(x, grad)` at the returned `x`, builds a projected
gradient infinity-norm against the (possibly ±∞) bounds, applies the configured
absolute or relative stationarity test, records an active-set readout in
driven coordinates, and produces an *advisory* `OptimStatus`.

For equality-constrained scalar problems, the companion audit additionally
recomputes \(h(x)\) and \(J_h(x)\), solves the rank-revealing least-squares
multiplier problem, and tests

```text
|| project_box(grad f(x) + J_h(x)' lambda) ||_inf
```

together with primal equality feasibility. Rank-deficient constraint
Jacobians are allowed and their numerical rank is recorded. With active box
bounds, multipliers are fitted from interior coordinates before the one-sided
bound KKT signs are applied; when multipliers are non-unique this is
conservative. This constrained audit is essential for nonlinear `==` models
and for PSD-ML's Cholesky lift, whose covariance links are nonlinear
equalities even though the resulting primitive covariance matrices are PSD by
construction.

### Options

```cpp
struct TerminalAuditOptions {
  enum class StationarityMode { Absolute, Relative };
  StationarityMode stationarity_mode = StationarityMode::Absolute;
  double absolute_tol      = 1e-3;
  double stationarity_tol  = 1e-6;
  double active_bound_tol  = 1e-12;
  double f_consistency_rel = 1e-6;
  double constraint_tol    = 1e-6;
};
```

- `stationarity_mode` selects the shape of the stationarity test.

  - **Absolute (v1 default)**: `‖Pg‖_∞ ≤ absolute_tol`. The default
    `absolute_tol = 1e-3` was motivated by lavaan's `optim.dx.tol`.
    This is a historical choice, not an equivalence claim: optimizer
    coordinates, parameter scaling, bound treatment, and which termination
    paths receive the check can differ. In particular, the inspected lavaan
    0.7-2 implementation excludes selected exact-bound coordinates rather
    than implementing our full covariance-cone audit. The common half-
    discrepancy objective scale is necessary but insufficient for numerical
    comparability. The current full-model L2 criterion is different again.
  - **Relative**: `‖Pg‖_∞ ≤ stationarity_tol · (1 + |f|)`. The shape an
    earlier revision shipped as the default. The argument for Relative is
    that `|f|` spans many orders of magnitude across the SEM corpus, so an
    absolute threshold may under- or over-reject at the extremes. The
    argument against is that there is no calibration study justifying a
    specific relative tolerance against downstream inference (SE, χ²,
    LRT), so the choice is opinionated. Relative remains available as a
    research-grade alternative, and the test file
    `cpp/tests/unit/terminal_audit_test.cpp` keeps explicit Relative-mode
    coverage so the path doesn't bit-rot.

  This is the first genuinely hard design call in magmaan and the choice
  is genuinely unstable. The mode flag encodes the decision in one place
  so future experiments are one option flip away; the "Tolerance
  calibration" section below sketches the empirical work that would let
  us revisit the default with data instead of a defensive choice.

- `absolute_tol` is consulted only in Absolute mode. `stationarity_tol` is
  consulted only in Relative mode. They are not interchangeable.
- `active_bound_tol` is a coordinate-distance threshold for the
  projected-gradient construction — masking a gradient component, not a
  diagnostic readout.
- `f_consistency_rel` catches the rare case where a backend leaves the
  *last-tried* iterate in `x` rather than the *best-found* one. Relative
  because the same floating-point noise that motivates the audit also
  applies here.
- `constraint_tol` is the maximum equality residual accepted by the
  constrained audit. It certifies the returned point; it does not control the
  optimizer's search tolerance.

### Promotion policy (v1)

The audit is **observation only**. The wrapper still owns the final
`OptimStatus` it returns. The v1 policy:

| Backend reports | Audit says stationary | Wrapper returns |
|---|---|---|
| clean success | yes | `OptimStatus::Converged` (unchanged) |
| clean success | no | `OptimStatus::Converged` (unchanged — v1 does NOT downgrade) |
| soft failure | yes | **PROMOTE** to `OptimStatus::LineSearchSalvaged` |
| soft failure | no | original `FitError` (unchanged) |
| hard failure | (audit skipped) | original `FitError` |

Hard failures (no usable `x` or `f`) bypass the audit. A future PR may add
symmetric downgrade — if a backend declares success at a non-stationary
point, surface that — but this is a semantics-change separate from v1.

### Wired backends

| Backend | File | Soft-failure sites that now audit |
|---|---|---|
| `NloptOptimizer` (NLopt C API) | `cpp/src/optim/nlopt_optimizer.cpp` | `MAXEVAL/MAXTIME`; `FORCED_STOP`; generic `FAILURE`; `ROUNDOFF_LIMITED` (kept as salvaged regardless, audit fills `grad_inf_norm`); SLSQP equality problems use the KKT audit |
| `PortOptimizer` (PORT `drmngb`) | `cpp/src/optim/port_optimizer.cpp` | IV(1)=8 noisy; IV(1)=9 false convergence; IV(1)=10 budget; IV(1)≥11 other; IV(1)=7 singular keeps `SingularConvergence` |
| `IpoptOptimizer` | `cpp/src/optim/ipopt_optimizer.cpp` | unconstrained problems use projected gradients; equality-constrained problems use the same KKT audit and may salvage stationary budget/small-step stops |

Every success path also calls the audit so `OptimResult::grad_inf_norm` has
a single source of truth (the bespoke projected-gradient loops are gone).
The audit costs one extra `f(x, grad)` evaluation per fit — exactly
what the success paths already did inline.

**Not wired in v1:** `PortNlsOptimizer` and `ceres_lm` (residual-driven,
no scalar `ObjectiveFn` to audit); `CeresBfgsOptimizer` (no surveyed
failures motivate it).

### Semantic shift: PORT IV(1)=8

PORT's "noisy gradient detected" status (`IV(1)=8`) was previously a hard
`FitError::NumericIssue`. After v1, IV(1)=8 plus a stationary audit becomes
`OptimStatus::LineSearchSalvaged` (success); IV(1)=8 plus non-stationary
keeps `NumericIssue`. The "noisy" message remains informative about *why*
the audit failed when it does.

## L2: fit finalization audit

**Files:**
- `cpp/include/magmaan/estimate/diagnostics.hpp` — `FitDiagnostics`,
  `DiagnosticsOptions`, `finalize_fit_diagnostics` free function.
- `cpp/src/estimate/diagnostics.cpp` — implementation; reuses the PD-check
  pattern from `cpp/src/measures/fit_measures.cpp:130-135`,
  `estimate::active_bounds`, and `model::ModelEvaluator::sigma`.

```cpp
struct FitDiagnostics {
  std::vector<bool>      sigma_pd_per_block;
  bool                   sigma_pd_all = false;
  double                 lin_eq_residual_inf = 0.0;
  bool                   lin_eq_satisfied    = true;
  Eigen::VectorXd        nl_eq_residual;          // empty when no NL constraints
  double                 nl_eq_residual_inf  = 0.0;
  bool                   nl_eq_satisfied     = true;
  ActiveBoundDiagnostics active_bounds_full;      // indexes full θ (distinct from L1 active_set)
  bool                   snlls_profile_fallback = false;  // see v1 non-goals
};
```

`finalize_fit_diagnostics(theta_full, ev, con, nl, bounds, ...)` is called
from every public `fit_*` entry point that holds a `ModelEvaluator` (i.e.
the MatrixRep paths: `fit_ml`, `fit_gls`, `fit_gmm`, `fit_snlls`,
`fit_snlls_gls`). The FCSEM path uses a different evaluator type and is
excluded — its `fit$diagnostics` is the default-constructed schema slot.

### What L2 *records* vs *gates*

L2 never blocks a fit. It records what downstream consumers need:

- `sigma_pd_per_block` / `sigma_pd_all` — ML/GLS need `Σ⁻¹` so a non-PD
  block invalidates SEs; ULS only needs finite, so a consumer reading this
  decides per-formula. (A blanket gate would wrongly reject ULS fits whose
  Σ has a singular block that ULS does not actually require to invert.)
- `lin_eq_residual_inf` — the K-reparameterization enforces `A_eq·θ = b_eq`
  by construction; a residual exceeding `lin_eq_residual_tol` signals the
  expansion itself is misbehaving (a correctness signal, not an optimizer
  signal).
- `nl_eq_residual` / `_inf` — IPOPT drives `h(θ̂) → 0`; recording the achieved
  infinity-norm tells the user whether the nonlinear constraints were actually
  satisfied (separate from the optimizer's terminal status).
- `active_bounds_full` — the Heywood-case detector: a variance at its 0
  bound has a one-sided derivative, and the standard info-matrix SE for it
  is not valid. Downstream `magmaan_se()` etc. can flag or fall back.

### Common-coordinate covariance-cone stationarity

The lifted Cholesky audit and an ordinary partable-coordinate audit are not
the same first-order question at a singular covariance boundary. For the
scalar analogue (v=ell^2), an inward objective gradient at (v=0) becomes
zero after differentiating through (ell=0). Consequently, equality KKT in
the lifted chart can declare the terminal point stationary even though a
feasible first-order direction exists in the original covariance variable.

`audit_geometric_stationarity()` avoids that chart singularity. It receives
the analytic objective gradient in ordinary full-`theta` coordinates and
computes two normal-cone projection residuals:

1. **Ambient residual:** linear-equality normals, nonlinear-equality tangent
   normals, and active box-bound normals.
2. **Cone residual:** the same normals plus
   $-D_b^*(U_{0b} H_b U_{0b}^{\mathsf T})$, $H_b \succeq 0$, for every
   singular primitive covariance block. Here $U_{0b}$ spans the numerical
   null space of $\Theta_b$ or $\Psi_b$, and $D_b^*$ maps a covariance
   differential back to full model coordinates.

The cone residual is therefore the distance of the objective differential
from the KKT normal cone of the original PSD-constrained model. At a
positive-definite point every covariance null space is empty, so it reduces
exactly to the ambient equality/bound residual.

A scalar residual requires a metric. The declared default is
`model_frobenius`: the product Frobenius metric induced by the assembled
LISREL matrices. Symmetric off-diagonal covariance entries have weight two,
and shared coordinates accumulate every matrix occurrence. This is invariant
to orthogonal changes of basis inside covariance blocks and avoids privileging
the Cholesky chart. It is not invariant to arbitrary changes of measurement
units; such invariance is neither claimed nor numerically possible without a
separate scale convention.

The stationarity verdict thresholds the metric-dual L2 distance, which is
unchanged by isometric changes of model coordinates. The coordinatewise
infinity residual is also retained as a familiar diagnostic, but it does not
drive the verdict because it depends on the selected basis. Both residuals use
the same half-discrepancy objective scale and numerical `1e-3` default as the
L1 audit; this common number does not make the L2 and infinity-norm criteria
identical. Feasibility uses the original
equalities, bounds, and PSD blocks. The multiplier projection is observation
only and never changes fit return semantics. Complete-data ML/GMM/LS, FIML,
PSD FIML, ML2S through its ML/GMM Stage 2, ordinal/mixed-ordinal LS, and CatML
fit paths supply full-model gradients. Profiled SNLLS and extra callback
constraints remain unchecked until their eliminated/extra constraint normals
are available in the common representation.

## R schema

Surfacing happens in `r-package/src/fit.cpp` via two helpers
(`audit_to_r`, `diagnostics_to_r`) called from `fit_result()` — the single
shared assembler for ML/GLS/ULS/WLS/SNLLS — and mirrored in
`fcsem_fit_result()`. Active-bound indices convert 0-based → 1-based at the
R boundary so they index `theta` / `partable` rows directly.

```
fit$audit$
  stationary       (logical)   geometric + primal-feasibility verdict
  grad_inf_norm    (numeric)   projected objective/Lagrangian gradient norm
  raw_grad_inf_norm (numeric)  unprojected objective-gradient norm
  grad_scaled_inf  (numeric)   scale-aware version of grad_inf_norm
  stationarity_rhs (numeric)   configured absolute/relative comparison RHS
  f_recomputed     (numeric)   f at x, recomputed by the audit
  f_consistent     (logical)   |f_recomputed - reported| ≤ rel·(1+|reported|)
  f_finite         (logical)
  constrained      (logical)   equality-KKT audit ran
  constraint_violation_inf (numeric) max primal equality residual
  constraint_jacobian_rank (integer) numerical rank of J_h(x)
  active_set       (integer)   in DRIVEN coords: {-1, 0, +1}
  advisory_status  (character) "converged" / "line_search_salvaged" / ...

fit$diagnostics$
  sigma_pd_per_block      (logical)   one per group
  sigma_pd_all            (logical)
  lin_eq_residual_inf     (numeric)
  lin_eq_satisfied        (logical)
  nl_eq_residual          (numeric)   empty when no NL constraints
  nl_eq_residual_inf      (numeric)
  nl_eq_satisfied         (logical)
  active_bounds_lower     (integer)   1-based θ indices (Heywood detector)
  active_bounds_upper     (integer)
  geometric_stationarity  (list)
    checked               (logical)
    metric                (character) "model_frobenius"
    ambient_stationary    (logical)   equality/bound geometry only
    ambient_residual_inf  (numeric)
    ambient_residual_l2   (numeric)   metric-dual norm; drives verdict
    cone_stationary       (logical)   adds primitive PSD normal cones
    cone_residual_inf     (numeric)
    cone_residual_l2      (numeric)   metric-dual norm; drives verdict
    covariance_nullity    (integer)   summed active null-space dimension
  snlls_profile_fallback  (logical)
```

**Back-compat:** `fit$converged` stays strict (true only for
`OptimStatus::Converged`). Existing R consumers
(`r-package/R/model_data.R:1300` print method; paper harness
`harness-benchmark.R`, `harness-sim-benchmark.R`) keep working bit-for-bit.
Compatibility code reads `fit$audit$stationary` for the driven-coordinate
lavaan-style verdict and `fit$optimizer_status` for the refined status
string. Cross-method studies read
`fit$diagnostics$geometric_stationarity$cone_stationary`.

**Two coordinate systems, two active-set readouts.** `fit$audit$active_set`
is in the driven (reduced/profiled) coordinates the optimizer minimized
over; `fit$diagnostics$active_bounds_lower/upper` indexes the expanded full
θ. They will not match in general — this is the L1/L2 split made visible.

## What the corpus survey revealed

The intended PR success criterion was "Newsom `Full-fail` drops 3 → 1." The
empirical result is *no change in Full-fail count* across all 290 corpus
models — and that turns out to be the more informative outcome.

For `ex5_4` / `ex5_4c` the audit confirms: the iterate sits at the
*objective value* of the optimum (`f = 0.003`, matching lavaan to 5
digits) but the projected gradient there is **0.0015 / 0.0073** — not
machine zero, not noise-floor tiny. The objective surface is near-flat in
some directions of the constraint-reduced α space; the optimizer's line
search stopped because no further `f` decrease was measurable, and the
non-zero gradient remains.

This is exactly the distinction the audit was built to make. The previous
ad-hoc salvage at `max(1e-3, 1e3·gtol)` wouldn't have caught these either
(`0.0015 > 1e-3` borderline, `0.0073 ≫ 1e-3`), so v1
introduces no behavioral regression on previously-salvaged iterates while
adding principled stationarity verification everywhere. The honest
classification of `ex5_4` / `ex5_4c` as **non-stationary** points to the
right next investigation (evaluator accuracy, re-parameterization), rather
than masking the problem with a looser tolerance.

`ex12_3` (a second-edition Newsom script, no longer in the first-edition
corpus) is a separate mechanism — NLopt L-BFGS gets stuck early at
`f = 982` (true optimum `3.06`); PORT and SNLLS both converge it. The audit
correctly reports non-stationary at `f = 982` (large gradient there). Not
fixed by tolerance tuning.

## v1 non-goals (explicit out-of-scope)

1. **Symmetric downgrade.** If a backend reports `Converged` but the audit
   says non-stationary, v1 keeps the reported status. The
   `fit$audit$stationary` field records the driven-coordinate verdict for
   consumers. A separate semantics-change PR can flip this once a survey
   pass confirms no surprise.
2. **`allFit`-style cross-backend agreement** (Layer 3).
3. **LS-backend audit wiring** (`PortNlsOptimizer`, `ceres_lm`):
   residual-driven, no scalar `ObjectiveFn`.
4. **Per-estimator tolerance tuning.** Every backend uses the same
   `TerminalAuditOptions` defaults. The v1 Absolute / 1e-3 default is a
   defensive cross-package match (lavaan), not a calibrated choice; per-
   estimator tuning, or a switch to Relative with a calibrated tolerance,
   may emerge from the "Tolerance calibration" study below if ML / GLS /
   ULS / DWLS / FIML show systematically different gradient noise floors.

## Tolerance calibration (high-priority follow-up)

The full-model model-Frobenius dual L2 cutoff of `1e-3` is a convention,
not a calibrated accuracy guarantee. The older L1 infinity-norm default was
motivated by a lavaan gradient check; reusing that number for the L2 audit
neither reproduces lavaan's criterion nor establishes comparable accuracy
across models. In particular, model-Frobenius geometry removes dependence on
the optimizer's Cholesky coordinates but does not remove measurement units,
objective scaling, dimension, or curvature from the numerical problem.

A scalar example is sufficient: for the half Gaussian discrepancy
`F(v) = (log(v/s) + s/v - 1)/2`, changing units by `y_new = c*y` sends
`v_new = c^2*v` and the gradient to `g_new = g/c^2`, while leaving the
statistical fit and discrepancy unchanged. A fixed absolute gradient cutoff
can therefore change its verdict solely because the measurement unit changes.
Feasibility tolerances and active-eigenvalue decisions need analogous review.

The useful optimization practice is to separate feasibility, stationarity,
and solver termination and declare their scales. For example,
[Ipopt's termination documentation](https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_Termination)
distinguishes scaled overall optimality from separate unscaled feasibility,
dual, and complementarity tolerances. This is guidance on the structure of an
audit, not a reason to borrow Ipopt's numerical defaults for a different norm.
[lme4's convergence guidance](https://lme4.github.io/lme4/reference/convergence.html)
recommends checking tighter tolerances, scaling, derivatives, restarts, and
agreement across optimizers; it also discusses the limitations of Hessian-based
gradient scaling. No universal scalar threshold follows from either source.

Calibration work should proceed as follows:

1. Keep the original full-model domain and objective normalization explicit.
   Record continuous stationarity residuals, feasibility violations, projection
   status, and solver status. Unchecked projection is not failure or success.
2. Test equivalent parameter representations and changes of units on fixed
   fitted points, separately from refitting. Evaluate a declared dimensionless
   scaling convention. An information/curvature metric is a candidate on
   regular identifiable interiors, not an automatic solution at singular or
   PSD-boundary points.
3. Compare candidate residuals against high-accuracy reference fits from
   tightened optimization and independent backends. These are reference fits,
   not known global optima. Inspect objective gaps, standardized parameter
   changes, and implied means/covariances; tiny objective changes alone can
   hide parameter changes in flat directions.
4. Report sensitivity across several tolerances without changing the
   prespecified primary cutoff after seeing which method benefits. Distinguish
   solver return, approximate stationarity at the stated accuracy, admissibility,
   and downstream stability. First-order stationarity does not prove a minimum.
5. Calibrate a practical point-estimation accuracy target on one collection
   and validate on held-out models, units, sample sizes, and boundary ranks.
   Extend to inferential stability only in a separate inference study; the
   current fitting-only experiments do not validate SEs or tests.

The September 2026 SLSQP control diagnostic illustrates why this matters:
stricter objective stopping recovered eight failed stationarity checks in a
30-sample subset, with objective changes at most `1.6e-9`. That is numerical
polishing, not evidence of finding a different likelihood maximum. Keep the
current residual and threshold visible pending calibration; do not silently
relax acceptance or replace the audit by a solver's success flag.

### Accuracy interpretation and boundary follow-up (2026-09-21)

A normal-cone stationarity residual is naturally a **backward error**: at a
feasible point in a closed convex set, it measures the smallest linear
perturbation of the objective gradient that makes that point stationary,
in the chosen metric. It is not a bound on parameter error. In a flat
quadratic, arbitrarily small gradients coexist with appreciable parameter
error. Accordingly, call the current result approximate first-order
stationarity at a declared tolerance, not an accuracy or minimum certificate.

A candidate revised assessment should distinguish:

- normalized feasibility, gradient/normal residual, projection reliability,
  and primal/dual complementarity;
- a dimensionless model metric with an explicit reference-scale convention;
- local accuracy diagnostics and conditioning, when available;
- optimizer termination and any separate recovery policy.

Near a PSD boundary, selecting an approximate nullspace by an eigenvalue
cutoff implicitly approximates complementarity. It does not bound
`trace(Z P)` independently of the dual multiplier magnitude. Record and
calibrate complementarity as well as primal/dual feasibility. In the scalar
variance problem `f(t) = (log(1+t) + .5/(1+t))/2`, `t >= 0`, the optimum is
zero; at `t=1e-10` its strict interior gradient is about `.25`. Thus the
exact normal residual can remain large while the parameter error vanishes.
A metric projected-gradient mapping is a candidate that avoids this hard
active-set discontinuity for convex restrictions, but requires a declared
step scale and an accurately solved joint projection. It is not yet an
implemented replacement, and nonlinear constraints need separate treatment.

In identifiable regular reduced coordinates, a positive-definite Hessian
`H` yields a local predicted step `-H^{-1}g` and predicted objective gap
`g' H^{-1} g / 2`. With our per-observation ML objective,
`N * g' H^{-1} g` approximates both squared displacement in information
units and the remaining twice-log-likelihood improvement. This is a local
approximation, not a global error bound. Inspect the standardized step as
well as the decrement; do not erase unidentified directions with an
unqualified pseudoinverse. Boundary and arbitrary-discrepancy cases do not
inherit the regular ML statistical interpretation.

For calibration, compare likelihood gaps, maximum standardized parameter
and moment changes, and block RMS/Frobenius changes against qualified
same-domain references. The maximum protects against dilution of a single
unstable component; aggregate block errors describe overall movement.
Illustrative budgets `2N*delta_f <= 1e-3` and standardized changes below
`1e-4` are exploratory reporting targets, not approved defaults. Reference
fits require tighter polishing and independent optimizer agreement; if
these fail, label numerical accuracy unresolved. Self-comparison to the
lowest objective returned by a failing portfolio is not accuracy evidence.
Freeze targets and controls before held-out model/unit/rank validation.

Sources: [Dennis and Schnabel, chapter 7](https://doi.org/10.1137/1.9781611971200.ch7),
[Boyd and Vandenberghe, section 9.5](https://web.stanford.edu/~boyd/cvxbook/),
[Bates and Watts (1981)](https://doi.org/10.2307/1268035), and
[Ceres' projected-gradient stopping rule](https://ceres-solver.readthedocs.io/latest/nnls_solving.html).
The inspected current lme4 `checkConv` additionally takes componentwise
minima of raw and scaled gradients and skips remaining gradient checks on
singular fits; its procedure is not a drop-in PSD-boundary audit.

### Regular-interior candidate profile (2026-09-21)

The advisory C++ prototype in `cpp/tests/checks/interior_newton/` uses the
canonical observed information and original ML gradient after linear-equality
reduction. Accuracy budget: Newton distance
`d=sqrt(N*g' H^{-1}g) <= .01`, equivalently total-negative-log-likelihood
EDM <=5e-5. This is a local accuracy approximation. It became part of the
authoritative verdict for complete-data ML on 2026-09-24 (item 3 above). Ineligible boundary/improper points and unreliable curvature remain
separate outcomes. The prototype's numerical conditioning/solve guards are
provisional and do not establish identification.

The paired study covers seven Gaussian model/settings, three sample sizes,
three repetitions, three measurement units, and nine L-BFGS control profiles,
with stock and extended backtracking (3,402 fits). A candidate control profile
is `ftol_rel=1e-12`, `xtol_rel=1e-10`, internal `tolg` default and automatic
memory. Legacy `gtol` names the step test and legacy `history` is not
forwarded to NLopt. Explicit `OptimOptions::nlopt` controls now expose
`xtol_rel`, `tolg` and `vector_storage` independently; see
[optimizer controls](../reference/optimizer-controls.md). Stock line-search
domain failure is a separate unresolved issue.
No production defaults changed. Full findings and limits are in
[the study note](../validation/interior-newton-audit.md).

### Remaining historical implementation items

5. **`fit$converged` boolean semantics:** unchanged.
6. **`snlls_profile_fallback` plumbing:** the flag exists on
   `FitDiagnostics` and surfaces to R, but the v1 SNLLS expand site leaves
   it `false`. Wiring requires a small flag on `GpProblem` — follow-up.
   Documented as a known v1 gap.
7. **No retry / fallback / warm-restart inside the audit.** The audit
   observes; it never re-runs the optimizer. That's Layer 3.

## References

- Implementation roadmap: [`project/architecture/roadmap.md`](../architecture/roadmap.md) — see the "Optimizer backends" section
  for the existing `OptimStatus` / `grad_norm` surfacing that the audit
  now sits behind.
- Open backlog: [`project/backlog/newsom-corpus-failures.md`](../backlog/newsom-corpus-failures.md) — section 1 amended
  after the audit's empirical findings on `ex5_4` / `ex5_4c`.
- Primary bug site (now fixed): `cpp/src/optim/nlopt_optimizer.cpp` — the
  `NLOPT_FAILURE` path that previously discarded any returned iterate
  without examining its geometry.
- Prior art:
  - Nocedal & Wright, *Numerical Optimization* — line-search convergence
    analyzed via gradient-norm / first-order stationarity, not "the line
    search returned success."
  - Ceres Solver: projected-gradient KKT termination — `‖x − Π(x − g)‖∞ ≤
    gradient_tolerance` (equivalent to the per-component active-bound
    masking used here, for box bounds).
  - NLopt manual: `ROUNDOFF_LIMITED` may still leave a useful minimum;
    the API returns `xopt`, `fmin`, and status separately to support
    auditing the returned point.
  - lme4 convergence guidance: optimizer warnings as diagnostics
    cross-checked with gradients, Hessians, and alternate optimizers;
    `allFit` as the practical cross-check pattern.
