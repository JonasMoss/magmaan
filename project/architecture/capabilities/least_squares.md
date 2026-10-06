# Least-squares numerical auditing

## Observed curvature in objective-Jacobian coordinates

Complete continuous moment-GMM adapters retain the total-scale analytic
Gauss-Newton term and observed correction independently:
`H = N J_tilde.transpose() J_tilde + C`. The correction is accumulated from
weighted moment second derivatives, never from subtracting rounded
cross-products. Existing Hessian callers still receive their sum.

Unrestricted equality-reduced and ambient sphere audits use column-scaled,
pivoted QR of the full-rank objective Jacobian. Triangular solves assemble the
congruent observed Hessian from the identity and transported correction. The
Newton step is solved there and transported to original reduced coordinates.
The retained residual is projected through Q to form the score. Sphere
transport includes its normalization-chain correction in both the full
Hessian and independent correction.

Rank loss never authorizes a solve. A clearly negative observed direction may
classify a rank-deficient point as a saddle, using a rounding margin; otherwise
rank uncertainty remains ill conditioned. No ridge or pseudoinverse is used.
Active boxes and PSD-face systems retain their previous paths. Adapters without
an independent correction retain the previous Hessian solve.

The curvature condition describes the equilibrated QR-coordinate Hessian.
The unsquared objective-Jacobian condition and QR reconstruction residual are
retained separately. Statistical weak identification remains present. The
sampling-factor condition cap, solve guard and .01 accuracy budget retain
their existing thresholds. Thin R artifacts expose the coordinate map,
equilibrated curvature, independent correction, Jacobian and transported step.
Owning C++ tests check rounded-cross-product failure, conventional-solve
agreement, decomposition, saddles and rank loss. R tests check ordinary and
sphere artifact identities.

Experiment [engineering/active/15-sphere-reference-fits](../../../experiments/engineering/active/15-sphere-reference-fits/report.qmd),
`uls_curvature` lane, judges 175 fixed points against independent 90-digit
derivatives. Both flat minima are resolved at conditions 1.20/1.59. Every
measured curvature error is below the exact smallest eigenvalue in computed
coordinates; the largest step discrepancy is 1.94e-11 in objective-curvature
units. Residual-construction error remains in those true-point checks and is
reported separately from correction assembly at the recorded residual.

This is development validation, not a runtime forward-error guarantee or
a fitting-default decision. The sampling-metric guard rejects all bank points.
The subsequent construction-bound lane below resolves those numerical minima;
production integration and broader sampled confirmation remain open.

## Conditional numerical intervals

`newton_metric_distance_interval` now verifies QR reconstruction, Q
orthogonality, triangular invertibility and projection arithmetic. It accepts
explicit scaled-factor/residual construction bounds and reports within budget,
above budget or unresolved. `newton_hessian_distance_interval` provides the
corresponding likelihood quadratic calculation using Cholesky perturbation
bounds. Active boxes and PSD faces are unsupported.

R `evaluate_at(..., audit_options=list(retain_newton_artifacts=TRUE))` exposes
a retained-input arithmetic interval for complete-data ML/LS. Optional
`interval_input_errors=list(matrix=...,vector=...)` adds a conditional
construction interval. These artifacts never replace the stored verdict.
ML artifacts respect explicit bounds and use the supplied chart.

See the [numerical interval contract](../../validation/interior-newton-audit.md).
Experiment 15 adds 329 independent point checks, including 15 fresh numerical
minima and all seven finite NTML witnesses. Its dimensional allowance remains
a sensitivity assumption, separate from the construction producer below.

## Calculated construction bounds

`newton_input_error_bounds` independently recomputes covariance-only linear SEM
moments, analytic first/second derivatives and sample roots using outward
long-double intervals. It supports ambient unboxed single-level ULS and
complete-data ML, with retained equality reductions and multiple sample blocks.
The native sphere extension below encloses loading normalization as well.
Means, other weights, active boxes and PSD faces are unsupported.
Primitive fixed/free matrix cells are exact binary64 inputs; derived matrices
are recomputed rather than taken from rounded model-evaluation intermediates.

The producer bounds the target-minus-retained factor, score and observed
curvature in the actual audit coordinates. ULS forms its Gauss-Newton term
after applying the QR map, preserving the separate observed correction.
`newton_curvature_lower_bound` verifies positive curvature using triangular
inverse and Cholesky reconstruction residuals. `newton_input_distance_interval`
requires that proof before composing a distance interval. Uncertain rank,
curvature or interval pivots never authorize a pass; no regularization is used.

R optional `derive_interval_input_errors=TRUE` returns explicit bounds/status
and the derived-input distance interval, preserving the stored verdict.
Experiment 15's `audit_construction` lane checks 329 points at 90 digits:
all 314 available construction bounds and positive-curvature lower bounds cover,
all 47 numerical minima qualify (25 retained ULS, 15 fresh ULS and seven NTML),
and 250 decisive interval classifications are correct. The 79 unresolved points
include 15 independently nonpositive NTML curvatures. Owning tests cover shared
labels, feedback and unequal-sized groups, plus interval rank/saddle controls.

The original default retains its 1e12 production guard; the opt-in policy below
uses the calculated interval instead.

## Opt-in terminal assessment

`audit_convergence_covariance` collects original-objective, consistency and
feasibility evidence alongside the retained ULS/ML audit and its construction
bounds. Set `ConvergencePolicy::require_verified_inputs` on an explicit Newton
policy to require the entire verified interval within budget. Above-budget and
nonpositive-curvature evidence fail; unavailable bounds and unresolved arithmetic
stay unchecked, with no fallback to condition thresholds or first-order checks.
Compatibility behavior and ordinary automatic fit acceptance remain unchanged.

`SphereOptions::verified_newton` adds construction bounds at the native endpoint.
The retained `NewtonSphereMap` allows independent reconstruction of the loading
normalization, its Jacobian and full second-derivative score contraction before
tangent/QR transport. Radial pin curvature is excluded. R exposes this as
`frontier_fit_sphere(..., control=list(verified_newton=TRUE))`; its selected
verdict concerns the native endpoint before translation/polish. The old full
native assessment remains in `gauge$native_audit$compatibility_assessment`.

The ordinary explicit R point assessment is
`evaluate_at(..., audit_options=list(verified_newton=TRUE))`; optional
`reported_objective` checks backend consistency at an endpoint. Both adapters
retain the previous verdict; selected unchecked evidence is NA. A call without
this option preserves the existing assessment. Default adoption, means/other
weights/constraints, and start/search choices remain separate work.


Experiment 15's fresh `audit_terminal` confirmation retains 120 fits across
30 sampled regular/weak/mixed/tied-loading/feedback/two-group problems, plus
120 variance perturbations. All 216 available construction bounds, curvature
margins and all finite distance intervals cover their 90-digit references;
213 decisive classifications are correct. Endpoints yield 106 passes, eight
independently nonpositive curvatures, three above-budget failures and three
unresolved assessments. Mixed-unit native ULS gains three passes with no losses
at identical endpoints. Of the unresolved endpoints, weak-marker sphere ML is
locally accurate (distance .000429), while two mixed-unit sphere ULS endpoints
are above budget (.697/.514). The interval remains conservative in all three.
All 120 perturbation controls remain unaccepted. Fit/evaluation took 6.1 seconds
and independent checking 214.9 seconds on one math thread. Five draws per family
do not settle start/search reliability or justify default adoption.


## Experimental ULS search portfolio

Experiment 15's `uls_search` lane fixes PORT-NLS/sample-unit scaling and the
opt-in audit while comparing actual API default starts, layered starts and four
sample-only signed moment starts. ULS defaults to FABIN3/native, whereas the
preceding terminal study held layered starts common. This is a default-start
comparison under fixed optimizer settings, not a full no-options baseline.

On ten fresh draws per family, native sphere with default plus four signed
starts qualifies 10/10 regular, weak-marker and mixed-unit cases and matches
every best independently refined local reference. Marker qualifies 10/6/8;
tighter controls increase mixed units to 10 but leave weak markers at 6 and
increase cost. Keeping the default avoids a signed-only weak-marker loss.
Across 1,040 retained/fresh cold fits, independent 90-digit checks cover 473
selected endpoints, including every qualified portfolio winner. All 398
available construction bounds and all finite distance intervals cover, with
no wrong decisive classification. Full raw fits are not all independently checked.

Local basin references use independent diagonal-profile ULS refinement after
cold fitting. Initial 75-digit residual refinement failures are retained; a
reference-only amendment to a 40-digit residual at 90-digit working precision
provides every reference, without changing fits or the 0.01 audit budget.
The comparison establishes agreement with best known local minima, not global
optimality. Some winners have negative residual variances or indefinite latent
covariances under this unrestricted target. The two-factor-specific signed recipe
needs broader-model construction and cost evidence; adoption requires a separate
prespecified actual-default decision study. No implementation/default changes.

## Ordinal and mixed numerical integration

Ordinal/mixed numerical integration (2026-10-05, isolated sphere-study) now
retains the total-scale whitened Jacobian, score residual, actual fitting factors
and analytic observed correction before adding Gauss-Newton. Ordinary telemetry
and explicit original full-threshold audits reuse QR curvature and square-root
distances with the existing fitting-weight working metric and acceptance policy.
The opt-in `magmaan_core$frontier_ordinal_newton_audit(fit, theta)` exposes raw
coordinates, the prepared partable and owning artifacts. Ordinal construction
bounds remain explicitly unsupported; this is not a sampling-Gamma or inference
policy change. Experiment 15's fresh confirmation has 160 passing cold fits,
252 agreeing independent 90-digit points and 46 agreeing empirical-threshold/
conditional-weight checks in delta/theta, including shared loadings, groups and
mixed units. All 80 displaced points and 12 saddles remain unaccepted. A corrected
replay retains six earlier sparse-theta failures at distance 0.44–1.11 (three
also ill-conditioned). TASK-33.10.6's focused follow-up (2026-10-06) reproduces
all six and banks an explicit local finite-theta domain limit: three 90-digit
Schur-profiled loading faces have positive tangent curvature and strict outward
descent; 15 finite approach points retain above-budget rejection. Negative
residual delta fits cannot be transported into finite positive-residual theta
coordinates. Tighter L-BFGS recovers none; no global nonattainment theorem or
alternate-basin exclusion is claimed. Fresh regular/sparse controls return 324
passing fits/independent points; two singular first-stage sparse draws reject.
All-ordinal ULS/DWLS/WLS bindings now retain the actual `start$theta` and producer
name `ordinal-simple`. PORT-NLS scalar recomputation telemetry remains unavailable,
so this lane explicitly checks reported versus full-threshold objectives.
The native path scope stops at factor variance 1e6; an ill-conditioned 1e8 probe
retains a diagnostic-distance precision failure and the correct rejection.
Existing first-stage and pairwise
oracle gates pass. Propagated construction bounds, fresh pairwise reliability,
sphere/PSD/barrier coverage, inference policy, identification/global recovery
and default adoption remain outside this bank.

`OrdinalNewtonParts` stores `A = stack_b sqrt(n_b) F_b' Delta_b` and
`b = stack_b sqrt(n_b) F_b' residual_b`, with actual `W_b = F_b F_b'`.
The working metric is `A'A`, and the distance is `sqrt(G' (A'A)^-1 G)` after
applicable equality reduction, with positive full observed curvature required.
The analytic residual correction is retained directly; it is not recovered by
subtracting rounded cross-products. This preserves the fit-time convention
that needs no NACOV, rather than substituting an estimated sampling sandwich.
The explicit point artifact is ambient; active bounds and nonlinear constraints
need their applicable route. Thresholds are included in full even when fitting
profiled them. No new propagated ordinal construction certificate is supplied.
