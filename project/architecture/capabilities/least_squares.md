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
Means, other weights, native sphere normalization and PSD faces are unsupported.
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

This implementation settles the covariance-only construction-bound mechanism.
Terminal-verdict integration, sphere transport, broader model/sample controls
and any default decision remain separate. The current 1e12 production guard
still rejects all retained ULS controls and the flat NTML minimum.
