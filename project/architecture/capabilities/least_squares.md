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
Usable uncertainty bounds, the flat finite NTML witness and fresh confirmation
remain open.
