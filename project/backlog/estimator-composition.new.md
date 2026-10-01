# Shared fitting composition: foundation merge update

Completion recorded 2026-10-01. The initial decisions and dependency order
have been merged into the
[roadmap](../architecture/roadmap.md#shared-fitting-composition) and
[active TODO](todo.md#shared-fitting-composition). This file is a focused
completion record for the maintainer editing those documents; fold it into
them and remove it after the foundation validation is recorded there.

## Foundation implementation

`model::MomentTarget` distinguishes covariance and correlation targets.
`model::correlation_evaluation()` projects grouped covariance values and their
optional vech Jacobians, drops mean moments, and fixes diagonal correlations
and their derivatives to one and zero respectively. Positive finite variances
are required; the consuming discrepancy owns the definiteness requirement.

`estimate::ml_objective()` accepts either target while retaining its existing
two-argument covariance entry point and half-discrepancy scale. Correlation
inputs must already have unit diagonals and no mean moments; input moments are
not repaired. Ordinary and PSD catML now use the same projection and ML kernel
as the covariance route. Their wrappers and fitting behavior remain available.

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
implemented contract above and these validation gates. The ordinal association
contract, unified dispatcher and shared penalty fitting/finalization remain
open; the scalar objective alone does not establish those fitting or inference
contracts.

## Next slice

Define the ordinal association-model contract before exposing ML on prepared
ordinal moments through `fit_model()`: keep Stage-1 thresholds saturated,
exclude inactive threshold/mean/scale coordinates from optimization, use the
active association-Jacobian rank for df, and reject unsupported constraints.
Validate overidentified, grouped and constrained models and preserve delta/theta
semantics and unchanged Stage-1 moments. Dispatcher/metadata consolidation and
barriers across retained fitting routes follow the order already in the TODO.
