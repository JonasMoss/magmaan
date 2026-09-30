# Implementation requirements

This folder implements the normal parameter-interval experiment.
The statistical design and run sizes are fixed in `report.qmd`. Preserve the
validated construction and source fingerprints when extending it; use fresh
confirmation seeds after statistical or numerical changes. This is an exploratory study, not authorization to change
ordinary-user inference defaults.

## Scope and dependencies

Use existing core/R bindings for ML fitting, derivatives and parameter
covariances. Experiment-local R may orchestrate candidate refits, root searches
and the Bartlett bootstrap. Do not implement parallel SEM fitting or sandwich
algebra in R when the library exposes it. Do not source or read another
experiment; this folder must remain an independent leaf. Lift genuinely shared
statistical primitives into core or the compiled R package, not the experiment
harness. No manuscript, private project or sibling results are dependencies.

Inspect `r-package/R/scores.R`, `r-package/R/nested_test.R`, and the
`magmaan_core$frontier_profile_lrt_parameter_ml` and
`magmaan_core$frontier_profile_lrt_ci_parameter_ml` bindings. These are starting
points, not a promise that every needed adapter is already exported. Use
labeled equality constraints (`b == candidate`) to preserve parameter slots;
fixed-value syntax may change the nested-model map. With unit factor variances,
the fitted factor covariance is the correlation target. Verify sign orientation
and identify free parameters by names/labels, never hardcoded indices.

For the normal score, at each restricted fit partition per-observation expected
information H between the tested scalar b and nuisance a. The efficient score
is e_i = s_bi - H_ba H_aa^{-1} s_ai, with information
h = H_bb - H_ba H_aa^{-1} H_ab. Invert n * mean(e_i)^2 / h against chi-square-1.
Do not freeze the unrestricted information and call the resulting Wald
approximation a score interval. Audit all n and 1/2 objective conventions.

## Implementation order

1. Record fitting/domain/bounds conventions, population covariance and target
   mapping. Check analytic covariance against an independently generated large
   sample and record simulation error rather than enforcing sample covariance.
2. Validate one ordinary ML profile-LR statistic against an independent lavaan
   constrained/unconstrained likelihood calculation under identical options.
   Validate one score restriction against the library's nested-score surface
   and an independent finite-difference derivative check. Confirm information
   scaling on a simple analytically checkable normal model.
3. Implement endpoint bracketing/refinement and candidate logging. At every
   candidate require library convergence and equality satisfaction. Verify LR
   nonnegativity up to numerical error; do not hide a bad optimum by clipping.
4. Verify ordinary endpoint statistics within 1e-4 of the cutoff, equality
   residuals within 1e-6 and target root tolerance 1e-5. Search outward on both
   sides of the estimate; scan the acceptance region for extra crossings.
   Report disconnected sets, unbounded ends and domain-limited ends explicitly.
   A failed bracket is not automatically an infinite or boundary endpoint.
5. Implement candidate-specific Bartlett resampling from the restricted normal
   covariance. Keep fixed base normal draws across candidates, deterministic
   warm-start/fallback rules, and separate data/bootstrap random streams.
   Every bootstrap replicate needs an unrestricted and a restricted refit.
   Record B attempted, B valid, failure reasons, mean LR and its MCSE.
6. Retry failed bootstrap fits once with a predetermined fallback start. If any
   bootstrap replicate still fails, mark that candidate correction unavailable
   in the primary pilot. Do not silently estimate a conditional-on-convergence
   LR mean or redraw until B successes. Such a diagnostic may be saved under
   a separate label. Nonpositive/nonfinite correction factors fail explicitly.
7. For corrected endpoints check T(b)/c(b), not the ordinary cutoff equation.
   Re-evaluate with the same bootstrap stream and test stability under the
   larger B and independent stream specified in the report. An optimizer/root
   tolerance is not a guarantee of bootstrap precision. Do not smooth away
   failures or silently freeze c across candidates.
8. Implement `--smoke`, `--main`, `--bartlett-pilot`, `--reps`, `--seed-base`,
   `--bootstrap-reps`, `--output`, and resumable replicate IDs. Measure pilot
   time before the nested bootstrap; report projected time and progress.
9. Run validations and smoke, then the prespecified stages. Keep full runs
   local. Replace the report's planned status only with observed results.

## Outputs and completion checks

Write metadata (source commit/dirty state, package versions, full command,
seeds, method definitions and tolerances), per-dataset/per-target/per-method
interval rows, candidate diagnostics, bootstrap correction diagnostics,
failures, timing and summary CSVs under the selected `results/<run-id>/`.
Use stable IDs to pair methods. Bootstrap seeds must not depend on method
iteration order. Save failed rows; truth outside an interval is not a failure.

Check truth-in-set against the pointwise test at truth for every inversion.
Do not use truth to set bounds, choose starts, tune corrections or find roots.
Ordinary and corrected LR are separate methods. No Bartlett factor derived
for a global goodness-of-fit statistic may substitute for the scalar profile
factor. Do not transfer the LR correction to Wald or score tests.

Run relevant binding/numerical checks for any changed primitives, plus
`just check-layering` and `just check-tracked`. Commit explicit completed
changes on the current branch, preserving unrelated work. If implementation
reveals a missing primitive, record the concrete gap in the active backlog;
do not broaden this into a new inference-policy project.
