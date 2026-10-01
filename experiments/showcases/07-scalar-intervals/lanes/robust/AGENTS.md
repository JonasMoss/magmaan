# Implementation requirements

The experiment implements the statistical design in `../../design/robust.md` using existing
library primitives. It is self-contained and does not change the ordinary-user
interval default. Keep the contracts below when extending or rerunning it.

`R/methods.R` owns generation, fitting and inversion; `R/validation.R` checks
normalization, score algebra, API conventions, lavaan agreement and endpoints.
The runner checkpoints deterministic replicate IDs and refuses to resume after
a source, package-binary, seed or cell change. `scripts/freeze_results.R` checks
completed output, method-order invariance and checkpoint reproducibility.
Tiny CSV summaries and provenance under `results/frozen/` are deliberately
tracked so the report remains reviewable; all raw runs remain ignored.

## Boundaries

Restrict the initial work to unpenalized, complete-data ML, correct covariance
specification, the stated populations and two interior scalar targets.
Do not add FIML, ordinal estimation, boundary asymptotics, Bartlett factors,
flips, bootstrap calibration or alternative optimizers to the method grid.
Keep existing core behavior unchanged unless a verified defect or genuinely
missing reusable primitive requires a focused change with appropriate tests.

This experiment owns its orchestration and outputs. Do not source another
experiment, depend on its results, or copy a statistical implementation into
the shared harness. Shared SEM methods belong in core/the compiled R package;
only path, seed, metadata and I/O mechanics belong in `experiments/_support`.

## Statistical contract

Write down the conventions in runner comments and output metadata before the
first smoke run. All methods use the same unrestricted estimate and parameter
coordinates. Identify targets from the fitted partable by label/name. Unit
factor variances make the factor covariance the correlation; do not change
identification conventions or apply a post-hoc Fisher transform to Wald.
Give each candidate a labeled equality restriction while preserving parameter
slots, for example `b == candidate`, rather than reconstructing a model with a
fixed-value parameter that disappears from the free map.

Let s_i be the per-observation Gaussian log-likelihood score, H the
per-observation sensitivity, and J = mean(s_i s_i') the **uncentered** empirical
score second moment. The mean structure and centering convention must match
the actual fitted likelihood. Scale parameter covariance as H^{-1} J H^{-1}/n.
Use the library's casewise scores and derivatives. Do not differentiate an
R reimplementation of the SEM model. Do not mix expected/observed bread,
centered/uncentered meat, or n/(n-1) conventions silently.

The primary methods are:

1. **Wald-O:** the current ordinary-user observed-information sandwich at the
   unrestricted fit, with normal 95% endpoints. Match `policy_inference()`
   covariance numerically before treating a manually composed covariance as
   its replacement.
2. **Score-E:** at every b-restricted fit, use expected H to eliminate nuisance
   scores. Define e_i = s_bi - H_ba H_aa^{-1} s_ai and j = mean(e_i^2).
   Invert S(b) = n * mean(e_i)^2 / j against chi-square-1. Recompute scores,
   sensitivity, nuisance projection and j at every candidate. This is the
   rank-one studentized efficient score; it is not an unrestricted Wald
   approximation.
3. **LR-E:** profile T(b) = 2 * (ell_unrestricted - ell_restricted).
   At the restricted fit use the same expected H and empirical J, with
   h = H_bb - H_ba H_aa^{-1} H_ab and c(b) = j/h. Invert T(b)/c(b) against
   chi-square-1. Equivalently c = (H^{-1} J H^{-1})_bb/(H^{-1})_bb.
   Recompute c at each candidate; do not use the global GOF scaling or
   subtract separately scaled model chi-squares.

Prespecified secondary methods replace O with E for Wald and E with O for
score and LR, preserving every other convention. This yields **six robust
methods**, not a factorial grid of all information/metric/meat combinations.
Uncorrected model-based Wald-E, score-E and profile LR are three diagnostic
arms. Expected and observed variants agree asymptotically under this study's
correct covariance model; that is not a claim of finite-sample equality.

For ordinary score the divisor is h rather than j. For a rank-one robust
score, the same-geometry scaled quadratic and direct sandwich statistic
should agree numerically. Do not treat SB and pEBA as independent scientific
candidates here. A scalar expected-information correction approaching one
under normality is a large-sample check, not an exact sample identity.

Observed H or nuisance blocks that are indefinite/singular, nonpositive h/j/c,
or failed fits are explicit failures. Do not replace them by expected
information, ridge them, truncate eigenvalues or silently switch methods.
This experiment makes no claim of misspecification robustness merely because
an existing API option happens to be named `misspec_scaled`.

## API audit before implementation

Read these local sources; inspect signatures rather than assuming old examples
match the installed package:

- `r-package/R/scores.R`: `prepare_inference`, `scores`,
  `inference_information`, `parameter_covariance`, `policy_inference`,
  `score_components`, `project_scores`, `score_sandwich` and explicit matrix
  adapters. Check full-model derivatives evaluated at the restricted point;
  an H0-only reduced parameter vector loses the tested direction.
- `r-package/R/nested_test.R`: `nested_score_test(H1, H0, data, sensitivity)`.
  Its generic `p_value` currently selects pEBA4. Select the intended scalar
  statistic/reference explicitly and verify it against the contract above.
- `magmaanlab::magmaan_core$frontier_profile_lrt_parameter_ml` and
  `frontier_profile_lrt_ci_parameter_ml`; definitions in
  `r-package/R/RcppExports.R` / `r-package/R/zzz_core.R` and
  `cpp/include/magmaan/estimate/fit.hpp` / `cpp/src/estimate/fit.cpp`.
  Audit the actual meat and normalization behind `robust = TRUE`,
  `reference = "robust_scaled"` and `reference = "misspec_scaled"` before
  mapping them to LR-E/LR-O. An API name is not proof of identical algebra.
- `cpp/include/magmaan/api/policy.hpp`: current covariance-policy contract.

Produce an API-to-method mapping in runner comments and run metadata. If a
landed profile option uses a different empirical meat, retain an explicit
diagnostic label or compose the specified correction from existing primitives;
do not silently relabel it. Verify actual accepted reference strings before
calling them. Missing access to full-model restricted derivatives is a
specific adapter gap to solve, not a reason to substitute Wald for score.

## Execution sequence

1. Implement deterministic population generation exactly as in the report.
   Check analytical covariance and moments, finite fourth moments, target
   identification and factor orientation. A separate large generated sample
   checks the implementation within Monte Carlo error. Never rescale each
   analysis sample to the true covariance.
2. Record the unpenalized fit/domain/bounds policy. Fit one normal and one
   skewed sample and verify all method ingredients against the above algebra,
   including n versus n-1 and half-discrepancy scaling. Validate a normal LR
   against installed lavaan under matching options. Validate restricted score
   derivatives independently by finite differences. Check Wald-O against the
   ordinary-user policy and the scalar score/scale identity.
3. Build candidate evaluation and inversion with deterministic starts/fallbacks.
   Log every candidate's target, constraint residual, convergence, raw and
   corrected statistic, correction and conditioning diagnostics. Require
   library convergence, constraint residual <= 1e-6, target root tolerance
   1e-5 and endpoint statistic gap <= 1e-4. Do not clip negative LR values
   beyond documented floating-point tolerance or hide failed optima.
4. Expand brackets outward from the estimate, refine both crossings and audit
   a wider grid for extra crossings. Preserve disconnected acceptance sets;
   do not call their convex hull a CI. If the numerical search cannot establish
   the set, label it unresolved. Distinguish domain endpoints, unbounded
   endpoints and bracket/search failure. No clipping Wald to the correlation
   domain for cosmetic agreement. Record out-of-domain Wald endpoints.
5. Test truth-in-set versus the pointwise test at truth for every inversion;
   flag differences within numerical tolerance separately. Re-evaluate both
   endpoints independently. Do not use the known truth during inversion.
6. Implement `--smoke`, `--reps`, `--seed-base`, `--cells`, `--output`, resumable
   replicate IDs and a timing estimate with progress. Run the two-replicate
   smoke and numerical checks before the 50-replicate pilot and full design.
   Use fresh confirmation seeds after fixes as specified in the report.
7. Write results and the actual report. Retain per-method failures and paired
   comparisons; a method cannot win by dropping difficult samples. Separate
   main results from information-choice diagnostics and normal controls.

## Required artifacts and completion

Keep generated data in `results/<run-id>/`. Write `metadata.csv` (command,
source commit/dirty state, package versions, seeds, population, method mapping,
bounds, optimizer and tolerances), `intervals.csv` (cell/replicate/target/method,
estimate, truth, endpoints or set components, validity, failure reason,
coverage, tail misses, width and refit counts), `candidates.csv`,
`validation.csv`, `summary.csv`, `paired.csv` and `timing.csv`. Store exceptions
as rectangular failure rows and preserve shared fit failures for all methods.
Random streams and results must be invariant to method iteration order.

Coverage conditional on success, successful-and-covering fraction of attempts,
common-valid paired differences with Monte Carlo uncertainty, domain events,
and runtime all belong in the report or its local result tables. Full grids
stay in CSV. Do not select methods or alter the preregistered populations using
the same pilot outcomes and then present a confirming run as untouched.

Run appropriate numerical/binding checks for any core change, refresh vendors
if canonical C++ changes, and run `just check-layering` and `just check-tracked`.
Update the active backlog for concrete missing primitives or completed scope;
update the roadmap only if implementation contracts change. Commit explicit
paths on the current branch, preserving unrelated edits. No cross-experiment
imports or private/session handoff files belong in this folder.


## Recorded implementation choices

The profile primitives with `robust_scaled` / `misspec_scaled` center empirical
contributions at the sample covariance. This study instead composes its fixed
uncentered-score correction from `scores()`, the matrix score adapter,
`project_scores(center=FALSE)`, `score_sandwich()` and `score_spectrum()`.
For a scalar candidate the two scales obey
`c_uncentered = c_centered + ordinary_score / n`; validation checks this
relationship rather than claiming the existing profile helpers are aliases.
`parameter_covariance()` applies the fit's equality constraints, so it is used
only at the unrestricted fit for Wald. Restricted score/LR geometry uses the
full parameter derivatives and an explicit nuisance projection.

Fitting uses SLSQP without bounds, PSD penalties or repairs; the implied
covariance must be positive definite. Both factor signs remain anchored by
positive first loadings. Primitive inadmissibility is recorded, including
latent correlations outside [-1,1]. The wider audit extends half an interval
width beyond each endpoint, samples nine equally spaced points plus the
estimate, and requires two acceptance transitions. It cannot certify the
acceptance set outside that finite grid. Failure there invalidates the reported
interval even when both local endpoints passed their checks.

The original-seed 50/cell pilot is retained locally as `results/main/`. A logging
refinement then preserved a successful constrained-fit convergence verdict
when later information evaluation failed. The confirming 50/cell pilot and its
500/cell extension use seed base 2026132751 and identical computation in
`results/confirmation/`; the pilot summary remains in its `pilot-50/` folder.
The fixed populations, methods and tolerances were not tuned from coverage.
