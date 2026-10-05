# Association-ML inference contract

TASK-32.1 audits the existing fitter and proposes implementation gates; no
inference is enabled by this note. The ordinary policy must satisfy the
[misspecification requirement](../scope.md#misspecification-robust-inference-requirement).
Initial scope is independent complete rows, all-ordinal indicators, saturated
thresholds, fixed response scales and linear association constraints. Missing
pairwise supports, mixed indicators, nonlinear restrictions, active PSD
boundaries and penalties require separate contracts and retain typed refusals.

## Estimand and existing implementation

Stage 1 estimates marginal normal-quantile thresholds and pairwise polychoric
correlations. `data::ordinal_stats_from_integer_data` delegates to
`pairwise_ordinal_stats_from_integer_data` in `cpp/src/data/ordinal.cpp`.
`OrdinalStats` (`cpp/include/magmaan/data/ordinal.hpp`) stores correlations,
thresholds, NACOV, OPG-convention moment influence/bread and integer cases.
These sampling conventions must not be silently relabelled as exact empirical
estimating-equation influence under latent distribution misspecification.

Stage 2 minimizes, with group proportions w_b = n_b/N,

    q(theta, r) = (1/2) sum_b w_b [log|C_b(theta)|
                  + tr(R_b C_b(theta)^(-1)) - log|R_b| - p_b].

Here C is the model-implied correlation matrix and r stacks the strict lower
triangles of the Stage-1 R matrices. This is Gaussian ML *on estimated
associations*, not the ordinal casewise likelihood. There is no fitted NACOV
weight: NACOV affects sampling inference only. DWLS instead minimizes a
threshold/association quadratic with an estimated diagonal NACOV weight.
`ml_prepare`, `ml_value_gradient` and `ml_objective` in
`cpp/src/estimate/nt.cpp` provide the discrepancy and gradient;
`model::correlation_evaluation` supplies its correlation target and Jacobian.
the ordinal `fit_ml` and `fit_ml_psd` routes in `cpp/src/estimate/fit.cpp`
use this target (optimizer `fmin = q`, discrepancy statistic `2N q`).

`validate_ordinal_association_model`, `ordinal_association_layout` and
`ordinal_association_info` in `cpp/src/estimate/ordinal_prepare.cpp` enforce
saturated Stage-1 thresholds, remove them from optimization via `EqConstraints`,
and count/rank the active association Jacobian. Threshold coordinates remain
in full theta but are data-driven nuisance estimands, not zero-variance fixed
parameters. `LatentStructure` carries the restrictions, `LatentNames` and its
`row_user` mask preserve user declarations, and `Starts` supplies hints; no
new estimand belongs in R formatting or labels.

## Sampling and inference recipe

Use active coordinates alpha with theta = theta_0 + K alpha. Define
s = partial q / partial alpha, H = partial s / partial alpha (the observed
Hessian, including correlation-map curvature), and D_b = (partial s / partial
r_b)/w_b (the unweighted group target derivative). Obtain centered Stage-1 influence g_bi by differentiating the stacked
marginal and pairwise estimating equations and solving their **empirical
Jacobian**, including threshold-to-polychoric coupling. Select association
rows only *after* that solve. Threshold uncertainty is thus transported even
though q has no direct threshold argument. For stratified groups,
B = sum_b w_b D_b Gamma_b D_b', Gamma_b = mean_i g_bi g_bi'.
Then Cov(alpha_hat) = H^(-1) B H^(-T)/N; parameter influence is
-H^(-1)D_b g_bi. Transport through K and, when reporting stored thresholds,
through the joint Stage-1/Stage-2 influence, retaining cross-covariances.

`data::mixed_moment_sampling_influence` in `cpp/src/data/ordinal.cpp` and
`MixedOrdinalStats::sampling_moment_influence` are the TASK-67 template.
TASK-74 adds `data::ordinal_moment_sampling_influence` for complete all-ordinal
data, returning centered empirical rows and their sampling Gamma, also exposed
by the lab IJ comparator. Association-ML transport and a supplied-channel
adapter remain to be implemented; the broader subcard-1 nonnormality and
stratified delete-one gates remain open.
Lavaan NACOV/OPG is a compatibility comparator only. A caller-provided NACOV
without a declared, validated sampling meaning cannot grant ordinary inference.
Unlike DWLS, this criterion has no estimated-weight/Gamma-influence channel;
D transports the estimated correlation target, not an estimated weight.

| Component | Required composition and reusable pieces |
| --- | --- |
| Covariance/Wald | Observed H and empirical B above, joint threshold transport; reuse correlation evaluation and equality-coordinate machinery, not Gaussian raw-data score meat. |
| Global fit | Null asserts the Stage-1 association target is in the model. Derive the residual-projection quadratic from the local ML metric and empirical association Gamma; use its exact weighted-chi-square spectrum. `2N q` has no automatic chi-square law. Under this null observed and expected curvature coincide; thresholds contribute no extra fitted df. |
| Nested tests | Same Stage-1 data, ordering and group weights; restrictions concern the larger model's pseudo-true association parameters. Use observed H for nuisance projection and the restriction influence covariance. A positive definite score metric may be expected information, with its matching spectrum. Derive the `2N(q_null-q_alt)` law separately; a DWLS difference or Gaussian LR reference is insufficient. |
| MI/releases | Differentiate q in an augmented association direction; use observed sensitivity for nuisance projection, empirical projected meat for the robust score law, and a declared metric for the statistic/EPC. Test fixed/absent association rows and linear equality releases; threshold/scale releases change the estimator and remain refused. |

`ordinal_rbm_parts`, `ordinal_casewise_influence_ij` and
`robust_ordinal_ij` (`cpp/src/estimate/ordinal_robust.cpp`) illustrate the DWLS
policy's influence/spectrum composition, but are LS workers, not association-ML
implementations. Candidate enumeration and release directions in
`cpp/src/estimate/ordinal_score.cpp` can be reused after separating LS kernels.
`require_ls_ordinal_estimates` there, `robust_ordinal` in ordinal_robust.cpp,
and dispatch in `cpp/src/api/sem.cpp` and `cpp/src/api/policy.cpp` deliberately
refuse current association inference. Preserve those guards until gates pass.

## Ordered implementation subcards (proposed)

1. **Exact all-ordinal Stage-1 sampling channel.** Add explicit empirical rows
   and sampling Gamma without altering fitting NACOV. Acceptance: marginal and
   pairwise score/Jacobian finite differences; independent replicated case-weight
   derivatives under latent nonnormality; threshold coupling and centered rows;
   stratified delete-one convergence; typed missing-data/invalid-Gamma refusals.
2. **Association evaluation-point score, sensitivity and covariance.** Add s,
   H, D and joint influence in active/full coordinates. Acceptance: independent
   q/gradient/Hessian and target derivatives; influence versus reweighted refits;
   exact-fit fixed-metric sandwich reduction; grouped linear constraints,
   rank-deficiency refusals and unit/coordinate transport. Gate threshold
   cross-covariance as well as active parameter covariance.
3. **Global and nested reference laws.** Compose spectral laws and reportable
   availability. Acceptance: saturated global zero-df and exact-fit quadratic
   reductions; independent matrix reconstruction of spectra and normalization;
   identical-model/one-restriction/grouped controls; nested pseudo-true restriction
   under larger-model misspecification. Compare lavaan only where its estimator
   and recipe demonstrably match; otherwise use independent derivative and
   quadratic references, without inventing a parity exemption.
4. **Association MI and linear releases; C++/R composition.** Acceptance:
   independent one-direction Schur reconstruction, augmented-model gradient and
   finite-difference EPC gates; grouped absent/fixed rows and equality releases;
   agreement with the nested score for the same restriction; C++/both-R results
   and typed unsupported threshold/scale/boundary/penalty routes. Remove only
   the guards covered by these gates.
5. **Frozen policy calibration before ordinary exposure.** Predeclare covariance
   coverage and global/nested/MI size panels over N, category imbalance, group
   constraints and latent nonnormality/model misspecification. Acceptance:
   reproducible frozen evidence, Monte Carlo uncertainty, failure accounting and
   recorded reporting decisions. Calibration cannot substitute OPG/expected bread
   for a required consistent recipe. Pricing/long runs need separate authorization.

The existing [MI capability matrix](../validation/capabilities.md) remains
unsupported until implementation and calibration establish these contracts.
