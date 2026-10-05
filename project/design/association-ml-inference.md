# Association-ML inference contract

TASK-32.1 audits the fitter and specifies implementation gates; TASK-32.2
implements the lab evaluation-point covariance component, and TASK-32.3 adds
lab global/nested reference laws. Ordinary exposure remains gated by subcards
4–5. The ordinary policy must satisfy the
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
by the lab IJ comparator. Association-ML transport is implemented by
`estimate::frontier::association_ml_ij` in `ordinal_association.cpp`, exposed
as the lab `association_ml_ij()`. It returns the evaluation-point score,
observed sensitivity (central differences of the analytic correlation score),
unweighted group target derivatives, active/full influence and covariance.
The full influence adds within-stratum threshold rows divided by w_b, retaining
threshold-parameter cross-covariance. A supplied-channel adapter remains open.
Subcard-1 nonnormality and stratified delete-one gates and subcard-2 validation
are recorded in `ordinal_ij_test.cpp`. The non-Gaussian copula gate uses centered
chi-square factors (3 df) and errors (5 df), with three categories and two
strata sharing loading constraints. Replicated case-weight central differences
at two steps keep the existing pairwise solver resolution visible and retain
the 1e-5 relative-error gate. Stratified delete-one covariance errors against IJ
are 0.117626, 0.029646 and 0.00728114 at total N = 250, 1000 and 4000. These
are deterministic numerical validation, not policy coverage calibration.
Subcards 1 (the exact sampling channel and its outstanding gates) and 2 (lab
score/sensitivity/covariance) are implemented; subcard 3 adds lab spectral laws
and subcards 4–5 remain open.
The lab `association_ml_global_test()` reports T = 2N q and All (negative
weights truncated), SB and PEBA4. For correlation basis E_j, its local null
metric is V_b[j,k] = w_b tr(C_b^-1 E_j C_b^-1 E_k)/2. With Delta the active
correlation tangent, U = V - V Delta (Delta' V Delta)^-1 Delta' V. Independent
strata have Cov(sqrt(N) r_b) = Gamma_b/w_b, where Gamma_b selects association
rows after the joint empirical Stage-1 solve. Consequently the reference
spectrum is eig(U blockdiag(Gamma_b/w_b)); no extra factor of two is needed.
The metric is evaluated at fitted C: under the global null it agrees with the
observed curvature at the population target. It does not give a global null
law under misspecification.

The lab `association_ml_nested_test()` reports T = 2N(q_null-q_alt). In the
alternative's active coordinates the exact parameter restriction A gives
C = A H^-1 A' and S = A H^-1 B H^-T A'; its spectrum solves S v = lambda C v.
This targets restrictions true at the larger model's pseudo-true parameter,
using observed H even when the larger model is misspecified. Thresholds are
held at their common Stage-1 values when embedding the null. Moment-only
nesting is refused; no tangent fallback is enabled. Both R fits must use the
same supplied raw rows, Stage-1 targets, group order and sizes. Zero df is a
typed unavailable reference (`zero_df` in R), with empty spectrum and NaN
p-values. Penalties, missing raw rows, fixed/restricted thresholds, nonlinear
constraints, covariance faces and singular active information remain refused.
Independent explicit metric/Gamma and restriction reconstructions, exact-fit,
saturated, identical-model, one-restriction and identical misspecified-stratum
gates live in `ordinal_ij_test.cpp`. These lab references do not select an
ordinary-user policy; MI/releases and calibration remain subcards 4–5.

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

## Ordered implementation subcards

1. **Exact all-ordinal Stage-1 sampling channel (complete raw-row channel).**
   Explicit empirical rows and sampling Gamma leave fitting NACOV unchanged.
   Acceptance gates completed by TASK-74 and TASK-32.2: marginal and
   pairwise score/Jacobian finite differences; independent replicated case-weight
   derivatives under latent nonnormality; threshold coupling and centered rows;
   stratified delete-one convergence; typed missing-data/invalid-Gamma refusals.
2. **Association evaluation-point score, sensitivity and covariance (complete
   lab component).** Supplies s, H, D and joint influence in active/full
   coordinates. Acceptance gates completed by TASK-32.2: independent
   q/gradient/Hessian and target derivatives; influence versus reweighted refits;
   exact-fit fixed-metric sandwich reduction; grouped linear constraints,
   rank-deficiency refusals and unit/coordinate transport. Gate threshold
   cross-covariance as well as active parameter covariance.
3. **Global and nested reference laws (complete lab component).** Compose spectral laws and reportable
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
