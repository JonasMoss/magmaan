# magmaan scope

Status: adopted scope decision, 2026-10-01.

This document owns the statistical targets and boundaries of magmaan's
development programme. The [roadmap](architecture/roadmap.md) owns implemented
capabilities and contracts; the [active backlog](backlog/todo.md) owns accepted
work; the [trigger register](backlog/speculative.md) owns banked extensions.
The [R-interface vision](design/r-interface-vision.md) owns the package split
and ordinary-user policy. A scope decision does not establish implementation,
validation coverage or a new default.

## Purpose and primary workflows

magmaan fits linear SEM and develops inference for population approximation
parameters. A fitted model can be a working approximation rather than the true
data-generating mechanism. Covariance restrictions, regression coefficients and
derived quantities retain estimator-specific interpretations; a structural
coefficient does not acquire a causal interpretation merely by appearing in a
SEM.

The current 0.2.0 programme is single-level normal-theory ML, FIML and
all-ordinal DWLS; PSD covariance constraints keep their current fitting and
boundary-inference contract. magmaan's own fitting reliability, PSD hardening,
mixed continuous/ordered completion and barrier-specific hardening/inference
are assigned to 0.3.0.
Noniterative development is indefinitely postponed and requires an explicit
user scope decision to reopen; existing APIs and regression gates remain.
The roadmap owns the estimator tiers and supported
model slices. Broader model families remain governed by their existing triggers;
this document does not expand them.

## Population target and sampling law

For independently and identically sampled observational units, the target is
an estimator-specific functional of the joint population distribution. For
example, an M-estimator may target

$$
\theta(P) = \operatorname*{argmin}_{\theta} E_P[\ell(Z;\theta)].
$$

Moment estimators and penalties need their own target definitions, including
population weights and the penalty's asymptotic scaling. Missing-data routes
also need an observation-law and identification contract. Identifiability,
regularity, finite moments and an applicable central limit theorem are required
for the usual asymptotic inference; robustness to model misspecification does
not remove those conditions. This functional interpretation follows
[Buja et al., *Models as Approximations II*](https://arxiv.org/abs/1612.03257).

When covariates X are part of the sampled unit, their sampling variability
belongs in inference for the population target. The best linear approximation,
for example, is

$$
\beta(P) = E_P[XX^\mathsf{T}]^{-1}E_P[XY].
$$

Under a nonlinear conditional mean this target depends on the covariate
distribution. Variation in sampled X contributes uncertainty even if Y is a
deterministic function of X. See
[Buja et al., *Models as Approximations I*](https://arxiv.org/abs/1404.1578).

Independent clusters can replace independent rows only through a separately
supported sampling and influence contract. Time-series dependence, arbitrary
dependent designs and design-based survey or randomization inference are not
implied by a robust covariance label. Ergodicity alone does not specify a
central limit theorem or an estimable variance.

### Misspecification-robust inference (requirement)

Adopted 2026-10-03. Every default inference component of the ordinary policy
must be misspecification-robust: its covariance or reference law must be
consistent whenever the component's own null hypothesis holds, including when
the fitted model is misspecified, unless that null itself asserts correct
specification. This is a requirement, not a preference among calibrated
options. What it demands depends on what each component's null asserts.

| Component | What its null asserts | Consequence |
| --- | --- | --- |
| Parameter covariance, Wald tests and intervals | Nothing about model correctness: the target is the pseudo-true $\theta(P)$ | Sandwich with observed-information bread and empirical score or moment meat; estimated-weight influence for data-dependent weights (DWLS, GLS, WLS) |
| Nested tests (score, likelihood ratio, fit-function difference) | The restriction holds at the larger model's pseudo-true value; the larger model may be wrong | Observed-information sensitivity and spectra; estimated-weight laws for least-squares estimators (the DWLS profile law). Fixed-weight or expected-information versions are comparators only |
| Global goodness-of-fit tests | The model is correct | Consistency is needed only under that null. With complete data, the expected normal-theory information is the limit of the observed Hessian there, even for nonnormal data, so it is admissible and simulations choose. Under MAR missingness the pattern-conditional expected information is not the Hessian even for a correct model, so FIML uses observed information |
| Score-test metric | A free choice: any fixed positive definite metric with its matching spectrum gives a consistent test | Expected information is allowed, and is used for stability |

Consequences:

- A recipe that is inconsistent under misspecification does not become a
  default because it calibrates better in correctly specified finite samples.
  Robust tests can be less accurate at small N than model-based ones; that
  cost is accepted.
- Calibration studies document the adopted robust recipes; they do not select
  inconsistent alternatives. A poorly calibrated robust recipe calls for
  finite-sample work that preserves consistency (corrected references,
  multiplier or bootstrap calibration), which stays research-tier until
  validated.
- Explicit lavaan compatibility output (`lavaan_compat`) reproduces lavaan's
  recipes, robust or not, and is labelled as compatibility rather than policy.
- The robustness concerns the structural model. Identifiability, regularity
  and the sampling law above remain required.

### Group allocation and likelihood-score covariance

Settled 2026-10-01: the sampling law determines whether between-group score
means belong in uncertainty. This is a covariance identity, not a centering
parameter to select by simulation. Let $w_g=n_g/N$ and
$m_g=E_g[s(Z;\theta^*)]$, with $\sum_g w_g m_g=0$ at the population
approximation target. For independent observations, identically distributed
within each of a fixed finite number of groups, with fixed counts and
$n_g\to\infty$, the per-observation covariance
of the summed score is

$$
B_{\mathrm{fixed}}=\sum_g w_g\operatorname{Cov}_g(s).
$$

It is consistently estimated by cross-products of score rows centered within
each sampling group. With iid sampling of the whole unit, including its group
label, $w_g$ instead denotes the population group probability; the covariance
includes variation in group composition:

$$
B_{\mathrm{joint}}
=\sum_g w_g E_g[ss^\mathsf{T}]
=B_{\mathrm{fixed}}+\sum_g w_g m_gm_g^\mathsf{T}.
$$

Raw score cross-products consistently estimate the latter at the target. The
two formulas agree when every group score mean is zero, including under a
correctly specified likelihood. Centering covariance rows leaves the estimating
equations and observed test score unchanged. Correct covariance alone does not
settle finite-sample score reference calibration. This distinction is consistent
with [Abadie, Imbens and Zheng (2014), Section 2](https://economics.mit.edu/sites/default/files/publications/Inference%20for%20Misspecified%20Models%20With%20Fixed.pdf);
the [grouped experiment](../experiments/decisions/03-score-centering/report.qmd)
provides an independently checked example.

Supplying `group` or observing group counts does not by itself choose conditional
fixed-allocation inference. The primary joint-sampling scope is unchanged.
The fixed-allocation extension is
[banked](backlog/speculative.md#fixed-design-inference-under-mean-misspecification),
with its covariance formula settled. Random missingness patterns are not fixed
sampling groups and do not inherit this centering rule.

## Covariates, conditional fitting and fixed design

Three distinctions determine scope:

| Concept | Meaning | Scope decision |
| --- | --- | --- |
| Random-X population inference | Repeated sampling of the joint observational unit, including X | Primary target, subject to each estimator's validation |
| Conditional fitting | A criterion or moment construction for Y given X | Compatible with random-X inference; each new route needs its own target and influence contract |
| Genuinely fixed-design inference | Repeated responses at the same specified covariate array | Outside the primary programme under unrestricted mean misspecification; banked |

Conditioning in an estimation criterion does not require conditioning in the
sampling law. A conditional estimating equation can define a population
functional under joint sampling without parametrically modeling the marginal
distribution of X. Conditional fitting and a joint working model can also
target different approximations under misspecification. Neither is an
automatic substitute for the other.

Compatibility options describe numerical conventions. In lavaan,
`fixed.x = TRUE` fixes exogenous means, variances and covariances at their
sample values, whereas `conditional.x = TRUE` constructs a conditional model
whose model-implied statistics exclude X. Categorical models default to the
conditional construction when exogenous covariates are present. Those settings
do not by themselves establish inference under arbitrary nonlinear mean
misspecification. See the official
[lavaan options reference](https://cran.r-project.org/web/packages/lavaan/refman/lavaan.html#lavOptions).

### Why fixed-design robustness needs a separate contract

Consider a full-rank fixed design with independent responses
$Y_i = m_i + \epsilon_i$, $E[\epsilon_i \mid X] = 0$ and
$\operatorname{Var}(\epsilon_i \mid X) = \sigma_i^2$.
The finite-design least-squares target is
$\beta_n(X) = (X^\mathsf{T}X)^{-1}X^\mathsf{T}m$, and its estimator's
conditional covariance is

$$
\operatorname{Var}(\widehat\beta\mid X)
= (X^\mathsf{T}X)^{-1}
  \left(\sum_i x_i x_i^\mathsf{T}\sigma_i^2\right)
  (X^\mathsf{T}X)^{-1}.
$$

If m is nonlinear in X, fitted residuals contain both response noise and
approximation error. Their squares do not generally isolate $\sigma_i^2$.
For a noiseless nonlinear response, the conditional covariance is exactly
zero, while the ordinary residual sandwich can be nonzero. Under random-X
sampling, approximation error interacting with sampled X instead contributes
to population-target uncertainty.

This is a first-principles counterexample to a general conditional-variance
guarantee for the ordinary residual sandwich, not an impossibility claim about
all fixed-design inference. Replication, restrictions on the conditional mean,
smoothness or an explicit randomization design can support narrower methods.
Such assumptions and the conditional target must be stated before choosing
the variance estimator or test reference law.

## Compatibility and inferential claims

The lab and C++ API retain their tested estimation and lavaan-compatibility
surfaces, including existing fixed-x conventions. Parity establishes the
specified component's behavior and tolerance. It does not establish a broader
sampling guarantee or choose the ordinary-user inference policy.

The ordinary policy promises only validated combinations of target, sampling
law, estimator and inferential component. A sandwich for a pseudo-true
parameter and an exact-fit test address different hypotheses; robustness of
one does not validate the other. Singular boundaries and penalized estimators
need their own inference rather than inherited interior formulas.

Categorical conditional stage-one moments and their sampling covariance are
not implemented. The existing rejection of categorical fixed-covariate models
remains. That extension is banked separately from genuinely fixed-design
inference: it could serve random-X population inference, but is not required
to complete the currently supported primary workflows.

### Ordinary fixed-x decision

Decided 2026-10-01: the
[ordinary API](design/r-interface-vision.md#ordinary-api) has no `fixed.x`
option and always fits the joint random-X model that the ordinary policy
geometry requires. A fixed-x fit is not a neutral numerical convention. Its
inferential meaning rests on further assumptions, such as a correctly
specified linear conditional mean (see the counterexample above), and an
argument would hide them. Fixed-x inference remains a banked extension, not an
ordinary option.

The removal changes estimators differently. For ML, the structural estimates
coincide, because the likelihood factors into the marginal of X and the
conditional of Y given X. For least-squares estimators they generally differ:
on a Holzinger–Swineford regression with two observed covariates, GLS and ULS
structural estimates moved by up to 0.05 and 0.13 (checked 2026-10-01).

The lab keeps its fixed-x conventions and their component gates. Ordinary
construction rejects lab specifications built with `fixed_x = TRUE` and says
how to rebuild them; it never silently converts a supplied fixed-x model or
refits a joint model in its place. The ordinary API implements this decision
from 0.2.0 (2026-10-02); 0.1.0 kept `fixed.x = TRUE` as its default, with
explicitly unsupported inference for fixed observed covariates.

## Reopening banked work

The [covariate trigger entries](backlog/speculative.md#covariates-and-sampling)
own the cheaper alternatives and build-if conditions. Before promoting a
bounded extension to the active backlog, name its consumer, target, sampling
law, identifying assumptions, estimator and requested inferential components.
Require independent derivation and calibration controls that distinguish mean
misspecification from heteroskedasticity and covariate randomness. Numerical
lavaan parity is a separate gate where compatibility is claimed.
