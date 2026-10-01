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

The primary programme is single-level normal-theory ML, FIML and ordinal/mixed
DWLS, with PSD covariance constraints and the multi-information barrier
developed alongside them. The roadmap owns the estimator tiers and supported
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

Current ordinary ML policy geometry requires random X; the public fitting
interface still defaults to `fixed.x = TRUE`. For models with fixed observed
covariates, this mismatch remains an active
restriction/documentation task, not a commitment to build general fixed-design
inference. Unsupported inference must be explicit, and no route may silently
refit a joint model. Any change to the fitting default needs a separate,
evidence-backed interface decision.

Categorical conditional stage-one moments and their sampling covariance are
not implemented. The existing rejection of categorical fixed-covariate models
remains. That extension is banked separately from genuinely fixed-design
inference: it could serve random-X population inference, but is not required
to complete the currently supported primary workflows.

## Reopening banked work

The [covariate trigger entries](backlog/speculative.md#covariates-and-sampling)
own the cheaper alternatives and build-if conditions. Before promoting a
bounded extension to the active backlog, name its consumer, target, sampling
law, identifying assumptions, estimator and requested inferential components.
Require independent derivation and calibration controls that distinguish mean
misspecification from heteroskedasticity and covariate randomness. Numerical
lavaan parity is a separate gate where compatibility is claimed.
