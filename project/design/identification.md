# Structural identification

Adopted 2026-10-07 (TASK-33.3). **User decisions (2026-10-07):** check
structural identification cheaply and once per model; always construct the
model; the ordinary package refuses to fit a structurally unidentified model
and says which parameters are free and how to fix them; document that magmaan
does this. Local Newton accuracy never certifies identification
([terminal audit](terminal-audit.md)).

## What is checked

A model maps parameters to implied moments
$m(\theta)=(\operatorname{vech}\Sigma_g(\theta),\ \mu_g(\theta))_g$, plus the
threshold and polychoric map on ordinal routes. The parameter is locally
identified at $\theta$ when $m$ is one-to-one near $\theta$; for these
polynomial maps that holds exactly when the Jacobian
$J(\theta)=\partial m/\partial\theta$ has full column rank in the coordinates
$\theta=\theta_0+K\alpha$ of the linear equality constraints (Rothenberg 1971).

The rank of $J(\theta)K$ is constant off a measure-zero set, so it is evaluated
at three seeded pseudo-random points (SplitMix64, identical on every platform).
Columns are scaled to unit length before a Jacobi SVD, so parameter units do
not matter, and no data enter, so rescaling variables cannot change the result.
With $s_{\min}$ the smallest relative singular value:

- **Identified** if $s_{\min}\ge 10^{-7}$ at some point;
- **Unidentified** if $s_{\min}\le 10^{-10}$ at every point (with the best
  point's rank clearing $10^{-7}$), or if the counting rule fails: more free
  reduced parameters than moment rows;
- **Unchecked** otherwise, and on routes without a supported moment map
  (nonlinear or inequality constraints, CatML, ordinal PSD and multi-information,
  two-level, FC-SEM, sphere charts, RBM).

Exact null directions sit near $10^{-15}$; identified models sit far above
$10^{-7}$. The TASK-33.3 corpus sweep records the observed gap.

Not checked here: global identification (discrete ambiguities such as a
factor's sign are harmless), and empirical underidentification, where a
generically identified model is singular at the estimate (a two-indicator
factor whose correlations vanish, a marker whose loading is near zero). The
latter remains an endpoint diagnostic (conditioning, Newton audit, standard
errors) and is never refused.

## How models become unidentified

1. **Scale freedom.** $\Lambda\to c\Lambda$, $\psi\to\psi/c^2$ leaves
   $\Lambda\psi\Lambda^\top$ unchanged; cured by a marker, `std.lv` or effects
   coding.
2. **Location freedom.** $\alpha\to\alpha+(I-B)a$, $\nu\to\nu-\Lambda a$ leaves
   $\mu=\nu+\Lambda(I-B)^{-1}\alpha$ unchanged (the shift is $\alpha+a$
   when $B=0$); cured by fixing the latent mean or an intercept, and across
   groups by intercept invariance with a reference group.
3. **Rotation freedom.** $\Lambda\to\Lambda T$, $\Psi\to T^{-1}\Psi T^{-\top}$;
   removed by the confirmatory zero pattern unless too many loadings are free.
4. **Information deficits.** Fewer moments than parameters, or free residual
   covariances and higher-order structure that consume the information.
5. **Structural confounding.** Reciprocal paths without instruments, constraint
   systems that leave a direction unpinned.

Types 1-3 are gauge freedoms of the coordinates and are detected exactly;
1-2 have standard fixes, 3 needs the analyst's zero pattern.
Types 4-5 are genuine deficits; no automatic constraint is neutral.

## Policy

**Core.** `check_structural_identification()` returns an
`IdentificationReport`: status, reason, the probe map, $q$, moment rows, rank,
singular values, and null directions named by parameter label. The result is
data-free, so it is computed once per model structure and cached with the
prepared model; fits reuse it. `with_identification()` turns Unidentified into
a failed verdict with reason `unidentified` under every convergence rule
(common, compatibility presets, explicit assessment). Unchecked leaves the
verdict unchanged and is reported.

**Ordinary package.** `magmaan_model()` computes the report at construction and
stores it; printing shows the status next to the identification setting.
`magmaan()` and refits refuse an Unidentified model before fitting with a
condition of class `magmaan_identification_error`, carrying the free
directions and suggested fixes, as Mplus refusals and schema errors already do.
There is no override in the ordinary package; deliberate fits of unidentified
models belong in magmaanlab. Unchecked models fit normally and report
identification as unchecked. magmaan never silently adds constraints beyond the
declared `identification` choice (marker or `std.lv`).

**Suggested fixes.** Each null direction is classified exactly, by testing
whether the null space contains the generator of a gauge freedom at the probe
point: the scale generator of each latent variable (the derivative of
$\Lambda\to c\Lambda$, $\Psi\to\Psi/c^2$ and the matching regression and mean
terms at $c=1$), its location generator, and, for each block of $k$ factors,
the $k^2$ generators of $\Lambda\to\Lambda T$, $\Psi\to T^{-1}\Psi T^{-\top}$
at $T=I+\varepsilon E_{ab}$, whose diagonal members are the scale generators.
Fixed and absent matrix entries and linear equalities first restrict the
generator span. Principal angles between that admissible span and the numerical
null space, in the same equilibrated reduced coordinates as the rank check,
give the contained freedoms. Individual contained scale and location generators
are reported first; remaining contained combinations identify rotation freedom.
Translations and linear factor transformations are intersected separately, so
the reported types do not come from a mixture of generator families. Reported
gauge columns retain their actual generators; an internal orthonormal basis
in those coordinates gives the information-deficit complement. Ordinal and
mixed routes test the same generators against their own moment Jacobians,
with unchanged threshold and response-scale entries included in the reduction.
Scale and location freedoms get a
standard fix (fix a loading or the variance; fix the latent mean or an
intercept); rotation freedom between factors $a$ and $b$ is reported with the
advice to add independent loading restrictions (each factor needs $k-1$ zeros
in a non-degenerate pattern). Covariance restrictions can help but are not
sufficient on their own; recheck identification after editing. No automatic
fix is applied. Any remaining null dimension is reported as an information
deficit with the parameters involved, and no automatic fix.

**magmaanlab.** Exposes the report on fits and through a standalone check, and
does not refuse: methods development sometimes fits ridges on purpose. Lab
verdicts still fail with reason `unidentified`.

## Cost

One Jacobian evaluation per probe point (the cost of one gradient), three
points, and an SVD of order $Mq^2$ for $M$ moment rows and $q$ free reduced
parameters: microseconds to milliseconds for ordinary models, about a second
for very large ones, paid once per model. The TASK-33.3 sweep records measured
costs.

## Validation

- Witnesses: the free-marker three-indicator CFA through the PSD fallback, and
  the direct-FIML complete one-factor model with a free marker and mean.
- Controls: marker, `std.lv`, effects coding, tau-equivalence, Little (2013)
  Table 10.3's fully pinned block, rescaled indicators, multigroup equality,
  correlated two-indicator factors, and an empirically underidentified fit,
  which must not be refused.
- Negative-control sweep: every lavaan-fitted model in the C++ golden and
  fixture corpora must be Identified; each exception is listed and explained.
- Classification: each gauge generator type is detected, and a deficit-only
  model gets no false fix.

## Documentation

`?magmaan_model` gains an *Identification* section beside *Schema checks*;
`?magmaan` documents the refusal and its condition class; the lab help
documents the report fields and the standalone check; NEWS records the change
under the development heading. This note, [terminal audit](terminal-audit.md)
and the [numerical-audit capability](../architecture/capabilities/numerical_audit.md)
hold the contract.

## Work

- TASK-33.3: core check, verdict wiring and R exposure on fits (complete).
- TASK-33.3.1: construction-time caching on immutable `api::Model` and native
  prepared handles, the ordinary refusal, printing and standalone lab report.
  Cached directions refer to the deciding data-free probe, not an estimate.
  Low-level callers can attach a report to their immutable `MatrixRep`; rebuild
  the representation after structural changes. Complete: full optimized and
  Debug C++ suites and both R testthat suites verified, including the corrected
  unsupported ordinal-schema fixture and construction-count probes.
- TASK-33.3.2: scale, location and rotation classification and suggested fixes.
