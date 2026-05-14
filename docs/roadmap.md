# magmaan roadmap

This is the live roadmap for complete-data normal-theory SEM work. Finished
phase plans are removed or folded into this file so there is only one active
roadmap.

Out of scope for this track: FIML/missing data, ordinal/DWLS/polychoric,
bootstrap SE, Bayesian, multilevel, and end-user `cfa(model, data)` ergonomics.

## Immediate Priorities

### 1. Stabilize bounded optimization

The current soft barrier catches non-PD regions but does not impose real box
constraints. Heywood warnings exist; bounded estimation is still incomplete.

Active work:

- Finish the bounded optimizer path around variance lower bounds.
- Keep equality-constrained models on the existing reparameterized path until
  bounded optimization and affine constraints are composed deliberately.
- Replace the Ceres single-residual `sqrt(F)` least-squares trick for LS
  discrepancies. It produces a rank-1 Jacobian and stalls on shallow ULS
  objectives.
- Add per-discrepancy multi-residual Ceres cost functions for LS estimators:
  ULS first, then GLS/WLS.
- Restore the single-group 3F + meanstructure fit fixture once the bounded
  multi-residual path can fit it reliably.

Validation targets:

- Holzinger 3F + means, where the current LBFGS defaults fail.
- Saturated 1F ULS, which exposes shallow-landscape conditioning.
- Heywood-prone ULS, where lower bounds must be enforced.

### 2. Keep namespace cleanup opportunistic

The package identity rename to `magmaan` is complete, and public transition
headers exist for the target domains:

- `spec`: model specification, lavaanify options, equality groups, linear
  constraints, starts.
- `lavaan`: lavaan-shaped partable projections and compatibility views.
- `estimate`: estimates, fitting orchestration, bounds, fixed.x resolution,
  equality reparameterization.
- `optim`: optimizer concepts and implementations.
- `nt`: complete-data normal-theory ML, inference, robust tests, measures,
  standardization, and defined parameters.
- `gls`: GLS-family discrepancy functions such as ULS.
- `data`: sample statistics, raw data, and moment/gamma helpers.

What remains:

- Move primary definitions out of the old `magmaan::fit` and
  `magmaan::partable` namespaces into the target namespaces.
- Convert repo code and R binding internals to include and call the target
  headers/namespaces directly.
- Keep old `include/magmaan/fit/*` and `include/magmaan/partable/*` headers as
  compatibility shims for one transition window.
- Keep one focused compatibility test that includes representative old headers
  and verifies the aliases still compile.
- Update stale source references in docs after the code migration lands.

This migration is API organization work, not a statistical feature. The public
`optim` facade already looks good; its headers currently alias definitions that
still live in `magmaan::fit`. Moving those canonical definitions into
`magmaan::optim` would clean up symbols and documentation, but it is not the
next functional blocker. Defer it unless touching optimizer code anyway.

### 3. Close the remaining observed/robust inference gaps

Expected-info inference, finite-difference observed inference, robust SEs,
UΓ/Satorra-Bentler families, Browne residual NT, Wald/LR/z tests, fit measures,
standardization, and Satorra (2000) nested tests are implemented and covered.

Open gaps:

- Implement `AnalyticObservedInfoSE` for mean-structure models. The missing
  terms are the `dmu` and `d2mu` contributions involving Λ×α, Λ×B, α×B, and
  B×B pairs.
- Generalize observed-bread robust SE / U-factor handling to multi-block
  models. The expected-bread path is multi-block; the observed-bread path still
  has single-block assumptions.
- Generalize `reduced_gamma_unbiased` to multi-block models, likely by
  carrying per-block reduced moment matrices instead of consuming only the
  already-summed form.
- Add Browne residual ADF by swapping empirical Γ into the existing Browne
  residual projection machinery.
- Verify `fit_extras` conditional log-likelihood for multi-group `fixed.x` and
  for meanstructure × `fixed.x` once fixtures exist.

### 4. Finish R/API parity polish

The C++ surface is ahead of the R boundary in several places.

Open items:

- Expose `compute_defined()` through R and add a golden comparison against
  `parameterEstimates()` for chained `:=` rows and `.pN.` references.
- Add a sample-moments-only input path for R/API users who already have `S`,
  `mean`, and `n`. The C++ `SampleStats` type already models this.
- Add `se = "none"` / `test = "none"` shortcuts to skip post-fit inference
  when callers only need point estimates.
- Add lavaan partable comparison helpers that compare by semantic row keys
  rather than raw row position.
- Keep fit-list extraction helpers as R boundary conveniences, but keep C++
  APIs explicit over primitive values.

## Estimator Track

### ULS

ULS is implemented and unit-tested as a discrepancy. Remaining work:

- Add lavaan golden fixtures for `estimator = "ULS"`.
- Re-check multi-group weighting against lavaan once ULS goldens exist.
- Use ULS as the first consumer of multi-residual bounded optimization.

### GLS and WLS/ADF

The architecture is ready: `fit<D, O>()` is discrepancy-generic, model
Jacobians are reusable, `SampleStats` is per-block, and Γ helpers already
exist.

Remaining work:

- Add `GLS`: `F = 1/2 tr((Σ^-1(S-Σ))^2)` with the corresponding analytic
  gradient.
- Add `WLS`/ADF: `F = 1/2 vech(S-Σ)' Γ^-1 vech(S-Σ)`, with constructor-time
  checks for the supplied weight matrix.
- Generalize robust/inference helpers currently hard-coded around
  `W = Γ_NT(M)^-1` so ULS/GLS/WLS can reuse the sandwich paths with arbitrary
  per-block weights.

## Reporting and Exploration

These are lavaan-parity features that sit above the core estimator.

- Modification indices, univariate score tests, and EPC are not implemented.
  They need scores and information for fixed or equality-constrained
  parameters at θ̂.
- RMSEA close-fit p-value (`rmsea.pvalue`, H0: RMSEA <= 0.05) is not computed.
- `std.all` beta-coefficient rescaling remains to be checked/implemented for
  structural regressions.
- Add more targeted goldens for scalar-invariance latent mean rescaling once
  the bounded optimizer can fit the needed meanstructure models reliably.

## Constraint Boundary

Linear equality constraints are implemented through affine
reparameterization. Effect coding for loadings is implemented. The current
boundary is intentional:

- Nonlinear equality constraints remain unsupported.
- Inequality constraints remain unsupported.
- Active bound/inequality inference must not silently report ordinary χ²/SE
  theory; boundary cases need explicit warnings or chi-bar-square style
  treatment before being presented as regular inference.

## Design Invariants

- No global mutable state.
- No in-place mutation of shared public structures. Entry points operate on
  values or local copies and return values/errors.
- No groups == one group. Single-group models use the same block/group shape
  as multi-group models.
- Lavaan-shaped partables are a boundary format for oracle comparison,
  interchange, and compatibility projection. Core work should prefer the model
  triple plus explicit derived structures.
- Extension points remain concepts and free functions, not virtual hot-path
  interfaces.
