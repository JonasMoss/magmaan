# Ordinal SNLLS and Gamma Workspace Plan

This note is the restart point for the ordinal SNLLS / Gamma work. It records
the intended architecture before touching C++ source again.

## Why Ordinal Is Not Plug And Play

The continuous SNLLS path works on a moment vector that is essentially

```text
s = vech(S)          or          s = [mean; vech(S)]
sigma(theta) = vech(Sigma(theta))
```

and `Sigma(theta)` comes directly from the LISREL block

```text
Sigma(theta) = Lambda (I - B)^-1 Psi (I - B)^-T Lambda^T + Theta.
```

Ordinal data changes the target. The observed statistics are thresholds plus
latent-response correlations/polychorics:

```text
s = [tau_hat; rho_hat]
m(theta) = [tau(theta); rho_star(theta)]
```

Thresholds are not entries of `Sigma(theta)`. They are an ordinal measurement
layer around the LISREL covariance model. Lavaan represents them with `|`
partable rows. Internally, magmaan should keep that separation:

- LISREL / `MatrixRep` owns the covariance and mean machinery.
- Ordinal threshold rows live in the lavaanified model/partable layer.
- The ordinal fit layer compares `[tau; rho]`, not just `vech(Sigma)`.
- The Gamma/NACOV object is the asymptotic covariance of the full ordinal
  statistic vector, not the continuous covariance statistic.

That is the core reason the continuous SNLLS implementation cannot simply take
a polychoric matrix and a Gamma matrix and continue unchanged.

## Current Starting Point

The current code already has a working bounded ordinal LS surface:

- `data::OrdinalStats`
  - `R`, `thresholds`, threshold metadata, `NACOV`, `W_dwls`, `W_wls`,
    levels, names, and `n_obs`.
- `data::MixedOrdinalStats`
  - mixed continuous/ordinal moments, `NACOV`, `W_dwls`, `W_wls`, and metadata.
- `estimate::fit_ordinal_bounded(...)`
- `estimate::fit_mixed_ordinal_bounded(...)`
- `estimate::robust_ordinal(...)`
- `estimate::robust_mixed_ordinal(...)`

The immediate design problem is that these stat structs conflate three things:

1. Observed moments and metadata.
2. Gamma/NACOV construction.
3. Weight construction for DWLS/WLS.

That is convenient for parity tests, but it is wrong for benchmark hygiene and
for a methods-developer architecture where fitting and inference should be
conceptually separable.

## Design Goals

- Keep fit-only speed experiments honest: do not compute full Gamma, full WLS
  weights, or robust inference products when the selected estimator does not
  need them.
- Still allow a one-call "fit plus inference" path that computes all needed
  Gamma pieces once when that is cheaper than rebuilding them downstream.
- Make DWLS cheap: compute only the diagonal Gamma entries needed by the fit.
- Make ULS cheaper: compute no Gamma for fit-only ULS.
- Make full WLS explicit: full WLS needs either full Gamma inversion or the
  equivalent profiled blocks.
- Preserve cached Gamma pieces so later inference can reconstruct full or
  reduced products without recomputing pairwise/polychoric estimating equations
  when the caller requested that.
- Treat threshold profiling as part of the ordinal LS layer, not as a LISREL
  matrix-representation concern.
- Implement the simple `H = I` threshold case first, then generalize to
  constrained thresholds.

## Proposed Data Split

Introduce a non-breaking internal split first; the public compatibility structs
can remain as adapters until the new path is stable.

### `OrdinalMoments`

Observed statistics and metadata only.

Likely contents:

- `R`
- `thresholds`
- threshold owner/level metadata
- mixed continuous means/variances/covariances when needed
- moment layout descriptors
- `n_obs`
- `n_levels`
- observed-variable names

No `NACOV`, `W_dwls`, or `W_wls`.

### `OrdinalGammaCache`

Lazy/cache object for asymptotic covariance pieces.

It should be able to provide, per block:

- diagonal of Gamma
- full Gamma
- Gamma subblocks for threshold/correlation partitions
- full inverse or factorization when WLS asks for it
- reduced products for robust inference where possible
- provenance flags saying exactly which pieces have been computed

The cache should not materialize full Gamma just because DWLS asks for diagonal
weights.

### `OrdinalWeightPlan`

Small object describing what the caller needs:

```text
purpose: fit_only | fit_plus_inference | inference_only
estimator: ULS | DWLS | WLS
parameterization: delta | theta
threshold_mode: free_identity | linear_map | fixed_or_constrained
materialization: none | diagonal | full | reduced
```

The plan chooses which cache pieces to build. The same plan should drive
benchmark reporting so we can separate:

- statistic construction
- diagonal Gamma construction
- full Gamma construction
- WLS inversion/factorization
- optimizer time
- post-fit inference time

### `OrdinalFitWorkspace`

Prepared object shared by fit and inference:

- prepared ordinal partable
- evaluator / matrix representation
- moment layout
- `OrdinalMoments`
- `OrdinalGammaCache`
- chosen `OrdinalWeightPlan`

This is where the "compute all in one go" option belongs. The conceptual API
can still say fitting and inference are separate; the workspace lets a caller
pay for shared setup once when desired.

## Threshold Profiling

Partition the ordinal residual as

```text
e = [e_tau; e_rho]
  = [tau_hat - (H alpha_tau + c); rho_hat - rho_star(beta, alpha_sigma)]
```

where `H` maps threshold parameters to implied thresholds and `c` carries fixed
threshold constants. For unconstrained automatic thresholds, `H = I` and
`c = 0`.

### ULS and DWLS, `H = I`

With free thresholds and a diagonal or identity fit weight, the optimal
threshold residual is zero:

```text
tau(theta) = tau_hat
e_tau = 0
```

So fit-only ULS and DWLS do not need to optimize thresholds. For DWLS, the fit
only needs the diagonal Gamma entries for the non-threshold moments that remain
in the profiled objective. If we keep constrained/fixed thresholds in a later
slice, the threshold block re-enters.

### Full WLS, `H = I`

Full WLS has threshold-correlation cross-weights. Let

```text
W = [W_tt W_tr;
     W_rt W_rr]
```

For a fixed correlation residual `e_r`, the profiled threshold residual is

```text
e_t = - W_tt^-1 W_tr e_r
```

and the profiled objective uses the Schur-complement weight

```text
W_profile = W_rr - W_rt W_tt^-1 W_tr.
```

Equivalently, the reconstructed thresholds are

```text
tau = tau_hat + W_tt^-1 W_tr e_r.
```

This means full WLS is not just "set thresholds to observed values"; the
threshold/correlation cross-block matters.

### General `H`

For equality-constrained or otherwise linearly mapped thresholds:

```text
alpha_tau = (H' W_tt H)^-1 H' (W_tt (tau_hat - c) + W_tr e_r)
```

The implementation now covers the full affine threshold design
`tau_b = c_b + H_b gamma`: free thresholds, fixed rows, equality-label merges
(including cross-group threshold invariance), and threshold-only linear
equality constraints folded through a null-space basis. The threshold normal
equations are joint across blocks with `n_b/N` sample weights — required as
soon as one `gamma` coordinate spans groups, where the per-block solve is
singular and the block weights no longer cancel. Block `b`'s profiled
thresholds then depend on every block's correlation residual through the
shared normal-matrix inverse, so the workspace threshold map consumes the
stacked correlation residual. Constraints that mix threshold and
non-threshold columns, equality groups linking thresholds to non-threshold
parameters, and infeasible/contradictory threshold constraint systems fail
clearly.

## Fit-Only Cost Rules

These are the rules the implementation and experiments should enforce.

| Estimator | Free thresholds (`H = I`) | Gamma needed for fit | Weight needed for fit |
| --- | --- | --- | --- |
| ULS | profile exactly | none | none / identity |
| DWLS | profile exactly | diagonal only for active non-threshold moments | diagonal inverse |
| WLS | profile with Schur complement | full relevant Gamma or inverse/factor blocks | full/profiled WLS |

For fixed threshold rows, ULS/DWLS keep the threshold residuals in the profiled
full moment vector and DWLS consumes the corresponding diagonal Gamma entries.
Shared-label merges (within or across groups) and threshold-only linear
constraints use the same joint design map above.

## Inference Rules

Inference is allowed to need more than fitting.

- Standard ordinal robust reporting needs the full or reduced Gamma machinery
  for sandwich SEs and scaled/shifted test statistics.
- If the fit was ULS or DWLS fit-only, inference may need to extend the cache
  after fitting.
- If the caller selects fit-plus-inference, the workspace may compute full
  Gamma once up front and retain it.
- Where possible, robust test paths should consume reduced Gamma products
  directly instead of forcing full materialization.

The magmaan rule still applies: never compute more than necessary for the
requested product, but make it possible to request a larger product bundle
intentionally.

## Mixed Gamma Construction

The mixed continuous/ordinal `NACOV` mirrors lavaan's muthen1984
estimating-equation sandwich exactly (validated to ~1e-8 per moment block on
the mixed fixtures):

- **Stage-1 stack** `[th | mu | var]`: threshold scores for ordinal variables;
  univariate normal ML scores `(y - mu)/sigma^2` and
  `((y - mu)^2 - sigma^2)/(2 sigma^4)` for continuous ones. The bread `A11` is
  block-diagonal per variable (threshold block, or the 2x2 `(mu, var)` OPG
  block for a continuous variable); the meat is the full score OPG.
- **Stage-2 pair scores** for every association type: polychoric rho scores,
  polyserial rho scores with mu/var coupling channels (the raw-metric pair
  scores `sigma dl/dmu = u - dlogP/du` and
  `sigma^2 dl/dsigma^2 = ((u^2 - 1) - u dlogP/du)/2`, exposed as
  `PolyserialPairScores::{mu_unit, var_unit}`), and bivariate-normal
  correlation ML scores for continuous-continuous pairs (chain-ruled from the
  covariance-metric `continuous_pair_normal_scores` into the
  `(mu, sigma^2, rho)` parameterization).
- **`A21`** holds the score-cross coupling of each pair's rho score with its
  margins' stage-1 scores (th/mu/var channels); **`A22`** is the diagonal rho
  score OPG. The sandwich inverts block-triangularly as before.
- **Delta rule** (lavaan's `H`): association influence rows transform from the
  correlation metric to the covariance metric using the *post-sandwich*
  variance influence columns — `cov_ij` rows are
  `sd_i sd_j IF_rho + (rho sd_j / 2 sd_i) IF_var_i + (rho sd_i / 2 sd_j)
  IF_var_j` (one variance term for polyserial rows) — replacing the former
  ad-hoc raw-residual variance patch. Mean rows are `-IF_mu` (the `-mu`
  moment sign), variance rows are `IF_var`.

The same construction backs the eager `mixed_ordinal_stats_from_data_impl`,
the lazy fit-only DWLS diagonal in `mixed_ordinal_workspace_from_data` (column
norms only), and the Huber-residual single-ordinal rebuild (whose no-clip
variant reproduces the ML Gamma exactly; its mu/var coupling channels use the
ML score units at the Huber rho, an estimating-equation approximation under
clipping). The polyserial-DPD pair bread continues to override the
`(thresholds, rho)` block; its mu/var channels use the ML-identity
approximation and its Gamma is a research surface allowed to differ from
lavaan.

## SNLLS Integration

The threshold-profiled ordinal SNLLS path covers:

- all-ordinal, delta parameterization (theta delegates to the full-threshold
  path)
- the full affine threshold design: free thresholds, fixed rows, equality
  merges within and across groups, and threshold-only linear constraints
- ULS, DWLS, and WLS (profiled Schur-complement weight)
- multi-group fits with joint `n_b/N`-weighted threshold normal equations

The nonlinear SNLLS block should operate on the profiled ordinal correlation
objective after thresholds have been eliminated. Threshold estimates are then
reconstructed for the returned full parameter vector / partable output.

For `H = I` and ULS/DWLS, reconstruction is trivial:

```text
tau = tau_hat
```

For WLS, use the formula above.

The SNLLS compatibility checker must reject cases where threshold constraints,
theta-parameterization details, or mixed moments require machinery not yet in
the ordinal SNLLS implementation.

## Benchmarks To Add

Keep two experiment families separate.

### Fit Speed

Fit speed should report:

- moment construction time
- Gamma diagonal/full construction time
- weight construction time
- optimizer time
- total fit-only time

For fit-only comparisons:

- ULS should not build Gamma.
- DWLS should not build full Gamma.
- WLS should report full Gamma/weight setup separately from optimizer time.

### Fit Plus Inference Speed

A later experiment should report:

- moment construction
- Gamma construction
- fitting
- robust SE/test construction
- total fit plus inference

This is where "compute all in one go" can be evaluated against "fit cheaply,
extend cache during inference."

## Implementation Slices

1. **Documentation only.**
   This file plus a backlog pointer.

2. **Internal cache skeleton.**
   Add `OrdinalMoments`, `OrdinalGammaCache`, and `OrdinalWeightPlan` behind
   existing `OrdinalStats` / `MixedOrdinalStats` adapters. No estimator behavior
   change yet.

3. **Cost-aware existing bounded fits.**
   Route existing ordinal ULS/DWLS/WLS through the plan/cache enough to verify
   that ULS/DWLS no longer build unnecessary full weights in new fit-only
   entry points. Keep compatibility constructors for lavaan parity fixtures.

4. **Threshold profiler, `H = I`.**
   Implement the ULS/DWLS exact profile and WLS Schur complement. Add tests that
   compare profiled and unprofiled bounded fits on small ordinal fixtures.

5. **All-ordinal SNLLS, ULS/DWLS.**
   Add the profiled residual/Jacobian path for ordinal correlations and compare
   against the full bounded LS path on supported fixtures.

6. **All-ordinal SNLLS, WLS.**
   Add the full-WLS profiled threshold block and verify parity with full WLS.

7. **Inference workspace integration.**
   Add fit-plus-inference and inference-only cache extension paths. Verify that
   robust ordinal SE/test results match the current materialized NACOV path.

8. **Fixed threshold rows.**
   Keep fixed threshold residuals in the profiled full-moment objective and
   verify bounded and SNLLS parity against the unprofiled path.

9. **Threshold-local pure-merge constraints.**
   Let shared-label / bare-merge threshold equality groups share columns in the
   threshold map and verify bounded plus SNLLS parity against the unprofiled
   constrained path.

10. **Mixed ordinal / general threshold maps.**
   Generalize only after the all-ordinal free/fixed/pure-merge threshold path is
   stable.

11. **Experiments.**
   Add fit-only and fit-plus-inference ordinal experiments with explicit setup
   time accounting.

## Open Questions

- Should the first public cache surface be C++ only, with R wrappers added only
  after the experiments need them?
- Should full WLS store the inverse weight, a Cholesky/factorization, or just
  the Schur-complement operator?
- How should threshold constraints be represented in `LatentStructure` without
  making `MatrixRep` aware of them?
- Can the robust U-Gamma path consume ordinal reduced products directly enough
  to avoid full Gamma in common inference calls?
- What is the smallest instrumentation we need to prove fit-only benchmarks are
  not accidentally computing full Gamma?
