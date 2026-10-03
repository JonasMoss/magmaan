# Ordinal workspaces and threshold profiling

This document records the shipped ordinal workspace and threshold-profile
contracts. Implementation state is summarized in the
[roadmap](../architecture/capabilities/ordinal_and_mixed.md#ordinal-and-mixed-categorical-ls);
remaining work belongs in the [backlog](../backlog/todo.md#shared-fitting-composition).

## Model and moment boundary

LISREL / `MatrixRep` owns implied covariance and mean evaluation. Ordinal
thresholds and response scales belong to the lavaanified model; the ordinal
fit compares observed thresholds and latent-response associations with their
model counterparts. Gamma/NACOV is the asymptotic covariance of that ordinal
moment vector, not the continuous covariance-statistic Gamma.

## Shipped data split

The public types live in
[`data/ordinal.hpp`](../../cpp/include/magmaan/data/ordinal.hpp):

- `OrdinalMoments` and `MixedOrdinalMoments` carry observed moments, variable
  names, threshold ownership/levels and sample metadata, without NACOV or weights.
- `OrdinalGammaCacheBlock` holds diagonal/full Gamma, DWLS/WLS weights and
  explicit flags indicating which products have been materialized.
- `OrdinalWeightPlan` selects purpose (`FitOnly`, `FitPlusInference`,
  `InferenceOnly`), estimator (ULS/DWLS/WLS), delta/theta parameterization,
  threshold-mode metadata and Gamma materialization. The partable supplies the
  actual free, fixed, shared or linearly constrained threshold design; the
  threshold-mode enum is bookkeeping rather than a replacement for that design.
- `OrdinalWorkspace` and `MixedOrdinalWorkspace` pair moments with their Gamma
  cache. They do not own the model evaluator or a prepared partable.
- `OrdinalStats` / `MixedOrdinalStats` remain compatibility objects with
  materialized NACOV and weights. Cache conversion and ensure helpers bridge
  them to the workspace paths.

Raw-data workspace constructors honor fit-only costs: ULS avoids Gamma, DWLS
builds its diagonal, and WLS needs full Gamma and its weight. Mixed WLS defers
inversion to the cache ensure helper. Fit-plus-inference can retain the full
Gamma products needed downstream. The `Reduced` materialization enum does not
establish a general reduced-Gamma inference implementation; that work remains
[speculative](../backlog/speculative.md).

Cache-aware bounded fits, all-ordinal delta SNLLS, theta threshold profiling,
mixed theta SNLLS, and robust all-ordinal cache reuse are implemented. The
threshold derivation below describes the affine delta profile. Theta uses the
separate standardized-threshold profile described under SNLLS Integration.

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
in the profiled objective. Fixed or constrained thresholds retain their threshold residuals and need
the corresponding Gamma diagonal entries.

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

- all-ordinal, delta parameterization with fixed unit response scales
- the full affine threshold design: free thresholds, fixed rows, equality
  merges within and across groups, and threshold-only linear constraints
- ULS, DWLS, and WLS (profiled Schur-complement weight)
- multi-group fits with joint `n_b/N`-weighted threshold normal equations

Non-unit fixed or free DELTA scales retain their original `~*~` coordinates.
They use the full-threshold GP moment map instead of the unit-scale Schur
profile: thresholds are `delta_i * (tau_i - mu_i)` and correlations are
`delta_i * delta_j * Sigma*_ij`. Scale coordinates stay nonlinear, while
threshold and mean coordinates are conditionally linear. Affine equalities
on scales remain in the generic constraint map. The compact-moment bounded
route likewise switches to full moments for these scales. Frozen ULS oracle
fixtures cover fixed, released, equal, constrained and grouped scales.
Nonlinear equality constraints are rejected by the shared GP classifier.

Theta now has a separate fast path for independently free, unbounded
thresholds with no active equality constraints. It eliminates standardized
thresholds before optimization and reconstructs raw thresholds as
`tau = mu + sqrt(diag(Sigma)) * z`. ULS/DWLS set `z` to observed thresholds;
WLS caches the threshold map and a QR square root of the Schur-complement
weight. All covariance parameters remain nonlinear because correlations use
model-dependent standardization. The endpoint receives the original
threshold-inclusive common audit. Fixed/shared thresholds and constrained
models retain the generic full-threshold implementation. The explicit
`fit_ordinal_snlls_full_thresholds()` entry point remains available as a
reference for eligible models too.

The nonlinear delta SNLLS block operates on the profiled ordinal correlation
objective after thresholds have been eliminated. Threshold estimates are then
reconstructed for the returned full parameter vector / partable output.

For `H = I` and ULS/DWLS, reconstruction is trivial:

```text
tau = tau_hat
```

For WLS, use the formula above.

The SNLLS compatibility checker rejects unsupported constraint and
parameterization combinations. Mixed theta SNLLS has its own fit path; the
affine all-ordinal delta derivation is not a general mixed-moment profile.

## Validation and remaining work

[`ordinal_test.cpp`](../../cpp/tests/unit/ordinal_test.cpp) covers workspace
materialization costs, cached-versus-legacy fits, fixed/shared/linearly
constrained thresholds, joint multi-group profiling, delta/theta fits and
mixed workspaces. The theta profiler also has off-optimum objective/gradient
checks. [`ordinal_golden_test.cpp`](../../cpp/tests/golden/ordinal_golden_test.cpp)
drives the profiled and bounded paths through threshold-invariance and linear
constraint fixtures (0013/0014); keyword theta invariance has separate fixtures
0017–0020. The roadmap records their numerical conventions and boundaries.

Keep benchmark boundaries explicit: moment construction, diagonal/full Gamma,
weight construction, optimization and post-fit inference. Compare fit-only
and fit-plus-inference workloads separately; a fit-only ULS/DWLS result must
not hide full-Gamma work in setup.

Remaining R/API polish and mixed invariance work are tracked in the
[active backlog](../backlog/todo.md#shared-fitting-composition); Gamma-influence
performance and weight storage have their own entries. Reduced-Gamma robust
products require a concrete size-driven consumer before promotion from the
speculative backlog. These pointers replace the completed implementation
sequence and its superseded public-surface questions.
