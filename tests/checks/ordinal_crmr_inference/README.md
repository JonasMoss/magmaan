# CRMR estimated-weight confidence-interval verification

Advisory local check for `estimate::ordinal_crmr_misspec_inference`, the
misspecification-aware confidence interval for the categorical CRMR fit index.
Outside the default test suite and CI; links the optimized static libs and runs
a Monte-Carlo study. Single-model absolute-fit analogue of
`tests/checks/ordinal_dwls_profile`.

## What it verifies

CRMR's criterion metric is fixed (it averages squared polychoric-correlation
residuals), so unlike RMSEA it carries no estimated-weight (γ) channel on the
metric side. But it is evaluated at the DWLS estimate `θ̂`, which is fit with the
estimated weight, so the weight re-enters through the estimator. This check asks
whether the CI that propagates that channel is calibrated, and how large the
channel actually is.

Population: the four-cycle (C4) binary design (p = 4 binary indicators,
thresholds 0, latent correlations `base = λ²` on the two opposite pairs and
`base + ε` on the four C4 edges). We fit the **congeneric one-factor** model; for
`ε > 0` it is misspecified (C4 is not rank-1) so the residual is nonzero and the
γ channel is live; `ε = 0` is correct specification. For each `ε` we draw `reps`
fresh samples of size `n`, fit DWLS, and call `ordinal_crmr_misspec_inference`
with `estimated_weight ∈ {true, false}`, comparing to a Monte-Carlo ground truth.

Checks (estimated-weight version):

- **Bias** (all ε): analytic `E[N·G] = N·G₀ + tr(QΓ_x)` vs the MC mean of `N·G`.
- **Variance under misspecification** (ε > 0): analytic `Var(N·G) = N·grad_var`
  vs the MC variance. This is the γ-channel magnitude **and sign** gate: a wrong
  sign would break the match. (The normal-theory variance is not the right
  object at the exact null, where the statistic is a χ² mixture; the exact-fit
  p-value covers that regime, so the variance is not compared to MC at ε = 0.)
- **Coverage** (all ε): the estimated-weight CI covers the population CRMR near
  the nominal level.
- **Dormancy** (ε = 0): with the residual ≈ 0 the γ channel is dormant, so the
  estimated and fixed-weight variances coincide.

## Result

The estimated-weight CRMR CI is calibrated. At `n = 1000`, `λ = 0.70`,
`reps = 2000`:

```
 eps    mean N*G (mc / analytic)   var N*G (mc / est)    coverage(.90)   gamma share
 0.00     2.13 / 2.08             4.6 / —  (dormant)      0.87            ~0.004
 0.12    20.9 / 21.3            59.6 / 61.9               0.88            ~0.026
 0.24    84.5 / 84.7           222.4 / 215.8              0.89            ~0.025
```

(`reps = 2000`, `n = 1000`, `λ = 0.70`.) The decisive
finding is the **gamma share**: the estimated-weight channel contributes only
about 2–3% of the CRMR sampling variance under misspecification, and essentially
zero at correct specification. That is the principled contrast with the
chi-square / RMSEA / nested-test family (where the weight *is* the metric and its
estimation dominates): **CRMR's fixed metric makes it largely robust to weight
estimation.** The CI is calibrated either way, so for CRMR the simpler
fixed-weight interval is already close; the estimated-weight version removes a
small, real residual miscalibration and quantifies exactly how small it is.

## Run

Requires `cmake --build --preset opt` first (links `build/opt/*.a`).

```sh
just quick   # reps=300, n=800   (fast, noisy coverage)
just all     # reps=2000, n=1000 (tight)
just clean
```

Override: `build/ordinal_crmr_verify --reps=… --n=… --n-pop=… --seed=…`.
