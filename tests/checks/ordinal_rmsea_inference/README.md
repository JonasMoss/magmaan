# RMSEA estimated-weight confidence-interval verification

Advisory local check for `estimate::ordinal_rmsea_misspec_inference`, the
misspecification-aware confidence interval for categorical RMSEA that propagates
the sampling variability of the estimated polychoric weight (the γ channel).
Outside the default test suite and CI; links the optimized static libs and runs
a Monte-Carlo study. The absolute-fit, large-γ counterpart to
`tests/checks/ordinal_crmr_inference`.

## What it verifies

RMSEA's criterion is the DWLS discrepancy `F = rᵀ W r` itself, whose metric **is**
the estimated weight `W = diag(Γ̂)⁻¹`, so its γ channel is large. Because the
criterion is the value function θ̂ minimizes, the envelope theorem gives the
gradient as the bare profile score `g_F = (−2Wr, −r²/γ²)` (no estimator
projection). The CI is the normal-theory interval on `F₀` with
`Var(N·F) = N·g_Fᵀ Γ_x g_F`, mapped through `sqrt(··G/df)` to RMSEA; the
exact-fit p-value uses the `QΓ_x` mixture at the null.

Population: the four-cycle (C4) binary design (p = 4 binary indicators,
thresholds 0, latent correlations `base = λ²` on the two opposite pairs,
`base + ε` on the four C4 edges). We fit the congeneric one-factor model; for
`ε > 0` it is misspecified (residual ≠ 0, γ channel live); `ε = 0` is correct
specification. For each `ε` we draw `reps` fresh samples of size `n`, fit DWLS,
and call `ordinal_rmsea_misspec_inference` with `estimated_weight ∈ {true,false}`,
comparing to Monte-Carlo. Checks: bias (`E[N·F] = N·F₀ + tr(QΓ_x)` vs MC mean);
variance (`N·grad_var` vs MC variance — the γ magnitude/sign gate); coverage of
the population RMSEA; ε=0 dormancy.

## Result

`reps = 2000`, `n = 1000`, `λ = 0.70`, nominal 90%:

```
 eps   RMSEA_pop   mean N*F (mc/an)   var N*F (mc / est / fixed)   coverage (est / fixed)   gamma share
 0.00    0.000      1.20 / 1.21        1.6 / — / —  (dormant)        ~1.0 / ~1.0 (boundary)   ~0.00
 0.12    0.079     13.70 / 13.58      20.1 / 23.0 / 25.9             0.91 / 0.93              −0.12
 0.24    0.166     56.60 / 56.45      48.5 / 54.5 / 98.2             0.92 / 0.98              −0.80
```

The estimated-weight RMSEA CI is **calibrated** (coverage ≈ nominal, ~0.91–0.92);
the bias correction matches MC to <1%. The headline is the **gamma share**: the
γ channel is large (−0.12 to −0.80 of the variance) and **variance-reducing** —
`r` and `γ` co-vary negatively, so properly accounting for the estimated weight
*shrinks* the variance. Ignoring it (the fixed-weight comparator) **over-states
the variance**, up to ~2× at ε = 0.24, so the fixed-weight CI is increasingly
**conservative / too wide** (coverage 0.93 → 0.98).

This is the large-γ contrast with CRMR (where the metric is fixed and the γ
channel is only ~2–3%): for RMSEA the estimated weight matters a lot, and the
correction *tightens* the interval to nominal rather than (as one might guess
from the nested-test result, experiment 36) widening it. Same theme as the rest
of the program — the fixed-weight object is miscalibrated — but here the
direction is conservative, not anti-conservative.

(The estimated-weight variance mildly over-states MC at the strongest setting —
54.5 vs 48.5 — a finite-sample normal-approximation effect, leaving the CI very
slightly conservative; it shrinks with `n`.)

## Run

Requires `cmake --build --preset opt` first (links `build/opt/*.a`).

```sh
just quick   # reps=300, n=800   (fast, noisy)
just all     # reps=2000, n=1000 (tight)
just clean
```

Override: `build/ordinal_rmsea_verify --reps=… --n=… --n-pop=… --seed=…`.
