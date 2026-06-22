# CFI/TLI estimated-weight confidence-interval verification

Advisory local check for `estimate::ordinal_cfi_tli_misspec_inference`, the
misspecification-aware confidence intervals for the categorical incremental fit
indices CFI and TLI that propagate the sampling variability of the estimated
polychoric weight (the γ channel). Outside the default test suite and CI; links
the optimized static libs and runs a Monte-Carlo study. The first *incremental*
(two-model) companion to `tests/checks/ordinal_{rmsea,crmr}_inference`.

## What it verifies

CFI and TLI compare the user model to the independence baseline through a ratio
of two noncentralities, `δ_u = T_u − Q̄_u` and `δ_b = T_b − Q̄_b`, that share one
joint NACOV `Γ_x` (generalized df `Q̄ = tr(Q Γ_x)`). CFI `= 1 − δ_u/δ_b` and
TLI `= 1 − (Q̄_b/Q̄_u)·δ_u/δ_b` are a ratio delta-method; the joint law is the
single bilinear form `Cov(T_u,T_b) = N gᵤᵀ Γ_x g_b` with the envelope-score
gradients `g = (−2Wd, −d²/γ²)`. See
`docs/research/notes/cfi_tli_misspec_inference.tex`.

Population: the four-cycle (C4) binary design (p = 4 binary indicators,
thresholds 0, latent correlations `base = λ²` on the two opposite pairs,
`base + ε` on the four C4 edges). We fit the congeneric one-factor model; for
`ε > 0` it is misspecified (`CFI₀ < 1`, user γ channel live); `ε = 0` is correct
specification (`CFI₀ = 1`, boundary). For each `ε` we draw `reps` fresh samples
of size `n`, fit DWLS, and call `ordinal_cfi_tli_misspec_inference` with
`estimated_weight ∈ {true,false}`. Checks: point unbiasedness (MC mean of
`1 − δ_u/δ_b` vs population `CFI₀`, likewise TLI); coverage of `CFI₀`/`TLI₀`
near nominal; with the delta-method variance, the γ share, and the
baseline-dominated leading term `V_uu/δ_b²` reported as diagnostics. Coverage
uses the UNCLAMPED normal interval (the reported CI is clamped to `[0,1]`).

## Result

`reps = 2000`, `n = 1000`, `λ = 0.70`, nominal 90%:

```
 eps   CFI_pop  TLI_pop   CFI cov(est/fix)  CFI γ-share  lead V_uu/δ_b² ratio   TLI cov(est)  TLI an_var/mc_var
 0.00  1.000    1.000      0.875 / 0.868      +0.04        0.99 (boundary)        0.875         0.0014 / 0.0007
 0.12  0.991    0.735      0.909 / 0.910      −0.03        0.84                   0.939         0.013  / 0.009
 0.24  0.980    0.024      0.904 / 0.888      +0.08        0.35                   1.000         0.053  / 0.006
```

Three findings:

1. **CFI is calibrated.** Coverage ≈ nominal (0.90–0.91 at `ε > 0`); the point
   is unbiased and the delta-method variance matches Monte-Carlo (e.g. 1.4e-4 vs
   1.2e-4 at `ε = 0.12`).

2. **CFI is largely robust to weight estimation.** The γ share of the CFI
   variance is small (≈ ±3–8%), so the estimated- and fixed-weight intervals
   nearly coincide. This is the CRMR-like end of the spectrum, *opposite* to
   RMSEA (γ share −12% to −80%): even though CFI's leading term `V_uu` is the
   RMSEA variance, the ratio structure and the `δ_b²` scaling attenuate the
   channel, and the `δ_b` (baseline) shift under fixed weight largely cancels it.

3. **The baseline-dominated leading order degrades with misspecification.** The
   ratio `(V_uu/δ_b²) / Var(CFI)` falls 0.99 → 0.84 → 0.35: at weak misfit
   `Var(CFI) ≈ V_uu/δ_b²` (a rescaling of the RMSEA-side variance), but as misfit
   grows the cross-term `−2r V_ub` and baseline-variance `r² V_bb` corrections
   take over. The implementation uses the full bivariate form, so coverage stays
   calibrated regardless; the simplification is intuition, not the computation.

**TLI** is unbiased and calibrated at weak/moderate misspecification
(coverage 0.94 at `ε = 0.12`) but its analytic variance **over-states** at strong
misspecification (`ε = 0.24`: 0.053 vs MC 0.006, coverage → 1.00). The cause is
`c = Q̄_b/Q̄_u`: when the user generalized df `Q̄_u` is small (and occasionally
near the signed-trace cancellation point at strong misfit), `c²` has a heavy
right tail, so averaging the per-sample `Var(TLI) = c²·Var(CFI)` over-states.
The TLI interval is therefore conservative, not anti-conservative. Fixed-weight
TLI coverage is 0.00 by construction: the fixed weight changes `Q̄`, hence `c`,
hence the **TLI estimand**, so its interval targets a different value than the
estimated-weight `TLI₀` (this comparison is not meaningful and is only printed).

Practical reading: for incremental-fit inference under estimated categorical
weights, **CFI carries a trustworthy interval**; TLI's point is fine but its
interval should be read as conservative and is unreliable at strong misfit. A
stabilized TLI variance (robust treatment of `c`) is future work.

## Run

Requires `cmake --build --preset opt` first (links `build/opt/*.a`).

```sh
just quick   # reps=300, n=800   (fast, noisy)
just all     # reps=2000, n=1000 (tight)
just clean
```

Override: `build/ordinal_cfi_verify --reps=… --n=… --n-pop=… --seed=…`.
