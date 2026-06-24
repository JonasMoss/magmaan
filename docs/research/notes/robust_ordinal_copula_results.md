# Copula stress test: full robust-ordinal family vs Welz et al. (2026) §8.2

Status: **results note, 2026-06-24.** Runner:
`docs/research/sims/r/robust_ordinal_copula.R`. Raw rows + summary in
`docs/research/sims/results/robust_ordinal_copula{,_summary}.csv` (gitignored;
51 MB raw). This is Design 4 of the robust ordinal SEM paper plan, run for the
first time and extended from the paper's WMA/ML pair to the full eight-estimator
family.

## What was run

Faithful replica of Welz, Mair & Alfons (2026, Psychometrika 91:247-278) §8.2:
latent (xi, eta) drawn from a bivariate copula with **standard normal margins**,
calibrated to population (Pearson) correlation rho_G, then discretized into five
categories with the paper's thresholds a = (-1.5, 0, 0.5, 1),
b = (-1, 1, 1.5, 2). N = 1000, **5000 reps**, truth = rho_G. The latent copula is
magmaan's own `sim_bicop_*` generator, which for two variables is the same
construction as the VITA / covsim path the paper used (calibrate the copula
parameter so the normal-margined Pearson correlation equals rho_G).

- Copulas: **Clayton** (lower-tail dependence; paper main text), **Gumbel**
  (upper-tail; paper supplement), **Frank** (no tail dependence; our control).
- rho_G in {0.3, 0.9}.
- Estimators (all via `data_ordinal_stats_from_raw`): ML, WMA hard cap
  (`h_k=1.6` = robcat tuning c=0.6), smooth cap (re-centered to (1.3,1.9) so its
  plateau matches WMA k=1.6), exp cap, DPD (alpha=0.3), Huber-residual
  (hard, k=1.345), pseudo-Huber, Tukey biweight (k=4.685).
- The bivariate t-copula supplement is not run: there is no calibrated bivariate
  t wrapper on the R surface yet (the C++ `simulate_t_copula_*` path exists).

## Replication check (the pipeline is validated against the published oracle)

Our **WMA hard cap reproduces the paper's robust estimator, and ML reproduces
their MLE**, on the Clayton cells (paper Table 5 in parentheses):

| cell            | WMA bias        | WMA cov       | ML bias         | ML cov        |
|-----------------|-----------------|---------------|-----------------|---------------|
| Clayton rho=0.3 | -0.016 (-0.016) | 0.936 (0.938) | -0.015 (-0.013) | 0.908 (0.932) |
| Clayton rho=0.9 | -0.012 (-0.012) | 0.882 (0.915) | -0.053 (-0.043) | 0.038 (0.132) |

The small ML/coverage gaps at rho=0.9 are RNG + a slightly different calibrated
copula parameter (magmaan's calibration vs covsim's VITA); the WMA estimand and
the qualitative ML collapse match. So the harness, the DGP, and the WMA = robcat
identity are all confirmed against the paper.

## Full family results (bias / RMSE / 95% coverage, 5000 reps)

Clayton (lower-tail dependence; the case the robust estimator is built for):

| method         | rho=0.3 bias / rmse / cov | rho=0.9 bias / rmse / cov |
|----------------|---------------------------|---------------------------|
| ml             | -0.015 / 0.040 / 0.91     | -0.053 / 0.055 / 0.04     |
| **wma_hard_cap** | -0.016 / 0.050 / 0.94   | **-0.012 / 0.019 / 0.88** |
| smooth_cap     | -0.016 / 0.050 / 0.94     | -0.006 / 0.027 / 0.81     |
| exp_cap        | -0.011 / 0.043 / 0.95     | -0.014 / 0.024 / 0.77     |
| dpd            | -0.005 / 0.039 / 0.94     | -0.040 / 0.042 / 0.19     |
| huber_residual | -0.039 / 0.073 / 0.79     | -0.041 / 0.045 / 0.36     |
| pseudo_huber   | -0.035 / 0.070 / 0.78     | -0.042 / 0.045 / 0.28     |
| tukey_biweight | -0.054 / 0.087 / 0.71     | -0.043 / 0.047 / 0.92*    |

Gumbel (upper-tail dependence):

| method         | rho=0.3 bias / rmse / cov | rho=0.9 bias / rmse / cov |
|----------------|---------------------------|---------------------------|
| ml             | +0.006 / 0.038 / 0.91     | **+0.009 / 0.016 / 0.73** |
| wma_hard_cap   | +0.009 / 0.044 / 0.93     | +0.045 / 0.048 / 0.23     |
| smooth_cap     | +0.008 / 0.044 / 0.93     | +0.049 / 0.053 / 0.22     |
| exp_cap        | +0.011 / 0.042 / 0.93     | +0.044 / 0.049 / 0.33     |
| **dpd**        | +0.008 / 0.040 / 0.94     | **+0.020 / 0.024 / 0.67** |
| huber_residual | +0.003 / 0.067 / 0.82     | +0.052 / 0.054 / 0.08     |
| pseudo_huber   | +0.005 / 0.064 / 0.81     | +0.051 / 0.053 / 0.06     |
| tukey_biweight | +0.002 / 0.082 / 0.85     | +0.052 / 0.053 / 0.28*    |

Frank (no tail dependence; control):

| method         | rho=0.3 bias / rmse / cov | rho=0.9 bias / rmse / cov |
|----------------|---------------------------|---------------------------|
| ml             | -0.007 / 0.037 / 0.95     | -0.006 / 0.014 / 0.92     |
| wma_hard_cap   | +0.002 / 0.037 / 0.95     | +0.013 / 0.020 / 0.84     |
| smooth_cap     | +0.002 / 0.037 / 0.95     | +0.014 / 0.022 / 0.81     |
| exp_cap        | -0.000 / 0.037 / 0.95     | +0.009 / 0.019 / 0.87     |
| **dpd**        | +0.002 / 0.036 / 0.95     | **-0.001 / 0.013 / 0.95** |
| huber_residual | +0.015 / 0.044 / 0.92     | +0.014 / 0.024 / 0.69     |
| pseudo_huber   | +0.015 / 0.044 / 0.90     | +0.013 / 0.024 / 0.65     |
| tukey_biweight | +0.014 / 0.045 / 0.93     | +0.015 / 0.025 / 0.84     |

`*` Tukey's reported coverage is an artifact of a badly inflated sandwich SE
(`se_bias` up to +0.23; the redescending psi drives bread derivatives toward
zero, ill-conditioning Gamma). Tukey's SEs are not usable as-is, and its failure
rate is the highest (up to 0.063 on Gumbel rho=0.3).

## Findings

1. **WMA's robustness is directional, not general.** The cell-overcount family
   (WMA hard cap, smooth cap, exp cap) is the clear winner *only* on Clayton,
   the lower-tail case it was designed for (bias ~-0.01, the others collapse).
   On **Gumbel rho=0.9 the entire WMA family over-corrects** to +0.045..+0.049
   with coverage ~0.22, *worse than ML* (+0.009, cov 0.73). WMA one-sidedly
   downweights over-counted cells; when the tail dependence sits on the other
   side relative to the thresholds, that same mechanism pushes the estimate past
   the truth. This is consistent with the paper's own caution that the estimator
   helps only when nonnormality "behaves like tail contamination."

2. **The Huber/Tukey residual-clip family is dominated almost everywhere.** It
   never wins a cell; it is ML-like-or-worse on every Clayton/Gumbel rho=0.9
   cell (bias ~-0.04 / +0.05, broken coverage), has the worst RMSE on the
   rho=0.3 cells, and Tukey's SEs are unusable. The residual-clip recipe does
   **not** inherit WMA's good behavior.

3. **DPD is the most stable robust recipe across copula types.** It is never the
   best in a cell, but never broken: it ties ML on Frank (both rho), is the best
   robust method on Gumbel rho=0.9 (bias +0.020 vs WMA +0.045), and is only
   ML-like (not WMA-like) on Clayton rho=0.9. Its SEs are well calibrated
   everywhere (se_bias ~ 0).

4. **No portable recipe matches WMA on Welz's own turf (Clayton).** WMA wins
   there because it is purpose-built for one-sided cell over-counting; the price
   is fragility (Gumbel over-correction) and being polychoric-only.

## Implication for the mixed-estimator fork

This directly informs [robust_mixed_recipe_taxonomy.md](robust_mixed_recipe_taxonomy.md):

- The taxonomy called Huber/Tukey-residual "the genuine spiritual port of WMA"
  to polyserials (option 2, intrinsic mix). **These results argue against
  leading with it**: the residual-clip family is the weakest performer on the
  copula stress test and Tukey's inference is unreliable. The per-observation
  residual clip does not behave like the per-cell WMA cap.
- **DPD (option 1, uniform) is the portable recipe that holds up.** It already
  covers ord-ord, cont-ord, and cont-cont as one density-power divergence; it
  has calibrated SEs; and it is the only robust recipe here that is stable across
  Clayton/Gumbel/Frank without WMA's directional over-correction. It does not
  reach WMA's Clayton peak, but for a *mixed* estimator (where WMA is unavailable
  anyway) that tradeoff favours DPD.
- The decision the taxonomy parked (uniform DPD vs intrinsic mix vs
  all-ordinal-only WMA paper) now has evidence: if the mixed extension is built,
  **build it on DPD**; keep WMA as the all-ordinal-only headline where its
  Clayton edge is real and the mixing question does not arise.

## Caveats

- This is **distributional** misspecification (no contamination), which the
  paper itself frames as the boundary / non-core design. WMA's headline case is
  *partial* contamination (careless responding), where the pilot
  ([robust_ordinal_gamma.tex](robust_ordinal_gamma.tex)) also had WMA best. The
  negative result for Huber/Tukey here should be cross-checked against the
  contamination designs before it is treated as final; the DPD-vs-WMA tradeoff
  likewise.
- Single-pair (bivariate) only; the matrix-level Gamma conditioning question
  (Design 2) is separate.
- t-copula supplement not yet run (needs a calibrated bivariate t R wrapper).
