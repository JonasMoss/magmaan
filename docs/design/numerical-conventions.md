# Numerical conventions

The single source of truth for magmaan's objective-scale and test-statistic
conventions. Code references this file from `inference::chi2_stat`
(`include/magmaan/inference/inference.hpp`) and the estimator adapters.

## The objective scale: `fmin = ½·F`

Every estimator stores `est.fmin = ½·F`, where `F` is the statistical
**discrepancy** function. The optimiser minimises `½·F` for **all** estimators:

| family | discrepancy `F` | optimiser objective = `est.fmin` |
| --- | --- | --- |
| ULS | `‖s − σ̂(θ)‖²` | `½‖s − σ̂(θ)‖²` |
| GLS | `‖s − σ̂(θ)‖²_Ŵ` | `½‖s − σ̂(θ)‖²_Ŵ` |
| WLS | `‖s − σ̂(θ)‖²_W` | `½‖s − σ̂(θ)‖²_W` |
| DWLS / ordinal / mixed | weighted moment quadratic | `½·F` |
| ML | `log\|Σ\| + tr(SΣ⁻¹) − log\|S\| − p` (+ mean term) | `½·F_ML` |
| FIML | per-pattern averaged deviance | `½·F` |

The `½` is not a sum-of-squares artifact. It is the scaling that makes the
objective's Hessian equal the Fisher information:

- For least squares, `∇²(½‖r‖²_W) = JᵀWJ` (Gauss–Newton) — the moment-metric
  information, with no stray factor of 2.
- For ML, `∇²F_ML = 2·I` (e.g. scalar: `F''(s) = 1/s²`, per-obs Fisher
  `= 1/(2s²)`), so `∇²(½F_ML) = I`. The "morally present" `½` lives in the
  normal-theory weight `W_NT = ½(Σ⁻¹⊗Σ⁻¹)` — which is exactly the `½` baked
  into the GLS weight (`gmm::moment_quadratic`, the `0.5 * Wcov` block).

So `est.fmin` is simultaneously (i) the optimiser's minimum, (ii) half the
discrepancy, and (iii) the quantity whose curvature is the information — for
every estimator. lavaan minimises `½F` for ML as well, so magmaan's stored
`fmin` matches lavaan's stored `fmin` element-for-element (the parity goldens
compare them directly, no factor).

### Where the `½` is applied

The `½` lives **only in the optimiser adapter** — the lambda handed to the
optimiser:

- `estimate::ml_objective` (ModelEvaluator and FcSemEvaluator overloads,
  `src/estimate/nt.cpp`).
- the FIML `eval_at` lambda (`src/estimate/fiml.cpp`).
- the LS path is already halved in `optim::scalarize` / `evaluate_ls_objective`
  (`0.5 * r.squaredNorm()`).

**Invariant — never halve the math kernels.** `ml_value`, `ml_value_gradient`,
`ml_gradient_block`, and `FIML::value` / `gradient` / `value_gradient` stay at
**full-F** scale, because:

- the information builders (`inference::information_expected`,
  `information_observed_fd`, `..._analytic`, and `fiml_observed_information`'s
  `fiml_observed_hessian_fd`) differentiate those kernels and carry their own
  structural `n_b/2` (ML) or `½·N` (FIML) factors — independent of the objective
  scalar. Halving a kernel would silently halve an information matrix and
  double the standard errors.
- the score / modification-index tests pair the kernel gradient (scale F) with
  the structured information (scale F).

`fit_ml_fisher`'s `expected_ml_hessian_f` is the one place that compensates: it
returns the expected Hessian of `½F` (its block weight carries the `½`) so the
Fisher step `(½H)⁻¹(½∇F) = H⁻¹∇F` is unchanged while gradient and objective
live on the uniform `½F` scale.

## The test statistic: `T = 2N·fmin = N·F`

The goodness-of-fit χ² is `T = 2·N·fmin = N·F` for every estimator, via the
contract documented at `inference::chi2_stat`. `N` (total sample size), not
`N−1`: this matches lavaan's `likelihood = "normal"` default. magmaan uses `N`
as the statistic multiplier throughout and reserves `N−1` only where it is
genuinely the unbiased quantity (the sample-covariance divisor is `N−1`).

### Deliberate exceptions (documented at their own sites)

1. **Continuous ULS standard GOF** uses Browne's residual normal-theory
   statistic, not `2N·fmin` — `N·F_ULS` is not asymptotically χ². See
   `continuous_ls_chisq` (`weight.empty()` branch). The robust ULS path
   (`se = "robust.sem"`) uses the `2N·fmin` base (`robust_ls_standard_chisq`);
   this standard-vs-robust split mirrors lavaan and is pinned by the
   `hs_3factor_ls*` parity fixtures.
2. **FIML standard GOF** is the likelihood-ratio statistic
   `−2(logl − logl_sat)` (`fiml::fiml_extras`), which recomputes the full-F
   deviance from `θ̂`; the `2N·fmin` identity holds but the LRT is the reported
   quantity.
3. **lavaan's `(N−G)·F` offset.** lavaan reports GLS/WLS χ² against `N−G`
   (Wishart/unbiased), differing from magmaan's `N·F` by exactly `(N−G)/N`.
   ULS already carries `N−G` via `browne_residual_nt`'s `n_used`, so it matches
   lavaan directly. The `(N−G)/N` factor is applied **test-side** in the parity
   goldens (`lavaan_parity_golden_test.cpp`, `ls_golden_test.cpp`), never inside
   the statistic. It is an honesty correction in the comparison, not an
   estimator choice.

## Implications for callers

- `chi2_stat(samp, est) == 2·N·est.fmin == N·F`. The R binding
  `infer_chi2_stat(ss, fit$fmin)` is unchanged in value (fit$fmin is now `½F`).
- Robust scaled tests (Satorra–Bentler, scaled-shifted, mean-var-adjusted)
  consume this base statistic; their values are unchanged.
- Fit measures (RMSEA, CFI, TLI) consume the χ² above; the independence
  baseline χ² is computed from `S` directly, not from `fmin`.
- Anything reporting `est.fmin` as a number now reports `½F` for ML/FIML/ordinal
  too (previously those stored full `F`); the GOF statistic is unchanged.
