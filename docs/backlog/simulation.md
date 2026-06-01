# Simulation TODO

Simulation-specific state and remaining work for `magmaan::sim`.

This file exists to keep simulation-generator state separate from the main SEM
estimation/inference backlog in [`todo.md`](todo.md). Use it for NORTA,
independent generators, Vale-Maurelli/Fleishman, marginal moment matching,
simulation fixture policy, and simulation-performance notes. Keep estimator,
parser, matrix-representation, robust-test, and R-package work in
[`todo.md`](todo.md) unless the item is specifically about data generation.

The implementation-state summary still belongs in
[`../architecture/roadmap.md`](../architecture/roadmap.md). This file is the
simulation work queue and decision log.

## Current Simulation Surface

- `magmaan::sim` exposes NORTA calibration/sampling, independent marginal
  generators, Foldnes-Olsson independent-generator calibration, and
  Vale-Maurelli/Fleishman calibration/sampling.
- Marginal moment matching is available through `fit_marginal_to_moments()`.
  Tukey g-and-h, Pearson-system, and Johnson SU/SB families feed the same
  `MarginalSpec` transform path used by NORTA and independent generators.
- Pearson marginals follow PearsonDS conventions. Types 0/I/II/III/V/VI/VII
  are supported and checked against PearsonDS 1.3.2 goldens. Type IV is
  classified but still rejected pending a CDF/quantile policy.

## Pearson Type IV

Type IV is not a SEM-model feature. It is triggered by requested marginal
moments in the Pearson moment-matching family:

- input skewness `gamma1`
- input excess kurtosis `gamma2`
- Pearson raw kurtosis `beta2 = gamma2 + 3`
- squared skewness `beta1 = gamma1^2`

In the PearsonDS classifier mirrored by `fit_pearson_moment_match()`, Type IV is
the `0 < kappa < 1` branch. It appears quickly for asymmetric, heavy-tailed
moment pairs where kurtosis is high enough relative to skewness. It should be
treated as an important Pearson-family case, not rare polish, if Pearson becomes
a serious default or alternative generator family.

Observed examples from the current classifier:

| skewness | excess kurtosis | Pearson type |
| ---: | ---: | --- |
| 1 | 1 | I |
| 1 | 2 | IV |
| 1 | 5 | IV |
| 2 | 5 | I |
| 2 | 8 | VI |
| 2 | 10 | IV |
| 3 | 20 | VI |
| 3 | 30 | IV |

Implementation note:

- The Type IV moment-parameter formulas are straightforward and already present
  in PearsonDS `pearsonFitM()`: `m`, `nu`, `location`, and `scale`.
- GSL is not the core blocker for magmaan. PearsonDS uses GSL only for the
  complex log-gamma normalization in the density, and also ships a no-GSL C
  fallback for that normalization.
- The simulation path needs quantiles, not density evaluation. A dependency-free
  CDF can be computed as a normalized finite integral after
  `theta = atan((x - location) / scale)`:

```text
CDF(x) =
  integral[-pi/2, theta] exp(-nu*t) * cos(t)^(2*m - 2) dt
  /
  integral[-pi/2,  pi/2] exp(-nu*t) * cos(t)^(2*m - 2) dt
```

- Quantiles can then be computed by safeguarded bisection in `theta`, followed
  by `x = location + scale * tan(theta)`. This avoids GSL, avoids porting GPL
  PearsonDS hypergeometric code, and matches the existing project preference
  for small, validated numerical kernels.
- The existing vendored QUADPACK `qagi` is for semi-infinite intervals; Type IV
  only needs finite-interval quadrature. Prefer a small private adaptive
  Simpson or Gauss-Kronrod helper unless a broader finite-integral need emerges.

Validation expected for Type IV support:

- Extend `tests/tools/regen_pearson_sim_fixtures.R` so Type IV cases are
  supported and checked against PearsonDS 1.3.2 quantiles.
- Include multiple Type IV cases, at least `(skew=1, excess=2)`,
  `(skew=1, excess=5)`, `(skew=2, excess=10)`, `(skew=3, excess=30)`, and a
  negative-skew mirror.
- Add a direct transform smoke through either `simulate_independent_matrix()` or
  `calibrate_norta()` using a Type IV Pearson marginal, so the implemented
  quantile path is exercised beyond parameter goldens.
- Keep the fixture oracle in R/PearsonDS only. Do not add PearsonDS, gsl, or R
  package dependencies to the C++ core.

## Remaining Work

- **M.** Implement Pearson Type IV support for `MomentMatchFamily::Pearson`:
  parameter formulas, `validate_marginal()` acceptance, finite-integral CDF,
  quantile bisection, PearsonDS goldens, and one transform-path smoke.
- **S.** Decide whether Johnson SL should be exposed beyond the direct
  `MarginalSpec::johnson()` constructor.
- **S.** Decide the long-term special-functions policy before expanding the
  Pearson/Johnson surface. `src/detail_distribution_math.hpp` currently holds
  hand-rolled regularized beta/gamma, inverse beta/gamma, Student-t, and F-tail
  helpers shared by Pearson simulation and FMG p-values. Either keep these with
  dedicated goldens, vendor/use Boost.Math, or choose another vetted dependency.
- **M.** Harden NORTA calibration for larger simulation grids: cache pairwise
  correlation maps when marginal specs repeat, expose/interpolate the
  `rho_Z -> Corr(X_i, X_j)` map for repeated target matrices, and add an
  explicit policy for pairwise-calibrated latent matrices that are not positive
  definite. Error-only is the current behavior.
