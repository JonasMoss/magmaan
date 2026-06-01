# Simulation TODO

Simulation-specific state and remaining work for `magmaan::sim`.

This file exists to keep simulation-generator state separate from the main SEM
estimation/inference backlog in [`todo.md`](todo.md). Use it for the simulation
sublibrary roadmap, NORTA, independent generators, Vale-Maurelli/Fleishman,
marginal moment matching, ordinal/mixed-data projection, model-implied data
generation, simulation fixture policy, and simulation-performance notes. Keep
estimator, parser, matrix-representation, robust-test, and R-package work in
[`todo.md`](todo.md) unless the item is specifically about data generation.

The implementation-state summary still belongs in
[`../architecture/roadmap.md`](../architecture/roadmap.md). This file is the
simulation work queue and decision log.

## Current Simulation Surface

- `magmaan::sim` exposes NORTA calibration/sampling, independent marginal
  generators, Foldnes-Olsson independent-generator calibration, and
  Vale-Maurelli/Fleishman calibration/sampling.
- Baseline multivariate-normal generation is available through
  `simulate_normal_matrix()` and `simulate_normal_raw()`, taking explicit
  population means and covariance matrices. This is the low-level normal
  generator target for future model-implied SEM simulation.
- Reusable observed-variable projection is available through
  `thresholds_from_probabilities()`, `project_ordinal_matrix()`, and
  `project_mixed_matrix()`. Projection thresholds continuous latent responses
  into one-based ordinal categories, preserves continuous columns in mixed data,
  and returns ordered masks, declared level counts, and category counts for
  downstream ordinal/mixed workspace builders.
- Marginal moment matching is available through `fit_marginal_to_moments()`.
  Tukey g-and-h, Pearson-system, Johnson SU/SB, and Fleishman polynomial
  families feed the same `MarginalSpec` transform path used by NORTA and
  independent generators. Fleishman uses the Vale-Maurelli coefficient solver
  and is deliberately treated as a normal generator transform; it is not exposed
  as a guaranteed quantile because the fitted cubic need not be monotone.
- Pearson marginals follow PearsonDS conventions. Types 0/I/II/III/IV/V/VI/VII
  are supported and checked against PearsonDS 1.3.2 goldens. Type IV uses a
  dependency-free finite-integral CDF and bisection quantile path.
- Johnson SU/SB marginals are checked against SuppDists 1.1.9.9
  `JohnsonFit`/`qJohnson` goldens (shape pair, type, and quantiles to ~1e-7).
  The fixture targets the realized moments of SuppDists's returned shape so the
  comparison is exact rather than capped by SuppDists's loose moment solve;
  `tests/tools/regen_johnson_sim_fixtures.R` regenerates it.

## Architecture Direction

The simulation sublibrary is intentionally maximalist: if a data-generation
mechanism is used in the SEM/simulation literature, or has an obvious extension
needed to compare those mechanisms fairly, it belongs here eventually. Keep that
coverage organized as a layered simulation stack rather than as unrelated
top-level named generators.

The preferred decomposition is:

1. **Population construction.** Build explicit population means/covariances,
   group-specific population moments, thresholds, and eventually model-implied
   population moments from a lavaanified model or fitted estimates.
2. **Continuous generator.** Draw a complete continuous response matrix from a
   named mechanism: normal, elliptical, copula/NORTA, independent-generator,
   Vale-Maurelli/Fleishman, PLSIM, VITA/covsim, or later mechanisms.
3. **Observed-variable projection.** Apply mean/intercept structure,
   threshold/discretization rules, group labels, missingness, contamination, and
   other observed-data transformations. Ordinal and mixed simulation should be
   projection layers shared by all continuous generators, not duplicated inside
   each generator.
4. **Diagnostics and calibration artifacts.** Return or expose achieved moments,
   calibrated latent/intermediate correlations, fitted marginal parameters,
   thresholds, infeasible pairwise maps, positive-definiteness failures, and
   stochastic validation summaries.

This shape keeps named literature methods narrow: each one either constructs a
population, generates continuous responses, projects observed variables, or
calibrates/diagnoses one of those steps.

## Planned Surface

- Add explicit population structs for continuous and mixed data, including
  means, covariance/correlation, group-specific variants, thresholds, ordered
  columns, and optional variable names/levels when wrapping `data::RawData`.
- Add model-implied simulation as a lowering step: convert `MatrixRep` /
  estimates / fitted objects into population moments and projection specs, then
  call the generic simulation stack. Mean structures, intercepts, latent means,
  group means, and thresholds should live here instead of being baked into each
  generator.
- Add ordinal/mixed correlation calibration after the direct projection path is
  stable. The first version can generate from latent correlations and report
  achieved observed correlations; a later version should calibrate latent
  correlations to requested observed Pearson/polychoric/polyserial summaries.
- Add an elliptical-generator family: multivariate Student-t, contaminated
  normal, slash, and a generic scale-mixture-of-normals hook. Add power
  exponential / generalized-normal variants if the literature use cases need
  them.
- Add pseudo-elliptical / transformed-elliptical mechanisms where a radial or
  spherical/elliptical core is combined with marginal transformations. Keep the
  boundary clear with NORTA/copula methods: pseudo-elliptical methods are
  continuous-generator mechanisms, not observed-data projections.
- Add copula families beyond Gaussian NORTA: t-copula first, then Archimedean
  Clayton, Gumbel, Frank, and Joe if needed for robust/ordinal simulation
  studies. Reuse the same marginal and projection layers.
- Add PLSIM and VITA/covsim later. Both are literature-relevant, but they should
  plug into the same continuous-generator interface after normal, mixed
  projection, and elliptical/copula foundations are in place.

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

Implemented policy:

- The Type IV moment-parameter formulas mirror PearsonDS `pearsonFitM()`:
  `m`, `nu`, `location`, and `scale`.
- GSL is not the core blocker for magmaan. PearsonDS uses GSL only for the
  complex log-gamma normalization in the density, and also ships a no-GSL C
  fallback for that normalization.
- The simulation path needs quantiles, not density evaluation. The implemented
  CDF computes a normalized finite integral after
  `theta = atan((x - location) / scale)`:

```text
CDF(x) =
  integral[-pi/2, theta] exp(-nu*t) * cos(t)^(2*m - 2) dt
  /
  integral[-pi/2,  pi/2] exp(-nu*t) * cos(t)^(2*m - 2) dt
```

- Quantiles are computed by safeguarded bisection in `theta`, followed by
  `x = location + scale * tan(theta)`. This avoids GSL, avoids porting GPL
  PearsonDS hypergeometric code, and matches the existing project preference
  for small, validated numerical kernels.
- The existing vendored QUADPACK `qagi` is for semi-infinite intervals; Type IV
  only needs finite-interval quadrature. Type IV currently uses a small private
  adaptive Simpson helper in `src/sim/norta.cpp`; revisit only if a broader
  finite-integral need emerges.

Validation:

- `tests/tools/regen_pearson_sim_fixtures.R` includes Type IV cases for
  `(skew=1, excess=2)`, `(skew=1, excess=5)`, `(skew=2, excess=10)`,
  `(skew=3, excess=30)`, and a negative-skew mirror.
- `tests/unit/norta_test.cpp` checks fitted Type IV parameters and quantiles
  against PearsonDS 1.3.2 and includes a direct `simulate_independent_matrix()`
  transform smoke.
- Keep the fixture oracle in R/PearsonDS only. Do not add PearsonDS, gsl, or R
  package dependencies to the C++ core.

## Remaining Work

- **M.** Extend the ordinal/mixed projection layer with group-specific
  thresholds, variable names/level labels for raw-data wrapping, and richer
  achieved-proportion diagnostics.
- **L.** Add model-implied simulation that lowers lavaanified/model evaluator
  state to population moments and projection specs before invoking the generic
  simulation stack. Treat mean structure as population construction/projection,
  not as a generator-specific option.
- **M.** Add first elliptical generators: Student-t, contaminated normal, slash,
  and a generic scale-mixture-of-normals entry point. Validate moments and tail
  behavior with deterministic formulas where available plus stochastic smokes.
- **M.** Add t-copula simulation as the first non-Gaussian copula extension.
  Keep it on the existing marginal/projection path and expose calibration
  artifacts analogously to NORTA.
- **M.** Add ordinal/mixed observed-correlation calibration once direct
  threshold projection is stable: pairwise thresholded correlation maps,
  calibration to target observed Pearson/polychoric/polyserial summaries, and a
  policy for non-positive-definite calibrated latent matrices.
- **S.** Add pseudo-elliptical / transformed-elliptical mechanisms after the
  first elliptical slice clarifies the shared radial/core interfaces.
- **S.** Add Archimedean copulas as needed by simulation studies: Clayton,
  Gumbel, Frank, and Joe.
- **S.** Add PLSIM/piecewise-linear simulation once the shared projection and
  diagnostics interfaces exist.
- **S.** Add VITA/covsim-style simulation once copula and projection
  foundations make the implementation local rather than architecture-driving.
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
