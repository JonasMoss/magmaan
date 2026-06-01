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
  generators, Foldnes-Olsson independent-generator calibration,
  Vale-Maurelli/Fleishman calibration/sampling, PLSIM piecewise-linear
  calibration/sampling, fixed-parameter t-copula sampling, fixed-parameter
  bivariate Archimedean copula sampling, and a first VITA/covsim-style
  bivariate copula observed-Pearson calibration layer.
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
- Population composition structs are available through `ContinuousPopulation`,
  `MixedPopulation`, `CopulaPopulation`, and `MixedPopulationDraw`. They
  compose explicit mean/covariance population moments or copula marginals with
  continuous generators and the observed projection layer; model-implied SEM
  simulation should lower into this surface.
- The first elliptical/scale-mixture generators are available:
  `simulate_student_t_*`, `simulate_contaminated_normal_*`,
  `simulate_slash_*`, and `simulate_scale_mixture_normal_*`. These APIs treat
  the supplied covariance as the target population covariance when the variance
  exists, rescaling the internal normal core as needed.
- Marginal moment matching is available through `fit_marginal_to_moments()`.
  Tukey g-and-h, Pearson-system, Johnson SU/SB, and Fleishman polynomial
  families feed the same `MarginalSpec` transform path used by NORTA and
  independent generators. Fleishman uses the Vale-Maurelli coefficient solver
  and is deliberately treated as a normal generator transform; it is not exposed
  as a guaranteed quantile because the fitted cubic need not be monotone.
- PLSIM is available through `fit_plsim_marginal()`, `diagnose_plsim()`,
  `calibrate_plsim()`, `simulate_plsim_matrix()`, and `simulate_plsim_raw()`.
  The first slice uses regular normal-quantile breakpoints, solves slopes for
  target marginal skewness/excess kurtosis, exposes Hermite coefficients, and
  calibrates pairwise intermediate normal correlations with selectable
  covariance evaluators: Hermite-series, bivariate Gauss-Hermite quadrature,
  conditional-normal rectangle moments, and Hermite-initialized refinement for
  either deterministic path. Hermite is the default fast path. The rectangle
  evaluator follows the Foldnes-Grønneberg segment decomposition directly,
  reducing each bivariate rectangle probability/first/cross-moment calculation
  to one-dimensional adaptive integration over the conditioning normal variable.
  `diagnose_plsim()` keeps pairwise feasibility bounds, per-pair errors, the
  partial intermediate matrix, achieved correlations, and the intermediate
  minimum eigenvalue available when calibration fails.
- Pearson marginals follow PearsonDS conventions. Types 0/I/II/III/IV/V/VI/VII
  are supported and checked against PearsonDS 1.3.2 goldens. Type IV uses a
  dependency-free finite-integral CDF and bisection quantile path.
- Johnson SU/SB marginals are checked against SuppDists 1.1.9.9
  `JohnsonFit`/`qJohnson` goldens (shape pair, type, and quantiles to ~1e-7).
  The fixture targets the realized moments of SuppDists's returned shape so the
  comparison is exact rather than capped by SuppDists's loose moment solve;
  `tests/tools/regen_johnson_sim_fixtures.R` regenerates it.
- t-copula sampling is intentionally a copula-parameter generator, not an
  observed-Pearson-correlation calibrator. `simulate_t_copula_matrix()` draws a
  multivariate Student-t copula with supplied correlation matrix and degrees of
  freedom, then feeds uniforms through the existing marginal quantile path.
  Fleishman generator transforms are rejected because they are not guaranteed
  quantiles. VITA/covsim work sits above the fixed-parameter generator layer as
  calibration from requested observed moments/correlations to generator
  parameters.
- Fixed-parameter bivariate Archimedean copula sampling is available through
  `BivariateCopulaSpec`, `simulate_bivariate_copula_matrix()`, and
  `simulate_bivariate_copula_raw()`. The first families are independence,
  Clayton, Gumbel, Frank, and Joe. Sampling uses conditional inversion of the
  bivariate copula and then the same marginal quantile path as t-copula
  sampling. `tests/tools/regen_copula_sim_fixtures.R` regenerates
  `tests/fixtures/sim/bivariate_copula_hfunc.json`, which checks the
  conditional CDF and inverse conditional CDF against rvinecopulib `hbicop()`.
  `bivariate_copula_tau()` and `bivariate_copula_from_tau()` expose the
  Kendall-tau parameterization used by vine/copula workflows; Clayton, Gumbel,
  and Frank use closed-form or Debye-function formulas, while Joe uses the
  standard convergent series. `bivariate_copula_observed_corr()` evaluates the
  deterministic quadrature-implied observed Pearson correlation after the
  marginal quantile transforms, and `calibrate_bivariate_copula_correlation()`
  bisects on Kendall tau to hit a target pairwise correlation for one family.
  `calibrate_bivariate_copula_correlation_matrix()` applies that calibration to
  every off-diagonal entry of a target correlation matrix and returns pairwise
  copula parameters, achieved correlations, feasible bounds, and iteration
  counts. It also reports the maximum achieved-correlation error and the raw
  minimum eigenvalue of the achieved matrix, with opt-in error/ridge/shrinkage
  repair toward a requested minimum eigenvalue. This is a matrix
  diagnostic/calibration layer, not an automatic matrix-to-vine fitter.
  Explicit three-variable C-vine sampling is available through
  `CVine3CopulaSpec`, `cvine3_copula_inverse_rosenblatt()`,
  `simulate_cvine3_copula_uniforms()`, `simulate_cvine3_copula_matrix()`, and
  `simulate_cvine3_copula_raw()`. The first structure uses variable 0 as the
  root, pair copulas for `0-1` and `0-2`, and a conditional pair copula for
  `1-2|0`. `tests/fixtures/sim/cvine3_inverse_rosenblatt.json` checks the
  deterministic inverse Rosenblatt transform against rvinecopulib
  `inverse_rosenblatt()` for the equivalent `cvine_structure(c(3,2,1))`.
  `cvine3_copula_observed_corr()` deterministically evaluates the implied
  observed correlation matrix, and `calibrate_cvine3_copula_correlation()`
  calibrates the two root pair copulas plus the conditional pair copula to a
  3x3 target observed-correlation matrix.
  `calibrate_cvine3_copula_correlation_select_root()` tries all three possible
  roots, restores achieved correlations to the caller's original variable
  order, and reports the selected root/order. `CVine3FamilySpec` and the
  overloaded `calibrate_cvine3_copula_correlation()` support per-edge family
  choices, while `calibrate_cvine3_copula_correlation_select_families()` tries
  a caller-provided candidate set for the fixed root-0 C-vine.
  `calibrate_cvine3_copula_correlation_select_structure()` tries all three
  roots and all candidate edge-family triples, skips infeasible fits, and
  returns achieved correlations in the caller's original variable order.
  Higher-dimensional vines, broader structure/family policies, and
  ordinal/polyserial/polychoric calibration remain separate work.

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

- Add model-implied simulation as a lowering step: convert `MatrixRep` /
  estimates / fitted objects into population moments and projection specs, then
  call the generic simulation stack. Mean structures, intercepts, latent means,
  group means, and thresholds should live here instead of being baked into each
  generator.
- Add ordinal/mixed correlation calibration after the direct projection path is
  stable. The first version can generate from latent correlations and report
  achieved observed correlations; a later version should calibrate latent
  correlations to requested observed Pearson/polychoric/polyserial summaries.
- Extend the elliptical-generator family beyond the first scale-mixture slice:
  add power exponential / generalized-normal variants if the literature use
  cases need them, and add diagnostics for theoretical vs achieved tail
  behavior.
- Add pseudo-elliptical / transformed-elliptical mechanisms where a radial or
  spherical/elliptical core is combined with marginal transformations. Keep the
  boundary clear with NORTA/copula methods: pseudo-elliptical methods are
  continuous-generator mechanisms, not observed-data projections.
- Add copula families beyond Gaussian NORTA: fixed-parameter t-copula and
  bivariate Archimedean independence/Clayton/Gumbel/Frank/Joe are landed.
  Reuse the same marginal and projection layers for later calibration and
  broader copula surfaces.
- Use `vinecopulib` / `rvinecopulib` as an oracle/reference for copula-family
  parameter conventions, simulation checks, and later vine work. Do not make it
  a core runtime dependency unless the exception/dependency boundary is
  deliberately revisited; the C++ core stays local and `std::expected` based.
- Extend the first PLSIM slice with pair-calibration caching, broader
  simulation-grid diagnostics, and possible tuning/replacement of the current
  adaptive rectangle integration kernel. Extend the first pairwise VITA/covsim
  copula calibration into a full matrix-oriented workflow.

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
- **M.** Extend population composition with group-specific population blocks
  and raw-data naming/level metadata once the `RawData` carrier grows those
  fields.
- **L.** Add model-implied simulation that lowers lavaanified/model evaluator
  state to population moments and projection specs before invoking the generic
  simulation stack. Treat mean structure as population construction/projection,
  not as a generator-specific option.
- **M.** Add elliptical diagnostics/goldens for Student-t, contaminated normal,
  slash, and finite scale mixtures: deterministic moment formulas where
  available plus stochastic smokes.
- **Landed, first non-Gaussian copula slice.** Add fixed-parameter t-copula
  simulation through `TCopulaSpec`, `simulate_t_copula_matrix()`, and
  `simulate_t_copula_raw()`. It reuses the existing marginal quantile path and
  validates that marginals are true quantile marginals. The population
  composition layer also exposes `simulate_mixed_population_t_copula()` to feed
  generated continuous copula data through the shared observed projection path,
  leaving observed-correlation calibration to later VITA/covsim-style work.
- **M.** Add ordinal/mixed observed-correlation calibration once direct
  threshold projection is stable: pairwise thresholded correlation maps,
  calibration to target observed Pearson/polychoric/polyserial summaries, and a
  policy for non-positive-definite calibrated latent matrices.
- **S.** Add pseudo-elliptical / transformed-elliptical mechanisms after the
  first elliptical slice clarifies the shared radial/core interfaces.
- **Landed, first bivariate Archimedean slice.** Add fixed-parameter bivariate
  independence, Clayton, Gumbel, Frank, and Joe copula simulation. The runtime
  implementation is local and uses conditional inversion; validate future
  parameter-convention fixtures against vinecopulib or its R interface unless a
  broader vine backend is explicitly chosen. `simulate_mixed_population_bivariate_copula()`
  composes the same generator with observed ordinal/mixed projection.
- **Landed, first VITA/covsim matrix slice.** Add deterministic bivariate copula
  observed-correlation evaluation and Kendall-tau bisection calibration through
  `bivariate_copula_observed_corr()`,
  `calibrate_bivariate_copula_correlation()`, and
  `calibrate_bivariate_copula_correlation_matrix()`. The current matrix scope
  is pairwise calibration/diagnostics for one bivariate family across quantile
  marginals, including maximum-error/minimum-eigenvalue diagnostics and opt-in
  error/ridge/shrinkage repair for indefinite achieved matrices. Joint copula
  assembly and ordinal/polyserial/polychoric calibration remain.
- **Landed, first explicit vine sampler.** Add `CVine3CopulaSpec` and
  `simulate_cvine3_copula_*()` for a fixed three-variable C-vine with root 0.
  It composes the local bivariate h-functions/inverses and supports explicit
  `0-1`, `0-2`, and `1-2|0` copulas.
- **Landed, first matrix-to-vine fit.** Add
  `cvine3_copula_observed_corr()` and
  `calibrate_cvine3_copula_correlation()` for the fixed root-0 C-vine. The fit
  calibrates root pairs against `r01`/`r02`, then bisects the conditional
  copula against the full C-vine implied `r12`.
- **Landed, root selection for 3-variable C-vines.** Add
  `calibrate_cvine3_copula_correlation_select_root()` to try roots 0, 1, and 2
  by permutation, skip infeasible fits, and return the best achieved matrix in
  the original variable order.
- **Landed, per-edge family selection for fixed root-0 C-vines.** Add
  `CVine3FamilySpec` plus `calibrate_cvine3_copula_correlation_select_families()`
  to fit `0-1`, `0-2`, and `1-2|0` with different family choices from a
  caller-provided candidate set.
- **Landed, combined root/family selection for 3-variable C-vines.** Add
  `calibrate_cvine3_copula_correlation_select_structure()` to try all roots and
  candidate edge-family triples, skip infeasible fits, and return the best
  achieved matrix in the original variable order.
- **Landed, first PLSIM slice.** Add piecewise-linear simulation through
  `fit_plsim_marginal()`, `diagnose_plsim()`, `calibrate_plsim()`,
  `simulate_plsim_matrix()`, and `simulate_plsim_raw()`. Unit coverage checks
  the skewness 2 / excess kurtosis 5 condition where Fleishman fails,
  Hermite-vs-quadrature covariance agreement, rectangle covariance evaluation,
  pairwise calibration under Hermite/quadrature/rectangle/refined strategies,
  pairwise infeasibility diagnostics, non-PD intermediate diagnostics,
  stochastic moments, and raw-data wrapping. `tests/checks/plsim/` provides an
  advisory calibration bench comparing speed and quadrature/rectangle agreement
  across strategies. Remaining PLSIM work is pair-cache/performance tuning and
  broader simulation-grid diagnostics.
- **S.** Extend VITA/covsim-style simulation from repaired matrix diagnostics
  plus 3-variable C-vine root/family selection to higher dimensions, richer
  structure/family search policies, and broader vine/multivariate-copula
  policies.
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
