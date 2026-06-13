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

The implemented `magmaan::sim` surface (NORTA, independent / Foldnes-Olsson IG,
Vale-Maurelli/Fleishman, PLSIM, fixed-parameter t-copula, bivariate Archimedean
copulas, 3-variable and generic fixed-order C-vines, elliptical / scale-mixture
generators, model-implied population lowering, the Tukey / Pearson / Johnson /
Fleishman marginal families, population composition, and the shared
observed-variable projection layer) is inventoried in the roadmap's "Simulation
primitives" section. The R surface
exposes `sim_*_batch()` convenience calls plus the reusable
`sim_ig_calibrate()` / `sim_ig_draw()`,
`sim_norta_calibrate()` / `sim_norta_draw()`,
`sim_plsim_calibrate()` / `sim_plsim_draw()`,
`sim_bicop_calibrate()` / `sim_bicop_draw()`, and
`sim_cvine_calibrate()` / `sim_cvine_draw()` and
`sim_cvine3_calibrate()` / `sim_cvine3_draw()`, and
`sim_model_calibrate()` / `sim_model_draw()` two-stage handles. Keep API-level
inventory in the roadmap; this file carries the work queue and decision log
below.

## Architecture Direction

The simulation sublibrary is intentionally maximalist: if a data-generation
mechanism is used in the SEM/simulation literature, or has an obvious extension
needed to compare those mechanisms fairly, it belongs here eventually. Keep that
coverage organized as a layered simulation stack rather than as unrelated
top-level named generators.

The preferred decomposition is:

1. **Population construction.** Build explicit population means/covariances,
   group-specific population moments, thresholds, and model-implied population
   moments from a lavaanified model or fitted estimates.
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

Calibrated generators must follow a two-stage contract at both C++ and R
boundaries:

1. `calibrate_*()` returns a concrete calibration/state object containing every
   deterministic object needed for future draws: fitted marginals, latent or
   intermediate correlation/root matrices, chosen copula parameters, repair
   diagnostics, achieved moments/correlations, and iteration counts.
2. `simulate_*()` / `draw_*()` accepts that calibration object and only consumes
   `n`, RNG/seed, and draw-time options. Repeated draws for different sample
   sizes or replicate batches must not redo deterministic calibration.
3. Convenience one-shot APIs may remain, but must be thin wrappers that call the
   explicit two-stage path and should be named/documented as convenience, not as
   the primary simulation design.
4. R wrappers should mirror the C++ split with opaque-but-inspectable S3/list
   calibration objects. For large simulation grids, users should be able to
   calibrate once per `(population, generator, distribution target, options)`
   cell and reuse that object for all `N` and replicate batches.

This shape keeps named literature methods narrow: each one either constructs a
population, generates continuous responses, projects observed variables, or
calibrates/diagnoses one of those steps.

## Planned Surface

- Keep the landed model-implied simulation bridge moment-based: it lowers
  fitted or hand-authored SEM state to population moments and projection specs,
  then calls the normal/elliptical mixed-population stack. Do not route it into
  copula/NORTA/vine/IG/VM/PLSIM generators until a separate marginal
  calibration layer can supply distribution targets; the model gives moments
  and thresholds, not marginal families.
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
- The generator API is normalized around reusable calibration objects: every
  calibrated path now exposes `sim_*_calibrate()` / `sim_*_draw()` R wrappers,
  with `sim_*_batch()` kept as thin convenience wrappers. Remaining: extend the
  first PLSIM slice with broader simulation-grid diagnostics and possible
  tuning/replacement of the current adaptive rectangle integration kernel, and
  extend the first pairwise VITA/covsim copula calibration into a full
  matrix-oriented workflow.

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

## IG Generator Family: Draw Cost and Moment-Matching (2026-06-01)

Findings from profiling the independent-generator (`sim_ig_*`) path while
scaling experiment 17 (Foldnes-Moss-Gronneberg 2024) to the full grid.

**Cost shape.** The ML fit is trivial and N-independent (~2-9 ms at any p/N: the
sample covariance is formed once, then optimization is on the p x p matrix). The
expensive parts are (a) the FMG eigenvalue battery, which scales with p/df and,
at p=40, with N; and (b) IG *data generation*. At p=40, N=3000 a single IG
dataset draw costs ~2.0-2.2 s/rep, dwarfing everything else. So large-model runs
are p-bound, and the IG generator is the single biggest line item at the corner.

**Why IG draws are slow: generator family.** `make_cell_sampler` requests
`generator_family = "pearson"` (chosen to mirror the lavaan/R IG reference). The
Pearson draw applies a per-element inverse CDF (`normal_cdf` then an inverse
beta/gamma/Student-t, or the iterative Type IV quantile) -- O(N*p) numeric
inversions per dataset. The Tukey g-and-h and Johnson SU families instead have
*closed-form* draw transforms (`((exp(gZ)-1)/g) exp(hZ^2/2)` and
`sinh((Z-gamma)/delta)`), so their draws are ~100x cheaper. The right default for
speed is a closed-form family.

**Why we cannot just switch families (today):**
- **Tukey g-and-h cannot represent the targets.** For skew 2-3 the `g` that sets
  the skew already overshoots the kurtosis, and the finite-fourth-moment cap
  `h < 1/4` leaves no room. Calibration fails even at a 10% moment tolerance.
  This is a representability limit, not a solver bug.
- **Johnson SU can represent them, but the moment-match solver is brittle.** It
  fits both shape params by a finite-difference Gauss-Newton whose candidate
  moments are computed by hand-rolled **Gauss-Hermite quadrature**
  (`raw_moment_summary`). For severe kurtosis the `z^4 sinh^4` integrand has
  heavy tails that 81-point GH integrates poorly, so the residual floors above
  `objective_tol` and the solver *throws* on what is otherwise a fine fit (and on
  non-convergence it throws rather than returning the best fit found). Symptoms:
  fails at the trivial (skew 2, kurt 7) under the default `max_iter=80`;
  convergence flips erratically with `max_iter`/`parameter_tol`; stalls ~10%
  short at (skew 3, kurt 21).

**Root cause and fix.** The fragility is the hand-rolled quadrature used to
*evaluate* candidate moments, not the optimizer. For Johnson **SU** the
standardized skewness and excess kurtosis are closed-form in `(gamma, delta)`
(raw moments of `W = sinh((Z-gamma)/delta)`: with `w = exp(1/delta^2)`,
`Om = gamma/delta`, `E[W]=-sqrt(w) sinh(Om)`, `E[W^2]=(w^2 cosh(2Om)-1)/2`,
`E[W^3]=-sqrt(w)(w^4 sinh(3Om)-3 sinh(Om))/4`,
`E[W^4]=(w^8 cosh(4Om)-4 w^2 cosh(2Om)+3)/8`, then central moments; verified
against Monte Carlo). Computing those analytically removes the quadrature error,
so the SU solve now converges to machine precision (resid ~1e-12 in ~10 iters)
where it previously stalled. Implemented in the SU branch of
`eval_johnson_moment_fit` with an SU regression test; SB (type 3) keeps the
quadrature path.

**This does NOT yet make Johnson a usable IG generator.** The IG construction
does not fit the marginal to the *target* moments: `solve_ig_moment_system`
solves a linear system for inflated *generator* moments (the linear mixing
`X = root . Y` averages independent generators and so reduces kurtosis; the
generators must overshoot to compensate, and the inflation grows with p). At
p=40, target `(skew 3, exkurt 21)` requires generator marginals ranging up to
`(skew 5.6, exkurt 53)`, and the generator-moment vector straddles the SU/SB
boundary -- a mix of SU points (now fixed) and SB points. Several SB-region
generator targets (e.g. `(3.5, 26)`, `(2.14, 7.7)`, `(3.72, 17.6)`) still fail
the quadrature SB solve, and any single failed marginal aborts the whole IG
calibrate. So **experiment 17 stays on Pearson** for now; Johnson IG needs a
robust SB moment-fit too (no elementary closed form -- candidates: AS99
Hill-Holder SB branch, or higher/adaptive quadrature that returns its best fit
instead of throwing). The orthogonal lever for IG draw speed is to spline the
Pearson inverse-CDF once per marginal at calibration (keeps lavaan-matching
fidelity, family-agnostic) rather than inverting per element at draw time.

**General lesson (carry forward).** Hand-rolled numerical integration in the sim
calibration (`marginal_moments`, the bivariate-copula and PLSIM quadratures) is a
recurring fragility source: it silently caps the achievable accuracy and turns
representable targets into "did not converge". Prefer closed-form moments wherever
the family admits them; where quadrature is unavoidable, the convergence test
must respect the quadrature floor and the solver should return its best fit
rather than throw. The draw-time standardization (`marginal_moments`) is benign
for skew/kurt (scale/location invariant) but is the same pattern.

## Remaining Work

Open work only; landed generator slices are inventoried in the roadmap.

- **M.** Extend the ordinal/mixed projection layer with group-specific
  thresholds, variable names/level labels for raw-data wrapping, and richer
  achieved-proportion diagnostics.
- **M.** Extend population composition with group-specific population blocks
  and raw-data naming/level metadata once the `RawData` carrier grows those
  fields.
- **M.** Add elliptical diagnostics/goldens for Student-t, contaminated normal,
  slash, and finite scale mixtures: deterministic moment formulas where
  available plus stochastic smokes.
- **M.** Add ordinal/mixed observed-correlation calibration once direct
  threshold projection is stable: pairwise thresholded correlation maps,
  calibration to target observed Pearson/polychoric/polyserial summaries, and a
  policy for non-positive-definite calibrated latent matrices.
- **S.** Add pseudo-elliptical / transformed-elliptical mechanisms after the
  first elliptical slice clarifies the shared radial/core interfaces.
- **S/M.** Remaining PLSIM work: lower-level pair-cache / performance tuning and
  broader simulation-grid diagnostics (the first PLSIM slice plus its R
  calibrate/draw wrappers have landed; see roadmap).
- **S.** Extend VITA/covsim-style simulation from repaired matrix diagnostics
  plus 3-variable C-vine root/family selection and generic fixed-order
  calibration to richer higher-dimensional structure/family search policies and
  broader vine/multivariate-copula policies.
- **S.** Decide whether Johnson SL should be exposed beyond the direct
  `MarginalSpec::johnson()` constructor.
- **S.** Decide the long-term special-functions policy before expanding the
  Pearson/Johnson surface. `src/detail_distribution_math.hpp` currently holds
  hand-rolled regularized beta/gamma, inverse beta/gamma, Student-t, and F-tail
  helpers shared by Pearson simulation and FMG p-values. Either keep these with
  dedicated goldens, vendor/use Boost.Math, or choose another vetted dependency.
- **S/M.** A robust Johnson SB moment-fit (no elementary closed form) so the
  Johnson family becomes a usable closed-form IG generator: candidates are the
  AS99 Hill-Holder SB branch or higher/adaptive quadrature that returns its best
  fit instead of throwing. Orthogonal IG draw-speed lever: spline the Pearson
  inverse-CDF once per marginal at calibration. (See the 2026-06-01 IG note.)
- **M.** Harden NORTA calibration for larger simulation grids: cache pairwise
  correlation maps when marginal specs repeat, expose/interpolate the
  `rho_Z -> Corr(X_i, X_j)` map for repeated target matrices, and add an
  explicit policy for pairwise-calibrated latent matrices that are not positive
  definite. Error-only is the current behavior.
