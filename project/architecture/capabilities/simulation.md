### Simulation primitives

- A first C++ simulation namespace, `magmaan::sim`, provides NORTA data
  generation as an explicit primitive rather than a fitting shortcut.
  `calibrate_norta()` maps a target observed correlation matrix to a Gaussian
  copula correlation matrix by deterministic Gauss-Hermite quadrature and
  pairwise bisection, then validates the calibrated latent correlation by
  Cholesky factorization. `simulate_norta_matrix()` samples a complete
  `Eigen::MatrixXd`; `simulate_norta_raw()` wraps the same matrix in
  `data::RawData` for downstream sample-stat builders. Both draw functions also
  accept `NortaCalibration` directly, so repeated draws can reuse the
  deterministic latent-correlation calibration.
- The same marginal-transform surface is reused by independent generators:
  `simulate_independent_matrix()` and `simulate_independent_raw()` draw
  independent latent standard normals, transform each column through its
  marginal, and skip NORTA correlation calibration entirely.
- Foldnes-Olsson style independent-generator simulation is available through
  `calibrate_ig()`, `simulate_ig_matrix()`, and `simulate_ig_raw()`.
  `calibrate_ig()` chooses a square root `A` of the target covariance
  (`IgRootKind::Cholesky` by default, or `IgRootKind::Symmetric` via
  eigendecomposition), solves the linear `A^3` and `A^4` systems for generator
  skewness and excess kurtosis, then moment-matches each independent generator
  marginal. The lower-level overload accepts an already chosen root and fitted
  generator marginals for experiments that want to inspect or cache the
  calibration. The exploratory R package exposes the same mechanism through
  `magmaan_core$sim_ig_batch()` and the reusable
  `sim_ig_calibrate()` / `sim_ig_draw()` split. Pearson IG draws use direct
  Pearson random generators instead of inverse-CDF transforms: closed-form
  RNGs where available and exact theta-scale rejection for Type IV, with a
  per-marginal log-concave envelope. The R wrapper additionally batches reusable Type VI
  Pearson IG draws through R's gamma RNG before splitting the returned samples.
- Vale-Maurelli / Fleishman polynomial simulation is available through
  `fit_fleishman_coefficients()`, `calibrate_vale_maurelli()`,
  `simulate_vale_maurelli_matrix()`, and `simulate_vale_maurelli_raw()`.
  The first slice implements the classic third-order Fleishman transform
  `a + bZ + cZ^2 + dZ^3`: coefficients are solved from target skewness and
  excess kurtosis by exact polynomial moments, with the conventional monotone
  increasing root preferred when the moment equations have multiple solutions;
  pairwise intermediate normal
  correlations are solved from the Vale-Maurelli covariance cubic, and the
  assembled intermediate correlation matrix is Cholesky-validated before
  sampling. The R package exposes the two-stage split as
  `sim_vm_calibrate()` / `sim_vm_draw()` plus `sim_vm_batch()` (calibrate from a
  target correlation + per-margin skewness/excess kurtosis; draw consumes the
  calibration's Fleishman coefficients + intermediate correlation), validated by
  `r-package/examples/sim_vm.R`.
- PLSIM piecewise-linear simulation is available through
  `fit_plsim_marginal()`, `diagnose_plsim()`, `calibrate_plsim()`,
  `simulate_plsim_matrix()`, and `simulate_plsim_raw()`. The first slice uses
  regular normal-quantile breakpoints, fits continuous PL slopes to target
  marginal skewness and excess kurtosis, exposes Hermite coefficients, and
  calibrates pairwise intermediate normal correlations with selectable
  covariance evaluators:
  `PlsimCovarianceMethod::Hermite`, `Quadrature`, `Rectangle`,
  `HermiteThenQuadrature`, or `HermiteThenRectangle`. The rectangle path
  evaluates the Foldnes-Grønneberg segment decomposition by reducing bivariate
  normal rectangle probability/first/cross moments to conditional-normal
  one-dimensional adaptive integration. Gauss-Hermite quadrature remains useful
  as a smooth deterministic comparison path, but the advisory PLSIM bench shows
  it can differ from the rectangle/Hermite paths around kinked transforms.
  Hermite is the default calibration path; `diagnose_plsim()` reports fitted
  marginals, feasible pairwise target ranges, per-pair calibration failures,
  achieved correlations, and the minimum intermediate-correlation eigenvalue
  before the stricter `calibrate_plsim()` wrapper returns a success value.
  `PlsimCalibration` is draw-ready in C++, and the R package exposes both
  `sim_plsim_batch()` and the reusable
  `sim_plsim_calibrate()` / `sim_plsim_draw()` split.
- Model-implied simulation is available as a population-construction bridge:
  `sim::lower_model_implied()` lowers a lavaanified `LatentStructure` plus
  `MatrixRep` and `theta` through `ModelEvaluator::sigma()` into per-group
  `MixedPopulation` blocks, shared observed-variable names/kinds, and
  projection thresholds. Empty implied means become zero population means;
  explicit mean structures become the continuous population mean. Threshold
  rows are read from the partable, sorted per group/observed variable, and
  projected on the raw latent-response scale, so Delta/Theta ordinal
  parameterizations require no simulation-side standardization option.
  `sim::simulate_model_implied_group()` and `sim::simulate_model_implied()`
  dispatch that lowered population through the existing normal, Student-t,
  finite scale-mixture, contaminated-normal, and slash generators. The R
  package exposes the same split as
  `sim_model_calibrate()` / `sim_model_draw()` plus `sim_model_batch()`,
  accepting either a fitted magmaan object or a lavaan-shaped population
  partable with an explicit `theta` vector. Copula/NORTA/vine/IG/VM/PLSIM
  bridges remain separate because the SEM supplies moments and thresholds, not
  marginal distribution specifications.
- Elliptical generator diagnostics report the radial scale second/fourth
  moments, the covariance-normalization multiplier used by the draw path, the
  kurtosis inflation factor, and implied marginal excess kurtosis for
  Student-t, finite scale mixtures, contaminated normal, and slash generators.
  Infinite fourth moments are explicit for the covariance-finite but
  fourth-moment-infinite Student-t/slash cases (`df`/`q` in `(2, 4]`).
  `cpp/tests/unit/elliptical_test.cpp` pins the deterministic formulas and fixed-seed
  covariance smokes.
- Ordinal/mixed observed-correlation calibration inverts a target observed
  correlation matrix plus per-variable marginals (ordinal category proportions
  or continuous) into a latent Gaussian correlation matrix + thresholds.
  `sim::calibrate_ordinal_correlation()` solves each off-diagonal independently
  (NORTA-style) under one of three `ObservedCorrelationMetric` options:
  `Polychoric` (ordinal x ordinal latent rho == target, closed form),
  `PearsonCodes` (monotone bisection so the integer-code Pearson r matches the
  target, using `data::ordinal_bvn_rect_prob` as the forward map), and
  `Polyserial` (ordinal x continuous closed form `rho*sum phi(tau)/sd`;
  continuous x continuous is identity). The assembled latent matrix is repaired
  through the shared `repair_correlation_matrix_if_requested`
  (None/Error/Ridge/Shrinkage), and the calibration object records per-pair
  diagnostics plus the achieved-correlation matrix at the shipped latent.
  `sim::ordinal_correlation_population()` lowers it to a `MixedPopulation` that
  feeds the existing `simulate_mixed_population_*` generators, while
  `sim::calibrate_ordinal_correlation_multigroup()` composes one such
  calibration per group with shared variable shape, group labels, and
  group-keyed achieved category proportions; its draw path threads one RNG
  sequentially across per-group `MixedPopulation` blocks just like
  model-implied simulation. `simulate_ordinal_correlation_normal()` is the
  single-group one-shot convenience. The R
  package exposes the two-stage split as `sim_ordcorr_calibrate()` /
  `sim_ordcorr_draw()` plus `sim_ordcorr_batch()`, and the multi-group split as
  `sim_ordcorr_mg_calibrate()` / `sim_ordcorr_mg_draw()` plus
  `sim_ordcorr_mg_batch()`. `MixedProjectionResult` now also carries achieved
  `category_proportions`; `sim::raw_data_from_mixed_projection()` wraps a
  projected block as a `data::RawData` with optional variable names and ordinal
  level labels, and `sim::raw_data_from_mixed_projections()` composes
  per-group projected blocks into multi-block `RawData` with populated
  `group_labels`. For workflows that already have estimated categorical
  summaries, `sim::calibrate_ordinal_correlation_summary()` and
  `sim::calibrate_ordinal_correlation_summary_multigroup()` accept observed
  kinds, latent-response thresholds, and a pre-estimated latent Gaussian
  correlation matrix directly (polychoric/polyserial summary `R`); they skip
  pairwise inversion, apply the same matrix-repair policies, and record
  `max_abs_error` as the largest off-diagonal repair delta. The R mirror is
  `sim_ordcorr_summary_calibrate()` and
  `sim_ordcorr_mg_summary_calibrate()`, whose outputs feed the existing
  `sim_ordcorr*_draw()` functions.
- Calibrated simulation generators are standardized around a two-stage
  contract: deterministic `calibrate_*()` calls return reusable state objects
  with fitted marginals, latent/intermediate matrices, achieved diagnostics, and
  iteration metadata; stochastic `simulate_*()`/draw calls accept that state
  plus only `n`, RNG/seed, and draw-time options. One-shot batch helpers remain
  as convenience wrappers. This path is now in place for IG, NORTA, PLSIM,
  pairwise Archimedean copulas, generic fixed-order C-vines, specialized
  three-variable C-vine root/family-selection policies, and single-/multi-group
  ordinal/mixed observed-correlation calibration at the R boundary and for IG,
  PLSIM, NORTA, copula/vine, and ordinal-correlation generators at the C++
  boundary.
  Long-running experiments persist those calibration objects explicitly through
  the `experiments/_support` cache helpers, which key ignored `results/cache/`
  entries by population, generator/options, magmaan package version, and git
  reference. Core and the R package do not perform hidden global disk caching.
- Initial NORTA marginals are standard normal, standardized lognormal, Tukey
  g-and-h, Pearson-system distributions, and Johnson-system SU/SB
  distributions. Fleishman polynomial transforms can also be passed through the
  same marginal-transform machinery, but they are generator transforms rather
  than guaranteed quantile functions. The Tukey path is a pragmatic
  skew/tail stress family with numerically standardized moments and
  `0 <= h < 0.25` so fourth moments are finite. Pearson marginals use the
  PearsonDS parameter convention for Types 0, I, II, III, IV, V, VI, and VII;
  Type IV quantiles are computed by finite-interval integration of the
  normalized theta-space density and safeguarded bisection, avoiding a GSL or
  PearsonDS runtime dependency. Johnson marginals use monotone transforms of a
  standard normal variate, fit SU/SB shape parameters to target skewness and
  excess kurtosis by deterministic quadrature, and plug into the same
  NORTA/independent-generator transform surface.
- Fixed-parameter t-copula simulation is available through `TCopulaSpec`,
  `simulate_t_copula_matrix()`, and `simulate_t_copula_raw()`. It draws a
  multivariate Student-t copula from a supplied correlation matrix and degrees
  of freedom, maps the resulting uniforms through the existing marginal
  quantile machinery, and rejects non-quantile generator transforms such as
  Fleishman. This is deliberately still a fixed-parameter generator; VITA/covsim
  calibration from requested observed moments/correlations to generator
  parameters lives above these copula draws. `vinecopulib`/`rvinecopulib` are
  treated as oracle/reference implementations for copula conventions, not as
  core runtime dependencies. `simulate_mixed_population_t_copula()` composes the
  same generator with the observed projection layer.
- Fixed-parameter bivariate Archimedean copula simulation is available through
  `BivariateCopulaSpec`, `simulate_bivariate_copula_matrix()`, and
  `simulate_bivariate_copula_raw()`. The first local families are independence,
  Clayton, Gumbel, Frank, and Joe. Sampling uses conditional inversion of
  `dC(u,v)/du`, then feeds the resulting uniforms through the same marginal
  quantile machinery as the t-copula path. The conditional CDF and inverse
  conditional CDF helpers are fixture-checked against rvinecopulib `hbicop()`.
  `bivariate_copula_tau()` and `bivariate_copula_from_tau()` provide the
  Kendall-tau parameter scale used for rank-based copula setup.
  `simulate_mixed_population_bivariate_copula()` composes the same generator
  with the observed projection layer. The first pairwise VITA/covsim-style
  calibration helper is also present: `bivariate_copula_observed_corr()`
  evaluates the quadrature-implied observed Pearson correlation after marginal
  transforms, and `calibrate_bivariate_copula_correlation()` bisects on Kendall
  tau for one bivariate family. `calibrate_bivariate_copula_correlation_matrix()`
  extends this to every off-diagonal entry of a target correlation matrix and
  returns pairwise parameters, achieved correlations, feasible bounds, and
  iteration counts. It also reports maximum absolute error and achieved-matrix
  eigenvalue diagnostics, with opt-in error/ridge/shrinkage repair toward a
  requested minimum eigenvalue. This is a matrix diagnostic/calibration layer
  rather than an automatic matrix-to-vine fitter. The first explicit joint vine
  sampler is available through `CVine3CopulaSpec` and
  `cvine3_copula_inverse_rosenblatt()` / `simulate_cvine3_copula_*()`: a fixed
  three-variable C-vine with variable 0 as root, bivariate pair copulas for
  `0-1` and `0-2`, and a conditional pair copula for `1-2|0`. The inverse
  Rosenblatt helper is fixture-checked against rvinecopulib
  `inverse_rosenblatt()` using the equivalent `cvine_structure(c(3,2,1))`.
  `cvine3_copula_observed_corr()` evaluates its implied observed correlation
  matrix by deterministic quadrature, and
  `calibrate_cvine3_copula_correlation()` fits the two root pair copulas plus
  the conditional copula to a 3x3 observed-correlation target.
  `calibrate_cvine3_copula_correlation_select_root()` tries all three possible
  roots by permutation and returns the best achieved matrix in the original
  variable order. `CVine3FamilySpec` allows different families on `0-1`, `0-2`,
  and `1-2|0`, and `calibrate_cvine3_copula_correlation_select_families()`
  searches a caller-provided family set for the fixed root-0 C-vine.
  `calibrate_cvine3_copula_correlation_select_structure()` combines the
  three-root search with per-edge family search for the three-variable case.
  The `simulate_cvine3_copula_*()` overloads that take a
  `CVine3CorrelationCalibration` simulate in the selected vine order and return
  columns in the caller's original variable order.
  `simulate_mixed_population_cvine3_copula()` composes explicit or calibrated
  three-variable C-vine draws with the observed projection layer.
  Unit validation now covers the full 3-variable path from target observed
  correlation through structure/family calibration to deterministic-seed
  simulated correlations. Generic fixed-order C-vine sampling is available
  through `CVineCopulaSpec`, `cvine_copula_inverse_rosenblatt()`, and
  `simulate_cvine_copula_*()` for arbitrary dimension, with inverse
  Rosenblatt validation against both the 3-variable specialization and a
  four-variable rvinecopulib fixture. `simulate_mixed_population_cvine_copula()`
  composes generic fixed-order C-vine draws with the observed projection layer.
  `cvine_copula_observed_corr()` and `calibrate_cvine_copula_correlation()`
  provide the first higher-dimensional fixed-order VITA slice: deterministic
  observed-correlation evaluation and sequential Kendall-tau calibration for
  one caller-supplied family/order. Broader structure/family policies and
  ordinal/polyserial/polychoric calibration are still future work.
- Scalar special-function helpers needed by Pearson quantiles and FMG F tails
  (regularized beta/gamma and their inverses, Student-t CDF/quantile, F upper
  tail) are centralized in the private `cpp/src/detail_distribution_math.hpp` header,
  and that is the settled long-term policy: hand-rolled local kernels with
  dedicated goldens vs base R (`cpp/tests/tools/regen_distribution_math_fixtures.R` ->
  `cpp/tests/fixtures/distribution_math.json`,
  `cpp/tests/unit/distribution_math_test.cpp`), no Boost.Math or other third-party
  special-functions dependency (Boost.Math throws under `-fno-exceptions`). Every
  new kernel ships a golden; `cpp/src/inference/inference.cpp` reuses the shared
  header rather than keeping its own copy of the incomplete-gamma helpers.
- Moment-matching now has an explicit simulation API:
  `fit_marginal_to_moments()` takes a `MomentMatchSpec` with target skewness
  and excess kurtosis, returning a fitted `MarginalSpec` plus achieved moment
  diagnostics. `MomentMatchFamily::TukeyGH` is implemented with a reusable
  two-moment finite-difference solve over Tukey `g,h`.
  `MomentMatchFamily::Pearson` implements the PearsonDS `pearsonFitM()`
  moment-classification formulas, converting magmaan's target excess kurtosis
  to PearsonDS raw kurtosis internally. Checked fixtures generated by
  `cpp/tests/tools/regen_pearson_sim_fixtures.R` compare fitted parameters and
  quantiles, including Pearson Type IV, to PearsonDS 1.3.2.
  `MomentMatchFamily::Johnson` fits Johnson SU and SB marginals numerically
  against the same two shape moments; SL (log-normal) is intentionally excluded
  from moment-matching (its skew and kurtosis lie on a 1-parameter curve, so it
  cannot 2-moment-fit) and remains reachable only via the direct
  `MarginalSpec::johnson(1, ...)` constructor. Checked fixtures generated by
  `cpp/tests/tools/regen_johnson_sim_fixtures.R` compare the fitted shape pair
  (gamma, delta), the SU/SB type, and quantiles to SuppDists 1.1.9.9
  `JohnsonFit`/`qJohnson`; because SuppDists's own moment fit is only
  loosely accurate, the fixture targets the realized moments of its returned
  shape (computed by independent high-accuracy quadrature) so both
  implementations describe the identical distribution and agree to ~1e-7.
  `MomentMatchFamily::Fleishman` reuses the Vale-Maurelli coefficient solver
  and exposes the resulting cubic as a regular transform marginal, while
  `marginal_quantile()` rejects it because the cubic need not be monotone.
