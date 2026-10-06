### Ordinal and mixed categorical LS

TASK-83 accelerates complete-data exact first-stage sampling influence with a
block-sparse central FD Jacobian of empirical mean scores. Each perturbation
visits only its marginal and incident pairs, or its single association pair;
ordinal means reduce over category counts and share bivariate-normal corner
CDF/PDF grids. Mixed polyserial and continuous-pair blocks retain FD. No
per-case Jacobians, perturbed bread matrices or perturbed Gamma matrices are
built. The dense implementation remains callable through private test probes.
Binary/five-category, skewed-threshold and one/two-group ordinal/mixed gates
compare Jacobians, rows and Gamma at 1e-7 relative tolerance. Existing
policy/lab delta/theta tests are unchanged.

Five-call medians with BLAS/OpenMP pinned to one thread, fresh policy caches,
12 indicators, two equally sized groups and five ordinal categories are:

| Data | Total N | Dense FD policy (s) | Sparse FD policy (s) | OPG IJ call (s) |
| --- | ---: | ---: | ---: | ---: |
| All ordinal | 1000 | 1.909 | 0.138 | 0.100 |
| All ordinal | 4000 | 3.933 | 0.195 | 0.120 |
| Six continuous + six ordinal | 1000 | 3.435 | 2.052 | — |
| Six continuous + six ordinal | 4000 | 12.603 | 7.758 | — |

The benchmark uses R seed 83, one Gaussian factor with loadings 0.7, thresholds
(-1.2, -0.3, 0.4, 1.1), and excludes data preparation/fitting from timing.
Before/after calls use newly fitted live handles; deserialized handles add
reconstruction overhead. OPG is the existing lab IJ covariance comparator.
A separate component profile (C++ seed 83, two 500-row blocks, five-call
medians) isolates the mean-score Jacobian at 3.541 s dense versus 0.021 s
sparse, confirming the targeted bottleneck.
Fresh ordinal policy calls cost 1.38x/1.63x that comparator; policy outputs
match their dense-FD snapshots within 1e-7. Mixed timing includes unchanged
estimated-weight/Gamma work beyond the exact sampling Jacobian.

- TASK-74 adds complete-data all-ordinal exact sampling rows and Gamma via
  `data::ordinal_moment_sampling_influence()`, reusing the mixed empirical
  score-Jacobian assembly. `robust_ordinal_ij()` accepts `OrdinalFirstStage`
  (OPG default, Exact comparator); the lab `first_stage = "exact"` also returns
  `sampling_moment_influence` and `sampling_gamma`. Both the sampling term and
  fitted-OPG-weight movement use the selected rows. Missing-data exact sampling
  is explicitly unavailable. TASK-69 adopts Exact for the ordinary all-ordinal policy, caching sampling
  rows once per `DwlsPolicyFit`; fitting NACOV, lavaan compatibility and lab
  defaults retain OPG. Global Gamma is the exact-row cross-product; nested
  parameter and moment-tangent laws use the exact estimated-weight IJ meat.
  Global All remains; nested All was adopted in TASK-81 (2026-10-06), with
  SB/PEBA4 comparators. Exact-first-stage
  calibration remains pending: TASK-79 registers fresh-draw exact/OPG paired
  reconfirmation and supplies a full-grid mechanics/pricing pilot; production
  awaits merger registration and compute approval. TASK-84 supplies a registered
  latent non-normality lane with streamed million-row pseudo-targets, independent
  Monte Carlo target checks, descriptive globals and paired exact/OPG coverage
  and identical-group nested tests. Local smoke/pricing evidence is retained;
  production calibration remains pending. The shared implementation differentiates the
  empirical score means with relative step 1e-5, then solves the centered
  equations (threshold Hessian and pair-score threshold/rho derivatives).
  Replicated case-weight row errors are below 8e-9 in the registered samples.
  Over eight Gaussian seeds, RMS IJ covariance gaps at N = 250/1000/4000
  are 0.0900/0.0483/0.0269: a 3.34-fold reduction over sixteenfold N, consistent
  with the O_p(N^-1/2) information-identity discrepancy. This numerical
  convergence gate is not a coverage calibration.
  TASK-69 policy gates compare Exact IJ covariance, exact-row global Gamma,
  parameter-space nested and moment-nested tangent spectra (1e-10), with one/two
  groups, delta/theta, binary and five-category items. Binary pairwise saturation
  agrees with OPG Gamma within 1e-6 at N = 4000: its three first-stage coordinates
  saturate each pair, so remaining discrepancy is numerical, rather than the
  unsaturated normal-pair O_p(N^-1/2) information-equality discrepancy.
  Before TASK-83, adopting Exact for a two-group 12-indicator five-category
  fit (N = 1000 total, one thread) cost median 80 ms before versus 1643 ms
  after on five fresh R policy calls;
  cached calls have median 1 ms. Sampling rows are built once per policy fit.
  The theta jackknife gate uses N = 2400 with the unchanged 4% tolerance
  (1.32% discrepancy); delta designs remain at N = 600. Exact covariance
  transport from delta to theta agrees within 1e-6 at both sizes; tighter
  refits leave the N = 600 discrepancy unchanged. The finite-sample theta
  reparameterization gives 5.23% at N = 600, decreasing approximately as 1/n.
  The Satorra comparator uses exact Gamma with unchanged OPG fitting weights;
  its monotone and terminal 2% gates pass. The OPG comparison is diagnostic only.

- C++ `fit_ordinal_configured()` supports the `lavaan-0.7.2` preset for
  all-ordinal DWLS, with delta/theta and ordered affine equality coordinates.
  Sample thresholds, polychoric FABIN3 loadings, unit response scales and
  theta residual starts feed the pinned PORT controls and four-attempt retry
  sequence. Search weights use `(n_g-1)/n_g` relative to native weights,
  reproducing lavaan's group-weighted half quadratic and acceptance gradient.
  Every attempt and the selected first-order verdict are retained beside
  native objective/stationarity diagnostics. Frozen single/grouped,
  loading/threshold invariance and invalid-start fixtures cover this contract.
  Both R packages route all-ordinal DWLS fitting options through this engine;
  live lavaan tests cover delta/theta, grouped loading/threshold invariance,
  starts, coordinates, acceptance gradients and invalid-start retries.
  Native fitting is unchanged.
  Complete mixed DWLS also uses this configured engine (TASK-91), with sample
  thresholds/continuous means, half continuous-variance residual starts,
  FABIN3 loadings and sample-response retry scales. Eight frozen HS cases and
  installed-lavaan retained WLSMV gates cover delta/theta, one/two unequal
  groups, a zero factor covariance and equal loadings. Fresh `fit_model()`,
  prepared `estimate()` and ordinary `magmaan()` agree. Missing observations,
  nonlinear constraints and finite bounds return explicit fit errors for the
  mixed preset. Native mixed fitting remains unchanged; ULS/WLS versioned
  presets remain unavailable.
- Explicit ordinal rows retain user provenance through lab DWLS/ULS/WLS,
  PSD, fixed-stage-2 and native prepared handles. DELTA response-scale `~*~`
  rows are live coordinates: free, labelled/equal, linearly constrained or
  fixed at a non-unit value. Their estimates and SEs retain lavaan's scale
  coordinates. Residual variances are derived as
  `theta_ii = delta_i^(-2) - (Lambda Mid Lambda')_ii`, with `free = 0`.
  Thresholds use `delta_i * (tau_i - mu_i)` and associations use
  `delta_i * delta_j * Sigma*_ij`; Jacobian and observed curvature include
  their product-rule derivatives. THETA keeps its residual coordinates.
  Seven frozen lavaan WLSMV/DWLS cases cover fixed/released/equal/constrained
  scales, threshold/loading invariance, P-IV2 SCALAR and longitudinal equality;
  live R gates include scale SEs and retained-estimate WLSMV reporting.
  Threshold augmentation preserves an existing source scale row, and automatic
  group release respects explicit fixed scales. Factor scores, std.all and
  fitted-correlation reliability use the corresponding response variance.
- Threshold (`|`) and response-scale (`~*~`) parser/partable projection.
- Integer all-ordinal complete/listwise sample statistics.
- Pairwise polychoric correlations.
- Public enum-backed polychoric h-score API under `data::eval_polychoric_h_score()`
  for ML, WMA hard cap, smooth cap, and exponential cap experiments, returning
  score values, derivatives, and objective contributions while leaving the
  default lavaan-compatible ordinal sample-stat path unchanged.
- Experimental fixed-threshold all-ordinal bivariate h-weighted rho fitting
  under `data::fit_ordinal_pair_rho_h_weighted()`. It keeps thresholds fixed,
  accepts the predefined h-score options, returns rho/objective/score,
  convergence and bound diagnostics, adjusted counts, fitted probabilities,
  expected/residual/Pearson tables, and per-cell robust weights. ML and
  `WmaHardCap(k = Inf)` delegate to the existing ML rho path, preserving the
  lavaan-compatible limit; finite caps remain diagnostics/prototype machinery,
  not the default ordinal moment builder.
- Experimental pair-local all-ordinal bivariate joint h-weighted fitting under
  `data::fit_ordinal_pair_joint_h_weighted()`. It estimates both pair-local
  thresholds and rho by minimizing the h-score/minimum-disparity objective,
  reusing ordered-threshold and bounded-rho transforms. It returns pair-local
  thresholds, rho, objective/gradient diagnostics, adjusted counts, fitted
  probabilities, expected/residual/Pearson tables, and per-cell robust weights.
  ML and `WmaHardCap(k = Inf)` delegate to the existing joint ML kernel; finite
  caps are validated as experimental bivariate diagnostics, not SEM moment
  construction or calibrated robust inference.
- Experimental pair-local all-ordinal h-weighted influence diagnostics under
  `data::ordinal_pair_h_weighted_influence()`. Given integer bivariate counts,
  pair-local thresholds, and rho, it expands casewise estimating-function rows,
  returns h-score ratios/values/derivatives/diagnostic weights, analytic
  probability-Hessian bread, centered `h'(t)`-weighted meat rows, influence
  rows, score Gamma, and sandwich Gamma with `S'S / n` scaling. The WMA
  hard-cap derivative convention is pinned as `dh(t) = 1[t < k]`, so the exact
  kink uses derivative zero. This remains the robust bivariate Gamma primitive
  used for pair-level validation.
- Experimental pair-local all-ordinal density power divergence fitting under
  `data::fit_ordinal_pair_joint_dpd()`. It estimates pair-local thresholds and
  rho with DPD tuning `alpha`, delegates `alpha = 0` to joint ML, and returns
  probabilities, expected/residual/Pearson tables, and `p^alpha` attenuation
  weights. This is the main non-h-score bivariate comparator and is not yet
  wired into SEM moment/Gamma construction.
- Experimental all-ordinal SEM integration under
  `data::pairwise_ordinal_stats_h_weighted_from_integer_data()` now uses a
  shared-threshold composite h-score estimator: one threshold block per ordinal
  variable plus one polychoric correlation per pair are optimized jointly under
  the h-weighted objective. It rebuilds casewise moment influence/Gamma from
  the shared robust estimating equations and refreshes DWLS/WLS weights.
  Optional robust-R handling records the raw minimum eigenvalue and can either
  fail explicitly or repair low-eigen correlation matrices by ridge/shrinkage
  toward the identity, with the same transformation applied to the correlation
  influence columns before Gamma and weights are rebuilt.
- Experimental all-ordinal shared-threshold DPD stats are available under
  `data::pairwise_ordinal_stats_dpd_from_integer_data()`. The parameter and
  output contract matches the h-weighted SEM path, but the composite objective
  is density power divergence with tuning `alpha`; `alpha = 0` delegates to
  the default ML/lavaan-compatible stats path.
- Experimental all-ordinal shared-threshold Huberized Pearson-residual stats
  are available under
  `data::pairwise_ordinal_stats_huber_residual_from_integer_data()`. The
  clipped residual is the cell Pearson residual; hard Huber, pseudo-Huber,
  Tukey biweight, and no-clip options share the same moment/Gamma contract as
  h-weighted and DPD stats, with no-clip delegating to the default
  ML/lavaan-compatible stats path. Robust-R ridge/shrinkage repair applies the
  same correlation-column influence transformation before Gamma and weights
  are rebuilt.
- Experimental mixed continuous/ordinal Huberized Pearson-residual stats are
  available under `data::mixed_ordinal_stats_huber_residual_from_data()`.
  Hard Huber, pseudo-Huber, Tukey biweight, and no-clip residual options are
  exposed for ordinal-containing threshold/correlation/polyserial rows while
  continuous-only means, variances, and covariances remain ordinary moments.
  The path rebuilds casewise moment influence, Gamma/NACOV, and DWLS/WLS
  weights, including the single-ordinal case where threshold uncertainty enters
  only through polyserial links. Unit-level simulation checks cover the no-clip
  ML limit, contaminated polyserial tails, sparse ordinal margins, positive
  DWLS diagonals, finite Gamma conditioning, and DPD comparator stability. This
  remains a methods comparator rather than a lavaan-backed compatibility claim.
- Robust ordinal moment builders are experimental methods-developer surfaces,
  not changes to the default lavaan-compatible ordinal builders. SEM-facing
  robust builders use shared ordinal thresholds and rebuild moment influence,
  Gamma/NACOV, and DWLS/WLS weights under the robust equations. Pair-local
  threshold estimators remain diagnostics/prototypes rather than SEM moment
  constructors. Mixed robust builders currently robustify ordinal-only and
  continuous-ordinal links only; continuous marginal moments and
  continuous-continuous covariances remain ordinary unless a separate design
  reopens them. R exposes predefined robust methods only; arbitrary C++
  h-functions remain internal.
- Public all-ordinal pairwise ML kernel and `PairwiseOrdinalStats` wrapper for
  complete/listwise ordinal data, exposing pair labels, count/adjustment
  diagnostics, fitted fixed-threshold rho diagnostics, fitted expected/residual
  tables, missingness/repair diagnostics, casewise moment influence, Gamma,
  and minimum eigenvalue diagnostics while preserving the existing
  `OrdinalStats` moment/weight path.
- Bivariate ordinal observed-pair table kernel for explicit pairwise
  observed-data composite-likelihood work. `NaN` skips the pair and increments
  missing-pair diagnostics; finite observed values must be positive integer
  categories inside the declared level ranges. Observed-pair wrappers feed
  those counts into both fixed-threshold rho ML and pair-local joint
  threshold/rho ML while preserving `n_obs` and `n_missing`.
- Experimental joint bivariate ordinal ML kernel for one complete pairwise
  table, estimating pair-local nuisance thresholds and rho. This kernel backs
  the pair-local complete/listwise and observed-pair composite objective
  prototypes, but it is not wired into the lavaan-compatible SEM moment
  construction path.
- Public SEM-facing all-ordinal pairwise composite objective API under
  `estimate::pairwise_ordinal_composite_objective()`. It consumes
  `PairwiseOrdinalStats` plus SEM-implied shared thresholds and correlations,
  maps each pair to its bivariate threshold margins, exposes per-pair boundary
  diagnostics plus score/Gamma rows for the bivariate threshold/rho margin,
  and makes scaling/weighting explicit. The current reporting contract
  deliberately does not report chi-square or degrees of freedom until a
  calibrated composite-likelihood test is implemented.
- Complete/listwise all-ordinal pair-local joint composite prototype under
  `estimate::pairwise_ordinal_joint_composite_objective()`. It consumes the
  same `PairwiseOrdinalStats` diagnostics and options surface, refits every
  pair with the complete bivariate joint threshold/rho ML kernel, and returns
  pair-local thresholds, rho, adjusted counts, fitted counts, residuals,
  score contributions, score Gamma, boundary flags, and objective scaling.
  This is a saturated/reference composite target for future SEM fitting, not a
  lavaan-compatible DWLS moment builder and not a calibrated global chi-square
  test.
- Observed-pair all-ordinal joint composite prototype under
  `estimate::pairwise_ordinal_observed_joint_composite_objective()`. It takes
  ordinal data blocks with `NaN` missingness plus declared level counts, fits
  each bivariate observed-pair table independently with the joint threshold/rho
  ML kernel, preserves per-pair `n_obs`/`n_missing`, and rejects all-missing
  pairs or empty marginal categories. It exposes the same bivariate
  threshold/rho score contributions and score Gamma as the complete/listwise
  path. This is pairwise observed-data likelihood, not multivariate MAR
  ordinal FIML.
- Frontier all-ordinal pairwise composite SEM fitting and inference (2026-06):
  `estimate::frontier::pairwise_ordinal_observed_data()` builds one raw-data
  observed-pair cache for complete or incomplete ordinal blocks;
  `fit_pairwise_ordinal_composite()` fits the SEM-implied shared-threshold /
  implied-correlation pairwise ML objective with the existing scalar optimizer
  backends and linear-equality reparameterization; and
  `pairwise_ordinal_composite_godambe()` returns Godambe sandwich SEs from
  casewise observed-pair score rows. The meat is built from each subject's sum
  over observed pairs, so cross-pair score covariance and pairwise missingness
  enter automatically. Bread is finite-difference on the composite score and is
  inverted in the equality-reduced alpha space before expanding the vcov back to
  full theta. `lr_test_pairwise_ordinal_composite()` reports nested composite
  LR tests through the existing Satorra-2000 result family. Global pairwise GOF,
  robust h/DPD/Huber composite objectives, nonlinear constraints, and a
  multivariate MAR ordinal FIML interpretation remain out of scope.
- Checked-in pairwise diagnostic fixture coverage under
  `cpp/tests/fixtures/pairwise/`: complete all-ordinal polychoric diagnostics,
  mixed pair labels and primitive ML helpers, complete/listwise joint
  composite diagnostics, and observed-pair composite missingness/count
  semantics.
- The SEM-facing ordinal moment path remains the lavaan-compatible
  shared-threshold path. Pair-local joint threshold/rho ML is reserved for
  diagnostics, robust pair experiments, and the frontier composite-likelihood
  surface rather than for constructing the current `OrdinalStats` moment vector.
- Shared-threshold multivariate missing ordinal modeling remains out of scope:
  the implemented missing-data ordinal path is pairwise observed-data
  composite likelihood, not multivariate MAR ordinal FIML.
- Muthen-style all-ordinal NACOV construction for thresholds plus
  polychorics.
- Ordinal workspaces separate observed moments from cached Gamma/weights and
  the requested fit/inference products. `OrdinalMoments` / `MixedOrdinalMoments`,
  `OrdinalGammaCache` and `OrdinalWeightPlan` are public in `data/ordinal.hpp`;
  legacy stats remain materialized compatibility objects. Fit-only ULS avoids
  Gamma; DWLS uses its diagonal; WLS needs the full weight. All-ordinal and
  mixed workspace construction, all-ordinal robust cache reuse, delta SNLLS
  and mixed/theta fitting are implemented.
- Threshold profiling supports fixed rows, shared labels, threshold-only
  linear equalities and joint multi-group equations with `n_b/N` weights.
  All-ordinal delta SNLLS uses this affine map; eligible theta SNLLS profiles
  standardized thresholds, with generic full-threshold fitting retained for
  fixed/shared/constrained cases. Non-unit or free DELTA scales use the full
  moment map in compact bounded and generic SNLLS fitting; frozen ULS oracle
  estimates gate these routes. Unit tests compare profiled/cache-aware fits with
  full-threshold and legacy bounded fits; lavaan fixtures 0013/0014 cover
  cross-group shared thresholds and threshold-only linear constraints.
  The maintained [workspace contract](../../design/ordinal-snlls-gamma-architecture.md)
  owns the data split, profiling algebra, cost rules and support boundaries.
- **Nonlinear equality restrictions (TASK-54.2).** All-ordinal native
  ULS/DWLS/WLS bounded fits enforce compiled nonlinear equalities through
  SLSQP (or an explicit IPOPT backend), together with affine restrictions.
  Compact DELTA fits use full moments when nonlinear equalities are present;
  DELTA free-set compaction remaps expression leaves. Configured DWLS retains
  the selected starts and records SLSQP/native convergence as a modified
  lavaan preset, since PORT's affine search cannot enforce nonlinear rows.
  Expected-information covariance and scaled/shifted global tests, nested
  Satorra tests, release scores, fit measures and retained-estimate conventions
  use the null space of the stacked affine and nonlinear Jacobians at the fit.
  Fitted df counts independent local restrictions; callers without fitted
  coordinates receive an error for nonlinear df. The ordinary DWLS policy,
  observed/IJ and misspecification profile routes return typed refusals naming
  the missing Lagrangian multiplier curvature. Expected-information scalar
  profile scaling includes the nonlinear tangent. SNLLS, association ML,
  pairwise composite and mixed ordinal nonlinear routes remain unsupported.
  Six pinned lavaan cases cover DELTA/THETA, binary/three-category outcomes,
  products, mixed affine/nonlinear and cross-group restrictions; a categorical
  Mplus nonlinear example has the Demo meaning and lavaan numerical gates.
  Nonlinear nested/combined release-score comparisons use the proved fitted-
  start transitive reference in [the oracle ledger](../../validation/oracle-defects.md);
  500 ordinal null replications give identical 26/500 rejection counts.
- **Keyword `group.equal`
  ordinal measurement invariance landed 2026-06-15 (theta).**
  `BuildOptions::group_equal`/`group_partial` ties the requested families across
  groups (synthetic shared labels feeding the same `compute_eq_groups` merge as
  explicit labels; the fixed marker is left untied) and `build` forces a mean
  structure when `Thresholds` is equated so the indicator-intercept rows exist.
  As in lavaan, equating `Intercepts` without `Means` frees the auto-added
  latent means in groups 2+ (user-fixed means stay fixed);
  `prepare_ordinal_*_partable` then applies the Wu-Estabrook (2016) release —
  free group-2+ residual variances (the released latent-response scale,
  binary-vetoed at `nth==1`) and indicator intercepts, marker/`f~1` left as is.
  The released block is fit under **theta** (the standard ordinal-invariance
  parameterization): the released scale is the free `~~` residual variance, which
  lavaan-theta reports identically, so theta_hat compares with no `~*~`
  projection. A shared ordinal moment-Jacobian block now covers the theta and
  released-delta cases used by fitting and robust nested tests: theta subtracts
  the freed intercept μ from standardized thresholds `(τ−μ)/√Σ*ᵢᵢ` and threads
  `J_mu`; released delta differentiates `(τ−μ)δ_i` plus the implied association
  rows, so freed latent means and released response scales carry nonzero
  moment-Jacobian columns. DELTA now retains lavaan's `~*~` coordinates rather
  than translating them into residual dimensions; threshold/loading invariance
  is gated by `delta_scales.json`, including scale SEs and scaled/shifted tests.
  lavaan-gated by the bounded golden `ordinal invariance (group.equal) theta
  fits match lavaan` over fixtures
  0017 (3-cat thresholds+loadings), 0018 (binary scale-veto), 0019
  (thresholds-only), and 0020 (thresholds+loadings+intercepts / scalar),
  matching df/chisq/theta_hat. At scalar, lavaan fixes the group-2+ indicator
  intercepts back to 0 and frees the group-2+ latent means; magmaan mirrors that
  in `prepare_ordinal_*_partable`. The released O(5) variances and scalar latent
  mean carry the documented `(n_g−1)/n_g` weighting gap, bounded at 3e-4 for
  0017/0020. The **Satorra-2000 nested LRT** ladder is gated too (`ordinal
  invariance nested LRT (satorra.2000 delta) matches lavaan`): configural→metric
  and thresholds→metric match scaled Δχ² 3.025 / Δdf 3 / p 0.388; metric→scalar
  matches scaled Δχ² 3.765 / Δdf 3 / p 0.288 with a scalar-only 1.5e-2 tolerance
  on the scaled statistic because the freed latent mean makes the Satorra scaling
  more sensitive to the same LS-weight gap. Configural→thresholds is explicitly
  recorded as a df=0 equivalence (same χ²/df; lavaan cannot form a positive-df
  `lavTestLRT` there). The Mplus Demo WLSMV DIFFTEST probe
  (`experiments/research/evidence/10-mplus-demo-wlsmv-difftest`) now gates the same shared
  released-delta moment Jacobian for the explicit 38-parameter scalar model
  under pairwise missing ordinal data: overlap-Gamma magmaan gives scaled-shifted
  Δχ² `22.365850` / Δdf 22 / p `0.438242`, matching Mplus Demo DIFFTEST
  `22.366000` / Δdf 22 / p `0.438200`; the raw LS objective also matches
  Mplus (`27.295091` vs `27.295100`). The R helper
  `mplus_wlsmv_invariance()` wraps that explicit Mplus-style delta ladder:
  configural, metric (`group_equal = "loadings"`), and scalar
  (`group_equal = c("loadings","thresholds")` plus fixed ordinal intercepts and
  freed non-reference latent means), returning the rung fits and
  DIFFTEST-style nested rows. It supports all-ordinal pairwise/listwise data and
  mixed continuous/ordinal listwise data; mixed pairwise missing is rejected
  until mixed pairwise NACOV construction exists. Checked fixtures live in
  `cpp/tests/fixtures/mplus_wlsmv_invariance`, and the executable R regression is
  `r-package/examples/mplus_wlsmv_invariance.R`. The ordinal golden chisq
  gates now apply the lavaan `Σ(n_g−1)F̂_g` convention rescale at 5e-3 (see
  numerical-conventions exception 4 and the test ledger). `experiments/_archive/ordinal-construction-boundary`
  now compares the legacy eager constructor with
  `ordinal_workspace_from_integer_data()`: fit-only ULS returns
  `OrdinalMoments` without Gamma, fit-only DWLS returns `OrdinalMoments` plus
  the Gamma diagonal, and WLS/fit-plus-inference still fall back to full
  `OrdinalStats`/Gamma materialization. `experiments/_archive/ordinal-snlls-speed`
  now includes delta/theta timing rows plus construction-aware raw-to-SNLLS
  rows: the legacy row rebuilds `OrdinalStats`/moments/starts/cache inside the
  timed operation, while the lazy ULS/DWLS row rebuilds `OrdinalWorkspace`,
  starts, and the profiled SNLLS fit. Theta rows use the cache-aware bounded
  comparator and threshold-only SNLLS profiling, so the report keeps them
  separate from delta's threshold-plus-covariance profiling split. The same
  benchmark/report now includes mixed continuous/ordinal delta rows comparing
  materialized full bounded DWLS/WLS with materialized full-threshold SNLLS,
  plus raw-to-fit mixed bounded/SNLLS construction timings. Fit-only mixed
  DWLS can also build a lazy `MixedOrdinalWorkspace` with
  `MixedOrdinalMoments` plus the exact Gamma diagonal, so the speed pilot
  carries legacy-versus-lazy mixed DWLS raw-to-fit rows; mixed WLS still
  materializes the full Gamma/inverse-weight path.
- DWLS diagonal weights, full WLS weights, bounded ordinal LS fitting, and
  thin R wrappers for ordinal stats plus DWLS/WLS fits.
- The R ordinal data boundary exposes consolidated dispatchers:
  `data_ordinal_stats_from_raw(robust = ...)` for all-ordinal data and
  `data_mixed_ordinal_stats_from_raw(polyserial = ..., ordered_mask = ...)`
  for mixed continuous/ordinal data. Method-specific Rcpp names remain
  callable compatibility aliases rather than entries in the displayed
  `magmaan_core` data group.
- Retained-fit `lavaan_nested_ordinal` composes lavaan 0.7.2's default
  WLSMV/ULSMV nested reporting: Satorra-2000 with delta restrictions,
  scaled-shifted reduction and larger-model information/Jacobian. Actual
  nesting uses the existing embedding gate. Objective counts are n_g−1;
  the nested sandwich retains original n_g/N fractions. Live R gates cover
  single-group delta/theta loading/covariance restrictions, two-group theta
  loading invariance and Wu-Estabrook thresholds→thresholds+loadings,
  saturated alternatives and explicit refusals. Plain DWLS/ULS retain the
  standard difference without a p-value; WLS reports the standard chi-square.
  Both ordinary `anova()` argument orders are gated. This changes only
  named compatibility reporting.
- Current ordinal fixtures validate thresholds, polychoric `R`, `NACOV`,
  `WLS.V`, `WLS.VD`, free sets, point estimates, degrees of freedom, and
  chi-square statistics across representative single-group, multi-group,
  skewed, sparse, near-empty, equality-constrained, and multi-group loading
  equality cases.
- The implemented ordinal LS boundary supports both lavaan delta and theta
  parameterizations for all-ordinal and mixed continuous/ordinal DWLS/WLS point
  estimates, and the all-ordinal cache-aware/SNLLS path now covers theta for
  ULS/DWLS/WLS point estimation. Theta post-fit support uses
  parameterization-aware threshold and association Jacobians for robust ordinal
  reporting, modification indices, score tests, and standardized-solution
  reporting.
- DWLS nested policy (task-17.4) uses the observed-Hessian parameter-space
  estimated-weight IJ law at H1 with the exact embedding restriction map.
  Its spectrum has exactly df_diff terms and matches the common-point profile
  and task-17.3 diagnostic construction. T = N(F_H0 − F_H1), All (SB/PEBA4 comparators), typed
  nesting reasons and unsupported nested score are preserved. The separate-point
  profile law remains an explicitly named lab comparator. Confirmed on fresh
  draws in decision study 05 (4.3-7.8%); the global test, ML and FIML recipes are unchanged.
- Moment-nested DWLS threshold comparisons (TASK-68) have limited validation;
  calibration is pending. `api::frontier::moment_nested_tangent` fits H1 to the
  null fit's implied moments using the shared DWLS weight, checks zero residual
  and equality-reduced tangent inclusion/rank, and returns the embedding and
  tangent T. An orthonormal row basis A annihilating T feeds the existing
  H1 observed-Hessian/IJ sandwich law, equivalent to the nonzero spectrum of
  `[H^-1 − T(T'HT)^-1T']B`, with q1 − q0 terms. This is a numerical local
  inclusion witness, not a symbolic proof of arbitrary manifold containment.
  Parameter-nested spectra agree with task-17.4 within 1e-10 relative. Two-group
  theta configural versus thresholds-equal has eight terms for four five-category
  indicators; three categories give typed `equivalent_models`. Thresholds+loadings
  versus loadings-only fails the implied-moment embedding and stays `not_nested`:
  item-scale releases change the standardized loading ratios. Policy and ordinary
  `anova()` route the threshold step through the same core. A 100-replicate
  correct-model check (1000/group, seed 68261004) gives mean 2.9735 versus trace
  2.8136 and variance 2.2289 versus 2 sum(lambda²) 2.0778; discrepancies fall within
  three Monte Carlo SEs (0.1441 and 0.4774). This is not a calibration claim.
- Reusable DWLS policy evaluation points (TASK-23): `api::DwlsPolicyFit` owns
  the fitted structure, parameters, ordinal statistics, parameterization and
  row provenance. It lazily retains IJ covariance and exact Newton Hessian
  results, including failures, and shares the IJ result between global
  covariance and larger-fit nested LR. Global policy output is retained too;
  original signatures remain fresh wrappers. R fits own process-local caches
  keyed by portable inputs (including data and weight recipe), rebuilt after
  serialization or PID changes. `inference_reuse(fit)` reports build counts.
  `policy_cache_test.cpp` gates exact fresh/cached covariance, statistics,
  spectra and probabilities for delta/theta, one/two groups and nested pairs;
  R gates restoration, repeated reporting/anova, fork rebuilding and input
  invalidation. Timing is recorded in [FIML](fiml.md#ordinary-policy).
- Study 05 retains global-only production-seed `--explore` and extends
  fresh-seed `--confirm` to 64 global and 40 nested cells (53–84, 139–146),
  seed base 817160001. Nested arms retain profile and fixed-weight comparators,
  evaluate the registered reference family on the policy's r-term spectrum,
  and save spectra. Runner and Modal `--family global|nested|all` select subsets;
  combine checks the selected cell count. The task-17.3 three-cell exploratory
  replay found common-point and parameter-IJ spectra agreeing with ten terms,
  with SB/PEBA4 size 5–6%; this is not fresh-seed confirmation.
- DWLS calibration study `experiments/decisions/05-dwls-policy-calibration`
  has a registered 146-cell runner for global/nested size, IJ coverage and
  loading/global power, with explicit policy-equivalence gaps and CPU pricing.
  Its approved theta pair constrains thresholds in H1 and adds loading
  equalities in H0 (Wu–Estabrook metric step); availability passes for binary
  and five-category indicators. The earlier loadings-to-thresholds-plus-loadings
  pair correctly returned `not_nested` because it changes fixed scales/intercepts.
  Threshold-shift power is excluded: both corrected fits impose threshold
  equality, so that perturbation is not threshold-invariance power. Smoke
  passed 292 draws with zero failures; the frozen 2,920-draw pilot records two
  nonconvergence failures and maximum policy gap 5.3e-14. Per-cell CPU timing
  extrapolates to 58.66 CPU-hours (14.66 ideal four-worker hours). Calibration
  remains open until separately authorized production evidence; pilot flags do not change the recipes.
- Friendly C++ `robust_ordinal`, `fit_measures`, `modification_indices` and
  `score_tests` replay all-ordinal/mixed preparation with the fit-time
  `LatentNames::row_user` mask, preserving explicitly fixed/free ordinal
  residual and intercept rows. Automatic ordinal/mixed starts receive the
  same mask, keeping their free-vector length consistent with the fitted
  model. Lower-level IJ, RBM/casewise and Satorra-2000 callers, plus DWLS
  policy nesting, may supply fit-time row masks for unprepared structures.
  Successful preparation records the per-group ordered set and binary vetoes
  in `LatentStructure::ordinal_preparation`. Repeated preparation validates
  that signature and preserves the free set, fixed values, constraints and
  starts, including a response-scale release transferred from `~*~` to `~~`.
  `LavaanParTable` carries the signature as header metadata; R projections
  preserve it in the `magmaan.ordinal_preparation` partable attribute, read by
  shared fit reconstruction. `LatentNames` still owns row provenance and
  `Starts` owns start hints. Explicitly fixed scales retain their original
  preparation rules; an incompatible ordered set or binary veto is an error.
- Explicit post-fit robust ordinal reporting returns sandwich SEs plus
  Satorra-Bentler, mean/variance-adjusted, and scaled/shifted statistics from
  the threshold-plus-polychoric moment vector. The implementation now uses a
  shared weighted-moment sandwich/U-Gamma primitive that can be reused by other
  LS moment stacks with arbitrary block weights and NACOV matrices. The same
  module also exposes `robust_weighted_moment_ij` for observed-bread
  infinitesimal-jackknife covariance from casewise moment rows plus optional
  estimated-weight influence corrections; scaled-test corrections remain
  estimator-specific. Complete continuous LS now has
  `robust_continuous_ls_fixed_weight_ij`, which treats the supplied
  second-stage weight as fixed and reduces to the observed-bread sandwich for
  ULS and caller-fixed weights, plus `robust_continuous_ls_gls_ij`, which adds
  the Hall-Inoue correction for the sample-built normal-theory GLS weight
  (`gmm::normal_theory_weight`). This GLS item is separate from the ML/FIML
  robust-score normal-theory path. Complete continuous WLS/ADF now has
  `robust_continuous_ls_wls_ij`, which rebuilds the dense empirical-Gamma
  weight from complete raw-data moment rows and carries the full
  estimated-weight influence, including meanstructure third-moment cross-blocks.
  Complete continuous DWLS now has `robust_continuous_ls_dwls_ij`, which uses
  the empirical-Gamma diagonal weight and carries only the diagonal
  estimated-weight influence.
  Complete continuous DLS now has `robust_continuous_ls_dls_ij`, which treats
  the mixing scalar as fixed and carries the mixed sample-built
  normal-theory/empirical-Gamma weight influence.
  The continuous fixed-weight GMM parameter profile-LR/CI surface can now opt
  into those complete-data IJ weight effects when `estimated_weight = TRUE`:
  GLS routes the robust or misspec meat through `SampleNormalTheory`, WLS
  through `SampleEmpiricalWls`, while ULS remains a fixed-weight case. The
  scalar profile reference is explicit (`Ordinary`, `RobustScaled`,
  `MisspecScaled`, or `MisspecMixture`); the misspec references use the
  analytic observed LS bread reduced through the equality-constraint basis and
  either the fixed empirical meat or IJ estimated-weight meat. Fitted-weight
  profile LR still rebuilds `W(theta)` at the endpoint and uses the fixed
  profile metric for robust/misspec references; full derivative-of-weight
  corrections remain research-tier.
  Continuous-LS observed-bread computation now uses an analytic moment Hessian:
  the Gauss-Newton `Delta' W Delta` term plus the residual-weighted LISREL
  second-derivative contraction, reduced through the equality-constraint basis
  `K`. All-ordinal and mixed ordinal/polyserial observed bread uses the same
  analytic shape, with scalar moment curvature for thresholds, standardized
  correlations/associations, continuous means/variances, and theta /
  Wu-Estabrook released-scale standardization. The public finite-difference bread
  helper remains as the validation oracle for weighted-moment bread checks.
  Complete all-ordinal ULS/DWLS/WLS now has `robust_ordinal_ij`: ULS reduces to
  the analytic observed-bread fixed-weight sandwich, DWLS carries the diagonal
  estimated-Gamma influence, and WLS carries the dense `IF(Gamma)` through
  `d' W IF(Gamma) W`. The dense ordinal Gamma influence is finite-difference
  gated against case-weight perturbations and its diagonal extraction is pinned
  to the DWLS helper. For observed/pairwise-missing all-ordinal stats with
  overlap Gamma, `ordinal_stats_from_observed_integer_data` now materializes
  case-aligned sparse moment-influence rows and missing-coded integer data.
  The rows' crossproduct reproduces the overlap NACOV; `robust_ordinal_ij`
  supports them for ULS/fixed-weight observed-bread covariance and uses
  support-aware observed Gamma influence/Jacobian helpers for DWLS/WLS
  estimated-weight corrections. The observed helpers are gated by
  missing-pattern case-weight finite differences and complete-data reduction.
  A first reliability consumer now sits on top of that stack:
  `estimate::frontier::ordinal_observed_omega` is a single-group all-ordinal
  post-fit functional for observed category-score omega. It reuses the fitted
  ordinal threshold layout and delta/theta/released-scale standardization rules,
  computes the category-score covariance via the same bivariate-normal rectangle
  probabilities as the polychoric stage, and reports the delta-method SE from
  `robust_ordinal_ij` (including DWLS/WLS estimated-weight influence). This is
  the canonical DWLS ordinal-omega proving slice; it is exposed in R through
  `magmaan_core$measures_reliability_ordinal_observed_omega` and has a smoke
  calibration probe in `experiments/engineering/banked/07-ordinal-observed-omega`. Multi-group
  pooling semantics and small-sample/profile-LR corrections remain separate
  follow-ups.
  Complete mixed ordinal/polyserial fixed-weight ULS now has
  `robust_mixed_ordinal_ij`, with mixed casewise moment influence rows stored on
  `MixedOrdinalStats`; it reduces exactly to the analytic observed-bread
  fixed-weight sandwich. The same entry point now supports ordinary complete-data
  mixed DWLS by carrying raw mixed blocks on `MixedOrdinalStats` and combining
  mixed data-direct diagonal `IF(Gamma)` with finite-difference
  `d diag(Gamma) / d kappa` in the mixed moment order. Mixed full WLS uses the
  same raw mixed blocks with dense mixed data-direct `IF(Gamma)` and
  finite-difference `d Gamma / d kappa`; its dense case-weight derivative is
  finite-difference gated and its diagonal extraction is pinned to the DWLS
  helper. The mixed data-direct diagonal influence contracts sparse case-bread
  blocks with precomputed transformed bread/Gamma factors, avoiding dense
  per-case matrix products. The retained full-matrix WLS path gates agreement
  at 1e-10 relative for binary/five-category, one/two-group, mixed/all-ordinal
  inputs under complete and observed assembly. TASK-88 local one-thread
  policy-call medians (three fresh fits, six continuous plus six five-category
  items, seed 88001, total N across groups) were:

  | Groups | N | Dense (s) | Sparse (s) |
  |---|---:|---:|---:|
  | 1 | 1000 | 3.511 | 1.069 |
  | 1 | 4000 | 8.497 | 4.263 |
  | 2 | 1000 | 1.900 | 1.220 |
  | 2 | 4000 | 11.003 | 4.054 |

  The matched decisions/05 mixed smoke cell 3 (one draw) took 0.982 versus
  0.629 s. Temporary native instrumentation separated assembly from case
  processing: for one group at N=1000/4000, assembly took 0.009/0.037 s
  dense and 0.010/0.043 s sparse; case processing took 0.225/1.002 s dense
  and 0.034/0.152 s sparse. Native sampling was unavailable (`perf` permission
  restriction). These workstation timings describe cost, not calibration;
  the remaining policy cost includes finite-difference Gamma movement.
  Timings use `mixed_draw` / `mixed_fit` from decisions/05 with five categories,
  then time only `policy_inference` on each fresh fit; smoke uses
  `--lane mixed --smoke --cell 3 --workers 1`. Run metadata and logs remain
  local under task88-named results and `~/.cache/magmaan-logs/`.

  For observed/pairwise-missing mixed ordinal/polyserial stats,
  `mixed_ordinal_stats_from_observed_data` now materializes support-aligned
  rows for thresholds, continuous means/variances, polychorics, polyserial
  covariances, and Pearson covariances; the rows reproduce the overlap NACOV
  and let `robust_mixed_ordinal_ij` handle ULS/fixed-weight observed-bread
  covariance under MCAR. The same entry point now routes observed mixed DWLS
  and dense WLS through support-aware observed mixed Gamma data-influence and
  finite-difference `d Gamma / d kappa` helpers, and now materializes the
  resulting `gamma_diag_influence` / `gamma_full_influence` rows directly on
  `MixedOrdinalStats`. `robust_mixed_ordinal_ij` consumes those rows first and
  keeps the NaN-coded raw blocks only as a complete-data/backward-compatible
  fallback. The same precomputed diagonal channel feeds
  `mixed_ordinal_dwls_profile_rmsea`, so observed-missing mixed profile RMSEA
  has the same explicit-data-object contract as the all-ordinal overlap path.
  The observed helpers reduce to the complete-data mixed helpers and are
  covered by deterministic MCAR fit-level IJ/profile tests. Robust/experimental
  mixed stage-1 variants need separate Gamma-influence derivations.
  ML2S now exposes the observed-bread Stage-2 regime through
  `TwoStageBread::Observed`, and the saturated-EM moment influence primitive
  is available as `saturated_em_moment_influence`. Raw complete-data ML2S
  observed-bread covariance for `TwoStageWeight::{Dwls,Adf,Dls}` now reduces
  through the continuous-LS IJ adapters, while fixed-weight `Uls` stays on the
  shared ML2S IJ assembly. Raw missing-data ML2S observed-bread covariance for
  those estimated weights now includes the FIML Stage-1
  sandwich-Gamma influence via a case-weight finite-difference over the
  saturated EM `(H,J,ACOV)` stack; scaled-test fields remain fixed-weight. The
  remaining performance follow-up is an analytic replacement for that
  finite-difference Gamma influence if the frontier path becomes hot. The
  default NT route is the ordinary normal-theory ML robust-score path; the
  moment-quadratic GLS IJ correction remains the complete continuous-LS adapter.
- `standardize_lv`/`standardize_all` and `compute_defined` accept
  ordinal/mixed-ordinal fits at both the C++ api and the Rcpp bindings. These
  parameterization-agnostic transforms operate over the
  *prepared* ordinal partable: `fit_ordinal_bounded` fixes the latent-response
  residual variances the delta constraint determines and compacts the free set,
  so the stored estimates/vcov live in that reduced space while `Model` carries
  the un-prepared structure. The api functions reconstruct the prepared
  structure on demand (an internal `prepared_structure` helper replaying
  `prepare_ordinal_partable`); the Rcpp bindings get it for free because
  `ctx_from_fit` parses the prepared partable. (Before this bridge the api-level
  guard-removal was dead: it fed the reduced theta into the un-prepared
  evaluator and aborted on the dimension mismatch.) `standardize_all` takes an
  `ordinal_delta_unit` flag: under the
  delta parameterization a categorical indicator's latent response is
  unit-variance, so its loading is standardized by the latent SD only (σ_rr = 1)
  rather than the assembled `λ²ψ + 1`. This is applied in both the plain-CFA
  `Lambda` slot and the all-y RAM `Beta` slot, so a mixed SEM's endogenous-factor
  loadings and structural paths standardize to lavaan `std.all`. The bindings
  read the parameterization from the partable attribute and the api from the
  fit's `EstimatorSpec`. Mixed-ordinal stats construction also no longer aborts
  DWLS when the full-WLS NACOV is singular (common at small N with many
  indicators): the inverse is non-fatal, `W_wls` is left empty, and DWLS / the
  robust sandwich proceed on the diagonal weight / NACOV.
- Ordinal/mixed-ordinal factor scores are a separate categorical estimator, not
  a guard flip over the continuous regression/Bartlett predictor. The measures
  layer exposes `factor_scores_ordinal()` and `factor_scores_mixed_ordinal()`;
  `api::factor_scores()` dispatches ordinal and mixed fits there; and the R
  `factor_scores()` wrapper defaults categorical fits to EBM while accepting
  `method = "EBM"`, `"ML"`, or `"EAP"`. EBM/ML score each unique complete
  response pattern by damped Newton with analytic ordinal interval-probability
  gradient/Hessian terms; EAP is supported for one-factor models through the
  vendored QUADPACK infinite-interval integrator. The same EAP quadrature also
  exposes one-factor posterior variance/SE, sample-moment PRMSE, and the
  direct concrete ordinal reliability through `factor_score_precision_*` /
  `api::factor_score_precision()` and the R `factor_score_precision()` helper.
  The current categorical scope is diagonal residual `Theta`; multi-factor EAP
  and correlated-residual orthant
  probabilities remain deferred. Checked-in lavaan parity is gated by
  `cpp/tests/golden/ordinal_golden_test.cpp` ("ordinal/mixed factor scores (EBM/ML)
  match lavaan", 5e-4) over the `fits.DWLS.fscores` oracle: single-group EBM
  (all-ordinal and mixed) and mixed ML. All-ordinal ML (unbounded mode on
  extreme patterns) and EAP (no categorical `lavPredict()` oracle) are not
  lavaan-gated (EAP stays self-checked). The EAP precision surface additionally
  carries a Monte-Carlo ground-truth gate (`cpp/tests/unit/api_sem_test.cpp`,
  "ordinal EAP factor-score precision tracks Monte-Carlo PRMSE"): on a
  five-indicator three-category one-factor model simulated with retained latent
  `Z` and fit under `std.lv`, the reported `pooled_prmse` matches the realized
  `corr(Z, E[Z|Y])²`, the mean posterior variance matches the realized EAP MSE,
  and the concrete reliability reduces exactly to `1 - mean Var(Z|Y)` under unit
  latent variance (gaps ~1e-3 at n=8000). Multi-group categorical EBM is correct
  and is validated transitively: lavaan's own multi-group categorical
  `lavPredict()` returns a non-stationary point for non-reference groups (it is
  not a usable oracle there), so the same golden instead checks that each
  group's multi-group EBM equals an independent single-group fit on that group's
  data (~3e-8) for the unconstrained two-group fixture, and single-group EBM is
  lavaan-gated.
- All-ordinal DWLS/WLS fit measures are exposed through
  `estimate::fit_measures_ordinal()` and `api::fit_measures()`: CFI/TLI/RMSEA
  use the categorical independence baseline over the polychoric moment stack,
  with WLS minimizing threshold nuisance residuals under the full weight
  matrix, and ordinal SRMR uses the lavaan correlation-metric denominator that
  includes zero diagonal residuals. The bfi ordinal parity fixture gates DWLS
  and WLS CFI/TLI/RMSEA/SRMR against lavaan.
- Lab `fit_measures()` uses the fitted categorical moment layout and equality-
  reduced dimension for df, retaining native `n F` objective/baseline reporting.
  Live lavaan gates cover grouped DELTA/THETA threshold/loading invariance and
  cross-group labels: df agrees exactly, while chi-square and dependent indices
  use lavaan inputs rescaled to the native convention. Continuous equality
  gates also pass.
- Weighted-χ² reducer formulas are shared across eigenvalue and trace-summary
  callers: Satorra-Bentler, mean/variance-adjusted, and scaled/shifted tests
  can consume either the UΓ spectrum or `(Σλ, Σλ²)` when a low-rank trick has
  already computed the traces.
- A first mixed continuous/ordinal path builds lavaan-ordered thresholds,
  continuous means/variances, polychoric/polyserial/covariance moments,
  NACOV/DWLS/WLS weights, and DWLS/WLS delta/theta fits. Four Newsom longitudinal
  mixed DWLS cases have compact derived-moment/NACOV-diagonal oracle fixtures
  under `textbook_mixed/`, with gates for same-point criteria, fitted moments,
  complete tables and refits. Their previously reported objective gaps came
  from omitted terminal-outcome covariances: `model_spec()` requires explicit
  `auto_cov_y = TRUE` to match those `sem()` calls. Categorical fmin comparisons
  apply lavaan's `(N - 1) / N` reporting factor. These gates do not validate
  textbook full-WLS or robust inference. Mixed delta SNLLS now
  has a materialized-stats full-threshold entry point,
  `estimate::fit_mixed_ordinal_snlls_full_thresholds()`, that profiles the
  conditionally linear threshold, mean, variance, and covariance parameters
  through the generic Golub-Pereyra split and matches bounded mixed DWLS/WLS on
  a focused unit test. The fit-only mixed DWLS workspace path now avoids full
  Gamma/WLS materialization by carrying `MixedOrdinalMoments` plus the Gamma
  diagonal into bounded and full-threshold SNLLS fits. Mixed full-Gamma cache
  reuse also covers robust DWLS/WLS reporting through a mixed-moments
  overload. The mixed Gamma construction mirrors lavaan's muthen1984
  estimating-equation sandwich exactly (stage-1 mu/var ML scores with
  per-variable bread blocks, pair-ML scores including mu/var coupling channels
  for polyserial and continuous-continuous pairs, and the delta-rule
  correlation-to-covariance transform applied to post-sandwich variance
  influence; see the design doc's "Mixed Gamma Construction"), so the mixed
  goldens gate NACOV/weights at 1e-6, point estimates at the all-ordinal
  theta 1e-5 / chisq 5e-3 contract, and the robust scaled-test fields at
  all-ordinal tightness — at lavaan's theta-hat and at magmaan's own. The
  same construction backs the lazy fit-only DWLS diagonal and the
  Huber-residual single-ordinal rebuild (no-clip reproduces the ML Gamma
  exactly). Mixed theta SNLLS runs through the same full-threshold stack:
  under theta only thresholds stay Golub-Pereyra linear (the standardized
  covariance moments make the rest nonlinear), gated against the bounded
  theta fit on a well-identified three-category design. Binary-indicator theta
  models carry a near-flat lambda/psi ridge where optimizer endpoints are
  arbitrary, so theta parity is only meaningful on identified designs.
  Observed-data mixed stats now cover both pure observed-pairwise and a basic
  hybrid first stage:
  `estimate::fiml::mixed_ordinal_stats_hybrid_fiml_from_observed_data` keeps
  the ordinal/polyserial pieces pairwise, but replaces continuous
  means/covariances and their influence rows with saturated continuous FIML
  estimates. The hybrid stats recompute NACOV and estimated-weight Gamma
  influence rows empirically from the combined influence matrix, then feed the
  existing mixed DWLS fit, IJ, and profile-RMSEA machinery. This is a
  correctness-first implementation; validation remains focused on MCAR/MAR
  efficiency and full-WLS stability.
  Mixed WLS and fit-plus-inference workspaces are lazy about weights: the
  builder carries moments plus full Gamma into the cache and defers the
  O(m³) WLS inverse (and DWLS weight extraction) to the
  `ordinal_gamma_cache_ensure_*` helpers at first use. Threshold-profiled
  mixed objectives remain a later slice; reduced-Gamma robust products sit in
  the speculative backlog. The lavaan-backed fixtures include a
  complete/listwise sparse 4-category boundary case.
- Covariance shrinkage is available under `data::frontier` for both continuous
  `SampleStats` and mixed continuous/ordinal `MixedOrdinalStats`. Mixed
  shrinkage leaves thresholds and continuous means in place, transforms the
  lower-triangle association/covariance block, propagates the moment
  transformation through `NACOV`, and rebuilds DWLS/WLS weights so C++ and R
  consume the same shrunk moment stack. Missing-data ML2S now has a narrower
  frontier analogue through `regularize_saturated_stage1`: it conditions the
  saturated FIML Stage-1 covariance input before Stage 2 and delta-propagates the
  transformed ACOV. The distinct direct-FIML nested-test problem — regularizing
  the H1 information/acov reference used by Satorra spectra — remains backlog
  work because it changes the reference law. The old
  `magmaan/data/shrinkage.hpp` include path remains a forwarding shim.
- Public complete-data polyserial pair kernel for mixed continuous/ordinal
  work, exposing fixed-threshold rho ML, likelihood, casewise threshold/rho
  scores, and pairwise score Gamma. The mixed sample-stat builder now reuses
  this kernel for polyserial associations.
- Experimental fixed-marginal polyserial DPD is available under
  `data::fit_polyserial_pair_rho_dpd()` and
  `data::polyserial_pair_dpd_scores()`. It keeps the shared ordinal thresholds
  and standardized continuous marginal fixed, estimates only the polyserial
  association, delegates `alpha = 0` to the ML kernel, and returns DPD
  attenuation weights plus score/Gamma/bread diagnostics.
- Experimental SEM-facing mixed polyserial DPD stats are available under
  `data::mixed_ordinal_stats_polyserial_dpd_from_data()`. The builder preserves
  the existing mixed moment order, shared ordinal thresholds, continuous
  means/variances, ordinal-ordinal polychorics, and continuous-continuous
  covariance moments, while replacing only continuous-ordinal association
  equations with fixed-marginal DPD and rebuilding NACOV/DWLS/WLS weights from
  the mixed casewise influence rows.
- Experimental pair-local full DPD polyserial fitting is available under
  `data::fit_polyserial_pair_joint_dpd()`. It jointly estimates continuous
  mean/scale, ordinal thresholds, and rho with DPD tuning `alpha`, and returns
  probabilities, joint densities, and `f(x, y)^alpha` attenuation weights. DPD
  here means density power divergence and is not part of robcat parity.
- Pair-local full polyserial DPD remains a bivariate diagnostic only and is not
  used to construct `MixedOrdinalStats`; SEM-facing robust mixed moments use
  the shared-marginal fixed-threshold contract instead.
- Public complete-data mixed pair helpers also expose continuous-continuous
  normal pair likelihood/diagnostics, casewise mean/variance/covariance scores,
  score Gamma, and labels for the exact threshold, negative-mean, variance,
  and lower-triangle pair order used by `MixedOrdinalStats`.
- The continuous normal pair likelihood is currently a mixed-pair primitive
  and benchmark against complete-data ML/FIML, not a supported standalone
  normal-data pairwise SEM estimator.
- Ordinal and mixed delta DWLS/WLS expose fixed-parameter modification indices
  and equality-release score tests over the same threshold/correlation moment
  vectors and weights used by fitting. Both families use score -N J'r and
  metric N J'J for the fitter's F/2 criterion. Mixed ordinary/fixed robust MI
  no longer carries the former extra factor two; EPC is unchanged. Fixture
  0005 has independent frozen-moment df=1 Schur/robust-variance reconstruction
  and explicit (N-1)/N oracle score transport. Complete mixed DWLS/WLS
  estimated-weight MI/releases now use exact empirical first-stage rows, the
  mixed IJ Gamma channel, and observed nuisance sensitivity by default in lab R.
  Expected sensitivity and fixed weights remain explicit comparators.
  The score metric remains expected; one-direction robust MI is N u²/B_eff.
  `mixed_ij_test.cpp` gates explicit H/B reconstruction, replicated augmented
  case-score derivatives, exact-fit cancellation and the common-point
  one-restriction nested-law identity across groups and parameterizations.
  In observed geometry the local nested quadratic N u²/H_eff, divided by its
  one-restriction eigenvalue B_eff/H_eff, equals the robust MI N u²/B_eff.
  ULS and genuinely absent estimated-weight recipes return UnsupportedInference.
  The ordinary package still defers its MI API; lab robust workers now reach
  the estimated-weight result instead of refusing. The plain C++
  `api::modification_indices()`/`api::score_tests()` keep the standard
  statistic for every estimator; the robust route is explicit.
- Complete mixed DWLS estimated-weight IJ uses a separate empirical sampling
  channel: `data::mixed_moment_sampling_influence` differentiates the marginal
  and pairwise score equations in threshold/negative-mean/variance/association
  coordinates and solves their empirical Jacobian. The pairwise Hessian and
  nuisance derivatives include threshold, continuous mean and variance coupling.
  `MixedOrdinalStats::moment_influence`, NACOV and fitting weights retain the
  lavaan OPG convention; `sampling_moment_influence` may supply empirical rows
  explicitly. Otherwise `robust_mixed_ordinal_ij` reconstructs them from complete
  raw data. The fitting-weight influence combines the unchanged NACOV direct
  channel with its moment Jacobian evaluated along the empirical sampling rows.
  `mixed_ij_test.cpp` gates independent replicated case weights under
  misspecification (parameter rows within 1e-5 up to covariance-unobservable
  sign; moment/Gamma row errors below 7e-8). Refits are Newton-polished to remove
  optimizer stopping noise, with an explicit objective-gradient check. A
  saturated delta/theta mixed model gates vanishing weight influence.
  Stratified delete-one diagonal errors at N = 600 → 1200 per group are
  0.491% → 0.282% (delta, one group), 0.510% → 0.275% (delta, two),
  4.169% → 1.920% (theta, one) and 4.147% → 2.027% (theta, two).
  Every slice beats the observed fixed-weight OPG sandwich; the N = 1200 gate
  is 2.5%, with shrinkage checked separately, not a universal O(1/N) claim.
  Pure ordinal/continuous endpoints remain rejected by the mixed builder with
  `NumericIssue`, so direct endpoint-fit reductions are unavailable. Continuous
  marginal sampling rows equal the analytic mean/ML-variance derivatives.
  Continuous DWLS uses empirical-moment Gamma weights, while mixed fitting
  retains the marginal/pair-score OPG NACOV, so their estimated-weight channels
  do not share a fitting-weight convention. ULS, missing-data and robust
  mixed builders, RBM and all-ordinal routes retain their existing influence
  contracts; this gate does not validate those routes as exact empirical IJ.
  Complete mixed WLS IJ now also selects empirical sampling rows for its full
  Gamma influence. The mixed policy and MI/release composition are available;
  calibration remains separate.
- Mixed continuous/ordinal DWLS/WLS fit-measures are exposed through the same
  `api::fit_measures()` surface as all-ordinal fits. The mixed independence
  baseline profiles the marginal threshold/mean/variance block under the fitted
  DWLS/WLS weight before testing the zero-association model, and SRMR is
  computed from standardized mixed association residuals; both are fixture-gated
  against lavaan's mixed ordinal CFI/TLI/RMSEA/SRMR fields.
- Ordinal and mixed categorical entry points validate block counts, threshold
  metadata, ordered masks, moment/weight/NACOV dimensions, finite values,
  positive `n_obs`, and positive NACOV diagonals before fitting or robust
  reporting.

#### Ordinal weighted-LS whitening is structure-aware

The weighted-LS moment residual is `r_b = sqrt(n_b/N) · F_bᵀ d_b`, where
`F_b F_bᵀ` is block b's moment weight. Only full WLS needs a dense `F`: ULS uses
the identity and DWLS a diagonal, the latter provably so, since every `W_dwls`
construction site writes a zero matrix and then fills only its diagonal.

`cpp/src/estimate/detail_whiten_factor.hpp` carries that structure in the type.
`detail::WhitenFactor` is an Identity/Diagonal/Dense left-multiplying operator
and `detail::MomentWeight` is the same idea for `W` itself. Producers
(`weight_factors`, `full_weight_factors`) pick the kind; consumers call
`t_apply`, so a diagonal weight costs O(rows · cols) instead of a GEMM and the
identity costs a copy. Two structural consequences fall out and are honored:

- a diagonal `W` has a structurally zero threshold-by-correlation block, so
  `G_corr` is exactly zero and the profiled thresholds decouple from the
  correlation residual. `ProfiledWeightWorkspace::corr_coupled` records that
  once and the consumers skip the product rather than multiplying by zero on
  every gradient;
- `theta_threshold_profile` keeps a diagonal input diagonal, since a diagonal
  weight leaves a diagonal Schur complement.

`prob.eval` is set on every ordinal `GmmProblem`, so `optim::scalarize` uses its
fused branch instead of evaluating the model twice per gradient.

The contract is that none of this moves a number. Estimation-only DWLS at
N=1000 dropped from 838 ms to 52 ms at p=50 with the empirical complexity
exponent in p falling from 4.47 to 3.00 — the floor for a dense O(p²)-row by
O(p)-column Jacobian — while fitted parameter vectors stayed bit-identical and
gradient counts unchanged. The talk-side harness `private/oslo-psychometric-gathering-2026/tools/
benchmark_ordinal_whitening.R` (outside this repository) is the before/after harness and prints both the
element-wise parity check and the fitted exponent pair.

The 2026-09-17 paired timing check (three-factor CFA, N=1000) recorded:

| Indicators | Before (ms) | After (ms) | Speedup |
|---|---|---|---|
| 12 | 1.5 | 0.7 | 2.1x |
| 20 | 10.4 | 2.8 | 3.7x |
| 30 | 66.7 | 7.2 | 9.3x |
| 40 | 307.6 | 24.4 | 12.6x |
| 50 | 838.4 | 52.1 | **16.1x** |

Against lavaan, the pipeline speedup was 21.7x at p=12 and 13.0x at p=50;
estimation-only speedup was 18.6x and 3.7x, respectively. These are timings
from that validation run. The fused callback change alone was within timing
noise: the removed residual-only evaluation was about 200x cheaper than the
remaining Jacobian evaluation, so the predicted independent 2x gain did not
materialize. The profiled workspace also detects the structurally zero
threshold/correlation coupling once and skips its products. Remaining weight
storage and workspace cleanup lives in the
[backlog](../../backlog/todo.md#ordinal-weight-storage-and-workspace-cleanup).

#### Cross-products (OPG) information performance

`inference::casewise_scores` / `information_cross_products` no longer form the
dense p*×p* normal-theory Γ_NT. The weight is applied through the trace identity
Γ_NT⁻¹ = ½·Dᵀ(Σ̂⁻¹ ⊗ Σ̂⁻¹)D: the σ-segment of WΔ is assembled column by column
from Σ̂⁻¹A_aΣ̂⁻¹ at O(p³) each, replacing an O(p⁶) Cholesky plus O(p⁴·n_free) of
triangular solves. Columns whose block slice is entirely zero are skipped, which
is what makes multi-group models cheap on this path. The PD check now guards Σ̂_b
rather than Γ_NT(Σ̂_b), matching the contract the header already documented.

Measured with `benchmarks/timing` (i7-1355U, `opt`, single-threaded, cfa_3f,
N=1000): p=48 33.3 ms → 11.0 ms (3.0×), p=96 1037 ms → 89 ms (**11.6×**), and
the 173 MB Γ_NT allocation at p=96 is gone. `information_expected` and
`information_observed_analytic` are unchanged within noise, confirming the change
is confined to this path. The residual cost is the Z_c·WΔ product at
O(N·q·n_free), which is the necessary work for an OPG estimator — forming
Z_cᵀZ_c first would be O(N·q²) and strictly worse whenever n_free < q, so the
current multiplication order is already the right one.

Guarded by `cpp/tests/unit/inference_test.cpp`'s "Γ_NT⁻¹·vech(A) matches the
trace-identity form casewise_scores uses", which checks the substitution against
a dense `data::gamma_nt` solve across p ∈ {1,2,3,5,8} to 1e-9 relative. The
pre-existing asymptotic OPG-vs-expected test is far too loose to catch an error
in this algebra.

#### Ordinal DWLS Gamma performance

Complete-data all-ordinal DWLS estimated-weight inference now evaluates both
Gamma-diagonal influence channels using local item/pair subsystems. The public
`ordinal_gamma_diag_data_influence` and `ordinal_gamma_diag_jacobian_fd` shapes
and moment ordering are preserved. The Jacobian still uses central finite
differences with the caller's `h_rel`; there is no statistical approximation or
change to the fitted criterion. Full-WLS Gamma helpers retain their existing
implementations. Pairwise-missing ordinal and mixed diagonal movement use the
sparse extensions described below.

The private workspace in `cpp/src/data/ordinal.cpp` builds marginal category counts
and score/bread blocks once per call. A correlation's local subsystem contains
its two items' thresholds and its own rho, so a binary pair needs only three
coordinates. Pair counts weight the category-cell scores returned by the
existing `ordinal_pair_scores` kernel. The local influence Gram yields the
required Gamma block; direct bread variation is evaluated on that same block,
using the equality of the two diagonal sandwich terms for symmetric Gamma.
Only the requested influence columns are scattered back to observations.
Threshold direct influence reduces to `Gamma_kk - g_ik^2` because its bread is
its marginal score Gram. No bread-variation term is omitted.

Finite differences reuse the marginal blocks at each positive/negative
threshold perturbation and evaluate only pairs incident to that threshold or
the pair whose rho changes. At 18 binary indicators this uses 918 pair-score
builds instead of 52,326. The returned 171×171 Jacobian is initialized to zero
and has only 477 possible nonzero entries. The full threshold block's relative
positive-definiteness cutoff is retained across items and at perturbed moments;
independent item cutoffs alone would accept badly scaled global blocks.

Pairwise-missing ordinal movement now uses the same cached marginal category
counts and incident-pair cell grids. The sandwich adds rows observing only one
endpoint through that endpoint's marginal influence; overlap counts alone do
not reproduce the diagonal. The full sample size, item-specific observed rows,
relative threshold-bread cutoff, and central FD step are retained. The previous
observed global assembly remains callable through the private
`detail_gamma_reference.hpp` reference interface; complete-ordinal validation
also uses the unchanged full-Gamma FD helper.

Mixed complete and observed movement assemble one marginal item or one incident
pair at a time and scatter the resulting diagonal FD rows into the original
moment order. A threshold affects its item's marginal block and incident pairs;
a continuous mean or variance affects its marginal block and every incident
association (including the covariance transform); an association affects only
its own row. Steps use the existing threshold-spacing, variance and correlation
margin rules. Unchanged marginal spectral bounds are cached, and each local
perturbation retains the full marginal bread's relative PD gate. The dense mixed
FD references remain callable through the same private reference interface.
The regression gate covers one/two groups, two/five/seven categories, skewed
thresholds, complete and pairwise-missing ordinal/mixed data, at 1e-8 relative
agreement with identical steps. Existing policy, IJ, MI and jackknife assertions
are unchanged.

TASK-89's one-thread policy timing uses a two-group, twelve-indicator one-factor
model, with either twelve five-category ordinal items or six ordinal and six
continuous items. Each group has the stated N. The generator uses seed 89001,
loading 0.65, residual SD 0.76, and cuts at -1, -0.3, 0.3 and 1; three fresh
converged fits are prepared outside the timer. Medians below time the first
`policy_inference()` call per fit, under the warm Clang opt build, `nice -n 10`,
and single-thread BLAS/OpenMP. Temporary scoped timers measured movement only;
a temporary switch selected the retained dense mixed FD within the same build.
Both were removed before final validation. The complete all-ordinal movement
was already sparse before this task, so that row is an unchanged control.

| Items | N per group | Policy seconds, before → after | Movement seconds, before → after | Policy speedup |
|---|---:|---:|---:|---:|
| 12 ordinal | 1000 | 0.097 → 0.095 | 0.04748 → 0.04671 | unchanged |
| 12 ordinal | 4000 | 0.147 → 0.146 | 0.04772 → 0.04668 | unchanged |
| 6 ordinal + 6 continuous | 1000 | 1.997 → 0.454 | 1.81713 → 0.27512 | 4.40x |
| 6 ordinal + 6 continuous | 4000 | 7.631 → 2.014 | 6.85624 → 1.17410 | 3.79x |

The mixed movement channel accounted for about 90–91% of policy time before
this change; its own median cost fell by 5.84–6.60x. These are local timings,
not statistical calibration or CI timing gates. The standalone R generator and
raw timing logs are untracked under `~/.cache/magmaan-logs/task89_timing.R` and
`task89-profile-{dense,sparse}-clean.log`. A separate uninstrumented baseline
agrees with the unchanged ordinal control and confirms the mixed bottleneck.

The small timing protocol lives in `benchmarks/README.md` and
`benchmarks/ordinal_gamma_influence_bench.cpp`: three fixed synthetic samples,
one warm-up and five repetitions, one process, the same Clang 21.1.8 `opt`
settings (`-O3 -DNDEBUG -march=native`, Eigen threading disabled), on an Intel
i7-1355U. The benchmark fits a one-factor DWLS model with a deliberately
misspecified second loading, then times the complete `robust_ordinal_ij` call.
Parsing and start construction are outside the measured stages. On 2026-09-09,
median milliseconds were:

| n | Indicators | Categories | Direct Gamma IF, before → after | Gamma Jacobian, before → after | Complete IJ, before → after | IJ speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | 18 | 2 | 219.83 → 0.29 | 350.92 → 1.14 | 576.92 → 2.58 | 224x |
| 1200 | 18 | 2 | 764.35 → 0.52 | 1069.12 → 1.23 | 2225.82 → 5.73 | 389x |
| 300 | 18 | 4 | 456.56 → 1.75 | 1461.87 → 21.48 | 1936.30 → 26.36 | 73x |

The sum of statistics, fit, and IJ medians fell from about 582 to 6.3 ms, 2233
to 13.0 ms, and 1949 to 36.9 ms, respectively. These sums are not separately
timed raw-data pipelines. The timings are advisory local comparisons, not a
replay of a paper's generator, a calibration result, or CI timing gates. Raw
CSV/min/max/CPU logs and binary hashes are ignored under
`benchmarks/results/ordinal_gamma_local_{before,after}.{csv,log}` and
`benchmarks/results/ordinal_gamma_local.sha256`. All 18 printed result
checksums agree at their ten-significant-digit precision. The pre-change core
was `f5bb13fc`; the old/new opt archive SHA-256 values are
`2c12c92f8e818bdbefcd3856db0b95ba4e0d3b449eb7ef2cdefbc154330a8734` and
`79546954d0a6e31bb8c94439e284d6595fd348a7cbe625b563d413f714c0c18e`.

Validation includes the existing case-weight finite differences and full/
diagonal/observed reductions, new full-Gamma reference comparisons across four
items with unequal category counts at fitted and perturbed moments and three
FD step sizes, sparse-cell/high-rho checks, and the global conditioning gate.
An end-to-end DWLS test combines the retained dense WLS Gamma channels with
the fitted moment-to-parameter map and matches the complete IJ covariance with
nonzero fitted residuals. The complete optimized ordinal suite passes all 134
tests and 3,798 assertions, including lavaan golden parity. Seven targeted
ASan/UBSan tests pass 511 assertions; leak detection was disabled because
LeakSanitizer cannot run under this environment's process tracer. The R
package's vendored core is refreshed and `just r-dev` reinstalls the development
package. Before/after R checks on 18 binary and four-category indicators give
identical fitted parameters, maximum covariance difference 4.87e-15, and maximum
SE difference 2.27e-14. The existing `ordinal_dwls_wls.R` workflow passes. The
repository-wide layering check still reports unrelated existing paper-to-tests
references in `covariance-honest-sem` and `target-specific-distinguishability`.

### Ordinary DWLS nested reference

Every all-ordinal and mixed DWLS nested spectral test uses All (TASK-81,
decision 2026-10-06), retaining SB/PEBA4 and the SB scale as comparators.
Parameter-nested and moment-nested paths share the observed-Hessian/IJ law;
All uses every positive eigenvalue with the global test's negative truncation.
The registered decisions/06 fresh-draw threshold confirmation passed [3%, 7%]
at N >= 500/group (3.45–6.0% in all 144 null cells). decisions/05's fresh-draw
parameter-nested All comparator gave 3.9–6.9% at N = 400 and 4.3–6.4% at
N = 1000 in 32 null cells, with equal size-adjusted power. decisions/07 remains
pending as cross-model confirmation. Ordinary `anova()` recommends only
fit_function_difference/all and retains the typed unsupported score row.

### Ordinary DWLS global reference

`api::policy_inference_dwls` reports n F with its unchanged `robust_ordinal`
spectrum and the exact weighted chi-square All tail (every positive sample
eigenvalue), exposed as `reference = "all"` and `p_all`. SB/PEBA4 fields are
unset for this component; ML/FIML select `peba4`; nested DWLS also selects
`all` (TASK-81, decision 2026-10-06), with SB/PEBA4 comparator fields retained. Decision
study 05 compares `policy_all` with its explicit `all` arm at 1e-7 and confirmed
it on fresh draws (2.9-6.8%).

Decision study 07 prepares broader global and nested reference-law validation
across eight frozen textbook latent-response populations, including structural,
longitudinal and grouped invariance models. Its development pilot retains full
exact-first-stage policy spectra for fourteen reference comparisons. Production
and TASK-81 adoption remain open; this preparation changes no inference recipe.

DELTA response-scale equalities, fixed non-unit scales and linear scale
constraints use live lavaan coordinates (TASK-53.1); see the ordinal LS contract
and the frozen `ordinal/delta_scales.json` gates above.

Mplus categorical lowering (TASK-53) supports thresholds, DELTA/THETA scales,
multigroup defaults and all-ordinal DWLS. Ten independent default-lavaan
goldens and live grouped/scale tests gate fitting and inference; Demo meaning
gates cover parameter count, df, estimates and scaled tests. Demo SEs remain
convention observations, as recorded in the test ledger. Mixed, conditional
and categorical ML fitting remain explicit unsupported routes.

### Association-ML lab inference

The [association-ML audit](../../design/association-ml-inference.md) records the
existing correlation-target ML criterion and proposes exact all-ordinal Stage-1
sampling influence, observed sensitivity, covariance, spectral tests and
MI/release gates. Lab-only `estimate::frontier::association_ml_ij` and
`magmaanlab::association_ml_ij()` now compose the empirical Stage-1 rows with
observed correlation-score sensitivity and unweighted group target derivatives.
Full covariance retains saturated thresholds and their association cross-covariance.
The independent derivative, nonnormal case-weight, stratified delete-one,
exact-fit and identification-chart gates pass in `ordinal_ij_test.cpp`. The
lab wrapper gates covariance reconstruction and threshold cross-covariance.
Lab `association_ml_global_test` and `association_ml_nested_test` compose
All, SB and PEBA4 spectra from the local correlation ML residual metric and
exact Stage-1 Gamma, or observed H/IJ meat and an exact parameter restriction.
Explicit matrix reconstructions, zero-df/exact-fit reductions, one-restriction,
grouped and misspecified pseudo-true restriction gates are in `ordinal_ij_test.cpp`.
Moment nesting is refused. Ordinary covariance, global/nested tests and
MI/releases remain typed unavailable pending subcard 5. Lab
`association_ml_modification_indices` and `association_ml_score_tests` now
compose observed Schur nuisance projection/EPC with exact empirical projected
meat and one-df chi-square statistics. Fixed/absent association rows and
linear equality releases retain per-candidate typed refusals. Independent
Schur, augmented gradient/curvature, released-refit and grouped gates are in
`ordinal_ij_test.cpp`; lab reconstruction/refusals are in
`test_association_ml_scores.R`. No policy guard is removed. TASK-32.5
registers [decisions/08](../../../experiments/decisions/08-association-ml-policy/report.qmd)
for covariance/global/nested/MI calibration, with bounded smoke/pilot evidence
and provenance-checked per-cell execution; production and ordinary exposure
remain pending.

### Mixed DWLS ordinary policy

Complete mixed DWLS has exact empirical first-stage IJ covariance, a global
fit-function statistic with the All reference, and a nested fit-function
difference with the observed-Hessian/IJ law and All reference; SB/PEBA4 remain comparators.
`api::MixedDwlsPolicyFit` retains sampling rows and caches the IJ and observed
Hessian; `policy_inference`/`policy_nested` dispatch through both R packages.
Global Gamma is the cross-product of exact sampling rows, while OPG NACOV
continues to define fitting weights. These spectra differ in finite samples
and converge under regular latent-normal first-stage specification.
Moment-nested comparisons use the shared implied-moment embedding and tangent.
One/two-group delta/theta composition gates give limited validation; calibration
is pending. TASK-80 adds a registered mixed calibration lane to decisions/05,
with a full-grid local pricing pilot; production requires merger review.
Nested score and mixed ULS/WLS policy remain unsupported. Exact sampling
rows or complete raw observations are required.

### Mixed DWLS lavaan compatibility

Complete mixed DWLS supports `lavaan_compat = "WLSMV"` through the C++
`lavaan_inference_mixed_ordinal` / `lavaan_nested_mixed_ordinal` composers and
both R interfaces. Covariance uses expected-bread NACOV sandwich geometry
with original n_g/N fractions and the total N/(N-G) covariance multiplier.
The global scaled-shifted test uses the criterion evaluated with n_g-1 counts;
nested reporting uses lavaan's default Satorra-2000 delta map, scaled-shifted
reference and original n_g/N sandwich fractions. The ordinary exact-IJ policy
and retained estimates are unchanged.

Installed-lavaan 0.7.2 gates cover single/two groups, delta/theta, covariance,
intervals, summary and configural-to-equal-loadings nested reporting, including
both ordinary argument orders and cached/serialized bundles. Stage-1 moments
and NACOV agree before reporting is compared. Reporting at identical oracle
parameter points has strict 1e-5 relative lab gates; a checked-in single-group
mixed golden also gates the composer. Single-group delta additionally has
strict retained-fit gates for native fits. Under `preset = "lavaan-0.7.2"`,
seven of eight one/two-group delta/theta configural/restricted retained fits
pass these same reporting tolerances, including nested statistics, df, p, scale
and shift. The grouped theta configural case is path-unstable (TASK-94): both
fits must converge, search objectives must agree within 1e-8 times max(1, |F|),
and the canonical objective must be no worse than lavaan within 1e-12 times
max(1, |F_lavaan|). Its global and nested reporting gates use the identical
lavaan parameter point at the unchanged 1e-5 relative tolerance. Search
objectives agree with black-box lavaan evaluations on
identical Stage 1 within 1e-14; the small independently computed Stage-1
polyserial differences are kept separate from the search-unit gate.
Native grouped/theta endpoints retain the stopping differences recorded with
objective and canonical tangent-gradient evidence in the oracle observations ledger. No
oracle exemption or tolerance widening is used. This is numerical compatibility,
not a policy calibration claim. Missing-observation mixed stats and uncomposed
mixed bundles (including WLSM, ULSMV and plain DWLS/WLS) return `unsupported_model`;
penalties and failed verdicts retain their typed reasons.

### Slow numerical validation gates

The all-ordinal and mixed DWLS delete-one jackknife gates retain their original
assertions and tolerances. CTest's per-case `slow` label (TASK-87) covers these
and seven other numerical gates observed above about 60 seconds in Debug;
`cpp/tests/slow_tests.cmake` records the list. Debug checks and primary CI
exclude these nine cases; full optimized CTest and merge validation run them
all, preserving the complete test count.
