# magmaan TODO

Accepted implementation work, ordered by release. The
[roadmap](../architecture/roadmap.md) owns current capabilities/contracts;
the [test ledger](../validation/test_ledger.md) and linked studies own evidence.
Remove completed items after their durable record exists.

**Scope adopted 2026-09-30; release plan revised 2026-10-02:** single-level
ML, FIML and all-ordinal DWLS in their supported slices. **0.2.0 delivers the
ordinary API, lavaan-compatible fitting for simulations and the primary
inference workflows. magmaan's own optimizer and PSD work, mixed
continuous/ordered workflows and barrier hardening/inference follow in
0.3.0.** Secondary estimators retain correctness gates;
extensions need a concrete consumer or inexpensive reuse of primary work.
Noniterative work is indefinitely postponed; reopening requires an explicit
user scope decision. Existing noniterative APIs and regression gates remain.
Two-level SEM, SAM and composites
have **no scheduled expansion work**, including inside shared normalization,
start, preparation and inference programmes. Existing APIs/tests remain.

The [statistical scope](../scope.md) adopts population approximation targets
under joint observation sampling. General fixed-design inference under mean
misspecification and categorical conditional-moment expansion are
[banked](speculative.md#covariates-and-sampling); existing compatibility routes
retain their correctness gates.

The limited-information correlation-ML capability (formerly catML) is now
ordinal association ML within shared fitting composition; remaining
model-penalty work follows in 0.3.0. This does
not establish new ordinary-user defaults or promise inference for every
combination. See the
[composition contract](../architecture/roadmap.md#shared-fitting-composition).

[Speculative work](speculative.md) is a trigger register, not a second queue.
[Simulation](simulation.md) owns generator detail. The
[experiment index](../../experiments/README.md) owns study activity, evidence
limits and reopening conditions; active research does not automatically commit
the library to new methods/APIs. Paper-local runs, manuscripts and handoffs
belong to their independent projects.

Effort: **S** bounded fix/fixture/wrapper · **M** focused implementation and
validation · **L** cross-module semantics. Every task states its completion
check; evidence links own detailed protocols and historical results.

## Release plan

Revised 2026-10-02. One version number covers the C++ library and both R
packages, which ship together through the vendored core. The former 0.0.1
label (the C++ project version) is retired.

| Version | Content |
| --- | --- |
| **0.1.0, shipped 2026-10-01** | Simulation prerelease of `magmaan` and `magmaanlab` (local tag `v0.1.0`); API hardening and its gates are recorded in the [roadmap](../architecture/roadmap.md#r-bindings-and-public-namespace-transition) and package NEWS |
| **0.2.0, current** | The adopted ordinary API; lavaan-compatible fitting through `options`, so simulations can rely on lavaan's fitting while magmaan's own optimizer work waits; ordinary-policy inference for ML, FIML and all-ordinal DWLS; MI/release-score completion across weights, including two-stage |
| **0.3.0, next** | magmaan's own fitting reliability (starts, optimization, convergence, PSD finalization, stress and normalization); mixed continuous/ordered workflows; barrier hardening and inference; association-ML MI |
| **After 0.3.0** | Review what remains and bank items without a consumer in the [speculative register](speculative.md) |

Sphere-chart development remains outside 0.2.0; the minimal unrestricted
starts/search/audit programme and chart failure handling are high-priority
[0.3.0 work](#optimization-and-convergence). Broader promotion stays in the
[trigger register](speculative.md#sphere-chart-development). Existing PSD and
barrier entry points, inference refusals and regression gates are unchanged in
0.2.0; PSD inference keeps its current boundary contract. No milestone label
promotes a new ordinary-user estimator or inference default without recorded
evidence.

### 0.2.0 exit criteria

0.2.0 is released when every criterion holds. A component can meet a criterion
by returning a typed unavailable reason; it never meets one by returning a
result under an unstated convention.

1. **Ordinary API.** `magmaan_model()` and `magmaan(model, data, estimator,
   covariance, inference, options)` work as
   [designed](../design/r-interface-vision.md#ordinary-api), including the
   `fixed.x`, `missing`, `cluster` and `meanstructure` removals with migration
   messages, and the checks of the [API tasks](#ordinary-api) pass.
2. **lavaan-compatible fitting.** `options = list(preset = "lavaan-0.7.2")`
   fits complete-data ML with linear equality constraints (shared labels,
   `group.equal`, `==` rows), FIML and all-ordinal DWLS, single- and
   multi-group. Pinned fixtures match lavaan's starts, search coordinates,
   final gradients and verdicts, and installed-lavaan R comparisons pass on a
   named simulation-model set. Nonzero bounds, nonlinear constraints and other
   routes error.
3. **Primary inference.** For ML, FIML and all-ordinal DWLS, single- and
   multi-group, the ordinary policy returns parameter covariance, global tests
   and nested tests, or a typed unavailable reason per component. Each computed
   component has a written recipe, an independent or convention-matched gate,
   its exact reductions where the geometry coincides (on complete data the
   FIML covariance and LR statistic equal the ML policy's) and
   target-regime calibration evidence: existing studies or one frozen
   confirmatory run per estimator.
4. **Open ML/FIML validity questions settled.** The Wald coverage shortfall is
   explained (point bias versus underestimated variance) and either fixed or
   documented, and the small-sample LR reporting decision is recorded.
5. **MI/release matrix.** Every estimator × weight cell of the
   [completion matrix](#mi-and-release-score-completion-020) is gated in C++
   and R or rejected with a typed reason, and no cell silently combines two
   weight recipes. Two-stage MI/release use Stage-1 influence for every
   Stage-2 weight.
6. **Capability inventory.** One published table records, per estimator,
   covariance domain and inference component, whether it is validated, has
   limited validation, is unsupported or is inapplicable; C++, lab and
   ordinary-package behavior agree with it. The table defines "supported" for
   0.2.0.
7. **Release mechanics.** The CMake project and both DESCRIPTION files carry
   0.2.0 with matching NEWS. `just check`, the `parity` label and `R CMD check`
   for both packages pass, `main` and the release tag are pushed, and CI is
   green.

## 0.2.0: ordinary API, lavaan-compatible fitting and primary inference

### Ordinary API

- [x] **M/L — implement the adopted ordinary API surface** (2026-10-02).
  `magmaan_model()` with frozen group/category schema and zero-row prototypes;
  `magmaan(model, data, estimator, covariance, inference, options)` with
  always-on means, random X, `covariance = barrier(lambda)` (session message,
  printed status, C++ `penalized` reason), `options$start`, typed
  `magmaan_schema_error` failures and migration errors for removed arguments.
  Likelihood-ratio refits keep the anchor's group order. Fits run through
  `fit_model()` on the constructed specification. See the
  [implementation record](../design/r-interface-vision.md#implementation-and-remaining-decisions).

- [ ] **M — fit constructed ordinary models through native prepared
  handles.** `magmaan_model()` should own a prepared model so repeated fits
  skip structural preparation (ordinal DWLS 9.1 → 3.5 ms in the 2026-10-01
  timing). First close the parity gaps of the prepared `estimate()` path
  against `fit_model()`: the layered start for continuous ML and GLS (it uses
  FABIN3), fitting options and presets, fit-time start tables, ML2S, and the
  recorded refit route. Native handles stay process-local and are rebuilt from
  the portable model on workers. **Check:** fresh/prepared parity in partable,
  estimates, objective, diagnostics and routes for every ordinary estimator
  and covariance policy; zero repeated structural-preparation calls;
  changed-data starts/thresholds; save/reload and worker reconstruction; and
  separately timed construction, data preparation, fit and inference.

- [x] **S/M — remove the ordinary fixed-x option under the adopted API**
  (2026-10-02). Ordinary construction uses the joint random-X model and
  rejects lab specifications with `fixed_x = TRUE` and observed covariates.
  ML structural estimates match the fixed-x fit; GLS estimates of an
  overidentified regression differ, as the
  [scope](../scope.md#ordinary-fixed-x-decision) records. Regressions on
  observed covariates get the ML policy's inference. General fixed-design
  inference remains
  [banked](speculative.md#fixed-design-inference-under-mean-misspecification).

### lavaan-compatible fitting

Consumer: simulation studies that need lavaan-identical fitting while
magmaan's own optimizer work waits for 0.3.0. The `lavaan-0.7.2` preset
already pins starts, nlminb/PORT controls and scaling, the four-attempt retry
sequence and lavaan's acceptance rule for ordinary complete continuous ML
without equality constraints; under it the selected rule decides `converged`
and magmaan's own check is reported beside it (2026-10-01). Inference stays on
magmaan's policy, so lavaan-identical fitting does not imply lavaan-identical
standard errors or tests. Retain effective controls, every attempt and
separate native diagnostics; `options$convergence = "newton"` keeps magmaan's
own verdict. Nonzero bounds, nonlinear constraints (lavaan's NLMINB.CONSTR
augmented Lagrangian) and PSD, pairwise and two-level routes remain explicit
errors under the preset. Every check compares starts, search, final
gradients, soft failures and retries, not just easy estimates.

- [ ] **M — fit linear equality constraints in lavaan's coordinates.** lavaan
  0.7.2's default (`ceq.simple = FALSE`) turns labels and `group.equal` into
  `==` rows and optimizes in K-reduced coordinates, K from a QR decomposition
  of the constraint Jacobian; starts and parscale are projected through K and
  bounds become ±Inf. magmaan's affine map θ = θ0 + Kα exists with its own
  basis (0/1 for pure merges, per-component orthonormal otherwise), and PORT
  is not basis invariant, so endpoint parity needs lavaan's exact QR basis,
  including sign and column order. Pure-merge `ceq.simple = TRUE` is not
  lavaan's default and can wait. **Check:** pinned fixtures for shared labels,
  `group.equal` loadings/intercepts and `==` rows, matching starts,
  coordinates, final gradients, constrained retries and verdicts, plus
  installed-version comparisons. Retry parity so far: a ×100 rescale matches lavaan's
  standardized retry exactly; at ×1000 and ×10⁵ starts, coordinates and
  verdicts match, but endpoints depend on floating-point paths (fixture
  `fitting/lavaan_0_7_2.json`).

- [ ] **M — fit FIML under the lavaan preset.** lavaan builds FIML starts from
  its EM H1 moments; feed magmaan's EM H1 to the pinned start code, evaluate the
  FIML objective and acceptance gradient in lavaan's units, and match the
  source of the standardized-retry scale. **Check:** MCAR/MAR fixtures and
  live comparisons, single- and multi-group, retaining every attempt and the
  native diagnostics.

- [ ] **M — fit all-ordinal DWLS under the lavaan preset.** Pin lavaan's
  starts (sample thresholds, unit delta scales, FABIN3 on the polychoric
  matrix) and evaluate the DWLS objective, group weighting and acceptance
  gradient in lavaan's units. **Check:** delta/theta, grouped and invariance
  fixtures plus live comparisons with lavaan's WLSMV fits.

### Primary inference workflows

0.2.0 completes the ordinary policy for ML, FIML and all-ordinal DWLS.
Ordinary SEs are not automatically valid at singular PSD endpoints; the policy
keeps its current boundary contract (computed, assuming an interior
population), and further PSD inference work follows in 0.3.0. The adopted
ordinary API exposes barrier fitting experimentally as
`covariance = barrier(lambda)` and reports its inference as unavailable.

- [ ] **S — expose effective pEBA block counts.** A requested pEBA-4 is clamped
  to the test df and can coincide with scaled-shifted at df=1 while keeping its
  requested label. Decide an explicit diagnostic/reporting convention.
  **Check:** low-df rows identify the effective method without changing computed
  tails or silently mislabelling simulation cells. See the
  [FMG example](../../r-package/examples/fmg.R).

#### ML and FIML

- [ ] **M — validate remaining likelihood-score component contracts in the primary sampling scope.**
  Retain uncentered score second moments as the baseline for ML/FIML
  parameter sandwiches and global/nested score-test calibration under joint
  population sampling. The
  [fixed-allocation covariance formula](../scope.md#group-allocation-and-likelihood-score-covariance)
  is settled; its ordinary-policy extension is
  [banked](speculative.md#fixed-design-inference-under-mean-misspecification).
  Keep raw likelihood scores distinct from centered moment influences and
  saturated from structured evaluation points. Preserve the observed test score.
  **Check:** stationary-fit equivalence and covariance/projection contracts for
  missingness, constraints and PSD boundaries; compare centered/raw calibration
  only for a component that meets the
  [bank's reopening trigger](speculative.md#likelihood-score-centering-alternatives),
  with size, coverage, matched-null power and numerical failures reported.
  Separate convention-matched lavaan parity from
  ordinary-policy evidence before changing defaults. Reopening established
  moment-covariance recipes is
  [banked](speculative.md#moment-covariance-centering-alternatives); multiplier
  promotion remains under the banked robust-score-flip entry.
  The preregistered [score-centering decision study](../../experiments/decisions/03-score-centering/report.qmd)
  compares paired meats on the same fits. Its 3,600-draw pilot and independent
  8,000-draw fixed-group confirmation separate a sampling-law identity from
  finite-sample calibration. The latter centered score reference fails at N=80
  (6.2% rejection, Wilson interval 5.2–7.3%); this does not reopen the settled
  fixed-allocation covariance formula or select raw for that conditional target.
  Independent ML confirmation (32,000 draws) retains raw in the tested regular
  complete-data families: no registered one-point size-error benefit, at most
  0.75-point rejection shifts, and identical empirical matched-null power.
  Prospective FIML confirmation adds 32,000 fresh MCAR/MAR datasets, separately
  under expected and observed sensitivity: no qualifying centering benefit,
  matched-null power differences at most 0.10 points, and stationary covariance
  agreement. FIML adopted observed sensitivity on 2026-10-02 (see the FIML
  policy item below); its finite-sample conservatism is
  [banked](speculative.md#fiml-observed-sensitivity-finite-sample-correction). The centering comparison is now banked in place, with
  no queued extension. Check excluded constraint/boundary component contracts
  independently; penalty-specific inference follows in 0.3.0.
  Missingness patterns are not sampling groups.

- [ ] **M — investigate ML/FIML sandwich and Wald coverage gaps.**
  The [centering confirmation](../../experiments/decisions/03-score-centering/report.qmd)
  finds 90.3% coverage for the known loading target in nested skewed N=80 nulls
  (95% Wilson interval 88.9–91.5%); raw and centered covariances agree numerically.
  **Check:** point-estimate bias, empirical estimator variance versus reported
  sandwich variance, and larger-N/normal controls with independent uncertainty.
  Verify the native covariance construction before selecting any finite-sample
  interval correction; score-test nominal size does not validate Wald coverage.
  The same study's prospective FIML MCAR nested N=80 null coverage is 92.35%
  (Wilson interval 91.10–93.44%), failing the registered coverage condition.
  Retain this normal incomplete-data control in covariance/interval validation;
  raw and centered arms agree, so this is not evidence of a centering defect.

- [ ] **M — decide how the policy reports small-sample LR tests.** The calibration
  battery finds substantial high-df over-rejection for LR-SB/PEBA4 while score
  arms fare better; heterogeneous-spectrum global calibration remains unresolved.
  Decide co-primary reporting, leading with score, or an evidence-backed caveat.
  **Check:** held-out size/power/failure comparisons and explicit reporting
  criteria. Evidence: the calibration battery; do not silently change defaults.

- [ ] **M — reconcile nested restrictions and reusable inference.** Fixed and
  dropped paths embed into larger-model slots (completed 2026-10-01; see the
  roadmap's prepared-inference contract). Diagnose
  the structural-path constant disagreement between `score_components(H1=)`
  and `policy_nested()`/`lavTestScore()`. Support mean-aware metric-to-scalar nesting with released
  latent means. **Check:** Kline Worland restrictions, residual-covariance
  controls, cross-group equalities and independent restriction maps; assess
  larger-model misspecification calibration. See
  [scores](../../r-package/examples/scores.R) and
  [inference reuse](../../r-package/examples/inference_reuse.R).

- [ ] **M — move the ML nested policy to observed geometry.** Adopted
  2026-10-02 in the [interface vision](../design/r-interface-vision.md):
  expected information gives an inconsistent reference law when the larger
  model is misspecified, the usual invariance case. In `api::policy_nested_ml`,
  the score uses observed sensitivity at the restricted fit with the expected
  metric (the FIML nested geometry), and the Satorra-2000 spectrum uses
  observed information at the larger model; both statistics are unchanged.
  The global ML test keeps expected information. No ordinary option: lavaan's
  expected-information compositions stay in the lab with their parity gates.
  **Check:** finite-difference and identity gates for the observed
  projections, unchanged statistics, PSD-boundary and fixed/dropped-path
  embeddings, and a component comparison with lavaan fits using
  `information = "observed"`. Exact equality is not assumed: lavaan keeps its
  normal-theory weight in the Satorra-2000 projector, and `lavTestScore()`
  uses observed information for the metric too. One frozen calibration run
  with correct and misspecified larger models (expected versus observed; size
  and size-adjusted power) precedes release.

- [ ] **M — compose the FIML policy.** The primitives exist and are mostly
  lavaan gated: the observed-information sandwich with casewise scores (the
  MLR bread and meat), global and nested score through
  `global_score_components` / `nested_score_components`, and the FIML LR with
  its UGamma spectrum. Add `policy_inference_fiml` / `policy_nested_fiml`
  modelled on the ML policy, a FIML counterpart of the cached
  `NTMLFit`/`NTMLHypothesis` contexts for evaluation-point-specific influence
  reuse, R dispatch, and typed per-component reasons in place of today's
  `unsupported_model` refusal. The geometry was adopted on 2026-10-02 from
  [research/44](../../experiments/research/banked/44-fiml-global-tests/report.qmd)
  and is recorded in the [interface vision](../design/r-interface-vision.md):
  observed-H0 sensitivity with the expected metric for global and nested
  score tests, and the saturated observed-H1 metric for the LR spectrum.
  Reduction map on complete data: covariance and LR statistic exact; LR
  spectrum and score asymptotic only. No frozen FIML confirmatory run is
  queued for 0.2.0; the capability inventory records the evidence limits (no
  MAR cells in the latent-model sensitivity panel, nested evidence one df-1
  normal family, conservatism as df grows). LR reporting follows the
  small-sample LR decision. **Check:** the reductions above, the adopted
  geometry, grouped and missing-pattern gates, and typed reasons; missing
  inference never refuses a fit.

- [ ] **M — pin FIML robust conventions before claiming parity.** Resolve `sb_ml`
  bread/meat/H1 choices and convention dispatch; distinguish Yuan-Bentler
  variants from SB labels and retain FMG missing-data oracle limitations.
  **Check:** convention-matched values and target-regime/grouped calibration.
  Evidence: [definitions and matrix fingerprints](../../experiments/replications/08-savalei-falk-2014-test-conventions/report.qmd)
  and [calibration policy](../validation/calibration-parity.md). These
  labels concern lavaan parity; they do not block the FIML policy.

#### All-ordinal DWLS

Mixed continuous/ordered completion is assigned to 0.3.0. Shared fixes
required by an all-ordinal primary workflow remain current work.

- [ ] **S/M — compose the DWLS policy covariance and global test.** Use the
  IJ covariance `robust_ordinal_ij` (observed bread, Stage-1 threshold and
  polychoric influence, estimated-weight term) and the n·F global statistic
  with the `robust_ordinal` UGamma spectrum, calibrated with SB and PEBA4.
  Weight influence vanishes under the global null, so the score and
  fit-function statistics coincide and are reported once. Lift the ML-only
  gates in `r-package/R/scores.R`, and route lab `vcov()` for categorical fits
  deliberately (today it uses the fixed-weight `robust_ordinal`). **Check:**
  jackknife, multi-group and theta gates for the IJ covariance (every current
  IJ test is single-group delta), the fixed-weight reduction and lavaan WLSMV
  agreement of the shared pieces.

- [ ] **S/M — compose the DWLS nested likelihood-ratio-type test.**
  Fixed-weight Satorra-2000 (`lr_test_satorra2000_ordinal`, gated against
  lavaan and Mplus DIFFTEST) composes with SB/PEBA4 directly. The
  estimated-weight law (`ordinal_dwls_profile_lrt`) kept its size under
  misspecification in
  [evidence 13](../../experiments/research/evidence/13-ordinal-dwls-profile-lrt/)
  (about 4.2% where the fixed-weight test reached 10.5%), but needs a
  reduction gate to Satorra-2000 at exact fit, a df versus `spectrum_size`
  decision and an evaluation-point choice: Satorra-2000 uses the H1 point,
  while the ML route moved to the common null on 2026-10-01. **Check:**
  reductions, grouped delta/theta ladders and unit invariance.

- [ ] **M — calibrate the DWLS policy.** No in-repo study covers DWLS SB versus
  PEBA4 size, global or nested; [evidence 12](../../experiments/research/evidence/12-misspec-robust-se/)
  covers IJ coverage for one parameter under misspecification. **Check:** one
  frozen confirmatory run across ordinal regimes (categories, threshold skew,
  N, groups), including misspecified larger models for nested tests,
  reporting size, coverage and failures before the recipe becomes the policy.

- [ ] **L — derive a nested DWLS score test, or defer it.** No joint
  least-squares nested score statistic exists; `score_tests_robust_joint` is
  ML-only. Specify sensitivity, metric, nuisance projection, weight influence
  and group normalization, then derive and calibrate; alternative score
  weights need a recorded decision study before adoption. Until then the DWLS
  policy reports the nested score as unavailable with a typed reason, which
  meets the 0.2.0 exit criterion; move this item to 0.3.0 if it is not needed
  sooner.

- [ ] **S — verify post-fit partable reconstruction for ordinal inference.**
  Post-fit `robust_ordinal`, the IJ covariance and Satorra-2000 re-prepare the
  partable without `row_user`, while the fit passes it (`cpp/src/estimate/ordinal.cpp`,
  the post-fit preparations versus the fit call). Models with explicit `~~`
  rows may then fail or differ. **Check:** explicit and implicit spellings of
  the same model give identical inference, or the defect is fixed.

- [ ] **M — finish primary reusable inference ownership.** Extend observed-bread
  covariance/score, delta-nesting and categorical influence adapters where
  independently validated. **Check:** centering, finite-sample/group scaling and
  retained geometry; equal statistics do not establish equal spectra. See
  inference reuse and [workspace contract](../design/ordinal-snlls-gamma-architecture.md).

With `missing` removed from the ordinary call, pairwise DWLS stays a lab route;
the capability inventory lists its policy inference as unsupported.

### MI and release-score completion (0.2.0)

Adopted 2026-10-01 after assigning mixed continuous/ordered completion to
0.3.0 and indefinitely postponing noniterative work. This slice covers
one-parameter modification indices and one-at-a-time equality releases in the
lab/C++ surface. Global/nested ordinary-policy work remains above; joint
multi-constraint tests retain their separately documented coverage. Existing
mixed/noniterative APIs keep their regression gates.

Coverage separates the fitting discrepancy/weight, score-information metric,
sandwich sensitivity, score/moment covariance and estimation of the weight.
Identity, normal-theory, diagonal/full empirical, DLS-mixture and caller-supplied
weights are distinct recipes. Supplied W is fixed unless its generating recipe
and influence are explicitly supplied. A computed number is not sufficient:
every applicable combination needs a stated sampling contract and a gate;
unsupported or inapplicable choices must be explicit and never silently ignored.
Independent information/bread or alternative score-weight choices need a derived
projection/reference law before exposure. This adds no ordinary-user default.

| Remaining family | Existing MI/release basis | Weight/covariance completion |
| --- | --- | --- |
| Complete-data ML | Expected/observed information; robust core and R paths | Gate matching information/bread, structured/unstructured supported NT covariance, empirical/Browne and caller-Gamma choices; reject unsupported moment-source combinations |
| Direct FIML | Analytic observed MI/release, robust core and R dispatch; expected geometry gates identification | Observed statistic/bread and observed-pattern casewise meat; expected-statistic, alternative covariance/moment and second-stage-weight choices explicitly rejected; broader convention/calibration work remains above |
| Continuous ULS/GLS/WLS | Shared moment-quadratic MI; fixed- and estimated-weight robust primitives | Identity, NT, diagonal/full empirical, DLS(a) and supplied block W: R honors empirical/model-implied covariance with explicit fitting W; remaining work retains recipe/a and weight influence, audits sensitivity/nuisance projection, and exposes caller-Gamma adapters |
| All-ordinal ULS/DWLS/WLS | Ordinary and robust threshold/association MI; estimated-weight DWLS/WLS path | Gate identity, diagonal/full NACOV, retained Stage-2 DLS and supplied weights with delta/theta and group conventions; shared relative rank is implemented, provenance/adapters remain |
| Prepared all-ordinal association ML | Ordinary/PSD fitting through the shared association-target contract; LS MI is not an ML-target score contract | Typed rejection in 0.2.0; the contract is [0.3.0 work](#association-ml-inference) |
| Two-stage/ML2S | R NT-ML MI is a naive Stage-2 comparator; retained Stage-1 and weighted inference primitives | Corrected MI/release for NT, ULS, DWLS, ADF and DLS Stage-2 recipes, using Stage-1 influence and the applicable estimated-weight term; preserve actual ML versus quadratic discrepancy provenance |

The DLS/custom-weight rows concern bounded reuse of retained weighted primitives,
not a general DLS research programme. Pairwise moment sources retain their MCAR
scope and need their own covariance law; direct composite-likelihood, two-level
SEM, SAM, mixed-data and noniterative expansion are outside this slice.
Automatic absent-row enumeration covers cross-loadings and covariances;
structural-path enumeration remains a separate model-builder contract.

The ordinary package does not expose MI in this release; the
[interface vision](../design/r-interface-vision.md) defers policy MI.

- [ ] **S — finish the weight-recipe fix in R.** The C++ guard landed
  2026-10-02 (roadmap: estimated-weight recipe guard): continuous IJ consumers
  refuse a fitting weight that differs from their recipe's rebuild, ordinal
  IJ and DWLS profile paths refuse NT/DLS/supplied weights, and
  `continuous_ls_ij_mode_for` maps a recorded recipe to its mode. Remaining
  R glue: read `fit$composition$weight`, `dls_a` and the supplied flag through
  that resolver in every `continuous_ij_mode(estimator)` caller (MI/release,
  robust IJ SEs, profile LRTs, RBM, estimated-weight residuals); use `fit$W`
  instead of requiring `weight =`; add `UnsupportedInference` to the R error
  kind names; and remove or validate the `ij_weight` override (pending
  decision). Verify whether association-ML ordinal fits can reach the LS
  ordinal MI worker and reject them until their 0.3.0 contract. **Check:**
  testthat cases for DWLS, DLS (non-default a), supplied-W and ordinal NT/DLS
  fits through each R entry point, with typed errors where refused.

- [ ] **M/L — complete weighted MI/release provenance and adapters.** Cover the
  retained continuous and all-ordinal weight recipes in the matrix, with stored
  fitting W or explicit supplied W, Gamma/NACOV source, recipe/a and fixed versus
  estimated-weight influence. Derive consistent sensitivity and nuisance
  projection for observed/estimated-weight GMM score variants before exposing
  them; the existing expected-metric sweep alone does not establish that regime.
  Association-ML MI is a 0.3.0 contract; reject it until then. Storing the
  recipe, a and W on fits, carrying them into the IJ mode and exposing
  `ij_weight`, `dls_a` and caller Gamma in R is wiring; the sensitivity and
  nuisance projection for observed/estimated-weight score variants is the
  genuine derivation.
  Audit every bread/information/covariance argument and expose applicable caller-
  Gamma paths through thin R adapters. **Check:** independent score, sensitivity,
  meat and weight-influence assembly; recipe endpoint reductions, retained-data
  versus supplied-data agreement, documented unavailable cells and no ignored
  options. No numerical recipe/default changes without evidence.

- [ ] **S — expose two-stage MI/release in R.** The C++ tier landed
  2026-10-02 (roadmap: two-stage MI and equality-release score tests):
  `{modification_indices,score_tests}_ml2s` report the naive Stage-2 statistic
  and its Stage-1-aware scaling for NT, ULS, DWLS, ADF and DLS, fixed or
  estimated weight, gated by exact complete-data reductions in one and two
  groups. Remaining: dispatch ML2S fits from `inference_modification_indices`
  and the robust score wrappers using the fit's Stage-1 object, `stage2_weight`
  and `stage2_dls_a` (raw data, pack and H1 for the estimated weight); keep the
  naive column labelled; check whether lavaan's `modindices()` on a
  `missing = "two.stage"` fit reproduces the naive NT statistic and freeze a
  fixture if so. **Check:** R reductions to complete-data ML/LS, MCAR controls
  and typed errors for observed information and missing mean structure.

- [ ] **M — close the MI/release estimator-by-weight validation matrix.** Gate
  the implemented cells above in C++ and R, including means, unequal groups,
  constraints, absent/fixed candidates and standardized EPCs. **Check:**
  independent df=1 score/Schur/sandwich reconstruction; Gamma = W^-1 reduction;
  correct weight-scale transport of ordinary MI/EPC and robust-statistic
  invariance; unit/rank controls; estimated-weight case perturbations; and
  convention-matched lavaan fixtures where applicable. Target-matched
  misspecified nulls/local alternatives need separate calibration evidence;
  an unproved generating-model restriction is not a pseudo-null. Keep sampling
  groups distinct from missingness patterns and record unresolved centering
  choices rather than silently adopting them. Each R-visible cell must agree
  with the corresponding core contract or report its unsupported reason.
  Known fixture gaps: no complete-data MLR oracle, no GLS/WLS ordinary MI
  fixture, ordinal ULS/WLS only reduction and rank tests, and loose absolute
  tolerances on the ordinary goldens (0.2 FIML, 0.1 DWLS, 5e-2 ULS) to tighten
  or justify.

### Release readiness

- [ ] **S/M — inventory validated primary capabilities.** Record model/data
  slice, domain, penalty, algorithm, API tier and evidence for estimation,
  verdict/admissibility, covariance, global/nested tests and intervals separately.
  Use validated, limited-validation, unsupported and inapplicable states;
  planned slices link here. **Check:** C++/R owners and ordinary-policy exposure
  agree. The table is 0.2.0 exit criterion 6. Keep one inventory; secondary breadth is consumer-gated. See
  [development priorities](../architecture/roadmap.md#estimator-development-priorities).

- [ ] **M — broaden primary CI checks.** Use appropriate `R CMD check` instead
  of hand-picked R tests; add scheduled sanitizers/optional parity and an
  interpretable coverage artifact. **Check:** clean-source portable installs
  and mount-independent default tests, avoiding unexplained percentage gates.
  See [local hardening](../validation/local_hardening.md).

- [ ] **S — align versions and publish.** Set the CMake project version
  (currently 0.0.1) and both DESCRIPTION files to 0.2.0 with NEWS entries;
  push `main` and the tags. As of 2026-10-02 `main` is 74 commits ahead of
  `origin`, the `v0.1.0` tag and its commit are local only, and CI last ran on
  2026-09-28. **Check:** a clean install from the pushed tag and green CI.

## Unscheduled: interfaces, composition and maintenance

Work here proceeds when convenient or when a release item needs it; none of it
gates a release unless an exit criterion names it.

### EQS language extension

The initial EQS model-section frontend is complete; its implemented scope and
validation limits are in the [EQS contract](../grammar/eqs.md). The accepted
extension target is the full documented EQS 6 linear SEM model language, with
original parameter meanings preserved in the Jöreskog/LISREL model contract.
Full EQS job execution, estimator/default emulation and numerical parity are
outside this target. Syntax/schema recognition does not promote fitting or
inference beyond [scope](../scope.md). Work order is C++ first, then validated
increments in the freely evolving `magmaanlab`; `r-magmaan` is deferred to the
explicit later task below. The [implementation sequence](../grammar/eqs.md#implementation-sequence-planned)
owns dependency order, affected interfaces and completion gates. Planning is
complete; the source milestones remain unimplemented.

- [x] **S — extract and review the model-language sources.** The maintained
  [source inventory](../grammar/eqs_source_inventory.md) identifies exact manual
  pages, documented versus derived rules, semantic fixture families and a small
  runtime probe list. Well-specified rules can proceed without an EQS install;
  ambiguous behavior must remain explicit until independently resolved.
- [ ] **S — add a resolved EQS document and explicit-input resolution (C++ 1).** Resolve numeric
  identities and aliases before ranges/parameter references; cover documented
  IDs, duplicate-predictor recovery and line conventions. Preserve source spans,
  original names and rebuild behavior for aliases invalid in lavaan syntax.
  Retain roles, declaration order and parameter identities before lowering.
- [ ] **M — complete general linear-equation lowering (C++ 2).** Indicator
  variables participating in structural regressions, nonunit/free error paths
  and general independent-error covariances need an exact shared builder/matrix
  contract. Preserve original estimands, references, starts and derivatives
  through any augmentation. **Check:** direct linear-system moments, parameter
  perturbations/derivatives and unchanged existing lavaan contracts; equivalent
  covariance alone cannot validate a reparameterization.
- [ ] **M — add MODEL shorthand expansion (C++ 3).** Cover Cartesian ON, combined
  equations, within-/cross-list COV, paired PCOV, independent-moment families,
  distinct bare-VAR defaults, generated residuals and identification fixes.
  Keep explicit-equation semantics separate from shorthand defaults; gate
  RELIABILITY separately on resolution of its contradictory manual example.
- [ ] **M — add V999 means, model segments and minimal schema (C++ 4).** Map
  constant paths to existing Nu/Alpha semantics without an observed constant
  column; retain unequal segment-specific models, hints and restrictions.
  Do not use blind group replication. Classify schema declarations separately
  from data/execution instructions and validate supplied group/column mappings.
- [ ] **M — adapt parameter restrictions and SET (C++ 5).** Resolve directed,
  diagonal/symmetric and group-qualified references to existing equality/linear
  constraint machinery; represent explicit bounds and fail unsupported fit
  routes. SET needs EQS dependent/independent pattern classification and
  forced-free exceptions rather than broad lavaan equality-family substitution.
  Simple exact parameter bounds reuse supported shared bounds; general
  inequalities remain explicit where fitting cannot enforce them. This task
  adds no optimizer or active-bound inference contract.
- [ ] **S — close language coverage and documented ambiguities (C++ 6).** Obtain targeted setup
  outputs or independent authoritative evidence for marker ordering, alias
  edge cases, repeated declarations, observed selection/variance repair, SET
  exceptions and the contradictory RELIABILITY expansion. The source inventory
  owns the probe details. HLM DEFINE lacks a complete production in this manual
  and needs additional evidence before any scoped adapter work. Gate growth
  examples through ordinary rows and publish inventory-keyed coverage.
- [ ] **M — expose validated EQS increments in magmaanlab.** Follow each C++
  gate with thin adapter/export changes. Preserve original source language,
  column/alias mappings, schema and construction settings; rebuild via C++ EQS
  or a lossless portable triple rather than requiring a row-only lavaan string.
  Reject overrides that contradict explicit EQS choices. **Check:** partable,
  prepared/fresh, refit/rebuild and worker save/reload parity for rows, starts,
  restrictions and implied moments; appropriate live-lavaan and portable-install
  gates. Run `just vendor` after canonical C++ changes, never edit mirrors.
- [ ] **M — integrate EQS into r-magmaan later.** Start after C++/lab language
  and round-trip gates pass. Accept validated EQS specifications through the
  ordinary model constructor `magmaan_model()`, retain schema/parameter identities and
  use the existing ordinary inference policy. **Check:** repeated simulation
  fits, worker reconstruction, prepared parity and explicit unsupported-policy
  results. Do not expand fitting/inference scope or change ordinary defaults.
  No ordinary-package code is part of the current C++/lab milestones.

Every parser milestone changes the normative EBNF first and adds independent
semantic expectations plus convention-matched lavaan/implied-moment gates where
applicable. None requires matching EQS's numerical fitting output.

### Shared fitting composition

The moment-target foundation, the all-ordinal association-ML contract,
non-mixed PSD/barrier fitting and shared fixed weights are complete; their
contracts and validation are in the
[roadmap](../architecture/roadmap.md#implemented-composition-2026-10-01).
Remaining order: cross-route dispatcher metadata → mixed and penalty
completion (0.3.0) → sampling/inference gates (0.2.0 primary inference and
0.3.0 barriers). No task here removes shared primitives or promotes an
ordinary-user default.

- [ ] **M — consolidate dispatch and composition metadata across routes.**
  Ordinal association ML (ordinary and PSD) already dispatches through C++
  `api::fit` and `fit_model(estimator = "ML")` with full provenance, and the
  catML wrappers are retired. Extend the same moment source / target /
  discrepancy / domain / penalty / algorithm record to every retained route;
  consolidate enums, R validation/defaults, stored labels and post-fit
  allowlists. Decide whether the C++ `api::EstimatorSpec` facade gains a
  covariance option. **Check:** staged/convenience/refit equivalence,
  metadata/errors, unchanged numerics and thin R wrappers. Mixed/polyserial
  ML's moments/mean/scale contract follows in 0.3.0; removed research APIs are
  not migration targets.

- [ ] **M — finish primary prepared ownership.** Reuse categorical stage-one
  score ingredients when building Gamma; make FIML H1 reuse/attachment and
  estimate-only behavior explicit. Preserve caller results until a compatibility
  path exists. **Check:** separate model/data/weight/fit timing and equivalence.
  Specialized ML2S, two-level, SAM and FC-SEM migration is deferred. See
  [preparation contract](../design/r-model-preparation.md).

- [ ] **S — retain deletion provenance in lab fits.** Store listwise counts per
  group/reason for all entry points. **Check:** staged/convenience agreement and
  estimator-specific unsupported deletion paths. See the R interface vision.

- [ ] **M — finish primary invariance adapters.** Review redundant manual scalar
  mean freeing in `continuous_invariance()`; mixed-model release follows in
  0.3.0 and is exposed only
  after validating its threshold/scale map. **Check:** explicit user means,
  metric-to-scalar nesting and theta/released-delta boundaries. Broader Mplus/
  mixed-pairwise compatibility is deferred. See workspace contract.

- [ ] **M — separate research dependencies from core.** Untangle ordinal
  h-score/pairwise builders before retiering remaining research `data/` headers.
  Retier interleaved misspecification/profile surfaces in one deliberate pass
  with forwarding shims, R glue and vendor updates. **Check:** no upward core
  dependencies and unchanged supported calls. Cosmetic constraint/start moves
  are [deferred](speculative.md#namespace-and-header-housekeeping).

### Ordinal DWLS Gamma influence performance

- [ ] **M — profile cell-score cost and repeated post-fit reuse.** Measure
  multi-category work before another algorithm change; retain reusable influence
  and cache ownership. **Check:** equivalent ingredients, setup/fit/reporting
  costs and peak memory on complete and supported mixed/missing primary slices.
  See [benchmark guide](../../benchmarks/README.md) and workspace contract.

### Ordinal weight storage and workspace cleanup

- [ ] **M — retain diagonal DWLS storage.** Give `W_dwls` a diagonal type;
  remove unnecessary `Ws`/`factors` from `build_joint_profiled_workspace`.
  **Check:** staged/direct bounded and SNLLS agreement without unnecessary dense
  materialization or WLS inversion. See workspace contract.

- [ ] **M — build only Gamma diagonals for fit-only DWLS.** Complete the direct
  `ordinal_stats_from_integer_data` path and reconcile staged/direct `active_set`
  audit shapes. **Check:** diagonal/full parity and separately measured setup/
  memory. A reduced-materialization enum does not establish reduced inference.

### Continuous moment-quadratic weight follow-ups

- [ ] **M — preserve DWLS diagonals through R.** Stop transporting them as
  Dense. **Check:** unchanged fit/covariance with reduced storage. Consolidate
  `WhitenFactor`/`BlockWeight` only after documenting weight-versus-factor and
  numeric-policy contracts. See the roadmap's continuous-weight implementation.

### Validation and maintenance

- [ ] **S — revalidate the retained Bell alternative-CFA controls.** The
  consolidated reliability showcase reproduces generating targets and main
  one-factor comparisons, but current-package population smoke changes some
  alternative-CFA solutions/convergence relative to the frozen sensitivity
  examples. **Check:** separate each model's convergence/admissibility from its
  objective and reliability target; independently verify any claimed optimum
  before extending or using model spread as evidence. Keep historical results
  distinct. See [reliability targets](../../experiments/showcases/08-reliability-targets/report.qmd).

- [ ] **S/M — review retained research capabilities one decision at a time.**
  Candidates: continuous/mixed covariance shrinkage; robust ordinal/polyserial
  menus and pair-local diagnostics; fixed-scalar DLS and Stage-2/IJ adapters;
  Fisher/Fisher-SNLLS/IRLS
  routes; ordinal pairwise composite likelihood; mixed pairwise/FIML hybrids
  and regularized Stage 1; RBM; SAM/LSAM and native FC-SEM as separate decisions.
  Mixed-only reviews follow in 0.3.0; noniterative reviews are indefinitely
  postponed. **Check:** concrete consumers, shared
  dependencies, evidence and a bounded keep/consolidate/remove decision.
  Recording this list does not authorize removals. Retain ordinary moment/Gamma,
  score/IJ and other shared primitives. Parking expansion is not code deletion.

- [ ] **M — audit tolerances and convention exemptions.** Loose parity gates
  must not absorb known divergences. **Check:** justified tolerances, count-pinned
  deferred buckets and independent/calibration proof for oracle exemptions.
  See test ledger and [oracle defects](../validation/oracle-defects.md).

- [ ] **M — add primary target-regime calibration checks.** Per-dataset oracle
  agreement does not establish size. Add advisory paired ML/FIML/DWLS robust
  test/covariance checks against oracle and nominal rates. **Check:** nonnormal,
  missing and ordinal regimes with failures retained; turn deterministic defects
  into tests. See calibration policy.

- [ ] **S — fix remaining example assertions.** Consolidate `ml_psd_fallback.R`
  with the ridge task; diagnose `score_flip_test.R`'s
  `mean_variance_relative_shift == 0` assertion. **Check:** meaningful current-
  contract assertions that do not conceal a discrepancy.

- [ ] **S/M — export named corpus gaps.** Freeze at-theta implied moments for
  `newsom_2015_ex9_3` and `little_2013_ch3_fig_3_6_1indicator` so their goldens
  need no optional mount. Restore corrected Little/Newsom fit gates after the
  documented failures are handled. **Check:** provenance, independent oracle
  values, no third-party data and file-size limits. See translation audit and
  [Newsom failures](newsom-corpus-failures.md).

- [ ] **S — explain Mplus `chapter6_ex6_10` ULS statistics.** Distinguish near-zero
  objective-derived chi-square from lavaan's reported ULS test (38.3).
  **Check:** identify the convention before gating a comparison.

- [ ] **S — finish admissibility fixtures/checkpoints.** Add an oracle warning-
  status witness for an improper complete-data fit; carry audit flags into
  experiments when their runners next change. **Check:** improper/admissible
  controls with convergence/domain kept separate. Composite attachment is deferred.

- [ ] **S — add `sqrt` to defined parameters.** Edit normative EBNF, parser and
  derivatives in that order. **Check:** Mplus twin expressions ex5.21/ex5.22,
  domain errors and derivative propagation.

### Benchmarks

- [ ] **S/M — maintain comparisons needed by current decisions.** Separate
  setup/fit/inference, pair numerics before speed claims and retain objectives,
  verdicts, evaluations, memory and failures. Remove unused fixture fields only
  after checking readers; retire duplicated timing loops when touching runners.
  **Check:** clean-source reproducibility and matched computation. General new
  grids and paper timing programmes are deferred. See benchmark guide and
  [speed attribution](../../experiments/showcases/06-speed-attribution/report.qmd).

## 0.3.0: magmaan's own fitting, mixed data and barriers

Assigned 2026-10-02. Until these land, simulations that need dependable
fitting can use the lavaan-compatible preset from 0.2.0.

### Optimization and convergence

- [ ] **M — complete the minimal ordinary-fit reliability programme.** Start
  with complete-data unrestricted NTML, ULS and GLS. Pair marker/sphere routes
  on identical moments; vary route-native versus shared sample-only FABIN3
  starts, and stock versus tighter L-BFGS/PORT stopping controls. Keep the
  library's native Newton audit fixed and record failed/unchecked endpoints,
  chart availability, backend stops, objective and implied-covariance gaps,
  and cost separately. **Check:** the bounded development pilot in
  [sphere references](../../experiments/engineering/active/15-sphere-reference-fits/report.qmd),
  then targeted failure replays and fresh draws before any default decision.
  First replay the retained heterogeneous-unit NTML/GLS sphere failures while
  marker fits pass, isolating conditioning/preconditioning and stopping;
  separately explain the shared-FABIN3 GLS worse local minimum.
  PSD/barriers, FIML, ordinal data and broad global-search guarantees are
  separate extensions, not prerequisites for this programme.

- [ ] **M — complete sphere-native constraint geometry (high priority after
  0.2.0).** Native unpinned audits now cover ML, FIML, continuous moment-quadratic
  fitting and PSD interiors before chart translation. Complete joint
  sphere/PSD curvature at singular faces and validate the requested-chart
  rejection criterion independently of endpoint accuracy. Active boxes and
  additional nonlinear equalities currently report unchecked; extend them
  only with the required feasible Newton/Lagrangian geometry. **Check:**
  retained pole, finite-improper, boundary and inaccurate-stop witnesses across
  units and identifications; derivative/retraction checks and explicit
  failed/unchecked R conditions. Global search and nonattainment classification
  are separate work. See [sphere implementation](../architecture/roadmap.md)
  and [sphere references](../../experiments/engineering/active/15-sphere-reference-fits/report.qmd).

- [ ] **M — recover from L-BFGS domain aborts across parameter scales.** Limited
  line-search reductions can exhaust infeasible trials at the initial point.
  Assess safeguarded backtracking or adapter recovery while retaining caller
  controls and terminal candidates. **Check:** the
  [domain probe](../../cpp/tests/checks/nlopt_lbfgs_domain.c), corrected corpus
  failures and rescaled fits. A larger evaluation budget alone is insufficient.
  See [corpus recovery](../../experiments/engineering/active/17-corpus-optimizer-recovery/report.qmd).

- [ ] **M — equality constraints that fully determine a block stall the ML fit.**
  Effects coding (each factor's loadings average 1) plus tau-equivalence (the
  loadings equal) fixes every loading of the factor at 1 through `==` rows.
  On Little (2013) Table 10.3 (24 indicators, 8 factors) magmaan's L-BFGS and
  PORT fits both stop at the start, unconverged, with $\chi^2 \approx 73{,}000$,
  while lavaan converges to $\chi^2 = 271$ on the same draw. Found by the
  sem-score-tests calibration study, where the case is supplement-only.
  **Check:** that model's restricted fit converges to lavaan's optimum, and the
  constraint projection handles constraints that pin parameters completely.

- [ ] **M — diagnose the layered-start Geiser latent-AR loss.** The fixture
  `latent_ar_cross_lagged_extended` stalls above the reference objective with
  L-BFGS and PORT; FABIN3 and std.lv succeed. **Check:** explain the valley/start
  failure, preserve a regression and verify corpus/held-out behavior before
  changing defaults. Evidence: corpus recovery.

- [ ] **M — finish unit-equivariant starts and optimizer coordinates.** Remove
  unit-dependent FABIN/layered fallback steps; extend supported start selections
  and fallback reports to all-ordinal preparation. Wire coordinates through
  primary all-ordinal and relevant SNLLS/IRLS paths; mixed extensions follow with
  mixed completion. **Check:** transported
  starts and fits across units, groups, constraints and identification;
  reject unsupported selections. Exclude two-level, SAM, FC-SEM and removed
  fitted-weight/automatic-identification routes. See
  [optimizer controls](../reference/optimizer-controls.md) and corpus recovery.

- [ ] **M — detect reflection-trapped starts and validate saddle escape.**
  Fixed-variance latent scales permit zero-gradient sign-reflection subspaces
  with negative curvature. Check all starts against fixed entries/constraints;
  assess a safeguarded original-objective step as explicit recovery. **Check:**
  phantom, second-order and Little Table 7.6 witnesses, retaining user-start
  and intervention reports. Evidence: corpus recovery.

- [ ] **S/M — reject unidentified exact-fit ridges reliably.** The free-marker
  CFA in `ml_psd_fallback.R` can pass a Newton check with seven parameters for
  six moments. Add scale-free identification checking alongside local accuracy.
  **Check:** the ridge and identified constrained controls under unit changes;
  avoid tolerance changes that merely move the failure. See
  [terminal audit](../design/terminal-audit.md).

- [ ] **M — finish common-verdict and stopping-control integration.** Preserve
  evaluable candidates on soft exits, report effective controls/raw reasons
  separately from the verdict, and migrate active backend-status consumers.
  Assess ML stopping controls for FIML/DWLS and accuracy-budget sensitivity with
  estimated Gamma. **Check:** corpus near misses/non-minima and consistent R
  TRUE/FALSE/NA projection; preserve failed/unchecked fits and rank diagnostics.
  See terminal audit and
  [Newton rollout](../../experiments/engineering/active/19-newton-verdict-migration/report.qmd).

- [ ] **S/M — decide the flat-ridge ordinal golden gate.** Newsom 2024 ex1.3c
  passes the accuracy budget but differs in raw parameters. **Check:** a
  justified information-metric gate or tighter stop, with independently checked
  objective before removing its `kKnownGaps` entry; Newsom 2015 ex9.2 (the
  free-delta Heywood bound) keeps the list non-empty. Free-delta bound relaxation is
  separately [consumer-gated](speculative.md#categorical-scope-extensions).

### PSD and remaining normalization

- [ ] **M — make caller-unit covariance audits scale robust.** Include standalone
  audits and normalization-disabled fits. **Check:** heterogeneous units,
  equalities, means and improper witnesses, retaining the distinction between
  accuracy and admissibility. Replay the paired normalization losses in
  [optimizer defaults](../../experiments/decisions/01-optimizer-defaults/report.qmd).

- [ ] **M — repair PSD finalization and retained route losses.** Diagnose std.lv
  equal-loading lift round-trip failures and constrained-chain objective losses
  (`eqchain_b20`), including starts and covariance-link mappings. **Check:**
  lifted/ordinary objectives, links and equality residuals across units. Direct
  PSD remains the ordinary route; fallback remains explicit. Any route change
  needs a fresh decision study. Evidence: optimizer defaults.

- [ ] **L — extend shared normalization to FIML.** Transport raw data/patterns,
  fixed-x inputs, groups, models, starts, bounds, affine equalities and applicable
  inference; return caller units. **Check:** all-observed reduction,
  missing-pattern derivatives and unit/group/constraint round trips before
  adoption. Complete-data ML/PSD normalization is already enabled.
  See optimizer controls.

- [ ] **L — close primary PSD stress gaps.** Localize ML, FIML and all-ordinal
  DWLS failures using the retained harness. Keep stage-one objects unchanged;
  separate solver status, cone stationarity, admissibility and competing basins.
  **Check:** independent objective/link recomputation, interior reductions,
  retained seeds and targeted confirmation with failures/cost tails reported.
  Promote deterministic defects to tests. Secondary breadth is consumer-gated;
  two-level and native FC-SEM stay excluded. Evidence:
  [PSD stress](../../experiments/engineering/active/13-psd-estimator-stress/report.qmd).

### Mixed continuous/ordered models

Existing mixed delta/theta fits and inference APIs retain their documented
regression gates. New mixed-only work and broader completion follow in this
release, including dependencies otherwise shared with the primary programme.

- [ ] **M/L — finish mixed fitting, preparation and PSD coverage.** Extend the
  shared starts/coordinates, moments/means/scales, association projection and
  retained composition/weight metadata to mixed models. Complete supported
  grouped/invariance release adapters and localize mixed-only PSD failures.
  **Check:** delta/theta, units, groups, equalities, independent objective/Jacobian
  gates and retained Stage-1 objects; unsupported moment/mean/scale compositions
  remain explicit. Shared all-ordinal correctness fixes can land earlier.

- [ ] **L — complete mixed policy and weighted MI/release inference.** Establish
  the mixed covariance and weight-influence law, then extend the current
  fixed-weight MI/release paths to justified estimated-weight and policy
  combinations. **Check:** continuous/all-ordinal reductions where defined,
  independent mixed influence/weight perturbations, rank/candidate consistency,
  grouped and misspecification calibration, and core/R agreement before exposure.

### Association-ML inference

- [ ] **L — define association-ML MI/release and sampling inference.** Gate
  the active association Jacobian, saturated-threshold/NACOV transport and the
  matching ML information before exposing MI/release, covariance or tests for
  ordinal association-ML fits. **Check:** reductions where defined,
  independent derivatives, grouped/constrained controls and calibration; until
  then every component is rejected with a typed reason.

### Barrier fitting and inference

- [ ] **L — finish whole-barrier normalization and stress validation.** Extend
  and validate the shared complete-data ML/PSD transformation for model/data,
  automatic and supplied starts, fixed
  values, means, groups, labels, affine equalities and supported bounds.
  Establish penalty/strength/schedule transport so changed units preserve the
  statistical criterion, up to accounted additive constants. Transport
  derivatives, repairs, stopping/audit geometry, warm starts and applicable
  post-fit artifacts; return caller units. **Check:** same-point criterion,
  derivative and constraint identities, complete fits under uniform/mixed units,
  groups, equalities, boundaries and poles, with every regression retained.
  Preserve requested identification, avoid double normalization; nonlinear
  equalities remain outside this slice. See
  [normalization contract](../reference/optimizer-controls.md#complete-data-ml-and-psd-sample-normalization-2026-09-27)
  and [barrier defaults](../../experiments/decisions/02-barrier-defaults/report.qmd).

- [ ] **M — fix barrier fallback units and near-pole verdicts.** At ×0.01 in
  equality-constrained models, `native-fabin-fallback` supplies wrongly scaled
  starts; extreme accepted marker-chart endpoints also need chart-proximity
  diagnosis. **Check:** replay retained failures, then a fresh paired lane with
  losses retained. Chart extent is not proof of nonattainment; do not change
  markers automatically. Evidence: barrier defaults and
  [sphere references](../../experiments/engineering/active/15-sphere-reference-fits/report.qmd).

The shared barrier fitting baseline is recorded in the
[roadmap](../architecture/roadmap.md#implemented-composition-2026-10-01).
Retain current barrier entry points and regression gates.

- [ ] **M — finish remaining model-penalty compositions (0.3.0).** Non-mixed
  continuous, pairwise, saturated-FIML, ordinal and direct-FIML penalty fitting
  has landed. Remaining: mixed discrepancies, an analytic penalized-ordinal
  Newton audit (ordinal penalty routes now provide geometric stationarity
  only), and penalized starts/verdict integration found by barrier stress work.
  **Check:** per-combination domain, strength/normalization and reduction
  identities. Preserve inputs; a model barrier does not repair indefinite
  polychoric/pairwise moments. Saturated Stage-1 penalties require a separate
  target, propagation and H1-reference contract; latent determinacy is zero
  without genuine latents.

- [ ] **L — establish applicable barrier inference and exposure contracts.**
  After normalized fitting, distinguish penalized-estimate covariance from
  ordinary inverse information, and single-face displacement tests from a
  general multi-face reference law. Use each moment source's Gamma/influence
  law, target-map derivatives and actual penalized estimating equation; retain
  structural-parameter and threshold uncertainty explicitly. Define scaling/
  domain and inference before claiming DWLS penalty inference or validated
  ordinary-user inference. The adopted ordinary API exposes fitting
  experimentally as `covariance = barrier(lambda)` and reports unavailable
  inference.
  **Check:** independent derivatives, fixed-penalty limits,
  regular/boundary covariance, global/nested tests and interval calibration,
  with explicit unsupported components. Reuse correlation-ML criterion
  evaluation at DWLS fits for robust RMSEA without implying ML refitting.
  See the composition contract and the research index's barrier studies.

## Related work

- [Simulation backlog](simulation.md) owns generator/projection/calibration
  detail, including model-implied simulation and group metadata. Only primary-
  workflow dependencies belong in this queue.
- [Speculative register](speculative.md) owns parked methods, model families,
  expensive verification and optional reporting/packaging. Promotion needs a
  named consumer, bounded scope and completion check.
