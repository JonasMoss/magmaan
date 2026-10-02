# magmaan TODO

Accepted implementation work, ordered by release. The
[roadmap](../architecture/roadmap.md) owns current state/contracts and indexes
the per-area implemented capability files;
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
| **0.1.0, shipped 2026-10-01** | Simulation prerelease of `magmaan` and `magmaanlab` (local tag `v0.1.0`); API hardening and its gates are recorded in the [roadmap](../architecture/capabilities/r_bindings.md#r-bindings-and-public-namespace-transition) and package NEWS |
| **0.2.0, current** | The adopted ordinary API; lavaan-compatible fitting through `options`, so simulations can rely on lavaan's fitting while magmaan's own optimizer work waits; ordinary-policy inference for ML, FIML and all-ordinal DWLS; MI/release-score completion across weights, including two-stage |
| **0.3.0, next** | magmaan's own fitting reliability (starts, optimization, convergence, PSD finalization, stress and normalization); mixed continuous/ordered workflows; barrier hardening and inference; association-ML MI |
| **After 0.3.0** | Review what remains and bank items without a consumer in the [speculative register](speculative.md) |

Sphere-chart failure handling is deferred to the
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
   derivatives at identical parameter points, endpoint/objective tolerances
   and verdicts; each actual endpoint is checked under the declared acceptance
   rule. Floating-point retry paths need not have identical final gradients.
   Installed-lavaan R comparisons pass on a named simulation-model set.
   Nonzero bounds, nonlinear constraints and other routes error; nonzero-RHS
   affine standardized retries and scaling that changes a homogeneous
   constraint surface are explicitly unavailable because of the
   [proved oracle defect](../validation/oracle-defects.md).
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

- [x] **M — expose named downstream inference conventions** (2026-10-02).
  `convention` on `vcov`, `confint`, `summary` and `anova`; C++ compatibility
  composers, on-demand reporting and optional `infer(fit, convention)` caching.
  Complete-data ML/MLM/MLR and all-ordinal DWLS/WLSMV, ULS/ULSMV and WLS
  covariance/global recipes; ML nested defaults. The
  [capability inventory](../validation/capabilities.md) records checked slices.
  **Check:** installed-lavaan whole-bundle comparisons, typed unavailable
  reasons, incompatible-regime rejection and preserved estimates/default policy.

- [ ] **M — extend checked reporting conventions to FIML and ordinal nested
  tests.** These are missing ordinary reporting compositions, not missing C++
  algorithms or unavailable lavaan features. Reuse FIML standard/robust
  covariance, global MLR and nested engines, and the ordinal Satorra-2000
  exact/delta engine, scaled-shifted reducer and existing lab adapters.
  FIML ML/MLR need covariance/global/default nested bundle gates; pin the
  observed-information, H1 and scale recipes rather than assuming the scalar
  SB2001 engine's spectrum-derived scales equal lavaan MLR's trace recipe.
  WLSMV/ULSMV nested reporting needs delta plus scaled-shifted composition and
  per-group reporting normalization; existing ordinal golden tests gate the
  mean-scaled result. Probe plain ordinal DWLS/ULS/WLS nested reports separately.
  **Check:** live lavaan defaults, full covariance and intervals, statistic/df/
  p-value/scale/shift, grouped incomplete FIML and ordinal delta/theta invariance,
  both model orders, actual nesting, saturated alternatives, convergence and
  penalty refusals, deferred/cached/serialized reporting, and preserved
  estimates/default policy. Extend the [inventory](../validation/capabilities.md#reporting-gap-audit-2026-10-02).
  Continuous LS, two-stage, mixed and additional historical bundles remain
  consumer-gated. This does not block primary policy composition or silently
  broaden 0.2.0's exit criteria.

- [ ] **S/M — close validation gaps in existing reporting bundles.** Add
  complete-ML grouped nested default comparisons with loading/intercept
  equalities and released means; grouped ordinal ULS/ULSMV/WLS covariance/global
  gates and threshold-restriction coverage beyond current loading equality.
  Inventory interior PSD, boundary, active-bound and degenerate-scale behavior
  separately from classical unrestricted parity; add a gate or an explicit
  unsupported/limited-validation entry for each retained slice. **Check:**
  source/fixture and installed-lavaan evidence identify the exact API and
  method, including group-specific n-minus-one normalization; no silent
  fallback or unsupported promotion. Future FIML/ordinal nested bundles use
  the same test obligations. PSD hardening remains 0.3.0.

- [ ] **S — decide the compatibility selector's name and reporting label.**
  The current argument is `convention = "magmaan"`. Proposed on 2026-10-02:
  `lavaan_compat = NULL` for the ordinary default and an explicit historical
  bundle such as `"MLR"` or `"WLSMV"` for reproduction/comparison. This spelling
  is not yet adopted; compatibility must not read as an endorsed policy choice.
  **Check:** agree the spelling/default, then update `vcov`, `confint`, `summary`,
  `anova`, `infer`, labels, result attributes, caches, docs and tests together;
  preserve default policy and exact bundle identity. No ingredient switches.

### lavaan-compatible fitting

Consumer: simulation studies that need lavaan-identical fitting while
magmaan's own optimizer work waits for 0.3.0. The `lavaan-0.7.2` preset
already pins starts, nlminb/PORT controls and scaling, the four-attempt retry
sequence and lavaan's acceptance rule for ordinary complete continuous ML
including linear equalities; under it the selected rule decides `converged`
and magmaan's own check is reported beside it (2026-10-02). Inference stays on
magmaan's policy, so lavaan-identical fitting does not imply lavaan-identical
standard errors or tests. Retain effective controls, every attempt and
separate native diagnostics; `options$convergence = "newton"` keeps magmaan's
own verdict. Nonzero bounds, nonlinear constraints (lavaan's NLMINB.CONSTR
augmented Lagrangian) and PSD, pairwise and two-level routes remain explicit
errors under the preset. Gates compare starts, search coordinates, derivatives
at identical points, endpoint/objective tolerances, soft failures and retries,
with each actual endpoint's declared acceptance checked separately.

- [x] **M — fit linear equality constraints in lavaan's coordinates**
  (2026-10-02). Ordered name-free affine rows preserve lavaan 0.7.2's QR basis,
  including row-order-sensitive equivalent systems; native fitting keeps its
  existing basis. The preset packs starts/scales/gradients, removes bounds and
  retains every attempt. Frozen and installed-version gates cover shared
  labels, `group.equal` loadings/intercepts, nonzero-RHS and redundant rows,
  partable round trips and homogeneous standardized retries. Approved retry
  validation compares derivatives at identical points and checks each actual
  endpoint's acceptance, while retaining endpoint/objective tolerances.
  Nonzero-RHS standardized retries and homogeneous scaling that fails
  `A*D^-1*K=0` return an explicit error before changing the affine surface;
  the independent proof and replacement gates are in
  [the oracle ledger](../validation/oracle-defects.md). Invalid constrained
  initial covariances error as in the oracle. `ceq.simple = TRUE`, nonlinear
  constraints, nonzero bounds and PSD/pairwise/two-level preset routes remain
  unavailable. The older unconstrained ×1000/×10⁵ witnesses still record their
  pre-existing endpoint sensitivity in `fitting/lavaan_0_7_2.json`.

- [ ] **Task-47 — diagnose opt equality retry acceptance** (2026-10-02).
  The equality fixture test now explicitly returns after prerequisite failures
  and skips attempt indexing after a count mismatch under `-fno-exceptions`.
  The ×100 oracle first-attempt gradient maximum is 0.0016149511731821235
  versus the 0.001 acceptance threshold (relative gap +0.6149511731821235).
  Opt accepts its first attempt with PORT status 4 and gradient maximum
  0.00094775120123813394 (relative gap -0.052248798761866076); this is not
  knife-edge acceptance. The targeted opt configured tests report 2 passed,
  1 failed, with no crash. **Needs decision:** authorize investigation of the
  non-marginal optimizer endpoint divergence before adapting retry parity.
  Parity assertions and tolerances remain unchanged.

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

- [x] **M — move the ML nested policy to observed geometry** (2026-10-02).
  `api::policy_nested_ml` passes `Information::Observed` to the nested NTML
  quadratics: the score projects with the larger model's observed
  information at the restricted fit and keeps the expected metric on the
  projected directions (the FIML recipe, so its statistic changes as well as
  its spectrum, decided with the author), and the Satorra-2000 LR spectrum
  reduces through the observed information at the larger model (statistic
  unchanged). `ntml_quadratic(hypothesis, score, geometry)` defaults to
  expected, which the lab Satorra-2000 test, the lab quadratic binding and the
  lavaan conventions keep. Gates (`policy_test.cpp`): closed-form observed
  information against second differences; first-principles score
  reconstruction from finite-difference casewise scores in both geometries;
  independent LR spectrum reconstruction; exact-fit identity (observed equals
  expected); the complete-data FIML nested score equals the policy score to
  1e-9 with a mean structure; the existing embedding, boundary and unit gates.
  Task-44 implements group-mean profiling for covariance-only complete-data
  lab nested/global components with single-/multi-group NTML gates at 1e-9;
  targeted opt validation passes (13 cases, 451 assertions), including existing
  score-flip and mean-structure FIML-pack checks.

- [ ] **M — calibrate the observed nested ML geometry.** One frozen run with
  correct and misspecified larger models comparing expected and observed
  nested geometry (size and size-adjusted power, score and LR, SB and PEBA4)
  precedes release. Needs compute: simbox is unavailable, so Modal (cost
  estimate first) or a small local run within the 5-minute rule. Optional
  side check: a component comparison with lavaan fits using
  `information = "observed"`; exact equality is not expected, since lavaan
  keeps its normal-theory weight in the Satorra-2000 projector and
  `lavTestScore()` uses observed information for the metric too.

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

- [x] **S/M — compose the DWLS policy covariance and global test** (2026-10-02).
  `api::policy_inference_dwls`: the IJ covariance (`robust_ordinal_ij`, which
  now takes the fit-time `row_user`) and one global test, the fit-function
  statistic n F with the fixed-weight `robust_ordinal` UGamma spectrum, SB and
  PEBA4, reported in `score` with label `fit_function`; `lr` is the new typed
  `inapplicable` reason. Only plain DWLS qualifies (Stage-2 NT/DLS, supplied
  weights, ULS and WLS stay `unsupported_model`). R: `policy_inference()`
  routes all-ordinal DWLS fits; the lab `vcov()` gains the explicit
  `sandwich_ij` regime (defaults stay fixed-weight, lavaan's `robust.sem`);
  `magmaan()` prints the test as "fit function" and the LR as a note. Gates
  (`policy_dwls_test.cpp`): wiring against `robust_ordinal_ij`/`robust_ordinal`;
  exact fit (saturated) reduces to the fixed-weight sandwich (1e-6);
  stratified delete-one jackknife at n = 600 per group agrees within 4% for
  single-group delta, two-group delta and theta (IJ 2.9/2.3/0.3% versus
  fixed weight 4.3/4.6/1.7%; theta separates from fixed weight clearly by
  n = 2400). Remaining: the calibration item below and lavaan WLSMV agreement
  of the shared spectrum, which the conventions already gate.

- [x] **S/M — compose the DWLS nested likelihood-ratio-type test** (2026-10-02).
  `api::policy_nested_dwls`: the fit-function difference T = n(F_null −
  F_alt) in the `lr` slot (label `fit_function_difference`) with the
  estimated-weight profile law (`ordinal_dwls_profile_lrt`, each model's
  profile at its own estimate over thresholds, polychorics and the DWLS
  weight diagonal; evidence 13). Decisions: the reference is the positive
  profile spectrum with values below 1e-8 of the largest treated as zero
  (otherwise theta's extra scale directions change PEBA4), padded to the
  restriction df; SB divides its full trace by the restriction df (mean
  matching over every term; `fmg_test`'s top-df truncation would drop the
  weight channel); PEBA4 uses the whole spectrum; negative profile
  eigenvalues are dropped as in the validated mixture (conservative). Nesting
  is verified with `robust::embed_nested_null` on the prepared partables. The
  nested score is typed `unsupported_model` (no derivation). R:
  `policy_nested()` routes two DWLS fits with identical ordinal statistics;
  `anova()` prints "fit-function difference". Gates (`policy_dwls_test.cpp`):
  statistic equals the difference of the global fit-function statistics; role
  swap is refused; delta and theta give identical statistics, spectra and
  PEBA4; two groups compose; under a true null the profile trace approaches
  Satorra-2000's (delta 1.9% to 0.19%, theta 2.3% to 0.20% from n = 4000 to
  64000). Calibration remains open (below).

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
| Continuous ULS/GLS/WLS | Shared moment-quadratic MI; fixed- and estimated-weight robust primitives | Identity, NT, diagonal/full empirical, DLS(a) and supplied block W: R uses the recorded fitting W and recipe/a for empirical/model-implied covariance and the weight influence (supplied W refused there); remaining work audits sensitivity/nuisance projection and exposes caller-Gamma adapters |
| All-ordinal ULS/DWLS/WLS | Ordinary and robust threshold/association MI; estimated-weight DWLS/WLS path | Gate identity, diagonal/full NACOV, retained Stage-2 DLS and supplied weights with delta/theta and group conventions; shared relative rank is implemented, provenance/adapters remain |
| Prepared all-ordinal association ML | Ordinary/PSD fitting through the shared association-target contract; LS MI is not an ML-target score contract | Typed rejection in 0.2.0; the contract is [0.3.0 work](#association-ml-inference) |
| Two-stage/ML2S | Naive Stage-2 MI/release (`mi_type = "naive_stage2"`) and Stage-1-scaled MI/release for NT, ULS, DWLS, ADF and DLS, fixed or estimated weight, in C++ and R | Gated by exact complete-data reductions; remaining cells are grouped and MAR checks in the matrix item. lavaan's two-stage `modindices()` uses the unstructured information, so a fixture needs that option for the naive row |

The DLS/custom-weight rows concern bounded reuse of retained weighted primitives,
not a general DLS research programme. Pairwise moment sources retain their MCAR
scope and need their own covariance law; direct composite-likelihood, two-level
SEM, SAM, mixed-data and noniterative expansion are outside this slice.
Automatic absent-row enumeration covers cross-loadings and covariances;
structural-path enumeration remains a separate model-builder contract.

The ordinary package does not expose MI in this release; the
[interface vision](../design/r-interface-vision.md) defers policy MI.

- [ ] **M/L — complete weighted MI/release provenance and adapters.** Cover the
  retained continuous and all-ordinal weight recipes in the matrix, with stored
  fitting W or explicit supplied W, Gamma/NACOV source, recipe/a and fixed versus
  estimated-weight influence. Derive consistent sensitivity and nuisance
  projection for observed/estimated-weight GMM score variants before exposing
  them; the existing expected-metric sweep alone does not establish that regime.
  Association-ML MI is a 0.3.0 contract; the score workers reject it until
  then. Fits record recipe, a and W, and every R estimated-weight consumer
  resolves its IJ mode from that record (roadmap: estimated-weight recipe
  guard); exposing caller Gamma in R is the remaining wiring, and the
  sensitivity and nuisance projection for observed/estimated-weight score
  variants is the genuine derivation.
  Audit every bread/information/covariance argument and expose applicable caller-
  Gamma paths through thin R adapters. **Check:** independent score, sensitivity,
  meat and weight-influence assembly; recipe endpoint reductions, retained-data
  versus supplied-data agreement, documented unavailable cells and no ignored
  options. No numerical recipe/default changes without evidence.

- [x] **S — preserve user-written rows in C++ API ordinal post-fit inference.**
  `api::robust_ordinal`, `fit_measures`, `modification_indices` and
  `score_tests` pass the fit-time `row_user` mask through all-ordinal and mixed
  ordinal preparation; automatic ordinal/mixed starts use the same mask.
  Regressions cover fixed non-default and free explicit residual variances
  (all-ordinal theta and supported mixed delta), fit-objective consistency,
  absent-row MI, and equivalent explicit/implicit default spellings. R glue is
  unchanged.

- [ ] **S — complete lower-level ordinal preparation provenance.** The IJ,
  RBM/casewise and Satorra-2000 entry points still prepare without `row_user`.
  A freed `~*~` is not idempotent: preparation transfers its free dimension to
  `~~` and pins `~*~` to 1; a second preparation loses that release and pins
  the auto `~~` too. Preserve the release in the model/preparation contract,
  rather than guessing from an explicitly fixed `~*~` row. **Check:** repeat
  preparation preserves the free set/constraints/starts, and low-level
  inference agrees with the fitted preparation for explicit ordinal rows
  (board TASK-37).

- [x] **M — close the MI/release estimator-by-weight validation matrix.** Gate
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
  Component matrix (2026-10-02): [23 estimator/recipe rows and 92 cells](../validation/capabilities.md#mi-and-equality-release-score-components)
  name C++/R gates or actual refusals, including unequal-group/MAR ML2S,
  continuous LS means/shared labels, ordinal recipes and adapter limits.
  GLS/WLS ordinary MI fixtures preserve raw lavaan values with explicit
  score-divisor transport and independent analytic score/Schur/EPC checks.
  FIML MI is gated at 2e-5; raw ULS/DWLS tolerances are tightened to measured
  finite-divisor floors. Remaining evidence limits: no direct complete-data
  MLR MI oracle; ordinal ULS/WLS use reduction/rank gates; mixed ordinary MI
  retains a factor-two discrepancy tracked as TASK-33.4 (0.3.0), not raw oracle
  parity. Component closure does not establish new calibration or defaults.

### Release readiness

- [ ] **S/M — inventory validated primary capabilities.** Record model/data
  slice, domain, penalty, algorithm, API tier and evidence for estimation,
  verdict/admissibility, covariance, global/nested tests and intervals separately.
  Use validated, limited-validation, unsupported and inapplicable states;
  planned slices link here. **Check:** C++/R owners and ordinary-policy exposure
  agree. [The inventory](../validation/capabilities.md) starts with reporting
  conventions; the wider policy/domain rows remain to do. The table is 0.2.0
  exit criterion 6. The [2026-10-02 reporting sweep](../validation/capabilities.md#reporting-gap-audit-2026-10-02)
  distinguishes existing C++/lab algorithms from missing ordinary composers,
  adapters and recipe gates. Finish the broader sweep of estimator × covariance
  domain × component × API tier, including retained-data/refit ownership,
  constraints, groups/missingness, penalties, verdicts and interval routes.
  Every gap must name its implementation owner, evidence and existing backlog
  task; identify intentional exclusions separately. Keep one inventory;
  secondary breadth is consumer-gated. See
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

### Inference interface follow-ups

- [ ] **S/M — compose the planned LR interval interface.** Ordinary
  `confint(test = "lr")` currently rejects the request despite existing C++ and
  lab profile-test/CI engines for ML, FIML and ordinal fits. After the relevant
  policy test recipe is settled, specify its scalar/defined-parameter scope,
  calibration, retained-data refits, covariance policy, bracketing/inversion
  failures and interaction with explicit lavaan compatibility. **Check:**
  endpoint statistic/cutoff gates, unit invariance, unchanged anchor fits and
  explicit failure reasons; Wald remains the default. This is not a 0.2.0 exit
  requirement and does not promote every lab profile route to ordinary use.

- [ ] **S/M — make lab estimated-weight comparisons consistent.** Add an
  `estimated_weight` off switch to `frontier_rbm()` so supplied-W fits can request
  fixed-weight RBM rather than always hitting the estimated-recipe refusal.
  Inventory and choose one documented lab default across robust MI/score,
  residual, case-influence, GMM/ML2S profile and ordinal misspecification
  fit-measure routes; explicit off/on currently differs between them. **Check:**
  fixed-weight reductions, estimated-weight identities, supplied-W acceptance/
  refusal and matched recipes across lab wrappers. Preserve numerical policy
  defaults and evidence; a lavaan bundle is not an isolated correction switch.

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

- [x] **S — fix remaining example assertions.** Completed 2026-10-02 (board
  TASK-28). The complete-data score variance shift is roundoff (2.09e-16),
  checked with a dimension-scaled machine-epsilon bound. ML/PSD starts are
  compared with constructors on normalized samples and caller-unit transport;
  the verbatim-sample control retains the 1e-12 vector tolerance. The PSD
  fallback example reports the unidentified ridge limitation (TASK-33.3) and
  checks its actual verdict/admissibility selection rule. **Check:** all three
  examples and the fitting-options/frontier-fit R tests pass against an
  isolated opt install; no library behavior or fit tolerance changed.

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
  six moments (board TASK-33.3). The 2026-10-02 isolated opt witness is accepted
  with Newton distance 5.54e-10, condition 8.14e10 and no reported null directions;
  scaling all loadings by c and the factor variance by 1/c^2 preserves Sigma.
  The example now retains and reports this limitation while checking selection.
  Add scale-free identification checking alongside local accuracy.
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
