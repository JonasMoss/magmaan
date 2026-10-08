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

Revised 2026-10-07. One version number covers the C++ library and both R
packages, which ship together through the vendored core. The former 0.0.1
label (the C++ project version) is retired. Work proceeded on 0.3.0 items in
parallel with 0.2.0; on 2026-10-07 the user decided that everything on `main`
ships as 0.2.0, so the release notes fold the "0.3.0 (in development)" NEWS
entries into 0.2.0. 0.2.0 was released on 2026-10-07 as tag `v0.2.0`
(`52471911`); later NEWS entries go under a development heading above it.

| Version | Content |
| --- | --- |
| **0.1.0, shipped 2026-10-01** | Simulation prerelease of `magmaan` and `magmaanlab` (tag `v0.1.0`); API hardening and its gates are recorded in the [roadmap](../architecture/capabilities/r_bindings.md#r-bindings-and-public-namespace-transition) and package NEWS |
| **0.2.0, released 2026-10-07** | The simulation release (tag `v0.2.0`). The adopted ordinary API; lavaan-compatible fitting through `options`; ordinary-policy inference for ML, FIML and all-ordinal DWLS (exact first stage, All/PEBA4 references, reconfirmed by decisions/05 and 07); uniform test tables with `references` for comparing tests; MI/release-score completion across weights; and, landed early, standardized estimates, robust `modindices()`, policy and lavaan-compatible `fit_measures()` |
| **0.2.1, in progress: complete lavaan-0.7.2 preset** | Cut from `main` when the hard-case parity gate passes (TASK-129, user 2026-10-08): lavaan 0.7.2's marker switch as a `marker` fitting-options component with an off switch, lavaan's `post.check` reported beside `converged`, and a hard-case gate against live lavaan for ML, FIML and DWLS. First consumer: the sem-psd rerun, which pins this release. The tag needs the user's explicit go |
| **0.3.0, current: feature completeness** | magmaan's own fitting reliability (starts, optimization, convergence, PSD finalization, stress and normalization); mixed continuous/ordered workflows and their calibration (TASK-80); barrier hardening and inference; association-ML policy calibration (decisions/08) and MI; fit-index intervals (TASK-103); a stable Mplus input frontend for the linear SEM subset; latent non-normality evidence for the DWLS policy (TASK-84) |
| **0.4.0: documentation, cleanup and clarity** | User and simulation guides; one decision register that states every default with its evidence; `magmaanlab` export tiering (stable, research, superseded) and retirement of legacy names; consolidated error messages and help; CRAN-readiness checks for both packages |
| **After 0.4.0** | Review what remains and bank items without a consumer in the [speculative register](speculative.md); then 1.0 |

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
   `group.equal`, `==` rows), FIML, all-ordinal and complete mixed DWLS, single- and
   multi-group. Pinned fixtures match lavaan's starts, search coordinates,
   derivatives at identical parameter points, endpoint/objective tolerances
   and verdicts for path-stable cases; each actual endpoint is checked under the
   declared acceptance rule. Rescaled retry endpoints and verdicts can depend
   on floating-point search paths.
   Installed-lavaan R comparisons must pass on the eight-case named set in
   [test_preset_simulation_parity.R](../../r-package/tests/testthat/test_preset_simulation_parity.R):
   HS CFA, PoliticalDemocracy SEM, school-invariant HS CFA (ML); HS MCAR/MAR
   (FIML); HS delta/theta and school-invariant theta (all-ordinal DWLS).
   Task-59 passes under the approved estimate-or-endpoint contract; see below.
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

- [x] **L — lavaan-compatible fit-measures composer** (task-109): C++
  standard/scaled/robust families, lab/ordinary compatibility dispatch and
  analytic categorical independence baselines. Complete ML/MLM/MLR and
  ordinal DWLS/WLSMV, ULS/ULSMV and WLS have live same-point 1e-6 gates,
  including two groups and delta/theta. FIML ML/MLR are validated by task-110
  after tightening compatibility H1 moment convergence to 1e-10; MAR and
  grouped-missingness gates retain 1e-6 without an oracle exemption.

- [x] **M — ordinary policy fit measures** (task-102): fixed index/estimate/reason
  rows and opt-in summary attachment/printing. Compatibility remains task-109;
  intervals await task-103.

- [x] **L — C++ policy fit-index point composer** (task-101): ML/FIML
  Takeuchi differences, ULS profile plus covariance moment bias, Exact
  ordinal/mixed DWLS corrections, corrected pooled residuals and lab adapter.
  Component validation only; the ordinary interface is implemented (task-102), and point
  bias / intervals require the registered task-103 evaluation. Unsupported
  estimator corrections carry explicit missing-ingredient reasons.


- [x] **M/L — implement the adopted ordinary API surface** (2026-10-02).
  `magmaan_model()` with frozen group/category schema and zero-row prototypes;
  `magmaan(model, data, estimator, covariance, inference, options)` with
  always-on means, random X, `covariance = barrier(lambda)` (session message,
  printed status, C++ `penalized` reason), `options$start`, typed
  `magmaan_schema_error` failures and migration errors for removed arguments.
  Likelihood-ratio refits keep the anchor's group order. Fits run through
  `fit_model()` on the constructed specification. See the
  [implementation record](../design/r-interface-vision.md#implementation-and-remaining-decisions).

- [x] **S — prepared-path fitting options for all-ordinal DWLS.** Prepared
  `estimate()` reuses the fresh fitter's configured ordinal engine for presets,
  starts, optimizer and convergence choices. `magmaan()` uses prepared handles
  for these fits; ML2S retains its fallback. Delta/theta, single/two-group and
  loading/Wu-Estabrook threshold invariance parity gates cover estimates,
  verdicts, attempts and reporting metadata.

- [x] **M — fit constructed ordinary models through native prepared
  handles.** `magmaan_model()` should own a prepared model so repeated fits
  skip structural preparation (ordinal DWLS 9.1 → 3.5 ms in the 2026-10-01
  timing). Prepared `estimate()` parity is complete (task-25.1): ordinary
  starts, fitting options/presets, fit-time start tables, mixed PSD fitting and
  replayable refit routes use the reference compositions. Task-25.3 closes
  ordinal loading-equality and theta loading/threshold-invariance parity,
  nested DWLS policy pairs, retained reporting moment/weight layouts (including
  ULS Gamma) and the mixed-ULS diagnostic. The permitted ML2S
  gap is explicit: `magmaan_unsupported_estimator` directs callers to
  `fit_model()`; no Stage-1 ML2S implementation was added. The ordinary caller
  now fits through these handles (task-25.2). Native handles stay
  process-local and are rebuilt from
  the portable model on workers. **Check:** fresh/prepared parity in partable,
  estimates, objective, diagnostics and routes for every ordinary estimator
  and covariance policy; zero repeated structural-preparation calls;
  changed-data starts/thresholds; save/reload and worker reconstruction; and
  separately timed construction, data preparation, fit and inference.
  Task-25.2 uses a lazy reference cache, detects serialized NULL pointers and
  process changes, and rebuilds transparently. Repeated-fit, save/reload and
  PSOCK-worker checks pass. ML2S and ordinal DWLS with fitting options retain
  the authorized `fit_model()` fallback; no broader fallback is needed after
  task-25.3. Full ordinary tests pass 1,399 assertions; full lab tests pass
  4,944 assertions (two warnings, two existing skips).
  HolzingerSwineford1939 phase timings (single thread, median of 20 runs
  after one warmup; milliseconds, before → after): ML construction 0 → 1,
  data preparation 0 → 0, fit 1 → 1, inference 1 → 1; all-ordinal DWLS
  construction 1 → 0, data preparation 1 → 1, fit 5 → 1, inference 2 → 2.
  These use the elapsed process clock (millisecond resolution); zero means
  below its resolution. Before fits consume separately constructed lab sample
  statistics; after fits consume prepared data with a warm model handle.
  Construction times exclude lazy native preparation; fit times include
  structural work before and weight preparation after. These are boundary
  measurements, not end-to-end speedup evidence.

- [x] **L — prepare realistic-model DWLS reference-law study** (task-86).
  `experiments/decisions/07-dwls-reference-law` registers eight textbook
  latent-response families, exact null restrictions, 192 cells and fourteen
  references on identical saved policy spectra. Frozen derived populations,
  development smoke/pilot, per-cell completion/provenance checks and pricing
  support production planning. Production (184 cells, simbox, 2026-10-07)
  confirms All for global and nested DWLS tests; the bifactor nested null
  (not exact under DWLS) and Worland nonconvergence stay open for those
  models; see the decisions/07 report.

- [x] **M — diagnose DWLS nested profile-law excess rank** (task-17.3).
  Three production-seed cells, 100 draws each, isolate separate-point profile
  cancellation and positive-tail truncation. Common-point spectra equal the
  ten-term observed-Hessian parameter-space IJ law; the latter needs registered
  fresh-seed confirmation before replacing a reference law. Policy unchanged;
  see the study's `diagnostics/nested_profile_law.md`.

- [x] **S/M — remove the ordinary fixed-x option under the adopted API**
  (2026-10-02). Ordinary construction uses the joint random-X model and
  rejects lab specifications with `fixed_x = TRUE` and observed covariates.
  ML structural estimates match the fixed-x fit; GLS estimates of an
  overidentified regression differ, as the
  [scope](../scope.md#ordinary-fixed-x-decision) records. Regressions on
  observed covariates get the ML policy's inference. General fixed-design
  inference remains
  [banked](speculative.md#fixed-design-inference-under-mean-misspecification).

- [x] **S — correct equality-constrained categorical fit measures** (task-62).
  Lab reporting uses ordinal/mixed moment counts minus the equality-reduced
  dimension, retaining native categorical n F reporting. Live DELTA/THETA
  group-equality and label-equality gates cover df and all df-based indices;
  continuous group/label equality reporting is checked alongside them.

- [x] **M — expose named downstream inference conventions** (2026-10-02).
  `lavaan_compat` on `vcov`, `confint`, `summary` and `anova`; C++ compatibility
  composers, on-demand reporting and optional `infer(fit, lavaan_compat)` caching.
  Complete-data ML/MLM/MLR and all-ordinal DWLS/WLSMV, ULS/ULSMV and WLS
  covariance/global recipes; ML nested defaults. The
  [capability inventory](../validation/capabilities.md) records checked slices.
  **Check:** installed-lavaan whole-bundle comparisons, typed unavailable
  reasons, incompatible-regime rejection and preserved estimates/default policy.

- [x] **S — repair grouped all-ordinal compatibility covariance (task-104).**
  Preserve n_g/N bread/meat fractions and use N-G only as the covariance
  denominator; n_g−1 remains the objective count. Unequal-group HS delta/theta
  WLSMV has retained/identical-point 1e-5 relative covariance gates and an
  independent expected-bread sandwich check. Removed the standardization
  covariance substitution.

- [x] **M — extend checked reporting conventions to FIML and ordinal nested
  tests** (2026-10-03, task-7.2 and task-14). Both reuse existing C++
  algorithms and lavaan features: FIML standard/robust covariance, global MLR
  and nested engines, and the ordinal Satorra-2000 exact/delta engine,
  scaled-shifted reducer and lab adapters.
  FIML ML/MLR covariance/global/default nested bundles are complete (task-14),
  with observed-Hessian covariance and trace-based YB-Mplus scales pinned
  against installed lavaan; they do not use spectrum-derived SB scales.
  Ordinal nested reporting is composed and live-gated (task-7.2): WLSMV/ULSMV
  use delta plus scaled-shifted Satorra-2000, n_g−1 objective counts and original
  n_g/N sandwich fractions. Plain DWLS/ULS retain statistic/df without p; WLS
  uses the standard difference. Single-group delta/theta, two-group theta
  loading and Wu-Estabrook threshold→threshold+loading invariance, saturated
  alternatives and typed refusals are covered at doctest-relative 1e-5.
  **Check:** live lavaan defaults, full covariance and intervals, statistic/df/
  p-value/scale/shift, grouped incomplete FIML and ordinal delta/theta invariance,
  both model orders, actual nesting, saturated alternatives, convergence and
  penalty refusals, deferred/cached/serialized reporting, and preserved
  estimates/default policy. Extend the [inventory](../validation/capabilities.md#reporting-gap-audit-2026-10-02).
  Continuous LS, two-stage, mixed and additional historical bundles remain
  consumer-gated. This does not block primary policy composition or silently
  broaden 0.2.0's exit criteria.

- [x] **S/M — close validation gaps in existing reporting bundles.** Grouped
  complete-ML loading/intercept/mean nested defaults are gated against installed
  lavaan 0.7.2 `lavTestLRT` (standard / SB2001, both model orders). Grouped
  ordinal ULS/ULSMV/WLS covariance/global gates cover delta/theta loading
  equality with each group's n-minus-one normalization, and theta threshold
  equalities. Delta threshold equality stays limited because lavaan's released
  delta scale is not identified. Interior PSD, boundary, active-bound and degenerate
  scales have separate API-specific limited/unsupported inventory entries.
  See the [inventory](../validation/capabilities.md#ordinary-reporting-conventions).
  Future FIML/ordinal nested bundles retain these test obligations; PSD hardening
  remains 0.3.0.

- [x] **S — decide the compatibility selector's name and reporting label** (2026-10-02).
  Adopted `lavaan_compat = NULL` for the ordinary policy default and explicit
  historical bundles such as `"MLR"` or `"WLSMV"` for comparison. Renamed
  reporting arguments, inference metadata, result attributes and caches together;
  output says lavaan compatibility. The unreleased argument has no alias.
  Default policy, compatibility rules and computed values are unchanged.

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

0.2.1 completes the preset (TASK-129, user 2026-10-08; design in
[the R-interface vision](../design/r-interface-vision.md#advanced-fitting-choices)).
lavaan's `meanstructure = FALSE`/`fixed.x = TRUE` defaults are deliberately
not emulated: comparisons run lavaan with `meanstructure = TRUE` and
`fixed.x = FALSE`. The constraint-violating standardized retry stays an error.

- [ ] **0.2.1 — marker switch for complete-data ML (TASK-129.1).** A `marker`
  fitting-options component (`"default"`, `"lavaan-0.7.2"`) set by the preset;
  lavaan 0.7.2's `bad.marker.crit = 0.1` rule in C++, a builder marker map with
  `lav_pt_flat` semantics, switched fit with all attempts and revert, and the
  fit reporting the identification actually fitted.
- [ ] **0.2.1 — report lavaan's `post.check` (TASK-129.2)** for every fit under
  lavaan's acceptance rule, separate from `converged` and admissibility.
- [ ] **0.2.1 — marker switch for FIML and ordinal/mixed DWLS (TASK-129.3)**,
  reading lavaan's saturated (h1) covariance of each route.
- [ ] **0.2.1 — hard-case parity gate (TASK-129.4).** Opt-in comparison with
  live lavaan at small N (Heywood cases, non-convergence, retries, marker
  switches), structured means with equality constraints, random covariates,
  FIML, DWLS and two groups, with pre-registered classes: no rule differences
  may remain; floating-point path divergence is reported per cell.

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
  pre-existing endpoint sensitivity in `fitting/lavaan_0_7_2.json`; their
  verdicts are build-dependent too.

- [x] **Task-47 — guard opt equality retry comparisons** (2026-10-02).
  Prerequisite and attempt-count failures cannot trigger out-of-range access
  under `-fno-exceptions`. The ×100 equality witness has path-dependent PORT
  endpoints: opt first-attempt gradient maximum 0.00094775120123813394 accepts,
  while lavaan's 0.0016149511731821235 retries at the unchanged 0.001 threshold.
  Approved validation keeps exact starts/search coordinates and derivatives
  at identical points for every case, checks each actual endpoint's acceptance
  rule, and retains retry/endpoint/verdict parity for path-stable cases only.
  No thresholds, tolerances, implementation or fixtures changed.

- [x] **M — fit FIML under the lavaan preset.** Implemented 2026-10-02:
  pinned diagonal/SQUAREM EM H1 supplies loading/location starts;
  available-case variances with the group ML divisor supply residual starts
  and standardized retry scales. The observed-pattern objective uses half
  the H1-relative deviance, ordered equality QR coordinates and pinned PORT
  controls/retries/acceptance, retaining native diagnostics and every attempt.
  Frozen MCAR/MAR and installed-lavaan gates cover single/multiple groups,
  shared labels and `group.equal`, invalid starts and x100 rescaling. The x100
  final gradient vector is path-dependent (opt max `1.03649e-5`, oracle
  `2.44914e-4`, threshold `1e-3`): follow the approved task-47 identical-point
  derivative and per-endpoint acceptance contract without changing tolerances;
  estimates/objectives, retries and verdicts still match. Nonzero affine RHS,
  nonlinear constraints and unsupported preset routes remain explicit errors.

- [x] **M — fit all-ordinal DWLS under the lavaan preset.**
  C++ `fit_ordinal_configured()` supplies sample-threshold/FABIN3 starts,
  unit response scales, theta residual starts, the DWLS objective and group
  weighting in lavaan's units, ordered equality coordinates and four-attempt
  PORT retries. Frozen delta/theta, grouped, invariance and invalid-start
  fixtures gate starts, coordinates, gradients, estimates and verdicts.
  Native DWLS remains on its existing path. Both R packages now route through
  the configured entry, with live installed-lavaan WLSMV comparisons for
  delta/theta, grouping, loading/threshold invariance and invalid-start retries.
  ULS/WLS presets remain unavailable.

- [x] **M — complete mixed DWLS under the lavaan preset (TASK-91).**
  The shared configured engine supplies mixed FABIN3/sample starts, ordered
  equality coordinates, `(n_g-1)/N` search weights and pinned PORT stopping.
  Eight frozen/live HS cases cover delta/theta, one/two unequal groups and
  configural/restricted pairs (zero factor covariance or equal loadings).
  Prepared `estimate()`, `fit_model()` and ordinary `magmaan()` retain matching
  endpoints; TASK-90's WLSMV retained covariance/global/nested gates pass at
  their existing tolerances for seven retained cases; TASK-94 gates the
  path-unstable grouped theta configural case by objective acceptance and
  identical-point reporting. Native outputs are bit-identical
  across the eight fits. Mixed missing data, nonlinear constraints and finite
  bounds remain explicit errors; ULS/WLS presets remain unavailable.

- [x] **Task-59 — named simulation parity set (exit 2).** The opt-in
  `MAGMAAN_PARITY=1` test above uses fitted public lavaan datasets as normal
  populations, fixed seeds, 20 replicates per case and N=300 per group.
  Ordinal indicators use four categories at common population quartiles;
  MCAR removes 15% per variable and MAR removes x2/x5/x8 depending on
  x1/x4/x7. Seeds are `590000 + 100 * case_index + replicate` in named list order.
  The per-parameter fixture gate remains
  `abs(a-b) <= 1e-5 * (1 + max(abs(a), abs(b)))`.
  The approved alternative for non-retry endpoints requires identical
  convergence/verdicts, both maximum gradients <=1e-3 in lavaan units,
  lavaan statistics (2N times the objective) agreeing within 1e-6 and every estimate difference
  <=1e-3 times its same-fit lavaan standard SE. Rescaled retries retain
  their separate task-47 contract. Contract-route counts and metrics print
  explicitly; no replicate is dropped and every disagreement fails the test.
  The [optimizer capability record](../architecture/capabilities/optimizers.md)
  retains seed 590214's trace and same-point derivative evidence supporting
  this decision. No fitting implementation change was needed.
  The installed-lavaan 0.7.2 gate passes all 160 replicates (about 35 seconds).
The split between routes varies slightly between builds because the drift
is rounding noise: in the merge run (2026-10-03) 157 pass estimates and three
PoliticalDemocracy seeds (590202, 590203, 590210) pass the endpoint
alternative, with chi-square differences at most 3.3e-8, gradients at most
1.7e-6 and SE-scaled differences at most 1.7e-4; all other models use zero
endpoint alternatives. All verdicts agree and no rescaled retry occurs. An
earlier 1e-9 relative objective condition proved build-sensitive: near an
optimum the objective gap is second order in the estimate difference, so it
is judged on the statistic's scale.

### Primary inference workflows

0.2.0 completes the ordinary policy for ML, FIML and all-ordinal DWLS.
Ordinary SEs are not automatically valid at singular PSD endpoints; the policy
keeps its current boundary contract (computed, assuming an interior
population), and further PSD inference work follows in 0.3.0. The adopted
ordinary API exposes barrier fitting experimentally as
`covariance = barrier(lambda)` and reports its inference as unavailable.

- [x] **S — expose effective pEBA block counts.** C++ FMG, policy and lab results
  report actual nonempty eigenvalue blocks from the existing partition. Ordinary
  global/nested reporting retains the `peba4` reference row and prints one footnote when fewer
  than four blocks formed, including df=5 or 6 (three blocks). Computed tails
  are unchanged; targeted checks cover df 1, 2, 3, 4, 5, 6 and 8.

- [x] **M — uniform ordinary test tables and simulation references (task-95).**
  Summary/anova expose one row per test/reference with typed unavailable rows;
  alternatives reuse the policy statistic/spectrum via C++ FMG. Lab calibration
  accepts the shared fixed-name and EBA/pEBA block grammar. Default p-values,
  compatibility metadata and nested recovery are retained.

- [x] **M — primary-policy test recommendations (task-96).**
  Recommended rows select only the primary policy test and its default laws;
  LR remains reported with its caveat. Global/nested gates cover ML, FIML,
  all-ordinal and mixed DWLS, including exact native policy p-values.

#### ML and FIML

- [x] **S/M — PEBA4-only ML/FIML defaults (TASK-97).** Decision 2026-10-06:
  global and nested score/LR select PEBA4; SB remains an explicit comparator
  through `references = c("sb", "peba4")`. Only score/PEBA4 is recommended.
  Complete-data geometry evidence and the FIML alignment are recorded in the
  interface vision; DWLS defaults remain unchanged.

- [x] **M — validate remaining likelihood-score component contracts in the primary sampling scope.**
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
  no queued extension. `policy_score_contracts_test.cpp` independently gates
  complete/MCAR/MAR free-direction stationarity, observed bread, raw sandwich
  meat, null-score projection, one/two-group equivalent constraints, two-group
  complete-data nested score reduction, a single-observed-variable MAR pattern
  and an actual Heywood PSD boundary. Constraint comparisons use one common
  evaluation point and relative 1e-8 tolerance; stationarity uses half the
  terminal 1e-3 deviance-gradient tolerance. These are component checks, not
  additional calibration. Penalty-specific inference follows in 0.3.0.
  Missingness patterns are not sampling groups.

- [x] **M — investigate ML/FIML sandwich and Wald coverage gaps** (2026-10-02,
  documented, exit criterion 4). The shortfall is sandwich-variance error, not
  point bias: in the [centering confirmation](../../experiments/decisions/03-score-centering/report.qmd)
  (`scripts/coverage_decomposition.R`) bias is at most 0.11 empirical SDs and
  intervals with the empirical SD cover 93.9–95.7% in every ML/FIML cell, while
  at N=80 the reported variance averages 0.87–0.89 of the empirical variance
  with skewed data (0.92–1.01 normal FIML) and its spread (SE coefficient of
  variation 0.28 versus 0.13–0.17) costs most of the 4–5 coverage points. Both
  shrink by N=300. The native construction matches lavaan's
  `robust.huber.white`. No small-sample correction is adopted; a bias-corrected
  meat or effective-df t reference is [banked](speculative.md#small-sample-distribution-free-intervals-for-covariance-functionals-kauermann-carroll).

- [x] **M — decide how the policy reports small-sample LR tests** (2026-10-02,
  user decision, exit criterion 4). The score test is primary: it comes first
  in `summary()` and `anova()`, and the likelihood-ratio row follows with a
  printed caveat about over-rejection when N is small relative to its df.
  Evidence and rationale are in the
  [inference policy](../design/r-interface-vision.md#inference-policy).

- [x] **M — reconcile nested restrictions and reusable inference** (2026-10-02).
  The structural-path "constant disagreement" between `score_components(H1=)`
  and `policy_nested()`/`lavTestScore()` was not about structural paths: the
  model was covariance-only, and the complete-data score components fixed the
  mean at zero. Profiling the mean (board TASK-44) removes it; on the
  research/52 Worland design the pre-fix library gave 1.1313 against lavaan's
  1.0645, the fixed one 1.0645 (regression test in `test_nested_embedding.R`).
  Mean-aware metric-to-scalar nesting with released latent means already works
  through the shared embedding: on HolzingerSwineford1939, metric versus
  scalar gives lavaan's LR difference 40.059 (6 df) and configural versus
  scalar 48.251 (12 df); `anova()` gates both in r-magmaan. Larger-model
  misspecification calibration belongs to 'calibrate the observed nested ML
  geometry'; FIML scalar nesting arrives with the FIML policy.

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

- [x] **M — calibrate the observed nested ML geometry** (2026-10-03).
  [decisions/04](../../experiments/decisions/04-nested-ml-geometry/report.qmd),
  36,000 draws, 24 cells (correct, mild and strong misspecification of the
  larger model). Under strong misspecification the expected-geometry LR rejects
  10–16% of true nulls and barely improves with N; the observed geometry reaches
  6.4% at N = 300 per group (normal). The observed score is conservative at
  N = 100 (2–3%), which tripped the registered reporting rule; under the
  misspecification-robust requirement that calls for finite-sample work, below.

- [ ] **M — finite-sample calibration of the observed nested score.** The
  observed-sensitivity nested ML score rejects 2–3% at nominal 5% with 100
  observations per group (decisions/04), where the inconsistent expected
  sensitivity holds about 5%. Find a correction that keeps consistency under
  misspecification (a corrected reference law, or multiplier/bootstrap
  calibration). **Check:** decisions/04 cells plus a larger-df family; size and
  size-adjusted power; no expected-sensitivity fallback. Not release-gating.
  Task-73 replay (decisions/04 diagnostics): 200 production draws per small-N
  correct/mild normal/skewed cell implicate fitted projection/metric variance
  compression rather than uniform meat inflation. TASK-93 adds the
  [finite-sample design](../design/nested-score-finite-sample.md)
  and [research/54 pilot](../../experiments/research/active/54-nested-score-small-n/report.qmd):
  recentered fixed/refitted-geometry bootstrap, 40 fresh draws/cell, B=199,
  9.9 minutes; eight bootstrap convergence failures affect three refitted tests.
  Wide intervals preclude a calibration claim.
  Next: freeze size/power registration with strong misspecification and ten
  restrictions; validate an exactly recentered objective before including it.
  No default change; production requires a later compute decision.

- [x] **M — compose the FIML policy.** Implemented C++ global/nested
  composers and R dispatch: observed-bread casewise-score covariance,
  observed-H0 score sensitivity with expected metric, saturated observed-H1
  global LR, and direct larger-fit empirical-score nested LR via the exact
  restriction map. SB/PEBA4 and typed per-component reasons are retained.
  The existing transported-influence lab LR route remains unchanged.
  See [FIML capability detail](../architecture/capabilities/fiml.md) for gates
  and the limited research/44 and decisions/03 calibration evidence; no new
  confirmatory calibration is claimed.

- [x] **M — pin FIML robust conventions before claiming parity.** Task-14
  composes observed-Hessian ML and Huber–White/Yuan–Bentler Mplus MLR
  reporting, with default SB2001 differences using the trace-based scales.
  Installed-lavaan MCAR/MAR, grouped invariance, missing random-X and saturated
  gates cover covariance/intervals/global/nested bundles and refusal reasons.
  The [inventory](../validation/capabilities.md) pins the recipes; this is
  compatibility evidence, not target-regime calibration or a policy change.

#### All-ordinal DWLS

- [x] **S — keep slow jackknife gates off Debug checks** (TASK-87).
  Nine numerical gates carry the per-case CTest `slow` label; Debug check,
  quick checks and CI exclude it, while optimized merge validation runs all
  cases. Assertions, tolerances and the complete test count are unchanged.
  Debug CTest wall time fell from the TASK-31.2 baseline of 2021.14 s to
  338.08 s (1541 selected cases, two workers); full opt CTest passed all
  1550 cases in 318.78 s.

- [x] **M — speed the exact first-stage score Jacobian** (TASK-83).
  Block-sparse FD differentiates only affected marginal/pair mean scores;
  ordinal category counts and shared bivariate-normal corners avoid repeated
  casewise assembly. Dense FD remains a private test reference. Jacobian,
  sampling-row and Gamma gates retain 1e-7 tolerance for ordinal/mixed,
  binary/five-category, skewed-threshold and one/two-group data. Public
  functions and inference policy are unchanged; timing evidence is recorded
  in the ordinal capability area.

- [x] **M — adopt exact first-stage influence for the ordinary policy** (TASK-69).
  Covariance and both nested laws use Exact estimated-weight IJ; global Gamma
  uses exact sampling rows cached once per policy fit. OPG fitting weights,
  estimates, lavaan compatibility and lab defaults are unchanged. Global All
  and nested SB/PEBA4 references remain. Earlier decisions/05 calibration
  applies to OPG. TASK-79 supplies a preregistered exact/OPG paired
  reconfirmation runner, full-grid local pricing pilot and preemption-safe
  Modal support. Production ran on simbox (2026-10-06, frozen in
  decisions/05 `results/exact-first-stage/production-2026-10-05`): global All
  3.1-6.05% and nested All 4.0-7.3% with no size flag at N >= 500; IJ
  coverage median 94.85%, converged-draw minimum 92.9%; the only coverage
  shortfall is nonconvergence in the two-group binary N = 300 cell, identical
  under OPG.
  TASK-84 adds the registered latent non-normality lane, streamed pair-table
  million-row pseudo-targets with independent Monte Carlo checks, exact/OPG
  paired arms and a bounded local pricing pilot; production ran on simbox and
  is frozen (TASK-127): OPG inconsistent for skewed five-category items, exact
  calibrated, binary identical. Exact delta/theta covariance transport agrees within 1e-6;
  tighter delete-one refits leave discrepancies unchanged. The theta jackknife
  design uses N = 2400 (1.32% error), retaining the 4% tolerance; delta designs
  retain N = 600. The N = 600 theta error (5.23%) reflects finite-sample
  reparameterization nonlinearity. The Satorra comparison uses the same exact
  Gamma and retains both original gates.

Mixed continuous/ordered completion is assigned to 0.3.0. Shared fixes
required by an all-ordinal primary workflow remain current work.

- [x] **S/M — compose the DWLS policy covariance and global test** (2026-10-02).
  `api::policy_inference_dwls`: the IJ covariance (`robust_ordinal_ij`, which
  now takes the fit-time `row_user`) and one global test, the fit-function
  statistic n F with the fixed-weight `robust_ordinal` UGamma spectrum, exact
  weighted chi-square All reference (task-17.5), reported in `score` with label
  `fit_function`; `lr` is the typed
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

- [x] **S/M — compose the DWLS nested likelihood-ratio-type test** (task-17.4).
  `api::policy_nested_dwls` reports unchanged T = n(F_null − F_alt) in `lr`
  (`fit_function_difference`), calibrated with SB and PEBA4 on exactly
  df_diff parameter-space estimated-weight IJ terms. Observed Hessian at H1,
  IJ covariance in the constraint coordinates (using K's left inverse), and
  `embed_nested_null`'s exact restriction map compose the reference. The
  separate-point `ordinal_dwls_profile_lrt` remains an explicitly named lab
  comparator. Nested score stays typed `unsupported_model`. C++ and lab gates
  compare against common-point profiles and the task-17.3 diagnostic; statistic,
  nesting refusals and true-null fixed-weight reduction are preserved.
  Delta/theta statistics agree, but their H1 affine restriction tangents need
  not coincide away from the nested null. Confirmed on fresh draws (decision
  study 05, 4.3-7.8%). Study 05 saves spectra and the reference family, retains profile
  and fixed-weight comparators, and confirms registered nested cells 53–84 and
  139–146 with seed base 817160001. Modal selects global, nested or both.

- [x] **M — calibrate the DWLS policy** (2026-10-03, task-17).
  [Decision study 05](../../experiments/decisions/05-dwls-policy-calibration/report.qmd)
  ran production (146 cells, Modal), a 13-reference exploration on the
  production seeds and two registered fresh-seed confirmations. The IJ
  covariance is calibrated (median coverage about 95% for every target). The
  global SB/PEBA4 references over-reject at df >= 53; the user chose the exact
  spectrum tail (All), confirmed at 2.9-6.8%. The separate-point nested
  profile law had an O_p(N^-1/2) cancellation artifact (78-373 terms for 10-15
  restrictions); the r-term parameter-space law (observed Hessian,
  estimated-weight IJ meat, exact restriction map) replaced it and was
  confirmed at 4.3-7.8% with SB/PEBA4. Out of scope and reopen triggers:
  heavier-tailed latent responses; threshold-invariance restriction maps.

- [x] **M — derive the misspecification-consistent moment choice for lab
  robust meats (TASK-77).** Empirical ML SE/score/MI defaults now use exact
  likelihood rows, sharing TASK-66's mean-shift and uncentered group constants;
  raw/Zc/Gamma gates agree with finite-difference scores. Explicit structured
  and unstructured weights remain comparators. Global U-factor/test-moment
  builders retain correct-null conventions. All 19 audit Q rows are resolved;
  derivations and supported sampling contracts live in
  `project/validation/lab_inference_defaults.md`. Continuous LS retains its
  sample-centered fitting-weight law: the grouped fixed-allocation influence
  is gated, while joint allocation under nonzero group estimating-equation
  means remains a documented limitation (banked fixed-design scope).
  TASK-75 registers
  decisions/04 structured-mean calibration, with frozen full-grid 20-draw pilot
  and cost. The original intermittent R/native conversion failure remains
  counted; 201 authorized replays passed. TASK-76 adds two-worker Modal cell
  fan-out for both decisions/04 lanes. The registered structured-mean
  production (Modal, 2026-10-05, 108,000 draws, no failures) returned no flag:
  the exact-row LR never lost to the pre-66 meat and was better with skewed
  data at N = 100. The complete-data ML policy's restricted-mean case is
  closed; the lab moment drivers remain (TASK-77).
  No production or default change in the lane. Not release-gating.

- [x] **M — threshold-invariance nested tests (Wu-Estabrook)** (TASK-68).
  The DWLS policy now falls back from failed parameter nesting to a numerical
  moment embedding and null-tangent inclusion check. The H1 observed Hessian
  and estimated-weight IJ meat give q1 − q0 terms; parameter-nested gates agree
  with task-17.4 within 1e-10 relative. Two-group theta configural versus equal
  thresholds is testable with 5 categories and reports typed `equivalent_models`
  with 3 categories. Thresholds+loadings versus loadings-only remains
  `not_nested`: released item response scales change the standardized loading
  ratios constrained by the loadings-only model. C++/R policy and ordinary
  `anova()` gates cover the threshold step. A 100-replicate correct-model check
  at 1000 observations/group agrees with the trace and variance within Monte
  Carlo uncertainty; this is limited validation. decisions/06 production
  (2026-10-05, 576,000 draws) flags SB/PEBA4 with seven categories (7.0–8.0% at
  N >= 500 per group) while All stays within 3.5–6.0%; the fresh-draw All
  confirmation passed (3.45–6.0% in all 144 null cells). The nested reference
  choice was adopted on 2026-10-06 (TASK-81): All for every DWLS nested
  spectral test, with SB/PEBA4 comparators retained. decisions/05 fresh-draw
  parameter-nested All confirmation gives 3.9–6.9% at N = 400 and 4.3–6.4%
  at N = 1000; decisions/07 confirms it across eight textbook models (15 of
  120 nested cells outside [3%, 7%] at N >= 500, tied fewest; SB 23).

- [ ] **L — derive a nested DWLS score test, or defer it.** No joint
  least-squares nested score statistic exists; `score_tests_robust_joint` is
  ML-only. Specify sensitivity, metric, nuisance projection, weight influence
  and group normalization, then derive and calibrate; alternative score
  weights need a recorded decision study before adoption. Until then the DWLS
  policy reports the nested score as unavailable with a typed reason, which
  meets the 0.2.0 exit criterion; move this item to 0.3.0 if it is not needed
  sooner.

- [x] **M — finish primary reusable inference ownership** (TASK-23).
  Owning `api::FimlPolicyFit` / `DwlsPolicyFit` snapshots lazily retain the
  evaluation-point bread/scores and IJ/Hessian results, respectively, including
  failures. Fit-owned R contexts reuse them for ordinary covariance, summary,
  global and repeated nested policy calls; serialization and PID changes rebuild
  transparently. Exact-output gates cover MAR, theta, groups and nested pairs.
  Delta/theta nested LR already uses the shared restriction embedding.
  Concrete residue outside this policy slice: shared ML2S/global score geometry;
  categorical `score_components`/resampling adapters need a separately derived
  joint score contract (nested DWLS score is already deferred above). Embedded-null
  score geometry remains pair-specific. See the
  [workspace contract](../design/ordinal-snlls-gamma-architecture.md).

With `missing` removed from the ordinary call, pairwise DWLS stays a lab route;
the capability inventory lists its policy inference as unsupported.

### Ordinary MI reporting (0.3.0)

- [x] **TASK-99 — ordinary modification indices and equality releases.**
  C++ policy composer and thin lab/ordinary adapters; robust one-df score
  statistics by default, optional embedded-start nested-policy LR refits;
  candidate selection/order, standardized EPCs and typed unavailable rows.
  ML/FIML and all-ordinal/mixed DWLS are covered.
- [x] **TASK-105 — absent structural regression candidates.** Shared continuous/
  ordinal candidate enumeration includes reverse paths among equation variables;
  augmentation rebuilds structural roles and matrix representation while keeping
  data and fitted-parameter coordinates. SEM/path/grouped lavaan inventory and
  normal-theory gates, explicit robust-score gates, LR/anova and ordinal SEM
  checks cover the unchanged policy. Unidentified paths retain typed rows.

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
| Continuous ULS/GLS/WLS | Shared moment-quadratic MI; fixed- and estimated-weight robust primitives | Identity, NT, diagonal/full empirical, DLS(a) and supplied block W: R uses the recorded fitting W and recipe/a for empirical/model-implied covariance and the weight influence (supplied W refused there); caller-Gamma adapters are exposed; remaining work audits sensitivity/nuisance projection |
| All-ordinal ULS/DWLS/WLS | Ordinary and robust threshold/association MI; estimated-weight DWLS/WLS path | Gate identity, diagonal/full NACOV, retained Stage-2 DLS and supplied weights with delta/theta and group conventions; shared relative rank and caller-NACOV adapters are implemented; broader provenance remains |
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

- [x] **M/L — complete weighted MI/release provenance and adapters.** Caller
  Gamma/NACOV adapters are complete (TASK-4.1). Observed-sensitivity scores
  with estimated-weight influence now cover continuous ULS/GLS/DWLS/WLS/DLS
  and all-ordinal ULS/DWLS/WLS (TASK-4.2), using exact objective Hessians,
  observed nuisance projection, and the expected quadratic metric. Unequal-group
  case-weight finite differences gate the pooled IJ normalization. Unknown
  supplied-weight recipes require fixed-weight inference; ordinal NT/DLS
  estimated-weight scores remain explicit unavailable cells. Complete mixed
  DWLS/WLS MI/releases now compose exact sampling rows, mixed Gamma influence
  and observed sensitivity (TASK-92), with independent augmented-score and
  one-restriction reconstruction gates. Association-ML
  MI remains a 0.3.0 contract. The maintained
  [component matrix](../validation/capabilities.md#mi-and-equality-release-score-components)
  records the comparator and supported-default regimes.

- [x] **S — preserve user-written rows in C++ API ordinal post-fit inference.**
  `api::robust_ordinal`, `fit_measures`, `modification_indices` and
  `score_tests` pass the fit-time `row_user` mask through all-ordinal and mixed
  ordinal preparation; automatic ordinal/mixed starts use the same mask.
  Regressions cover fixed non-default and free explicit residual variances
  (all-ordinal theta and supported mixed delta), fit-objective consistency,
  absent-row MI, and equivalent explicit/implicit default spellings. R glue is
  unchanged.

- [x] **S — complete lower-level ordinal preparation provenance.** Preparation
  stamps the ordered-indicator set and binary vetoes on `LatentStructure`;
  repeated calls preserve the free set, constraints and starts or reject a
  changed preparation layout. The lavaan projection and R fit partable retain
  this header metadata, distinguishing a prepared released `~*~` from an
  explicitly fixed scale. IJ, RBM/casewise, Satorra-2000 and DWLS policy nesting
  accept fit-time row masks for unprepared inputs (board TASK-37).

- [x] **M — separate ML/FIML robust MI sensitivity and metric (TASK-106).**
  Observed nuisance sensitivity with expected metric/bread preserves saddle
  candidates, uses expected-information EPCs, and retains typed numerical
  failures. HS and PoliticalDemocracy complete/MAR candidates have explicit
  projected-meat gates; positive observed-curvature statistics reduce to the
  previous observed/observed result. LS/ordinal numeric recipes are unchanged.

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
  MLR MI oracle; ordinal ULS/WLS use reduction/rank gates. Mixed MI's
  factor-two defect is fixed (TASK-33.4): ordinary/fixed robust MI and releases
  have independent frozen-moment Schur gates, unchanged EPC and explicit
  finite-divisor oracle transport; raw mixed DWLS MI is gated at 0.005. Component closure does not establish new calibration or defaults.

### Release readiness

- [x] **S/M — inventory validated primary capabilities** (2026-10-02). The
  [primary inventory](../validation/capabilities.md#primary-inventory-020)
  records the ordinary policy per setup and component (gate, calibration, open
  task), domains, penalties and fit states, estimation/verdict/preset state and
  cross-cutting ownership; `test_capability_inventory.R` gates the ordinary
  states and typed reasons. Composing a new policy (FIML, task-13) or adopting
  calibration evidence (DWLS, task-17; nested ML, task-43) must update its rows
  and that test together. Exit criterion 6.

- [x] **M — broaden primary CI checks.** Use appropriate `R CMD check` instead
  of hand-picked R tests; add scheduled sanitizers/optional parity and an
  interpretable coverage artifact. **Check:** clean-source portable installs
  and mount-independent default tests, avoiding unexplained percentage gates.
  See [local hardening](../validation/local_hardening.md).
  Completed 2026-10-02: `just r-cmd-check` checks both portable source packages
  (both Status OK, no NOTEs); primary CI uses it, and weekly/manual hardening
  adds dev sanitizers, fixture parity and LLVM domain coverage artifacts.

- [x] **S/M — release check dry run (task-31.1).** `just check` passes
  (1,486 Debug C++ tests, R examples and ordinary-package tests), and the opt
  parity label passes (20 tests). Both portable packages finish `R CMD check
  --no-manual` with Status OK and no NOTEs. Fixed stale help defaults, a duplicate
  Rd details section, an unqualified `stats::setNames`, and a socket-unavailable
  PSOCK test skip that preserves local serialization coverage. The worker test
  also passes with sockets available. Versions and publishing remain pending.

- [x] **S/M — release dry run 2 and NEWS consolidation (task-31.2).** On the
  integrated TASK-32.3 baseline, `just check` passes (1,550 Debug C++ tests,
  all R examples and 1,980 ordinary-package assertions); opt parity passes
  32 tests. Both portable packages finish `R CMD check --no-manual` with
  Status OK and no NOTEs. Fixed the lab IJ help usage, made all seven live
  DELTA model declarations self-contained in the source package, and updated
  the score example to create a distinct FIML snapshot despite fit-owned
  caching. Both 0.2.0 NEWS sections are grouped and retain the in-development
  heading; earlier OPG calibration evidence is distinguished from pending
  exact-first-stage reconfirmation. The capability inventory gate passes
  52 assertions; stale wording proposals are recorded on the board without
  promoting validation statuses. Versions and publishing remain pending.

- [x] **S/M — refresh release help (task-112).** Summary fit measures now
  document lavaan-compatible output including FIML; policy help records PEBA4
  for ML/FIML and All for DWLS, exact first-stage influence and completed
  reference-law evidence. Policy intervals remain pending evaluation.

- [x] **S — align versions and publish (2026-10-07).** CMake and both
  DESCRIPTION files at 0.2.0 with consolidated NEWS; `main` and the tags
  `v0.1.0` and `v0.2.0` (`52471911`) pushed. CI green on the tagged commit;
  the local release check passed opt ctest including `parity`, the Debug
  build, both testthat suites, examples and `R CMD check` of both packages.

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

- [x] **S/M — make lab estimated-weight comparisons consistent.**
  `frontier_rbm(estimated_weight = FALSE)` supports supplied-W continuous
  fits and fixed-weight ordinal/mixed and ML2S corrections. All lab
  estimated-weight switches default to TRUE for misspecification-robust
  inference; explicit FALSE retains fixed-weight comparisons. Estimated-weight
  requests retain recipe-resolved corrections and supplied-W refusal. NEWS
  lists the routes; examples exercise the default. Ordinary policy
  defaults are unchanged.

- [x] **M — audit lab inference defaults (TASK-61).** The
  [inventory](../validation/lab_inference_defaults.md) records robust generic
  defaults, explicit compatibility/diagnostic routes and typed unsupported
  cases; ordinary implementation is unchanged.
- [x] **M — derive structured versus saturated moment evaluation for lab robust
  covariance and nested components** (TASK-77). All 19 audit Q rows in
  [lab_inference_defaults.md](../validation/lab_inference_defaults.md) are
  resolved: empirical ML SE/score/MI defaults use exact uncentered likelihood
  projections; structured/unstructured remain explicit comparators; GOF moment
  primitives keep their named conventions; grouped centered LS meats are
  fixed-allocation.

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

- [x] **TASK-88 — sparse mixed estimated-weight diagonal influence.**
  Contract ordinal outer products, continuous 2x2 blocks and association rows
  against precomputed transformed factors. Retain dense WLS as the test
  reference; binary/five-category and one/two-group mixed/all-ordinal gates
  require 1e-10 relative agreement. Policy and smoke timings are recorded in
  the task completion summary.

- [x] **TASK-89 — sparse Gamma-diagonal kappa movement.** Extend the
  complete-ordinal category-count FD to pairwise-missing data, including
  singleton-observed marginal contributions. Mixed complete/observed FD uses
  incident item/pair subsystems with the original step rule and global marginal
  PD cutoff. Dense callable references gate binary/five/seven-category,
  skew-threshold, one/two-group cases at 1e-8 relative agreement; policy/IJ/MI
  and jackknife tests retain their existing assertions.

- [ ] **M — profile cell-score cost and repeated post-fit reuse.** Measure
  multi-category work before another algorithm change; retain reusable influence
  and cache ownership. **Check:** equivalent ingredients, setup/fit/reporting
  costs and peak memory on complete and supported mixed/missing primary slices.
  See [benchmark guide](../../benchmarks/README.md) and workspace contract.

### Ordinal weight storage and workspace cleanup

- [x] **TASK-123 — skip unused full-WLS inversion in DWLS inference workspaces.**
  Raw complete-data `FitPlusInference` DWLS retains full Gamma and DWLS weights
  without constructing the WLS inverse. Cache WLS availability requires a
  nonempty payload, including when a requested inverse fails on singular Gamma.
  Focused workspace gates cover full-Gamma agreement, deferred inverse creation,
  singular Gamma with valid diagonals, lazy fitting and robust cache reporting.
  Direct fit-only materialization and diagonal storage remain open below.

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

- [x] **S — textbook WLSMV DELTA parity regression (TASK-63).** The
  implied-correlation gate reads live prepared `~*~` response scales and
  applies `delta_i * sigma_ij * delta_j`; THETA retains diagonal
  standardization. The old free-residual mask belonged to the retired DELTA
  translation. Fitting semantics, fixtures, tolerances and known gaps are
  unchanged; the 19 supported textbook cases remain the regression gate.

- [x] **M — split FIML implementation and ordinal unit tests (TASK-42).**
  FIML preparation, curvature, Gamma influence, moments, robust reporting,
  measures, Stage 1/2, two-stage inference, profiles, fitting and frontier
  methods now compile separately with private shared declarations. Ordinal
  unit tests compile by topic with shared test helpers; public headers and
  numerical bodies are unchanged, and vendors are refreshed.

- [x] **M — split R fitting glue (TASK-40).** Topical estimation, ordinal,
  missing-data, inference, robust, measures, frontier, profile, noniterative
  and prepared-interface files share private helper declarations and a utility
  translation unit. Exported signatures and numerical behavior are unchanged.

- [x] **M — split ordinal estimation implementation (TASK-41).** Preparation,
  moments, curvature, fitting, robust/IJ covariance, score tests, fit measures
  and nested tests now compile separately, sharing a private internal header.
  Public headers and numerical behavior are unchanged; vendors refreshed.

- [x] **S — revalidate the retained Bell alternative-CFA controls (TASK-115).**
  The bounded current-package check covers the low-reliability higher-order
  population's four CFA candidates and the low-reliability one-factor control.
  All five returned converged, admissible endpoints, with original objectives
  reproduced on re-evaluation; historical evidence and initial diagnostic
  lookup failures are retained separately. This closes the selected-control
  validation, not every historical cell. Reference optima and identification
  remain unresolved: independently verify any claimed optimum before extending
  or using model spread as evidence. See
  [reliability targets](../../experiments/showcases/08-reliability-targets/report.qmd).

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

- [ ] **S/M — export named corpus gaps.** At-theta implied moments for
  `newsom_2015_ex9_3` and `little_2013_ch3_fig_3_6_1indicator` are frozen in
  compact pinned-lavaan fixtures (TASK-119); their 1e-8 gates run without the
  optional mount and independently check keyed theta alignment. Deterministic
  export retains source hashes and provenance without observations. Remaining:
  restore corrected Little/Newsom fit gates after the documented failures are
  handled. **Check:** provenance, independent oracle
  values, no third-party data and file-size limits. See translation audit and
  [Newsom failures](newsom-corpus-failures.md).

- [x] **S — explain Mplus `chapter6_ex6_10` ULS statistics.** Same-data
  lavaan 0.7.2 reproduces `fmin=0.0383293433492224`, standard ULS
  `2N*fmin=38.3293433492224`, and the default reported Browne NT statistic
  `-1.86144433200752e-11`. The former backlog wording reversed the quantities.
  Fixed-x NT metric and tangent both have rank 14, leaving a zero residual
  projection despite nonzero full-moment residuals; see the TASK-117 entry in
  [oracle-defects.md](../validation/oracle-defects.md). **Check:** convention
  reproduced; any rank/df or inference-policy change needs a separate scoped
  investigation, without changing this fixture or its acceptance.

- [ ] **S — finish admissibility fixtures/checkpoints.** Complete-data oracle
  warning-status phase is banked: the synthetic three-indicator ML pair retains
  lavaan convergence, post-check warnings, keyed estimates and implied covariance.
  At-oracle-theta C++ gates separate negative residual admissibility from PD
  implied Sigma and retained backend convergence. Carry audit flags into
  experiments when their runners next change; composite attachment is deferred.

- [x] **S — validate `sqrt` in defined parameters.** EBNF, parser and AD
  already support it. Fixed-theta effects gates check values, correlated delta
  variance, nested-definition propagation, negative/varying-zero typed errors
  and constant zero. An independent Mplus ACE twin-expression construction checks
  SQRT lowering and value/variance semantics; the constraint AD gate checks its
  Jacobian. This closes the implementation/semantic item, not source-fidelity
  validation of ex5.21/ex5.22: no checked-in gate for those exact inputs was found.

### Benchmarks

- [ ] **S/M — maintain comparisons needed by current decisions.** Separate
  setup/fit/inference, pair numerics before speed claims and retain objectives,
  verdicts, evaluations, memory and failures. Remove unused fixture fields only
  after checking readers; retire duplicated timing loops when touching runners.
  **Check:** clean-source reproducibility and matched computation. General new
  grids and paper timing programmes are deferred. See benchmark guide and
  [speed attribution](../../experiments/showcases/06-speed-attribution/report.qmd).

## 0.3.0: magmaan's own fitting, mixed data and barriers

- [x] Ordinary standardized estimates and R-squared (task-98): opt-in summary
  columns and endogenous-variable table, C++ row-level map and delta SEs with
  the selected covariance; fixed markers, groups, equality constraints and
  ordinal latent-response scales. Defined standardized scales remain explicitly
  unavailable.

Assigned 2026-10-02. Until these land, simulations that need dependable
fitting can use the lavaan-compatible preset from 0.2.0.

### Optimization and convergence

- [x] **S/M — seed the larger nested model from the restricted fit.** Under
  strong misspecification the larger model sometimes converges to a worse local
  optimum than the restricted model (7 of 12,000 strong-cell draws in
  [decisions/04](../../experiments/decisions/04-nested-ml-geometry/report.qmd);
  lavaan reaches the same endpoints), so the LR difference is negative and the
  nested test reports a typed failure. Starting the larger model at the
  restricted estimate (embedded) supplies a feasible starting endpoint.
  Implemented explicit native/lab `refit_from_null()` and automatic ordinary
  `anova()` recovery for policy and lavaan compatibility comparisons. Recovery
  requires an improved objective no worse than H0 and a passing native verdict;
  it records and prints the refit and leaves input fits unchanged. Internal
  refit warnings are captured in `reseed$warnings` and printed as notes;
  explicit lab refits retain fit warnings. **Gates:**
  reconstructed failing draw, unchanged well-behaved pairs, and ML/FIML/DWLS
  refits with retained data and original fitting options.

- [x] **TASK-33.10.5.4.1 — additive objective-coordinate scale primitive.**
  Equality-reduced sample units and scalar-objective GN diagonal determine
  bounded two-sided or downward-only scales, with typed input/output failures.
  Independent residual geometry and unit transport gates cover the helper;
  parent TASK-33.10.5.4 retains fit wiring and opt-in selection.

#### Numerical banking programme (2026-10-04)

Board TASK-33.10 groups the remaining numerical work by shared objective
mechanism and route integration. Banking means retained, reproducible evidence
for a declared numerical scope with failures, unchecked cases and reopening
triggers. It does not mean ordinary default adoption, statistical calibration,
global optimality or completion of every constraint/domain combination.
The sphere-study branch (through 1fe7792f) was merged into main on 2026-10-07
(TASK-33.11). The merge adds opt-in audits only and changes no default;
"isolated sphere-study" in the rows below records where each card was banked.
Existing adapters and regression gates count as evidence;
new studies fill concrete gaps rather than duplicate an estimator-by-weight grid.

| Card | Remaining outcome | Evidence to share and additional checks |
| --- | --- | --- |
| 33.10.1 | Banked 2026-10-05 in isolated sphere-study: fixed-weight LS mechanism and complete-data producers | Identity/diagonal/dense/NT construction bounds cover complete means, unequal groups, linear equalities and sphere maps. Fresh 78-fit/150-point confirmation: all 136 available bounds/intervals and 64 recipe checks agree with 90-digit references; 72 identified fits qualify, six redundant controls remain failed/unchecked. ADF/DLS Gamma inversion now removes moment units before the rank gate; deficient Gamma still rejects. No default adoption, first-stage error propagation or basin guarantee. |
| 33.10.2 | Banked 2026-10-05 in isolated sphere-study: ordinal/mixed numerical audit integration | Retained whitened Jacobian, actual factors and direct observed correction reuse QR curvature/metric. Fresh 160 fits/252 independent points and 46 threshold/conditional-weight checks agree in delta/theta, including groups, shared loadings and mixed units. All fresh endpoints pass; 80 displaced points and 12 saddles reject. Six earlier sparse-theta failures remain correctly rejected and feed 33.10.6. Existing first-stage/pairwise oracle gates count; propagated construction bounds remain unsupported. No inference/default/sphere/PSD/barrier expansion. |
| 33.10.3 | Banked 2026-10-06 in isolated sphere-study: direct-FIML numerical integration | Owning observed-pattern artifacts rebuild from serialized raw fits. All 70 fresh and 37 corrected development points agree with independent 90-digit likelihood/gradient/observed-curvature calculations; actual pattern counts/means/covariances and eight complete-data ML reductions agree. Fresh actual-default fits pass 16/16, SLSQP 15/16; one unit endpoint and eight early-stop errors remain recorded. Earlier unit/weak-marker failures and two SLSQP budget errors feed 33.9/33.10.5. The proved scale ridge can pass the raw local test and remains explicitly unchecked for identification (33.3). No propagated construction interval, raw-sample normalization extension, inference/default/PSD claim. |
| 33.10.4 | Banked 2026-10-06 in isolated sphere-study: conditional ML2S numerical composition | R Stage-1 retains raw derivatives/repair and solver-stop provenance; fits capture caller-unit Stage-2 moments/counts/recipe/mixing/ACOV/supplied weights/bounds/transformation before fitting. The opt-in composed audit preserves separate Stage-1/handoff/Stage-2 reports and unchecked legacy/missing evidence. Fresh 156 stage-point, 210 conditional producer and 78 handoff checks agree at 90 digits; all 12 Stage-1 endpoints and 56 returned fits pass (54/60 ordinary plus two transformed controls). Six mixed-unit ULS/DWLS/ADF L-BFGS errors, three development errors and early/mismatched/repaired controls remain recorded. NT uses ML; other weights use their original quadratics. Repaired ACOV at nonpositive raw curvature stays unchecked and never rescues an audit. No propagated input-error/inference/default/PSD claim; unit recovery feeds 33.10.5. |
| 33.10.5 | Partially banked 2026-10-06 in isolated sphere-study: pinned starts and NLopt candidate retention | Fresh 51-fit confirmation: 28 equality/fixed fits agree with controlled-unit lavaan and original likelihoods; seven early stops retain failed candidates and exact controls; six ordinary defaults pass. All ten retained mixed-unit ML2S fits return, but six ULS/DWLS/ADF endpoints still fail separate Stage-2/composed audits. R mean ownership and automatic inference gating are repaired. Frozen failed reference/crash/tail runs remain visible. Generic domain recovery, complete fallback/backend histories, outer-loop stopping and broader held-out recovery remain open; no default or PSD/barrier adoption. |
| 33.10.6 | Banked 2026-10-06 in isolated sphere-study: explicit local theta boundary limit | All six failures reproduce exactly; actual starts are identical across settings. Independent 90-digit Schur-profiled loading coordinates verify three locally optimal faces with strict outward descent and 15 finite approach points. Negative-residual delta solutions have no finite positive-residual theta transport; tighter L-BFGS recovers none. Fresh 324 fits/point checks pass from 18 usable draws, with two singular first-stage sparse draws retained. No global nonattainment theorem, recovery/default promotion or audit relaxation; native distance precision beyond the conditioning guard remains unchecked. |
| 33.3 | Identification separate from local numerical accuracy | Implemented 2026-10-07 in lane/task-33.3: a data-free generic-rank check of the moment Jacobian fails every verdict of a structurally unidentified model and names its null directions. Both witnesses (the PSD-fallback free-marker CFA and the direct-FIML ridge, which passes the counting rule) are rejected; identified/equality/unit-change controls keep their verdicts. Pre-fit refusal in `magmaan()` awaits a user decision. |
| 33.6.7 | Accurate unresolved sphere ML interval | Attribute the conservative numerical term; improve only justified bounds or document why it remains unresolved. Preserve saddle/rank/above-budget failures. |
| 33.7 | Banked: PORT/PORT-NLS endpoint consistency recovery | All 38 retained reported/original mismatches are fixed (maximum original-objective gap 2.03e-16 relative to 1+abs(f)); all 38 common/native audits remain failed. Both adapters recompute the returned objective and recover the best evaluated point beyond 1e-14 relative, preserving raw stop/stored/returned-point telemetry without changing verdicts or controls. Twelve fresh endpoints are consistent with zero objective/audit losses; ten unchanged fresh fits and six healthy ULS/GLS/WLS PORT controls retain bit-identical estimates. Scalar bounded/unbounded PORT exposes 32 mismatches among 100 baseline endpoints; all 68 exactly matching endpoints remain bit-identical. Frozen before/after evidence: engineering/active/15-sphere-reference-fits report lane port_nls. No default promotion or basin-success claim. |
| 33.9 | Bounded sample-only starts beyond the two-factor design | Retain API defaults and reuse actual-start comparisons. Include applicable layered/FABIN unit fallbacks, Geiser start loss and reflection traps. The four signed ULS starts are development evidence, not a universal recipe. |
| 33.6.6 | Scoped audit/default adoption decision | Prespecify supported scope, full no-options baseline, numerical criteria, every loss and cost. A covariance-only ULS/ML decision can precede other families; broader promotion needs its applicable completed gates. |

Use three levels of evidence: independent fixed-point mechanism checks,
bounded fresh route confirmations, then larger decision studies only for a
proposed default or concrete failure needing broader diagnosis. Report local
audit accuracy, objective consistency, chart availability, admissibility,
identification and best-known basin recovery separately. All decisive audit
claims must agree with their independent checks; any missed construction bound
or unexplained verdict blocks banking that claimed verification scope. An
unresolved accurate endpoint can remain explicitly unchecked rather than force
a looser acceptance rule. Confirmation retains every baseline loss and cost;
unsupported regimes inherit no pass. Freeze the confirmation recipe before its
fresh draws, and use a prespecified decision study for default selection.

Fixed in the fitting parameters is distinct from estimated from a sample.
Conditional Stage-2 numerical evidence is reusable only with the actual moments,
weight recipe and factor supplied to that fit. ML2S requires a separately checked
Stage-1 and handoff; separate stage passes are not a propagated numerical-error
bound. Ordinal and pairwise moments need their producer/count/weight checks;
they cannot be treated as a synthetic complete dataset for sampling inference.
SE/test variants with identical fits share numerical evidence, while TASK-69,
TASK-71 and the inference backlog retain their own statistical questions.

Primary single-level ML/FIML/all-ordinal DWLS lead. Supported mixed and secondary
fixed weights enter through inexpensive reuse. This programme does not schedule
two-level, SAM, FC-SEM or noniterative expansion, or PSD-face/barrier work; their
existing gates and separately owned scope remain. Work order is shared correctness
witnesses and weighted-LS verification first, with starts and the ML interval
investigation independent; ordinal and ML2S then reuse the weighted mechanism,
while direct FIML follows its pattern-likelihood checks. Adoption remains a
separate final decision for each explicitly covered scope.

- [ ] **M — complete the minimal ordinary-fit reliability programme.** Start
  with complete-data unrestricted NTML, ULS and GLS. Pair marker/sphere routes
  on identical moments; vary route-native versus shared sample-only FABIN3/layered
  starts, and stock versus tighter L-BFGS/PORT stopping controls. Keep the
  library's native Newton audit fixed and record failed/unchecked endpoints,
  chart availability, backend stops, objective and implied-covariance gaps,
  and cost separately. **Check:** the bounded development pilot in
  [sphere references](../../experiments/engineering/active/15-sphere-reference-fits/report.qmd),
  then targeted failure replays and fresh draws before any default decision.
  The matched layered replay gains one sphere ULS and one GLS fit in the pilot
  and recovers 3/6 qualified retained ML minima versus marker's 1/6 with default
  L-BFGS. Warm diagnostic ML fits recover all six; cold search still misses
  qualified minima. The sphere sample-unit repair (2026-10-02) clears retained
  mixed-unit NTML/GLS failures: 12/12 configurations per estimator, with all
  48/48 paired configurations also passing and matching on four fresh regular
  draws. The internal gauge latents are dimensionless; LS and PORT now honor
  requested rest-coordinate scaling. The pilot loses no accepted fits; the
  hard-case replay retains two shared-FABIN3 PORT ML acceptance gains and two
  losses, with layered ML target recovery unchanged. The mixed-unit ULS probe
  independently establishes five finite strict local minima at 60/90 digits;
  warm diagnostic fits recover all five in both charts, but all ten fail the
  unchanged sandwich conditioning guard (metric conditions 1e16–1e18, despite
  Hessian conditions below 1e12). Three implied covariances are indefinite under
  this unrestricted target. Existing PORT-NLS reaches three references from
  most cold configurations; two lower signed-variance basins remain missed.
  The sphere audit now distinguishes failed accuracy-metric factorization from
  failed Hessian curvature, without changing acceptance. Next study LS
  preconditioning/damped steps and bounded sample-only signed-basin starts.
  The 2,400-fit fixed-target follow-up recovers all five retained and 20 fresh
  regular/weak-marker references with sphere PORT-NLS and four sample-only
  signed-moment starts (seven gains, no losses against one shared FABIN3 start).
  Profiling improves L-BFGS but does not outperform that portfolio. All 25
  exact references and all native fit audits still fail certification; no
  default follows. Point checks now specify infinite bounds, preventing the
  evaluator's positive-variance preset from substituting first-order verdicts.
  The square-root metric implementation now computes all 100 fixed-point
  distances, including 25/25 minima, versus 48/100 and 12/25 previously.
  Independent 90-digit point calculations agree within 6.4e-7, with zero
  .01 distance classification errors on 75 accurate and 25 inaccurate controls.
  Full-rank QR retains the factor through ordinary and sphere coordinates;
  singular metrics and objective saddles remain failures. The production
  1e12 condition guard is unchanged. A diagnostic factor-condition rule
  retains 23/25 minima without accepting an inaccurate control; two weak-marker
  minima still exceed the Hessian cap. The 175-point guard follow-up adds
  75 near-budget controls: raw distance comparison flips 16 nominal threshold
  decisions, while conditional intervals using independently measured errors
  decide 140 correctly and leave 35 unresolved. Construction-error allowances
  remain assumptions; a 64-epsilon sensitivity covers observed input errors
  but leaves 83 controls unresolved. At 90 digits, objective-Jacobian QR
  coordinates reduce the two flat Hessian conditions to 1.2/1.6 and preserve
  all observed curvature. Core now assembles the analytic correction separately
  and implements the transported full-rank QR Newton solve, including sphere
  normalization curvature. All 175 fixed-point curvature checks pass against
  independent 90-digit derivatives; step discrepancy is below 1.94e-11 in
  objective-curvature units. Owning tests retain saddle and rank-loss failures.
  The sampling-metric condition guard still rejects every control.
  Conditional interval kernels now verify projection/solve arithmetic for LS
  and likelihoods, with 329 independent fixed-point checks including 15 fresh
  numerical minima and all seven finite NTML witnesses. Dimensional construction
  sensitivity covers the comparable points, but leaves the flat NTML minimum
  and many retained ULS controls unresolved; it is not a proved bound or a
  production policy. The calculated construction producer now independently
  encloses covariance-only ambient ULS/ML moments, derivatives, sample roots
  and observed curvature. All 314 available bounds pass 90-digit checks; all
  47 numerical minima qualify, including every retained ULS minimum and the
  flat NTML witness. Its 250 decisive distance classifications are correct,
  with 79 unresolved points including 15 independently nonpositive NTML
  curvatures. Opt-in terminal integration and native sphere
  transport now include the normalization's full derivative chain. Fresh sampled
  confirmation covers six families with 30 problems, 120 fits and 120 variance
  perturbations: all 216 available bounds cover and all 213 decisive interval
  classifications agree with 90-digit references. Endpoints yield 106 passes,
  eight saddles, three above-budget failures and three unresolved assessments.
  Native mixed-unit ULS gains three passes and loses none. One unresolved
  weak-marker ML endpoint is locally accurate; two mixed-unit ULS endpoints
  are above budget. The focused ULS search follow-up (2026-10-03) holds
  PORT-NLS/sample-unit scaling fixed and uses actual API default starts
  (FABIN3/native for ULS, not layered). On ten fresh draws per family, the
  sphere default-plus-four-signed portfolio qualifies all regular, weak-marker
  and mixed-unit cases and matches every best independently refined local
  reference. Marker portfolios qualify 10/6/8; tighter controls raise mixed
  units to 10 but leave weak markers at 6, with greater runtime cost. Keeping
  the default avoids a signed-only weak-marker loss. All 398 available bounds
  cover across 1,040 retained/fresh cold fits and 473 selected endpoint checks;
  no decisive audit classification is wrong. This unrestricted target includes
  negative residual variances and indefinite latent covariances. The
  two-factor-specific recipe needs broader-model start construction and cost
  evidence before adoption. Default acceptance is unchanged. Next prespecify
  adoption criteria and compare against the full actual-default fitter on
  held-out models; separately resolve the conservative accurate ML endpoint. This evidence selects no
  regularization or acceptance default. Retain the PORT curvature
  failures, investigate the 38 retained profiled PORT-NLS objective inconsistencies,
  and explain the shared-FABIN3 GLS worse local minimum. Confirm any signed-start
  sphere candidate on broader populations and ordinary units. Preserve LS-specific
  warm references without treating a fitted ML point as an LS optimum.
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

- [x] **M — warn about continuous LS measurement-unit spread (TASK-33.10.5.3).**
  Core diagnostics record the maximum within-block continuous variance ratio;
  ULS/DWLS/WLS/ADF advise rescaling only above 1000. Complete-data, ML2S
  Stage-2 and mixed LS share the diagnostic, surfaced by both R packages.
  ML/GLS/FIML, NT/DLS and ordinal-only fits receive no advice. This diagnostic
  does not change search coordinates or certify numerical accuracy.

- [ ] **M — recover from L-BFGS domain aborts across parameter scales.** Limited
  line-search reductions can exhaust infeasible trials at the initial point.
  Assess safeguarded backtracking or adapter recovery while retaining caller
  controls and terminal candidates. **Check:** the
  [domain probe](../../cpp/tests/checks/nlopt_lbfgs_domain.c), corrected corpus
  failures and rescaled fits. A larger evaluation budget alone is insufficient.
  The isolated shared-adapter repair now retains finite failed candidates,
  raw NLopt codes and resolved controls; it does not retry or lengthen the
  line search. The scalar variance witness at $10^{-6}$ remains a domain abort.
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
  The isolated sphere-study repair recognizes fully pinned parameters as
  constants in the layered measurement/scale layers, rather than projecting
  only the final vector. Experiment 15 now checks equivalent fixed/equality
  formulations on the optional Table 10.3 corpus and fresh rescaled draws.
  The original supplement's particular draw ($\chi^2=271$) has not been replayed;
  keep its completion check distinct from the repaired shared mechanism.

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

- [x] **S/M — reject unidentified exact-fit ridges reliably.** Implemented
  2026-10-07 (board TASK-33.3, lane/task-33.3): a data-free structural
  identification check (`estimate/frontier/identification.hpp`) ranks the
  route's moment Jacobian at seeded random points in linear-equality
  coordinates; an unidentified model fails every verdict, including selected
  compatibility rules, and names its null directions. The PSD-fallback and
  direct-FIML witnesses are rejected; identified, equality-constrained,
  multi-group, two-indicator, unit-changed and empirically underidentified
  controls keep their verdicts; no Newton threshold changed. Coverage, gap-rule
  calibration and the corpus sweep are in the
  [terminal audit](../design/terminal-audit.md#structural-identification-2026-10-07).
  Open: ordinal association ML, ordinal/mixed PSD and multi-information,
  two-level and FC-SEM routes report unchecked. TASK-33.3.1 adds immutable-model
  construction caching, the standalone lab report and ordinary refusal before
  fitting with `magmaan_identification_error`; unchecked models fit normally.
  TASK-33.3.2 adds generator-membership classification of scale, location and
  rotation freedom, respecting fixed cells and equality constraints, with
  specific suggested fixes in lab reports and ordinary refusals. Remaining
  null dimensions are information deficits without an automatic fix.

- [ ] **M — finish common-verdict and stopping-control integration.** Preserve
  evaluable candidates on soft exits, report effective controls/raw reasons
  separately from the verdict, and migrate active backend-status consumers.
  Assess ML stopping controls for FIML/DWLS and accuracy-budget sensitivity with
  estimated Gamma. **Check:** corpus near misses/non-minima and consistent R
  TRUE/FALSE/NA projection; preserve failed/unchecked fits and rank diagnostics.
  Isolated NLopt candidate/raw-control retention is banked, including early
  stops across complete LS/ML, direct FIML and ordinal routes; complete fallback
  histories, other backend telemetry and outer-loop evidence remain open.
  See terminal audit and
  [Newton rollout](../../experiments/engineering/active/19-newton-verdict-migration/report.qmd).

- [x] **M — bound noncentral chi-square tail cost in fit-index inference (TASK-107).**
  Retained the rejected mixed-unit ULS ML2S witness as literal C++ regressions.
  Mode-centered incomplete-gamma recurrences replace per-term gamma evaluation;
  geometric Poisson remainder bounds and explicit work limits return NaN when
  unavailable. RMSEA bisection and standard/scaled/robust consumers preserve
  unavailable intervals. The 60-digit mpmath grid (df 1..500, ncp 0..1e5,
  lower through upper tails) has maximum absolute error 8.8e-15. The old first
  lower-interval midpoint took 15.68 seconds; extreme witness noncentralities
  now return immediately; the full explicit witness call takes 0.002 seconds.
  Optimizer recovery and interval policy are unchanged.

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

- [x] **M — mixed DWLS WLSMV compatibility bundles (TASK-90).** Compose
  NACOV sandwich covariance/intervals, scaled-shifted global and lavaan-default
  Satorra-2000 delta nested tests in C++ and both R reporting interfaces.
  Complete one/two-group delta/theta Stage-1 and identical-point parity gates
  pass at 1e-5 relative tolerance; single-group delta has retained-fit gates.
  TASK-91 validates seven retained complete mixed preset delta/theta
  grouped/restricted cases; TASK-94 uses objective acceptance and identical-point
  reporting for the eighth, grouped theta configural. Native grouped/theta
  stopping differences
  remain recorded in the oracle observations ledger. Missing stats
  and uncomposed mixed bundles remain typed unavailable; policy is unchanged.

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
  TASK-67 gates complete mixed DWLS covariance and weight influence using a
  separate empirical score-Jacobian sampling channel while preserving lavaan
  NACOV fitting weights. Independent case weights, exact-fit weight-channel
  cancellation and stratified delta/theta jackknife shrinkage pass (2.5%
  diagonal tolerance at N = 1200 per group). Pure endpoint routes remain
  typed unavailable, and continuous fitting-Gamma conventions differ.
  TASK-71 composes exact-IJ covariance, the global exact-sampling All law and
  the observed-Hessian/IJ nested parameter-space law (including moment tangents)
  in C++ and both R interfaces. One/two-group delta/theta numerical composition
  gates give limited validation. TASK-92 composes mixed DWLS/WLS estimated-weight
  MI/releases with observed sensitivity and exact sampling rows; explicit H/B,
  replicated augmented scores, exact-fit cancellation and common-point nested
  identity gates cover one/two groups and delta/theta. Mixed calibration remains
  separate. TASK-80 registers the mixed lane in
  `experiments/decisions/05-dwls-policy-calibration/criteria/mixed_policy.md`;
  its pilot prices production without establishing calibration. Merger review
  precedes production. Other mixed influence routes retain their contracts.

- [x] **M — all-ordinal exact first-stage lab comparator (TASK-74).**
  Complete empirical score-Jacobian sampling rows and Gamma, plus explicit
  `robust_ordinal_ij(first_stage = "exact")`; OPG remains the lab default and
  ordinary policy is unchanged. Case-weight, delta/theta one/two-group and
  Gaussian convergence gates cover the comparator. TASK-69 owns subsequent
  policy adoption; the explicit sampling Gamma also supplies TASK-32's first
  association-ML building block.

### Association-ML inference

- [ ] **L — define association-ML MI/release and sampling inference.** The
  [audit and ordered contract plan](../design/association-ml-inference.md) is
  complete (TASK-32.1). TASK-32.2 implements the lab-only evaluation-point
  score, observed sensitivity and joint exact Stage-1 covariance, with independent
  derivatives, nonnormal case weights, stratified delete-one, exact-fit and
  identification-chart gates. TASK-32.3 adds lab global/nested All, SB and PEBA4
  reference laws with explicit spectrum/normalization, zero-df, exact-fit and
  misspecified pseudo-true restriction gates. TASK-32.4 adds lab fixed/absent-row
  MI and linear releases with observed Schur EPC and exact projected meat,
  independent reconstruction, derivative/refit and grouped gates. Policy
  calibration remains open (subcard 5). TASK-32.5 registers
  [decisions/08](../../experiments/decisions/08-association-ml-policy/report.qmd)
  with exact-IJ coverage, global/nested references and robust MI panels;
  production is frozen at `b37ebefb` under `results/production-2026-10-05/`;
  TASK-128.1 completes the report against the registered criteria. TASK-128.3
  passes the post-hoc MI population approximation check: maximum implied size
  5.364183% across 96 cells, below 5.5%; MI exposure is the next implementation step. Ordinary
  exposure and global/nested reference decisions remain pending user review.
  Ordinary association-ML components retain typed
  refusal until those contracts and their calibration pass.

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

### Mplus input frontend

Planned 2026-10-02 in the [Mplus plan](../grammar/mplus.md), which owns the
target, input boundary, increments, evidence and stability bar. magmaan reads a
whole Mplus input file and lowers its linear SEM model into the model triple
with Mplus's model defaults reproduced exactly. It imports no estimator
conventions and adds no fitting or inference scope. Development starts now,
alongside 0.2.0, without touching 0.2.0 release surfaces; it is not a 0.2.0
exit criterion. Each increment merges only when complete, so every merged state
is stable for its documented subset and rejects everything else. Board cards
carry the label `mplus`.

- [x] **M — source inventory and grammar baseline** (2026-10-03). The
  [source inventory](../grammar/mplus_source_inventory.md) covers the User's
  Guide v8 and the 8.1–9.1 addenda: 123 rules with printed pages and evidence
  classes, a classification of every command and option, and 41 Demo probes,
  29 of them for increments 1–2. The increment-1
  [grammar](../grammar/mplus_grammar.ebnf) is drafted. Findings that change
  the plan: the reference group is the lowest grouping value; `MODEL label:`
  without GROUPING can mean longitudinal time points (rejected); the guide
  contradicts itself on the marker with repeated BY statements and on label
  lists, which probes settle before increment 1.
- [x] **S/M — Demo probes** (2026-10-03). `cpp/tests/tools/regen_mplus_probes.R`
  runs 41 probes (100 variants) through the local Mplus 9.1 Demo and writes
  derived summaries to `cpp/tests/fixtures/mplus/probes.json`
  reproducibly. Every rule needed by increments 1–2 is now documented or
  Demo-confirmed. Where 9.1 departs from the guide (setting stems, name
  length, label lists, labels on fixed parameters, observed–latent residual
  covariances, NOMEANSTRUCTURE), the inventory follows 9.1.
- [x] **M — increment 1a: input reader/classification (TASK-51.1).**
  `parse::MplusParser::read()` owns source bytes and spans, expands NAMES,
  selects USEVARIABLES ranges by NAMES position, preserves selection order,
  imports supported ANALYSIS schema
  settings and reports every data-description/execution item. Unsupported
  commands/options produce aggregated rule-ID diagnostics with explicit
  later-increment boundaries. Independent rule tests and the 100-variant
  Demo reader gate encode the deliberate LX02, NM02, MS11 and CL11 deviations;
  MODEL statement parsing/lowering belongs to TASK-51.2.
- [x] **S/M — increment-1 independent golden fixtures** (2026-10-03).
  `cpp/tests/tools/regen_oracle_mplus.R` writes 13 hand-written paired
  Mplus/lavaan models to `cpp/tests/fixtures/mplus/golden.json`: explicit
  partables, starts, N-divisor moments, estimates, expected SEs and implied
  moments. Mplus 9.1 Demo agrees on parameter counts, df, chi-square and
  every printed estimate within the fixed tolerances. Population-quality
  gates reject negative variances and chi-square p-values below 0.001;
  regeneration is byte-for-byte deterministic. C++ consumption belongs to
  the increment-1 lowering card (TASK-51.2).
- [x] **M — increment 1b: MODEL statements and lowering (TASK-51.2).**
  Continuous single-group relations, pairing, means/variances, line-local
  labels/equality numbers, starts/fixing, ordered ranges and explicit Mplus
  role/default rules lower into owned flat rows. Reader refinements and
  rule-ID rejections accompany independent expected rows, the 13-model
  pinned-lavaan numeric gate, Demo TECH1 status/equality checks and the
  optional extracted local corpus sweep. Combined start/label modifiers
  accumulate through duplicate flat rows; no spec or matrix changes.
- [x] **M/L — increment 1: input file and single-group continuous models.**
  Command classification, BY/ON/WITH/PWITH/PON, means, `@`/`*`, labels and
  label lists, NAMES-order ranges and the single-group defaults; C++ lowering,
  `api::` constructor and `magmaanlab::mplus_model()`. **Check:** independent
  expected-row fixtures, rejection fixtures, Demo TECH1 probes, local corpus
  match on free-parameter count and df, lab round trips and pinned-lavaan
  numerics of the projection. TASK-51.3 adds source-preserving API/lab
  construction, canonical spelling, BY-set rejection and serialization gates;
  the local end-to-end gate matches all 15 accepted cases among 68 eligible.
- [x] **M — increment 2: multiple groups.** Explicit integer GROUPING, ordered
  source labels/codes, cumulative group sections, invariance defaults and
  overrides, asymmetric fixed-zero topology and generated provenance; one
  CONFIGURAL/METRIC/SCALAR with marker or variance identification. Group-specific
  variable-role changes are rejected. **Check:** 14 independent Demo/lavaan
  grouped goldens; numeric 1e-5 gates, API/lab rebuild/serialization and row-order
  checks; corpus 22/68 accepted and matched, sweep reader 253→291 and MODEL
  189→217 on the same 2,440-input manifest. TASK-52.
- [x] **M — increment 3: categorical outcomes.** Thresholds, delta/theta
  parameterizations and their multigroup defaults; all-ordinal DWLS fits,
  other categorical routes return unsupported-fit. **Check:** as increment 1;
  ten independent categorical goldens, live grouped/scale/prepared gates,
  P-IV2 TECH1 probes; corpus 31/68 accepted, 25 matched and six unsupported
  fit routes, with two verified convention differences among matches. Sweep
  reader 291→456 and model 217→359 on 2,440 inputs. TASK-53: ten Demo
  meaning gates pass; SE convention observations are recorded separately,
  while default-lavaan inference gates pass.
  TASK-53.1 prerequisite: DELTA response scales retain live lavaan coordinates,
  including equality labels, fixed non-unit values and linear constraints.
  Seven frozen WLSMV/DWLS/ULS cases and live ordinal tests gate estimates,
  scale SEs and tests; THETA retains its residual coordinates.
- [x] **M — increment 4: growth, MODEL CONSTRAINT and MODEL INDIRECT.**
  Polynomial/free-time/piecewise and categorical/group growth defaults; NEW
  auxiliary coordinates and definitions, affine/nonlinear continuous equality
  restrictions, DO expansion and shared sqrt/pnorm/log10; total/specific/VIA
  indirect effects with delta-method SEs. Inequalities remain the deliberate
  CN01 boundary with the PSD/barrier remedy. TASK-54.2 closes all-ordinal
  nonlinear LS fitting and expected-information lavaan inference; ordinary
  policy and observed/IJ sensitivity remain typed unsupported pending
  Lagrangian curvature. **Check:** 14 continuous and four categorical frozen independent
  Demo/lavaan models, live lab fits and serialization, active ML/LS/FIML
  restriction checks. Auxiliary-coordinate regressions cover ML/GLS/ULS/FIML,
  PSD/barrier, linear DWLS, covariance/standardization/policy and typed
  noniterative refusals; full C++ and both R suites pass. Corpus 31→50 accepted, 25→44 matched, six unsupported
  fit routes; sweep reader 456→658 and MODEL 359→614, all classified. TASK-54.
- [x] **M — increment 5: data files.** Data plan in C++ (including FORMAT) and
  `magmaanlab::mplus_data()`, FILE/summary groups and summary mean-structure
  lowering. **Check:** free and
  fixed format, summary data and missing codes against independently written
  data frames; saturated summary ML agrees with Demo and lavaan after
  group-specific N-1 to N divisor conversion.
- [x] **S/M — stability closeout.** Standalone ASan/UBSan reader/parser and
  API-lowering sweeps cover 2,440 original inputs with per-input deadlines:
  658 reader / 614 parser/API accepted; zero crashes, sanitizer reports, hangs
  or unclassified rejections. All 33 firing IDs have rule-specific explanations
  and remedies. One matrix covers 121 primary inventory IDs and settled aliases;
  lab help matches the accepted subset. Seven model kinds pass partable,
  fresh/prepared, rebuild/refit and fresh-process serialization gates. Corpus:
  50/68 accepted, 44 matched, six unsupported fits (also dimension-gated), five
  convention differences, zero failures. Evidence is in the test ledger. TASK-56.
- [x] **M — ordinary integration.** Explicit `magmaanlab::mplus_model()` specs
  enter `magmaan_model()` without source detection or semantic conversion.
  P-MS08b establishes that every x variance mention (`x1 x2;`) frees the joint
  x means, variances and default covariances; complete-data ML and missing-x
  FIML match Demo/lavaan. Partial mentions name the completion statement.
  Construction preserves inspectable tables and records fittability; fitting
  refuses conditional X, NOMEANSTRUCTURE and summary-without-MEANS with a
  classed reason and input edit. Prepared/fresh parity and continuous/grouped/
  categorical fresh-process reconstruction pass; ordinary defaults are unchanged.
  Full C++ and both R suites and the unchanged corpus gate pass. TASK-57.

- [x] **M — output-only corpus meaning gate (TASK-85).** Scan covers
  1,933 outputs / 683 distinct Mplus inputs; all 130 accepted inputs
  match groups, equality-reduced free counts and df. Later bracket
  segments retain labels/modifiers, gated by nine Demo variants and independent
  rows. Constraint rank and printed-category fallback complete counting;
  full frontend checks pass. See the test ledger.

- [x] **M — real Mplus data-file check (TASK-108).** Local gate reads all
  373 resolvable pairs; available references match, including 34 accepted
  verified User's Guide cases. Terminal DOS EOF handling fixes nine reads on
  two Brown files. Embedded markers are rejected instead of silently truncating
  data, with isolated Demo evidence and FREE/fixed synthetic regressions.
  Missing references/files and six rejected verified inputs remain explicit
  in the derived summary and test ledger; analysis-sample rules stay in the gate.

- [x] **S — ordinary Mplus refusal reporting.** One error lists all needed
  input edits in stable order, with vector reason/edit fields; model printing
  shows fittability and the edits. Full ordinary R suite gates the contract.
  TASK-78.

## Related work

- [Simulation backlog](simulation.md) owns generator/projection/calibration
  detail, including model-implied simulation and group metadata. Only primary-
  workflow dependencies belong in this queue.
- [Speculative register](speculative.md) owns parked methods, model families,
  expensive verification and optional reporting/packaging. Promotion needs a
  named consumer, bounded scope and completion check.
