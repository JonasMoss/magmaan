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
| **0.3.0, next** | magmaan's own fitting reliability (starts, optimization, convergence, PSD finalization, stress and normalization); mixed continuous/ordered workflows; barrier hardening and inference; association-ML MI; a stable Mplus input frontend for the linear SEM subset |
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
  support production planning; no production or default change is authorized.

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
  ULS/WLS/mixed presets remain unavailable.

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
  global/nested reporting retains `p.peba4` and prints one footnote when fewer
  than four blocks formed, including df=5 or 6 (three blocks). Computed tails
  are unchanged; targeted checks cover df 1, 2, 3, 4, 5, 6 and 8.

#### ML and FIML

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
  compression rather than uniform meat inflation. Next: register nuisance-
  refitting bootstrap or higher-order joint-moment correction; no default change.

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
  Modal support; production awaits merger registration and compute approval.
  TASK-84 adds the registered latent non-normality lane, streamed pair-table
  million-row pseudo-targets with independent Monte Carlo checks, exact/OPG
  paired arms and a bounded local pricing pilot; production remains pending. Exact delta/theta covariance transport agrees within 1e-6;
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
  choice is with the user (TASK-81).

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
  supplied-weight recipes require fixed-weight inference; ordinal NT/DLS and
  mixed estimated-weight scores remain explicit unavailable cells. Association-ML
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
- [ ] **M — derive structured versus saturated moment evaluation for lab robust
  covariance and nested components.** TASK-61 decisions retain existing
  moments arguments pending a joint bread/meat derivation. Resolve every
  inventory Q row before claiming either moment evaluation universally valid.

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
  TASK-67 gates complete mixed DWLS covariance and weight influence using a
  separate empirical score-Jacobian sampling channel while preserving lavaan
  NACOV fitting weights. Independent case weights, exact-fit weight-channel
  cancellation and stratified delta/theta jackknife shrinkage pass (2.5%
  diagonal tolerance at N = 1200 per group). Pure endpoint routes remain
  typed unavailable, and continuous fitting-Gamma conventions differ.
  TASK-71 composes exact-IJ covariance, the global exact-sampling All law and
  the observed-Hessian/IJ nested parameter-space law (including moment tangents)
  in C++ and both R interfaces. One/two-group delta/theta numerical composition
  gates give limited validation; mixed calibration and estimated-weight
  MI/release remain open. TASK-80 registers the mixed lane in
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
  the bounded registration/pilot card does not authorize production or exposure.
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
