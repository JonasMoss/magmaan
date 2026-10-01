# magmaan TODO

Accepted implementation work, ordered by workflow. The
[roadmap](../architecture/roadmap.md) owns current capabilities/contracts;
the [test ledger](../validation/test_ledger.md) and linked studies own evidence.
Remove completed items after their durable record exists.

**Scope adopted 2026-09-30; release priorities amended 2026-10-01:** single-level
ML, FIML and all-ordinal DWLS in their supported slices. **0.0.1 focuses on
ordinary and PSD estimation and inference; barrier-specific hardening and
inference and mixed continuous/ordered workflows follow in 0.0.2.** Secondary
estimators retain correctness gates;
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

The existing limited-information correlation-ML capability (currently catML)
is retained and folded into shared fitting composition. Common ordinary/PSD
fitting serves 0.0.1; remaining model-penalty work follows in 0.0.2. This does
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

| Milestone | Focus |
| --- | --- |
| **0.0.1 — current** | Ordinary and PSD estimation and inference: shared starts/units/constraints, reliable convergence and admissibility, PSD finalization/stress gaps, and validated parameter covariance, tests and intervals in supported primary slices |
| **0.0.2 — following** | Mixed continuous/ordered fitting and inference; barrier-specific hardening and inference: stricter domains, near-face curvature, factor disappearance, fallback/pole regressions and penalty-specific sampling/exposure contracts |

Continue with [fitting reliability](#fitting-reliability) and
[primary inference](#primary-inference-workflows). Shared fixes required by
ordinary/PSD routes remain 0.0.1 work even when barriers benefit. Existing
barrier code/tests remain; barrier-only validation is not a 0.0.1 release gate.
Mixed-data completion is assigned to 0.0.2; noniterative development has no
scheduled release. No milestone label promotes a new
ordinary-user estimator or inference default.

## Fitting reliability

### Optimization and convergence

- [ ] **M — extend the pinned lavaan fitting setup beyond its initial ML gate.**
  Independent starts/search/acceptance and the `lavaan-0.7.2` preset are wired
  for ordinary complete continuous ML; `newton` retains the native common
  verdict with internal thresholds. Gate pure-merge/general affine and
  augmented-Lagrangian coordinates, nonzero bounds and constrained retries,
  then primary FIML and all-ordinal DWLS. Retain effective controls, every
  attempt and separate native diagnostics; reject unsupported combinations.
  **Check:** pinned offline components and installed-version R comparisons of
  starts, search, final gradients, soft failures and retries, not just easy
  estimates. Inference conventions remain magmaan's policy. Consumer:
  simulation studies that need lavaan-identical fitting. Retry parity so far:
  a ×100 rescale matches lavaan's standardized retry exactly; at ×1000 and
  ×10⁵ starts, coordinates and verdicts match, but endpoints depend on
  floating-point paths (fixture `fitting/lavaan_0_7_2.json`).

- [ ] **M/L — decide how magmaan's verdict relates to inference under a
  non-native acceptance rule.** Under a compatibility preset such as
  `lavaan-0.7.2`, the selected acceptance rule sets `converged` and gates
  ordinary inference. This is deliberate: simulations compared with lavaan
  need lavaan-identical outcomes. magmaan's common Newton verdict remains in
  the fit diagnostics, and the two can disagree. On exact-fit moments with x1
  scaled by 1000 and x3 by 1/1000, lavaan 0.7.2 accepts a standardized-retry
  endpoint with fmin 0.276 (same fixture). Decide whether the inference status
  reports both verdicts, adds a detail when they disagree, or gates on both,
  without changing lavaan-identical estimates or `converged`. Expect threading
  through `api::policy_fit_state` and the ordinary summary. **Check:**
  disagreement cases in both directions, unchanged lavaan-identical fits, and
  documentation of the verdict that gates inference.

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
  primary all-ordinal and relevant SNLLS/IRLS paths; mixed extensions follow in
  0.0.2. **Check:** transported
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

- [ ] **M — distinguish requested-chart failures from inaccurate sphere stops.**
  Validate a general chart gate with the retained weak-marker pole witness;
  require a clean/audited driven stop before `magmaan_user_chart_singular`.
  **Check:** poles, finite extremes, runaways and numerical failures across units;
  alternative charts remain diagnostics. Failed multistarts do not establish
  nonexistence. Evidence: sphere references. Broader geometry and ordinary-user
  promotion are [deferred](speculative.md#sphere-chart-development).

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

## R simulation prerelease

Scope adopted 2026-10-01: prepare a versioned ordinary-user `magmaan` and
matched `magmaanlab` for simulations. Completion is API correctness and a
reproducible install; broader estimator or inference coverage is not a release
gate. EQS remains lab-only; ordinary-user integration follows the C++/lab
language-extension gates below.

The 0.1.0 API hardening and local release gates are complete, recorded in
the [roadmap](../architecture/roadmap.md#r-bindings-and-public-namespace-transition)
and package release notes. Completed checklist items have been folded into
those records. Remaining inference expansion stays under the workflows below;
it is not required to use this prerelease for supported simulations.

The library and ordinary-user inference policy remain unfinished. Beyond this
packaging milestone, the goal is to fix remaining bugs and complete and validate
coverage of the main estimators and inferential procedures. The active items
here track that work; completing the 0.1.0 gates does not complete it.

## Primary inference workflows

0.0.1 prioritizes ordinary and PSD inference in supported primary workflows.
Ordinary SEs are not automatically valid at singular PSD endpoints; require
validated sampling contracts or explicit unsupported results. Barrier-specific
covariance, tests and intervals belong to the 0.0.2 section below. The
adopted ordinary API exposes barrier fitting experimentally as
`covariance = barrier(lambda)`; availability does not claim validated
inference.

### ML and FIML

- [ ] **S/M — remove the ordinary fixed-x option under the adopted API.**
  Decided in the [scope](../scope.md#ordinary-fixed-x-decision): ordinary
  construction uses the joint random-X model. Reject lab specifications built
  with `fixed_x = TRUE`, with instructions, instead of silently replacing their
  model. Preserve compiled/lab compatibility conventions. Coordinate with the
  ordinary API task below. **Check:** random-X partable/fitting parity, ML
  estimates unchanged and LS changes as documented, direct/deferred and nested
  inference gates, typed rejection and retained metadata. General fixed-design
  inference remains
  [banked](speculative.md#fixed-design-inference-under-mean-misspecification).

- [x] **M — accept nested pairs written by dropping or fixing parameters.**
  Completed 2026-10-01: one inference-side parameter-key embedding lifts fixed
  and omitted paths into H1 slots, preserves affine offsets, and checks the
  null's implied moments. Score paths evaluate H1 at the embedded H0 point;
  all exact restriction-map consumers and SB2010 injection use the shared map.
  Complete-data ML/FIML also support interior nesting through moments with
  both tangents at a fitted common null point. Other estimator routes return
  `unsupported_nesting` for unavailable key correspondences; singular
  tangents/factor covariance receive `boundary_nesting`. Mixed-point delta is
  still the lavaan-parity option. **Checks:** dropped/fixed/equality spelling
  invariance at one numerical null point, frozen lavaan HS1939 score/exact-LR
  references, live R ML/FIML/ordinal/weighted/multigroup gates, pairwise
  composite and same-point/boundary controls. Independent optimizer runs
  retain their ordinary estimate tolerance.

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
  agreement. Observed-sensitivity calibration remains unresolved; it belongs to
  FIML integration below. The centering comparison is now banked in place, with
  no queued extension. Check excluded constraint/boundary component contracts
  independently; penalty-specific inference follows in 0.0.2.
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

- [ ] **M — reconcile nested restrictions and reusable inference.** Embed fixed
  values (`0*`) into larger-model slots; diagnose the structural-path constant
  disagreement between `score_components(H1=)` and `policy_nested()`/
  `lavTestScore()`. Support mean-aware metric-to-scalar nesting with released
  latent means. **Check:** Kline Worland restrictions, residual-covariance
  controls, cross-group equalities and independent restriction maps; assess
  larger-model misspecification calibration. See
  [scores](../../r-package/examples/scores.R) and
  [inference reuse](../../r-package/examples/inference_reuse.R).

- [ ] **M — make Satorra-2000 rank checks unit invariant.** Unscaled pooled
  information checks reject the identified Kline Lynam model. Judge rank in
  normalized/whitened geometry. **Check:** rescaling a variable by 100 preserves
  the statistic, with genuine singular controls. Evidence:
  [calibration battery](../../experiments/showcases/09-robust-test-calibration/report.qmd).

- [ ] **L — bring FIML into the ordinary policy.** Compose adopted observed-H0
  sensitivity and expected metric, with distinct score/LR SB and PEBA4 spectra.
  Extend evaluation-point-specific influence reuse and grouped coverage.
  **Check:** all-observed ML reduction, missing-data covariance/global/nested
  calibration and typed component-level unsupported reasons; missing inference
  must not refuse a fit. Evidence lives in the consolidated
  [global study](../../experiments/research/active/44-fiml-global-tests/report.qmd)
  and [invariance study](../../experiments/research/active/06-fiml-invariance-tests/report.qmd).
  The banked [centering confirmation](../../experiments/decisions/03-score-centering/report.qmd)
  adds 32,000 normal MCAR/MAR datasets: observed-sensitivity N=80 global PEBA4
  rejects only 0.8–1.8% across raw/centered arms, while nested MCAR rejects
  6.5% raw and 7.25% centered. Check finite-sample geometry/calibration against
  independent references; changing covariance centering does not resolve this.
  Freeze publication-model adaptations and estimator-level nulls before larger
  grids; preserve scalar-nesting and size-matched-power gaps. Legacy smoke
  comparisons and completed flip expansion do not queue new runs. See the R interface vision.

- [ ] **M — pin FIML robust conventions before claiming parity.** Resolve `sb_ml`
  bread/meat/H1 choices and convention dispatch; distinguish Yuan-Bentler
  variants from SB labels and retain FMG missing-data oracle limitations.
  **Check:** convention-matched values and target-regime/grouped calibration.
  Evidence: [definitions and matrix fingerprints](../../experiments/replications/08-savalei-falk-2014-test-conventions/report.qmd)
  and [calibration policy](../validation/calibration-parity.md).

### All-ordinal DWLS

Mixed continuous/ordered completion is assigned to 0.0.2 below. Shared fixes
required by an all-ordinal primary workflow remain current work.

- [ ] **L — compose DWLS policy covariance and global/nested tests.** Include
  weight-estimation influence. Global score and fit-function statistics coincide
  for the estimator's fixed weight; report once with calibrated SB/PEBA4.
  Specify nested sensitivity, metric, evaluation point, nuisance projection,
  centering and group normalization before adapters. **Check:** fixed-weight
  reductions, independent influence derivatives, jackknife controls and
  misspecified-larger-model calibration. Alternative score weights need a
  recorded decision study before adoption.

- [ ] **S/M — decide the flat-ridge ordinal golden gate.** Newsom 2024 ex1.3c
  passes the accuracy budget but differs in raw parameters. **Check:** a
  justified information-metric gate or tighter stop, with independently checked
  objective before removing `kKnownGaps`. Free-delta bound relaxation is
  separately [consumer-gated](speculative.md#categorical-scope-extensions).

- [ ] **M — finish primary reusable inference ownership.** Extend observed-bread
  covariance/score, delta-nesting and categorical influence adapters where
  independently validated. **Check:** centering, finite-sample/group scaling and
  retained geometry; equal statistics do not establish equal spectra. See
  inference reuse and [workspace contract](../design/ordinal-snlls-gamma-architecture.md).

### MI and release-score completion (0.0.1)

Adopted 2026-10-01 after assigning mixed continuous/ordered completion to
0.0.2 and indefinitely postponing noniterative work. This slice covers
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
| All-ordinal ULS/DWLS/WLS | Ordinary and robust threshold/association MI; estimated-weight DWLS/WLS path | Gate identity, diagonal/full NACOV, retained Stage-2 DLS and supplied weights with delta/theta and group conventions; close ordinary-rank and adapter gaps |
| Prepared all-ordinal correlation-ML | Retained catML fitting and shared association-target composition work; LS MI is not an ML-target score contract | Gate the active association Jacobian, saturated-threshold/NACOV transport and matching ML information before MI/release exposure; follow the shared fitting contract |
| Two-stage/ML2S | R NT-ML MI is a naive Stage-2 comparator; retained Stage-1 and weighted inference primitives | Corrected MI/release for NT, ULS, DWLS, ADF and DLS Stage-2 recipes, using Stage-1 influence and the applicable estimated-weight term; preserve actual ML versus quadratic discrepancy provenance |

The DLS/custom-weight rows concern bounded reuse of retained weighted primitives,
not a general DLS research programme. Pairwise moment sources retain their MCAR
scope and need their own covariance law; direct composite-likelihood, two-level
SEM, SAM, mixed-data and noniterative expansion are outside this slice.
Automatic absent-row enumeration covers cross-loadings and covariances;
structural-path enumeration remains a separate model-builder contract.

- [ ] **S/M — unify ordinary ordinal MI rank and candidate checks.** The separate
  ordinary ordinal worker retains an absolute efficient-information floor while
  the shared robust worker uses the relative check. **Check:** all-ordinal
  identification-only and genuinely identified fixed loadings, thresholds and
  equality releases across units, nearby fits and unequal groups; ordinary/
  robust candidate agreement in matching metrics, delta/theta parity and fixed-
  row/absent-row moment scales. Mixed controls follow in 0.0.2.

- [ ] **M/L — complete weighted MI/release provenance and adapters.** Cover the
  retained continuous and all-ordinal weight recipes in the matrix, with stored
  fitting W or explicit supplied W, Gamma/NACOV source, recipe/a and fixed versus
  estimated-weight influence. Derive consistent sensitivity and nuisance
  projection for observed/estimated-weight GMM score variants before exposing
  them; the existing expected-metric sweep alone does not establish that regime.
  Complete prepared all-ordinal correlation-ML MI only after the shared fitting
  contract defines its active association coordinates and NACOV transport.
  Audit every bread/information/covariance argument and expose applicable caller-
  Gamma paths through thin R adapters. **Check:** independent score, sensitivity,
  meat and weight-influence assembly; recipe endpoint reductions, retained-data
  versus supplied-data agreement, documented unavailable cells and no ignored
  options. No numerical recipe/default changes without evidence.

- [ ] **M — add Stage-1-aware two-stage MI/release tests across weights.** Reuse
  retained saturated moments and their joint mean/covariance influence for the
  NT, ULS, DWLS, ADF and DLS Stage-2 choices. Propagate weight-estimation influence
  where the recipe requires it, with fit-null versus test-evaluation provenance
  and observed/expected sensitivity declared. Keep the naive NT-ML comparator
  labelled separately. **Check:** all-observed reductions to the corresponding
  complete-data estimator, incomplete/grouped influence assembly, missingness
  pattern and unequal-group controls, DLS endpoints and typed unavailable cases.
  This is post-fit reuse, not a wholesale prepared-ML2S migration.

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

## API consistency and performance

### API and R boundary

- [ ] **M/L — implement the adopted ordinary API.** Add
  `magmaan_model(model, prototype, ordered, group, group.equal, group.partial,
  identification, parameterization)` and fit immutable prepared models through
  `magmaan(model, data, estimator, covariance, inference, options)`. Support
  zero-row grouped/ordinal schema frames; keep native structural preparation
  outside repeated fits. Every ordinary model carries a mean structure.
  `covariance` takes `"unrestricted"`, `"psd"`, `"barrier"` or
  `barrier(lambda)`; barrier fits are experimental (one session message,
  recorded and printed status, typed unavailable inference). Merge the top-level
  `start` and `options$starts` into `options$start` with one documented
  vocabulary, removing today's `"fabin3"` clash. Remove `fixed.x`, `missing`,
  `cluster` and `meanstructure` from the ordinary call, with a versioned
  migration; preserve row provenance. The syntax shortcut errors on undeclared
  ordered factors. Refits already replay every recorded `fit_model()` argument
  (2026-10-01). Extend prepared adapters instead of rebuilding partables per
  draw. **Check:** fresh/prepared parity, structural-preparation counters,
  changed-data starts/thresholds, schema, ordered-factor and fixed-x rejection,
  mean-structure invariance of the other estimates, SEs and tests, option
  precedence and start vocabulary, barrier λ validation, zero-λ reduction,
  session message and inference refusal, the documented distinction between
  changing the covariance domain and the objective, worker reconstruction, and
  separately timed small-model setup/data/fit/inference. See the
  [design](../design/r-interface-vision.md#ordinary-api).

#### EQS language extension

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

#### Shared fitting composition

Dependency order: moment-target foundation → ordinal association contract →
dispatcher/metadata → shared model penalties → sampling/inference gates. The
last gate is tracked under primary inference above. Current catML entry points
remain until equivalent replacement routes are validated; no task here removes
shared primitives or promotes an ordinary-user default.

- [ ] **M — complete and validate the moment-target foundation.** Extract
  covariance-to-correlation values/Jacobians into model code and compose
  covariance/correlation targets with one ML objective builder. Route ordinary
  and PSD catML through the shared map/kernel; preserve existing two-argument
  ML calls and results. **Check:** independent finite-difference derivatives,
  observation-unit invariance, input/domain errors and correlation-ML plus
  barrier composition. Foundation work is in progress; API presence alone is
  not completion. See the composition contract.

- [ ] **L — define the ordinal association-model contract.** Keep Stage-1
  thresholds saturated in this slice; remove inactive threshold/mean/scale
  coordinates from optimization and derive df from active association-Jacobian
  rank. Reject unsupported threshold/mean/released-scale constraints. Share
  correlation projections used by ordinal measures while preserving delta/theta
  semantics. **Check:** identification, overidentified/grouped/constrained
  controls and unchanged Stage-1 moments before expanding coverage.

- [ ] **L — unify fitting dispatch and composition metadata.** Separate moment
  source, model target, discrepancy/weight, covariance domain, model penalty and
  algorithm. Enable ML on prepared all-ordinal moments, ordinary or PSD, in C++
  and `fit_model()`; refits must reconstruct the same composition. Consolidate
  enums, R validation/defaults, stored labels and post-fit allowlists. Preserve
  or explicitly reject operations using provenance; moments alone do not supply
  an inference contract. **Check:** staged/convenience/refit equivalence,
  metadata/errors, unchanged numerics and thin R wrappers. Retire separate catML
  wrappers only after replacement gates pass. Mixed/polyserial ML's
  moments/mean/scale contract follows in 0.0.2; removed research APIs are not
  migration targets.

- [ ] **L — finish remaining model-penalty compositions (0.0.2).** Extend
  the scalar penalty wrapper/finalization across continuous ML/ULS/GLS/fixed
  WLS, pairwise MCAR and saturated-FIML moments, ordinal/mixed discrepancies and
  the direct observed-pattern FIML likelihood. Integrate derivatives, admissible
  starts, equalities, units and verdicts; validate primary combinations first.
  Store unpenalized discrepancy and penalized objective separately and audit
  the latter. **Check:** per-combination domain, strength/normalization and
  reduction identities; penalized sum-of-squares may require scalar optimization.
  Preserve inputs; a model barrier does not repair indefinite polychoric/pairwise
  moments. Saturated Stage-1 penalties require a separate target, propagation
  and H1-reference contract; latent determinacy is zero without genuine latents.

- [ ] **S — expose effective pEBA block counts.** A requested pEBA-4 is clamped
  to the test df and can coincide with scaled-shifted at df=1 while keeping its
  requested label. Decide an explicit diagnostic/reporting convention.
  **Check:** low-df rows identify the effective method without changing computed
  tails or silently mislabelling simulation cells. See the
  [FMG example](../../r-package/examples/fmg.R).

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
  0.0.2 and is exposed only
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

## Validation and maintenance

- [ ] **S — revalidate the retained Bell alternative-CFA controls.** The
  consolidated reliability showcase reproduces generating targets and main
  one-factor comparisons, but current-package population smoke changes some
  alternative-CFA solutions/convergence relative to the frozen sensitivity
  examples. **Check:** separate each model's convergence/admissibility from its
  objective and reliability target; independently verify any claimed optimum
  before extending or using model spread as evidence. Keep historical results
  distinct. See [reliability targets](../../experiments/showcases/08-reliability-targets/report.qmd).

### Capability inventory

- [ ] **S/M — review retained research capabilities one decision at a time.**
  Candidates: continuous/mixed covariance shrinkage; robust ordinal/polyserial
  menus and pair-local diagnostics; fixed-scalar DLS and Stage-2/IJ adapters;
  Fisher/Fisher-SNLLS/IRLS
  routes; ordinal pairwise composite likelihood; mixed pairwise/FIML hybrids
  and regularized Stage 1; RBM; SAM/LSAM and native FC-SEM as separate decisions.
  Mixed-only reviews follow in 0.0.2; noniterative reviews are indefinitely
  postponed. **Check:** concrete consumers, shared
  dependencies, evidence and a bounded keep/consolidate/remove decision.
  Recording this list does not authorize removals. Retain ordinary moment/Gamma,
  score/IJ and other shared primitives. Parking expansion is not code deletion.

- [ ] **S/M — inventory validated primary capabilities.** Record model/data
  slice, domain, penalty, algorithm, API tier and evidence for estimation,
  verdict/admissibility, covariance, global/nested tests and intervals separately.
  Use validated, limited-validation, unsupported and inapplicable states;
  planned slices link here. **Check:** C++/R owners and ordinary-policy exposure
  agree. Keep one inventory; secondary breadth is consumer-gated. See
  [development priorities](../architecture/roadmap.md#estimator-development-priorities).

- [ ] **M — audit tolerances and convention exemptions.** Loose parity gates
  must not absorb known divergences. **Check:** justified tolerances, count-pinned
  deferred buckets and independent/calibration proof for oracle exemptions.
  See test ledger and [oracle defects](../validation/oracle-defects.md).

- [ ] **M — add primary target-regime calibration checks.** Per-dataset oracle
  agreement does not establish size. Add advisory paired ML/FIML/DWLS robust
  test/covariance checks against oracle and nominal rates. **Check:** nonnormal,
  missing and ordinal regimes with failures retained; turn deterministic defects
  into tests. See calibration policy.

- [ ] **M — broaden primary CI checks.** Use appropriate `R CMD check` instead
  of hand-picked R tests; add scheduled sanitizers/optional parity and an
  interpretable coverage artifact. **Check:** clean-source portable installs
  and mount-independent default tests, avoiding unexplained percentage gates.
  See [local hardening](../validation/local_hardening.md).

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

## 0.0.2: mixed-data workflows and barrier inference

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

Barrier-specific work follows the 0.0.1 ordinary/PSD milestone. The existing
shared fitting baseline is recorded in the
[composition merge update](estimator-composition.new.md); reconcile completed
slices when folding that record into this backlog. Retain current barrier
entry points and regression gates.

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
