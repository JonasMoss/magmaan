# magmaan TODO

Accepted implementation work, ordered by workflow. The
[roadmap](../architecture/roadmap.md) owns current capabilities/contracts;
the [test ledger](../validation/test_ledger.md) and linked studies own evidence.
Remove completed items after their durable record exists.

**Scope adopted 2026-09-30:** single-level ML, FIML and ordinal/mixed DWLS,
with PSD covariance constraints and the multi-information barrier alongside
them. Secondary estimators retain correctness gates; extensions need a concrete
consumer or inexpensive reuse of primary work. Two-level SEM, SAM and composites
have **no scheduled expansion work**, including inside shared normalization,
start, preparation and inference programmes. Existing APIs/tests remain.

The existing limited-information correlation-ML capability (currently catML)
is retained and folded into shared fitting composition. Common moment targets
and model penalties are near-term work; this does not establish new ordinary-
user defaults or promise inference for every combination. See the
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

## Next: barrier fitting normalization

- [ ] **L — normalize the whole barrier fit.** Reuse the shared complete-data
  ML/PSD transformation for model/data, automatic and supplied starts, fixed
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

Barrier normalization remains the next requested fitting task. The dependency
order inside [shared composition](#shared-fitting-composition) starts with the
moment-target foundation; that extraction can proceed alongside normalization.

## Fitting reliability

### Optimization and convergence

- [ ] **M — recover from L-BFGS domain aborts across parameter scales.** Limited
  line-search reductions can exhaust infeasible trials at the initial point.
  Assess safeguarded backtracking or adapter recovery while retaining caller
  controls and terminal candidates. **Check:** the
  [domain probe](../../cpp/tests/checks/nlopt_lbfgs_domain.c), corrected corpus
  failures and rescaled fits. A larger evaluation budget alone is insufficient.
  See [corpus recovery](../../experiments/engineering/active/17-corpus-optimizer-recovery/report.qmd).

- [ ] **M — diagnose the layered-start Geiser latent-AR loss.** The fixture
  `latent_ar_cross_lagged_extended` stalls above the reference objective with
  L-BFGS and PORT; FABIN3 and std.lv succeed. **Check:** explain the valley/start
  failure, preserve a regression and verify corpus/held-out behavior before
  changing defaults. Evidence: corpus recovery.

- [ ] **M — finish unit-equivariant starts and optimizer coordinates.** Remove
  unit-dependent FABIN/layered fallback steps; extend supported start selections
  and fallback reports to ordinal/mixed preparation. Wire coordinates through
  primary ordinal/mixed and relevant SNLLS/IRLS paths. **Check:** transported
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

- [ ] **L — close primary PSD stress gaps.** Localize ML, FIML and ordinal/mixed
  DWLS failures using the retained harness. Keep stage-one objects unchanged;
  separate solver status, cone stationarity, admissibility and competing basins.
  **Check:** independent objective/link recomputation, interior reductions,
  retained seeds and targeted confirmation with failures/cost tails reported.
  Promote deterministic defects to tests. Secondary breadth is consumer-gated;
  two-level and native FC-SEM stay excluded. Evidence:
  [PSD stress](../../experiments/engineering/active/13-psd-estimator-stress/report.qmd).

## Primary inference workflows

### ML and FIML

- [ ] **M — complete ML policy inference for fixed-x models.** The policy
  geometry requires random X while ordinary fitting defaults to `fixed.x=TRUE`.
  Define conditional geometry or an explicit policy restriction. **Check:**
  moments, covariance, global/nested tests and data ownership; never silently
  substitute a joint model. See [R interface vision](../design/r-interface-vision.md).

- [ ] **S — test fully specified models under the policy.** Complete the
  zero-direction geometry (identity U factor) and nontrivial global test for
  zero-free-parameter models. **Check:** independent
  statistic/reference geometry and saturated/zero-df handling. See the vision.

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

- [ ] **S — exclude identification-only score candidates reliably.** A freed
  marker's efficient information is zero, but rounding can make
  `score_for_coordinate_robust` admit it under an absolute cutoff.
  **Check:** identification-only restrictions stay excluded across nearby
  fits/units; genuine low-information restrictions retain a justified relative
  rank check. Preserve the FIML robust-MI witness and identified controls.

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
  must not refuse a fit. See the R interface vision.

- [ ] **M — pin FIML robust conventions before claiming parity.** Resolve `sb_ml`
  bread/meat/H1 choices and convention dispatch; distinguish Yuan-Bentler
  variants from SB labels and retain FMG missing-data oracle limitations.
  **Check:** convention-matched values and target-regime/grouped calibration.
  Evidence: [test map](../../experiments/replications/08-savalei-falk-2014-test-map/report.qmd)
  and [calibration policy](../validation/calibration-parity.md).

- [ ] **L — establish applicable barrier inference and exposure contracts.**
  After normalized fitting, distinguish penalized-estimate covariance from
  ordinary inverse information, and single-face displacement tests from a
  general multi-face reference law. Use each moment source's Gamma/influence
  law, target-map derivatives and actual penalized estimating equation; retain
  structural-parameter and threshold uncertainty explicitly. Define scaling/
  domain and inference before claiming DWLS penalty inference or ordinary-user
  exposure. **Check:** independent derivatives, fixed-penalty limits,
  regular/boundary covariance, global/nested tests and interval calibration,
  with explicit unsupported components. Reuse correlation-ML criterion
  evaluation at DWLS fits for robust RMSEA without implying ML refitting.
  Ordinary SEs are not automatically valid at singular PSD endpoints. See the
  composition contract and the research index's barrier studies.

### Ordinal and mixed DWLS

- [ ] **L — compose DWLS policy covariance and global/nested tests.** Include
  weight-estimation influence. Global score and fit-function statistics coincide
  for the estimator's fixed weight; report once with calibrated SB/PEBA4.
  Specify nested sensitivity, metric, evaluation point, nuisance projection,
  centering and group normalization before adapters. **Check:** fixed-weight
  reductions, independent influence derivatives, jackknife controls and
  misspecified-larger-model calibration. Alternative score weights need a
  recorded decision study before adoption.

- [ ] **M — implement conditional categorical moments.** Fixed observed
  covariates (`exo` rows) are explicitly refused by C++/R preparation;
  `fixed_x=FALSE` remains a distinct joint model. Build conditional stage-one
  moments, sampling covariance and stage-two model before removing the guard.
  **Check:** eight covariate corpus cases and direct/staged/cached routes against
  conditional references. See [translation audit](../validation/textbook-translation-audit.md).

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

## API consistency and performance

### API and R boundary

The initial EQS model-section frontend is complete; its maintained scope and
validation limits are in the [EQS contract](../grammar/eqs.md). Additional EQS
syntax (notably `/MODEL`, labels and constraints) is unscheduled until requester
files establish the needed subset; full EQS job/estimator emulation is outside
this request. Indicator variables participating in structural regressions
currently fail explicitly at the frontend; lifting this restriction requires
resolving their shared model-builder representation first.

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
  wrappers only after replacement gates pass. Mixed/polyserial ML needs its
  own moments/mean/scale contract; removed research APIs are not migration targets.

- [ ] **L — compose model penalties with retained fitting routes.** Generalize
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

- [ ] **S/M — settle friendly covariance-policy naming and migration.** The
  proposed `covariance = "unrestricted" | "psd" | "barrier"` replaces an
  overloaded `psd` option, but exact naming and existing-call migration remain
  open. Internally domain constraints and penalties stay independent.
  **Check:** explicit combinations, compatibility/refit metadata and documented
  distinction between changing the covariance domain and changing the objective.

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
  mean freeing in `continuous_invariance()`; expose mixed-model release only
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
  menus and pair-local diagnostics; noniterative CFA clamps/conditioning/H
  repair; fixed-scalar DLS and Stage-2/IJ adapters; Fisher/Fisher-SNLLS/IRLS
  routes; ordinal pairwise composite likelihood; mixed pairwise/FIML hybrids
  and regularized Stage 1; RBM; SAM/LSAM, native FC-SEM and the main noniterative
  CFA menu as separate decisions. **Check:** concrete consumers, shared
  dependencies, evidence and a bounded keep/consolidate/remove decision.
  Recording this list does not authorize removals. Retain ordinary moment/Gamma,
  score/IJ and other shared primitives; review recorded noniterative no-go screens
  separately from estimator maps. Parking expansion is not code deletion.

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

## Related work

- [Simulation backlog](simulation.md) owns generator/projection/calibration
  detail, including model-implied simulation and group metadata. Only primary-
  workflow dependencies belong in this queue.
- [Speculative register](speculative.md) owns parked methods, model families,
  expensive verification and optional reporting/packaging. Promotion needs a
  named consumer, bounded scope and completion check.
