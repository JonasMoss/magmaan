# Speculative backlog

Deferred possibilities, reviewed 2026-10-01. This is a **trigger register**:
none of these entries schedules implementation, a simulation or API promotion.
Promote a bounded slice to [todo.md](todo.md) only for a named downstream
consumer, with an explicit result and completion check. Existing APIs and
regression gates remain; current capability limits belong in the
[roadmap](../architecture/roadmap.md).

Every entry records a gap, an available alternative and a build-if condition.
Evidence stays in maintained contracts, the [test ledger](../validation/test_ledger.md)
or independent studies indexed by [experiments/README.md](../../experiments/README.md).
Paper simulations, manuscript tasks and handoffs belong to their own projects.
Completed derivations and run diaries are not retained as future commitments.

Removed research APIs (ridge continuation, automatic identification fitting,
adaptive composites, MI4, EB-DLS, fitted-weight GMM, PNTML and dedicated GLSpw)
are historical only. Alternatives below use retained surfaces. Pairwise moments
are an MCAR data axis; supplying them does not validate complete-data inference
or establish general MAR consistency.

## Parked model families

### Two-level SEM

**Gap:** multi-group models, within-/between-only variables, constraints,
analytic Hessians, robust/categorical inference and broader corpus parity.
Random slopes and 3+ levels also need new model design.
**Available:** the tested single-group shared-variable complete-data ML slice;
unsupported shapes remain explicit. See the roadmap's two-level contract.
**Build if:** a named user/study needs one specific extension and its estimation,
inference and validation scope is agreed. No shared start/normalization/R
preparation programme implicitly reactivates this family.

### SAM and LSAM

**Gap:** broader robust conventions, misspecification/estimated-weight
sandwiches, new preparation adapters and larger parity grids.
**Available:** the existing explicit SAM frontier surface in its tested scope,
or ordinary joint SEM fitting. See the roadmap's SAM contract.
**Build if:** a named methods workflow requires the two-step procedure and an
unsupported slice. Promote only that slice; no scheduled expansion or new
default inference work.

### Composite models

**Gap:** multi-group weights, ordinal/FIML/LS routes, robust inference,
mean structures and dedicated benchmark expansion.
**Available:** the tested native single-group ML FC-SEM frontier slice;
unsupported dispatcher calls can use its explicit entry point.
**Build if:** a concrete consumer needs the extension or repeatedly hits an
opaque dispatcher failure. Broader support needs independent parameterization
and parity contracts; it is not part of primary workflow completion.

## Covariates and sampling

Banked 2026-10-01 under the [statistical scope](../scope.md). Conditional
fitting can serve random-X population inference; the two entries below have
different targets and must not be promoted as one generic fixed-x feature.

### Fixed-design inference under mean misspecification

**Gap:** a conditional-on-design target, assumptions that make its variance
estimable, and covariance/global/nested inference under nonlinear mean
misspecification. Ordinary residual sandwiches do not provide a general
conditional-variance guarantee.
**Available:** validated random-X population inference for supported models;
existing lab fixed-x estimation and compatibility conventions in their tested
slices. When conditional fixed-design uncertainty is required, report the
inference limitation rather than reinterpret the target or silently refit.
**Build if:** a named study needs a specific fixed-design estimand and supplies
additional structure (for example replication or a restricted conditional
mean). Agree the design sequence, response law and requested components;
derive the variance and test references independently. Calibration must
distinguish heteroskedasticity, nonlinear mean misspecification and noiseless
controls. This is outside primary workflow completion.

### Conditional categorical moments with observed covariates

**Gap:** conditional stage-one thresholds/intercepts/slopes and residual
moments, their sampling covariance/influence, and the corresponding stage-two
model. Fixed observed covariates (`exo` rows) are currently refused by C++/R
preparation, including cached moment routes.
**Available:** explicit joint `fixed_x=FALSE` fitting when its different target
and moment assumptions are appropriate, or an external conditional fitter.
The rejection guard remains; marginal statistics are not a substitute for
conditional statistics. See the
[translation audit](../validation/textbook-translation-audit.md).
**Build if:** a concrete consumer needs conditional categorical estimation.
Specify whether inference targets a random-X population functional or a
fixed-design parameter, and validate the relevant influence law. Reopen only
the bounded slice: the eight covariate corpus cases and direct/staged/cached
routes provide compatibility gates, with independent target-regime controls
for any inference claim. This does not reactivate general fixed-design inference.

## Optimizers and coordinates

### Convergence-audit extensions

**Gap:** singular-PSD/box interactions, nonlinear/nonconvex constraint geometry,
specialist adapters, owning R audit artifacts and automatic provenance capture.
**Available:** retained Newton artifacts and explicit reports under the
[terminal-audit contract](../design/terminal-audit.md), with unsupported geometry
left unresolved. Primary verdict bugs remain active.
**Build if:** a concrete workflow cannot express its audit/provenance reliably
using those owners. Validate the required geometry before changing acceptance.

### Ultimate verifier and tolerance redesign

**Gap:** high-precision terminal verification and empirical justification for
a universal tolerance or Absolute-to-Relative default change.
**Available:** current original-objective Newton/verdict diagnostics, focused
regressions and decision-specific comparisons. See terminal-audit.md and
[convergence engineering](../design/convergence-engineering.md).
**Build if:** a reproducible primary failure remains unexplained after ordinary
diagnostics, or a named default decision requires precision/tolerance evidence.
Do not build a broad verifier merely to enlarge the audit surface.

### Runaway estimates and nonattainment diagnostics

**Gap:** local stationarity does not prove a finite optimum exists; large
coefficients do not distinguish escape, weak identification and chart poles.
**Available:** separate accuracy, covariance-domain, chart and parameter
diagnostics; requested-chart correctness fixes remain active. Evidence:
[sphere references](../../experiments/engineering/active/15-sphere-reference-fits/report.qmd).
**Build if:** a consumer needs an automatic distinction, with a study separating
finite improper optima, genuine nonattainment and numerical/chart failures
across units and identifications. Bounds/penalties change the estimator.

### SEM PSD optimization beyond the engineering baseline

**Gap:** automatic chart selection, general boundary restarts and global
certificates beyond the direct PSD route and explicit fallback.
**Available:** retained fitting routes, starts and audits; concrete primary
round-trip/unit failures remain active. Historical native-start evidence:
[sem-total-variance](../../experiments/_archive/sem-total-variance/report.qmd).
**Build if:** a supported case supplies a material objective miss or bottleneck
and fresh relevant cases show repeatable benefit with retries/failures counted.
Boundary restarts need explicit graph/equality eligibility; global bounds need
their own consumer. Historical scaling comparisons are not current defaults.

### Start constructors beyond the layered moment start

**Gap:** MIIV, SAM-based initialization, spectral selectors and automatic
multistart beyond tested constructors and supplied vectors.
**Available:** layered/simple/FABIN and retained research starts; unit/fallback
bugs are active. Evidence: [corpus recovery](../../experiments/engineering/active/17-corpus-optimizer-recovery/report.qmd)
and sphere references.
**Build if:** a named family demonstrates constructor bias or reproducible
basin dependence and held-out comparisons justify a new recipe. Multistart
must be opt-in and seeded; preserve requested markers. No blanket defaults grid.

### Sphere chart development

**Gap:** ordinary-user promotion, broader gauge theory, automatic failure-cause
classification, chart/start attribution and new estimator/constraint charts.
**Available:** explicit frontier fitting/reidentification and retained chart
diagnostics. Concrete primary chart-verdict failures remain in TODO. See
[parameterization geometry](../design/parameterization-geometry.md) and sphere references.
**Build if:** a named consumer requires new chart support or the current
investigation establishes a bounded, validated interface change. Broader theory,
new grids and multistart are not prerequisites for fixing current failures.

### Convergence bench beyond the decisions study

**Gap:** a general optimizer tournament, reusable problem generator, automatic
problem classes and universal canary/reference infrastructure.
**Available:** [optimizer defaults](../../experiments/decisions/01-optimizer-defaults/report.qmd),
[barrier defaults](../../experiments/decisions/02-barrier-defaults/report.qmd),
corpus recovery and focused tests. See convergence engineering.
**Build if:** a new default decision needs a bounded lane; share problem code
only when a second real consumer appears. Add a canary when an actual regression
escapes tests. No broad backend comparison is queued.

### Exact Hessians for IPOPT

**Gap:** exact objective/Lagrangian Hessians in the optimizer bridge.
**Available:** IPOPT's approximation and retained alternative backends.
**Build if:** a supported constrained primary case demonstrates a material
accuracy or runtime benefit. See [optimizer controls](../reference/optimizer-controls.md).

## Estimation and inference

### Moment-covariance centering alternatives

Banked 2026-10-01: reconsidering established moment-covariance recipes is
outside the active likelihood-score centering decision.
**Gap:** evidence for changing the centering already built into empirical
Gamma/NACOV and moment-influence covariances: continuous WLS/DWLS/DLS weights
and inference, noniterative delta inference, complete-data moment-based robust
SE/GOF routes, and global/nested LR moment references (including Satorra--2000).
This also covers revisiting already-derived Stage-1 moment/influence covariance
maps; direct score OPG choices at saturated versus structured points remain in
the [active score-covariance decision](todo.md#primary-inference-workflows).
**Available:** retain each method's existing covariance/influence definition,
normalization and regression/parity gates. The fact that a covariance uses
centered contributions does not require a new centered/raw experiment.
**Build if:** a reproducible defect or named consumer establishes that an
existing recipe targets the wrong sampling covariance, or a specific alternative
has a justified statistical benefit. Derive the target and influence law first;
validate the affected estimates when Gamma also determines fitting weights.
This banks centering redesign, not primary estimator/inference completion or
fixes to established covariance formulas.

### Penalized ML for small-sample convergence and structure (parameter-space)

**Gap:** new variance/structural penalties, selection rules and penalized
inference beyond the retained multi-information barrier.
**Available:** explicit parameter bounds, retained covariance shrinkage and
barrier fitting, each with its own estimand and support contract.
**Build if:** a consumer specifically needs parameter-space regularization
and existing options are inadequate. Selection and nonstandard inference need
their own validation; removed ridge continuation is not an alternative.

### Regularized H1 references for two-stage and nested SEM

**Gap:** calibrated regularization for near-singular direct-FIML H/Gamma and
ML2S stage-one references; a reference-only change differs from input shrinkage.
**Available:** fail-closed guards, unregularized routes and explicit frontier
stage-one regularization with retained raw inputs. See
[test map](../../experiments/replications/08-savalei-falk-2014-test-map/report.qmd).
**Build if:** a named regularized-reference project validates bias, coverage,
size and reference-law effects on frozen hard cells and fresh draws. Do not
present oracle value agreement as calibration or silently change the estimator.

### Likelihood-ratio interval policy

**Gap:** an ordinary-user LR/score interval policy with robust small-sample
calibration, domain recovery and acceptable constrained-refit cost.
**Available:** Wald policy intervals, explicit profile primitives and
the consolidated [normal/robust scalar-interval showcase](../../experiments/showcases/07-scalar-intervals/report.qmd); `confint(test="lr")` remains reserved.
**Build if:** a policy decision is explicitly commissioned with held-out
coverage/failure/cost criteria. Reconcile centering conventions and boundary
failures before changing defaults; existing studies alone do not adopt a method.

### Reduced-bias estimation and specialist covariance extensions

**Gap:** new RBM performance/regular-region claims, additional estimating-
function adapters and secondary estimator policy completeness.
**Available:** retained explicit RBM/IJ methods and ordinary primary workflows;
fixed-weight reductions and analytic/finite-difference checks remain guards.
Evidence: [RBM risk](../../experiments/research/banked/53-rbm-estimation-risk/report.qmd).
**Build if:** a named workflow needs a specific adapter or performance repair,
with independent derivative/jackknife and misspecification controls. Paper-local
replications and expanded simulation grids stay with their owners.

### Profile and fit-index extensions

**Gap:** CRMR/SRMR pooling decisions, signed-trace TLI stability, profile-curvature
shortcuts, close-fit/boundary calibration, threshold-inclusive residual indices
and additional per-index bindings.
**Available:** explicit bundles and documented core/frontier targets; core
mean-of-roots and frontier root-of-mean are distinct multigroup quantities.
See the roadmap's moment-quadratic/profile contracts and test ledger.
**Build if:** a concrete reporting/calibration consumer needs a named target
or demonstrates an unstable result. Preserve dense mixture weights where needed;
analytic sign counts and cheap diagnostic gates need independent proofs.

### Robust score flips and ordinal permutation

Centered-versus-raw multiplier promotion remains banked here; the ordinary
policy's analytic likelihood-score covariance decision is active in TODO.

**Gap:** new standardized-flip promotion, robust MI scope and proof of raw-label
permutation validity for heterogeneous ordinal invariance nulls.
**Available:** retained explicit score/flip and nested-test primitives;
experimental resampling remains local. Evidence: the research index's score-
flip and permutation studies.
**Build if:** a named inference workflow needs resampling beyond the primary
policy. Require nuisance-singularity, allocation, sparse-category and matched-
power checks; a fixed-statistics shuffle is not a raw-label test. Any ridge or
shrinkage rescue defines a separate tuned method.

### Fixed-rank flip standardization

**Gap:** whether nuisance dimension, unequal allocation or heterogeneous
information makes flip-specific standardization useful at fixed test rank.
**Available:** effective flips, score-SB/pEBA4 and the maintained dense-covariance
oracle in `cpp/tests/unit/score_robust_test.cpp`; the completed homogeneous
illustration does not justify standardization from rank alone.
**Build if:** a named grouped-invariance consumer exposes a size/power or
variance-displacement gain large enough to justify standardization cost.
Start with paired controls at G=8, df=28, p=5 versus p=20 and small unequal
groups; validate moments and matched-null power before widening a grid.

The former research/34 harness was retired without local substantive results.
Its proposed broad 128-cell and focused 32-cell designs are historical, not
queued work. Legacy remote identifiers were app `exp34-flip-calibration-frontier`
and volume `exp64-flip-results`; inspect any remote artifacts before claiming
that no results ever existed. The library methods and regression gates remain.

### `spectral_truncate` weight policy for degenerate ADF/WLS Γ̂

**Gap:** pseudo-inverse fitting on a retained subspace with rank diagnostics.
**Available:** explicit rank/conditioning refusal and advisory weight telemetry.
See [numerical conventions](../design/numerical-conventions.md).
**Build if:** a real methods/parity case needs this estimator and its sampling
behavior can be validated. It is not a free numerical parity patch.

### `statistic = "N" | "N-1"` selector for the GLS/WLS test multiplier

**Gap:** selectable test normalization.
**Available:** the documented fixed convention and rescaled parity gates.
**Build if:** a methods consumer needs a side-by-side alternative. See numerical conventions.

### Batched-`trsm` LS-shape whitening for the NormalTheory weight

**Gap:** cheaper Jacobian whitening for genuine LS-shape backends.
**Available:** scalar trace-identity evaluation used by default GLS backends.
**Build if:** a measured PORT-NLS/Ceres/IRLS large-model workflow is dominated
by whitening. See [benchmark guide](../../benchmarks/README.md).

### Reduced-Gamma ordinal robust-inference products

**Gap:** influence/cache plumbing for robust reporting without full Gamma.
**Available:** cached full Gamma, lazy fitting workspaces and influence factors;
the `Reduced` enum alone does not implement reduced inference.
**Build if:** a supported model's measured Gamma memory/time dominates reporting
(e.g. moment dimension around 1000). See
[workspace contract](../design/ordinal-snlls-gamma-architecture.md).

### Categorical scope extensions

**Gap:** per-row free-delta bound relaxation (Newsom ex9.2), more flexible
threshold-profile eligibility and automatic robust baseline fit-index dispatch.
**Available:** current explicit bounds/profile fallbacks and robust helpers;
known exclusions are documented rather than hidden.
**Build if:** a concrete consumer needs a named case. Validate implied-variance
domain versus additive residual bounds, reconstruction/full-model audits or
baseline scaling as applicable. See the translation audit and workspace contract.

### Mplus/lavaan WLSMV invariance extensions

**Gap:** mixed-pairwise helper input, wider Mplus parity and delta-release diagnostics.
**Available:** current explicit Mplus-style helper and lavaan-theta parity paths.
**Build if:** a mixed missing-data consumer or compatibility claim requires it;
validate moments/NACOV first and keep CI free of Mplus. Degenerate lavaan delta
release is not an oracle target. See the workspace contract.

### Robust mixed/polyserial moments under missing data

**Gap:** a selected robust polyserial recipe with coherent influence/Gamma and
missing-data semantics, including optional complete-data h-weighted polyserials.
**Available:** robust all-ordinal moments and retained ML mixed/observed/hybrid
paths in their documented scopes. WMA's polychoric cell cap is not a polyserial
recipe. Evidence: the robust-ordinal studies in the research index.
**Build if:** a concrete mixed-robust consumer exists and the recipe fork is
resolved. Validate copula/contamination behavior and missingness assumptions;
do not port an intrinsically mixed recipe without evidence.

### ESEM (exploratory structural equation modeling): free loading blocks + rotation

**Gap:** rotated exploratory measurement blocks and rotation-aware inference.
**Available:** explicit CFA/cross-loading syntax, or dedicated external tools.
**Build if:** a named consumer requires exploratory blocks inside SEM and
identification, rotation and SE contracts are agreed. No current commitment.

## Measures and reporting

### Multi-factor EAP and non-diagonal residual ordinal factor scores

**Gap:** multidimensional EAP and residual-correlated ordinal scoring.
**Available:** tested diagonal-Theta EBM/ML and one-factor EAP, with explicit
scope errors. See [ordinal reliability](../reference/ordinal_reliability.md).
**Build if:** a named scoring consumer needs posterior means or residual
covariances in an unsupported shape and integration/parity validation is feasible.

### Ordinal precision and observed-score omega extensions

**Gap:** new precision coefficients, multi-factor PRMSE and broader observed-
score/true-score omega targets and interval calibration.
**Available:** one-factor posterior precision and explicitly named retained
reliability targets. Evidence: ordinal reliability and the research index's
omega target/coverage studies.
**Build if:** a consumer names the target and needs that extension. Do not
equate covariance omega, direct ordinal true-score reliability and determinacy.

### MI effect sizes (dMACS / EDM family) for fitted multi-group models

**Gap:** integrated invariance effect-size reporting.
**Available:** exposed means/loadings and current invariance tests, or external
effect-size tools.
**Build if:** a workflow needs a first-class effect size with a named scale/
rotation convention and independent closed-form validation.

### Model-based omega (omega_u / omega_H / omega_ho) reliability

**Gap:** additional ordinal/higher-order targets and default reporting.
**Available:** retained continuous fit-based omega and explicit reliability
primitives; current quantities must keep distinct targets/names.
**Build if:** a named reliability consumer needs a new target and its point/
SE/misspecification behavior is validated. See ordinal reliability and the roadmap.

### Maximal reliability / corrected coefficient H for bifactor models

**Gap:** additional first-class bifactor/maximal-reliability reporting and
categorical extension beyond retained quantities.
**Available:** fitted matrices and score weights, explicit reliability helpers.
**Build if:** a concrete reporting consumer needs the coefficient and oracle/
independent target validation. Avoid duplicating a quantity already exposed.

### Small-sample-corrected profile-LR confidence intervals for the reliability family

**Gap:** a stable operational correction across bounded reliability functionals.
**Available:** explicit profile primitives, Wald/delta intervals and retained
profile studies. Evidence: [profile study](../../experiments/research/active/20-profile-lr-reliability-ci/report.qmd)
and the research index's ordinal calibration studies.
**Build if:** a named reliability workflow requires calibrated profile intervals;
validate boundary mass, corrections, centering, failures and endpoint cost on
fresh data. Neither oracle factors nor per-dataset bootstrap success imply a default.

### Small-sample distribution-free intervals for covariance functionals (Kauermann-Carroll)

**Gap:** general variance-of-variance/effective-df intervals.
**Available:** retained delta/Wald inference and study-local prototypes.
**Build if:** a named covariance-functional consumer has demonstrably inadequate
ordinary coverage and the correction passes held-out target-regime checks.

### Non-iterative CFA inference extensions

**Gap:** new maps, structural paths, cross-loadings, residual covariances,
identification choices and new reference-law calibration.
**Available:** explicit supported closed-form maps and ordinary iterative CFA;
existing formula/derivative gates remain. See the roadmap and research index.
**Build if:** a named consumer needs a bounded extension, with independent map
and influence validation. Aligned-map paper reruns are paper-local work.

### Null-bootstrap GOF calibration for non-iterative CFA

**Gap:** calibrated bootstrap reference laws for new non-iterative targets.
**Available:** retained asymptotic/reference methods and ordinary ML inference.
**Build if:** a concrete non-iterative workflow needs bootstrap calibration and
can justify null construction, failures, Monte Carlo error and compute cost.

### Efficient leave-one-out / infinitesimal jackknife for closed-form (non-iterative) CFA

**Gap:** additional influence/LOO adapters for closed-form maps.
**Available:** explicit refits and current influence primitives.
**Build if:** a measured large-N workflow makes refitting material and map-level
derivatives/reconstruction can be independently validated.

### Residual and case-influence reporting extensions

**Gap:** additional RMR/CRMR residual summaries, mixed estimated-weight residuals,
approximate fit-measure changes and new case-drop domain/inference corrections.
**Available:** tested cor.bentler summary, supported estimated-weight residuals,
exact refits and one-step parameter changes. See roadmap/test ledger and
[oracle defects](../validation/oracle-defects.md).
**Build if:** a reporting workflow needs a named unsupported target or exact
refits become a demonstrated bottleneck. Upstream issue/PR drafting is a
separate maintainer task; it does not schedule external communication.

### Structural-model fit indices, tests, and CIs (two-step / SAM)

**Gap:** structural-only corrected fit indices and intervals.
**Available:** full-SEM GOF and external two-step implementations.
**Build if:** SAM is explicitly reactivated for a named structural-GOF consumer;
validate stage-one covariance influence and reference law first. No implicit
commitment from existing full-SEM fit-index machinery.

## Pairwise and spectrum performance

### Pairwise observed-bread streaming and mean inference

**Gap:** observed-bread inference still materializes pairwise Gamma; additional
mean weighting/influence conventions need validation.
**Available:** expected-bread/operator-first and empirical reduced pairwise
paths, plus materialized observed-bread references in the supported MCAR scope.
**Build if:** a pairwise consumer needs the missing route or hits measured
dense-memory cost. Check all-observed collapse and small-fixture agreement;
removed dedicated GLSpw fitting is not a target.

### Pairwise Browne-unbiased

**Gap:** finite-sample-unbiased pairwise meat.
**Available:** empirical pairwise influence and complete-data unbiased reducers.
**Build if:** a consumer needs unbiased pairwise inference and the empirical
alternative is inadequate. Validate a literal overlap reference before shortcuts.

### Overlap-corrected Γ for WLSMV / ordinal missing data (PD repair)

**Gap:** broader overlap/PD-repair conventions beyond supported ordinal and
mixed observed-data paths.
**Available:** current explicitly scoped moments, influence and Gamma builders;
complete-data inference is not a missing-data substitute.
**Build if:** a named missing-ordinal consumer needs an unsupported convention.
Validate exact overlap sets before pragmatic shortcuts and treat weight repair
as its own estimator/inference change. See the workspace contract.

### Row-space eigensolve for the unbiased spectrum

**Gap:** a shortcut when df is much larger than N.
**Available:** full reduced unbiased spectrum, which is the numerical reference.
**Build if:** a measured unbiased-spectrum workflow is dominated by the df×df
solve; preserve the sample NT term rather than applying a scalar shift.

### Rank-one secular update for the unbiased spectrum

**Gap:** cheaper paired biased/unbiased eigensolves.
**Available:** independent values-only solves.
**Build if:** a measured implementation beats both solves including eigenvector
rotation/setup, with numerical equivalence. See benchmark guide.

## Build, layout and documentation

### Namespace and header housekeeping

**Gap:** ownership of constraint/evaluation headers and `cfa_utils`, and gathering
start constructors in `estimate::starts`.
**Available:** documented current owners and forwarding-shim migration patterns.
**Build if:** a real dependency/API problem requires the move or primary work
already touches the affected boundary. Settle spec-versus-estimate evaluation
ownership first; cosmetic consolidation is not scheduled.

### Documentation system (two-surface manual)

**Gap:** a public manual and promotion of private derivations into concise
implementation-checked supplementary docs.
**Available:** roadmap, maintained design/reference notes and package examples.
**Build if:** a public release or recurring reader need names the missing page.
Promote derivations individually; paper-track ideas remain with their owner.

### Dependency-license manifest for binary artifacts

**Gap:** a redistribution-specific dependency manifest.
**Available:** source licenses and vendored dependency provenance.
**Build if:** a binary/packaged artifact is actually being shipped; inventory
what it includes and validate that distribution rather than guessing from builds.

### Ceres preset in regular validation

**Gap:** scheduled Ceres-specific coverage.
**Available:** on-demand optional preset and primary backend tests.
**Build if:** a concrete backend regression or coverage gap justifies the lane.

### Opt-in precompiled headers for Eigen-heavy builds

**Gap:** potentially shorter changed-TU builds.
**Available:** current measured build loop.
**Build if:** measured PCH benefits exceed setup/no-op/full-build costs.

### Additional benchmark, corpus and replication breadth

**Gap:** new OpenMx/Mplus/textbook panels, larger SNLLS grids and replication
variants without a current primary defect or decision.
**Available:** frozen oracle fixtures, existing harnesses and source-verified
cases; named fixture/failure repairs remain active.
**Build if:** a named support claim, regression or default decision needs the
additional case. Do not expand a grid merely because a harness exists.

### Exact-moment Likert populations beyond the Gaussian copula

**Gap:** non-Gaussian discretized generators preserving a specified SEM moment target.
**Available:** supported Pearson-code calibration and current simulation generators.
**Build if:** a named calibration study requires that population and can verify
its moments independently. Promote generator work to [simulation.md](simulation.md).
