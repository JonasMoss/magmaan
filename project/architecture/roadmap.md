# magmaan roadmap

This document summarizes the current implementation state and architectural
contracts for magmaan. It is not the active backlog. Remaining work lives in
[project/backlog/todo.md](../backlog/todo.md).

The [statistical scope](../scope.md) owns population targets and sampling-law
boundaries. Adopted 2026-10-01: primary inference targets population
approximations under joint observation sampling, including random covariates.
General fixed-design inference under mean misspecification and categorical
conditional-moment expansion are [banked](../backlog/speculative.md#covariates-and-sampling).
Conditional fitting can support random-X inference; compatibility options do
not by themselves establish its sampling guarantee. Existing APIs, fitting
defaults and parity gates remain in force. The ordinary fixed-x policy
restriction remains an active clarity/correctness task.

Out of scope for this track: Bayesian SEM, latent
interactions/mixtures, EFA, inequality constraints (and active-bound
inference), and end-user lavaan
replacement ergonomics. Multilevel SEM is no longer wholesale out of scope:
two-level random-intercept normal-theory ML is landed (see the two-level ML
capability below); 3+ levels and random slopes remain out of scope.


Engineering studies now live under `experiments/engineering/{active,banked,evidence}/`.
The collection index owns activity and reopening triggers: four unresolved
implementation studies remain active, three ideas are banked, and the completed
PSD basin audit is retained as paper evidence. Settled probes are archived; the
robust-score demonstration was removed after its checks became maintained tests.
The ordinal observed-score omega studies share one leaf with separate target
and sampling result trees; RBM estimation risk is research/banked/53.

Research studies use `experiments/research/{active,banked,evidence}/`: 8
active studies (including protected paper pipelines), 11 banked ideas and
5 retained evidence studies. Three consolidated showcases retain scalar
intervals (normal/robust), reliability targets (coefficient/Bell examples),
and robust test calibration (battery/mechanisms/diagnostics). The finite-sample
GOF probes share one banked study with distinct DGPs. Redundant ordinal probes
were removed after naming maintained regression checks; the unrun fixed-rank
flip harness is replaced by a trigger entry in the speculative backlog.
The Bell main population targets reproduce, while historical alternative-CFA
sensitivity fits need convergence/objective revalidation (see the active backlog).
Dedicated pairwise efficiency/speed/composite probes have been removed at the
user's request; existing API regression gates remain. FIML global tests now share
research/active/44 (SEM, geometry and banked legacy lanes), while nested invariance
shares research/active/06 (calibration, score-flips, pilot power and oracle lanes).
Parameter-information research/39 is completed evidence and remains separate.
Alpha–omega equality is banked with its nonregularity derivations; covariance-
functional intervals use a general name. Bifactor and reliability profile
applications share banked research/18, while delta and fitted-profile ordinal
inference share active research/22 with distinct targets and failure accounting.
Savalei–Falk definitions and matrix fingerprints share one active reference
study; numerical similarity still does not identify the original EQS statistic.
Generating-model nonnormal MAR stress is not automatically a FIML pseudo-null;
the reports state the target and numerical-availability limits.
The collection index owns reopening conditions.
Frozen evidence, original metadata and interval source fingerprints remain unchanged. Navigation
changes do not change policy or authorize incompatible checkpoint resumption.
The final artifact audit limits Chen missingness, latent-chart and early
replication claims to their retained pilot/smoke runs. The active parameter-profile report excludes
retired fitted-weight rows; local missing artifacts are listed in the index.

## Repository layout

The C++ project is self-contained under `cpp/` (source, headers, tests,
fixtures, CMake, and vendored dependencies). Root `just` commands orchestrate
its builds under `cpp/build/` and the R package. R build helpers belong to
`r-package/tools/`; its disposable development mirror is `r-package/build-rdev/`.
Public maintainer knowledge lives under `project/`; research working notes
are maintained independently and are not build or test dependencies. External
textbook and paper collections share `external/`; tests retain frozen fixture
snapshots.

Agent guidance uses a concise root `AGENTS.md` plus directory instructions for
C++, both R packages, experiments and papers. Versioned `.agents/skills/`
workflows cover oracle validation, R bindings/vendoring and experiments, with
task-specific references loaded only as needed. `CLAUDE.md` imports the adjacent
instructions. The tracked-content guard allows repository skills under `.agents/`
while excluding agent configuration/session files.

Experiments are grouped by purpose under `experiments/{decisions,showcases,
replications,research,engineering}/`; `_archive/` and `_support/` remain at collection level.
Numbering starts at 01 within each live category; archived folders use unnumbered
slugs. Each study remains an independent leaf, including within a category.
Engineering is a live workspace for current implementation/default decisions;
research remains exploratory. Existing output provenance and cloud storage IDs
are preserved through renames.
Benchmarks and experiments retain their existing dependency roles.
The marker-chart sanity and PSD optimization-default studies are
archived under `experiments/_archive/` as of 2026-09-24; their recorded findings
remain historical evidence, with sources and local results preserved.

## Estimator development priorities

Adopted 2026-09-27; active scope narrowed 2026-09-30; release priorities
updated 2026-10-01. Development priority applies
to complete workflows:
estimation, convergence and admissibility diagnostics, parameter covariance,
global and nested tests, intervals, and the R interface. It is independent of
API stability (`core` / `frontier`), statistical evidence, ordinary-user
availability, and the decision to make a method a default.

| Development tier | Initial scope | Commitment |
| --- | --- | --- |
| **Primary classical** | Single-level normal-theory ML (NTML, exposed as `ML`), FIML, and all-ordinal DWLS | 0.2.0: ordinary API, lavaan-compatible fitting and complete inference workflows; magmaan's own fitting reliability and mixed continuous/ordered completion follow in 0.3.0 |
| **Priority frontier** | PSD covariance constraints; multi-information barriers in the following release | PSD keeps its current fitting and boundary-inference contract in 0.2.0; PSD finalization/stress work and barrier-specific hardening and inference follow in 0.3.0, with estimator-specific validation |
| **Secondary classical** | GLS, continuous ADF/WLS, ULS, ordinal full WLS; provisionally ML2S and pairwise moment routes | Preserve correctness and existing support; extend for concrete users or inexpensive reuse of shared work |
| **Research collection** | DLS, robust alternatives, SAM, FC-SEM and other specialist methods | Maintain tested, explicit contracts without promising the primary workflows' breadth |

The first two tiers form the main development programme. A frontier method
may receive more attention than a classical method with a stable API. Lower
priority does not relax correctness requirements or remove existing support.
Two-level ML, SAM and composites remain supported in their documented slices,
with existing APIs and regression gates. They have no scheduled expansion work:
multi-group/level-specific two-level support, broader SAM inference and composite
estimator families are parked in the [trigger register](../backlog/speculative.md#parked-model-families).
Shared normalization, starts, prepared R ownership and ordinary-policy inference
do not implicitly extend to these families. Reactivation requires a named
consumer and a bounded, validated scope. Broader chart development, specialist
methods and general optimizer/benchmark expansion are likewise consumer-gated;
concrete primary correctness failures remain active.
The provisional secondary assignments can be revisited for a concrete use.

Adopted 2026-10-01: mixed means models containing both continuous and ordered
indicators. Their remaining fitting, inference and interface work is assigned
to 0.3.0; all-ordinal workflows remain current. Noniterative development,
inference expansion and capability-review work are indefinitely postponed,
with reactivation requiring an explicit user scope decision. Retained APIs and
regression gates preserve their documented capabilities. The
[MI/release-score completion matrix](../backlog/todo.md#mi-and-release-score-completion-020)
records current coverage and remaining work across estimator and weight choices;
it schedules no new ordinary-user default or parked model-family expansion.

PSD and barrier are capabilities across estimator families. PSD changes the
covariance domain; the barrier changes the objective. Track those separately
from the estimator/data combination and the numerical algorithm. Fisher
scoring, IRLS and SNLLS receive priority through the workflows they improve.
A fitter's existence does not establish its inference coverage, and extending a
penalty or constraint requires validation for each applicable combination.

Priority, ordinary-user exposure, API stabilization and default adoption are
separate decisions. PSD has ordinary-user exposure with policy inference;
`covariance = barrier(lambda)` is exposed as experimental, with estimates only
(every inference component returns `penalized`). Promotion of either to a
default requires recorded evidence.

The [capability inventory](../validation/capabilities.md#primary-inventory-020)
(2026-10-02, exit criterion 6) records validated, limited-validation,
unsupported and inapplicable components per setup, domain and API tier, and an
ordinary-package test gates its states; these tiers alone make no new
availability claims.

Release plan revised 2026-10-02; the
[active backlog](../backlog/todo.md#release-plan) owns versions and exit
criteria. One version number covers the C++ library and both R packages.
**0.2.0 delivers the adopted ordinary API, lavaan-compatible fitting, primary
inference and MI/release-score completion**: `options = list(preset =
"lavaan-0.7.2")` extends to linear equality constraints, FIML and all-ordinal
DWLS so simulations can rely on lavaan's fitting; the ordinary policy covers
ML, FIML and all-ordinal DWLS; MI/release covers every estimator × weight cell
including two-stage, or rejects it with a typed reason. Explicitly reject
unvalidated inference; fitting support alone does not establish inference.
**0.3.0 owns magmaan's own fitting reliability, mixed continuous/ordered
completion and barrier-specific hardening and inference**: shared starts,
units, convergence/admissibility, PSD finalization, stress and normalization;
stricter starting domains, near-face curvature, factor disappearance, fallback
units, marker poles and penalty-specific sampling/exposure contracts. Existing
PSD and barrier implementation, inference refusals and regression gates
remain, and PSD inference keeps its current boundary contract. Existing mixed
APIs retain correctness gates. Noniterative development and capability
reviews are indefinitely postponed pending an explicit user decision.

The active queue is organized by release: 0.2.0 (ordinary API,
lavaan-compatible fitting, primary inference, MI/release completion and
release readiness), unscheduled interface/composition/maintenance work, and
0.3.0 (fitting reliability, PSD, mixed data and barriers). Tasks
state a result and completion check; completed work and run histories live in
the maintained contracts, test ledger and experiment reports. Simulation has its
own backlog, research activity belongs to the experiment index, and paper-local
planning belongs to the independent paper. Neither an implemented method nor
an active research study creates an obligation to expand the library surface.

### Shared fitting composition

Direction recorded 2026-09-30. Retain the limited-information capability
currently called catML: normal-theory ML discrepancy on Stage-1 polychoric
moments and model-implied correlations, with saturated thresholds. Consolidate
its moment-target map and ML kernel rather than maintaining a separate
estimator identity. Remaining mixed and barrier-specific composition work
follows in 0.3.0. Existing frontier status,
entry points and ordinary-user defaults remain until validated replacement or
migration decisions.

Composition distinguishes **moment source**, **model target**,
**discrepancy/weight**, **covariance domain**, **model penalty** and **algorithm**.
Moment sources include complete continuous, pairwise MCAR, saturated-FIML and
polychoric/polyserial inputs in their supported slices. Provenance retains
sampling covariance/influence; an S matrix alone is not an inference contract.
Direct observed-data FIML retains its pattern-likelihood composer. Mixed ordinal
ML does not follow automatically from all-ordinal correlation-ML support.

Model barriers should compose with applicable retained discrepancy/moment
combinations and direct FIML. Preserve input moments unless a separately
requested Stage-1 transformation changes them. PSD constrains fitted primitive
covariance blocks; a barrier changes the objective and protects the faces
specified by its target. Neither repairs indefinite input moments for an
inverse/log-determinant criterion. Penalized sum-of-squares need an optimizer
that handles the changed scalar objective.

Store unpenalized discrepancy and penalized objective separately, audit the
actual optimized objective, and retain penalty target, strength and
normalization. Each combination needs its own domain/strength and sampling-law
validation. Gaussian log-likelihood/AIC/BIC claims do not follow from applying
the ML discrepancy to estimated ordinal moments. Interior, PSD-boundary and
barrier covariance/tests/intervals are distinct inference contracts.

Penalizing saturated Stage-1 FIML differs from penalizing the final SEM.
Latent determinacy is zero without genuine latents: a Stage-1 penalty needs an
observed-covariance target, uncertainty propagation and a separate H1-reference
contract. The adopted
[ordinary API](../design/r-interface-vision.md#ordinary-api) selects friendly
`covariance = "unrestricted" | "psd" | "barrier"`, with λ through
`barrier(lambda)`, the penalty target fixed by magmaan, and start overrides in
`options`. Implementation/migration remain pending; internal domain
constraints and penalties remain independent. Experimental barrier fitting is
exposed before complete development; barrier-specific hardening and validated
inference remain 0.3.0 work.

#### Implemented composition (2026-10-01)

**Moment targets.** `model::MomentTarget` distinguishes covariance and
correlation targets. `model::correlation_evaluation()` projects grouped
covariance values and optional vech Jacobians to correlations, drops mean
moments, and fixes unit diagonals with zero derivatives. Positive finite
variances are required; the consuming discrepancy owns definiteness.
`estimate::ml_objective()` accepts either target and keeps its two-argument
covariance entry point and half-discrepancy scale. Correlation inputs must
already have unit diagonals and no mean moments; inputs are never repaired.

**Ordinal association ML.** The methods-development surface calls the former
catML capability **ML**. `api::fit(model, ordinal_data, api::ml())` and the
`estimate::frontier::fit_ml()` / `fit_ml_psd()` overloads on `OrdinalStats`
dispatch to it. The separate `fit_catml()`, `fit_catml_psd()`,
`catml_objective()` and R `frontier_fit_catml_psd()` names are removed;
historical diagnostic formulas, Newton audit identifiers and PSD stress arm IDs
keep their categorical-ML names. In `magmaanlab`, `fit_model(spec, data,
estimator = "ML")` with every variable ordered, or with prepared ordinal stats,
selects the route; `psd = TRUE` uses the same discrepancy over PSD primitive
covariance blocks, and `frontier_fit_ml_psd()` accepts ordinal stats.
Stage-1 thresholds are affine restrictions at their fitted values, excluded
from the search; fixed unit residual variances supply the association gauge;
delta/theta are reporting conventions over one correlation target. The active
association-Jacobian rank sets df (`npar_active`, `association`); an
unidentified association map is an error. Linear loading equalities, including
grouped ones, are supported. Threshold constraints, mean/intercept requests,
released response scales, non-unit response-variance constraints and nonlinear
equalities are rejected before preparation. Provenance records moment
source/target, discrepancy, domain, algorithm, saturated-threshold policy and
unvalidated inference state. Gaussian likelihood/AIC/BIC and post-fit sampling
inference are unavailable pending their own contract. Mixed/polyserial ML is
unsupported.

**Covariance policies.** `fit_model()` and prepared `estimate()` accept
`covariance = "unrestricted" | "psd" | "barrier"`; `psd = TRUE` remains a
compatible spelling. Barrier options select `target = "joint" |
"determinacy"` and a finite non-negative `weight` (default 0.25);
contradictory options and extra barrier bounds are rejected.

| Moment source | Shared PSD / barrier fitting coverage |
| --- | --- |
| Complete continuous and pairwise MCAR moments | ML, ULS, GLS, explicit fixed-weight WLS |
| All-ordinal polychorics | Association ML, ULS, DWLS, WLS |
| Saturated continuous FIML/EM | ML2S with NT-ML or fixed ULS/DWLS/ADF/DLS Stage 2 |
| Direct continuous FIML | Observed-pattern likelihood, separate raw-data route |

In C++, the ML/FIML penalty entry points and
`estimate::frontier::fit_gmm_multiinfo()` / `fit_ordinal_multiinfo()` compose
one scalar penalty wrapper; the `api::EstimatorSpec` facade has no covariance
option. The objective is `f - weight / N * P`. Fits keep unpenalized `fmin`
and report `penalized_fmin`; stationarity uses the penalized objective.
ML/FIML/GMM Newton audits include analytic penalty curvature; ordinal penalty
routes provide geometric stationarity only. Observed Sigma must be PD for a
positive-weight penalty, including LS routes; zero weight keeps the base
domain, and an undefined penalty at an improper solution is NaN. Continuous
ML, fixed-weight quadratics and direct FIML barriers use the unit-normalization
contract: quadratic weights move with the moment units, and direct FIML
reports the caller-unit likelihood including its missingness-pattern Jacobian
constant. Continuous pairwise fitting is a moment source selected with
`missing = "pairwise"`, retaining pairwise stats, overlap and missingness
provenance. ML2S retains its complete Stage-1 object (moments, sample sizes,
ACOV/influence, regularization provenance); PSD and barriers change only the
final SEM fit. Refits and composition metadata retain these choices.

**Shared fixed weights.** Complete continuous and all-ordinal fitting share
ULS, NT/GLS, DWLS, WLS/ADF, fixed-a DLS and supplied W through
`estimate::gmm::FixedWeightKind`, `FixedWeightOptions` and
`fixed_moment_weight()`; ordinal and saturated-FIML weight names alias them.
The ordinal NT weight uses empirical threshold covariance, normal-theory
correlation covariance and zero cross blocks. DLS interpolates Gamma before
inversion (a = 0 is NT, a = 1 is ADF). NT is a quadratic weight for LS; the ML2S
`stage2_weight = "nt"` spelling still selects the ML discrepancy. R selects
weights with `fit_model(..., estimator = "WLS", weight = "dls", dls_a = .3)`,
prepared `estimate(..., weight = )`, `prepare_weight()` or
`estimator = "DLS"`. Weights stay fixed during optimization although they may
be estimated from the data; composition records `weight_frozen` and DLS a.
Refits rebuild data-derived weights and preserve supplied ones. Pairwise MCAR
empirical weights are unavailable. Continuous LS `vcov()` uses the weighted
sandwich with retained W, without an estimated-weight influence term.

**Limits.** General penalized and pairwise sampling inference are
unvalidated; new compositions reject generic SE/test and likelihood-based
fit-measure helpers instead of returning complete-data formulas.
Mixed/polyserial ML and barriers, two-level policies, an analytic
penalized-ordinal Newton audit, cross-route dispatcher metadata and
penalty-inference calibration remain open.

**Validation.** Each slice passed every non-`parity` C++ test in an optimized
build (1,339 rising to 1,367 tests), including independent finite-difference
derivatives, grouped projection, observation-unit invariance, invalid inputs,
overidentified/grouped/loading-constrained ordinal models, fixed-weight
criterion preservation, zero-weight reductions at improper solutions, affine
constraint transport, threshold-free penalty derivatives, fit-only ordinal
ULS/DWLS without sampling Gamma, and a strict L-BFGS mixed-unit gate for ML and
direct FIML. Isolated development installations passed the full compiled and
ordinary-user R suites with live lavaan checks, covering every tabled route
with both barriers, prepared/convenience/refit equivalence, unchanged Stage 1
and inference exclusions.

The [active composition queue](../backlog/todo.md#shared-fitting-composition)
owns remaining dependency order and gates. Further pruning requires individual
consumer/dependency decisions and preserves shared primitives; parking
SAM/FC-SEM expansion does not authorize deleting their existing surfaces.

## Current State

2026-10-03: the ordinary FIML policy is composed for single-level random-x
models with affine equalities, one or more groups. Covariance is the
observed-bread casewise-score sandwich. Global and nested score tests use
observed sensitivity with the expected metric. The global LR uses the saturated
observed-H1 spectrum, and the nested LR uses the larger fit's observed bread
and direct casewise-score meat through the exact restriction map, reported with
SB and PEBA4. Calibration is limited (research/44, decisions/03). The lab's
transported-influence FIML Satorra driver is unchanged and is no longer a default.

2026-10-03: misspecification-robust inference is a
[scope requirement](../scope.md#misspecification-robust-inference-requirement).
Lab `estimated_weight` switches default to TRUE, and generic lab covariance and
nested scores default to robust geometry. Named compatibility and diagnostic
routes stay explicit, and unsupported observed or estimated-weight laws fail.

2026-10-03: the ML nested geometry is calibrated (decisions/04). The score test
is primary (observed-H0 sensitivity, expected metric); the LR uses the observed
larger-model Satorra-2000 spectrum. Both report SB and PEBA4.

2026-10-03: the `lavaan-0.7.2` preset fits all-ordinal DWLS in both R packages.
The named eight-case simulated parity set passes all 160 replicates under the
approved endpoint contract (exit criterion 2). `LatentStructure` records its
ordinal preparation, projected through the partable and an R attribute;
re-preparation is a no-op or an error.

2026-10-03: `magmaan_model()` lazily caches native prepared structure;
`magmaan()` fits through it and rebuilds after serialization, falling back for
ML2S and ordinal fitting options. Prepared ordinal fits match fresh
equality-constrained fits and retain the reporting moment/weight layout.

2026-10-03: the DWLS policy calibration study (decisions/05) has a frozen
runner and pilot; production runs on Modal, one container per cell.

2026-10-02: ordinary reporting selects historical lavaan bundles through
`lavaan_compat = NULL` (policy default), with matching cache keys, inference
metadata and result attributes. Explicit output says lavaan compatibility;
the unreleased `convention` argument was renamed without an alias.

2026-10-02: RBM exposes fixed/estimated-weight selection. (That day's FALSE
lab default was reversed on 2026-10-03.)

2026-10-02: CI is configured for both portable R source-package checks, with
weekly dev-preset sanitizers and domain-level LLVM coverage artifacts (no
percentage gate).

On 2026-09-30 the research surface was pruned: ML ridge continuation,
automatic marker/std.lv fitting, adaptive Guttman composites, MI4/structured
fourth-moment weights, empirical-Bayes DLS selection, fitted-weight GMM and its
profile/audit adapters, PNTML, and the dedicated GLSpw fitter were removed.
Fixed-scalar DLS, ordinary fixed-weight GMM/WLS, the start transport pipeline,
and sphere/gauge coordinates remain supported. Archived experiments retain
historical source references; rerunning them requires their original revision.

Pairwise handling is a moment/data axis under MCAR, independent of the fitting
discrepancy. `data::pairwise_sample_stats` supplies covariances and marginal
means that can be composed as `SampleStats` with ML, ULS, GLS or caller-fixed
WLS/GMM, subject to each objective's domain. `data::gamma_nt_pairwise` and the
pairwise casewise/reduced-Gamma inference primitives remain available. A fixed
weight can be built from that Gamma and passed to ordinary GMM/WLS; it has no
separate estimator label. Pairwise sampling covariance is still required for
inference: supplying moments does not make complete-data SEs or test statistics
valid, nor establish general MAR consistency.

Pairwise and cluster sample-summary builders now return `PostError::NumericIssue`
for non-finite computed statistics, including overflow from finite observations.
Cluster input checks cover only selected columns; output checks cover the grand
mean, within scatter, and every size-pattern cluster-mean sum and cross-product.
Multigroup cluster errors identify the failing block. Pairwise missingness
inference and explicit masks are preserved, and finite singular summaries remain
valid. Regressions cover non-finite selected observations, mean/product/sum
overflow, between-moment overflow with zero within scatter, unused cluster
columns, missing-value placeholders, and later-block error attribution.

Pairwise normal-theory Gamma now uses the same observation rule as pairwise
sample summaries: an explicit mask when supplied, otherwise finite raw entries.
The dense metric, streamed expected-information bread, and reduced Gamma all
apply missingness corrections to NaN-coded data without a mask. Shared private
validation rejects inconsistent raw/covariance/availability/mask shapes,
non-finite covariances, and invalid availability probabilities before indexing;
reducers also require raw row counts to match their retained inference geometry.
Regressions compare inferred and explicit masks, independently counted four-way
overlaps, and complete-data identities, and exercise malformed-input errors.

Complete-data moment helpers (`sample_stats_from_raw`, `empirical_gamma`,
`empirical_gamma_with_means`, `gamma_nt`, and `gamma_nt_with_means`) reject
non-finite inputs and arithmetic overflow with `PostError::NumericIssue`
rather than returning successful objects containing NaN or infinity. Finite
singular moments remain valid outputs; these helpers do not require positive
definiteness. Raw-data unit regressions cover NaN, both infinities, multiblock
error attribution, and overflow of means, second moments, and fourth moments.

Research experiment `research/50-normal-parameter-intervals` composes existing
ML fitting and score primitives into a validated normal-theory Wald/score/LR
interval comparison and a candidate-specific bootstrap Bartlett pilot. Its
600-dataset ordinary run gives 4,800 valid intervals with no truth-test/inversion
disagreements; the profile LR agrees with installed lavaan and the score passes
an independent finite-difference check. The study records admissibility events,
bootstrap convergence failures, endpoint-noise checks and compute cost. This
is experiment-local orchestration, not a new exported score-interval API or a
change to the ordinary-user policy.

Research experiment `research/51-robust-parameter-intervals` adds the paired
normal, t(10) and heterogeneous-gamma comparison: observed-sandwich Wald,
expected-information robust score/LR inversion, information-choice variants
and uncorrected controls. Its 66 numerical checks cover likelihood scaling,
full-model restricted scores, the policy covariance, lavaan LR agreement and
endpoint inversion. The experiment fixes uncentered score second moments;
the landed profile helpers instead center empirical contributions, and the
scalar relationship is checked explicitly. This is experiment-local
composition of existing primitives, not a new exported interval API or an
ordinary-user default change. The 3,000-dataset confirmation records all
54,000 interval attempts, failures, domain events, paired coverage and cost.


Research experiment `research/49-spectral-tail-calibration` adds local R
prototypes for analytical nonlinear covariance shrinkage, increasing-index
cycle moments through order six, positive-spectrum reconstruction, and
truncated-CGF right-tail rules, compared with pEBA-4 for complete-data global
score and ML LRT tests. It checks observation-level spectra against the existing
core and estimates moments on an additional independent sample as a diagnostic. These are
advisory prototypes, not exported methods or changes to inference policy;
covariance-loss shrinkage is not population-spectrum recovery, and fitted
projections invalidate exact cycle-moment unbiasedness on the fitting sample.
Validation includes brute-force cycles, fixed-projection Monte Carlo moments,
scale identities, and a scaled-chi-square saddlepoint reference. The report
records the 500-replication-per-cell null pilot and failure rates.
The focused `--mv` follow-up adds an all-distinct, translation-invariant
U-statistic for the second spectral moment, comparing tau2-only and both-moment
Satterthwaite corrections without leave-one-out refits. The 12,000-dataset
one/two-factor normal/skewed pilot improves conservative score calibration but
can overshoot for LRT; corrected moments frequently violate the nominal-rank
spectral bound. The positive four-observation kernel agrees with brute-force
enumeration and fixed-transform Monte Carlo checks. Same-sample nuisance bias,
nominal-rank moment feasibility, and power remain unresolved; no core or R
package default changes.
The paired `--mv --metrics` replay holds expected nuisance sensitivity fixed
and compares expected versus observed-H0 score weights. Expected-score/LRT
outputs reproduce exactly. Observed weighting makes tau2-corrected MV liberal
(39.1–41.3% versus 4.9–5.6% on paired skewed p=12, n=100 fits) and adds 26
nonpositive-metric failures; keep expected weighting for this candidate.
The `--oracle` replay adds analytic population fourth moments and constrained
MV, with Gaussian-mixture calibration checks and statistic-moment diagnostics.
The original latent-component CFA generator has an exactly flat oracle spectrum;
perfect eigenvalues still leave finite-sample score/LRT errors there. A smaller
`--mv --ig` grid now covers 10,000 two-factor datasets (p=8/12, n=100/500,
normal and moderate/severe Pearson IG with symmetric/Cholesky roots). Symmetric
roots remain nearly flat; severe Cholesky weights extend to 5.20. At p=12,
n=500, severe Cholesky oracle rejection is 4.2% score/4.6% LRT versus corrected
MV's 9.8%/9.6%; the lower-bound constraint leaves those rates unchanged. There
are 332 common fit failures, including 19–22% in severe Cholesky n=100 cells,
so results condition on usable fits. Population geometry, generator moments,
and bound-active SB equivalence are independently checked. These diagnostics
support investigating feasible moment underestimation on nonflat spectra;
they do not promote corrected MV or change inference policy.

The corrected-corpus optimizer study (`engineering/active/17-corpus-optimizer-recovery`)
compares current ordinary ML/GLS starts across L-BFGS settings, PORT, SLSQP, and
explicit recovery, retaining backend status, fit verdict, objective quality,
preparation exclusions and wall-clock censoring separately. It is advisory and
changes no production default. Among 304 prepared cases per estimator, default
L-BFGS and PORT produce 239 versus 287 accepted, objective-matching ML fits,
and 282 versus 287 GLS fits; tighter L-BFGS improves GLS to 286 but not ML.
The two historical Newsom GLS failures do not
reproduce on the corrected inputs; evaluator cancellation remains an unproven
explanation, not an established defect. Early scale-sensitive L-BFGS domain
failures still reproduce in a scalar probe and corpus models. The Little phantom
poor stationary point fails the curvature audit; a verified-start restart passes.
A targeted start cross-check recovers 10 of 12 ML curvature cases in both PORT
and lavaan by replacing zero free latent-path starts with 0.5. This is diagnostic
evidence for start-policy work, not a new default. Lavaan also stalls at the
phantom CFA's zero-path start; the original LISREL input supplies nonzero starts.
Three GLS cases retain a PORT recovery gap even with starts shared with lavaan.
An equivalent-model probe confirms that latent regressions and higher-order
loadings currently receive different simple starts despite representing the same
scale paths. Structural initialization beyond the loading constructors remains
an explicit gap; no blanket nonzero-regression policy has been adopted.
The zero-path failures are exact sign-reflection traps: a latent scaled by a
fixed variance (std.lv or phantom) with all sign-odd free paths at zero has zero
gradient in them, so no gradient optimizer leaves. The layered moment start
(`estimate::layered_start_values`, R `start = "layered"`, the complete-data ML
and GLS default since 2026-09-26) builds measurement shapes, the measured-latent covariance, a joint
identification scale solve and a GLS latent-level structural fit, with
sign-generic moment magnitudes for variance-carrying paths, then means, a
unit-weighted constraint projection and PD repair. It is identification- and
spelling-invariant in its implied start covariance and equivariant under
observed rescaling on the unit-test models, but not yet on 32 of 239
unit-invariant corpus pairs (engineering/active/17-corpus-optimizer-recovery). On the 608-pair engineering/active/17-corpus-optimizer-recovery scan it gains
113 and loses 4 case fits across ML/GLS × PORT/L-BFGS; the losses are optimizer
terminations from lower start objectives. Simple/FABIN no longer read a zero
disturbance as a std.lv scale. Trait-state blocks whose latent covariance only
the structure identifies keep FABIN3.
Observed-only blocks bypass the empty latent solve (2026-10-02); their
layered starts retain sample means and variances, including saturated models.
Default decisions for starts, optimizers and the PSD route are made in
`experiments/decisions/01-optimizer-defaults`, whose report opens with the
register of those defaults. Its criteria are committed before each run, the
library verdict is the only judge, and it uses held-out simulated populations
under unit transforms. Since 2026-09-26 the layered start is the default of
complete-data ML and GLS (`ml_start_values`, `api::ml()`/`gls()`, R `fit_ml`,
`fit_gls`), and the optimizer stays NLopt L-BFGS. PORT certifies more, but the
Newton check accepts runaway points along divergent paths, and PORT reaches
about eight times as many points flagged by the study's extent screen. Since
2026-09-27 the next step is a small sphere-reference investigation separating
chart proximity, extreme parameters and numerical accuracy; the screen is not
a proof of nonattainment and PORT promotion remains undecided. For PSD ML,
FABIN3 and the diagonal
preconditioning stay, in the direct fit and in the ordinary stage of the
explicit fallback; a second pre-registered run after the ML promotion
rejected the layered start for both. The two-stage route beats the direct fit
only in rescaled units, so the direct fit stays the PSD route for now
(author, 2026-09-27). The lifted PSD information-scale clamp is now relative
to sample-derived units: original/equality-reduced coordinates reuse the shared
unit machinery, and Cholesky entries take their row variable's unit. Bounds are
`[1e-4*u, 1e4*u]` and the zero-information fallback is `u`; unclamped information
scales retain their original arithmetic. Model constraints and the PSD domain
are unchanged. Tests cover transported starts across uniform/mixed units,
marker/std.lv identification, correlated residuals/equalities, zero information,
and fitted boundary agreement under ×0.01 and ×100. The full estimate suite
passes (535 tests).
A small paired follow-up on 1,356 fresh problems per route keeps FABIN3-auto,
ordinary L-BFGS and constrained SLSQP explicit. The coordinate fix alone does
not settle the route comparison: direct certification stays at 1,328/1,356
(eight gains, eight losses), and two-stage changes 1,333→1,332. Under the fix,
two-stage matches 1,326 best observed objectives versus direct's 1,298, while
native certification remains slightly better for direct (351 versus 350).
The retained failures motivate investigating the still-absolute start eigenvalue
floor and covariance-link residual units.
Direct PSD remains the ordinary route, two-stage remains explicit frontier,
and spectral starts remain banked for a later defaults study.
A subsequent study-local unconstrained normalization pilot standardizes the
sample before constructing FABIN3-auto starts and backtransforms the fitted
parameters, preserving the requested markers or std.lv identification. All
returned normalized points reproduce their covariance/objective in original
units (errors below 6e-16 and 3e-13 respectively). Across 900 unit comparisons,
direct marker PSD goes from nine verdict differences/two certified objective
differences to zero/zero; direct std.lv has no remaining certified objective
differences but five verdict differences. This does not establish better basin
selection: normalized two-stage fits sometimes reach worse local minima.
**Default adoption closed (2026-09-27):** retain normalized fitting for
single-level, continuous complete-data ML and direct PSD ML. The existing
`normalize_sample=true` default is confirmed; no estimator/domain or route is
silently changed. Groups, means, equal labels and affine linear equalities are
inside this scope. Arbitrary `<`/`>` constraints are not supported; FIML,
multilevel, ordinal and barrier normalization remain outside it. The scope and
validation contract are in [optimizer controls](../reference/optimizer-controls.md#complete-data-ml-and-psd-sample-normalization-2026-09-27).
Closeout reran 13 normalization/fallback estimation tests (270 assertions),
20 staged API tests (459 assertions), and the R affine/start/fallback integration
example; all passed. Existing C++ tests include unequal-size groups, cross-group
loading equalities, means and mixed scales. The empirical challenge bank remains
single-group development evidence, not an exhaustive feature/estimator claim.

Complete-data ordinary ML and PSD ML now share a model/data normalization
before optimization (author request, 2026-09-27). The staged C++ and R entry
points construct automatic starts there; explicit starts, hints and ML bounds
keep their caller-unit interpretation. The transformation supports single-level
single/multigroup models, means, fixed/structural cells, and cross-group equal
labels/general linear equalities. Equalities become weighted rows A D z = b;
constraint rank must be preserved. Returned theta and post-fit model evaluation
remain in the user's identification and units. Fallback stages each transform
their supplied original-unit starts, including warm starts.
`OptimOptions::normalize_sample` / R `control$normalize_sample` enables this
behavior by default; `Estimates::sample_normalized` / R `fit$sample_normalized`
reports whether it applied. Optimizer tolerances, terminal audit and numerical
fit diagnostics refer to the internal fitting representation. The separate
optimizer preconditioning choice remains available within that representation.
Both ambient and PSD complete-data ML Newton audits use the shared normalized
model, preserving accuracy thresholds and returning retained full-space
derivatives and geometry maps in caller coordinates. Earlier saved-point
regressions recovered five unit-induced PSD false failures among 4,742 points;
that development evidence is not a fresh comparison of full fitting defaults.
A paired development revisit of 1,356 saved cases (including linear equalities
and means) finds no unit-dependent verdict/objective/covariance differences for
normalized direct PSD; equality-family successes improve from 167/180 to 177/180.
ML and fallback results remain mixed. Saved-point inspection identifies 23 old
mixed-unit fallback solutions accepted despite indefinite covariance blocks;
caller-unit admissibility scale robustness remains open. All convergence and
objective regressions are retained in the optimizer-defaults study.
The separate engineering/active/15-sphere-reference-fits challenge rerun (120 saved draws, 960 fits) finds
no target-recovery gains for layered/native ML or FABIN3-auto direct PSD.
Later-batch two-stage PSD loses three targets (two budget failures and a worse
local solution); six of seven saved finite ML witnesses still pass direct
accuracy checks. Normalization does not close the start/solution-search gap.
Fixed-endpoint diagnostics reproduce the three two-stage regressions outside
the wrapper: the handoff endpoint and PSD path both matter. A fivefold budget
does not recover the lost targets. Certified-only warm starts gain two targets
and lose one across the 120 saved cases; this retrospective tradeoff is not
promoted. Two-stage retains its existing handoff and remains provisional.
The user's clarified independent-fallback policy was then fitted on all 120
cases: accept certified/admissible ordinary ML, otherwise construct PSD's own
FABIN3-auto start. It matches direct PSD's outcomes and objectives, stops after
ordinary ML in 34 cases, and gains two targets while losing one against warm
fallback. This is a tested candidate; the production fallback still uses warm
starts. Repeated balanced timings on that bank show no speed advantage: direct PSD
costs 0.553 s per pass, independent fallback 0.703 s, and warm fallback 0.839 s
(sum of per-case median costs). Direct is fastest in each of six rounds.
These are current R-workflow costs, including failures; other workload mixes
and a future native independent-fallback implementation are unmeasured.
Paired off/on timing subsequently finds nearly unchanged median fit costs
(ML 0.47→0.48 ms, direct PSD 1.07→1.09 ms). Direct PSD's 14% aggregate increase
is dominated by one newly failed budget-limited case, not routine per-fit
scaling overhead. The six normalized PSD failures consume 69% of its bank time.
Normalization can still change which local solution is found. Barrier, FIML,
multilevel and nonlinear-equality fitting retain their existing paths, and the
remaining cross-route programme follows the release priorities above. The direct
versus ordinary-then-PSD route decision is unchanged.
The complete-data ML barrier fitter (`frontier_fit_ml_multiinfo`) keeps
transported FABIN3 with PORT after lane barrier-ml
(`experiments/decisions/02-barrier-defaults`, 2026-09-27): the layered start
stalls next to the factor-disappearance set that the determinacy penalty
favours, and L-BFGS stalls in line searches. That lane also found a ×0.01
start fallback failure in equality-constrained models and certified fits near
marker poles; both are in the backlog.

magmaan is a C++23 library for methods developers working on linear SEM. It is
built under `-fno-exceptions -fno-rtti`, Eigen runs under
`EIGEN_NO_EXCEPTIONS`, and fallible APIs return `std::expected<T, Error>` —
the single C++23 feature the library depends on. Extension points are free
function templates over structural (duck-typed) interfaces rather than virtual
hot-path interfaces or `concept` constraints.

Lavaan remains the oracle. Parser output, lavaanified partables, point
estimates, standard errors, and chi-square statistics are compared against
checked-in lavaan fixtures where support is claimed. Fixture regeneration is
done with `cpp/tests/tools/regen_oracle.R`; CI does not invoke R.

The lavaanified model contract is the triple:

- `LatentStructure`: the estimable model, name-free except where estimator or
  identification choices require structure.
- `LatentNames`: the verbal model, including variable names, labels, groups,
  group levels, and `.pN.` plabels.
- `Starts`: free-parameter start hints.

`to_lavaan_partable()` and `from_lavaan_partable()` project this model to and
from `LavaanParTable`, which is the compatibility format used by R bindings and
golden `parTable()` fixtures.

`effect_coding = TRUE` uses lavaan's loading and intercept identification:
loading sums equal the number of indicators, indicator intercept sums are zero,
and automatically supplied latent means are free. A fixed indicator intercept
suppresses that factor's intercept coding; explicit latent-mean fixes remain
binding. Explicit fixed loadings suppress the corresponding loading constraint.
The builder applies the lavaan reference-group convention under scalar
invariance. Covariance-only models retain loading-only coding.

Ordinal and mixed fitted parameter tables derive fixed-scale delta residual
variances and fixed theta response scales through
`estimate::ordinal_parameter_values`. Delta residuals use the fitted explained
variance and unit response variance; theta scales use the inverse square root
of the fitted response variance. The helper returns owning values
in structure-row order; it leaves preparation values, start hints, and free
coordinates unchanged. R result constructors share this reconstruction, including
prepared and post-fit ordinal routes. Theta residuals and free response-scale
coordinates retain their fitted values. Released-scale delta invariance remains
outside validated coverage. Targeted pinned-lavaan fixtures cover effect-coded
single/multigroup and scalar-invariance fits, explicit fixes, and single/multigroup
ordinal/mixed delta/theta reporting; independent checks cover mean/covariance
invariance, constraint sums, residual-plus-explained variance, and unit variance
after response scaling. Nonpositive or nonfinite theta response variances fail
explicitly instead of producing invalid reported scales.
Delta reconstruction also covers residuals stored in `Psi` on observed-variable
phantom latents in reduced models; it uses observed indices for implied variances.

Explicit single-level `group:` headers select separate group templates in
header order instead of being interpreted as levels and replicated across
groups. `FlatPartable::block_kinds` preserves the parsed header axis, including
through composite expansion. Header counts must match the requested groups;
mixed axes and generic `block:` construction fail explicitly. `group_equal`
matches explicit templates by parameter term even when row order/count differs.
Frozen lavaan parser/partable fixtures and R ordinal delta/theta, mixed/continuous,
and staged-model regressions cover grouping and complete fitted tables,
including theta response scales.
The two-level `level:` path retains its existing templates and mean rules.

Continuous multi-group `group.equal = "intercepts"` releases auto-added zero
latent means in groups 2+ unless `means` is also equal. Explicit user mean
rows remain authoritative, and `group.partial` does not suppress the release.
This identification change lives in `spec::build`, alongside the equality
labels, so keyword scalar invariance no longer requires explicit latent-mean
syntax. Unit guards cover release, equal means, fixed user means and partial
intercept invariance, three-group marker/std.lv models, per-group explicit
means and growth identification. The R scalar-invariance suite gates ML/FIML
parameter rows, estimates, df and chi-square, three-group CFA and growth fits,
and complete-data scalar nested tests against lavaan. The ordinary R package
also gates scalar-invariance MLR standard errors. FIML scalar nested scaled
statistics now match lavaan MLR under the explicit lavaan convention, including
marker/std.lv identification and overlapping missingness patterns. An offline
HS school fixture freezes both fits' parameters and gates all three computation
modes independently of optimization. These are value-parity, not calibration,
gates.

### Optimizer control semantics

`OptimOptions` exposes optional backend blocks for NLopt, PORT, IPOPT and
Ceres. The thin R interface accepts matching named sublists in `control`.
Explicit settings override legacy mappings;
NLopt gradient/step tolerances, memory and evaluation budgets are distinct,
as are PORT step/function criteria and evaluation/iteration budgets. See
[optimizer controls](../reference/optimizer-controls.md) for support, sentinel
values and compatibility rules. These settings govern search termination;
they do not alter the independent terminal audit. Effective-control and raw
stopping-code reporting across fitted results remains backlog work.

Complete-data ML now defaults to the validated NLopt profile (5000 evaluations,
relative objective/step tolerances 1e-12/1e-10). High-level ML uses transported
std.lv FABIN starts where safe, with native fallback and user-hint preservation.
The start pipeline separates native construction, preparation of an auxiliary
identification and transport of a supplied vector (`estimate/start_pipeline.hpp`).
Policies select native, automatic fallback or required transport; typed reasons
distinguish unsupported layouts from numerical failures. Optimizer-coordinate
scaling remains independent. R's ML and start-helper defaults agree on automatic
transported FABIN3, and explicit `default` matches omission. Native method names
remain available; start metadata reports the actual branch and fallback reason.
`StartMethod::Layered` is never transported (it constructs in the target
identification) and reports constructor notes on `StartValues::notes`.
Simple and FABIN starts interpret observed loadings, residual variances and
intercepts by their model meaning even when Reduced LISREL stores them in
Beta/Psi/Alpha phantom-state cells. This fixes skipped FABIN loadings,
full-variance residual starts and zero intercept starts in structural models;
the Geiser latent-path marker default then passes ordinary L-BFGS/SLSQP and
PSD-SLSQP without oracle starts or optimizer changes.
Every scalar backend searches complete-data ML, GLS, the moment least-squares
family, pairwise GLS, the constrained ML/GMM entries and FIML in shared
unit-equivariant optimizer coordinates (`estimate/coordinates.hpp`): the
equality-reduced parameter in sample units (observed SDs and each latent's
identification), refined downward by the expected information at the start,
with means and intercepts centered at their starts. This is the default
(`OptimOptions::coordinate_scaling = Information`); `SampleUnits` and `None`
remain selectable. The objective, constraints, bounds and the reported terminal
audit keep model coordinates. PSD ML keeps its lifted information scaling and
constraint tolerance 1e-8; ordinal, two-level and several frontier routes stay
in raw coordinates (backlog). See the optimizer-controls reference.
Explicit complete-data ML recovery is available as
`estimate::frontier::fit_ml_psd_fallback` (header
`estimate/frontier/ml_psd_fallback.hpp`) and R's
`frontier_fit_ml_psd_fallback()`. It accepts ordinary L-BFGS only when the
common fit verdict and covariance admissibility both pass; otherwise it
runs PSD-SLSQP once. A finite ordinary return with PD implied covariance and
satisfied model equalities supplies parameter starts, even if its accuracy
check fails or primitive covariances are improper. Hard errors or unusable
returns fall back to the original start. The PSD initializer projects factor
starts, not final estimates; initial link equalities may be violated. The
policy preserves both attempts as estimates or errors and returns no accepted
fit if recovery fails its verdict/admissibility checks. R exposes the selected
fit separately from the attempts, trigger reason, and warm-start flag. Both
stages retain independent optimizer controls. Ordinary ML defaults and the
separate ordinary L-BFGS-to-SLSQP backend remain unchanged. Focused C++/R
gates cover skipping, improper warm recovery, ordinary errors, inaccurate
ordinary returns, and failed recovery; this is not a global-optimality policy.
Acceptance does not yet certify local identification: the free-marker CFA
with seven parameters for six moments can pass Newton accuracy and be selected
by PSD recovery (board TASK-33.3). The R example retains this witness, reports
that limitation and checks the current verdict/admissibility selection rule.
Complete-data ML fits (ordinary, equality-constrained, PSD, Fisher scoring
and IRLS; not penalized fits) carry the Newton accuracy check in
`FitDiagnostics::newton_accuracy` (R `fit$diagnostics$newton_accuracy`):
d = sqrt(G' I^-1 G) from the total score and observed information after
linear-equality reduction, with the accepted budget d <= .01, equilibrated
condition <= 1e12 and relative solve residual <= 1e-10. Since 2026-09-24 the
common verdict uses it at regular interior points of the fitting domain:
every ambient fit, improper estimates included, and every PSD fit without a
singular primitive covariance block. There d <= .01 passes, and
nonpositive curvature, ill conditioning or an unreliable solve fail. PSD ML
fits carry the covariance-domain version (`newton_accuracy_ml_psd`): at a
boundary point the Newton step is restricted to the face of the PSD cone the
estimate lies on (null directions with positive multipliers held, the face's
curvature 2 tr(M dC C^+ dC) added), so PSD fits are judged by d <= .01
everywhere. Since 2026-09-25 FIML (ordinary and PSD) and every
moment-quadratic fit (GLS, ULS, WLS, DWLS and fixed-weight GMM, SNLLS, and their PSD versions) carry the same check with the exact
analytic Hessian of their objective: FIML measures the step with its observed
information, the least-squares fits with the normal-theory sandwich
(`d^2 = G' Omega^{-1} G`, Omega the variance of the total gradient), which
makes `d` free of units. The verdict requires a positive-definite implied
Sigma only for likelihood objectives. On the textbook corpus the change
rejects 56 of 833 FIML/GLS/ULS/DWLS fits the first-order check accepted, each a
stationary point that is not a minimum, a stop far from the optimum, or a
near miss of the budget (experiment engineering/active/19-newton-verdict-migration). The ordinal and mixed
least-squares fits (bounded, SNLLS, full-threshold, PSD) carry it too, with
their exact Hessian and the Gauss-Newton sandwich metric
`sum_b n_b Delta_b' W_b Delta_b`, and so do the multi-information barrier
fitters (ML and FIML), whose Hessian `N grad^2 fmin - lambda grad^2 P` uses the
new analytic `frontier::multiinfo_penalty_hessian`. The first-order check still
decides at active box bounds, under nonlinear equalities, and on the CatML and
two-level paths, and it remains telemetry elsewhere. Like any local check, the Newton check can pass a point far along
a divergent path (no attained maximum), where the remaining gain is tiny.
General escape/nonattainment diagnostics are deferred to the
[speculative backlog](../backlog/speculative.md#runaway-estimates-and-nonattainment-diagnostics).
`fit$diagnostics$verdict$criterion` says which check decided. The standalone
`estimate::frontier::newton_accuracy_ml` and R `frontier_newton_accuracy(fit)`
recompute it with other options.
Continuous and FIML interfaces now share explicit constructor/transport controls:
R uses `control$start` / `control$start_transport` and retains `fit$start`;
the friendly C++ API accepts `api::start_policy` and retains `Fit::starts()`.
Explicit vectors are validated for size and finiteness. Existing defaults are
preserved per entry point. Marker-only CFA constructors fall back to their
native method with a reason under automatic transport, and reject required
transport. Retained vectors precede fitter projection/PSD repair/profiling;
per-factor constructor fallback evidence and specialized start adapters remain
in the active backlog. See the start-policy section of
[optimizer controls](../reference/optimizer-controls.md).

Newton computations are available as owning, reusable C++ artifacts under
`estimate::frontier`. `evaluate_newton_ml` retains the full total-gradient and
analytic observed Hessian; geometry preparation retains equality and PSD
tangent bases, the separate curvature correction and reduced system; numerical
preparation retains equilibration and an LLT factorization; solving returns the
signed Newton correction. `assess_newton_accuracy` changes runtime acceptance
budgets without recomputation. `audit_newton_ml` and `audit_newton_derivatives`
compose those stages. Existing ML summary wrappers use the same implementation.

`estimate/frontier/newton_adapters.hpp` adds explicit post-fit adapters for
ULS, GLS, fixed-weight WLS/DWLS/GMM, expanded ordinary LS-SNLLS, FIML,
all-ordinal and mixed-ordinal LS, CatML, two-level ML, and multi-information
penalized complete-data ML/FIML. `audit_newton_ml2s` reuses
retained Stage-1 moments for NT Stage-2 ML and ULS/DWLS/ADF/DLS Stage-2 LS;
neither adapter certifies the outer iteration or Stage-1 convergence. Focused
tests cover all five policies, frozen weights away from an optimum and
unequal-group LS normalization. The check-by-path coverage matrix lives in
`project/design/terminal-audit.md`. FIML uses its analytic observed information;
the moment-quadratic LS adapters use the exact analytic Hessian
(`gmm::moment_quadratic_hessian`) with the normal-theory sandwich metric
(`gmm::moment_quadratic_nt_gradient_variance`), with an explicit Gauss-Newton
option; the ordinal, mixed and penalized adapters use their exact analytic
Hessians, and the CatML and two-level adapters differentiate their actual
gradients. The all-ordinal observed bread and the frontier robust ordinal
paths (misspecification and IJ sandwiches, RBM parts, casewise influence,
profile references) now include the mean columns of the moment Jacobian,
which they omitted whenever latent means or intercepts were free, and skip the
closed-form second-derivative term for free parameters without a model-matrix
cell. Numerical Hessians retain step sizes and h-versus-h/2 and
symmetry diagnostics, with runtime controls and no silent approximation fallback.
LS artifacts also retain their whitened residuals and Jacobians. CatML holds
its Stage-1 threshold coordinates fixed by explicit geometry, rather than
removing numerical Hessian null directions. Explicit box-bound audits now solve
an equality-reduced convex quadratic, retaining the feasible correction,
multipliers, working set and KKT residuals. Fixed coordinates are reduced first;
active bounds may release inward directions. The distance is sqrt(2*predicted
gain), an objective-gain budget at binding inequalities. The reduced Hessian
must be positive definite; iteration limits and singular working sets leave
evidence unavailable. Explicit policies accept this box-aware evidence while
compatibility fit routing remains unchanged. PSD-interior boxes and redundant
nonnegative-variance bounds share existing geometry; genuinely interacting boxes
on singular PSD faces and nonlinear equalities remain unsupported. Exhaustive
two-dimensional face minima, rescaling, equality and boundary tests validate the
new solver; see `project/design/terminal-audit.md`.

Every adapter labels its objective family, curvature source and native-to-total
normalization. Two-level uses its native total negative log likelihood; the
other adapters multiply their per-observation objective by N. The full original
Hessian remains separate from PSD-adjusted curvature. LS and penalized curvature
is not sampling information, and the default .01 distance budget is not a
cross-estimator standard-error calibration. These adapters do not change
non-ML fit-time convergence policy or the R summary interface. Artifact reuse
requires the same model, point, data, weights, objective and parameter order.

`estimate/frontier/convergence.hpp` now provides owning evidence reports and
`convergence_policy.hpp` provides separate runtime assessment. Explicit policies
require objective finiteness, declared-domain feasibility, and first-order,
Newton, or both checks; reported-objective consistency is independently
requestable. Missing required evidence stays unchecked, without a weaker
fallback. Threshold changes reuse measurements and the Newton solve; geometry
changes require recollection. Generic scalar and analytic ML collectors are
available, and existing retained estimator adapters compose without objective
or Hessian reevaluation. Reports retain the point, effective request and full
computations; source data/model fingerprints are not yet recorded.
`common_fit_verdict` delegates to the named compatibility policy, preserving
existing fit-time acceptance. `evaluate_at(ML)` now attaches matching Newton
evidence and accepts an explicit domain. Its legacy default variance bounds
remain; the new report API uses only explicit bounds. Newton distance, step
and gain survive numerical guard rejection so summaries and owning reports
can both be reassessed. Focused tests verify fit/post-fit parity, missing checks,
policy differences, domain feasibility and reassessment without recomputation.
The detailed contract and example are in `project/design/terminal-audit.md`.

FIML joins the versioned fitting adapter on 2026-10-02: EM H1 supplies
loading/location starts, available observations supply residual-variance
starts and standardized retry scales, and the observed-pattern deviance uses the
same ordered QR and PORT retry/acceptance machinery. Both R interfaces accept
the preset for unrestricted continuous FIML; details and parity gates live in
[the FIML capability](capabilities/fiml.md).

The advanced per-call fitting setup (2026-10-01) separates start construction,
search and acceptance in `estimate/configured_ml.hpp`. Both R packages expose
it through `options`; `newton` names the existing common verdict, while the
pinned `lavaan-0.7.2` preset selects native FABIN3/OLS starts, PORT controls and
start scaling, bounded/unbounded R dispatch, standardized/simple-start retries,
and raw-termination plus exactly bound-masked gradient acceptance. Linear
equalities use an ordered Householder QR basis matching lavaan 0.7.2; the
model retains name-free ordered affine rows alongside its native merge/reduced
constraint system. Explicit rows followed by synthetic shared-label rows
survive triple construction and lavaan-partable round trips. The preset packs
starts, scales and gradients through this basis; native fitting retains its
existing basis. Frozen and live gates cover shared labels, `group.equal`
loadings/intercepts, nonzero affine RHS, redundant rows and equivalent systems
with different row orders. Retry derivatives are compared at identical
parameter points; each actual endpoint is checked under the declared acceptance
rule, with unchanged endpoint/objective tolerances. Floating-point search paths
need not produce identical final gradient vectors. A standardized retry with
any nonzero affine RHS returns an explicit error before entering the invalid
scaled surface documented in [the oracle ledger](../validation/oracle-defects.md).
Zero-RHS standardized retries require finite nonzero scales and preservation
of the null space (`A*D^-1*K=0`); homogeneous ratio constraints can also fail
this property. Unstandardized affine fits and shared-label/group-equality
standardized retries remain supported. Invalid constrained initial covariances
return an explicit error;
the unconstrained four-attempt soft-failure contract remains. Component
overrides are reported as a modified preset. Thresholds remain internal.
Frozen fixtures and live 0.7.2 comparisons cover CFA identification, observed
and latent regressions, higher-order/single-indicator models and unequal groups.
The pinned failure gate retains the oracle's original-covariance preflight
across all retries, including its driven-coordinate early return.
`Estimates::selected_verdict` makes C++/R consumers and ordinary-policy gating
agree; common diagnostics remain independently inspectable. Fits retain requested
and resolved settings, numeric controls, starts/scales and attempt selection.
Every `fit_model()` fit records all of its arguments as its route, and every
refit (modification-index and equality-release likelihood-ratio refits, case
reruns) replays them, overriding only the model; arguments and options added
later carry over without changes. Likelihood-ratio refits rebuild their spec
with the anchor's group labels and order rather than grouping by appearance
in the supplied data (2026-10-02), so group-specific candidates stay in their
groups. Lavaan acceptance on magmaan's own PORT
search is judged with `lavaan_acceptance_gradient()` in lavaan's units; the
lavaan search reproduces the same measurement. The supported preset slice is
ordinary complete continuous ML with linear equalities and zero/infinite
bounds. Merged free slots (`ceq.simple = TRUE`), nonlinear/inequality constraints,
other versions and unsupported sources, including pairwise-moment input,
error. Retry gates retain starts, coordinates and verdicts and compare
derivatives at identical points; the older unconstrained ×1000/×10⁵ fixtures
retain their documented endpoint sensitivity. FIML and all-ordinal preset
gates remain in the active backlog; inference conventions and default fits
are unchanged. PORT
status codes follow its source: 8 false convergence (reported as a noisy
objective), 9 evaluation limit and 10 iteration limit (both budget stops).
A compatibility acceptance rule keeps deciding `converged` and inference, and
magmaan's common verdict still runs. `api::PolicyFitState::native_converged`
and `api::verdict_disagreement()` carry a disagreement between them into
policy inference and nested results without changing either; the ordinary
package records `fit$inference$convergence` and prints notes. The witness is
the rescaled exact-fit fixture case that lavaan 0.7.2 accepts at fmin 0.276.

`estimate/frontier/ml2s_audit.hpp` adds standalone saturated-likelihood endpoint
reports and composed ML2S reports for all five Stage-2 weight policies. Stage 1
retains structured per-block EM stopping/repair telemetry in `FIMLH1` and
`SaturatedMoments`; the latter now preserves `raw_H` and total `raw_gradient`
separately from its potentially repaired inference `H`, with repair metadata.
Endpoint audits reuse labelled analytic raw derivatives or evaluate them at the
supplied moments, without EM. Original moments and optional transformed Stage-2
inputs remain separate; transformations clear stale raw derivative slots.
Composed assessment uses independent stage policies and a handoff comparison
against an optional caller-retained fit-input record. Missing handoff evidence
stays unchecked by default; solver-stop evidence can be required separately.
Existing fitting/repair defaults are unchanged. Tests cover missing-data groups,
all five weights, early stops, raw negative curvature versus repaired information,
handoff mismatches and transformed inputs. R report bindings and automatically
captured fit-input records are deferred with the other remaining audit extensions
to [the speculative backlog](../backlog/speculative.md#convergence-audit-extensions);
the design document specifies limits.

### Admissible covariance-model contract

For each group, the continuous complete-data model uses the reduced-LISREL
system

```text
eta = alpha + B eta + zeta
y   = nu + Lambda eta + epsilon
```

with `Cov(zeta) = Psi`, `Cov(epsilon) = Theta`, and
`Cov(zeta, epsilon) = 0`. Writing `A = (I - B)^-1`,

```text
P     = Cov(eta) = A Psi A'
Sigma = Cov(y)   = Lambda A Psi A' Lambda' + Theta.
```

The statistical model's admissible parameter space requires, in every group,
`Psi` and `Theta` symmetric positive semidefinite, `I - B` nonsingular, and
the implied observed `Sigma` positive definite wherever an ordinary Gaussian
density is evaluated. For moment-only estimators, positive-semidefinite
`Sigma` is a covariance matrix, but downstream inverse-based normal-theory
methods still require positive definiteness. Fixed cells, structural zeros,
shared parameters, and equality constraints remain restrictions on the
original LISREL entries.

In pure CFA, `B = 0`, so `Psi` is the latent covariance itself. With structural
regressions, `Psi` is the innovation/disturbance covariance and the derived
latent covariance `P = A Psi A'` is automatically positive semidefinite when
`Psi` is. `B` is a regression matrix and has no PSD requirement; recursive
models make `I - B` nonsingular by construction, while cyclic models need that
condition checked explicitly. Traditional LISREL's separate exogenous
covariance `Phi` and endogenous disturbance covariance are the diagonal
blocks of the reduced system's `Psi`; if cross-block covariances are ever
allowed, the complete joint innovation block must be PSD.

PSD is the model requirement, not merely nonnegative diagonal entries. It
implies the fitted Cauchy bounds on every covariance and all higher-order
joint restrictions. In particular, a zero variance forces its entire
covariance row and column to zero. Singular-but-PSD component matrices are
valid covariance matrices, although they are boundary cases with nonregular
inference.

Core estimation remains lavaan-compatible and does not constrain the optimizer
to this full covariance domain. Continuous ML, LS, FIML, two-level, and related
ordinary `MatrixRep` paths instead finalize with an estimator-neutral
admissibility audit. It records
per-block minimum eigenvalues and PSD/PD status for `Theta` and `Psi`, source
partable rows, negative variance rows, defined correlations outside
`[-1, 1]`, and the implied-`Sigma` PD result. The audit does not change
optimizer convergence or discard the estimate. R exposes it at
`fit$diagnostics$admissibility`, emits one concise warning for an inadmissible
high-level fit, and prints covariance admissibility separately from
convergence. Opt-in `variance_bounds` remain a diagonal Heywood barrier, not a
joint PSD constraint.

The frontier covariance-honest estimators compile each `Theta_b` and `Psi_b`
covariance graph, splitting structurally disconnected variables into separate
lower-triangular factors, and evaluate the estimator objective only at the
resulting block-diagonal collection of `L_c L_c'`. Thus a diagonal residual
covariance uses scalar square factors rather than a dense residual Cholesky factor;
correlated or fully free components retain the joint PSD restriction. Equality
constraints link every lifted within-component entry back to the original
affine-reduced partable parameter, so fixed cells, structural zeros, shared
labels, and linear/nonlinear equality constraints retain their ordinary
semantics and the returned parameter count is unchanged. Finalization projects
the affine-reduced coordinates back onto the terminal covariance links and
recomputes the objective there before exposing the ordinary partable-shaped
result. This prevents small feasible-link residuals from being amplified by
poorly scaled loadings into an implied covariance different from the one the
lifted optimizer evaluated. A correlated-residual/two-factor GLS boundary
regression gates this round trip. Zero fixed variances are removed from the
factor support, consistently forcing their covariance rows and columns to zero
without a degenerate Cholesky diagonal constraint. The
analytic objective and link Jacobians use the sparse rank-two derivative of
`L_c L_c'` and feed the same backend-neutral constrained scalar problem used
elsewhere: NLopt SLSQP is the required default and IPOPT is an optional
cross-check. Partable `<`/`>` rows remain unsupported, but they are not needed
to express the covariance cone. R exposes the explicit frontier entry point as
`frontier_fit_ml_psd()`. Equality-constrained scalar backends finalize with a
primal-feasibility plus Lagrangian-stationarity KKT audit rather than applying
the unconstrained objective-gradient test to a constrained solution. R exposes
the KKT residual, raw objective-gradient norm, equality violation, and
constraint-Jacobian rank under `fit$audit`. Boundary fits are reported by the
existing covariance diagnostics; ordinary interior information-matrix
inference is not yet promoted as valid at a rank-deficient component solution.

Complete-data `frontier::fit_ml_psd` also has opt-in diagonal preconditioning
(`PsdFitOptions::diagonal_preconditioning`, R `preconditioning = "diagonal"`).
It freezes per-observation expected-information scales at the PSD start. Ordinary
SEM sensitivities supply the equality-reduced original-parameter block; lifted
sensitivities supply the covariance-factor block. This avoids assigning infinite
scales to original covariance coordinates whose lifted-objective columns vanish.
Zero-sensitivity coordinates retain unit scale; nonzero scales are capped at
`[1e-4, 1e4]`. The caps affect coordinates only, not covariance eigenvalues or the
admissible domain. Objective gradients and constraint Jacobian columns transform
together, and the fitted point is mapped back before original link feasibility,
partable round-trip, and cone-stationarity audits. The default remains unscaled;
other PSD estimator entry points reject this option rather than ignore it.

The engineering default decision from completed experiment _archive/sem-total-variance is to retain native
starts, NLopt SLSQP and no preconditioning for supported complete-data PSD ML.
The supplied identification and SEM constraints are retained; automatic chart
selection and boundary-targeted restarts are not adopted. The six-structure,
ten-setting comparison found increased median evaluations under diagonal scaling
in every setting. A known limitation remains: at a zero mediator disturbance,
equivalent path parameters can expose a descent direction unavailable to the
first-order check at the returned representation. Audit acceptance is therefore
not a local/global optimality certificate at singular points. Further optimization
policies are deferred to concrete downstream failures or performance needs.

The frontier multi-information penalty is the soft counterpart of that
covariance domain (`cpp/include/magmaan/estimate/frontier/multiinfo_penalty.hpp`;
`estimate::frontier::fit_ml_multiinfo`, `estimate::fiml::frontier::fit_fiml_multiinfo`;
R `frontier_fit_ml_multiinfo()` / `frontier_fit_fiml_multiinfo()`). It maximizes
`l(theta) + lambda * sum_b log det Corr(v_K)`, where `v = [eta; y]` is the complete
latent-plus-observed vector in RAM form and `K` keeps every variable whose
residual-variance cell is free or fixed nonzero. Fixed.x covariates stay in `K`,
because dropping them removes the barrier on every equation that regresses on
them. Phantom observed copies and zero-variance latents drop out. The penalty is
scale invariant, so marker and std.lv give the same implied `Sigma`, and it is
bounded above by zero. For recursive `B` it equals `sum log(1 - R2_i) + log det
Corr(S_KK)` with lavaan's `R2_i = 1 - S_ii / C_ii`, so it tends to minus infinity
exactly as `S_KK` loses rank. The penalized estimate is therefore a point on the
interior-point central path of PSD-ML at barrier parameter `lambda / N`.
Nonrecursive models are fitted and flagged (`penalty$recursive`). The fit returns
the unpenalized half-discrepancy as `fmin`, so chi-square and fit measures read
the ordinary criterion at the penalized estimate. Standard errors are ordinary
information SEs at that point. The default is `lambda = 0.25` (`eta = 1.25`).
Both wrappers default to the PORT optimizer: NLopt L-BFGS stalls in its line
search where the optimum hugs a face, and PORT's trust region does not
(experiment _archive/barrier-optimizer).
Experiment research/47 found that any `lambda > 0` removes all improper, boundary, and
failed fits, that `lambda = 0.25` matches or beats PSD-ML accuracy with near-PSD-ML
chi-square and Wald calibration at `N = 50`, and that the originally proposed
`lambda = 1` over-shrinks high-R2 equations and correlations near one. SNLLS,
LS/ordinal paths, and two-level models are out of scope.

The same machinery carries a second target, `PenaltyTarget::Determinacy` (R
`target = "determinacy"`): the latent-determinacy barrier `lambda * log det Q`,
with `Q = Var(eta_L | y)` standardized by `Var(eta_L)` over the genuine latents
`L`. Phantom ov.y / ov.x slots and zero-error single-indicator latents count as
observed; an error-free indicator of several latents is rejected at layout
time. It is the joint barrier with the observed margin conditioned out,
`log det Corr(eta_L, y) = log det Q + log det Corr(Sigma)`, and equals
`-2 [TC(eta_L) + I(eta_L; y)]`. It is exactly zero on models without genuine
latents, which cannot be improper where `Sigma` is positive definite. The
Cholesky factorization of `C_JJ` (`J` = observed plus `L`) is the domain check:
with `Sigma` positive definite it succeeds exactly when every residual
covariance block is positive definite (inertia identity
`n_-(Var(eta | y)) = n_-(Psi)`). The report gives per-latent
`log(1 - rho_j^2)` (factor-score determinacy), `log det Corr(V)`, `TC`, and
`I`, and the decomposition is exact for nonrecursive `B`. Vanishing latents are
not faces of this barrier, so in charts that represent negative latent
variances (marker, sphere) the domain check, not the barrier value, keeps
estimates proper. The default target stays `joint` until the paper experiments
(`papers/sem-barrier`) settle it.

The exploratory sphere reference study (`engineering/active/15-sphere-reference-fits`,
renamed from the local-convergence study on 2026-09-27) uses canonical, layered
and three random starts on 60 datasets from three two-factor designs at N=20/100.
It compares ML L-BFGS/PORT and PSD SLSQP with/without diagonal scaling, with
polish disabled. Endpoints are cross-checked without refitting in a
strongest-indicator marker chart using the existing Newton diagnostic. Full
sphere coverage, first-order/norm checks, PSD admissibility where applicable,
and an explicitly heuristic extent screen define candidate eligibility.
A sphere-native curvature assessment remains missing. Repeated best-observed
candidates occur on 46 ML and 59 PSD problems; nine ML problems have no screened
reference. One has a screened lavaan point below all the original sphere
attempts; both sphere backends recover it when started there. This confirms a
portfolio miss, not nonexistence. The study keeps original verdicts, chart
proximity, local accuracy and extremes separate; ordinary fits never define its
sphere reference. Historical results are retained with their classification
limitations stated. A study-local start-design follow-up recovers five of the
nine missing ML references and improves one existing reference. On 60 fresh
draws, canonical starts match the expanded best-observed reference on 37
problems; adding three signed spectral alternatives reaches 55. The tested
all-positive spectral alternative adds no coverage. A witness ablation shows
that an alternative loading direction can recover the point even with positive
initial latent variance; sign alone is insufficient. This motivates a smaller
purposeful reference portfolio, with general-model construction and independent
confirmation still open. Expansion to 120 draws from one-factor five-indicator,
correlated two-factor four-indicator, and three-factor chain models raises
best-observed coverage from 93 canonical to 118 with the full start portfolio.
The positive spectral control adds one reference, so dropping it universally
is not supported. Single-negative alternatives miss ten references reached by
combined sign patterns. The exhaustive construction remains study-local,
simple-structure, and capped at three factors; its exponential cost is not a
production policy. Two problems remain unresolved (one extent-only exclusion).
A direct ordinary-marker follow-up runs sample-only spectral starts on all 60
original and 60 retained fresh draws without sphere optimization or reference-
dependent selection. With information scaling on fresh ML draws, layered alone
matches 35 references under L-BFGS and 36 under PORT; adding four spectral
recipes reaches 49 and 53 (55 available references). For development PSD,
FABIN3-auto plus the positive spectral start reaches 57/60 with no preconditioning
and 56/60 with diagonal preconditioning, against 54/60 for FABIN3-auto alone.
All available reference targets are marker-representable at 1e-6; unresolved
cases are not classified as folds or poles. This remains a study-local portfolio.
The follow-up now includes five deterministic geometry witnesses: interior,
marker pole, zero residual PSD face, std.lv disturbance pole, and a proved
one-factor unrestricted-ML nonattainment covariance. Fixed second-indicator
marker and sphere routes recover the four finite targets for both ML and PSD.
Requested std.lv fits can pass local checks along a finite approximation to
an unattained chart target; convergence must not be reported as attainment.
No unrestricted-ML attempt passes the screen on the analytic closure witness;
all tested PSD routes agree on a checked positive-discrepancy boundary candidate.
A PSD-only reference run on the retained fresh 60 draws supplies 58 repeated
and two single-start references. FABIN3-auto plus positive spectral ordinary
fits match 51/60 under each scaling choice (47/60 FABIN3-auto alone); eight
remaining draws have inferior screened candidates and one has no screened
ordinary candidate. Boundary nominations use scaled primitive eigenvalues;
they remain separate from feasibility, local accuracy and chart translation.
No generic nonattainment classifier or production chart fallback is implemented.
Inspection of the nine missing ML references (four development, five fresh)
now compares nine marker placements, effect coding, positive std.lv and sphere,
with common transported spectral points and tighter endpoint continuations.
All nine yield finite locally checked candidates; two pass the original screen,
six have stable candidates excluded by the magnitude cutoff, and one remains
numerically delicate across charts. Two high-R2 cases have moderate invariant
primitive components despite large standardized ratios near zero total latent
variance. Positive std.lv excludes negative-disturbance solutions and is not
an equivalent chart for all unrestricted-ML sectors. Missing screened references
must not be equated with nonconvergence or nonattainment. Raw historical screens
are retained; component magnitude, accuracy and attainment evidence stay separate.
A translation-only inspection of the nine lowest-objective saved spherical
endpoints preserves their covariance/objective in marker coordinates. Better
markers reduce maximum loadings to about 1--1.24, but negative primitive
variances remain in every case. Eight endpoints cannot enter positive std.lv
because of negative latent/disturbance variances; the ninth can, but has negative
indicator residuals and unresolved local accuracy. Six of the nine spherical
endpoints pass local checks; three do not. The report retains their actual
parameter tables, separating successful translation from numerical acceptance.
The adopted requested-identification policy (2026-09-27) rejects a numerically
near-pole result rather than changing markers automatically. Fresh weak-marker
N=100 draw 4 is a required-failure witness: the saved chart level is 5.44e-6,
and its roughly 184,000 loading is rejected by an explicit study pole tolerance
of 1e-4 but still admitted by the current 1e-6. A regression check records this
gap without changing library tolerances. Another saved endpoint also fails
that study threshold and already fails accuracy; the other seven do not show
marker proximity at that threshold. This is a numerical rejection policy,
not an exact-pole or general nonattainment claim. Production validation of the
requested-chart gate remains open; automatic marker substitution is not planned.
The seven saved endpoints not rejected by that study chart check have now
been investigated independently with high-precision derivatives of the same
six-indicator ML objective, retaining the original x1/y1 markers. Five refine
directly to finite local minima. Fresh weak-marker N=20 draw 1 has a losing
sphere path toward var(X)=0 and diverging regression/disturbance, but the better
negative-X-variance point is a finite local minimum; both sphere backends
recover it from the witness. Development weak-marker N=100 draw 2 has a finite
profile minimum at loading 740.4555345, confirmed at 60 and 90 decimal digits,
with positive but extremely weak curvature. All seven finite witnesses reproduce
the library objective and pass the explicit study chart check. Six pass the
ordinary-precision accuracy check; the profile-resolved case remains
ill-conditioned (condition measure about 5e12), so it is still a numerical
failure for routine use. These are local numerical witnesses, not global or
exact-arithmetic proofs, and all retain negative primitive variance components.
No optimizer, estimator, inference or paper protocol changes.
A subsequent retrospective single-start check selects spectral signs from sample
correlations or ranks the four spectral starting discrepancies, without using
fitted endpoints in selection. On the retained 120 draws, small gains come with
losses, and every rule misses the inspected negative-X finite witness. The
strongest signed rule adds two ordinary-ML matches per backend on the later 60
but loses two previous matches; it has no net gain on the earlier 60. Starts
remain unchanged. These checks use the frozen reference screen, not a new
production acceptance policy; neither multistart nor automatic marker switching
is adopted.

The frontier sphere chart (`cpp/include/magmaan/estimate/frontier/gauge.hpp`,
`sphere.hpp`; design in `papers/global-gauge-sem/work/notes/`) changes only the
chart the optimizer walks in, not the estimator. `analyze_gauge` reads the
partable and, per (latent, block), decides whether the latent's scale can be
re-sliced by a unit-norm loading direction without changing the model. Each
row carries a scale weight (loading +1, latent variance -2, latent covariance
-1 per side, regression -1 on the outcome and +1 on the predictor, latent
intercept -1). A unit's admissible loadings form an affine set A. A marker,
effect coding or fixed loading ratio gives 0 not in A, and the sphere runs in
span(A). A std.lv latent gives a linear A plus one fixed variance, which is
released. Metric invariance ties group copies into one unit. Every other
restriction must be scale covariant: fixed nonzero values only on weight-0
rows, linear constraints only among equal weights, and only sign bounds on
scale-dependent rows. Anything else demotes the latent to its user chart, with
a stated reason, iterated to a fixpoint. No model is rejected.
`fit_ml_sphere`, `fit_gmm_sphere`, `fit_gls_sphere` and `fit_fiml_sphere`
drive the unit-norm direction through `optim::reparameterize(problem,
ParameterMap)`, the nonlinear sibling of the affine reduction, with the gauge
pin `rho (||beta||^2 - 1)^2`. `fit_ml_psd_sphere` instead hands the sphere to
`fit_ml_psd` as one smooth equality per unit. The metric is unit-free by
default (sum lambda^2 / s_jj = 1). The fitted point is translated to the
user's chart, and by default (`SphereOptions::polish`) the ordinary fit is
restarted from it, so the result is the ordinary estimate under the ordinary
convergence criteria whenever that estimate exists. The polish matters: without
it, stationarity checked in a near-pole user chart rejected 103 of 107
optimal sphere fits on the Ernst N = 10 draws, while the median polish moves
parameters by about 1e-10. A marker at a pole (|l(a)| below `pole_tol`),
or a fixed-variance latent whose sphere-chart variance falls below `pole_tol`
times its fixed value (the std.lv pole, for example an endogenous latent with a
vanishing residual variance), returns `user_chart = false` with the sphere
report. Experiment _archive/sphere-chart-sanity checks the R surface deterministically: 68
fits of 37 models reproduce the ordinary results and lavaan, exact population
moments are recovered in every identification that holds them and flagged in
every one that does not, and translations between identifications match the
direct fits. The driven start is canonical by default: FABIN on the gauge-free model,
with each unit identified by its most inter-correlated indicator. That model
is the same whichever identification the user wrote, so sphere solutions
agree across identifications to optimizer precision. Least-squares fits start
from the sphere ML solution, because without ML's log-determinant barrier the
symmetric sphere lets LS reach points where a factor collapses onto one
indicator (points the marker chart keeps at infinity). Explicit start values
select the user's start instead. The driven ML run uses `fit_ml`'s
sample-based coordinate scaling on the non-unit coordinates, so a model
without gauge units is driven like the ordinary fit. Every fit is still
single-start and local.
On the Ernst N = 10 and 20 draws, PSD-ML has several optima on different
boundary faces in 28 to 47% of draws. There, one canonical start misses the
best optimum in 6.5 to 16.5% of draws (the marker route in 11 to 24%). Global
optimality is a separate, later question. Evidence for promoting the sphere
judges convergence to a certified local optimum. `reidentify` translates any
fit between identifications of the same model (marker on any indicator, std.lv,
effect coding) and refuses when the target partable describes another model.
The `sphere_route` and `sphere_route_parity` test executables compile the
lavaan goldens with `MAGMAAN_TEST_SPHERE_ROUTE`, which routes the
`cpp/tests/test_fit.hpp` seam through the sphere. Every case matches lavaan.
R exposes `frontier_fit_sphere(model, data, estimator, groups, ..., psd)` for
ML, ULS, GLS, WLS and FIML (psd = TRUE for ML). It shares `fit_model()`'s model
and data preparation (`.magmaan_prepare_spec`), returns a finalized
`magmaan_fit` with `fit$gauge` (units, pass-through reasons, the sphere-chart
partable) and `fit$options$chart = "sphere"`. It signals a classed
`magmaan_user_chart_singular` condition carrying the sphere solution when the
user chart does not hold the point. `frontier_reidentify(fit, model)` wraps
`reidentify`. `case_rerun()` and `modification_indices_lrt()` refit sphere fits
through the sphere. Since 2026-09-27 every exported frontier fitter (PSD,
barrier, PSD fallback, LS/FIML/ML2S/ordinal PSD) and
`fit_twolevel()` return a finalized `magmaan_fit` carrying
`fit$options$route` (the fitter and its non-data arguments), as do
`fit_model(psd = TRUE)` fits. So `vcov()` and `residuals()` apply as for
ordinary ML, and the refit-based methods refit through the route: a PSD fit
stays PSD and a barrier fit keeps its penalty and weight. `case_rerun()`
refuses routes it cannot rebuild from down-dated moments (FIML, ML2S, ordinal,
two-level) instead of refitting plain ML, and the likelihood-ratio refits of a
frontier route are single-group. Out of
scope for now: two-level, SNLLS, ordinal, composites, partial invariance via
nested spheres, the PSD LS/FIML siblings, and Wald inference when the user
chart is singular.

Fit finalization supplies the authoritative common numerical verdict through
`estimate::fit_verdict(estimates)`, independently of optimizer termination or
the driven-coordinate `fit$audit`. It verifies the original half-discrepancy
objective and full-model gradient on a declared per-observation scale (two-level
ML uses an explicit `1/N` multiplier), then applies the model-Frobenius metric-dual
L2 stationarity criterion. Fit entry points explicitly select ambient
(equality/bound) or PSD-cone geometry; covariance admissibility remains a
separate requirement for ordinary fits. Status is passed, failed, or unchecked.
R's `fit$converged` projects this to TRUE, FALSE, or NA and exposes the component
verdict, objective values/tolerance, and existing geometric residuals.

The common audit is wired for continuous ML and fixed-weight GMM/LS,
FIML, their PSD counterparts, Fisher/IRLS, two-level ML, ML2S Stage 2, ordinary/profiled ordinal and mixed
ordinal LS, and CatML. Continuous SNLLS audits eliminated covariance/mean
coordinates, including all-linear closed-form fits. Profiled ordinal fits
reconstruct and audit the original threshold-inclusive objective. Extra callback
constraints and uncovered specialized paths remain explicitly unchecked.
Backend soft-exit candidate retention and research-consumer migration remain
open; see `project/design/terminal-audit.md` for the contract and rollout boundary.

The continuous R SNLLS primitives reject supplied bounds instead of silently
ignoring them; the unbounded profile does not enforce variance or PSD bounds.
The fast inner Cholesky solve uses a heuristic estimated-rcond screen plus a
scaled normal-residual check against the original design, falling back to
column-pivoted QR. The threshold carries no guaranteed digits-lost bound.
GP's residual callbacks explicitly supply Kaufman's approximate Jacobian:
its scalar gradient is exact locally at fixed rank with an accurate inner
solve, while PORT-NLS/Ceres use approximate Gauss–Newton curvature. Focused
gates compare scaled/full-rank/near-dependent/rank-deficient profiles with
SVD and compare the final common audit with the original LS gradient. The
shared GP classifier rejects nonlinear equality constraints for every caller.
All-ordinal SNLLS explicitly rejects released-scale delta models: fitted
response standardization breaks the affine covariance/threshold assumptions
of the existing delta profile. This boundary does not affect ordinary delta
or supported theta fits.

All-ordinal theta SNLLS specializes independently free, unbounded thresholds
when the model has no active equality constraints. It optimizes correlation
residuals, reconstructs raw thresholds using fitted response means/variances,
and retains the original full-model audit. ULS/DWLS use direct reconstruction;
WLS caches a QR square root of the Schur-complement weight. Fixed/shared
thresholds and constrained models retain the generic full-threshold GP path,
which also remains explicitly callable for comparisons. Theta covariance
parameters remain nonlinear. Mixed ordinal paths are unchanged.

The production lifted-objective compiler has a private derivative-probe seam
used only by regression tests. Central finite differences now gate the complete
lifted ML gradient and equality Jacobian for mean structures, shared covariance
coordinates, general linear and nonlinear equalities, and multi-group block
stacking. Deterministic exact-model fits separately gate shared off-diagonal
covariances, nonlinear equalities, independent multi-group blocks, and agreement
with ordinary interior mean-structure ML.

The covariance lift now exposes an estimator-neutral implied-moment/Jacobian
callback internally. `estimate::frontier::fit_gmm_psd` applies any caller-fixed
continuous moment weight to that callback: an empty weight is ULS, while a
fixed supplied weight covers WLS, ADF, and diagonal-WLS objectives over the
ordinary `[mean; vech(covariance)]` moment stack. The
`estimate::frontier::fit_gls_psd` specialization constructs the same
sample-based normal-theory GLS weight as ordinary `fit_gls` and then uses the
fixed-weight path. R exposes these explicitly as `frontier_fit_uls_psd()`,
`frontier_fit_gls_psd()`, and `frontier_fit_wls_psd()`. They deliberately use
the constrained scalar backends rather than Ceres or SNLLS: the exact partable
links are nonlinear in the Cholesky coordinates, so preserving the specialized
least-squares algorithms would require a separate formulation. Focused gates
cover the lifted fixed-weight gradient and link Jacobian by central finite
differences, ordinary/PSD agreement at interior ULS, GLS, and nonidentity-WLS
optima, a ULS negative-residual repair, and the three R result contracts.



`estimate::fiml::frontier::fit_fiml_psd` applies the same covariance lift to
raw-data FIML. It reuses `FIMLPack` unchanged, evaluates the existing
observed-pattern likelihood and analytic gradient on lifted moments, and keeps
the ordinary fixed-x missingness policy. Primitive `Theta`/`Psi` blocks stay
PSD by construction, while each pattern-specific observed covariance
`Sigma_oo` must be PD for its likelihood contribution to be finite. The raw-only
and retained-pack C++ overloads return ordinary partable coordinates; R exposes
`frontier_fit_fiml_psd()` and marks the result with
`covariance_policy = "psd"`. Focused gates cover lifted central differences,
all-observed reduction to PSD-NTML and ordinary FIML, an interior missing-data
fit, a missing-data negative-residual repair, and the R result contract.
Inference at a rank-deficient component solution is not part of this surface's
current validation contract.

`estimate::fiml::frontier::fit_ml2s_psd` completes the missing-data two-stage
composition without altering Stage 1. The saturated FIML/EM means,
covariances, and ACOV-derived weights remain ordinary estimated objects. The
NT Stage-2 member calls `fit_ml_psd`; ULS, DWLS, ADF, and fixed-`a` DLS build
their existing weight once and call `fit_gmm_psd`. The raw overload performs
the same deterministic saturated-EM build. R exposes
`frontier_fit_ml2s_psd()`, preserves `stage1` and `raw_data`, and deliberately
omits the ordinary ML2S covariance/test correction because boundary inference
has no automatic policy. Focused gates cover all five fixed Stage-2 policies,
interior reduction, unchanged Stage 1, and an improper-solution repair.



`estimate::frontier::fit_ordinal_psd` and `fit_mixed_ordinal_psd` now compose
the same lift with the existing all-ordinal and mixed continuous/ordinal
ULS/DWLS/WLS residual engines after their respective partable preparation.
The transformed-coordinate seam supplies both the lifted implied-moment
Jacobian and the ordinary threshold-coordinate Jacobian, so the existing
delta, theta, released-scale, threshold, continuous-mean/variance,
polyserial, and polychoric formulas remain the single implementation. Stage-1
thresholds, continuous moments, estimated association matrices, and fixed
DWLS/WLS weights are not projected or otherwise repaired; even an indefinite
polychoric or hybrid mixed association matrix remains a valid ULS moment
target. Only the fitted primitive `Theta`/`Psi` blocks are constrained to be
PSD. The returned vectors and R partables retain ordinary prepared
coordinates, and `frontier_fit_ordinal_psd()` plus
`frontier_fit_mixed_ordinal_psd()` label the covariance policy explicitly.
Focused gates cover central-difference gradients in delta and theta
coordinates, interior ULS/DWLS/WLS reduction to the ordinary estimators,
unchanged Stage-1 moments, ULS acceptance of indefinite Stage-1 association
matrices, and repair of an otherwise exact mixed fit with an improper
continuous residual variance.

The separate inverse/log-determinant categorical-ML domain is ordinal
association ML: the `estimate::frontier::fit_ml()` / `fit_ml_psd()` overloads
on `OrdinalStats` (see [shared fitting composition](#implemented-composition-2026-10-01),
which replaced the former `fit_catml` names). This is
limited-information cML: normal-theory ML applied to the Stage-1 polychoric
correlation matrix under a saturated threshold structure, not a
full-information ordinal likelihood. The implied covariance is standardized
to a correlation matrix with an analytic Jacobian before the ML value/gradient
is evaluated. Because the sample polychoric log determinant is part of the
criterion, a non-PD Stage-1 matrix is an explicit domain error and is never
silently projected. The PSD variant constrains fitted primitive `Theta`/`Psi`
blocks and retains the input polychorics and thresholds exactly. R selects it
through `fit_model(..., estimator = "ML")` on all-ordered data, and
`frontier_fit_ml_psd()` accepts prepared ordinal data. Derivative,
interior-reduction, domain-error, unchanged-Stage-1, and
R-schema gates cover this slice; broader boundary-geometry validation remains
deferred. Inference at fitted covariance boundaries remains out
of scope. PSD two-level ML and native FC-SEM are not part of the supported
covariance-honest extension.

Experiment engineering/active/13-psd-estimator-stress (`experiments/engineering/active/13-psd-estimator-stress/`) now supplies the common
cross-estimator validation harness. Its first smoke profile covers continuous
ML/ULS/GLS/fixed-WLS, FIML, all five ML2S Stage-2 policies,
ordinal and mixed delta/theta LS, CatML, a near-residual-boundary geometry, and
an expected non-PD CatML input rejection. Each returned criterion is recomputed
independently in R, input objects are fingerprinted before and after fitting,
ML2S shares one fingerprinted saturated-EM object, and ordinary/PSD pairs retain
comparable parameter, implied-moment, convergence, eigenvalue, and timing rows.
The current two-replication smoke produced 128 attempts: all 125 returned fits
passed their independent objective and geometric-projection checks, all 69
cone-stationary PSD fits were admissible,
all 28 ML2S Stage-1 checks were unchanged, and both CatML domain sentinels were
rejected as intended. These counts validate the harness only; stochastic rates,
larger structural geometries, conditioning, basin behavior, and scaling remain
the planned pilot work.

The first evidence-bearing Experiment engineering/active/13-psd-estimator-stress tranche now covers 23 sparse
continuous complete-data cells with 100 replications each: the one-factor
residual boundary and sample-size axes, correlated residual and latent
covariance blocks varied separately plus one joint-boundary cell, and a latent
regression disturbance boundary. Across 20,700 attempts, 20,406 returned and
all passed independent objective recomputation. All 11,500 covariance-honest
fits returned; 11,424 were driven/lifted-audit stationary and every one was
admissible. The
76 nonstationary covariance-honest returns were all GLS fits in correlated-
block boundary cells. All 2,353 eligible stationary/admissible interior pairs
passed the prespecified objective and fitted-moment agreement gates. The pilot
also exposed the terminal-link amplification defect described above before the
final replay.

The second continuous tranche expands that panel to 29 cells with a
misspecified two-factor model, two grouped anchor/boundary cells carrying a
shared residual variance and a general linear equality, and fixed-WLS weight
condition numbers `1, 1e4, 1e8`. Across 24,000 attempts, 23,506 returned and
all returned objectives recomputed correctly. The common audit classified
13,082 of 13,297 returned covariance-honest fits as cone-stationary, and every
one was admissible; all 893 cone-stationary misspecified fits retained nonzero
discrepancy, all 1,797 returned grouped fits
honored their equalities to at most `4.44e-16`, and all 3,253 eligible interior
pairs passed agreement. Conditioning is the first sharp computational limit:
ordinary fixed WLS returned in `100%, 3%, 0%` of the three cells, while PSD WLS
returned throughout but was cone-stationary in `100%, 100%, 6%` (versus
`100%, 100%, 8%` under the lifted audit). Across the full panel, 2,824 PSD
returns required the PSD normal cone to pass relative to the ambient audit; 43
passed the lifted audit but not the common cone audit, with no reverse cases.
All 23,506 returned fits had finite gradients and completed both normal
projections. The weights retained the requested spectra and full effective
ranks. The remaining Experiment engineering/active/13-psd-estimator-stress work is calibrated missing-data/categorical
stress and targeted multistart/corpus validation rather than more replication
of the continuous structural panel; those estimator families already have
smoke anchors.

The optional IPOPT backend has now been measured against the required SLSQP
backend in experiments _archive/psd-ml-timing and research/42. It agrees on the deterministic admissible
optima but is roughly 40 times slower backend-to-backend on the small interior
CFA panel, and it returned fewer usable fits in the paired \(N=10\) stress
cell. Most failures were `Invalid_Number_Detected`. This is consistent with a
known formulation/backend mismatch: the lifted primitive covariances remain
PSD along the IPOPT path, but the complete-data ML callback can still be
non-finite when the implied observed covariance or \(I-B\) transform becomes
singular. The current adapter does not retain callback-stage failure telemetry.
NLopt treats the resulting infinite objective as a rejected trial point;
IPOPT's smooth finite-callback contract does not. IPOPT therefore remains an
optional cross-check rather than a candidate default. Changing that conclusion
would require a principled finite-domain formulation, not merely looser
convergence tolerances.

The terminal audit establishes feasible KKT stationarity, not global
optimality. Experiment engineering/evidence/12-psd-ml-basin-audit therefore treats the smallest objective found across
a prespecified multistart portfolio as “best attained,” never as a proved
global maximum. In its 100-dataset random cores, the default PSD fit hit that
reference in 76%, 94%, and 100% at \(N=10,20,50\). Most misses were already
admissible KKT points: 22% and 5% at \(N=10,20\), respectively. Initial
competing-objective rates were 51%, 23%, and 6%; tight representative restarts
retained distinct clusters in 44, 20, and 3 datasets. All flagged
equal-objective/different-moment cases collapsed under the tighter solve, while
bidirectional fixed-\(\beta\) profiles showed at least two feasible-envelope
minima in 9 of 13 targeted persistent cases. Thus a successful one-start KKT
return is not a globality certificate in the tiny-\(N\) design. The audit
clusters objective values, original partable estimates, and implied moments
separately so Cholesky representation, observational equivalence, and genuinely
inferior stationary basins are not conflated. Boundary estimates are retained:
stabilization would change the estimator and remains outside this PSD-ML
validation track.

The advisory continuous-corpus audit in
`cpp/tests/checks/psd_ml_corpus/` re-estimates the checked-in textbook and paper
model/data summaries by ordinary NTML and PSD-ML. **Source fidelity
(2026-09-25):** the textbook corpus was source-audited and rebuilt; every case
now reproduces its book's own output or the author's own lavaan call (see the
[translation audit](../validation/textbook-translation-audit.md)). The original
97-case run characterized mistranslated inputs: six ordinary solutions were
inadmissible and all were repaired, but its four "inadmissible Little cases"
were translation artifacts. On the corrected 102-case fixtures, ordinary NTML
is inadmissible in two cases, both Geiser (`cfa_second_order`,
`growth_quadratic`). The evidence still supports ordinary lavaan-compatible NTML
plus the post-fit audit as the core default, with an explicit warm PSD refit as
the methods-development policy. It does not support silent automatic
replacement or ordinary boundary inference. The decision record is
`psd_ml_corpus_audit.tex`, whose case counts predate the correction. A compact
generated fixture at `cpp/tests/fixtures/psd_ml/corpus_geometries.json` promotes
four geometries to default regression coverage without running the complete
scan: same-fit covariance reallocation and a materially different constrained
optimum (two frozen synthetic specifications on Little's NegAFF data, not
Little's models), a negative structural disturbance, and a joint-indefinite
covariance whose individual variances are positive (both Geiser).

Experiment research/43 (`experiments/research/active/43-psd-ml-repair-risk/`) turns those four
geometries into a compact repair-anatomy panel and adds a controlled
near-boundary risk path. The Gaussian DGP fixes three unit-loading,
single-indicator residual variances at 0.2 and varies the eigenvalues of the
latent covariance as `(1, 0.4, lambda_min)`, so the observed covariance remains
uniformly positive definite even at the primitive boundary. In the
prespecified 4,000-dataset pilot, ordinary NTML was covariance-inadmissible in
1,540 datasets; all 4,000 ordinary-warm PSD refits were KKT-converged and
admissible. Conditional on an ordinary inadmissible estimate, PSD-ML reduced
mean relative error by 0.00462 for the primitive covariance and 0.00367 for the
implied observed covariance, and reduced mean population Gaussian KL loss by
0.0354. Every repaired dataset improved on all three losses in this transparent
design. The median warm-PSD/ordinary time ratio was 1.20. These are
model-specific risk results, not an inferential or universal dominance claim;
they support the explicit audit/refit policy and show why fitted boundary rank
must remain descriptive rather than being treated as automatic rank selection.

The paired small-sample convergence benchmark
`experiments/research/evidence/42-psd-ml-small-n-convergence/` uses the six-indicator SEM and
sample-size grid shared by De Jonckere--Rosseel and Ernst et al. In the
1,000-replication run, audit convergence at \(N=10\) was 54.1% for ordinary
L-BFGS NTML and 96.8% for direct PSD-ML. Against ordinary SLSQP on the same
datasets, PSD-ML still rescued 47.1% and harmed 0.4%, so the improvement is not
just a backend change. The qualification is substantive: 86.1% of the
\(N=10\) PSD fits lay on a covariance boundary, and weakly identified raw
structural coefficients retained extreme tails. PSD-ML therefore repairs
optimizer and admissibility failures but does not turn very-small-sample
boundary fits into regular estimates or justify ordinary interior inference.
The full 1,000-replication artifact was refreshed under the constrained-KKT
terminal audit on 2026-07-30. It reproduced the \(N=10\) 96.8% versus 54.1%
headline exactly. Every returned PSD fit had finite KKT telemetry and equality
violation at most \(1.46\times 10^{-9}\); the two clean solver returns rejected
by the audit were covariance-admissible and equality-feasible but had KKT
residuals 0.851 and 24.744. The result is therefore not an artifact of applying
an unconstrained gradient test to constrained solutions.

## Implemented Capabilities

- [Parser, lavaanify, and matrix representation](capabilities/parser.md).
- [Complete-data ML and inference](capabilities/complete_data_ml.md).
- [Continuous FIML](capabilities/fiml.md).
- [Two-level (multilevel) ML](capabilities/multilevel_ml.md).
- [Staged C++ facade](capabilities/cpp_facade.md).
- [Simulation primitives](capabilities/simulation.md).
- [Optimizer backends](capabilities/optimizers.md).
- [Least-squares estimators](capabilities/least_squares.md).
- [Ordinal and mixed categorical LS](capabilities/ordinal_and_mixed.md).
- [R bindings and public namespace transition](capabilities/r_bindings.md).
- [Local build workflow](capabilities/local_build.md).
- [Argument-minimality sweep](capabilities/argument_minimality.md).
- [Robust-test naming and compatibility](capabilities/robust_test_naming.md).
- [Testing and validation](capabilities/validation.md).

## Current Boundaries

- Complete-data and FIML support are fixture-backed only for the documented
  estimator/model slices.
- The complete-data ML parity layer includes the `lavaan::growth()` defaults
  for linear latent growth models: observed-variable intercepts are fixed to
  zero, latent growth-factor means are freely estimated, and the relevant
  growth-factor covariance is auto-added. The R boundary exposes this through
  `model_spec(..., model_type = "growth")`.
- FIML with missing observed exogenous variables under `fixed.x = TRUE`
  remains unsupported.
- FIML score tests and modification indices use observed-pattern gradients
  plus analytic observed information with relative efficient-rank checks;
  fixture parity is against lavaan's observed-information score output for the
  covered fixed-row and equality-release cases.
- Delta and theta parameterizations are supported for all-ordinal and mixed
  continuous/ordinal DWLS/WLS point estimates. Ordinal robust reporting,
  modification indices, score tests, and standardized reporting use the fitted
  parameterization; remaining lavaan parity is fixture-backed for the covered
  ordinal slices and smoke-tested where lavaan fixture coverage is not yet
  available.
- Categorical models with fixed observed covariates (`exo` rows) are explicitly
  unsupported: conditional moments (`conditional.x`) are not implemented.
  This extension is [banked](../backlog/speculative.md#conditional-categorical-moments-with-observed-covariates)
  pending a named consumer and a target/sampling/inference contract.
  C++ preparation rejects these models, including cached moment routes; R
  fit/data/augmentation helpers reject them before constructing marginal
  statistics or entering the fitter. Explicit joint random-x models
  (`fixed_x = FALSE`) retain their existing path. Regressions cover direct and
  precomputed-data routes, staged model preparation, PSD dispatch, and a
  lavaan-matched joint-model regression slope; they do not establish broader
  mixed-model parameter-table parity.
- WLS ordinal point estimates and standard chi-square are lavaan-backed.
  Robust WLS scaled-test reporting remains shape-only because lavaan rejects
  Satorra-Bentler-family `test=` requests with `estimator = "WLS"` for the
  current categorical fixtures; DWLS robust reporting remains lavaan-backed.
- Continuous LS robust ULS reporting is lavaan-backed for non-fixed.x cases;
  fixed.x robust LS still follows lavaan's conditional exogenous bookkeeping
  and is not part of the generic continuous adapter coverage.
- Mixed categorical NACOV/weight, fit, fit-measure, and robust-reporting parity
  remains looser than the all-ordinal path, but now includes a sparse listwise
  4-category boundary fixture. Pairwise polyserial ML, fixed-marginal
  polyserial DPD, pair-local full DPD polyserial diagnostics, and
  continuous-normal score/Gamma primitives are exposed for follow-on inference
  work. The SEM-facing mixed polyserial DPD path robustifies only
  continuous-ordinal associations; robust continuous marginals and
  continuous-continuous covariances remain outside that contract.
- Observed-pair ordinal table kernels are pair-level primitives only. Missing
  ordinal SEM estimation is limited to the explicit pairwise observed-data
  composite prototype; shared-threshold multivariate missing ordinal modeling
  remains unsupported and requires a separate design note before
  implementation.
- Browne residual ADF is complete-data only.
- Score-test EPC is raw, unstandardized EPC only; standardized EPC and absent
  row-generation helpers are not yet part of the public contract.
- Nonlinear *equality* constraints are supported for ML, complete-data LS, and
  FIML through NLopt SLSQP or the IPOPT backend (Jacobian-projected vcov/df),
  including in combination with linear equality constraints in the same model
  — but not for ordinal or the separable SNLLS path; those combinations fail
  explicitly.
- Inequality constraints (`<` / `>`) and active-bound inference remain
  unsupported: inequality-constrained estimation needs boundary
  (chi-bar-squared) asymptotics magmaan does not implement. They fail with an
  explicit early error rather than silently reporting ordinary χ²/SE theory.

## Design Invariants

- No global mutable state.
- No in-place mutation of shared public structures; entry points operate on
  values or local copies and return values/errors.
- No groups == one group. Single-group models use the same block/group shape
  as multi-group models.
- Unsupported statistical combinations fail with an explicit error rather than
  silently approximating nearby lavaan-compatible behavior.
- Lavaan-shaped partables are boundary formats for oracle comparison,
  interchange, and compatibility projection. Core work should prefer the model
  triple plus explicit derived structures.
- Extension points remain free function templates over structural interfaces,
  not virtual hot-path interfaces or `concept` constraints.
- Parser behavior follows `project/grammar/grammar.ebnf`; if parser code and the
  EBNF disagree, the parser is wrong.
- Function signatures should be argument-minimal. A computation should receive
  the smallest structure or primitive values it needs: no names for numeric
  fitters, no fitted object for degrees of freedom, no full fit list where a
  parameter vector or sample-size vector is enough. Convenience adapters may
  unpack larger objects at the boundary, but core APIs should not depend on
  those bundles.

## Planning Documents

Use this file to understand the current state and contracts before structural
changes. Remaining work lives in the backlog:

- [project/backlog/todo.md](../backlog/todo.md) — the active SEM/parser/estimation
  backlog (open work only; completed items are folded into this roadmap and the
  test ledger).
- [project/backlog/simulation.md](../backlog/simulation.md) — the `magmaan::sim`
  work queue and decision log.
- [project/backlog/speculative.md](../backlog/speculative.md) — may-never-build
  ideas kept findable, with the cheaper alternative and the build-if trigger.
- [project/backlog/newsom-corpus-failures.md](../backlog/newsom-corpus-failures.md)
  — open optimizer/evaluator-accuracy cases surfaced by the Newsom corpus.

Validation and design context:

- [project/validation/test_ledger.md](../validation/test_ledger.md) — the
  test-protection map by subsystem plus the regression notes for fixed
  cross-subsystem bugs.
- [project/validation/lavaan_tutorial_parity.md](../validation/lavaan_tutorial_parity.md)
  — section-by-section audit of magmaan against the lavaan tutorial.
- [project/design/documentation_proposal.md](../design/documentation_proposal.md)
  — the proposed Quarto manual split, documentation vocabulary, and first
  documentation milestones.
