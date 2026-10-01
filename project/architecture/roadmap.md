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
| **Primary classical** | Single-level normal-theory ML (NTML, exposed as `ML`), FIML, and all-ordinal DWLS | 0.0.1 priority for reliable, efficient estimation and complete inference workflows; mixed continuous/ordered completion follows in 0.0.2 |
| **Priority frontier** | PSD covariance constraints; multi-information barriers in the following release | PSD estimation and inference accompany primary classical workflows in 0.0.1; barrier-specific hardening and inference follow in 0.0.2, with estimator-specific validation |
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
to 0.0.2; all-ordinal workflows remain current. Noniterative development,
inference expansion and capability-review work are indefinitely postponed,
with reactivation requiring an explicit user scope decision. Retained APIs and
regression gates preserve their documented capabilities. The
[MI/release-score completion matrix](../backlog/todo.md#mi-and-release-score-completion-001)
records current coverage and remaining work across estimator and weight choices;
it schedules no new ordinary-user default or parked model-family expansion.

PSD and barrier are capabilities across estimator families. PSD changes the
covariance domain; the barrier changes the objective. Track those separately
from the estimator/data combination and the numerical algorithm. Fisher
scoring, IRLS and SNLLS receive priority through the workflows they improve.
A fitter's existence does not establish its inference coverage, and extending a
penalty or constraint requires validation for each applicable combination.

Priority, ordinary-user exposure, API stabilization and default adoption are
separate decisions. PSD already has ordinary-user exposure; barrier remains
lab-only pending its exposure and inference contract. Both retain their current
API status. Promotion of either to a default requires recorded evidence.

The [active backlog](../backlog/todo.md#capability-inventory) tracks
execution and the remaining capability inventory. The inventory will distinguish
implemented and validated, implemented with limited validation, planned,
unsupported, and inapplicable components; these tiers alone make no new
availability claims. Existing entry points and numerical defaults are unchanged.

Release direction adopted 2026-10-01: **0.0.1 focuses on ordinary and PSD
estimation and inference in supported primary workflows**. Prioritize shared
starts, units, constraints, convergence/admissibility, PSD finalization and
sampling-law validation. Explicitly reject unvalidated inference, including
singular PSD endpoints; fitting support alone does not establish inference.
**0.0.2 owns mixed continuous/ordered completion and barrier-specific hardening
and inference**: stricter starting
domains, near-face curvature, factor disappearance, fallback units, marker
poles and penalty-specific sampling/exposure contracts. Existing barrier
implementation and regression gates remain; barrier-only work is not a 0.0.1
release requirement. Shared fixes needed by ordinary/PSD fitting stay in 0.0.1
even when they also improve barriers. Existing mixed APIs retain correctness
gates; mixed-only completion is a 0.0.2 task. Noniterative development and
capability reviews are indefinitely postponed pending an explicit user decision.

The active queue is organized around ordinary/PSD fitting reliability,
primary inference workflows, API/performance, validation/maintenance and the
0.0.2 mixed/barrier programme. Tasks
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
estimator identity. Shared ordinary/PSD composition serves the 0.0.1 focus;
remaining barrier-specific work follows in 0.0.2. Existing frontier status,
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
inference remain 0.0.2 work.

The [active composition queue](../backlog/todo.md#shared-fitting-composition)
owns dependency order and remaining gates. Foundation work in progress is not
reported here as completed support; its owner must record validation before
closing the task. Further pruning requires individual consumer/dependency
decisions and preserves shared primitives; parking SAM/FC-SEM expansion does
not authorize deleting their existing surfaces.

## Current State

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

The advanced per-call fitting setup (2026-10-01) separates start construction,
search and acceptance in `estimate/configured_ml.hpp`. Both R packages expose
it through `options`; `newton` names the existing common verdict, while the
pinned `lavaan-0.7.2` preset selects native FABIN3/OLS starts, PORT controls and
start scaling, bounded/unbounded R dispatch, standardized/simple-start retries,
and raw-termination plus exactly bound-masked gradient acceptance. Component
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
later carry over without changes. Lavaan acceptance on magmaan's own PORT
search is judged with `lavaan_acceptance_gradient()` in lavaan's units; the
lavaan search reproduces the same measurement. Every lavaan component,
including starts alone, rejects equality constraints. The first gate is
ordinary complete continuous ML without equality constraints, with
zero/infinite bounds only. Other versions and unsupported sources, including
pairwise-moment input, error. Retry parity covers a standardized retry
exactly; ill-conditioned retries match starts, coordinates and verdicts but
not endpoints. Constrained, FIML and all-ordinal parity gates remain in the
active backlog; inference conventions and default fits are unchanged. PORT
status codes follow its source: 8 false convergence (reported as a noisy
objective), 9 evaluation limit and 10 iteration limit (both budget stops).

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

The separate inverse/log-determinant categorical-ML domain is now implemented
as `estimate::frontier::fit_catml` and `fit_catml_psd`. This is
limited-information cML: normal-theory ML applied to the Stage-1 polychoric
correlation matrix under a saturated threshold structure, not a
full-information ordinal likelihood. The implied covariance is standardized
to a correlation matrix with an analytic Jacobian before the ML value/gradient
is evaluated. Because the sample polychoric log determinant is part of the
criterion, a non-PD Stage-1 matrix is an explicit domain error and is never
silently projected. The PSD variant constrains fitted primitive `Theta`/`Psi`
blocks and retains the input polychorics and thresholds exactly. R exposes
`frontier_fit_catml_psd()` with explicit `covariance_policy` and Stage-1 policy
labels. Derivative, interior-reduction, domain-error, unchanged-Stage-1, and
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

### Parser, lavaanify, and matrix representation

- EQS model-section frontend (2026-10-01): `parse::EqsParser` and
  `api::model_from_eqs` lower explicit single-group continuous equations,
  variances and covariances into the existing model triple. Fixed/free values,
  starts, error ownership, default-zero covariances and explicit identification
  are preserved; EQS job/estimator settings are not imported. The lab exposes
  `eqs_model()`; EQS remains excluded from the ordinary-user simulation
  prerelease. Ordinary-user integration is deferred until the C++/lab
  language-extension and round-trip gates pass.
  The [EQS contract](../grammar/eqs.md) defines the subset, unsupported cases
  and evidence limits. Offline pinned-lavaan fixtures gate rows, starts, ML
  estimates, implied covariance, expected SEs, df and chi-square; R tests
  compare installed lavaan. There is no live EQS oracle.
  The [manual source inventory](../grammar/eqs_source_inventory.md) records
  the full documented SEM model-language target, source pages, model-triple
  adapter requirements and focused runtime checks for unresolved cases.
  The inventory does not promote additional parser or fitting capabilities.
  The [planned implementation sequence](../grammar/eqs.md#implementation-sequence-planned)
  orders C++ resolution, general-equation representation, shorthand, means/
  groups and restrictions, with lab exposure after each validated increment.
  The [backlog](../backlog/todo.md#eqs-language-extension) records the later
  `r-magmaan` task separately. This is planning, not additional implementation.
- Lavaan-style syntax parser with normative grammar in `project/grammar/`,
  including fixed numeric intercept shorthand (`x ~ 0`) and parenthesized
  modifier labels (`(label)*x`), signed numeric modifiers/starts, chained
  modifiers, multi-LHS regressions, tolerated repeated `+` separators, and
  RHS continuations after the operator newline.
- Checked-in lexer, parser, partable, matrix, and fit oracle fixtures.
- Single- and multi-group LISREL matrix representation.
- Reduced LISREL lowers measurement rows whose RHS observed variable is already
  promoted into `lv_ext_order` into the structural `Beta` matrix, while keeping
  the mechanically inserted `Lambda[observed, phantom] = 1` identity. This
  keeps latent-change-score and single-indicator state chains on the same
  state vector as their autoregressions.
- Fixed.x resolution, mean structures, marker/std.lv/effect-coding
  identification, lavaan-style single-indicator residual fixing, start hints,
  and linear equality constraints.
- **Scaling conventions are changes of coordinates, and are held to that —
  wherever the loadings are actually free to move.** The marker and `std_lv`
  parameterizations of the same model then reach the same optimum, hence the same
  df and chi-square; only the raw parameter count moves. Pinned for two-level
  (per-level application, npar unchanged), single-indicator latents (`std_lv` and
  `auto_fix_single` are orthogonal, npar unchanged), second-order and
  endogenous-latent models (fmin to 13+ digits), and multi-group metric
  invariance. **The exception is a model whose loadings are all user-fixed**:
  `std_lv` does not override a user fix, so it pins the latent variances without
  freeing anything in exchange, and the result is a strictly more restricted
  model. `growth(std_lv = TRUE)` is the case that bites — lavaan itself goes npar
  9 → 7, df 5 → 7, χ² 8.07 → 106.85 on `Demo.growth`, silently. magmaan matches
  that, so this is faithful behaviour rather than a bug, but "std_lv is just a
  reparameterization" is false there and the mean structure is never
  standardized by `std_lv` at all.
  The one place the coordinate-change property is not automatic is multi-group
  `group_equal = Loadings`:
  `apply_std_lv` fixes `lv ~~ lv` at 1.0 per group, which with Λ tied across
  groups also forbids group differences in factor variance, making the std.lv
  invariance model strictly more restrictive than its marker twin. Step 8a-bis in
  `spec/build` therefore releases the latent variances in groups 2..G when
  `Loadings ∈ group_equal` and `LvVariances ∉ group_equal`, matching lavaan
  (`lav_partable_flat.R`, upstream `fecaf6b7`, 2019-06-27 — behaviour stable from
  0.6-4 through 0.7-2). Net effect: df unchanged across conventions, raw npar
  higher by (G−1)·n_lv under `std_lv`, the surplus absorbed by the extra
  cross-group loading equalities. Gated both by the `fit_stdlv` multi-group
  golden fixture and self-consistently by "std.lv multi-group metric invariance agrees with marker scaling"
  (`cpp/tests/unit/constraints_test.cpp`), which fits one deliberately misspecified
  two-group model both ways and requires equal df/chi²/fmin — an oracle-free
  check, and the cheapest guard against any future scaling convention silently
  changing the fitted model rather than its coordinates.
  Numerical scaling remains a research question. The one-factor measurements in
  [project/design/parameterization-geometry.md](../design/parameterization-geometry.md)
  and experiment research/46 show marker sensitivity to a weak indicator; they do not
  establish a globally optimal chart. For endogenous latents, `std_lv` fixes
  disturbance variance, which can make coordinates extreme at high explained
  variance. Experiment _archive/sem-total-variance compares marker, disturbance-unit, and total-variance-unit
  coordinates across six recursive structures using an experiment-local R
  prototype, common starts, analytic derivatives, and the same strictly PD
  component domain. Native fitting behavior is unchanged. Boundary solutions,
  practical starts, feedback and shared/equality-constrained parameters remain
  outside the initial pilot; a fit gap across unequal admissible domains is not
  evidence of optimizer failure. Its Bollen follow-up adds diagonal/frozen full
  information scaling, a local chart selector, and existing native PSD references
  from two starts. It distinguishes interior audit failures from matching
  cone-stationary boundary candidates; no native parameterization policy changed.
- Linear equality constraints through affine reparameterization (θ = θ₀ + K·α)
  for the ML, GMM/GLS, and bounded ordinal LS paths; per-θ box bounds fold onto
  the reduced α for the pure-merge case.
- Nonlinear equality constraints (`a == b*c`, `b1 == (b2+b3)^2`) for the ML
  and complete-data LS paths: compiled to name-free expression trees by
  `resolve_lin_constraints`, enforced by NLopt SLSQP or the optional IPOPT
  interior-point backend, with the constrained vcov / df projected through the
  constraint Jacobian H(θ̂). They may be combined with linear equality
  constraints in the same model — the constrained optimizer then runs in the
  linear-constraint-reduced α-space. FIML has the same NLopt SLSQP / IPOPT
  nonlinear constraint support; ordinal and the separable (SNLLS) path reject
  them.
- The expression sub-language shared by `:=` defined parameters and `==`
  constraints supports `+ - * / ^`, unary `+ -`, and the unary functions
  `exp` / `log`; both the defined-parameter evaluator and the
  nonlinear-constraint evaluator evaluate them with forward-mode AD.
- Effect coding for loadings.

### Complete-data ML and inference

- Normal-theory ML fitting defaults to NLopt L-BFGS. Heuristic start values
  sign each free loading by its indicator's covariance with the factor's
  marker, and terminal audit records projected-gradient stationarity at the
  returned iterate so soft optimizer failures can be classified by geometry.
- Frontier profile-LR scalar infrastructure for complete-data ML and
  moment-quadratic fits:
  `estimate::frontier::fit_ml_constrained()` appends caller-supplied nonlinear
  equality closures `h(θ)` / `J_h(θ)` to any partable nonlinear `==` rows and
  runs the existing SLSQP/IPOPT constrained scalar optimizer. The first helper
  layer, `profile_lrt_scalar_ml()` / `profile_lrt_parameter_ml()`, refits
  `g(θ)=g0` and reports the ordinary df-1 LR statistic
  `2N(fmin_constrained - fmin_unrestricted)`. The moment-quadratic sibling,
  `fit_gmm_constrained()` plus `profile_lrt_scalar_gmm()` /
  `profile_lrt_parameter_gmm()`, applies the same programmatic constraint path
  to a caller-fixed ULS/GLS/WLS/DWLS/DLS weight. The first CI-inversion seed,
  `profile_lrt_ci_parameter_{ml,gmm}()`, bisects the df-1
  profile statistic and returns root diagnostics. The complete-data ML and
  caller-fixed GMM parameter helpers also accept raw data for opt-in profile
  reference tiers: `RobustScaled` keeps ordinary `T`/`p_value` and adds the
  Satorra-style `scaling_factor`, `T_scaled`, and `p_value_scaled`, while
  `MisspecScaled` / `MisspecMixture` add the observed-bread misspecification
  fields (`misspec_scaling_factor`, `T_misspec_scaled`, mixture eigenvalues,
  mixture p-value, and endpoint cutoffs for CI inversion). ML routes the
  misspecification reference through observed Hessian bread plus empirical
  score meat; caller-fixed GMM routes it through the continuous-LS observed
  bread plus fixed-weight or IJ estimated-weight meat, including the supported
  sample-normal-theory, empirical-WLS, empirical-DWLS, and fixed-`a` DLS IJ
  weight families. R exposes the parameter special cases as
  `frontier_profile_lrt_parameter_*` and
  `frontier_profile_lrt_ci_parameter_*`, with `reference=` selecting the
  target. The all-ordinal frontier companion,
  `fit_ordinal_constrained()` plus `profile_lrt_scalar_ordinal()` /
  `profile_lrt_parameter_ordinal()`, applies the same programmatic equality
  pattern to the full ordinal ULS/DWLS/WLS threshold + polychoric moment stack;
  R exposes the parameter helper as
  `frontier_profile_lrt_parameter_ordinal()`, and
  `profile_lrt_ci_parameter_ordinal()` / the matching R helper invert that
  ordinary df-1 statistic by bisection. The first no-integration ordinal
  functional slice, `profile_lrt_ordinal_polychoric_omega()` /
  `profile_lrt_ci_ordinal_polychoric_omega()` and the matching R helpers,
  profiles model-implied latent-response/polychoric omega through the same
  fixed-weight path. All ordinal parameter and polychoric-omega profile helpers
  support the same ordinary, robust-scaled, misspec-scaled, and misspec-mixture
  references. The robust/misspec scaling is recomputed at each constrained
  profile point from the complete ordinal IJ sandwich used by robust SE/MI,
  including fitted DWLS/WLS weight influence; misspec references combine that
  meat matrix with the analytic observed ordinal bread. Bartlett /
  small-sample correction, observed-score / Green-Yang-like ordinal omega
  targets, and functional R callbacks remain research layers above this seed
  surface.
  The same parameter-profile reference family also spans the missing-data and
  mixed-categorical estimators: `estimate::fiml::frontier::fit_fiml_constrained`
  plus `profile_lrt_parameter_fiml()` / `profile_lrt_ci_parameter_fiml()` refit
  direct FIML against `theta_k = theta0` using the retained `FIMLPack`;
  `profile_lrt_parameter_ml2s()` / `profile_lrt_ci_parameter_ml2s()` cover
  ML2S Stage-2 `nt`, `uls`, `dwls`, `adf`/`wls`, and fixed-`a` `dls`
  weights (`*_ml2s_nt` remains the NT compatibility wrapper); and
  `profile_lrt_parameter_mixed_ordinal()` /
  `profile_lrt_ci_parameter_mixed_ordinal()` apply the same df-1 inversion to
  mixed continuous/ordinal ULS/DWLS/WLS fits. These helpers support ordinary,
  robust-scaled, misspec-scaled, and misspec-mixture references. Direct FIML
  builds the scalar profile scale from observed-pattern score meat and observed
  FIML bread; ML2S uses the Stage-1 saturated-moment sandwich and switches to
  analytic observed Stage-2 bread for misspec references, with estimated-weight
  IJ meat for data-dependent non-NT Stage-2 weights when raw/FIML internals are
  available; mixed ordinal uses the same
  IJ meat as robust mixed-ordinal SE/MI, including DWLS/WLS fitted-weight
  influence, and analytic observed mixed bread for misspec references. R exposes
  these as
  `frontier_profile_lrt_{parameter,ci}_fiml`,
  `frontier_profile_lrt_{parameter,ci}_ml2s` /
  `frontier_profile_lrt_{parameter,ci}_ml2s_nt`, and
  `frontier_profile_lrt_{parameter,ci}_mixed_ordinal`, with `reference=`
  selecting the endpoint statistic used by CI inversion.

- Frontier non-iterative CFA inference (2026-07) turns closed-form CFA
  estimators into delta-method-inferable ones. `estimate::frontier::
  noniterative_cfa_theta` maps `sigma -> theta` and returns a complete
  `(Lambda, Phi, psi)`. The legacy `guttman` selector preserves the existing
  lavaan-like Spearman/incidence Guttman map; `guttman_aligned` is the
  promoted research selector, using the blockwise triad-GMM H diagonal and the
  aligned score reconstruction from the communality experiment. The Guttman
  regression also has an explicit composite-weight axis:
  `auto` resolves to `unit` for `guttman_lavaan` and to `standardized` for
  `guttman_aligned`; `unit` uses incidence weights,
  `standardized` uses `diag(S)^-1/2 Z` and is rebuilt from the live covariance
  in every map evaluation. Aligned maps read loadings from the own-composite regression
  `K_if = (HB)_if / Q_ff` (the least-squares simple-structure fit; 2026-09-29),
  not the multiple regression `HB Q^-1` that `guttman_lavaan` keeps, and report
  residual variances as the communality split `diag(S) - diag(H)`.
  `estimator_map_jacobian` is its `J = dtheta/dvech(S)`: configural Guttman
  maps use an analytic regular-interior derivative (including the
  correlation-standardization, triad-GMM communality, fixed-rank
  Moore-Penrose-weight, composite-weight, inverse, and marker-scaling chains)
  with central differences retained as a boundary/rank-change fallback and as
  the generality seam for future FABIN2/Bentler/JS/MIIV maps. The promoted
  `guttman_aligned` configural path batches the block-GMM communality
  Jacobian and downstream score-regression derivative over all covariance
  coordinates; the GMM communality Jacobian now works on block-active
  derivative columns, applies the pseudo-inverse derivative through its
  required `dW e` action instead of materializing dense `dW` columns, and uses a
  guarded full-column-rank row-weight pseudo-inverse fast path. The aligned map
  has an explicit sample-size-resolved communality
  admissibility policy: `raw` is the exact historical no-op default, `hard`
  clips correlation-scale h2 to `[margin, 1-margin]`, and `soft` uses a smooth
  box with `beta = beta0 * n^rate`. Value, directional, batched, and
  constrained-KKT Jacobian paths compose the same clamp derivative. C++ and R
  fit/inference surfaces carry the policy and per-block activation counts;
  `guttman_lavaan` remains untouched. The aligned `unit` and `standardized`
  paths also have opt-in score-covariance conditioning for
  `Q = B' H B`. With `D = diag(Q)` and `R = D^-1/2 Q D^-1/2`, the hard map
  adds the exact diagonal intensity needed to attain
  `delta_n = floor0 n^-rate`, while the soft map uses a smooth spectral
  minimum and softplus activation; both return `(Q + tD)/(1+t)`, preserve the
  score variances, and use the repaired matrix consistently in the score
  inverse and factor covariance. The batched analytic Jacobian caches one
  eigensystem per block; hard activation/tie boundaries fall back to central
  differences and soft repeated eigenvalues use the invariant spectral
  projector. Fit objects retain the conditioning configuration and per-block
  raw/repaired eigenvalues, normalized eigenvalues, intensity, floor violation,
  score variance, and marker diagnostics, so post-fit inference reconstructs
  the identical map. Conditioning remains `raw` by default, is rejected for
  legacy `guttman_lavaan` . The retired engineering/10
  screen calibrated hard and smooth score repairs jointly with the
  communality-clamp finalists; its 24-cell/300-rep run (2026-07-10) produced no
  survivor (findings kept in the guttman-inference paper's notes since
  2026-09-29):
  raw-H arms frequently hit the existing improper-communality-split guard, and
  even clamped-H arms could have non-positive score variances. Fixed-diagonal
  Q repair intentionally cannot cure that latter failure because it preserves
  diag(Q). A separate, opt-in point-estimation feasibility branch now repairs
  the normalized H proxy itself toward the identity while preserving diag(H),
  before score construction. It is restricted to aligned unit/standardized
  maps, records raw/repaired H spectra and intensity, and has no post-fit
  inference claim until its derivative and calibration work are complete. Its
  18-cell/100-rep stress feasibility screen is also a no-go: it restored 100%
  point-fit availability and PD Phi, but every tested H repair worsened loading
  RMSE, often sharply, because the required normalized-H shrinkage was large
  (median intensity roughly 6--12 with substantial upper tails). The soft and
  hard variants saturated to the same repair in this failure region. Keep this
  as documented research evidence, not a production or inference candidate.
  Restricted
  estimator-side maps differentiate the regular constrained communality KKT
  system and the loading projection, with central differences retained for
  rank-changing pseudo-inverse or singular projection boundary cases. The
  restricted path batches the constrained h2 KKT right-hand sides for
  `triad_ls`, `extended_triad_ls`, and `triad_wls`; the GMM branch uses the same
  active-column `dW e` derivative as configural. The grouped restricted path
  embeds the active block RHS in the stacked communality system, reuses one
  joint KKT factorization across all covariance columns, feeds every block's
  constrained `dH` diagonal into the batched score-regression derivative, and
  applies the loading-projection derivative in action form, avoiding per-column
  inverse-derivative matrices. When no residual-communality rows are active,
  restricted maps bypass the stacked KKT system and evaluate the selected H
  diagonal directly; this keeps the `extended_triad_ls` +
  `standardized` configural proxy on the fast path. The correlation-standardizing
  Jacobian scatters each `dR_ab` row through its three nonzero covariance
  coordinates rather than adding dense `p*` rows. `triad_wls_joint` remains
  analytic but direction-wise.
  `robust::frontier::noniterative_se*` is the SE-only primitive:
  normal-theory paths contract `J Gamma J'/N`, while empirical paths stream
  casewise moment rows in parameter space rather than materializing dense
  moment-space Gamma. `robust::frontier::noniterative_inference*` is the full
  GOF bundle: it builds the residual projector `M = I - Delta J` and reuses the
  existing weighted-chi2 spectrum reducer for ULS or model-implied NTML tests,
  then includes the same delta-method SEs. `noniterative_wald` and
  `noniterative_difference_test` are the nested tests. R surface:
  `magmaan_core$noniterative_cfa_{fit,se,inference,wald,difference,
  pseudo_lrt}_impl`.
  The R fits are now first-class `magmaan_fit` objects (classed
  `magmaan_noniterative_fit`), so the ordinary post-fit measures apply
  unchanged: `vcov()` (regime `model` -> NT Gamma, `robust` -> empirical Gamma),
  `standardized()`, `residuals()` / `lav_residuals()` (SRMR), `factor_scores()`,
  `composite_weights()`, `compute_defined()`, the general `parameter_table()`
  (est/se/z/p/CI), and `fit_measures()`. The last routes to
  `fit_measures_noniterative()`: the residual GOF supplies the user statistic
  under the NT (`ntml`) or ULS discrepancy, referred to a same-discrepancy
  independence baseline, giving naive and robust/mixture-scaled CFI/TLI/RMSEA
  plus SRMR (the ULS baseline and the baseline scaling `c_0` are closed-form; NT
  `c_0 = 1` exactly). Multi-group non-iterative fits route through the grouped
  residual inference path, empirical baseline scaling is fixture-checked against
  raw cross-product variances, and likelihood information criteria
  (`logl`/AIC/BIC/BIC2) are reported as `NA` because the closed-form map is not
  an ML estimator. The ML score / LRT machinery
  (`modification_indices`, `score_tests`, their robust/LRT variants,
  `case_rerun`, `nestedTest`) is guarded off with a message pointing to the
  residual-based diagnostics and nested tests, because a closed-form map is not
  a gradient-zero minimizer. The explicit
  `noniterative_cfa_modification_indices()` surface reports three diagnostics
  for each fixed-zero or absent one-parameter candidate: a raw residual score,
  a `V`-tangent-residualized residual score, and local one-step
  discrepancy-drop/EPC values. It is single-group covariance-only in v1 and
  intentionally does not reuse the ML score-test name. Glue in
  `r-package/R/noniterative_postfit.R` and `r-package/R/noniterative.R`;
  derivations in the
  paper notes `noniterative_fit_indices` and
  `noniterative_modification_indices`; validated by
  `r-package/examples/noniterative_postfit.R`.
  Theory in
  the guttman-inference paper's derivation notes
  (`papers/guttman-inference/dev/notes/noniterative_cfa_tests`); validated on the
  legacy map by the retired research/24 study, whose runner and findings moved
  to the guttman-inference paper on 2026-09-29 (empirical Gamma calibrated across
  normal / independent-component / ordinal-as-continuous generators and across a
  0.3-0.7 reliability sweep, NT Gamma asymptotically miscalibrated on non-normal
  data, GOF power near 1; Guttman's efficiency gap vs ML is small at high
  reliability and grows on the structural parameters as reliability drops and,
  under a heterogeneous loading pattern, lands on the weak indicator's
  factor-mates via the triad-based communality step; the closed form is more
  robust than ML to uniformly weak signal but less robust to a single weak
  indicator).
  The H-diagonal communality rules from the follow-up Guttman work are now a
  separate frontier primitive:
  `estimate::frontier::estimate_h_communalities()` / R `guttman_h()` compute
  AR, RS, triad least squares, anchor triad least squares
  (identity-weighted one-same-block anchor triads), blockwise triad-GMM, and
  full selected-triad-GMM diagonals for a fixed simple-structure indicator
  block vector. They return `h2`, `diag(H)`, and
  `H = S` with only the diagonal replaced. The ordinary `guttman_aligned`
  point-estimator lane consumes the blockwise triad-GMM rule. The
  residual-restricted Guttman map can also select any least-squares-form
  H-diagonal rule (`triad_ls`, `extended_triad_ls`, `triad_wls`, `triad_wls_joint`) and
  any composite weight (`unit`, `standardized`), reusing both
  choices in its restricted analytic-first Jacobian and grouped inference;
  AR/RS remain
  low-level H-estimation diagnostics because they are not constraint-compatible
  LS systems. The retired research/27 smoke study (findings in the
  guttman-inference paper's notes) ranked extended triad LS first.
- Frontier multi-group / constrained / mean-structure non-iterative CFA
  (2026-07) extends the closed-form estimator to measurement invariance. The map
  fits each group's Guttman block independently and stacks them, so
  `robust::frontier::noniterative_inference_grouped*` carry a block-diagonal
  delta-method `Omega`, a joint residual GOF, and `block_of_param`. Mean
  structure is supported for free intercepts with latent means fixed at 0
  (`nu_g = m_g` saturated); by Proposition 2 of the note the mean part is inert
  for fit, so only `Omega` gains the intercept block (via `gamma_nt_with_means`).
  The estimator-side metric map `fit_noniterative_cfa_metric` estimates a common
  standardized loading shape across groups by a Sigma/H-level rank-one
  reconstruction and then converts to the partable's marker chart. The
  estimator-side restricted map `fit_noniterative_cfa_restricted` imposes
  separable loading/residual linear constraints inside the Guttman
  reconstruction: residual rows enter the selected LS-form communality/H step
  (default `triad_wls`). Loading rows confined to one factor, and fixed
  non-marker loadings, are homogeneous linear restrictions on that factor's
  own-composite coefficients and are imposed exactly by a partable-only
  Euclidean projector before the marker rescaling (tau-equivalence averages,
  independent of the marker); rows spanning factors or groups fall back to a
  second-stage marker-chart projection over all loading rows. Unsupported
  mixed/factor/mean rows error. Grouped restricted
  inference uses the restricted map's full stacked analytic-first Jacobian
  with the same communality and composite choices, so cross-block constraints
  propagate into `Omega`, GOF, and pseudo-LRTs. General linear equality testing
  still has the
  `Omega`-metric minimum-distance projection `noniterative_constrained_fit`,
  whose statistic is an exact chi2_k (Wald = min-distance duality). True
  (free-latent-mean) scalar
  invariance is `noniterative_scalar_invariance`, the reference-group mean map
  `alpha_g = (Lr'Lr)^-1 Lr'(m_g - m_r)` with a linearized pseudo-inverse Wald on
  the mean residual orthogonal to the loadings, df `(G-1)(p-#factors)`. R surface:
  `fit_noniterative_cfa_{metric,restricted}` plus
  `magmaan_core$noniterative_cfa_{se,grouped_inference,pseudo_lrt,constrained,
  scalar}_impl`.
  Theory in the guttman-inference paper's constrained-CFA note
  (`constrained_noniterative_cfa`, removed from the paper's notes on 2026-09-29
  when invariance left its scope; recoverable from that repo's history); validated on the
  legacy map by the retired research/26 study, 300 reps, findings in the
  guttman-inference paper's notes (metric Wald tracks the ML LRT on
  normal data with matched power; on non-normal data the NT-Gamma metric Wald
  over-rejects and the empirical Gamma restores the level, mirroring the ML
  NT-vs-robust split; the scalar Wald is exactly nominal on normal data, far more
  robust to non-normality, and delivers true scalar in one closed-form step where
  the ML nested test cannot). The retired constraint-charts check (archived
  study, deleted 2026-09-29) showed for the estimator-side metric map that both the configural and
  metric-constrained implied covariances are marker-chart invariant at roundoff,
  including deliberately off-surface metric-violation cells; the raw theta
  coordinates differ, as they should.
- Frontier empirical reduced-bias estimation (2026-06) implements the
  Kosmidis-Lunardon trace adjustment for raw-data normal-theory SEM and the
  moment-quadratic family. The C++ surface covers
  `rbm_{explicit,implicit}_{ml,fiml,continuous_ls,ordinal,mixed_ordinal,two_stage}`;
  the R research surface is `magmaan_core$frontier_rbm(fit, raw_data, weight,
  stage2_weight, dls_a, method = "explicit" | "implicit")`. Complete-data ML
  uses literal per-row normal score rows, FIML uses observed-pattern score rows,
  continuous ULS/GLS/WLS uses the complete-data moment IJ rows, ordinal and
  mixed ordinal use the estimated polychoric/polyserial IJ rows, and ML2S uses
  the same saturated-moment IJ stack as weighted inference. The trace,
  correction, and implicit penalty are computed in the linear-constraint-reduced
  alpha space (`K' J K`, `K' E K`); full and reduced bread/meat matrices are
  reported for diagnostics. Nonlinear equality constraints are still rejected.
- Frontier structural-after-measurement (SAM / LSAM) estimation is available as
  `estimate::frontier::fit_sam()` and from R as
  `magmaan_core$frontier_sam()` / `sam()`. The C++ core fits local or global
  measurement blocks, builds the Croon/Wall-Amemiya latent covariance with
  ML/GLS/ULS mapping and Fuller lambda correction, then fits the promoted
  structural submodel. Lavaan `sam()` parity is covered for point estimates and
  classic `se = "standard"` / `se = "twostep"` standard errors from
  `SampleStats`. The raw-data overload adds lavaan-parity
  `se = "twostep.robust"` for the landed scope: single-group, local,
  covariance-only, complete continuous data with `alpha_correction = 0`; it is
  lavaan's nonnormality-robust Yuan-Chan correction, not the frontier
  misspecification / estimated-weight sandwich still tracked in the backlog.
- Experimental complete-data ML IRLS paths fit the same ML objective through
  outer Fisher reweighting and inner GLS solves:
  `estimate::fit_ml_irls()` uses the full parameter block, while
  `estimate::fit_ml_irls_snlls()` uses Golub-Pereyra profiling for separable
  models. Mean-structure IRLS adjusts each frozen inner covariance target by
  the current mean residual outer product so the inner score matches the ML
  score up to scale. Linear equality constraints are handled in reduced
  coordinates; nonlinear constraints are rejected, and the SNLLS variant also
  rejects box bounds.
- Experimental local Fisher scoring for complete-data ML is exposed as
  `estimate::fit_ml_fisher()` and `magmaan_core$fit_ml_fisher()`. It computes
  the analytic ML gradient and expected-information Hessian approximation on
  the true `F_ML` scale, solves a damped local Fisher equation, and accepts
  steps by Armijo backtracking on the ML objective. This is separate from the
  IRLS paths above, which reoptimize a frozen GLS subproblem at each outer
  iterate. A companion `estimate::fit_ml_fisher_snlls()` /
  `magmaan_core$fit_ml_fisher_snlls()` path solves the same local Fisher
  equation through a Schur complement over the SNLLS β/α split; this is local
  block elimination, not Golub-Pereyra objective profiling.
- Objective-scale convention (unified 2026-06-09): `est.fmin = ½·F` for EVERY
  estimator (ML/FIML, ULS/GLS/WLS, ordinal, mixed) — the optimiser's minimum,
  half the discrepancy, and the quantity whose Hessian is the Fisher
  information. The goodness-of-fit statistic is `T = 2N·fmin = N·F` uniformly
  (`inference::chi2_stat`), matching lavaan's stored `fmin` element-for-element.
  The `½` lives only in the optimiser adapters; the math kernels stay full-`F`
  so the information/SE and score paths are untouched. Deliberate exceptions
  (ULS-standard Browne, FIML-standard LRT, the test-side `(N−G)/N` lavaan
  offset) are documented at `chi2_stat` and in
  [project/design/numerical-conventions.md](../design/numerical-conventions.md).
- Expected information, finite-difference observed information, and analytic
  observed information for covariance and mean-structure models.
- Vcov/SE, Wald/z tests, chi-square/df helpers, LR/Satorra-2000 and
  Satorra-Bentler 2001/2010 compatibility nested tests, robust U-Gamma
  machinery, Satorra-Bentler-family statistics, robust SEs,
  FMG eigenvalue p-value tests (explicit method/options API, no parser),
  Browne residual NT/ADF, fixed-parameter modification indices,
  equality-release score tests, fit measures including RMSEA close-fit
  p-values and lavaan's saturated-user-model `TLI = 1` convention,
  lavaan-style robust/scaled fit-measure formula helpers for the core
  `chisq.scaled`/baseline/CFI/TLI/RMSEA family, the FIML corrected robust
  fit-measure reduction (`estimate::fiml::fiml_corrected_fit_measures`) that
  builds missing-data `XX3`/`df3`/`c.hat3` and baseline counterparts with
  lavaan's FIML-C(V3) trace correction, using the missing-data saturated H1
  information plus the complete-data H1 information at the EM moments, and ML2S
  `robust.two.stage` robust/scaled global indices
  (`estimate::fiml::two_stage_fit_measures`),
  structural-aware standardization, C++ defined-parameter evaluation, and the
  first frontier reliability covariance functionals
  (`measures::frontier::reliability`: alpha, Guttman's lambda6, and
  Spearman-Guttman covariance omega with delta-method covariance-scale SEs;
  exposed in R through `magmaan_core$measures_reliability_cov`). Extended with
  closed-form **multidimensional** omega: `omega_multidim` for
  `OmegaTarget::Total` (the weighted composite `w'CX(X'CX)^-1 X'C w / w'Sw`) and
  `OmegaTarget::Hierarchical` (two-stage centroid Schmid-Leiman general factor,
  k>=3), with Spearman ratio-of-sums communalities, a finite-difference gradient,
  and the full-Gamma `omega_multidim_delta` SE; exposed through
  `magmaan_core$measures_reliability_omega_multidim`. Consumed by
  `papers/closed-form-omega`. Model-based CFA omega is also present:
  `omega_from_fit` computes continuous single-group `omega_total` /
  `omega_hierarchical` from fitted LISREL matrices and reuses ML/GLS/ULS robust
  parameter vcovs for delta SEs; the all-ordinal frontier slice adds
  `ordinal_observed_score_covariance` / `omega_ordinal_observed` plus
  `estimate::frontier::ordinal_observed_omega` for a fitted single-group
  DWLS/WLS/ULS model. That ordinal path maps fitted thresholds and
  latent-response correlations to the observed integer category-score covariance
  (0,1,...,K-1), then contracts the finite-difference scalar gradient with the
  complete IJ parameter sandwich from `robust_ordinal_ij`. It intentionally has
  no Bartlett/profile-LR/small-sample scaling policy; those remain research
  layers above the delta surface. The R surface is
  `magmaan_core$measures_reliability_ordinal_observed_omega`. The simpler
  no-integration ordinal reliability target,
  `magmaan_core$measures_reliability_ordinal_polychoric_omega`, computes
  closed-form omega directly on the all-ordinal polychoric correlation matrix:
  it extracts the correlation block of the ordinal `NACOV`, pads it into the
  off-diagonal slots of a full vech-correlation Gamma with fixed unit diagonals,
  and reuses `omega_multidim_delta` for the robust delta SE.
- `inference::frontier` robust (generalized / Satorra-Bentler-scaled)
  modification indices and equality-release score tests: each candidate carries
  the ordinary `mi` and a `mi_scaled = mi / c` with the per-direction scaling
  `c = gᵀB1g / gᵀA1g`, where A1/B1 are the parameter-space sandwich bread/meat
  surfaced by `robust::param_space_sandwich` (the same Δ'WΔ / Δ'WΓ̂WΔ that
  `robust_se` uses) and g is the efficient-score direction. Goes beyond lavaan,
  which falls back to the ordinary statistic when `se != "standard"`. Covers
  complete-data ML, both breads (`Information::Expected` ≈ robust.sem/MLM;
  `Information::Observed` ≈ robust.huber.white/MLR), single or multi-group (the
  per-block `n_b/N`-weighted sandwich pools across groups); reduces to the
  ordinary statistic exactly under the model-implied Γ_NT meat (Expected bread).
  Friendly entries under `api::frontier::{modification_indices,score_tests}_robust`.
  The fixed-parameter robust MI sweep batches eligible candidate rows into one
  augmented null-point evaluation, so score/info and the parameter-space sandwich
  are built once per sweep rather than once per candidate; duplicate matrix-cell
  edge cases fall back to the conservative one-candidate path. The sandwich
  helper also accepts precomputed casewise contributions (`Zc`) for callers that
  reuse the same raw-data meat.
  One-dimensional NT and robust candidate tests judge efficient-information
  rank relative to the marginal and nuisance-removed information terms
  (`I_eff > 1e-10 max(|I_dd|, |I_da I_aa^-1 I_ad|)`), rather than an absolute
  information floor. The robust bread uses the cancellation bound
  `g' A1 g > 1e-12 |g|' |A1| |g|`. These homogeneous checks exclude
  identification-only releases without rejecting identified directions merely
  because their parameter units make information small. Observed-information
  workers first check candidate rank in expected-information tangent geometry:
  residual-gradient curvature on a nonlinear identification orbit at nearby
  parameter values must not create a hypothesis. This gate covers fixed and
  equality releases without changing the information used for their statistics.
  Gates cover the batched ML sweep across nearby fits, observation counts and
  units; the original FIML
  robust-MI normal-data witness now requires marker exclusion, and R MI tables
  are compared with live lavaan for complete/missing data across indicator units.
  Ordinary FIML MI/equality releases also use the existing analytic observed
  information, matching the robust path: a fixed finite-difference step could
  otherwise manufacture curvature in redundant directions. Their `h_step`
  argument remains validated for compatibility and no longer tunes information.
  Validated four ways (lavaan implements no robust score test to diff against):
  exact reduction-to-NT, independent A1/B1 re-assembly, an R-internals oracle
  built from lavaan's delta/wls.v/gamma/ceq.JAC (`regen_robust_score.R`,
  convention-free θ-space scaling), and an advisory calibration + Wald/LRT-trinity
  simulation (`cpp/tests/checks/robust_score/`).
- **RLS is Browne's statistic with a model-based Γ (2026-09-19):**
  `inference::rls_chi2()` claimed lavaan `browne.residual.nt.model` parity but
  computed `Σ_b n_b·½·tr((Σ̂_b⁻¹(S_b−Σ̂_b))²)` from moments alone. Reading
  `lav_test_browne.R`, lavaan's statistic is
  `N·(r'Γ⁻¹r − b'A⁻¹b)` with `b = Δ'Γ⁻¹r`, `A = Δ'Γ⁻¹Δ`, over a residual `r`
  that **includes the mean block** whenever the model has a mean structure, and
  `browne.residual.nt` vs `.model` differ *only* in whether Γ is built at `S` or
  at `Σ̂`. The trace form equals that projected quadratic only when `r` is
  already Γ-orthogonal to the model tangent space — true at the ML optimum for
  covariance-only or **saturated-mean** models, false once means are genuinely
  restricted, where it was wrong by 24-50%.
  - Fix: `browne_residual_nt` gained an `inference::GammaAt {Sample, Model}`
    parameter (the only behavioural difference), and `rls_chi2` is now
    `browne_residual_nt(…, GammaAt::Model)`. Mean structures are handled by the
    shared residual vector, so no separate mean-aware entry point exists:
    `frontier::rls_mean_cov_chi2` is retired.
  - The moments-only overload is **removed**, not fixed: the projection cannot
    be recovered without the Jacobian. What it actually computed — the
    unprojected `N·r'Γ(Σ̂)⁻¹r`, mean block included — survives under an honest
    name as `frontier::nt_moment_quadratic`, which is a legitimate primitive
    (the closed-form CFA `rls_check` and `noniterative_cfa_test` want exactly
    the trace form) but is **not** a lavaan test statistic and not χ²(df).
  - Why it went unnoticed: every fixture carrying an `rls_chi2` oracle value had
    saturated or absent means, where the two formulas agree exactly. The only
    restricted-mean fixture, `0023_scalar_invariance_3f_hs`, had *null* oracle
    values because `regen_oracle.R` used `tryCatch(…, warning = function(w)
    NULL)`, and that fixture emits the benign "a single label per parameter in a
    multiple group setting implies imposing equality constraints" warning — which
    is precisely what the fixture intends. One informational warning silently
    nulled 7 statistics for 4 of 19 fixtures (0013, 0015, 0018, 0023). Replaced
    by a shared `fit_or_error()` helper that retries under `suppressWarnings`
    and treats only errors as errors; the restored 0023 now gates the
    restricted-mean case, and RLS there is 187.339 vs the old 115.586.
- **Matched ML speed pilot (2026-09-20):** `benchmarks/r/timing.R` now supplies
  shared adaptive batching, balanced arm order, and raw batch records for R
  workloads. The experiment collection contains a two-case continuous-ML
  pilot with directly timed raw/prepared/post-fit boundaries and separate
  covariance, SB, pEBA-4, and adapter checks. The HS native pEBA-4 comparison
  remains rejected on relative tail agreement even at identical statistic and
  spectrum. Follow-up isolates lavaan's default absolute integration tolerance:
  a tighter tail integral and an independent high-precision Erlang-chain
  reference agree with magmaan. The timing rejection remains until an
  accuracy-matched reference route is timed; there is no statistical-convention
  discrepancy. Controlled cross-engine evaluator/backend attribution and public
  promotion remain open under `project/validation/benchmark_plan.md`.
- **Expected information via the whitened Jacobian (2026-09-18):**
  `inference::information_expected_per_case_blocks` and
  `expected_info_covariance_only` no longer materialize
  `T[k][b] = Σ_b⁻¹·unvech(J[:,k])` for every free parameter × block. Both terms
  of the expected information are the same bilinear form — the normal-theory
  inner product `⟨X,Y⟩_A = ½·tr(A⁻¹XA⁻¹Y)` — which the NormalTheory
  `estimate::gmm::BlockWeight` already carries in factored form, so with
  `Y_b = Fᵀ[dμ/dθ ; dvech(Σ)/dθ]_b` the per-case block is exactly `Y_bᵀ Y_b`
  with no residual scaling. One whitened Jacobian per block, built and
  discarded in turn, plus one BLAS-3 syrk, replacing an
  `n_free × n_blocks` array of p×p matrices and an
  `n_free²·n_blocks·p²` elementwise reduction. Multi-group models benefit most,
  since a parameter usually touches one group and the rest of that array was
  explicitly-stored zeros: at p=48 with 4 groups, 14.4x faster and 27.0 MB →
  3.45 MB (`benchmarks/expected_info_bench.cpp`, paired/rotated via `benchmarks/timing/timing.hpp`). Pinned against an independent
  explicit-trace reference in `cpp/tests/unit/expected_info_whitened_test.cpp`.
  This is the surviving half of the retired "share the p×p factor" backlog item;
  the CPU-sharing half was retired (ceiling 0.006% at p=48; see
  the continuous-weight capability below).
- **Persistent continuous NTML inference (2026-09-16):** the existing
  U-factor shared phase is now the owning `robust::NTMLGeometry`, exposed by
  `prepare_ntml_geometry` and consumed by expected/observed U-factor tails.
  `robust::frontier::{NTMLData,NTMLFit,NTMLHypothesis,NTMLQuadratic}` retain
  complete-data sample/pattern setup, casewise contributions, fitted moments,
  derivatives, weight factorizations, expected information/covariance and
  test-specific reductions. These mutable native caches are session-local and
  not thread-safe; prepared inputs must remain immutable.

  R `prepare_inference_data` provides a shared dataset to `prepare_inference`.
  `prepare_hypothesis` owns an exact nested pair. Parameter-key embedding
  expresses dropped or fixed H0 paths as affine restrictions in H1's slots;
  existing same-slot affine pairs retain their calculation path.
  `inference_quadratic` produces global score/ML GOF or nested
  score/exact H1-anchored Satorra–2000 LR without invoking a test wrapper;
  `inference_rows` returns its casewise rows (a score statistic is the squared
  norm of their column sums, and their crossproduct is the spectrum's reduced
  matrix), so moment-based calibrations can be composed outside the core.
  `inference_covariance` shares expected information and empirical contributions
  with Wald consumers. Existing compatible GOF (including unbiased spectra),
  expected-information, expected sandwich-SE and exact empirical streaming LR
  wrappers reuse these native snapshots; edited extracted fit lists invalidate
  their handles. Other inference conventions retain the existing paths.

  Casewise storage expands centered moment contributions once across models;
  automatic storage switches to tiled projection above a 64-MiB contribution
  budget. Global score and GOF share a projection. Score applies the fitted-mean
  linear/constant correction without reconstructing fourth-moment contributions.
  The large-N tiled path accumulates both reduced matrices in one pass instead
  of retaining N-by-df rows. The spectrum uses row space when N < df; SB-only
  calibration uses a trace and spectra are cached on demand. Distinct tiled
  hypotheses may require distinct raw-data passes. Explicit unbiased GOF
  retains the existing casewise correction, which may materialize contributions.
  Neither full empirical Gamma nor full U is a prerequisite of sharing.

  Validation: the optimized inference suite passes 325 cases / 39,332 assertions;
  focused reuse tests cover unequal group sizes, mean/covariance layouts,
  tiled/casewise parity, smaller row space and construction counts. R checks
  additionally cover global and nested score/LR parity, unbiased GOF, sandwich
  covariance/Wald, restricted means, repeated calls and ownership/invalidation.
  `inference_reuse` exposes construction counters and
  `benchmarks/inference_reuse.R` provides a bounded timing comparison.
  Shared geometry for FIML/ML2S and additional bread/nesting conventions remains
  on the backlog; the first-class score interfaces below remain available.

- **Reusable score primitives (2026-09-16):**
  `inference::frontier::{global_score_components,global_score_components_ml2s,
  nested_score_components}` now own construction independently of tests.
  `ScoreGeometryOptions` selects sensitivity and metric without any resampling
  option. `ScoreComponents` retains the observed total score, casewise rows,
  nuisance/test directions and geometry. NT-ML2S marks its rows as influence
  contributions and preserves the separately computed observed Stage-2 score.
  `project_scores` constructs an owning `ProjectedScore` with a retained metric
  Cholesky factor, quadratic and meat; rows are optionally retained for explicit
  multiplier resampling. `score_spectrum`, trace-only `score_mean_scale`,
  `score_sandwich` and `resample_scores` consume that object separately. PSD
  meat remains valid for mixture inference when sandwich inversion is unavailable.
  Legacy nested/global score-flip wrappers now consume the same construction
  and projection; their historical exact-mixture/sandwich diagnostics remain
  compatibility behavior, not prerequisites of the primitive path.
  The friendly `api::frontier::score_components` adapters accept global ML/FIML
  fits or an H1 model plus H0 fit for nested hypotheses.

  R exposes `prepare_inference`, `scores`, `score_components`,
  `score_components_from_matrices`, `project_scores`, `score_quadratic`,
  `score_spectrum`, `calibrate_quadratic`, `score_sandwich` and
  `resample_scores` as thin wrappers over native construction/calculation.
  Preparation snapshots the fitted model, parameters, raw data and FIML pack
  once. Locked environments own native objects; restored process-local handles
  require re-preparation. Bare likelihood scores use summed log-likelihood
  units; globally centered covariance rows are an explicit projection option,
  distinct from legacy within-pattern centering. `inference_information` and
  `parameter_covariance` expose reusable ML/FIML matrices; `wald_test` accepts
  an existing covariance. Tagged matrices reject a different snapshot.
  Existing `fmg_tests`, `fmg_nested` and `robust_nested_lrt` also accept these
  snapshots and reuse their native fit context. Extracted fit lists invalidate
  that cache when their structural/sample objects change. LR/GOF-specific
  geometry still follows its own method contracts; `quadratic_reference`
  retains its computed statistic/spectrum for repeated downstream calibration.

  Validation: the optimized inference suite passes 323 cases / 39,248
  assertions; score-focused API checks pass 4 cases / 90 assertions. The R
  `examples/scores.R` checks ML/FIML/ML2S reductions, nested scores without an
  H1 fit, reference reuse, PSD meat, supplied matrices, Wald composition and
  snapshot ownership/invalidation. A bounded 15-call benchmark at N=400 and
  p=20 measures the new component/projection/spectrum/SB+pEBA pipeline at
  about 8 ms (ML), 17 ms (FIML), and 21 ms (ML2S), compared with 15/23/29 ms
  through the legacy score-flip wrapper; preparation is separately timed.
  Recalibration from an existing spectrum takes about 0.5 ms. These laptop
  timings are advisory; see `benchmarks/score_primitives.R`. They do not
  replace the historical talk simulation runtime or imply identical
  finite-sample score/LR spectra.

- `inference::frontier::score_flip_test` adds Monte Carlo Rademacher calibration
  for affine nested complete-data ML and direct-FIML models. It derives the
  tested directions from the exact H1/H0 restriction map, evaluates individual
  Gaussian observed-data likelihood-score contributions at the H0 fit, and
  reports three references:
  basic score flips, Hemerik-Goeman-Finos nuisance-effective flips, and the
  De Santis-Goeman-Hemerik-Davenport-Finos flip-specifically standardized
  quadratic statistic. The effective direction
  `G = D - K(K'IK)^-1 K'ID` is orthogonal to H0's nuisance tangent `K`; each
  transformed statistic additionally recomputes the conditional variance
  induced by estimating that nuisance vector. Complete data use per-group
  expected-information blocks; FIML uses conditional Fisher-information blocks
  per `(group, observed-pattern)` stratum. Both compress the correction without
  storing an `n x n` hat matrix, and the all-observed FIML path is unit-gated to
  the complete-data statistics, p-values, and mixture spectrum. The observed
  identity is included in the Monte Carlo rank,
  `p_value` aliases the standardized p-value, and deterministic seeds, Monte
  Carlo SEs, asymptotic score comparators, nuisance-stationarity, and variance-
  conditioning diagnostics are returned. The result also measures mean/max
  flip-specific covariance displacement and high-resolution setup,
  basic/effective resampling, standardization, asymptotic-comparator, and total
  timings. `ScoreFlipOptions::calibration` can now request asymptotic-only,
  effective-only, effective-plus-standardized, or the full historical battery;
  skipped references do not draw signs or build their covariance geometry and
  are returned as unavailable. `api::frontier::score_flip_test` and the R
  `score_flip_test()` wrapper expose both the original fit-pair route and a
  model-plus-H0 route: the latter uses H1's tangent without fitting H1. The R
  `nested_score_test()` convenience selects the zero-resampling route and
  reports score pEBA4 as its primary p-value alongside SB, exact-mixture, and
  direct-sandwich diagnostics. `ScoreFlipOptions::sensitivity` additionally
  exposes an opt-in observed-information correction. It replaces the expected
  metric in the nuisance projection
  `G_A = D - K(K'AK)^-1K'AD` with the realized likelihood Hessian while
  retaining expected information as a stable common quadratic metric. This is
  the pseudo-true/MAR path when
  information equality fails; the historical expected-information
  construction remains the default comparator. Observed sensitivity is
  effective/asymptotic-only and deliberately forbids within-pattern centering,
  whose conditional means need not vanish under MAR. Direct FIML reuses the H0
  fit's raw data and cached missingness pack. The current contract rejects
  fixed-X rows, nonlinear/inequality constraints, boundary null fits, and
  estimators other than complete ML or direct FIML.
  The C++ frontier option additionally has a deterministic verification-only
  exact-enumeration path capped at n=20. Its unit gate enumerates all 4,096
  sign vectors at n=12, is seed-invariant, and returns zero Monte Carlo error;
  a separate independent dense per-case nuisance-adjustment oracle matches the
  production group-sufficient covariance formula over every sign vector in a
  two-group n=5 construction. The R surface remains Monte Carlo-only.
- `inference::frontier::global_score_flip_test` is the curved-model global-GOF
  extension for continuous ML/FIML, with
  `inference::frontier::global_score_flip_test_ml2s` providing the
  normal-theory two-stage counterpart. The ML/FIML path evaluates direct
  casewise saturated likelihood scores at the fitted model moments, builds the
  conditional Fisher metric for each observed-data pattern, and projects the
  scores off the local SEM moment tangent. Its opt-in observed-sensitivity
  variant instead uses the analytic realized saturated-moment Hessian at the
  restricted fitted moments for the tangent projection; a projection identity is unit-gated against the
  existing structural observed-H1 information. A separate opt-in global-score
  metric can now use that same observed H0 information for the score quadratic
  and generalized-eigenvalue bread, while the established default retains the
  conditional Fisher metric. A second observed-information diagnostic uses
  the realized saturated H1 Hessian for either the nuisance projection or the
  quadratic/spectrum metric. At a pseudo-null it has the same population target
  as observed H0 information, while evaluation at the saturated optimum keeps
  its finite-sample curvature positive definite. The R surface names this
  choice `observed-h1`; its score rows and observed statistic remain evaluated
  at H0, so it is an H1-geometry score test rather than an LR construction.
  The global result also exposes the summed df-dimensional projected score,
  quadratic metric, and raw OPG meat, allowing experiment-side checks of the
  direct sandwich without reconstructing the saturated tangent. Two explicitly
  diagnostic observed-H0 sensitivity variants shrink the realized Hessian
  toward expected Fisher information with weights based only on
  `tangent_rank / n` (light and square-root rules); both weights vanish under
  fixed-dimensional asymptotics and the realized weight is returned.
  The full-observed H0 variant fails closed when its
  tested-complement information is not positive definite and is unavailable
  for ML2S. The tested dimension is therefore
  the saturated mean/covariance dimension minus the numerical tangent rank; no saturated H1
  fit and no refit per multiplier draw is required. Complete data and an
  all-observed FIML mask are unit-gated to the same statistic and multiplier
  ranks, while a separate missing-pattern gate exercises stratum centering.
  The first production contract is effective/asymptotic calibration only,
  random-X, affine equality constraints, no active bounds, and a required mean
  structure when observations are missing. `api::frontier` and the R
  `global_score_flip_test()` wrapper expose the result together with saturated
  dimension, tangent-rank, singular-value, conditioning, stationarity, and
  runtime diagnostics. ML2S instead evaluates the observed Stage-2 saturated
  score at the fitted model and propagates the Stage-1 saturated-EM casewise
  moment influence through the same fixed fitted-model NT metric and tangent
  projection. It deliberately supports only the unregularized, fixed-NT
  Stage-2 estimator: estimated/non-NT weights would require their additional
  weight-influence channel. Complete-data ML2S is unit-gated to the ordinary ML
  observed statistic and multiplier rank/p-value, while its asymptotic
  spectrum is only first-order equivalent because it uses centered moment
  influence rather than raw likelihood-score rows. The R
  `global_score_flip_test()` wrapper dispatches `estimator = "ML2S"` to this
  route and rejects regularized or non-NT two-stage fits.
  Experiment research/44 (now `fiml-global-tests`, global lane)'s 20,000-fit representative-SEM null gate found mean cell
  rejection .058 across 40 normal/VM/IG complete/MCAR cells (range
  .018--.122; 33/40 in [.025,.075]). Twenty-five finite calls lost numerical
  tangent rank and were strongly rejection-prone; downstream experiments must
  therefore report the prespecified-rank denominator separately and fail closed
  on non-nominal geometry rather than treating a finite p-value alone as full
  test success.
  A 300-fit, 30-cell `n=120` triage of the full-observed metric across all five
  pilot models, normal/VM/IG generators, and complete/MAR data found only
  226/300 usable calls; all 74 failures were non-positive-definite projected
  observed information. Among usable calls, observed-metric score pEBA4
  rejected 58%, versus 2.7% for the established observed-sensitivity/Fisher-
  metric score pEBA4 on all 300 fits. Ten replications per cell are not a size
  study, but the magnitude and conditioning failures rule this construction
  out as a default; retain it only as a diagnostic comparator.
  Experiment research/32's 300-replication
  probe found the basic test extremely conservative, effective flips close to
  nominal, and standardization a small improvement concentrated at n=30; the
  hard t5 / threefold factor-variance cell rejected at 0.054, 0.077, and 0.050
  for n=30/50/100, versus 0.190, 0.163, and 0.107 for the mean-scaled
  Satorra-2000 difference test. The 2026-07-13 expansion held a 62-slot
  two-factor nuisance model fixed while crossing 1/4/8 restrictions, total
  N=60/100/200, 1:1/1:3 allocation, three group-information geometries, and
  normal/t5/skew distributions (162 cells, 200 attempts each). Across 32,243
  successful fits, standardization changed 70 decisions, all from effective
  rejection to acceptance. At N=60, effective/standardized rejection was
  0.060/0.059 (df=1), 0.069/0.064 (df=4), and 0.083/0.077 (df=8); by N=200 the
  differences were 0.0006--0.0017. A warm serial 499-flip call cost 2.3/5.4/8.7
  ms total at df=1/4/8, of which standardization cost 0.7/2.7/4.2 ms. This is
  frontier evidence, not a support claim. The follow-up power pass added 540
  sparse/dense alternative cells (150 attempts each) and the complete-data
  nested LR/FMG battery: unscaled, SB, mean/variance-adjusted, scaled-shifted,
  exact mixture, FMG SB/MV/SS/SF, EBA2/4/6, pEBA2/4/6, PALL, pOLS, and ALL.
  FMG SB/MV/SS matched their nested-test routes to machine precision. Raw
  nominal power was not rankable because overall null rejection ranged from
  0.063 for standardized flips through 0.110 for MV/ALL to 0.138 for SB.
  After method- and design-cell-specific empirical-null calibration,
  standardized/effective flips averaged about 0.186 power and MV, pEBA4, and
  ALL about 0.184; the apparent nominal FMG/SB advantage disappeared. Sparse
  violations were markedly easier than equal-Euclidean-norm dense violations,
  and adding null restriction directions reduced power at df=8. Under
  16-worker contention, median latency was 13 ms for two fits, 22 ms for the
  499-flip battery, 3 ms for the nested LR battery, and 8 ms for all 13 FMG
  transforms. The null and power sweeps took 147 and 396 wall-clock seconds.
  A later audit separated the score base from its spectrum transform and added
  the direct sandwich-studentized joint statistic
  `u'(G'B1G)^-1u ~ chi-square(df)` to `JointScoreTestResult` and
  `ScoreFlipTestResult`, including availability, minimum-meat-eigenvalue, and
  condition diagnostics. It reduces to the ordinary joint score under normal
  theory and to the existing robust scalar release at df=1. Experiment research/32 shows
  why it is a comparator rather than a small-sample default: main-grid rejection
  was 0.036 overall and 0.024 at df=8, with median df=8 meat condition numbers
  of 36/22/14 at N=60/100/200. Under the severe PL/VM copulas it rejected
  0.029/0.067 while mean cellwise condition numbers reached 251/96. Score-SB's
  near-nominal aggregate is therefore interpreted as trace-only spectral
  shrinkage and partial error cancellation, not correctness of the scaled
  chi-square law; pEBA4 was the most stable compromise between scalar collapse,
  noisy plug-in ALL, and unregularized meat inversion. A separate 32-cell
  dimension replay found direct-sandwich rejection between 0.038 and 0.061;
  with adequate sample ratios it was 0.050/0.046/0.043/0.058 at
  p=10/20/30/40 while median condition numbers improved from 45 to 22 as N grew.
  The derivation and iteration decisions live in experiment research/32's
  `notes/score-sb-audit.tex`.
  Experiment research/33 then replaced that broad synthetic grid with a focused,
  published weak-invariance design: exact Foldnes-Grønneberg-Moss Study 2/3
  loadings at p=5/20, G=2/8, n=400 per group; normal plus severe VM/IG/PL;
  null plus published power alternatives (32 cells, 200 replications, 199
  flips, no t5 or unbiased-Gamma variants). The covariance-equivalent marker
  parameterization keeps the paper's `(G-1)(p-1)` weak-invariance pair affine.
  All 6,400 fits/flips/nested batteries succeeded. Effective/standardized null
  rejection was 0.0553/0.0547 and differed on only 2 of 3,200 decisions; raw
  power was 0.2422/0.2419 and matched-null power 0.2131/0.2125. Thus restriction
  rank alone did not make standardization useful at n/group=400: median
  flip-covariance displacement was 0.00093 and its correlation with |delta p|
  only 0.09. Cost nevertheless exploded with rank. At df=4/19/28/133, median
  standardization time was 0.8/23/87/4,080 ms; the df=133 flip call took 7.35 s,
  of which score resampling was 1.94 s. The eight-worker run took 36.7 minutes
  with zero failures (severe PL calibration at p=20,G=8 added about six minutes
  per cell). In this homogeneous, adequately sized design the effective flip is
  the practical choice; standardization needs small-n/leverage or visible
  variance-displacement evidence, not merely many restrictions.
  The compact comparator battery reinforced the earlier caution about aggregate
  size: direct-sandwich score, score-SB, score-pEBA4, and nested ALL/MV rejected
  0.049/0.054/0.045 and 0.050/0.052 overall, but each moved across the VM/IG/PL
  families. Their matched-null powers lay between roughly 0.22 and 0.24, versus
  0.21 for the flips, at only 200 null draws per cell; this is illustrative
  evidence rather than a method ranking.
  The former research/34 fixed-rank frontier had no local substantive results
  and its unused harness has been retired. Its p=5 versus p=20 nuisance-dimension
  question at G=8, df=28 is now a consumer-gated hypothesis in
  `project/backlog/speculative.md#fixed-rank-flip-standardization`, with cheaper
  effective-flip/score alternatives and historical remote identifiers. It is
  not a queued simulation; maintained flip methods and dense-oracle gates remain.
  Experiment research/35 is the first concrete residual/RLS-flip GOF derivation, kept
  leaf-local pending calibration. For one complete covariance block it forms
  model-centred saturated covariance contributions and projects them through
  the expected-information residual U-factor; the all-plus quadratic equals
  the structured RLS GOF statistic (independently gated to `1.2e-7`). Random
  signs calibrate that df-dimensional residual sum. An optional empirical-meat
  whitening acts on the already projected residual scores and is deliberately
  reported unavailable when its covariance is rank deficient; it is not the
  nested test's flip-specific nuisance correction and adds no hidden ridge.
  The 16-cell normal/PL null-power probe (p=5/20, n=100/400, one omitted
  residual covariance, 100 replications, 199 flips) completed 1,600 batteries
  without failure in 36.4 seconds. Effective null rejection was plausible but
  noisy at df=5 (.01--.08 across four cells), and poor at df=170: .12/.13 under
  normal n=100/400 and .33/.08 under PL. Empirical standardization was
  impossible at df=170,n=100 and worsened n=400 rejection to .23/.11 while the
  median projected-meat condition number reached about 37/42,000 under
  normal/PL. Thus the algebraic GOF bridge survives, but broad core promotion
  does not: the next gate is a larger low/moderate-rank n/df calibration, not a
  regularized high-rank default.
  Former research/36 (now research/06's pilot lane) extends direct-FIML nested scores and probes
  the published FIML--FMG two-group, six-indicator configural-to-metric design
  (`df=5`) at group-1 n=50/100/200, 0/15/30% MCAR, normal/severe PL data, and
  null/loading-power truths (36 cells, 100 replications, 199 signs). Basic,
  nuisance-effective, and standardized flip null rejection averaged
  0.009/0.064/0.062; effective versus standardized differed on only 3 null
  decisions. The important split was normal versus PL (effective 0.038/0.090),
  not missingness. Pattern-specific covariance displacement increased with
  missingness, but standardization rose from about 0.8 ms complete to 5--7 ms
  at n1=100 with missingness and supplied no material calibration gain. Score
  pEBA4 rejected 0.038/0.060 under normal/PL, whereas nested-LR pEBA4 rejected
  0.035/0.185. Across 3,600 attempts there were 27 fit failures and 29 further
  nested-battery conditioning failures, but zero flip failures conditional on a
  successful fit; the four-worker run took 41.9 seconds. The subsequent screen
  below supersedes the proposed larger null gate. Its unique power control is
  retained, with no independent replication claim from overlapping seed bases.
  Former research/37 (now research/06's score-flips lane) completed a 240-cell atlas: normal and
  severe VM/IG/PL data, group-1 n=50/100/200/400 (group 2 at 70%), complete,
  15/30% MCAR, paper-style 30% MAR, stronger logistic 30% MAR, and loading-
  equality ranks 1/3/5. Its 500-attempt, 199-sign screen reuses one configural
  FIML fit across ranks and produced 120,000 rank-specific test attempts.
  Equal-cell rejection was 0.0128/0.0693/0.0678 for basic/effective/
  standardized flips, 0.0526/0.0496 for score SB/pEBA4, and 0.1135/0.1102/
  0.1004 for nested LR SB/pEBA4/exact mixture. Normal effective/standardized
  rejection was 0.0558/0.0549, while VM and PL were about 0.075/0.073 and
  0.076/0.074; small n and higher rank, not missingness percentage alone,
  marked the liberal frontier. Score pEBA4 put 97.9% of cells in 0.025--0.075.
  An independent predefined 11-cell confirmation (2,000 attempts, 999 signs)
  put effective/standardized flips at 0.0808/0.0778 and score pEBA4 at 0.0473;
  VM/PL n1=50, 30%-MCAR flips remained around 0.11--0.12. Standardization
  changed 0.145% of screen and 0.301% of confirmation decisions. Its median
  phase cost rose from 3.8 ms at 199 signs to 94.8 ms at 999 signs, versus
  12.1 ms for the effective sign sums in the latter run. Flip versus nested
  numerical availability was 99.49% versus 98.85% in the screen and 98.17%
  versus 96.10% in the hard-cell run. The 12-worker runs took 14.3 and 10.8
  minutes. This makes the effective flip a useful non-ad-hoc diagnostic, not a
  uniformly calibrated small-n default under severe nonnormality; at rank <=5,
  pattern standardization is not the missing correction. Score SB's 0.0526
  aggregate is retained only as empirical error cancellation: at ranks 3/5 its
  p-values differ from pEBA4 by about 0.0048/0.0063 while the score-spectrum
  coefficient of variation averages 0.57/0.72. These historical aggregates include
  nonnormal MAR, whose generating-model restrictions are not independently proved
  FIML pseudo-nulls. The consolidated report separates those stress cells from
  complete/MCAR and normal-MAR calibration. Further flip expansion is banked
  pending a named incomplete-data consumer or new rank/missingness regime.
- The same scaling in the moment metric for the LS estimator tiers (2026-06).
  Continuous ULS/GLS/WLS/DWLS: `inference::frontier`
  `{modification_indices,score_tests}_robust` overloads taking the
  `estimate::gmm::Weight`, with A1 = Σ_b (n_b/N)·Δ'WΔ and
  B1 = Σ_b (n_b/N)·Δ'WΓ̂WΔ built by
  `estimate::continuous_ls_param_space_sandwich` (expected bread only; Γ̂
  empirical from raw, caller-supplied per-block, or model-implied Γ_NT per
  `WeightMoments` — the GLS weight with the Γ_NT(S) meat collapses to c ≡ 1
  exactly). All-ordinal and mixed DWLS/WLS (plus all-ordinal ULS):
  `estimate::frontier`
  `{modification_indices,score_tests}_{ordinal,mixed_ordinal}_robust` over the
  [thresholds ; associations] moment metric, reusing the `robust_ordinal`
  block assembly (W = estimation weight, Γ̂ = `stats.NACOV`) through the shared
  `estimate::weighted_param_space_sandwich`; full WLS (W = NACOV⁻¹) reduces to
  the ordinary statistic exactly, and `mi`/`mi_scaled` keep the lavaan-matched
  row-type moment-scale convention (c carries no moment-scale factor). The
  per-direction worker is shared as
  `inference::frontier::score_for_direction_robust` (c is not W-scale
  invariant: A1/B1 must be on the same weight scale as score/info). The
  continuous ML/LS and ordinal/mixed-ordinal tiers are single- or multi-group
  (see the multi-group bullets). api/R wrappers deferred until a concrete
  consumer appears.
  Oracles:
  continuous DWLS (`se = "robust.sem"`, so lavaan's wls.v/gamma are the
  ADF NACOV — gaps ~1e-9) and all-ordinal WLSMV (polychoric NACOV — c gap
  ~6e-10) release-score fixtures 0007/0008, plus the two-group ordinal WLSMV
  release-score fixture 0012, in `regen_robust_score.R`; exact WLS/GLS
  reductions and a primitives re-assembly live in
  `cpp/tests/unit/score_robust_test.cpp`.
- Continuous-LS robust MI/release covariance dispatch (2026-10-01): the R
  wrappers honor `cov="empirical"` versus `cov="model_implied"` independently of
  the WLS estimator label, preserving explicit fitting W. Empirical covariance
  needs complete fitting observations; normal-theory covariance uses the selected
  structured/unstructured moments. A full empirical WLS weight reduces to the
  ordinary statistic only with the matching empirical Gamma. The C++ workers
  reject unimplemented Browne-unbiased covariance and non-empirical
  estimated-weight covariance, including the old R model-implied-to-empirical
  substitution. Caller-Gamma overloads remain core-only; broader weight
  provenance/adapters stay in the MI completion matrix. Gates cover raw versus
  supplied Gamma, means/unequal groups, diagonal/full weights, scaling transport,
  empirical versus NT controls, estimated-weight mode and explicit unsupported
  errors in `score_robust_test.cpp` and `test_wls_robust_covariance.R`.
- FIML (missing-data) robust MI and equality-release score tests, the MLR corner
  (2026-06): `inference::frontier::{modification_indices,score_tests}_fiml_robust`
  build the bread A1 = (N/2)·H (the analytic observed FIML information) and the
  meat B1 = ¼·scoresᵀscores from the casewise observed-pattern deviance
  gradients, reusing the FIML-FMG machinery via the shared
  `estimate::fiml::fiml_score_meat_bread` (also feeds `fiml_robust_mlr`). Because
  the FIML score is `-½·Σ_i scores_i`, the unscaled `mi` equals the non-robust
  FIML MI and the per-direction `c = gᵀB1g/gᵀA1g` is the Huber-White correction,
  → 1 under a correct normal model. Observed bread only (no expected-info FIML
  analogue), no H1 EM needed (two data passes per candidate, no EM), single- or
  multi-group via the same block-stacked Hessian / casewise-score layout used by
  `fiml_robust_mlr`. FIML has no batch augmentation, so this uses the one-by-one
  robust sweep.
  R binding completion (2026-10-01): `modification_indices_robust()` and
  `score_tests_robust()` dispatch direct FIML to these entries. Omitted
  bread/information select observed; explicit expected information, alternative
  covariance/moment recipes, supplied weights and estimated-weight mode are
  rejected. Retained raw data reuse the fitted missingness pack; explicit data
  (including masks/group blocks) rebuild their pack, without an H1 EM. Binding
  gates cover retained/rebuilt/explicit-data agreement, unequal groups, equality
  releases, marker exclusion versus identified fixed loadings, and indicator
  units in `test_fiml_robust_score.R` and `test-score-rank.R`.
  Oracle: FIML/MLR release-score fixture 0009 (`information.observed` /
  `lavScores`, θ-space assembly, c ≈ 2.16 on heavy-tailed + MCAR data) plus a
  non-robust-`mi` match and a c → 1 normal-data anchor in
  `cpp/tests/unit/score_robust_test.cpp`.
- Multi-group robust MI / score tests for the continuous ML and LS tiers
  (2026-06-13): the single-group guards in `inference::frontier` are removed; the
  per-block `n_b/N`-weighted sandwich (`robust::param_space_sandwich` /
  `estimate::weighted_param_space_sandwich`), per-group candidate enumeration
  (each absent statement is one candidate per group), and the full-θ-space
  nuisance projection all carry over unchanged, so no new statistics were needed.
  Validated by a two-group Γ_NT reduction (c = 1 exactly across unequal groups),
  a heavy-tailed empirical case, a GLS multi-group reduction, and the cross-group
  loading-invariance golden 0010 — the latter pins the `n_b/N` weighting that
  within-group reductions cannot (`A1 = lavInspect(fit,"information")` equals
  `Σ_b (n_b/N)·Δ_b'V_bΔ_b` exactly, c ≈ 1.24).
- Frontier fixed-misspecification profile-RMSEA / profile-LRT primitives for the
  continuous moment-quadratic tier (2026-06-22): the weighted-moment RMSEA
  helper forms the full moment-space profile Hessian
  `Q = W - W D B^{-1} D' W` from the same observed-Hessian bread used by the
  misspecification-robust SE path, computes the `QΓ` spectrum with
  `robust::compute_profile_contrast_spectrum`, and reports both the signed
  trace `tr(QΓ)` used by the RMSEA correction
  `sqrt(max(F - tr(QΓ)/N, 0) * G / df)` and the positive-spectrum summaries used
  by mixture-tail approximations (`bias_trace`, `spectrum_size`, negative count,
  and rank are carried separately). When the first-stage data metric is positive
  definite, the RMSEA result also carries the small profile pencil
  `ν_j = eig(B^{-1} Ã)` plus its predicted positive/negative/rank counts; dense
  `QΓ` eigensolve remains the source of actual mixture weights.
  With `G_hat = V_o^-1/2 W_p D`,
  `Q = V_o^(1/2)(I - G_hat B^-1 G_hat')V_o^(1/2)` and
  `A_tilde = D' W_p V_o^-1 W_p D`, the inner spectrum is `(m-p)` unit
  eigenvalues plus `1-nu_j`, where `nu_j = eig(B^-1 A_tilde)`. For full-rank
  Gamma, congruence preserves inertia: positive count is
  `(m-p) + #{nu_j < 1}`, negative count is `#{nu_j > 1}`, and rank is
  `(m-p) + #{nu_j != 1}`. Singular Gamma restricts these counts to its range.
  The counts are Gamma-free only conditional on fixed Q, which can move with
  moments/weights. Small residuals make integer rank floor-sensitive; signed
  trace and positive-tail trace remain distinct when the contrast is indefinite.
  `estimate::weighted_moment_profile_lrt` compares two such profile Hessians in
  a common first-stage moment space, uses the positive spectrum of
  `(Q_H0 - Q_H1)Γ` for mixture and adjusted tails, and reports nominal `df_diff`
  separately from the actual `spectrum_size`. The continuous wrappers
  `estimate::continuous_ls_profile_rmsea` and
  `estimate::continuous_ls_profile_lrt` supply
  the observed bread plus Γ from either caller blocks or complete raw data.
  `estimate::weighted_moment_profile_rmsea_two_metric` generalizes the same
  dense engine to two-metric profile Hessians
  `Q = V0 - W* D B^{-1} D' W*`. The complete-data ML adapter
  `estimate::ml_profile_rmsea` / `estimate::ml_profile_lrt` uses the
  sample/saturated normal-theory metric for `V0`, the fitted-implied metric for
  `W*`, and the observed ML Hessian scaled to the per-unit profile bread.
  Covariance-only models use the vech(S) moment; mean-structure models use the
  stacked `[mean; vech(S)]` moment and empirical Gamma from
  `data::empirical_gamma_with_means` when raw data is supplied. ML2S-NT reuses
  the same two-metric ML adapter through
  `estimate::fiml::two_stage_nt_profile_rmsea` /
  `two_stage_nt_profile_lrt`: saturated EM moments are treated as the Stage-2
  complete-data sample statistics, while Stage-1 uncertainty is supplied by
  `two_stage_gamma_from_acov(sm, false)` over the stacked `[mean; vech(cov)]`
  moment blocks; overloads accept either precomputed `SaturatedMoments`, raw
  data, or raw data plus a precomputed `FIMLPack`/`FIMLH1`. Raw-data FIML is
  wired through `estimate::fiml::fiml_profile_rmsea` /
  `fiml_profile_lrt`: it uses the EM saturated metric `H/n_b` for `V0`, the
  model-implied observed-pattern H1 metric for `W*`, the observed FIML
  information scaled per observation as the bread, and the caller's FIML LRT
  chi-square as `N*fmin`; the same overload pattern reuses precomputed
  `FIMLPack`/`FIMLH1` and `SaturatedMoments`. These are basic dense research
  surfaces, not lavaan-parity fit-measure dispatch.
  `estimate::weighted_moment_profile_rmsea_estimated_weight` adds the
  diagonal-weight (categorical DWLS) case where the weight `W=diag(1/γ)` is
  itself a first-stage quantity: it assembles the value-function Hessian over the
  *extended* moment vector `x=(u,γ)`,
  `Q = [[W,R],[R,S]] - [[WD],[RD]] B^{-1} [D'W,D'R]` with `R=diag(r/γ²)`,
  `S=diag(r²/γ³)`, via the two-metric engine (extended `jacobian=[D;D]`,
  `V0=[[W,R],[R,S]]`, `W*=blkdiag(W,R)`, joint NACOV `Γ_x`) and restates `df` to
  the classical u-moment count. The residual-driven γ channel is dormant at
  exact fit (`Q` collapses to `W − W D B^{-1} D' W`) and reshapes the reference
  law under fixed misspecification. The all-ordinal estimator wiring that
  *produces* `(D, γ, r, Γ_x)` is `estimate::ordinal_dwls_profile_rmsea` /
  `ordinal_dwls_profile_lrt`: it pulls `(D, γ, r, B)` from the ordinal DWLS fit
  and builds the joint NACOV `Γ_x` of `(u, γ)` from stacked per-case influence
  rows `[g_i | IF_i(γ)]`, reusing the same `ordinal_gamma_diag_*` influence
  channels as `robust_ordinal_ij`; its `chisq_standard`/`df` match
  `robust_ordinal`. Mixed continuous/ordinal DWLS is wired through
  `estimate::mixed_ordinal_dwls_profile_rmsea` /
  `mixed_ordinal_dwls_profile_lrt`, which assemble the same `(D, γ, r, Γ_x)`
  block from `MixedOrdinalStats`, `mixed_moment_jacobian`,
  `mixed_observed_bread_analytic`, and the mixed `mixed_gamma_diag_*`
  influence/Jacobian channels (including observed/missing variants), so the
  mixed path also shares the `robust_mixed_ordinal` standard χ² and df. The R
  package exposes these research surfaces through
  `magmaan_core$ordinal_profile_rmsea` / `ordinal_profile_lrt` and
  `magmaan_core$mixed_ordinal_profile_rmsea` / `mixed_ordinal_profile_lrt`,
  taking the same explicit ordinal or mixed stats object used for fitting.
  The absolute-fit companion is `estimate::ordinal_crmr_misspec_inference`
  (`OrdinalCrmrInference`; `estimate::ordinal_crmr` point +
  `OrdinalFitMeasures.crmr`): a criterion-at-estimator sandwich
  `Q_G = Dφᵀ V0 Dφ` with `V0` the correlation-selector and `Dφ` the full extended
  `(u,γ)` residual jacobian, reusing the catml projector and the profile `Γ_x`. It
  returns a bias-corrected CRMR/SRMR point, an exact-fit mixture p-value, and a
  CI that propagates the estimated weight (normal-theory `g_Gᵀ Γ_x g_G` under
  misspecification, `weighted_chisq` mixture at the null); an `estimated_weight`
  flag gives the fixed-weight comparator. Single-group only so far. Empirically
  the γ channel is only ~2–3% of the CRMR variance — CRMR's fixed metric makes it
  largely robust to weight estimation, unlike the metric-dominating RMSEA / nested
  test. Verified by `cpp/tests/checks/ordinal_crmr_inference` (bias/variance/coverage
  vs Monte-Carlo). R bindings and lavaan `crmr` parity are deferred.
  The large-γ absolute-fit case is `estimate::ordinal_rmsea_misspec_inference`
  (`OrdinalRmseaInference`): RMSEA's criterion is the discrepancy `F = rᵀWr`
  itself, so the envelope theorem gives the gradient as the bare profile score
  `g_F = (−2Wr, −r²/γ²)` (no projector); it returns the bias-corrected RMSEA, an
  exact-fit mixture p-value, and a normal-theory CI on `F₀` with
  `Var(N·F)=N·g_Fᵀ Γ_x g_F`, reusing the profile `Q`/`Γ_x`/bias/spectrum.
  Empirically the γ channel is large and variance-reducing (`r` and `γ` co-vary
  negatively): the estimated-weight CI is calibrated while the fixed-weight one
  is conservative (over-covers, increasingly with misspecification) — accounting
  for the estimated weight tightens RMSEA's interval. Verified by
  `cpp/tests/checks/ordinal_rmsea_inference`. Single-group; R bindings deferred.
  The first *incremental* (two-model) case is
  `estimate::ordinal_cfi_tli_misspec_inference` (`OrdinalIncrementalFitInference`):
  misspecification-robust CFI and TLI with CIs. The user model and the analytic
  independence baseline (linear in its thresholds, so `Q_b` is exact) run through
  the *same* profile primitive with the *shared* `Γ_x`, so the joint law of
  `(T_u,T_b)` is one bilinear form `Cov(T_u,T_b)=N gᵤᵀΓ_x g_b`; CFI `=1−δ_u/δ_b`
  and TLI `=1−(Q̄_b/Q̄_u)δ_u/δ_b` (noncentralities `δ=T−Q̄`, generalized df
  `Q̄=tr(QΓ_x)`) are a ratio delta-method, the TLI interval being the CFI interval
  scaled by `Q̄_b/Q̄_u`. To leading order `Var(CFI)≈Var(T_u)/δ_b²`, so CFI
  inference is a rescaling of the RMSEA-side variance. Derivation in
  `cfi_tli_misspec_inference.tex`; gated by `ordinal_test.cpp`
  and the C4 MC harness `cpp/tests/checks/ordinal_cfi_inference`. Empirically CFI's CI
  is calibrated and (unlike RMSEA) largely robust to weight estimation (γ-share
  ≈±3–8%); the leading-order simplification holds only at weak misfit. TLI's point
  is calibrated but its variance over-states at strong misfit (`c=Q̄_b/Q̄_u`
  ill-conditioned), so CFI is the index to trust for an interval.
  The whole estimated-weight fit-index family (RMSEA + CRMR/SRMR + CFI/TLI with
  CIs) is R-exposed through one consolidated surface,
  `estimate::ordinal_fit_measures_misspec_inference`
  (`OrdinalMisspecFitMeasures`), bound as `infer_ordinal_fit_measures_misspec`
  and the `@export`ed `fit_measures_misspec(fit, ordinal_stats, ...)`; the
  per-index C++ entry points stay unbound. All of RMSEA/CRMR/CFI/TLI are now
  **multi-group**: the criteria pool as `Σ_b n_b·crit_b` with a block-diagonal
  `Γ_x`, the gradients stack with `√(n_b/N)` weights, and the baseline is the
  per-group independence model (validated by a duplicate-group reduction:
  points invariant, statistic/df double, intervals tighten). A stabilized-`c`
  TLI variance remains deferred.
  Mixed continuous/ordinal DWLS has the first reportable fit-index slices:
  `estimate::mixed_ordinal_rmsea_misspec_inference` reuses
  `mixed_ordinal_dwls_profile_rmsea` and adds the same estimated-weight
  envelope-score CI and exact-fit mixture p-value for RMSEA, while
  `estimate::mixed_ordinal_crmr_misspec_inference` adds CRMR/SRMR inference over
  the existing standardized mixed association residual convention (including
  observed continuous-variance scale derivatives), and
  `estimate::mixed_ordinal_cfi_tli_misspec_inference` adds CFI/TLI inference
  with the mixed DWLS independence-baseline convention. The mixed consolidated
  surface is `estimate::mixed_ordinal_fit_measures_misspec_inference`, exposed as
  `magmaan_core$mixed_ordinal_fit_measures_misspec` and the exported R companion
  `fit_measures_misspec_mixed_ordinal()`.
- Multi-group robust MI / score tests for the ordinal and mixed-ordinal tiers
  (2026-06-13): the `require_single_group_ordinal` guard in
  `estimate::frontier` is removed; the ordinal sandwich already loops over
  `stats.R.size()` and pools `A1/B1` through the shared `n_b/N`-weighted
  `estimate::weighted_param_space_sandwich`. The per-block threshold and
  association moment Jacobians write into the correct full-θ columns, so the
  continuous-tier nuisance projection carries over. Validated by exact
  two-group full-WLS reductions for all-ordinal MI/score and mixed-ordinal
  MI/score, a finite non-trivial two-group DWLS ordinal scaling case, and the
  two-group WLSMV golden 0012 (`c ≈ 0.856`) assembled from lavaan's
  per-group `delta` / `wls.v` / `gamma` lists.
- df>1 total release (2026-06-13): `inference::frontier::score_tests_robust_joint`
  releases all active equality constraints at once. The NT joint statistic is the
  multivariate score (Lagrange-multiplier) form `T = uᵀV⁻¹u` over the
  df-dimensional efficient-score subspace `G` (release normals made
  info-orthogonal to the nuisance subspace; u = Gᵀs, V = GᵀIG), which reproduces
  lavaan's `lavTestScore(fit)$test$X2`. The robust report mean-scales by
  `c̄ = tr((GᵀA1G)⁻¹(GᵀB1G))/df = Σλ/df` and also gives the exact eigenvalue-mixture
  p-value `Pr(Σλⱼχ²₁ > T)` via the QUADPACK `imhof_upper`, with λ the generalized
  eigenvalues of (GᵀB1G, GᵀA1G) (`JointScoreTestResult`; the shared worker
  `score_for_subspace_robust`). At df=1 it reduces to the per-row `c` bit-for-bit.
  Complete-data ML (raw or caller Γ̂); single- or multi-group. Oracle: df=2 joint
  fixture 0011 (mi = lavaan total, c̄ ≈ 2.03, p_mixture vs `CompQuadForm::imhof`)
  plus a df=1-reduces-to-per-row unit check.
- Estimated-weight ("complete-sandwich") robust MI / score tests (2026-06-22):
  an `estimated_weight` flag routes the per-direction scaling `c = gᵀB1g/gᵀA1g`
  through the complete Hall-Inoue infinitesimal-jackknife meat (the
  data-dependent-weight `IF(Ŵ)` term), not the fixed-weight `Δ'WΓ̂WΔ` — the
  per-parameter robust denominator lavaan never builds (it scales MI only by the
  global SB scalar). Core: `estimate::weighted_param_space_sandwich_ij` plus the
  IJ adapters `continuous_ls_param_space_sandwich_ij` (GLS/WLS/DWLS/DLS) and
  `ordinal_param_space_sandwich_ij` (all-ordinal DWLS/WLS), the latter built from
  `build_ordinal_ij_blocks` shared with the `robust_ordinal_ij` SE path. ML/FIML
  and mixed-ordinal reject the flag. Leading-order only under misspecification
  (ULS, fixed weight, unaffected). R: `{modification_indices,score_tests}_robust(…,
  estimated_weight=)`.
- Misspecification-robust ("complete-sandwich") case influence (2026-06-23): the
  casewise dual of the estimated-weight SE.
  `estimate::continuous_ls_casewise_influence_ij` decomposes the complete-sandwich
  covariance into its per-case contributions `c_i = (1/N)·K·A⁻¹·(Δ_b K)ᵀ·v_i`
  (observed bread `A`; `v_i = g_i·W + IF(Ŵ)`), the one-step leave-one-out change
  carrying the data-dependent-weight term that the naive (semfindr / Pek-MacCallum)
  influence drops. It reuses the same `build_continuous_ls_ij_blocks` as the MI /
  SE paths, so `Σ_i c_i c_iᵀ` reproduces the `robust_continuous_ls_*_ij` vcov to
  1e-9 (a free self-check); a `naive` field drops `IF(Ŵ)`, so the difference is
  the per-case `Δ'W'_d` diagnostic. R: `infer_casewise_influence_ij_fit`,
  `est_change_{raw_,}approx(fit, type = "estimated.weight")` (continuous
  GLS/WLS/ULS). Self-validated against the exact GLS leave-one-out engine (which
  re-estimates the weight per drop): the complete one-step tracks it at RMSE ~3e-4
  regardless of misfit, while the naive degrades with it (the Hall-Inoue order
  promotion). Writeup in `papers/estimated-weight-se`. The per-case row extraction
  is the shared `estimate::casewise_influence_from_ij_blocks(blocks, K, bread)`
  (the per-case dual of `robust_weighted_moment_ij`), reused by the continuous
  accessor and the ordinal `estimate::ordinal_casewise_influence_ij` (categorical
  DWLS/WLSMV — the headline cell, on the existing `build_ordinal_ij_blocks`).
  Both case-influence regimes (`standard`, `estimated.weight`) are multiple-group
  (block-stacked `g{b}_{row}` ids); `est_change_*_approx(type = "estimated.weight")`
  covers continuous GLS/WLS/ULS, ordinal DWLS/WLSMV, and two-stage **ML2S**
  (`estimate::fiml::two_stage_casewise_influence_ij` — the missing-data member,
  fusing the Stage-1 saturated-moment influence with the Stage-2 weight term;
  NT/robust.two.stage carries a zero correction, non-NT DWLS/ADF/DLS the live
  one), routing ordinal/ML2S fits to their bindings automatically. Every
  estimator/group cell self-checks `Σ_i c_i c_iᵀ ≡` the matching observed-bread
  `robust_*_ij` vcov.
- Observed-bread robust SEs and observed-Hessian U-factors use total-N scaling
  and work on block-stacked multi-block covariance and mean-structure models.
- Browne's unbiased reduced gamma has a single-block reduced-matrix shorthand
  and a casewise multi-block primitive.
- The weighted-sum-of-chi-squares tail behind the FMG/pEBA/pOLS p-values
  (`robust::weighted_chisq_upper`) uses Ruben's positive-weight
  central-χ² series first, avoiding oscillatory-cancellation failures in deep
  tails. If the series does not converge it falls back to Imhof's characteristic
  function inversion through vendored QUADPACK `qagi`
  (`cpp/third_party/quadpack/`, f2c-translated, public domain;
  cpp/cmake/QuadpackVendor.cmake — the second vendored static library after PORT),
  with a dense-Simpson fallback for weakly damped small-df tails where qagi's
  extrapolation breaks. A self-contained C++ golden pins fixed-spectrum FMG
  p-values for each method against constants generated from R `stats` and
  `CompQuadForm::imhof`; unit tests also pin deep equal-weight tails to exact
  χ² references. The same module exposes `robust::weighted_chisq_quantile`
  for deterministic positive-mixture cutoffs, and
  `robust::compute_profile_contrast_spectrum` computes the positive `QΓ`
  spectrum for regular profile-Hessian nested-test research primitives.

### Continuous FIML

- Direct observed-pattern ML over raw continuous data with missingness masks.
- Rows are compressed into observed-value patterns; the observed-pattern
  objective and analytic gradient reuse `ModelEvaluator` Jacobians.
- The explicit frontier `fit_fiml_psd()` path evaluates that same cached
  objective on Cholesky-lifted primitive covariance blocks and exposes the thin
  R wrapper `frontier_fit_fiml_psd()`. It preserves ordinary FIML's fixed-x
  missingness rules and returned partable parameterization; boundary inference
  is deliberately not claimed.
- `fit_fiml()` defaults to NLopt L-BFGS with an SLSQP retry when the L-BFGS
  run fails or returns a non-clean optimizer status; explicit `nlopt-lbfgs` and
  `nlopt-slsqp` remain available for diagnostics and parity checks.
- Cross-call precomputation is value-based, with no mutable cache state:
  `FIMLPack` (immutable pattern cache + pairwise-complete start statistics,
  built by `fiml_pack`) and `FIMLH1` (per-block saturated EM moments plus the
  converged H1 objective value, built by one EM run in `fiml_h1_moments`).
  The H1 EM fails closed by default when a missing-data block reaches its
  iteration cap, returning `FitError::OptimizerNonConvergence` rather than a
  last-iterate H1 value. The default relative EM tolerance is `1e-6`.
  `FIMLH1Options` exposes `max_iter`, tolerance,
  covariance floor/warning thresholds, and an explicit
  `error_on_nonconvergence = false` diagnostic mode that restores the old
  last-iterate-with-warning behavior. Near-singular EM covariance updates are
  still diagonally floored before iteration continues, with
  `FIMLH1::warnings` recording any repairs or low eigenvalues.
  Every post-fit helper (`fiml_extras`, `fiml_observed_information`,
  `fiml_robust_mlr`, `saturated_em_moments`, `fiml_eta_jacobian`,
  `fiml_ugamma_spectrum`, `fiml_baseline_chi2`), the FIML score/MI helpers
  (ordinary plus MLR robust), the FIML Satorra-2000 nested-test helper, and
  `fit_fiml` carry pack overloads next to the raw-only signatures, which now
  build the pack/H1 once and delegate. `api::sem` FIML fits build both eagerly
  at `fit()` time and expose them via `Fit::fiml_pack()` / `Fit::fiml_h1()`.
  The R FIML fit list mirrors this with opaque `fiml_pack` / `fiml_h1` external
  pointers: `fit_fiml_impl()` uses the retained pack for optimization, and the
  Rcpp MLR, FMG, and score/MI wrappers consume the retained pack/H1 when
  present, falling back to raw-data rebuilding only for old/minimal fit lists.
  A fit → test → fit_measures → SE/MLR session therefore runs the saturated EM
  exactly once (the H1 value and H1 moments also share that single EM run, where
  the raw-only `fiml_extras` previously ran two).
- Current checked-in fixtures cover single- and multi-group CFA, three-factor
  CFA, labeled equality CFA, latent structural models, observed-variable path
  models under random-x and complete fixed.x policies, equality-constrained
  structural regressions, dense non-monotone missingness, complete observed-row
  equivalence, multi-group fixed.x with complete exogenous variables and
  missing outcomes, explicit mean structures, and a full 25-item five-factor
  bfi real-data FIML parity case.
- Post-fit FIML extras include observed-data normal constants, saturated/H1
  likelihood, baseline/independence likelihood accounting, chi-square,
  information criteria, and fit-index inputs for the current fixture tranche.
- `SaturatedMoments` carries propagated H1 warnings plus warnings from the
  saturated information repair. The saturated information matrix is symmetrized
  and diagonally floored only when needed before forming `H^-1 J H^-1`; routine
  well-conditioned cases are unchanged, while high-dimensional sparse-missing
  cases now return a diagnostic Stage-1 object instead of aborting on a singular
  495x495 H1 information matrix. The advisory stress harness is
  `cpp/tests/checks/fiml_h1_edge`.
- Two-stage EM ML (`ML2S`) is a packaged missing-data estimator path alongside
  direct FIML. Stage 1 fits the saturated EM mean/covariance model and exposes
  `(H, J, ACOV)` plus casewise saturated-moment influence rows through
  `saturated_em_moment_influence`; Stage 2 runs complete-data ML on those
  saturated moments.
  `estimate::fiml::two_stage_em_ml_inference` converts the Stage-1 ACOV to the
  moment Gamma scales expected by the shared robust SE and U-Gamma reducers,
  returning Savalei-Bentler-style sandwich SEs, ML chi-square, df, the corrected
  U-Gamma spectrum, scaling factor, and scaled chi-square.
  `TwoStageBread::Observed` is available on the inference path for the
  misspecification-robust Stage-2 bread regime; complete-data tests reduce it
  to unstructured observed-bread robust SEM for the NT weight and observed-bread
  `robust_continuous_ls` for the ADF weight. The default remains the
  lavaan-parity expected-bread convention.
  The separate single-group/mean-structure frontier diagnostics
  `estimate::fiml::frontier::{two_stage,fiml}_information_choices` enumerate
  the historical matrix-estimation axes without changing either estimator's
  default. The ML2S audit crosses four saturated/structured,
  observed/expected Stage-1 breads, two score-meat evaluation points, and four
  complete-data Stage-2 residual metrics (32 rows). The direct-FIML audit
  crosses the six Equation-37 residual metrics catalogued by Savalei and
  Rosseel (2022) with four sandwich breads and two meat points (48 rows).
  Experiment replications/09 applies all 80 choices to the same generated samples to identify
  the Savalei--Falk/EQS configurations. Its 2 x 1,000 targeted run identifies
  the two-stage source candidate (11.5% rejection versus the published 10.0%):
  saturated observed Stage-1 bread, saturated score meat, and structured
  observed Stage-2 information. Direct FIML remains unresolved. The literal
  all-structured observed Equation-(5/6) row was usable in only 935 samples and
  rejected 36.5%, whereas the closest row used saturated expected residual
  information and a saturated expected/saturated-score sandwich (61.9% versus
  63.3%). This conflict is now an explicit matrix-oracle gate before any
  paper-era direct-FIML route is added.
  `estimate::fiml::two_stage_fit_measures` adds the matching TS global-index
  layer: baseline scaling (`cB`), scaled CFI/TLI/RMSEA, and robust CFI/TLI/RMSEA
  with RMSEA confidence intervals and p-values. Complete-data multi-group tests
  anchor the corrected SE/spectrum path against the ordinary complete-data
  robust.sem machinery; missing-data golden tests gate the TS scaling and global
  indices against lavaan `missing = "robust.two.stage"`.
  - *Frontier:* the Stage-2 weight is selectable
    (`estimate::fiml::TwoStageWeight` ∈ {`Nt`, `Uls`, `Dwls`, `Adf`, `Dls`},
    built by `two_stage_stage2_weight`). `Nt` is the lavaan
    `robust.two.stage` default and is unchanged; the non-NT members are
    weighted-LS Stage-2 estimators (`Uls` = identity, `Dwls` =
    `diag(Γ_FIML)⁻¹`, `Adf` = `Γ_FIML⁻¹`, `Dls` = Browne
    `Γ_NT`/`Γ_FIML` mix) that route the single-model GOF/SE path
    through `robust_continuous_ls` and the ML2S Satorra-2000/2001 difference
    tests through the same eigen-cores (a `(V, Γ)` swap, no new test machinery).
    R: `fit_model(estimator = "ML2S", stage2_weight =, dls_a =)`. Opt-in
    Stage-1 covariance conditioning is also exposed as
    `stage1_regularization = TRUE` / `list(...)`: it regularizes the saturated
    EM covariance before Stage 2 with a diagonal/scaled-identity/identity target,
    chooses the smallest intensity needed to meet the requested condition/eigen
    cap when no fixed intensity is supplied, preserves `$stage1_raw`, and
    propagates the same moment map through `$stage1$acov` by delta method.
    Regularized ML2S is frontier/non-lavaan-parity; the default remains raw
    robust.two.stage. Rationale and the open heavy-MAR efficiency/calibration
    sim: `two_stage_weighting.tex`.
- Robust FIML MLR post-fit reporting computes observed-pattern casewise
  sandwich SEs and Yuan-Bentler Mplus scaled-test traces for fixture-backed
  non-saturated single- and multi-group cases. The R surface also exposes
  `fiml_observed_vcov()` / `inference_fiml_observed_vcov()` and routes
  `vcov(fit, regime = "model" | "robust")` for FIML to the inverse observed
  information or the MLR sandwich respectively; saturated (`df = 0`) FIML fits
  still return robust `vcov`/`se`, with scaled-test scalars set to `NaN`. The
  observed FIML information
  (`fiml_observed_information`, the MLR sandwich bread, and the `api` FIML
  information path) is analytic: the per-pattern moment-space Hessian chained
  through the model Jacobian plus the pattern-aggregated moment gradient
  contracted with the closed-form LISREL second derivatives (shared with the
  complete-data analytic observed information via `detail_second_order.hpp`);
  `diagnostic::fiml_observed_information_fd` retains the central-difference
  route as a regression comparator.
  The comparison primitive
  `inference_fiml_information_vcov()` additionally exposes the structured
  expected Fisher, observed-H1, and full observed-Hessian matrices at one
  retained fit. Each information matrix is returned with its inverse and with
  the same observed-pattern score-cross-product sandwich, preserving equality-
  constraint projection through the model covariance. Experiment research/39 validates
  these six SE conventions against explicit lavaan settings and studies their
  calibration without changing the high-level FIML defaults.
- The public fixed.x policy rejects missing observed exogenous variables rather
  than approximating lavaan's conditional likelihood behavior.
- The R boundary exposes `df_to_fiml_data()`, estimate-only `fit_fiml()` with
  retained FIML pack/H1 state, and packaged `fit_ml2s()` /
  `estimate_two_stage_em(..., kind = "ml")`. `fit_measures()` dispatches FIML
  fits to the FIML likelihood and independence-baseline path rather than the
  complete-data `2N*fmin` baseline helper; `robust = TRUE` / `"MLR"` adds the
  corrected `XX3`/baseline scaling and robust CFI/TLI/RMSEA fields from
  `estimate::fiml::fiml_corrected_fit_measures`, using lavaan's FIML-C(V3)
  trace and reusing a fit's retained Stage-1 saturated moments when present.
  ML2S attaches its corrected
  `vcov`, `se`, `chisq`, `df`, `chisq_scaled`, and `scaling_factor` fields
  directly because those corrections are part of the named two-stage estimator
  rather than optional post-fit reporting; the C++ post-fit surface also exposes
  the lavaan `robust.two.stage` scaled/robust CFI/TLI/RMSEA family. On the R
  surface, `fit$ml2s` carries the same baseline/scaled scalar list so
  `fit_measures(fit, robust = fit$ml2s)` reports the lavaan two-stage robust
  global-index fields.
- FMG single-model goodness-of-fit p-values are first-class complete-data ML
  and continuous ULS/GLS/WLS post-fit diagnostics on the R surface.
  `fmg_tests()` returns the p-value,
  df, ML/RLS source statistic, method/parameter, UG flag, chi-square-equivalent
  diagnostic, truncation count, and UGamma/lambda spectra; `fit_measures(...,
  fmg = ...)` attaches the same table to the ordinary fit-measure path, while
  the legacy `fmg_pvalues()` remains a named-vector compatibility view. Fits
  built from `fit_model(..., data.frame, estimator = "ML")` or
  `fit_ml(model, df_to_data(...))` retain listwise-complete raw blocks in
  `fit$raw_data`, so FMG no longer requires a separate raw-data argument in the
  normal fit workflow. Continuous LS composes the existing
  `estimate::robust_continuous_ls` primitive and exposes empirical or
  normal-theory Gamma explicitly; continuous WLS also takes the caller's
  fitting weight because fit lists do not retain it. The public
  `df_to_data(scaling = "n" | "n-1")` choice makes covariance scaling explicit
  for cross-implementation work. The fused C++ spectra primitive supports
  complete-data single- and multi-group ML, including mean structures.
  FIML/missing-data fits
  are also supported (`fmg_tests()` accepts a `fit_fiml()` /
  `fit_model(..., estimator = "FIML")` fit, single- or multi-group): the
  missing-data UGamma spectrum is built first-principles by
  `estimate::fiml::fiml_ugamma_spectrum` from the saturated-moment ACOV
  `Gamma_mis = H^-1 J H^-1` (`saturated_em_moments`, with analytic observed-row
  Hessians for saturated H1 information and a C++-only finite-difference
  diagnostic comparator) plus the saturated observed H1 information `V = H` as
  the projector metric (PD by second-order optimality — the FMG-spectrum
  convention; a selectable structured-at-θ̂ variant was removed 2026-06-24, see
  the backlog). The route uses
  `U = V - V Delta (Delta' V Delta)^-1 Delta' V` and the df eigenvalues of
  `U Gamma_mis`, with the FIML LRT as the base statistic. Equality constraints
  are honored by projecting `Delta` into the local free-coordinate space:
  affine linear constraints use their `K` reparameterization, while nonlinear
  equality constraints use the tangent-space null basis of `[A_eq ; dh/dtheta]`
  at `theta_hat`.
  The saturated H1 is the `h1.information = "unstructured"` convention
  natural to FIML's EM saturated model: on complete data it reproduces lavaan's
  unstructured UGamma spectrum element-for-element (~1e-7), validated in
  `examples/fmg.R`. semTests'
  rescale-the-mis-normalized-lavInspect-UGamma FIML hack is unsound and is
  deliberately NOT matched. Under FIML only the biased Gamma-hat and the ML base
  are defined; `_ug` and `_rls` are rejected.
  Two-stage ML (`ML2S`) fits are supported the same way: `fmg_tests()` accepts a
  `fit_ml2s()` / `fit_model(..., estimator = "ML2S")` fit and applies the
  eigenvalue-tail transforms to the df-dimensional two-stage UGamma spectrum and
  Stage-2 ML base chi-square already attached as `fit$ml2s` (`eigvals`, `chisq`,
  `df`) by `two_stage_em_ml_inference`. The two-stage spectrum uses the
  saturated-moment EM sandwich ACOV as its meat and an expected normal-theory
  Satorra-Bentler weight built from the *unstructured* (sample/saturated h1)
  moments as the U-metric; as under FIML, `_ug` is rejected. Unsuffixed ML2S
  FMG requests use the Stage-2 ML discrepancy. An explicit `_rls` suffix uses
  `inference::rls_chi2()` with the same ML2S UGamma spectrum — the same helper
  the FIML and nested paths use, so `ml` and `rls` are now commensurable bases
  (both model-projected) under one spectrum. This previously routed to a
  separate `frontier::rls_mean_cov_chi2()` because `rls_chi2()` was
  covariance-only; that split is gone (see the RLS entry below). ML2S must be
  dispatched before FIML
  because a two-stage fit also carries a `magmaan_fiml_data` raw object. The
  two-stage scaling and SEs match lavaan's `missing = "robust.two.stage"`
  convention (Huber-White sandwich Stage-1 ACOV) to machine precision - base,
  `pvalue.scaled`, and SEs agree to ≲`1e-4` across the exp-research/05 grid (typically
  ~`1e-7`), the residual being EM/optimizer convergence tolerance, not a
  convention difference. It is *not* lavaan's plain `missing = "two.stage"`, which uses a
  normal-theory ACOV that collapses toward the naive test under non-normality (its
  implied reference-law mean diverges sharply from magmaan's while
  robust.two.stage's tracks it); the *base* matches both lavaan conventions
  (identical point estimates). `trace(UGamma) = E[T]` (normal-data `ncp_hat =
  mean(base) - mean(trace) ~ 0`) is an independent first-principles check.
  (The U-metric weight was previously built from the *structured* model-implied
  moments, leaving a 1-3% trace gap to robust.two.stage that grew with
  non-normality; the unstructured weight - the convention lavaan two-stage forces
  and FIML FMG already used - closed it exactly.) The historical comparison runner is
  `experiments/research/active/44-fiml-global-tests/lanes/two-stage`; only a ten-rep
  local smoke survives there. Substantive SEM calibration and its target limits
  are summarized in that study's parent report. A separate literature reconstruction in
  `experiments/replications/08-savalei-falk-2014-test-conventions` separates
  modern oracle targets from Savalei and Falk's (2014) written conventions.
  The article describes analytic observed information (`SE=EXACT`) and
  structured-model residual projections, with an observed-information
  two-stage correction. Numerical rejection fingerprints instead favor
  expected-information-like FIML choices; original EQS 6.1 identity is still
  unresolved. The current MLR trace-difference, FIML saturated-H1 FMG metric
  and ML2S unstructured-H1 metric retain their existing oracle meanings.
  Any paper-era route needs independent same-data matrix/trace validation;
  numerical similarity must not relabel a current route as an exact replication.
  Nested/model-pair FIML FMG is
  available through the existing `robust_nested_lrt()` / `nestedTest()`
  `method = "restriction_map"` route when both fits are FIML and carry
  compatible `raw_data`: for empirical Gamma it builds the H1-anchored split
  sandwich `A1 = K1' I_obs(theta_H1) K1`,
  `B1 = (V Delta K1)' Gamma_mis (V Delta K1)`. The default
  `convention = "magmaan"` uses the saturated-EM eta-space metric
  (`V = SaturatedMoments::H`, `Gamma_mis = SaturatedMoments::acov`). The
  `convention = "lavaan"` compatibility mode instead mirrors
  `lavTestLRT(method = "satorra.2000")`: parameter bread is the per-observation
  observed FIML information, `WLS.V` is the per-group expected information
  over observed missingness patterns at the fitted larger model covariance,
  and `Gamma_g = n_g * SaturatedMoments::acov_g` is the saturated EM sandwich
  covariance. Its bread is the saturated observed Hessian and its meat is the
  saturated casewise score crossproduct. This matches lavaan MLR's default
  observed/structured information settings. The correction on 2026-09-25
  replaces the former masked raw-residual Gamma and saturated-covariance weight;
  native defaults and the ML2S compatibility convention remain unchanged. The R
  wrapper defaults `A.method` to `"delta"` when
  `convention = "lavaan"` is requested. Both conventions report the existing
  unscaled/scaled/mean-variance/scaled-shifted/exact-mixture nested result
  shape. `GammaSource::NT` intentionally keeps the saturated eta-space bread so
  every difference eigenvalue collapses to one. Nonlinear equality constraints
  are supported by local tangent bases for both models; when `"exact"` is
  requested for such a pair, the route warns and uses the local tangent
  restriction because no global affine exact map exists.
  Frontier reference regularization is opt-in through
  `h1_reference_regularization` on `robust_nested_lrt()` / `nestedTest()` for
  direct-FIML restriction-map tests. Defaults remain raw/lavaan-parity. In the
  lavaan convention, the option transforms the saturated covariance and
  propagates that transformation into its ACOV by the delta method; the weight
  remains evaluated at the fitted larger model. In the native eta-space
  convention it
  floors saturated H1 information blocks before inversion and recomputes the
  coherent H1 reference Gamma. The returned nested-test list includes
  `$h1_reference_regularization` diagnostics when the option is enabled. ML2S
  does not consume this option; its separate Stage-1 input regularization stays
  fit-time only.
  Two-stage ML (`ML2S`) pairs are routed the same way
  (`robust_nested_lrt()` / `nestedTest()` dispatch ML2S before the FIML check,
  since an ML2S fit's `raw_data` is a `magmaan_fiml_data`; `computation =
  "ml2s_eta"`). The ML2S difference reduction uses the selected Stage-2
  two-stage weight (`Nt` by default; `Uls`/`Dwls`/`Adf`/`Dls` for weighted
  Stage-2 fits, not the saturated `V = H` of the FIML route) with the EM-ACOV
  meat `two_stage_gamma_from_acov(sm)` and the Stage-2 chi-square difference as
  the base. Under `convention = "lavaan"`, ML2S keeps the selected Stage-2
  `WLS.V` but swaps the meat to lavaan's model-based raw-moment Gamma, which
  matches `missing = "robust.two.stage"` nested differences in the paper parity
  gate. A `GammaSource::NT` collapse (every difference eigenvalue exactly 1)
  gates the NT convention. The eigenvalue-tail battery is applied to any of these
  difference spectra (continuous ML / FIML / ML2S) through `fmg_nested(fit_H1,
  fit_H0, ...)`, the model-pair analogue of `fmg_nested_ordinal()`: it harvests
  `(T_diff, df_diff, eigenvalues)` from the restriction-map result and runs the
  `pEBA`/`pOLS`/`all`/`penalized-all` transforms, returning a `magmaan_fmg_tests`
  table. Complete-data ML pairs additionally accept an explicit biased `_rls`
  base: `T_RLS,H0 - T_RLS,H1` is evaluated at the two ML estimates and fed
  through the same restriction-map spectrum. The `_ug` suffix selects the
  grouped, mean-structure-aware Browne/Du-Bentler finite-sample Gamma
  correction; unsuffixed nested names remain ML for compatibility, while
  FIML/ML2S reject both `_ug` and `_rls`. Continuous ULS/GLS/WLS pairs use the
  estimator's quadratic-form difference and the continuous-moment sandwich
  restriction map; ULS defaults to normal-theory Gamma, GLS/WLS to empirical
  Gamma, and WLS requires its caller-supplied fitting weight. Their FMG rows
  use the `_ls` base suffix. The R workflow checks source statistics directly
  and all biased/unbiased ML plus continuous GLS/ULS transformed tails against
  `semTests::pvalues_nested(method = "2000")`. This is an asymptotically
  equivalent RLS comparator, not an RLS estimator or casewise RLS score-flip
  method.
  FIML convention note (2026-06-30): lavaan's public
  `lavTestLRT(method = "satorra.2000")` default is delta/scaled-shifted with
  the lavaan `WLS.V`/Gamma convention above. `convention = "lavaan"` matches
  that oracle for complete, MCAR, and MAR FIML/ML2S paper parity cells; the
  independent saturated-EM convention remains the magmaan default. The
  strict-invariance `"exact"` row space can still differ from lavaan's internal
  exact helper because lavaan carries earlier invariance rows into that exact
  body before projection; use `A.method = "delta"` for the lavaan-default oracle.
  See `project/validation/satorra2000_parity.md`.
- **Method-2001 difference spectrum (`U_D = U0 - U1`) for FIML/ML2S.** Alongside
  the Satorra-2000 restriction map ("method 2000", `U_D` from the H1 fit),
  `robust_nested_lrt()` / `nestedTest(ud_method = "2001")` builds the
  Satorra-Bentler (2001) difference of the two single-model residual projectors
  (`compute_diff_spectrum_2001`, the top `df0 - df1` eigenvalues of `(U0-U1)·Γ`
  in a common saturated meat space; FIML uses observed model-information bread
  with `V = sm.H, Γ = sm.acov` unless the same direct-FIML
  `h1_reference_regularization` option is enabled, while ML2S uses
  `V = ml2s NT weight, Γ = two_stage_gamma`). It feeds the same scaled/mixture
  readouts and the FMG/pEBA transforms, and unlike method 2000 accepts non-`==`
  nesting (different parameter counts) but can carry negative eigenvalues
  (flagged for fallback). This is the second `U_D` estimator the FMG-nested paper
  and `semTests::ugamma_nested(., "2001")` cross against; validated to ~5e-5 vs
  semTests on complete data (single + multi-group). The scalar two-constant
  baselines also exist for FIML/ML2S: `nestedTest(method =
  "satorra.bentler.2001"/"2010")` (the trace-based SB2001 and the M10 positivity
  fix; 2010 embeds the restricted point into the alternative slots). NOTE:
  under missing data,
  `lavInspect("UGamma")` uses a non-official normalization; magmaan matches
  lavaan's *official* scaling factor / `trace.UGamma` (1e-7), so validate the
  spectrum against complete-data semTests, not the missing-data `lavInspect`.
  The helper computes biased and optional Browne/Du-Bentler unbiased U-Gamma
  spectra through C++, keeping the U-factor, tiled casewise-contribution
  projection, grouped reduced matrices, and eigensolves out of R list
  roundtrips. The unbiased correction mirrors lavaan's complete-data rule: the
  covariance block uses `Gamma_NT(S)` at the *sample* covariance plus Browne's
  finite-sample coefficients, and mean-structure blocks keep `S` for means with
  the mean-covariance third-order block scaled by `n/(n-2)`. Biased-only
  single-block spectra can still eigensolve in row space when `N < df`; grouped
  and unbiased paths take the full reduced route so per-block denominators and
  the non-identity sample NT term are honored.
  FIML FMG is validated for the full multi-group measurement-invariance workflow
  (configural -> metric -> scalar: cross-group loading/intercept equality plus
  mean structure), for both the GOF spectrum and the nested restriction map, by
  C++ algebra cases in `cpp/tests/unit/fiml_test.cpp` and by
  `experiments/research/active/06-fiml-invariance-tests/lanes/oracle`, whose `--lavaan-parity` run
  reproduces lavaan's FIML LRT chi-square (~1e-7) and, on complete data, the full
  unstructured UGamma eigenvalue spectrum (~1e-5) across all three invariance
  levels and normal / heavy-tailed / MCAR cells. That audit also found and fixed
  a complete-data robust bug: the `build_u_factor` Expected-info projector had
  dropped the per-group weight `n_b/N`, biasing the UΓ spectrum (SB scaling, FMG
  p-values, robust difference test) for models with unequal group sizes plus a
  cross-group equality constraint; it was masked by equal-group designs where the
  weight is a global scalar (regression note in
  [project/validation/test_ledger.md](../validation/test_ledger.md)).
  Saturated-moment reuse now covers the FIML nested path too: the
  `lr_test_satorra2000/2001_fiml_from_data` core functions take an optional
  precomputed `SaturatedMoments* sm_precomputed` (mirroring the ML2S overloads),
  and the `infer_fiml_lr_test_satorra2000` glue reconstructs it from a fit's
  `$stage1` (`saturated_from_stage1`) before falling back to a rebuild -- so a
  caller that stamps `$stage1` onto a FIML fit (or an ML2S fit that already
  carries it) skips the from-scratch saturated EM + observed-information build in
  the difference-spectrum driver. `fit_ml2s(stage1 = ...)` likewise skips the
  rung-independent Stage-1 EM when handed a precomputed object. Bit-identical to
  the rebuild (the EM is deterministic; asserted in `fiml_test.cpp` and at the R
  surface). One residual remains by design: each `fit_fiml` still computes its own
  `fiml_h1_moments` (the cheap mu/Sigma-only EM, no H/J), so the structured H1 is
  rebuilt per rung; injecting it would need a `fit_fiml` h1 argument plus a
  FIMLH1-from-R reconstructor, deferred as the structured optimization dominates.
- Ordinal and mixed-ordinal (polychoric/polyserial least-squares) fits are FMG
  supported via `fmg_tests_ordinal()` / `fmg_tests_mixed_ordinal()`. These reuse
  the UGamma spectrum, base LS chi-square `N*F_min`, and df that
  `robust_ordinal()` / `robust_mixed_ordinal()` already build from the polychoric
  NACOV sandwich (validated against lavaan ordinal robust internals in the
  ordinal goldens), then apply the estimator-agnostic `robust::frontier::fmg_test`
  eigenvalue-tail transforms - no new spectrum machinery. An ordinal LS fit has a
  single base statistic (no ML/RLS split) and the polychoric NACOV is already the
  asymptotic Gamma, so `_ml` and `_ug` test suffixes are rejected. The categorical
  sample statistics are passed explicitly, mirroring `robust_ordinal(fit, stats,
  weight)`; single- and multi-group fits are supported through the same
  per-block `n_b/N` robust ordinal sandwich. Anchored by
  `cpp/tests/unit/ordinal_test.cpp` ("Ordinal FMG transforms consume the
  robust_ordinal UGamma spectrum"; the FMG SB reproduces the stored
  Satorra-Bentler scaling to 1e-12) and `r-package/examples/fmg_ordinal.R`,
  which checks single-group and two-group all-ordinal/mixed SB parity. This is
  the gate for the polychoric-FMG paper track; the pEBA/pOLS/PALL transforms
  remain magmaan-original with no external oracle, as on the complete-data and
  FIML paths.
- Ordinal nested Satorra-2000 tests are wired for all-ordinal and mixed
  continuous/ordinal DWLS/WLS fits via `nestedTest()` / `robust_nested_lrt()`
  when the caller supplies the same `magmaan_ordinal_data` or
  `magmaan_mixed_ordinal_data` statistics object used for fitting. The C++
  workers build the H1 ordinal/mixed moment Jacobian, project through the
  equality reparameterization, pool the `n_b/N` weighted `{A1, B1}` sandwich over
  polychoric/polyserial NACOV blocks, derive either exact parameter restrictions
  or the lavaan-style delta tangent map, and reuse the generic Satorra-2000
  reduced eigenproblem. Mixed pairwise missing remains unsupported because the
  mixed pairwise moment/NACOV constructor does not exist yet. Validation lives
  in `r-package/examples/nested_test_ordinal.R`:
  a single-group loading equality and a two-group configural-vs-metric ordinal
  WLSMV comparison match lavaan's `lavTestLRT(..., method = "satorra.2000",
  A.method = "exact", scaled.shifted = FALSE)` row under lavaan's ordinal WLSMV
  convention, where the displayed difference statistic/`Df diff` are the
  spectrum-derived mean/variance-adjusted `T_adjusted`/`d0` for `m > 1`;
  `r-package/examples/mplus_wlsmv_invariance.R` pins the mixed listwise bridge
  on a small deterministic mixed CFA.
  Friendly FMG composition is available for both branches:
  `fmg_nested_ordinal()` and `fmg_nested_mixed_ordinal()` consume the existing
  Satorra-2000 difference triple and keep both `A.method = "delta"` and
  `"exact"` selectable.
- All-ordinal pairwise-deletion sample statistics are implemented for the
  categorical LS path (`data::ordinal_stats_from_observed_integer_data`,
  surfaced in R as `missing = "pairwise"` with `pd_gamma = "overlap" |
  "nominal"`). Thresholds use itemwise observed cases and polychorics use
  observed pairs; both variants feed the existing `OrdinalStats` / DWLS / WLSMV
  machinery. The `overlap` Gamma is the default and applies the support-overlap
  finite-sample scaling from `ordinal_pd_gamma.tex`; the
  `nominal` variant keeps the same PD point estimates but suppresses the
  overlap reweighting for replication of the conventional wrong implementation.
  Experimental second-stage all-ordinal weights can now be built from the same
  precomputed `OrdinalStats` without rerunning threshold/polychoric/Gamma
  estimation: `estimate::frontier::ordinal_stage2_weight_blocks()` supports
  ULS, DWLS, observed full WLS/ADF, a GLS-like NT association weight, and DLS
  interpolation, with R helpers `ordinal_stage2_weight_blocks()` and
  `fit_ordinal_stage2()`. The NT endpoint is intentionally a research
  construction, not a lavaan oracle: threshold uncertainty remains the observed
  ordinal Gamma while the association block uses the MVN normal-theory
  correlation Gamma; robust reporting keeps the observed pairwise Gamma as the
  meat.
  Mixed continuous/ordinal pairwise missingness remains unsupported. Regression
  coverage lives in `cpp/tests/unit/ordinal_test.cpp`. The redundant construction
  probe was removed; its smoke outputs were not calibration evidence.

### Two-level (multilevel) ML

- Two-level normal-theory ML (v1) is landed. Scope: random-intercept models
  over a SHARED observed variable set (the same observed variables decompose
  into a within and a between part), single group with multi-group two-level in
  progress, complete data, no constraints. The `level:` block header is a real
  `(group, level)` axis: `level: 1` is within / L1, `level: 2` is between / L2,
  one `block` per (group, level) pair.
- Standard errors are available both observed (Hessian-inverse) and via the
  analytic expected information; chi-square is the LRT against the two-level
  saturated (H1) moments, df-exact and parity-matched to
  `lavaan::sem(model, data, cluster=)`.
- Entry points: `estimate::twolevel::fit_ml_twolevel` (core fit),
  `api::twolevel_ml()` + `api::data_from_cluster()` (the staged facade), and the
  `fit_twolevel` R binding. Cluster sufficient statistics are built from raw
  data plus a cluster id; the within/between sample-stat decomposition is an
  independent check against lavaan's `lavInspect(fit, "sampstat")`.
- Optional stabilizer bounds are supported on the two-level path. Explicit
  `Bounds` flow through the C++ facade; the R `fit_twolevel()` /
  `fit_model(..., cluster=)` `bounds =` presets reuse the ordinary
  lavaan-compatible bound builders after projecting the two-level H1
  within/between moments into per-block `SampleStats`, so `bounds = "standard"`
  uses level-specific observed variances rather than a pooled covariance.
- Validation: `cpp/tests/golden/twolevel_golden_test.cpp` drives the public api end
  to end against checked-in lavaan oracle fixtures
  (`cpp/tests/fixtures/twolevel/*.json`, regenerated by
  `cpp/tests/tools/regen_oracle_twolevel.R`), asserting theta-hat, standard SEs,
  chi-square, and df. Unbalanced full-information cells are additionally
  cross-checked against the Mplus 9.1 Demo via
  `cpp/tests/tools/regen_oracle_twolevel_mplus.R` (a maintainer tool; CI invokes
  neither R nor Mplus).
- Remaining work (see [project/backlog/todo.md](../backlog/todo.md)): multi-group
  two-level (in progress), between-only / within-only observed variables,
  constraints under two-level, categorical/robust two-level, 3+ levels, and
  random slopes.

### Staged C++ facade

- `magmaan::api` provides a tested staged facade over the currently
  implemented core primitives. Model construction, data construction, fitting,
  standard SEs, robust reporting, tests, modification indices, score tests,
  Wald/z tests, standardization, defined parameters, fit measures, and nested
  tests remain explicit calls rather than a lavaan-style one-shot summary.
- The facade supports complete-data ML, raw continuous FIML, continuous
  ULS/GLS/explicit-weight WLS, and ordinal/mixed DWLS/WLS through the same
  underlying fit and post-fit primitives described above. Unsupported
  combinations fail with `api::ErrorStage::UnsupportedCombination`.

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

### Optimizer backends

User-facing the optimizer is selected by a single kebab-case `optimizer = "..."`
string threaded through `fit_model()` and the underlying `fit_ml/fit_uls/fit_gls/
fit_wls/fit_*_snlls/fit_*_ordinal` entries; the table lives in
`cpp/include/magmaan/estimate/backend_strings.hpp` and parses into the C++
`Backend` enum. Accepted strings: `"ceres"`, `"ceres-bfgs"`,
`"nlopt-slsqp"`, `"nlopt-bobyqa"`, `"nlopt-tnewton"`, `"nlopt-var2"`,
`"nlopt-lbfgs"`, `"nlopt-lbfgs-slsqp-fallback"`, `"ipopt"`, `"port"`,
`"port-nls"`. The R side passes the same
string through to a single Rcpp shim per fit family — no per-Backend wrapper
explosion. Solver tuning rides on a generic `control = list(max_iter, ftol,
gtol, history)` argument.

The R complete-data ML and LS helpers also expose box constraints through
`bounds = list(lower, upper)` or the named bound builders
`bounds_variance()` / `bounds_pos_var()`, `bounds_standard()`,
`bounds_wide()`, and `bounds_loading()`. The high-level `fit_model()` ML, ULS,
GLS, WLS, and ordinal LS paths thread the same bounds object into the C++
fit layer; FIML remains unbounded at the R surface.

Optimizer outputs carry function/gradient evaluation counts plus a refined
success status and final stationarity diagnostic. The C++ `OptimResult` and
`estimate::Estimates` report `OptimStatus` (`Converged`,
`LineSearchSalvaged`, `SingularConvergence`, or `Unknown`) and a final
(projected, when bounded) gradient infinity norm when the backend can compute
one. R fit lists expose these as `optimizer_status` and `grad_norm`; the
`converged` field projects the common verdict (TRUE/FALSE/NA), independently
of the optimizer stop. A returned estimate need not pass that verdict.


- `Backend::NloptLbfgs` is the complete-data scalar default. FIML defaults to
  `Backend::NloptLbfgsSlsqpFallback`, which retries NLopt SLSQP from the same
  start if L-BFGS fails or returns a non-clean optimizer status. NLopt is a
  required dependency in the ordinary build, so both pieces of the fallback are
  always available.
  Known limitation: the NLopt Luksan L-BFGS line search allows ten step
  reductions, which can be insufficient to reach a finite objective from a
  valid start when parameter scales produce a large gradient. This can abort
  after 12 evaluations regardless of the overall evaluation budget. The
  standalone variance probe in `cpp/tests/checks/nlopt_lbfgs_domain.c` isolates the
  issue; the domain-recovery task is tracked in `project/backlog/todo.md`.
- `Backend::Port` is the trust-region cross-check: vendored PORT (Bell Labs)
  `drmngb_` (TOMS 611 Dennis-Gay-Welsch model-Hessian trust region; the
  algorithm behind R's `nlminb`), supports bounds natively. Vendored at
  `cpp/third_party/port/` from AMPL/ASL + Fermi-LAT (both BSD-3, manifest in
  `cpp/third_party/port/README.md`). Replaces the previous CppNumericalSolvers
  `Backend::TrustRegion`.
- `Backend::PortNls` is the least-squares-shape counterpart: PORT `drn2gb_`
  (TOMS 573 NL2SOL adaptive trust region — the algorithm behind R's `nls`).
  Drives the multi-residual `GmmProblem` directly through reverse
  communication (alternating R/J requests), so NL2SOL sees the true
  Gauss-Newton-plus-secant model Hessian instead of the scalarised
  ½‖r‖² collapse. On non-convex SNLLS problems with multiple local
  optima (e.g., Bollen's democracy SEM under GLS) it may converge to a
  different basin than the gradient backends; the cross-check tests
  document this rather than enforce agreement.
- `Backend::Ceres` / `Backend::CeresBfgs` cover Ceres Levenberg-Marquardt and
  dense line-search BFGS on the least-squares path (build with
  `MAGMAAN_WITH_CERES=ON`).
- `Backend::Ipopt` is the optional system-IPOPT interior-point backend (build
  with `MAGMAAN_WITH_IPOPT=ON`). It is available as a general scalar optimizer
  and accepts nonlinear equality constraints. v1 uses IPOPT's limited-memory
  Hessian approximation, so magmaan supplies objective gradients and
  constraint Jacobians but not exact Lagrangian Hessians.
- `Backend::NloptSlsqp` exposes NLopt's SLSQP for ML, FIML, and LS scalar
  paths (gradient SQP, Kraft 1988). It accepts simple bounds and nonlinear
  equality constraints via analytic constraint Jacobians.
- `Backend::NloptBobyqa`, `Backend::NloptTnewton`, `Backend::NloptVar2`,
  `Backend::NloptLbfgs` round out the NLopt roster with distinct algorithm
  ideas: Powell 2009 derivative-free quadratic-model trust region (BOBYQA,
  finite bounds required); Nash 1985 preconditioned truncated Newton with
  CG inner solve (TNEWTON); Shanno-Phua 1980 full (dense) variable-metric
  BFGS (VAR2); and NLopt's own L-BFGS. All five share the one
  `NloptOptimizer` adapter parameterised over an opaque
  `NloptAlgorithm` enum.
- The local `ceres` and `ipopt` presets are optional optimizer comparison
  builds. Ceres is FetchContent-managed; IPOPT is deliberately a system
  dependency because its BLAS/LAPACK and sparse-linear-solver stack is a
  toolchain choice. `just r-install-ceres` and `just r-install-ipopt` mirror
  the relevant compile definition into the R shared object so the accepted
  optimizer strings are executable from the R dev surface in one install.

### Least-squares estimators

- ULS, GLS, and explicit-weight WLS discrepancies with scalar
  value/gradient and residual/Jacobian interfaces.
- Bounded LS fitting through scalar NLopt L-BFGS and optional Ceres.
- Optional Ceres dense line-search BFGS is exposed only for unbounded SNLLS LS
  research comparisons. It consumes the already-profiled nonlinear block, so
  within-block linear equalities remain compatible through the SNLLS
  reparameterization. General-linear equality constraints use a
  component-wise null-space basis, so independent loading-only and
  intercept-only constraints stay block-separable instead of being rotated
  together by one global SVD. It is not a general bounded or constrained
  backend.
- Automatic nonnegative variance bounds.
- Linear equality constraints on the LS path via the affine α-reparameterization
  (θ = θ₀ + K·α), shared with the ML path — no quadratic penalty.
- Fixed-parameter modification indices and equality-release score tests reuse
  the estimator-specific LS residual/Jacobian weighting for ULS/GLS/WLS.
- Representative modification-index and equality-release score-test fixtures
  now pin fixed rows plus generated absent covariance rows against lavaan across
  complete ML, observed-information FIML, continuous ULS, ordinal DWLS, and
  mixed ordinal DWLS. `ModificationIndexOptions` exposes absent cross-loadings
  and covariances for those paths; generated structural-regression rows remain
  deliberately out of scope.
- Continuous LS fixtures cover point estimates, degrees of freedom, and
  estimator-specific chi-square reporting for representative CFA,
  multi-group, labeled-equality, mean-structure, and observed-exogenous
  fixed.x cases. Mean-structure LS fixtures also cross-check SNLLS against
  the full LS path, and Ceres against NLopt L-BFGS when the Ceres backend is
  enabled.
- The Geiser textbook GLS and ULS corpora are checked in as parity-tier golden
  fixtures generated by `cpp/tests/tools/regen_geiser_fixtures.R` from the
  source-verified `external/textbook-corpus/cases/geiser_2013` plus installed
  lavaan. The tests exercise all curated Geiser cases and compare full/PORT and
  SNLLS/PORT-NLS implied moments against lavaan. Manifest fixed.x path models
  resolve exogenous observed moments from the sample before the comparison and
  gate strictly; the latent AR cross-lagged family is gated strictly via a
  multi-start recipe (lower-objective of a simple-start and an ML-warm-started
  fit). All Geiser cases — including the two manifest fixed.x cross-lagged path
  models (`manifest_ar_cross_lagged`, `…_extended`) — now gate Σ/μ strictly
  against lavaan. The cross-lagged pair previously looked like a magmaan
  fixed.x-propagation defect but was an observed-order mismatch in the golden
  harness: `data::SampleStats` is name-free and positionally aligned to magmaan's
  `[ov.y, ov.x]` `ov_order`, whereas the fixtures store moments in their data
  column order, so the harness now reconciles by variable name before fit/compare
  (see the test ledger).
- A local ignored Mplus SEM source corpus can be built from
  `external/textbook-corpus/raw/MPLUS/*.zip` into
  `external/textbook-corpus/raw/mplus_sem` with
  `cpp/tests/tools/build_mplus_sem_corpus.R`, which uses the corpus's own
  Mplus translator and retains an example only if it reproduces the shipped
  Mplus `.out`. The tracked oracle layer
  (`cpp/tests/tools/regen_mplus_sem_fixtures.R`,
  `cpp/tests/fixtures/mplus_sem/`, and
  `cpp/tests/golden/mplus_sem_golden_test.cpp`) freezes derived lavaan sample
  statistics and estimates only. The current strict tranche gates seven
  continuous growth examples across ML/ULS/GLS/WLS (28 optimizer fits); theta,
  df and fixed-x-resolved implied moments fail the test, not only messages.
  Retained observed-path, ordinal, and mixed examples are recorded for corpus
  classification without committing Mplus raw data.
- Local ignored Little and Newsom textbook corpora can be built from
  `external/textbook-corpus/raw/little/*.rar` and
  `external/textbook-corpus/raw/newsom/*.zip` with
  `cpp/tests/tools/build_little_corpus.R` and
  `cpp/tests/tools/build_newsom_corpus.R`. The shared fixture regen script
  (`cpp/tests/tools/regen_little_newsom_fixtures.R`) writes grouped
  continuous/ordinal/mixed/observed manifests and lavaan oracles under
  `cpp/tests/fixtures/little/` and `cpp/tests/fixtures/newsom/`. Little's cases
  come from `cpp/tests/tools/lisrel_translate.R`, a LISREL 8 interpreter whose
  every translation is verified against the `.OUT` LISREL wrote for the input
  (106 of 108 inputs); the fixture holds the 48 single-group cases with at most
  18 observed variables. Newsom's are the 14 strict first-edition cases whose
  corpus call reproduces the author's script. The broad Little/Newsom
  continuous golden stays skipped: magmaan's default ordinary fit fails on 12
  corrected Little cases and `newsom/ex5_5b` (see the translation audit). A consolidated `magmaan_textbook_corpus_v1` index under
  `cpp/tests/fixtures/textbook_corpus/manifest.json` summarizes Geiser, Mplus SEM,
  Little, and Newsom as one textbook corpus without duplicating their heavy
  oracle payloads. The same directory also carries an advisory generated
  overlap graph (`overlap.json`) plus an empty curation hook
  (`overlap_overrides.json`) so future paper work can find same-data,
  same-syntax, same-shape, and same-oracle-structure examples while preserving
  every source case as its own fixture record; cluster members are linked by a
  spanning star so the graph grows linearly. Kline Guo measurement-invariance
  parity is now represented by a compact checked-in export
  (`case_exports.json`, regenerated by
  `cpp/tests/tools/regen_textbook_case_fixtures.R`) plus an optional
  submodule-backed cross-check in `textbook_corpus_golden_test.cpp`; both
  re-fit the four Guo invariance rungs from per-group summary statistics and
  assert df-exact + chisq parity. The tests document that lavaan's
  `guo_mi_strong` reference is under-converged (magmaan reaches a strictly lower
  chisq at identical df, confirmed by three independent optimizers), so that
  rung is gated by df-exact + no-worse-than-oracle.
- Paper-corpus curation now lives in the ignored nested Git repository
  `external/paper-corpus`. That repository owns raw downloads, minimal derived
  lavaan-ready data/models, raw-to-derived validation, and magmaan-facing JSON
  exports. magmaan keeps copied snapshots under `cpp/tests/fixtures/paper_corpus/`;
  the C++ tests never read `external/paper-corpus` directly. The compact
  `zxqvn` empirical mediation/SEM project is the first promoted case: its
  export is a core complete-data ML point-estimate fixture, while the source
  script's clustered-SE option (a single-level cluster-robust sandwich
  correction, distinct from the two-level ML decomposition above) is catalogued
  outside the current core parity surface. The next promotion candidate is the
  richer `hwkem` measurement-invariance tutorial.
  A second batch now adds 14 aggregate-only paper/tutorial fixtures: five
  Boivin WLSMV models, empirical pregnancy FIML mediation, five Kievit
  latent-change models plus a labelled missingness variant, and two DASS
  factor models (one intentionally inadmissible). The parity target checks
  imported partables, estimation from initial values, df, implied moments,
  ordinal MI/EPC and equality releases. It uses ordinal moments/weights and
  FIML pattern sufficient statistics, not participant rows. The new fixed-row
  MI checks exposed and fixed double counting of all-ordinal explicit fixed
  parameters in both ordinary and robust MI; the single-group categorical
  gradient scaling remains an explicit test-side convention.
- Continuous ULS/GLS/WLS robust adapters reuse the shared weighted-moment
  sandwich/U-Gamma primitive with either supplied Gamma blocks or raw-data
  Gamma construction. ULS `robust.sem` SEs and Satorra-Bentler-family
  statistics are lavaan-backed for the non-fixed.x continuous LS fixtures;
  GLS/WLS robust paths have shape and scaling coverage because lavaan does not
  expose matching robust scaled-test targets for those estimators.
- GLS/WLS reporting follows lavaan's `2 * N * fmin` convention; ULS
  standard chi-square is pinned to lavaan's Browne residual NT statistic,
  including lavaan's `fixed.x` convention of zeroing the fixed exogenous
  rows/columns in the NT inverse weight. ULS robust scaled-test reporting
  follows lavaan's robust LS `2 * N * fmin` base statistic.
- Continuous ULS/GLS/WLS objectives can be evaluated explicitly at any
  supplied parameter vector via `estimate::evaluate_ls_objective()`, keeping
  discrepancy evaluation separate from chi-square reporting scale. RLS exposes
  a matching theta-based `nt::infer::rls_chi2()` overload that builds implied
  moments from the model structure before applying Browne's model-based
  residual quadratic.
- Distributionally weighted least squares (DLS) is available as an explicit
  weight-matrix builder over the existing moment-quadratic LS surface. The
  fixed-scalar builder mixes covariance-moment Gamma matrices between
  normal-theory GLS and empirical ADF/WLS endpoints. The mixed Gamma
  is inverted through a strict eigen-gated SPD inverse
  (`detail::symmetric_inverse_pd_gated`, `tol = 1e-10·max(1,λmax)`): a
  numerically rank-deficient fourth-moment Gamma returns an explicit
  `FitError::NumericIssue` with dim / numerical rank / rcond / smallest
  eigenvalue rather than a barely-PD inverse that would later trip the terminal
  stationarity audit.

- Separable nonlinear least squares profiling exists for LS estimators where
  conditionally linear parameters can be profiled out.

#### Continuous moment-quadratic weights

`gmm::Weight` is `std::vector<BlockWeight>` with Identity, Diagonal, Dense and
NormalTheory blocks. NormalTheory stores a p×p Cholesky factor of A and applies
`blockdiag(A⁻¹, ½ Dᵀ(A⁻¹ ⊗ A⁻¹)D)` without a dense moment-weight factor.
GLS uses A=S; Fisher/IRLS uses A=Σ(theta_k). Empirical DWLS weights are diagonal;
Pairwise normal-theory Gamma and fixed-scalar DLS weights remain dense.
`BlockWeight::to_dense()` fills NT entries by the closed-form vech-pair identity
in O(p⁴), replacing the former O(p⁷) sequence of sparse-basis GEMMs.

Scalar GLS uses `gmm::normal_theory_objective` and the trace-form gradient;
residual/Jacobian backends retain their corresponding LS formulation.
Structured ML2S Stage-2 weights choose Identity for ULS, NormalTheory for NT,
Diagonal for DWLS, and Dense for ADF/DLS. `gls_scalar_objective_test.cpp` gates
Gamma-inverse equivalence (<1e-9 relative, p=2,3,4,5,6,8,10) and scalar
objective/gradient agreement (1e-11/1e-9). Same-path paired fits had
`|delta fmin| <= 1.1e-16` and parameter differences at most 7.1e-13.

The original `opt` comparisons below are historical validation measurements,
not fresh timing claims. A Jacobian with n_free=2p shows the residual-shape
constant-factor tradeoff: NT whitening loses below about p=20, while storage
falls from O(p⁴) to O(p²).

| p | q | dense µs/it | NT µs/it | speedup | dense MB | NT MB |
|---|---|---|---|---|---|---|
| 10 | 65 | 4.7 | 10.1 | **0.5x** | 0.03 | 0.0008 |
| 20 | 230 | 72 | 73 | 1.0x | 0.40 | 0.003 |
| 30 | 495 | 489 | 286 | 1.7x | 1.87 | 0.007 |
| 45 | 1080 | 3531 | 1050 | 3.4x | 8.90 | 0.015 |
| 60 | 1890 | 15302 | 2336 | 6.6x | 27.25 | 0.028 |

Single-group CFA, eight indicators per factor, N=500 and fixed NLopt L-BFGS:

| p | q | n_free | Dense µs/eval | Trace µs/eval | Evaluation speedup | Before ms | After ms | Fit speedup | Max abs dtheta |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 136 | 33 | 32.2 | 3.7 | 8.7x | 1.7 | 0.2 | 10.2x | 2.2e-16 |
| 24 | 300 | 51 | 195.6 | 9.6 | 20.3x | 12.5 | 0.7 | 17.8x | 2.8e-15 |
| 32 | 528 | 70 | 743.9 | 20.3 | 36.6x | 57.6 | 1.9 | 30.7x | 5.1e-14 |
| 40 | 820 | 90 | 2390.7 | 41.5 | 57.6x | 277.9 | 5.0 | 55.5x | 3.6e-13 |
| 48 | 1176 | 111 | 6267.0 | 72.4 | **86.5x** | 584.8 | 9.0 | **64.9x** | 7.1e-13 |

The fitted-cost exponent fell from 4.80 to 2.71. The same designs measured IRLS
with the structured weight rebuilt on each outer iteration:

| p | n_free | Before ms | After ms | Speedup | fmin before/after |
|---|---|---|---|---|---|
| 16 | 33 | 31.2 | 12.4 | 2.5x | 2.809e-02 both |
| 24 | 51 | 223.5 | 37.3 | 6.0x | 7.367e-02 both |
| 32 | 70 | 992.3 | 134.8 | 7.4x | 1.399e-01 both |
| 40 | 90 | 3692.2 | 399.1 | 9.2x | 2.199e-01 both |
| 48 | 111 | 11959.4 | 953.2 | **12.5x** | 3.133e-01 both |

Printed IRLS objectives agree throughout. Sharing the remaining p×p Cholesky
was retired as a performance task: `benchmarks/nt_factor_share_bench.cpp`
measured maximum whole-fit savings of 0.229%, 0.100%, 0.016% and 0.006% at
p=6,12,24,48. Whitened expected-information assembly is documented above;
remaining R-boundary diagonal transport and factor/weight API consolidation
are in the [backlog](../backlog/todo.md#continuous-moment-quadratic-weight-follow-ups).

### Ordinal and mixed categorical LS

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
  fixed/shared/constrained cases. Released-scale delta is rejected by the
  all-ordinal SNLLS paths. Unit tests compare profiled/cache-aware fits with
  full-threshold and legacy bounded fits; lavaan fixtures 0013/0014 cover
  cross-group shared thresholds and threshold-only linear constraints.
  The maintained [workspace contract](../design/ordinal-snlls-gamma-architecture.md)
  owns the data split, profiling algebra, cost rules and support boundaries.
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
  moment-Jacobian columns. Under **delta** lavaan's own released `~*~` scale is
  structurally unidentified (it stays pinned at 1 with a singular vcov), so
  lavaan-delta `group.equal` invariance remains ungated; the delta released
  branch is gated only by the explicit Mplus-style scalar probe below.
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
  helper. For observed/pairwise-missing mixed ordinal/polyserial stats,
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
  vectors and weights used by fitting.
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
[backlog](../backlog/todo.md#ordinal-weight-storage-and-workspace-cleanup).

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
change to the fitted criterion. Full-WLS, observed/missing, and mixed Gamma
helpers retain their existing implementations.

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

### R bindings and public namespace transition

- **Prepared R interface:** `prepare_model`, `prepare_data`, `prepare_weight`
  and `estimate` retain separate, immutable process-local native handles for
  continuous ML/ULS/GLS/WLS/DWLS, FIML, ordinal ULS/DWLS/WLS and mixed DWLS/WLS.
  Ordinal schema augmentation and native matrix representation are prepared
  once; each dataset refreshes moments/patterns/starts. Weights are dataset-bound,
  and full categorical Gamma is optional. New FIML estimation does not compute
  H1 eagerly. Existing fit lists and numerical audits are preserved. Legacy
  entry points remain supported for compatibility and specialized families;
  see [the rollout status](../design/r-model-preparation.md) and
  [R usage](../../r-package/README.md#reusable-model-data-and-weights).

- Exploratory R bindings cover lavaanify, fitting, sample-stat bundles, robust
  inference, fit measures, model implied moments, LS estimators, SNLLS, Ceres
  paths when enabled, and data-frame-to-model sample statistics. Frontier
  parameter profile-LR wrappers expose robust scaling for ML and continuous
  GMM, including opt-in estimated-weight GLS/WLS scaling for caller-fixed GMM
  weights.
- The `magmaan_core` data/robust surface includes explicit observed-missing
  mixed continuous/ordinal builders:
  `data_mixed_ordinal_stats_observed_from_{raw,df}` for fully pairwise
  observed stats and `data_mixed_ordinal_stats_hybrid_fiml_from_{raw,df}` for
  the continuous-FIML hybrid. `robust_mixed_ordinal_ij` exposes the
  misspecification-robust IJ covariance for mixed ULS/DWLS/WLS fits using the
  same explicit `magmaan_mixed_ordinal_data` object as fitting and profile
  RMSEA/LRT.
- **Case-level influence diagnostics** (exact leave-one-out engine; semfindr
  parity) landed 2026-06-23 as pure-R `r-package/R/case_influence.R`:
  `case_rerun()` (drop one case, down-date the sample statistics, warm-started
  refit; continuous ML/ULS/GLS, single- or multiple-group — the latter by
  passing the original `data` frame, with group-label-suffixed columns), then
  `est_change_raw()` (raw or std.all DFBETA), `est_change(se = )` (DFTHETAS +
  generalized Cook's distance `gcd`, standardized by the leave-one-out
  covariance under a `standard` / `robust.sem` / `robust.huber.white` regime),
  `fit_measures_change()`, and `mahalanobis_rerun()`. It reuses existing primitives only (`fit_ml`/`fit_uls`/
  `fit_gls`, `df_to_data`, `inference_information_expected`/`inference_vcov`,
  `fit_measures`), so no core C++ was touched. Output format mirrors
  `semfindr` (Cheung & Lai, 2026; Pek & MacCallum, 2011), validated live by
  `r-package/examples/case_influence_semfindr.R` and frozen against semfindr
  fixtures (`cpp/tests/fixtures/case_influence/`, regen
  `cpp/tests/tools/regen_semfindr_fixtures.R`, pin `semfindr_version.txt`).
  Parity is to ~1e-5 for the estimate/gcd changes and machine precision for
  fit-measure changes and Mahalanobis distance. The **approximate one-step
  (no-refit) engine** also landed: `est_change_raw_approx()` /
  `est_change_approx()` via `θ̂ − θ̂₍ᵢ₎ ≈ (N/(N−1))·V·s_i`, built on a new core
  accessor `inference::casewise_scores` (the N×n_free per-case ML score matrix
  `Z_c·WΔ` that `information_cross_products` is now the Gram of; bound as
  `infer_casewise_scores_fit`, verified equal to `lavaan::lavScores`).
  `est_change_raw_approx` matches semfindr to machine precision;
  `est_change_approx` uses the *correct* finite-sample scaling and so diverges
  from semfindr 0.2.0 by two documented constant factors (semfindr applies
  `N/(N-1)` twice to DFTHETAS and once too few inside gCD — a recorded oracle
  defect, gated transitively up to those factors). `fit_measures_change_approx`
  is intentionally not provided (the exact `fit_measures_change` covers that leg
  cheaply since refits are reused from `case_rerun`; the no-refit version would
  be lossier). The multi-group and robust-regime exact variants also landed
  2026-06-23 (see above), as did the **misspecification-robust ("complete-sandwich")
  one-step** — `est_change_{raw_,}approx(fit, type = "estimated.weight")`, the
  casewise dual of the estimated-weight SE, via the new core accessor
  `estimate::continuous_ls_casewise_influence_ij` (see the estimated-weight stream
  above and `papers/estimated-weight-se`). Still in the backlog: DWLS/categorical
  and multigroup for that one-step path.
- **`fit_measures()` baseline is now `fixed.x`-aware** (2026-06-23). It calls
  `infer_baseline_fit(fit)` → `measures::baseline_chi2(pt, samp)`, which frees
  the exogenous (co)variances in the independence/baseline model exactly as
  lavaan does under `fixed.x` (baseline `df` drops by `px(px-1)/2`). Previously
  `fit_measures()` used the partable-unaware `infer_baseline(ss)`, so CFI/TLI
  diverged from lavaan for models with observed exogenous predictors; they now
  match to machine precision. A no-op for fits without exogenous variables.
- **Multiple-group order follows lavaan** (2026-06-23). The R group helpers
  (`fit_model()`, `df_to_data`, `df_to_fiml_data`, the ordinal/mixed data builders)
  derive group labels by data-appearance order (`unique(as.character(g))`),
  matching lavaan, instead of factor `levels()`. Previously a factor group
  column whose level order differed from its appearance order made magmaan and
  lavaan index groups differently. Explicit ordering remains available via the
  model spec's `group_labels`.
- **Packaging is portable** (2026-06-17). `r-package/` is self-contained: the
  C++ core plus `cpp/third_party/{port,quadpack}` is vendored into
  `r-package/src/{core,magmaan,third_party}/` by `r-package/tools/vendor-cpp.sh`
  (`just vendor`), so `R CMD INSTALL` / `remotes::install_github` builds it with
  no CMake and no prebuilt library. NLopt comes from a system install
  (pkg-config) or the `nloptr` CRAN package (bundled/self-built; `LinkingTo`/
  `Imports: nloptr`), so no system NLopt module is required (the Saga fix). The
  fast dev loop stays `just r-dev` (glue-only
  compile + prebuilt `libmagmaan.a` link via the `r-package/build-rdev/` mirror with
  `r-package/tools/r-makevars-dev`); the portable build is `just r-install`; `just
  vendor-check` guards vendor drift in `just check`. Cluster install (Saga /
  Sigma2): `r-package/tools/saga/`. This required one GCC-portability fix (R compiles with
  GCC, the canonical build uses clang): `ordinal.cpp` local lambdas renamed to
  the file's `*_fn` convention; the vendored f2c C is pinned to `-std=gnu11`.
- Composite (`<~`) model specifications are visible at the R boundary as
  folded `<~` partable rows while the hidden expanded Henseler-Ogasawara
  partable is retained as internal metadata for fitting and post-fit helpers.
  The R `composite_weights(fit, vcov)` accessor exposes recovered composite
  weights and delta-method SEs from the C++ post-fit primitive. The C++
  `BuildOptions` default is `CompositeMode::None`: core callers that accept
  `<~` must explicitly select either the historical Henseler-Ogasawara
  expansion or the native FC-SEM path. The R lavaanify boundary selects the
  historical expansion, while `api::frontier::model_from_lavaan_fcsem()` and
  the R FC-SEM helpers select native FC-SEM and require at least one `<~` row.
  R-side ordinary SEM helpers and native FC-SEM helpers reject each other's
  model/data classes and native FC-SEM partable data.frames rather than
  reinterpret a prebuilt object under the other composite semantics.
- A parallel native FC-SEM composite spec path is scaffolded behind
  `spec::BuildOptions::composite_mode = CompositeMode::FcSem`. In that mode
  `<~` rows are preserved, the first composite weight is marker-fixed by the
  ordinary auto-fix-first rule, composite indicator T-block rows and composite
  self-variance rows are stamped as fixed/derived placeholders, and both verbal
  `LatentNames::composites` and name-free
  `LatentStructure::composite_blocks` carry the composite contract. MatrixRep
  intentionally rejects native `<~` rows because native FC-SEM is evaluated by
  a separate W/T evaluator rather than the ordinary LISREL matrix path.
- The first native FC-SEM covariance evaluator exists as
  `model::FcSemEvaluator`. It assembles W and sample-backed T blocks, derives
  composite loadings, solves the derived composite disturbance variances through
  the structural system, and returns implied covariance matrices. The evaluator
  is covariance-only and separate from the ordinary LISREL `ModelEvaluator`.
  `estimate::ml_objective(FcSemEvaluator, SampleStats)` wraps it in the
  complete-data normal-theory ML discrepancy and supplies a central
  finite-difference gradient for the first native composite optimization
  tranche. `estimate::simple_fcsem_start_values` and
  `estimate::fit_ml_fcsem` provide the first low-level complete-data ML fitting
  path; the pure-composite, composite-plus-factor, and composite-structural HS
  fixtures fit from native starts and match lavaan's native `<~`
  objective/implied covariance plus the lavaan-reported weights, loadings, and
  regressions. `FcSemEvaluator::dsigma_dtheta` supplies a central
  finite-difference covariance Jacobian, and
  `inference::information_expected_fcsem` uses it for native expected
  information/vcov; the same three fixtures now match lavaan SEs for reported
  weights, loadings, and regressions. Native
  `measures::standardize::standardize_lv_fcsem` and
  `standardize_all_fcsem` use the evaluator's sample-backed W/T and total
  construct covariance semantics; the same fixture trio matches lavaan
  `std.lv`/`std.all` values and delta-method SEs for reported free rows.
  `standardized_rows_fcsem` lifts that into a lavaan-like row surface for
  native `<~`, `=~`, and `~` rows, including fixed marker rows with
  delta-method standardized SEs. Native FC-SEM df subtracts the composite
  indicator T-block moments from the user model, while the independence
  baseline remains the ordinary observed-variable baseline; `fit_extras_fcsem`
  and the existing fit-measure helper match the same lavaan fixture trio for
  chi-square, df, CFI/TLI/RMSEA, SRMR, loglik, AIC, BIC, and sample-size
  adjusted BIC. `api::frontier` now exposes a native FC-SEM model builder,
  complete-data ML fitting, expected SEs, fit measures, and standardized row
  reporting. The R frontier mirrors that slice with `fcsem_model_spec()`,
  `df_to_fcsem_data()`, `fit_ml_fcsem()` / `magmaan_fcsem()`,
  `fcsem_standard_errors()`, `fcsem_fit_measures()`, and
  `fcsem_standardized_rows()`, plus `magmaan_core$frontier_*` aliases for
  method-development workflows.
- Single-group native ML FC-SEM lavaan parity is fixture-gated for the
  pure-composite, composite-plus-factor, and composite-structural HS cases under
  `cpp/tests/fixtures/composite/`. The public-surface golden
  (`cpp/tests/golden/composite_golden_test.cpp`) fits through `fit_ml_fcsem()` and
  checks objective/chi-square, implied covariance in lavaan observed-variable
  order, df/npar, fit measures, reported raw rows, SEs, and `std.lv`/`std.all`
  rows against lavaan's native `<~` output. Native W/T matrix construction
  remains covered as evaluator/unit-test machinery rather than as a public
  fixture contract; the R bridge is still a methods-developer frontier surface,
  not a lavaan replacement interface.
- `fit_model(model, data, estimator, groups)` (named `magmaan()` before the
  2026-09-25 package split) is the high-level estimate-only
  convenience of the compiled R package `magmaanlab`. It composes `model_spec()`, data-frame sample-stat/raw-data
  construction, and the matching point-estimation wrapper for complete-data
  ML/ULS/GLS/WLS, FIML, ML2S, and ordinal/mixed DWLS/WLS where the lower-level
  inputs are available. `model_spec(meanstructure = "default")` enables means
  for grouped models; ordered declarations and explicit intercept syntax imply
  means as in lavaan. Saved specs retain both the resolved option and the
  requested default, so adding groups or ordered variables later re-evaluates
  defaults without losing explicit choices. For FIML and ML2S syntax calls it
  auto-enables a mean structure (and rebuilds syntax-backed no-mean specs) because the raw-data
  missing-data paths are mean-based; explicit `meanstructure = FALSE` errors
  early. Lavaan-style `se = "none"` and `test = "none"` are accepted as explicit
  point-estimate-only shortcuts; other values error and point users to explicit
  post-fit inference calls. It returns a `magmaan_fit` list: the raw primitive
  fit fields plus the source `model_spec`, syntax, estimator options,
  ordered-variable metadata, parameterization, and grouping metadata. Its print
  method reports only point-fit status and directs users to explicit post-fit
  primitives. `psd = TRUE` dispatches every estimator branch except two-level
  to its PSD-constrained frontier fitter and records `options$psd`.
- Lab `vcov()` exposes formula-named regimes: expected/observed inverse
  information (ML/FIML), expected/observed-bread empirical sandwiches,
  normal-theory/empirical delta covariances (non-iterative CFA), and stored SAM
  covariance. Unsupported combinations error. Omission and legacy aliases
  preserve the existing estimator-specific defaults. Raw-data methods use
  retained observations when available; FIML requires its own retained data.
  These wrappers compose existing C++ inference primitives and do not change
  the ordinary-user inference policy.
- Ordinary API adopted 2026-10-01, not implemented:
  [`magmaan_model()` then `magmaan()`](../design/r-interface-vision.md#ordinary-api)
  separates immutable native model/schema preparation from repeated fitting.
  Grouped/ordinal skeleton frames declare levels; each dataset refreshes
  moments, thresholds, weights and starts. The fit takes model, data,
  estimator, covariance policy, inference and options; structural choices
  belong to construction and optimization details to `options`. Every model
  carries a mean structure. `fixed.x`, `missing`, `cluster` and
  `meanstructure` leave the ordinary call (fixed-x rationale in the
  [scope](../scope.md#ordinary-fixed-x-decision)). Barriers are exposed
  experimentally as `covariance = barrier(lambda)` with explicit unavailable
  inference until their own gates pass. Migration and adapter work remain
  open; the following entries describe current runtime.
- The ordinary-user R package `magmaan` (`r-magmaan/`, pure R, imports
  `magmaanlab`) is a scaffold of the two-package design
  ([r-interface-vision.md](../design/r-interface-vision.md)). `magmaan()`
  takes lavaan-named options, rejects estimator-plus-correction names such as
  MLR and WLSMV, delegates `meanstructure` defaults to the lab, reports rows used
  and deleted listwise, and fits through `fit_model()`. `infer()` runs the
  inference policy (next entry) and records each component (covariance,
  global score, global LR) as available or with a typed reason; for an
  unavailable component `vcov()` and `confint()` raise a
  `magmaan_inference_unavailable` condition carrying that reason.
  Its tests check parameter rows and free estimates against lavaan for ML,
  `std.lv`, multi-group `group.equal`, syntax intercepts and ordinal DWLS.
  The 0.1.0 simulation prerelease (2026-10-01) accepts lavaan syntax and
  lavaan-backed saved specs, inheriting ordered variables, parameterization
  and grouping before validation; conflicting explicit options error. EQS
  and partable-only specs stay lab-only. Per-group row counts use the resolved
  grouping. Defined estimates are evaluated through the C++ evaluator during
  lab fit reconstruction even without inference; uncertainty remains absent.
  Confidence arguments are checked and unchecked convergence stays distinct
  from failure. Help pages and the ordinary README specify simulation
  extraction and version-pinning contracts; the matched lab dependency is
  versioned 0.1.0 too.
  Release verification: 189 ordinary-package assertions and 82 focused lab
  assertions pass; the ordinary source archive has a clean `R CMD check`.
  The compiled source archive installs without CMake or a prebuilt core,
  exercising the bundled `nloptr` fallback. Its native sources match the frozen
  release projection. Both packages are distributed as source archives for
  the local `v0.1.0` simulation prerelease; broader inference is not a gate.
  This completes the simulation packaging milestone only. Development remains
  unfinished: remaining bugs need fixing, and coverage of the main estimators
  and inferential procedures needs completing and validating before a finished
  release; the active backlog tracks that work.
- The banked [score-centering decision study](../../experiments/decisions/03-score-centering/report.qmd)
  (2026-10-01) records a fixed-allocation grouped-mean counterexample to universal
  raw likelihood-score meat. Independent confirmation (8,000 draws) and
  closed-form replay find N=300 parameter-variance ratios 1.118 raw/global versus
  0.994 within groups, against the exact 1/N target. Within-group parameter
  covariance meets the registered criteria in this scope; centered score
  calibration fails the N=80 acceptance rule (6.2% rejection, 95% Wilson interval
  5.2–7.3%). Package defaults are unchanged. The
  [sampling-law covariance formula](../scope.md#group-allocation-and-likelihood-score-covariance)
  is settled: fixed allocation uses within-group covariance; joint sampling
  retains between-group score-mean variation. The ordinary fixed-allocation
  extension is [banked](../backlog/speculative.md#fixed-design-inference-under-mean-misspecification),
  with no queued experiment to choose that formula. Promotion needs a named
  consumer and native contract/regression gates. Regular complete-data ML
  confirmation (32,000 datasets) retains the raw reference under the registered
  rule: centering changes null rejection by at most 0.75 percentage points and
  empirical matched-null power is identical, with no qualifying size-error
  benefit. All nominal-size Wilson intervals are inside 3–7%, but two global
  N=80 SB paired noninferiority bounds exceed the one-point margin. Parameter
  covariances agree numerically; skewed nested N=80 Wald null coverage is 90.3%
  (95% Wilson interval 88.9–91.5%), a separate active validation issue. Prospective
  FIML confirmation adds 32,000 fresh MCAR/MAR datasets with no qualifying
  centering benefit in either sensitivity stratum and matched-null power changes
  at most 0.10 points. Observed-sensitivity N=80 global PEBA4 rejection is
  0.8–1.8%; nested MCAR rejection rises from 6.5% raw to 7.25% centered. Both
  strata retain raw as a comparator, without selecting an ordinary FIML default.
  Stationary covariances agree, but nested MCAR N=80 null Wald coverage is 92.35%
  (Wilson interval 91.10–93.44%), failing the registered coverage condition.
  The [likelihood-score bank](../backlog/speculative.md#likelihood-score-centering-alternatives)
  owns reopening; no further generic centering comparison is queued. Absolute
  FIML calibration/interval validation and excluded boundary/penalty cases remain
  open under their existing backlog owners. Random missingness
  patterns must not inherit the fixed-group centering rule.
- The inference policy for single-level complete-data ML is
  `api::policy_inference_ml()` (`api/policy.hpp`), exposed as
  `magmaanlab::policy_inference(fit)` and run by `magmaan::infer()`. The
  parameter covariance is `robust::frontier::ntml_score_sandwich()`: the
  observed-information sandwich of exact casewise likelihood scores at the
  fitted point, so a misspecified structured mean does not bias the meat (the
  centered-moment `ntml_covariance(fit, true)` and `casewise_scores()` use the
  sample mean). The global score and likelihood-ratio tests come from the
  shared expected-information NTML geometry, each calibrated with SB and PEBA4.
  Components carry typed reasons (`not_converged`, `saturated`,
  `unsupported_model`, `numeric_failure`, `not_nested`) instead of substitute
  results. A PSD estimate on the cone boundary gets every component, flagged
  `psd_boundary`: the inference assumes an interior population (since
  2026-09-26; before, it was refused). Nested tests are
  `api::policy_nested_ml()`, exposed as `magmaanlab::policy_nested()` and
  `magmaan::anova()`: the likelihood-ratio difference and the efficient score
  at the restricted fit, each with SB and PEBA4, for a restricted model that
  drops, fixes or constrains alternative paths. The shared inference-side
  `robust::embed_nested_null` matches formula keys including group and level,
  lifts the null's affine constraints and estimate into H1, and verifies its
  implied covariance and mean moments to relative tolerance 1e-10. Omission
  defaults to zero only for loadings, regressions and off-diagonal covariances;
  intercepts, means, variances, thresholds and scales need resolved rows.
  The model triple (`LatentStructure`, `LatentNames`, `Starts`) and its
  partable projection are unchanged. Score inference uses H1 derivatives at
  the embedded null point, with the lifted null nuisance tangent. ML/FIML
  moment reparameterizations use a damped analytic moment fit of H1 to H0's
  fitted moments and both Jacobians at that solution; tangent inclusion and
  interior rank/covariance checks are required. Other estimators support the
  key embedding and return typed `unsupported_nesting` for unavailable
  correspondences. Singular tangent/factor-covariance nestings return
  `boundary_nesting`. The shared map also feeds complete-data/FIML/ML2S exact
  Satorra-2000, ordinal/mixed-ordinal, pairwise-composite, continuous weighted
  inference and the SB2010 null-point injection. Mixed-point delta remains
  the lavaan compatibility option. SB matches lavaan's
  `satorra.2000` with the exact restriction map. A model without free
  parameters is evaluated at its fixed values instead of optimized
  (`evaluate_fixed` in `estimate/fit.cpp`), passes the verdict vacuously, and
  gets an empty covariance. Its global score and LR tests use the full moment
  space: the zero-direction expected-information projector is identity before
  normal-theory whitening, and fixed means retain their moment rows even with
  no parameter columns. The ML objective also includes the fixed-mean
  discrepancy when its mean Jacobian has zero columns. Independent
  saturated-normal formulas and empirical
  spectrum calculations gate covariance-only/mean-structure fits, single and
  unequal-sized multiple groups, and casewise/tiled storage; saturated models
  still have no global test. Ordinary R direct/deferred inference is covered,
  including LR discrepancy parity against lavaan's fitted ML objective (lavaan
  suppresses global tests for zero-free-parameter models). `fitted()` in
  `magmaan` returns model-implied moments, so such a model gives a population's
  moments. C++ unit tests check the
  covariance against finite-difference casewise scores, including two-group
  scalar invariance with misfitting means; the R tests match lavaan's MLR
  standard errors (single- and multi-group with structured means), its
  Satorra-Bentler statistic, scaling factor and p-value, and its delta-method
  standard errors for defined parameters.
- `compute_defined(model, fit, vcov)` exposes C++ defined-parameter evaluation
  for `:=` rows through R. It keeps covariance selection explicit, supports
  chained definitions, and resolves `.pN.` plabel references using the fitted
  lavaanified model.
- Friendly R post-fit wrappers expose routine inspection without hiding
  statistical choices: `standardized(fit, vcov, type)` requires an explicit
  covariance matrix, `stats::residuals(fit, standardized)` / `lav_residuals(fit)`
  wrap raw or standardized residuals including the lavaan-style continuous
  residual z-statistics and the per-block `$summary` table (the
  `lavResiduals(fit)$summary` analogue: cor.bentler SRMR/USRMR with SE,
  exact-fit and close-fit z-tests, and a close-fit CI for cov/mean/total,
  the raw-metric residual ACOV congruence-scaled into the correlation metric),
  with an `estimated_weight = TRUE` frontier route
  (`measures::frontier::standardized_residuals_estimated_weight`) that swaps the
  NT residual ACOV for the Hall-Inoue infinitesimal-jackknife sandwich so the
  residual SE/z and `$summary` reflect an estimated continuous-LS second-stage
  weight (beyond lavaan; bootstrap-calibrated under non-normality where the NT
  residual SE is anti-conservative),
  `factor_scores(fit, data, method)` requires complete raw data
  and dispatches continuous regression/Bartlett vs ordinal/mixed EBM/ML/EAP
  according to the fitted data type, and `modification_indices(fit, data,
  candidates)` / `score_tests(fit, data)`
  forward to the explicit scaffold primitives. Ordinal and mixed-ordinal fit
  objects retain the categorical stats object used for fitting, so ordinary
  categorical MI/score calls work directly; callers may still pass the stats
  object explicitly.
- R post-fit inference helpers now separate primitive-shaped entry points from
  fit-list adapters for the first audited slice: vcov, z tests, Wald tests,
  RLS chi-square, U-factor construction, and robust SE helpers can be called
  with explicit `partable` / `sample_stats` / `theta` pieces, while existing
  fit-list calls remain available with explicit `*_fit` aliases.
- The R sample-moment path accepts `list(S = , nobs = , mean = )`, reorders
  named covariance matrices to model observed-variable order, and rejects
  malformed group counts, non-square or wrong-sized covariance matrices,
  non-finite moments, nonpositive `nobs`, and wrong-length means before calling
  C++ fitters.
- Future public API direction is a staged, explicit workflow rather than
  lavaan-style compound estimator strings. Users should build a model, build
  data or sample moments, fit point estimates, and then explicitly request
  standard errors, robust corrections, test statistics, fit measures, defined
  parameters, or summaries. High-level objects may retain model/data/fit
  context for ergonomic chaining, but each statistical choice must remain
  inspectable.
- The planned C++ facade sits above the existing primitive namespaces as value
  objects such as `Model`, `Data`, `Fit`, `Analysis`, and `Summary`. It may
  support fluent usage, while the primitive namespaces remain the
  methods-developer surface. Convenience recipes can be added later only if
  they expand to visible choices such as estimator, moment builder, standard
  error method, test statistic, and fit-measure inputs.
- Bindings should keep the same split between a small friendly surface and an
  inspectable primitive layer. Python can expose a friendly top-level API plus a
  `magmaan.core` submodule for C++-shaped primitives. R should avoid exporting
  every primitive into the package namespace; instead, the friendly staged API
  should be exported directly and low-level functions should live behind a
  single `magmaan_core` object.
- The staged C++ facade and R `magmaan_core` surface expose the experimental
  robust ordinal moment builders explicitly: all-ordinal h-weighted,
  all-ordinal DPD, mixed continuous/ordinal fixed-marginal polyserial DPD, and
  mixed continuous/ordinal Huberized residual stats. The staged C++ facade also
  exposes all-ordinal Huberized residual stats. Default ordinal and mixed data
  builders remain lavaan-compatible ML paths.
- The R `magmaan_core` surface also exposes mixed continuous/ordinal covariance
  shrinkage as an explicit `data::frontier` transformation, rebuilding mixed
  moments, `NACOV`, and DWLS/WLS weights before fitting rather than hiding
  shrinkage inside estimator strings.
- The R package is intended as a methods-developer interface over the C++
  library, not a second SEM implementation.
- Standard errors, information matrices, Wald/z tests, robust corrections,
  fit measures, defined parameters, and nested tests remain explicit post-fit
  calls outside `fit_model()`.
- Primary public declarations and internal implementation now live in the
  target namespaces: `parse`, `spec`, `model`, `data`, `estimate`,
  `inference`, `robust`, `measures`, `sim`, `optim`, and `compat::lavaan`. Repository
  code and R binding internals use those namespaces directly.
- The `optim` namespace owns optimizer interfaces/backends, terminal audit
  helpers, and the equality-constraint reparameterization transforms that map
  theta-space scalar/GMM problems into constraint-reduced alpha coordinates.
  Constraint construction still lives in `estimate` until the broader
  constraint-header retiering lands; the old `estimate/reparameterize.hpp` path
  is a forwarding shim.
- Old `fit/*` and `partable/*` compatibility headers have been removed; use the
  target namespace headers directly.

### Local build workflow

- The normal C++ edit loop is the `fast` preset and `just test-fast` / `just
  test`: Debug, no sanitizers, Ceres off, with ccache and mold enabled.
- The `dev` preset is the sanitizer validation loop: Debug plus ASan/UBSan.
  Use `just test-dev` before handing back risky core changes, and prefer
  targeted `ctest --test-dir cpp/build/dev --output-on-failure -R ...` while narrowing failures.
- The C++ doctest suite is split into labeled executables (`smoke`, `spec`,
  `estimate`, `inference`, `ordinal`, `api`, `parity`, and `robcat`) behind
  the aggregate `magmaan_tests` target. This keeps
  `cmake --build --preset <p> --target magmaan_tests` working while allowing
  narrower relinks and `ctest -L`.
- R bindings are intentionally outside the default C++ loop. The fast dev loop
  is `just r-dev` (glue-only compile, prebuilt-core link); `just r-install` is
  the portable self-contained build (vendored core, system NLopt);
  `just r-install-ceres` / `just r-install-ipopt` are the explicit backend
  dev paths (now aliases over `just r-dev`).

#### Build-loop timings

Fast-loop snapshot refreshed on 2026-05-19 (commit `97dd197`), on a 13th-gen
i7-1355U (12 threads), clang 19.1.7, with ccache and mold enabled.
Wall-clock; approximate orientation, not a benchmark. Rows marked
"not remeasured" are older orientation values retained until the corresponding
loop is refreshed.

| Loop step (`just` recipe)                   | Time   | Notes |
|---------------------------------------------|--------|-------|
| no-op `fast` build (`just fast`)            | 0.03 s | nothing changed |
| touched core TU, `fast`                     | 1.2 s  | not remeasured; mtime only, ccache hit, relink `libmagmaan` + test exes |
| edited core TU, `fast`                      | 7.5 s  | not remeasured; content change, cold clang compile of one core TU |
| touched test TU, `fast`                     | 0.5 s  | not remeasured; mtime only, ccache hit, relink one test exe |
| edited test TU, `fast`                      | 5.6 s  | not remeasured; content change, cold compile of one test TU |
| C++ suite (`just test-fast`)                | 81.5 s | 458 tests; no-op build + `ctest` |
| C++ suite minus parity (`just test-quick`)  | 37.4 s | 454 tests; excludes the 4 parity tests |
| sanitizer suite (`just test-dev`)           | 287 s  | not remeasured; old ASan/UBSan orientation value |
| R install, opt (`just r-install`)           | 12 s   | not remeasured; warm core; rebuilds the 5 R-glue TUs + link |
| R install, fast (`just r-install-fast`)     | 15 s   | not remeasured; warm core |
| R install, Ceres (`just r-install-ceres`)   | 123 s  | not remeasured; included a one-time post-refactor rebuild of the Ceres core; ~15 s once warm |

The everyday inner loop (edit a core file, `just test-quick`) is comfortably
under a minute on the measured fast tree; the sanitizer suite and the Ceres R
path are minutes-scale and run deliberately rather than on every change.

### Argument-minimality sweep

One core architecture rule is that functions should take exactly the data they
need, and no more. The current C++ core mostly follows this:

- Fit results are plain `Estimates` and carry no back-pointer to the model.
- Continuous fitters take the estimable structure, matrix representation,
  sample/raw data, starts, discrepancy, and optimizer. They do not take
  parameter names.
- Information, covariance, SE, chi-square, degrees of freedom, Wald/z tests,
  robust U-Gamma reducers, both casewise-Zc and caller-supplied-Gamma robust
  test reductions, both-bread robust test trace-moment reducers, the
  materialized empirical-Gamma reference reducer, and fit measures are exposed
  as separate primitives rather than as methods on one fitted-object bundle.
- Satorra-Bentler-family reducers take scalar test statistics, degrees of
  freedom, and eigenvalues or trace summaries rather than fit objects.
- The Satorra-2000 C++ helper takes the model, reparameterization matrices,
  raw-data pieces, chi-square values, degrees of freedom, and an explicit
  `A.method` option (`exact` default, `delta` for lavaan-style
  moment-Jacobian column-space restrictions). Its empirical-Gamma computation
  can run in the default streaming casewise-reduced mode, in a materialized
  full-Gamma reference mode, or in a dense full-`UΓ` diagnostic mode for
  benchmarks. The same computation flag is honored by the FIML lavaan-
  convention restriction-map path: streaming projects through the lavaan-style
  expected `WLS.V` columns at the fitted larger model before contracting the
  retained saturated EM ACOV, without materializing another full Gamma.
  Materialized builds each `n_g * acov_g` block and dense eigendecomposes the
  full moment-space product. The numeric kernels fail closed on non-finite or
  singular pencils:
  model-implied covariance blocks, the pooled expected-information `P`, and
  restriction companion `C` are finite/SPD-gated; reduced meats are
  finite/PSD-gated; the FIML sandwich route allows finite nonsingular observed
  bread that is indefinite in finite samples, but still rejects singular bread
  and degenerate `C` before any generalized eigensolver is called. Continuous
  pairwise expected-bread U-factors also avoid materializing `Gamma_NT^pw`: the
  raw/pairwise `build_u_factor` overload builds the residual subspace from the
  model Jacobian, applies the pattern-grouped pairwise NT metric as an operator,
  and whitens only the reduced `df x df` Gram matrix. Observed-bread pairwise
  inference still materializes the pairwise NT metric because that
  representation stores `A = L_Gamma^{-1} Delta`.

The remaining pressure point is the R boundary. Several exported R wrappers
currently accept the transparent fit list for convenience and then unpack the
partable, sample statistics, and estimates internally. That is acceptable as an
interactive convenience while the R package is exploratory, but it is not the
final architectural ideal. Thin R wrappers should gradually mirror the C++
primitive signatures; fit-list helpers can remain as explicit convenience
adapters layered on top.

### Robust-test naming and compatibility

The core robust-test API should use statistical names for the object being
computed, not historical SEM estimator labels. The preferred core surface is:

- weighted chi-square mixture tests from explicit eigenvalues or trace
  summaries;
- moment reductions of that same mixture: mean-scaled, mean/variance-adjusted,
  and scaled/shifted;
- likelihood-ratio / nested-model helpers named for the statistical contract
  they implement, for example exact restriction-map LRT rather than a default
  public name centred on "Satorra-2000";
- robust SEs as named sandwich constructions.

Historical labels remain important for lavaan/Mplus parity, but they belong in
`compat::lavaan` or similarly explicit compatibility wrappers. This includes
estimator shortcuts such as `MLM`, `MLMV`, `MLMVS`, and `MLR`, lavaan test
labels such as `satorra.bentler`, `scaled.shifted`,
`mean.var.adjusted`, `yuan.bentler`, and `yuan.bentler.mplus`, and legacy
nested-test formulas such as Satorra-Bentler 2001/2010. The compatibility
wrappers should bind each label to the exact lavaan/Mplus bundle it denotes;
they should not expose a combinatorial menu that lets callers freely pair a
test label with unrelated bread, gamma, vcov, or scaling choices.

`yuan.bentler.mplus` is especially a compatibility target rather than a core
weighted-mixture reducer. Lavaan's MLR default computes a scalar trace
difference, H1 minus H0, relying on Mplus-style approximations such as
`A0 ~= Delta' A1 Delta` and the analogous first-order matrix. The plain
Yuan-Bentler formula instead constructs the H1 residual-space trace directly.
Both should be documented when exposed, but neither should force historical
names into the core naming scheme.

The R boundary now stages this policy with `robust_nested_lrt()` as the
friendly statistical name for nested robust likelihood-ratio work. Its default
`method = "restriction_map"` names the exact restriction-map contract, while
`"lavaan_sb2001"` and `"lavaan_sb2010"` are explicit compatibility methods.
The restriction-map route also exposes `computation = "streaming"` versus
`"materialized"` so paper-local benchmarks can compare the algebra without
using lavaan as the timing denominator. When both fits are FIML,
`robust_nested_lrt()` dispatches only to `method = "restriction_map"` and uses
the retained FIML `raw_data` rather than a caller-supplied complete-data
argument; mixed FIML/complete-data pairs and the lavaan SB2001/SB2010
compatibility methods are rejected for this boundary.
The complete-data ML restriction map handles mean structure: when either fit
carries free intercepts / latent means the per-group moment vector is augmented
to `[μ_g; vech(Σ_g)]` (mean rows on top), so `lr_test_satorra2000_from_data`
stacks `dmu_dtheta` over `dsigma_dtheta` for both the per-group `Pi_alpha` and
the delta restriction, and `compute_satorra2000` carries the augmented block
through `SatorraGroup.mu_dim` (μ-block of `V` is `Σ⁻¹`, the casewise meat picks
up the full μ×σ cross-block). This lifts the former covariance-only limitation
(mean-structured pairs used to error with a rank-deficient pooled `P`); it
matches lavaan's mean-structured `lavTestLRT(method = "satorra.2000")` and is
validated by `cpp/tests/unit/satorra2000_test.cpp` (augmented streaming/materialized/
dense agreement, an independent full-UΓ oracle, and the NT collapse) and by
`r-package/examples/nested_test_satorra2000.R` (meanstructure configural/metric
and a genuine intercept-invariance restriction vs lavaan).
The older `nestedTest()` spelling and lavaan labels (`"satorra.2000"`,
`"satorra.bentler.2001"`, `"satorra.bentler.2010"`) remain as compatibility
aliases during exploration. Low-level `magmaan_core` exposes both
`robust_nested_lrt_restriction_map` and `compat_lavaan_nested_lrt_*` aliases
over the same C++ primitives so methods scripts can choose the intended
surface directly.

### Testing and validation

Validation has three deliberately separate surfaces:

- **Corpus golden tests** — breadth. 26 small synthetic models in
  `cpp/tests/fixtures/corpus.json` exercise every parser, lavaanify, matrix, fit,
  and inference stage against checked-in lavaan fixtures. Oracle:
  `cpp/tests/tools/regen_oracle.R`.
- **Parity golden tests** — depth. `cpp/tests/golden/lavaan_parity_golden_test.cpp`
  gates magmaan against lavaan on the real-data benchmark cases
  (HolzingerSwineford1939, PoliticalDemocracy, Demo.growth, bfi, Mplus ex5.1):
  real sample sizes, real conditioning, real SEs and fit measures, and the
  raw-data ingestion path. The FIML tranche includes both a quick two-factor
  bfi missing-data sentinel with robust MLR post-fit reporting and a full
  25-item five-factor bfi missing-data convergence/global-fit gate.
  Self-contained fixtures live in
  `cpp/tests/fixtures/parity/`; oracle: `cpp/tests/tools/regen_parity_fixtures.R`.
  The same parity executable also includes the Mplus SEM corpus golden,
  generated from local `external/textbook-corpus/raw/mplus_sem` into
  `cpp/tests/fixtures/mplus_sem/`.
- **Benchmarks** — `benchmarks/` fits *live lavaan* on every run and gates
  timing, not CI correctness. Active advisory cases include complete-data ML,
  controlled-missingness FIML, and continuous ULS/GLS smoke paths. The harness
  is R-dependent and advisory; the parity layer is the bridge that freezes its
  correctness checks into the gated C++ suite.

The suite builds as eight doctest executables — `magmaan_test_{smoke, spec,
estimate, inference, ordinal, api, parity, robcat}` — so areas build and run
independently (`ctest -L parity`). CI never invokes R; fixture regeneration is
a manual developer step. Property and boundary tests are expected to catch
structural mistakes early, before they surface as hard-to-debug parity
failures.

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
