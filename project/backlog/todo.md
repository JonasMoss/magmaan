# magmaan TODO

- **Closed 2026-09-30 — research estimator pruning.** Removed ML ridge
  continuation, automatic marker/std.lv fitting, adaptive composites, MI4,
  empirical-Bayes DLS selection, fitted-weight GMM/profile/audit adapters,
  PNTML and the dedicated GLSpw fitter. Pairwise moments and their inference
  primitives remain estimator-independent MCAR inputs; ordinary fixed-weight
  fitting composes them without a special estimator label. General user-facing
  pairwise moment/inference routing belongs to the estimator-axis work below.

**Research collection reviewed 2026-09-30.** The
[experiment index](../../experiments/README.md#research) owns study activity,
local-evidence limits and reopening triggers: 16 active, 16 banked, 9 retained
evidence studies. The polychoric-omega coverage/stress arms share research/22;
construction/illustration predecessors 07, 33, 36 and 45 are archived with their
sources/results. Detailed method-development entries below do not automatically
queue a banked simulation. Protected paper pipelines and frozen metadata remain
intact; this reorganization does not change the library's inference policy.

## Next: barrier fitting normalization

- [ ] **L — extend the shared ML/PSD normalization to barrier fitting.** This
  is the first outstanding implementation task (requested 2026-09-27). Transport
  the model, automatic/user starts, linear equalities and group structure through
  the full barrier fit. Establish how every supported penalty, its strength and
  continuation schedule transform so a change of measurement units does not
  change the statistical criterion. Return estimates in caller units; align
  penalty derivatives, optimizer controls, accuracy/stationarity checks and
  applicable post-fit calculations with that contract. Validate same-point
  criterion/derivative equivalence, boundary/pole behavior, and complete fits
  under uniform/mixed units, including groups and linear constraints. Preserve
  explicit markers and retain every regression. Reuse the complete machinery
  programme below; do not implement another experiment-only normalization wrapper.

## Remaining normalization machinery beyond complete-data ML/PSD

**The single-level complete-data ML/direct-PSD default decision is closed.**
Normalization stays enabled, including groups, means and affine linear equalities;
see the [scope contract](../reference/optimizer-controls.md#complete-data-ml-and-psd-sample-normalization-2026-09-27).
The programme below concerns remaining objective families and broader validation,
not another decision about that default. Arbitrary linear inequalities are not
covered. Direct PSD remains the route; independent two-stage API adoption is a
separate outstanding item. Barrier stays first above.

- [ ] **L — shared end-to-end normalization (requested 2026-09-27).** Build
  one reusable model/data coordinate transformation for ordinary ML, direct PSD,
  ordinary-then-PSD fallback, and barrier fitting. Cover each route's supported
  data/model slices; completion is not a single-group PSD accuracy check or an
  experiment-only wrapper. For each remaining objective family, implement and
  validate the transformation before enabling it by default. Complete-data ML/PSD
  already satisfy the documented implementation slice and default decision.
  - **Model and identification:** normalize observed variables using sample-based
    units; transport latent units, free parameters, fixed values, structural
    cells, means/intercepts, equal labels, general linear equalities (including
    nonzero constants), and explicit parameter bounds. Preserve the user's
    markers, identification and model in the returned result. Nonlinear
    equalities remain outside this task's supported scope. Check rank and
    round-trip equivalence; never silently drop or alter a constraint.
  - **Groups and data:** include single-group and multigroup models, unequal
    group sizes and cross-group equalities. Cover complete-data and supported
    missing-data ML routes, transforming raw data, observed patterns, sample
    moments and fixed-x inputs consistently. Specify and test the coordinate
    contract for supported multilevel routes too; track any remaining unsupported
    slice explicitly rather than describing partial coverage as complete.
  - **Whole fitting path:** construct automatic starts in normalized coordinates,
    transport user starts, and use consistent units for covariance repairs,
    floors/clamps, preconditioning, gradients, optimizer tolerances, constraint
    residuals and stopping checks. Share the transformation across ordinary,
    PSD and barrier stages, including fallback/warm-start/polish transitions;
    avoid double normalization. Preserve explicit start and optimizer choices;
    automatic marker changes and multistart are separate decisions.
  - **PSD and barrier semantics:** preserve the covariance domain and transport
    lifted/Cholesky coordinates. Establish the transformation of each supported
    barrier penalty, its strength and continuation schedule: equivalent fits must
    optimize the same statistical criterion, allowing only accounted-for additive
    constants. Do not silently change regularization by changing measurement units.
  - **Outputs and checks:** return estimates, implied moments, partables and
    reusable derivatives in the requested coordinates. Make accuracy, stationarity,
    admissibility and boundary checks consistent with the fitting representation.
    Preserve the contracts of applicable information, covariance/SE, score and
    fit-statistic calculations, including the needed derivative transformations
    and cached artifacts. Expose effective normalization consistently through the
    C++ and R entry points and retain useful failures when scaling is unavailable.
  - **Validation and adoption:** test uniform and mixed units, groups with very
    different scales, marker/std.lv identification, constrained/fixed-value and
    mean models, interior/boundary points, poles and inaccurate/invalid points.
    Separate same-point objective/penalty/derivative/constraint/round-trip checks
    from complete fits; assess changed convergence and local solutions, not just
    normalized residuals. Preserve the existing accuracy thresholds. Use the saved
    difficult cases as development controls and fresh held-out draws under
    precommitted criteria for any default change; retain every regression and
    compare direct, fallback and barrier routes separately.
  - **Delivered slice (2026-09-27):** complete-data ordinary ML and PSD ML now
    use the shared normalized model in optimization, including single/multigroup
    models, means, fixed/structural cells, equal labels and linear equalities.
    Staged C++/R automatic starts, original-unit explicit starts/hints/bounds,
    fallback warm starts, returned estimates, and ambient/PSD Newton checks use
    the same transformation. The effective behavior is exposed by
    `sample_normalized`, with `normalize_sample=false` retaining the former
    fitting path. This implements the user's requested ML/PSD fitting change;
    FIML, barrier, multilevel and the broader validation programme above remain.
  - **Paired development revisit (2026-09-27):** 1,356 saved scaling cases,
    including linear equalities and means, compared with normalization on/off
    under the same current judge. Direct PSD has zero unit-dependent verdict,
    objective or covariance differences; equality-family successes rise
    167/180 → 177/180. ML is roughly unchanged; fallback remains mixed.
    Preserve the six direct/five ML/eleven fallback convergence losses and the
    one direct/two unexplained fallback objective losses as regression targets.
    Twenty-three other apparent fallback objective losses compare against
    old accepted but indefinite covariance solutions under mixed units.
    **Remaining:** make caller-unit covariance admissibility auditing robust
    to heterogeneous scales (including normalization-disabled fits and
    standalone audits), and investigate constrained PSD finalization failures.
    Full per-family evidence is in the optimizer-defaults report's paired
    fitting revisit; this is development evidence, not a held-out route choice.
  - **Starting evidence:** the unconstrained normalization pilot improves unit
    consistency but sometimes finds worse local solutions. Normalized PSD Newton
    accuracy first covered single-group complete-data ML with linear equalities.
    These are starting components, not completion of the full programme. The detailed
    pilot evidence remains under Optimization below.

Repository reorganization completed 2026-09-24: C++ under `cpp/`, public
maintainer material under `project/`, R build tools alongside the package, and
external source collections under `external/`. **Deferred:** regenerate the
paper corpus from its sources, validate derived cases and exports in its own
repository, then refresh magmaan fixture snapshots and run parity checks.
The reorganization deliberately preserves existing fixture contents. Experiment
classification and category-folder moves are complete; deeper cleanup and
further archive decisions remain deferred (see `experiments/README.md`).
Category-local numbering starts at 01; archived studies have unnumbered slugs.

- [ ] Focus the engineering inventory on live implementation/default decisions:
  state each current choice, alternatives, acceptance criteria, latest evidence,
  and reopening trigger; retire settled studies after preserving their contracts.
  Review inherited lifecycle labels rather than assuming they are current.
  Research can remain exploratory; preserve paper-supporting work and results.

Remaining-work backlog. Current state, architecture, and contracts live in
[project/architecture/roadmap.md](../architecture/roadmap.md); this file only tracks
unfinished work. Completed items are folded into the roadmap (capabilities) and
[project/validation/test_ledger.md](../validation/test_ledger.md) (regression notes
for fixed cross-subsystem bugs), not kept here. May-never-build ideas live in
[speculative.md](speculative.md).

Effort tags: **S** bounded project/fixtures/wrapper cleanup · **M** focused
implementation or test slice · **L** new estimator plumbing or cross-module
semantics · **XL** statistical design/research track before implementation.

## Estimator priority programme

Follow the [development tiers](../architecture/roadmap.md#estimator-development-priorities)
adopted 2026-09-27. Primary classical (NTML, FIML, ordinal/mixed DWLS) and
priority frontier (PSD and barrier) share the main development programme.
Effort tags describe work size, not priority. The thematic sections retain the
detailed tasks; their order does not put every classical estimator ahead of
frontier work.

- **S/M — inventory actual capabilities.** Audit public C++ and R entry points
  and their validation evidence. Record estimator/data slice, covariance domain,
  penalty, algorithm, development tier and API status. Track estimation,
  convergence/admissibility diagnostics, parameter covariance, global tests,
  nested tests and intervals separately, including ordinary-policy availability.
  Distinguish implemented and validated, implemented with limited validation,
  planned, unsupported and inapplicable. Use one inventory with links to evidence,
  not another roadmap; automate its checks when the schema is settled.
- **Primary workflows first.** Close the complete-data NTML policy gaps below
  (including fixed-x and fully specified models), bring FIML policy inference
  forward, and prioritize ordinal/mixed DWLS within the least-squares policy,
  nested-test and weight-influence work. Include interface consistency,
  reliability and performance in each workflow's completion criteria.
- **PSD and barrier alongside those workflows.** Continue the active barrier
  decision work and PSD route follow-ups under optimization. Complete their
  applicable NTML, FIML and DWLS estimation/inference combinations first;
  establish estimator-specific penalty and inference contracts before claiming
  new support. Then expand to other applicable estimators. Existing broad PSD
  fitting coverage does not establish corresponding inference coverage.
- **Secondary and research work.** Preserve existing correctness gates; extend
  secondary classical workflows for a concrete use or low-cost shared benefit.
  Research methods retain their explicit supported slices. Revisit the
  provisional ML2S/pairwise assignments and two-level expansion priority when
  there is a concrete downstream need.
- **Incremental surface cleanup.** Use the decomposition task under
  [API and R boundary](#api-and-r-boundary) to separate domain and penalty from
  algorithms. Prioritization takes effect now; namespace moves, entry-point
  renaming and default changes are separate work.

## Two-package R interface

Adopted 2026-09-25: an opinionated pure-R `magmaan` package (one call,
automatic inference, `r-magmaan/`) over the compiled `magmaanlab` package
(`r-package/`). The split and the `magmaan` scaffold landed the same day; the
design, starting-point inventory and validation rules are in
[r-interface-vision.md](../design/r-interface-vision.md). Tasks are grouped below;
execution follows the estimator priority programme above.

- **Nested comparisons: `anova(fit0, fit1)` — DONE for complete-data ML
  (2026-09-26), with open parts.** `api::policy_nested_ml`,
  `magmaanlab::policy_nested()` and `magmaan::anova()` give nested LR and score
  tests with SB and PEBA4 on the shared expected geometry. SB matches lavaan's
  exact-map Satorra (2000). The geometry is in the design doc. Open:
  - **M — misspecified larger model.** The expected geometry assumes the larger
    model is correct. Under misspecification (the usual invariance-testing
    case) the choice of sensitivity is first order; specify it (observed
    sensitivity?) and check it before claiming calibration there.
  - **M — restrictions written as fixed values.** `0*` in the restricted model
    changes the parameter slots, so `anova()` reports "not nested". Recognize
    it by embedding the restricted fit in the larger model's slots (the value
    becomes an equality constraint). Until then users write `b == 0` on a
    labeled parameter.
  - **Calibration evidence for the nested score test — first study landed
    (2026-09-28).** `experiments/research/evidence/52-robust-calibration-battery` has
    four natural restrictions (df 1-18: lagged residual covariances, structural
    paths, growth residual variances, a cross-group path), normal,
    Vale-Maurelli, IG and 5-point discretized data, n = 100-1000, and 5,000
    replications per cell. Score SB rejected 3.6-5.8% at nominal 5% in all 48
    cells; score pEBA4 rejected 2.8-5.5%. Still open: a misspecified larger
    model (above) and power. The covariance-honest paper's interior-inference
    rerun remains the natural second study.
  - **M — the LR half of the policy over-rejects at small n with many df.**
    In the same study, LR-SB and LR-pEBA4 global tests reached 20.8% and 15.8%
    at n = 100 (18-indicator CFA, df 102, normal data), and nested LR-SB
    reached 17.7% (growth, IG). Score arms stayed near 5% in the same cells.
    Decide whether the policy keeps LR as a co-primary test, reports it with
    a small-sample warning, or leads with the score test. Global fit under
    IG-type heterogeneous spectra is unsolved for every arm (best: score
    pEBA4, worst cell 12.3%).
  - Other estimators follow the policy's extension (the L item below).
- **S — the test of a fully specified model.** Since 2026-09-26 a model with
  no free parameters is evaluated at its fixed values (lavaan does the same),
  so `fitted()` gives a population's moments. Its global tests report
  `unsupported_model`: the NTML geometry assumes at least one model direction
  and crashed on none. Build the zero-direction geometry (the U factor is the
  identity), then the policy can test the fully specified model.
- **M — likelihood-ratio intervals: `confint(fit, test = "lr")`.** The value
  is reserved (2026-09-26).
  The engine exists in the lab: `frontier_profile_lrt_ci_parameter_ml()` (and
  the FIML, ML2S, GMM and ordinal variants), with a Satorra-scaled robust
  option, which is Falk's (2018) robust LR interval.
  - Decide the policy's robust calibration of the profiled statistic. The
    funLR notes say small-sample correction is the open problem.
  - Exploratory normal comparison implemented in
    `research/50-normal-parameter-intervals`: 600 datasets, 4,800 ordinary
    intervals, validated score/LR endpoint inversion, and candidate-specific
    bootstrap Bartlett feasibility and endpoint-noise checks. The N=100
    correlation favors score/LR over raw Wald in this pilot; this is not a
    robust-inference default decision. `research/51-robust-parameter-intervals`
    now implements the robust comparison (3,000 datasets, 54,000 interval
    attempts), validated score/LR inversion, failure accounting and paired
    common-valid comparisons. Its uncentered-score correction differs from
    the centered empirical contributions in the landed profile helpers;
    the relationship is validated, but choosing the policy convention remains
    open. The skewed N=100 loading still undercovers (88.6–90.8% across
    primary methods), and domain/search failures remain. The study changes no
    default and adds no exported score-CI API.
  - Price it: each interval needs a root search over constrained refits, for
    every parameter.
  - Choose the default by a decisions study (Wald or LR coverage and
    non-convergence on held-out families, including boundary-near variances
    and correlations) before changing it.
- **M — least-squares estimators under the policy (DWLS first).** For fixed-weight GLS, ULS,
  WLS and DWLS the global score statistic against the saturated model equals the
  fit-function statistic; report it once with SB and PEBA4 from the policy
  geometry. Include the weight-estimation influence in the covariance for
  GLS, WLS and DWLS. The machinery exists (C++
  `robust::robust_weighted_moment_ij`; R `robust_ordinal_ij` and
  `robust_mixed_ordinal_ij`); wire it into the policy for each estimator.
- **L — nested score test for the least-squares estimators.** `score_flip_test()`
  and `nested_score_test()` accept ML and FIML only. Construct the test from
  the larger model's gradient at the embedded smaller-model estimate, with the
  nuisance projection and the restriction spectrum for SB and PEBA4, for
  continuous and ordinal/mixed fits. Specify the nested geometry (sensitivity,
  metric, evaluation point, moment covariance, centering, normalization) before
  coding, including the misspecified-larger-model case.
- **M — experiment: weight choice in the score tests.** Complete-data ML is
  settled by experiment _archive/complete-ml-global-test-geometry: keep expected information throughout.
  Remaining: the least-squares estimators. Compare score tests
  built with the estimator's own weight against alternative weights (for
  example the normal-theory weight at the fitted model, or the full ADF weight
  for DWLS fits), global and nested, on size and power under non-normal and
  misspecified populations. With a weight other than the estimator's, the
  nuisance score does not vanish at the estimate, so every arm must use the
  effective (nuisance-projected) score; `score_components_from_matrices()` and
  `project_scores()` already carry the projection. Include the
  observed-information U for the global tests as an arm. The policy default
  cites the result.
- **S — record listwise deletion in the lab fit.** `magmaan()` reports rows
  used and deleted per group from the input data frame and the fit's `nobs`.
  Store the deleted-row count, per group and per reason, in the lab fit
  instead, so every entry point reports it. List any estimator path that
  cannot delete listwise as its own item here.
- **S — continuous WLS (ADF) in `fit_model()`.** It requires an explicit `W`;
  default it to the empirical-Gamma weight the prepared path already builds
  (`prepare_weight(data, "WLS")`). `magmaan()` refuses continuous WLS until
  then.
- **Done 2026-09-25 — lab-owned `meanstructure` defaults.** `model_spec()`
  resolves grouped, ordered and intercept-syntax defaults; staged grouping or
  ordered declarations re-evaluate omitted defaults while retaining explicit
  choices. `fit_model()` supplies the FIML/ML2S requirement. `magmaan()` delegates
  to the lab. R regressions compare mean rows and grouped estimates with lavaan.
- **L — extend the policy to FIML, ML2S, ordinal/mixed and two-level fits,**
  starting with FIML under the decided geometry (observed-H0 sensitivity,
  expected metric; score and LR each with their own SB and PEBA4 spectra, both
  reported),
  with a component-level capability table. Unimplemented components report a
  typed reason; the fit is never refused for missing inference.
- **M — fixed-x models in the policy composer.** The shared NTML geometry
  (`prepare_ntml_fit()`) requires random X, but `magmaan()` defaults to lavaan's
  `fixed.x = TRUE`, so models with exogenous observed covariates get no policy
  inference. Extend the geometry to conditional-on-x moments, or state the
  random-x policy for them.
- **Done 2026-09-25 — lab `vcov()` contracts and retained data.** Explicit
  information, sandwich, delta-method and stored covariance regimes name their
  formula. Existing defaults and legacy aliases preserve numerical behavior.
  Empirical covariances use retained observations when available, grouped raw
  wrappers are unpacked, and errors identify the data helper or `vcov()` caller.
  FIML rejects replacement data instead of silently ignoring it. Focused R
  tests cover ML oracle parity, FIML conventions and estimator-specific guards.
- **S — finish migrating callers.** In-repo experiments, benchmarks,
  examples, fixture tools and CI use `magmaanlab` and `fit_model()`. Left: the
  `experiments/showcases/` renumbering that was in flight during the rename
  (`06-speed-attribution/`), and the nested paper and private repositories,
  which still load `magmaan` and call `magmaan()`.
- Deferred until after the first release: `ordered = TRUE`, `fit_measures()` (which statistic
  feeds CFI and RMSEA), modification indices under the policy, `predict()`
  factor scores, a `control` option and summary-statistic input.

## Optimization and convergence

Starting values, optimizer defaults and backends, and the convergence verdict.
Most open items come from real models: the source-verified textbook corpus
(engineering/active/17-corpus-optimizer-recovery and engineering/active/19-newton-verdict-migration) and the covariance-honest-sem banks. A
failure that turns out to be a model-setup gap belongs to its estimator's
section: the categorical corpus saddles are under
[Categorical models](#categorical-models-gaps-found-on-the-textbook-corpus).

### Starting values and saddle escape

- **M — finish uniform start-policy coverage.** The continuous-data/FIML slice
  shares constructor/transport selection and retained supplied-vector reports;
  see the roadmap and [optimizer controls](../reference/optimizer-controls.md).
  Next: ordinal/mixed LS and CatML preparation (thresholds and parameterization),
  two-level summaries, ML2S Stage-1 initialization versus Stage-2 starts, and
  native FCSEM/spherical routes. Route supported selections through the common
  policy and reject unsupported selections instead of silently ignoring them.
  Retain per-block/factor constructor fallback evidence (including FABIN3 to
  FABIN2 and simple-baseline substitutions), user-hint overrides, and separately
  report fit-time projection/PSD repair/profiling. The current retained vector
  is the input supplied to fit, not a claim about the first optimizer iterate.
  Validate group/constraint/identification combinations before changing defaults.
  **Layered moment start (2026-09-25; the ML/GLS default since 2026-09-26, see
  "ML and GLS defaults" below).**
  `start = "layered"` (`estimate::layered_start_values`) builds the
  start in layers: measurement shapes on the standardized covariance, a
  least-squares latent covariance, a joint log-scale identification solve
  (markers, effect coding, pinned variances, cross-block loading equalities), and
  a GLS latent-level fit of paths and latent covariances from sign-generic
  moment magnitudes. Means come last, then a unit-weighted constraint projection
  and a PD repair. On the 608 engineering/active/17-corpus-optimizer-recovery ML/GLS pairs with one engine it
  raises accepted, best-matching fits from 287/238 to 298/297 (ML PORT/L-BFGS)
  and 286/280 to 303/302 (GLS), gaining 113 and losing 4 case fits; every loss
  starts lower and fails in the optimizer. The simple/FABIN misread of a zero disturbance as a std.lv
  scale is fixed. Still open for other routes: FIML and ordinal (not covered by
  the decision); simple/FABIN3 stay selectable for lavaan parity. Known limits:
  trait-state blocks whose latent covariance only the structure identifies keep
  FABIN3 (a full-model GLS polish from the layered start would cover them);
  latent-basis growth relies on the latent-level fit alone (no mean information);
  the Chapter 8 ALT ML models reach a second optimum (0.5476 against 0.5432).

- **S/M — reflection-trap check and saddle escape.** Independent of the start
  constructor. A latent whose scale is set by a fixed variance (std.lv, phantom
  unit variance) has a sign-reflection symmetry. A start on its fixed subspace
  (every sign-odd free parameter at zero) has exactly zero gradient there, and
  L-BFGS, PORT, SLSQP and nlminb never leave it. engineering/active/17-corpus-optimizer-recovery measured gradient 0
  and second derivatives from -1.9 to -17 on the phantom, second-order and
  Table 7.6 paths. *Check:* give each free parameter a character over
  (latent, block) in GF(2): a loading of j gets e_j, a path or covariance
  between j and l gets e_j + e_l, variances and intercepts 0, a latent mean e_j.
  The sign group G is every s that leaves each fixed nonzero entry and each
  constraint invariant (markers and effect coding remove their latent's
  reflection). A start is trapped iff some s in G flips at least one free
  parameter and every flipped parameter starts at zero. Run it on every supplied
  start (including user and lavaan-compatibility vectors) and report it.
  *Escape:* when the terminal curvature audit finds a negative eigenvalue (all
  12 engineering/active/17-corpus-optimizer-recovery ML trap endpoints), step along the eigenvector with a line
  search on the original objective and refit; schedule with the Newton-check
  work. Validate on the engineering/active/17-corpus-optimizer-recovery trap cases with the simple constructor.

### Optimizer failures on the textbook corpus

- **Optimizer coordinates — DONE (2026-09-26), follow-ups below.** Every
  scalar backend (PORT, L-BFGS, SLSQP and their fallback, TNEWTON, VAR2,
  BOBYQA, IPOPT) now searches complete-data ML, GLS, the moment least-squares
  family, pairwise-moment fixed-weight fitting, the constrained ML/GMM entries and FIML in
  unit-equivariant coordinates, information-refined sample units by default
  ([optimizer controls](../reference/optimizer-controls.md#optimizer-coordinates-2026-09-25),
  `estimate/coordinates.hpp`). This fixes the Kline Roth PORT early stops and
  replaces the old ML-only L-BFGS/SLSQP scaling, whose phantom latents took the
  block-mean SD because their unit loadings are structural cells. Evidence:
  `engineering/active/17-corpus-optimizer-recovery`, section "Optimizer coordinates".
- **S — optimizer coordinates for the remaining routes.** Still in raw
  coordinates: ordinal and mixed ordinal fits (mixed models keep continuous
  columns in data units), CatML, two-level ML, the frontier pairwise
  likelihood, SNLLS's outer loading block, the IRLS inner GLS solves and the
  fitted-weight loop (each a one-line wiring through `optimizer_coordinates`),
  and RBM. PSD ML keeps its own lifted information diagonal, whose absolute
  clamps [1e-4, 1e4] and unit fallback are not unit-equivariant; the sphere
  route uses the sample units only. Ceres LM and NL2SOL scale themselves and stay
  out. Wire each remaining route with a unit-rescaling test, as
  `coordinates_test.cpp` does for ML.
- **S/M — L-BFGS first-step domain aborts.** NLopt's Luksan L-BFGS takes an
  unbounded first step along the driven gradient and ends with a generic
  failure (about 12 evaluations) when trial points leave the positive-definite
  region; it can also run along a flat ridge to a far "stationary" point
  (Kievit's latent change score FIML from lavaan's start: disturbance variances
  near 1e6, gradient under the audit tolerance), which the L-BFGS/SLSQP fallback
  does not retry. Information coordinates size the first step and remove most
  of these on the corpus, not all. Candidates: a first-step cap in driven
  coordinates, a backtracking wrapper that maps infeasible trial points to a
  finite merit value, a fallback trigger on the fit verdict rather than the
  backend status, or PORT (whose trust region backs off) as the ordinary default.
- **S — starts that are not unit-equivariant.** Under a per-variable change of
  units, the auto-transported FABIN3 start moves on 148 of the 239
  unit-invariant engineering/active/17-corpus-optimizer-recovery pairs, and on higher-order and cross-group
  models in `coordinates_test.cpp`. The layered start moves on 32: Geiser's
  latent autoregressive and state-trait models, Newsom 3.1c, 5.5b and 9.4,
  Little's Table 7.6 and 3.7 models, the Chapter 10 MTMM, Guo's invariance
  model and a few others (the rank-deficient fallbacks to FABIN3 among them).
  Find and remove the unit-dependent steps before defaulting the layered start.
- **Route nonlinear equality constraints to a constrained optimizer — DONE
  (2026-09-26).** `fit_ml` and `fit_gls` run NLopt SLSQP when the selected
  backend cannot take nonlinear `==` constraints, instead of failing. They
  record it in `Estimates::substituted_backend` (R `fit$optimizer_substituted`);
  see the optimizer controls reference. This fixes Mplus User's Guide ex6.17,
  the one engineering/active/17-corpus-optimizer-recovery pair where lavaan succeeded and magmaan errored.
  From the layered start, SLSQP reaches the best objective (ML 0.001234678,
  GLS 0.001227317) and passes the Newton check. PORT also gained an explicit
  iteration budget, `port.max_iter`.
- **S — score tests admit unidentified candidates by rounding.** A freed marker
  loading is unidentified, so its efficient information is zero at every
  estimate; the absolute tolerance in `score_for_coordinate_robust` admits or
  drops it depending on information-matrix rounding (FIML robust MI on a
  one-factor model: MI 1e-18 and c 2e-7 at one estimate, excluded at an estimate
  6e-6 away). Exclude candidates that only change the identification, or judge
  the efficient information relative to the candidate's own.

- **M, default ordinary fits on the corrected textbook corpus.** The corpus is
  now source-verified (every case reproduces its book's output; see the
  [translation audit](../validation/textbook-translation-audit.md)). On the
  faithful Little models, the historical low-level ordinary-fit test (NLopt
  L-BFGS, simple starts and generic controls; the skipped Little/Newsom
  continuous golden with `--no-skip`) fails
  with a generic solver failure on 12 cases (the 2-by-3 invariance models, the
  Chapter 6 card-sorting simplex, Figure 3b) and stops at a worse stationary solution
  on the Chapter 3.11 phantom model. The current-interface optimizer study
  (`engineering/active/17-corpus-optimizer-recovery`) is a different, wider protocol;
  do not substitute its counts for this historical test. At the current phantom
  endpoint, L-BFGS, PORT and SLSQP have small gradients but nonpositive curvature;
  the fit audit rejects it. A verified-solution restart passes at the lower
  objective. The focused start cross-check recovers 10 of 12 ML curvature
  rejections in both PORT and lavaan by setting free, zero-start latent paths to
  0.5; lavaan defaults also miss many reference solutions. For the phantom CFA,
  the original LISREL input supplies 0.7 on those paths. The layered moment
  start (start-policy item above) recovers 19 of the 21 targeted pairs under
  both PORT and L-BFGS, including both invalid-start models and all three GLS
  cases; do not adopt blanket 0.5 starts. Three ML pairs end at an accepted
  second local minimum under every magmaan arm and under lavaan from the
  layered start: Little's two Chapter 8 ALT models (0.5476; only the book's
  LISREL starts reach 0.5432, lavaan's defaults stop at 0.5591) and Kline's
  Worland step 2a (0.1574; only lavaan's default start reaches 0.1181). These
  are basins, not optimizer failures; only a start policy or multistart
  (speculative backlog) reaches them. The optimizer losses that remained from layered
  starts (L-BFGS's early line-search abort on Geiser's quadratic growth, PORT and
  L-BFGS stopping short on raw-unit variances in Kline's Roth and Lynam models)
  are recovered in information coordinates, the default since 2026-09-26.
  Investigate escape from poor stationary points before unskipping the
  low-level golden. Geiser's marker default-start failures are
  resolved by correcting Reduced representation handling in simple/FABIN
  starts. Remaining: std.lv L-BFGS robustness and transported-start coverage
  for structural models. The start pipeline now separates
  constructor/transport/scaling and reports fallback reasons; R
  `start="default"` agrees with omission. Do not conflate these remaining
  issues with the repaired observed-parameter start mapping.

- **S/M, newsom corpus.** The Little/Newsom continuous golden
  (`cpp/tests/golden/textbook_corpus_golden_test.cpp`) is currently skipped because
  NLopt L-BFGS does not converge `newsom/ex5_5b` from `simple_start_values`, and
  now also fails on 12 corrected Little cases (see the default-fit item above).
  Same family as the documented `ex12_3` case (a second-edition Newsom script,
  no longer in the first-edition corpus) in
  [newsom-corpus-failures.md](newsom-corpus-failures.md): NLopt stalling early on
  a structurally awkward ML objective. Unskip once the starting-value path or a
  harness-level cross-backend fallback handles it.

- **S — ordinal flat ridge (Newsom 2024 ex1.3c).** A saturated
  theta-parameterization model whose factor variance (near 85) is poorly
  determined. From lavaan's starts, L-BFGS stops at fmin 5.8e-9 with the
  factor variance at 83.2 against lavaan's 88.0 and the loadings a few percent
  off; lavaan reaches 1e-16, and started at lavaan's estimates magmaan stays
  there, so the model is right. The R path's Newton check accepts the same
  kind of endpoint (fmin 4.5e-9, d = 0.0023, condition number 3.8e5;
  engineering/active/19-newton-verdict-migration): within magmaan's accuracy budget the fit has converged, and
  the parameter gap is the ridge's flatness. It is the one non-semantic case
  among the known gaps of `textbook_ordinal_golden_test.cpp`. Decide how the
  golden gates ill-conditioned cases (a tighter stopping rule, or parameter
  differences in the information metric), then move it out of `kKnownGaps`.

### High priority: reliable optimizer defaults and L-BFGS domain recovery

- **ML and GLS defaults — DONE (2026-09-26): the layered start, L-BFGS kept.**
  [decisions/01-optimizer-defaults](../../experiments/decisions/01-optimizer-defaults/report.qmd),
  lane ml-gls, three pre-registered runs.
  - **Start.** The layered start is now the complete-data ML and GLS default,
    moved to core. In the third run it certified, of the attainable test
    problems:
    - GLS: 38,489 of 40,812, against 30,747 for FABIN3;
    - ML: 42,062 of 43,096, against 41,364.
  - **Losses.** Five Chen draws at $N = 25$, where the layered start finds a
    worse basin.
  - **Optimizer.** Runs 1 and 2 over-credited PORT, because the Newton check
    certifies runaway points and PORT walks further along divergent paths.
    With runaways scored as failures, PORT still certifies more, but it
    returns about eight times as many runaway fits marked converged. So
    L-BFGS stays, pending the next item.
- **TODO — spectral starts in the next defaults study (banked 2026-09-27).**
  Retain the sample-only recipes and evidence from
  [engineering/active/15-sphere-reference-fits](../../experiments/engineering/active/15-sphere-reference-fits/report.qmd).
  Unrestricted ML candidates include positive and signed spectral directions;
  PSD candidates include only the positive construction. Compare explicitly
  against layered/native with information scaling for ordinary ML, and
  FABIN3-auto with diagonal preconditioning for direct PSD ML. The saved
  portfolios improve coverage, but the three retrospective single-start
  selectors do not establish a reliable replacement. This is a defaults-study
  TODO, not adoption: keep the recipes banked, preserve requested markers,
  and do not add multistart now. Revisit after the PSD scale/route work, using
  fresh problems, per-family gains and losses, and the revised chart/accuracy
  checks rather than treating magnitude alone as failure.
- **High — the layered start stalls on Geiser's latent AR cross-lagged model.**
  This is a loss of the new default found by the former automatic-identification example
  (removed on 2026-09-30; data in
  `cpp/tests/fixtures/geiser/gls_reference.json`, case
  `latent_ar_cross_lagged_extended`, marker identification, $N = 569$, $p = 12$).
  - From the layered start, the start objective is 0.66 against 3.10 for
    FABIN3, but the fits stall short of the optimum 0.0411: L-BFGS aborts
    with a line-search failure at 0.0485, and PORT exhausts its budget at
    0.0478.
  - FABIN3 and std.lv identification converge.
  - The corpus holds only the strong-invariance variant, where the layered
    start works.
  - Diagnose the start: likely a flat or ill-conditioned valley reached from
    the latent-level fit. The example times the former start until then.
  - Consider a verdict-triggered safety net: when a default fit from the
    layered start fails the verdict, refit from FABIN3 and keep the better
    certified fit. It needs a lane run before it becomes the default.
- **High — separate chart failures, parameter escape and numerical failures
  before revisiting the optimizer (2026-09-27).** The next step is the light
  sphere reference study in
  [engineering/active/15-sphere-reference-fits](../../experiments/engineering/active/15-sphere-reference-fits/report.qmd),
  alongside the start-unit and PSD-clamp fixes below. The current ML/PSD
  defaults remain shipped choices, open to revision; PORT promotion is not
  the predetermined outcome.
  - The decisions study's extent > 10 rule is a scoring screen, not a proof
    of nonattainment. Preserve its historical results and distinguish local
    accuracy, covariance admissibility, chart proximity and extreme parameters.
  - Engineering/15 now supplies best-observed sphere candidates from a small
    multistart portfolio, assessed without user-chart polish. Its interim
    Newton check uses a strongest-indicator marker chart. A sphere-native
    tangent/curvature assessment remains missing; use candidate labels, not
    certificates. No passing candidate means unresolved, not nonexistent.
  - Inspect disagreements and the single-start references before expanding
    the portfolio. A pilot case has a screened lavaan point missed by all
    original sphere attempts; starting sphere there recovers it with both
    backends. Failed multistarts alone cannot classify nonexistence.
  - Validate any eventual escape diagnostic across units and identifications
    before changing the verdict. The speculative nonattainment cautions still
    apply. A fresh decision run follows concrete fixes and settled labels.
  - The finding that ML's 5000-evaluation budget worsened PORT's results
    remains evidence to explain, not a reason to add another control profile.
- **PSD ML start — DONE (2026-09-26): FABIN3 stays in both PSD routes.**
  Lane psd-ml, second run (pre-registered after the ML/GLS promotion).
  - The layered start fails rule A again for direct PSD fits (five of six
    test families) and rule E for the fallback's ordinary step (four of six).
    Most losses exhaust SLSQP's budget.
  - So `frontier_fit_ml_psd_fallback` keeps its ordinary step pinned to
    FABIN3 with L-BFGS, apart from complete-data ML's layered default. Code
    that runs "ordinary ML first" by hand, with `fit_ml` defaults, gets the
    worse variant.
  - Exception: in the equality-constrained family the layered start avoids
    most of FABIN3's exhausted PSD budgets (203 against 23). A per-model
    start choice would need its own rule.
- **Barrier ML start and optimizer — DECIDED (2026-09-27): defaults stay.**
  [decisions/02-barrier-defaults](../../experiments/decisions/02-barrier-defaults/report.qmd),
  lane barrier-ml, one pre-registered run. No candidate was eligible:
  transported FABIN3 with PORT stays for `frontier_fit_ml_multiinfo`.
  - The layered start loses in five of six test families. It begins next to
    the factor-disappearance set (a latent variance near zero, a loading of
    the wrong sign), which the determinacy penalty favours, and PORT stops
    there at nonpositive curvature. Keep the barrier's start pinned apart from
    the ML default.
  - L-BFGS still stalls in line searches (Boomsma, engineering/active/15-sphere-reference-fits, Wolf).
- **High — barrier start at ×0.01 in equality-constrained models.** Found by
  lane barrier-ml: the default fails 2,653 of the constrained family's
  attainable ×0.01 fits (L-BFGS 506, layered start 132). The start transport
  falls back to native FABIN3 (`native-fabin-fallback`), not in the data's
  units, and PORT reports convergence after about 12 evaluations without
  moving; the verdict rejects it. Fix the fallback's units, then rerun lane
  barrier-ml on a fresh seed base. Check whether the PSD lane's ×0.01 budget
  exhaustion in the same family (the clamp item below) is partly this
  fallback, since PSD also starts from transported FABIN3.
- **High — the verdict certifies barrier fits near a marker pole.** PORT
  returns certified barrier fits with a marker-chart extent above 1000 (287
  default, 510 layered start in lane barrier-ml; L-BFGS none). Investigate
  chart proximity separately from parameter escape under the sphere-reference
  programme above. The study screen is `chart_extent` in
  `decisions/02-barrier-defaults/R/fits.R`; it is not an exact-pole proof.
  Barrier composition with the sphere is not implemented by this pilot.
- **High — revisit the PSD ML route after the barrier lane.** Decided for now
  (author, 2026-09-27): the direct fit (`fit_model(psd = TRUE)`,
  `magmaan(psd = TRUE)`) stays the route; two-stage (`fit_ml_psd_fallback`)
  stays an explicit frontier call. The sample-relative preconditioning clamp
  and a small fresh paired route check are now implemented (see the S item
  below). The clamp alone does not explain the route gap. Next inspect the
  remaining start projection, covariance links and equality-constrained losses
  before a larger route decision.
  - Replicated in both runs: two-stage certifies more (second run 22,132
    against 21,980), reaches the best known objective more often (21,850
    against 21,224), and takes about 40% less time.
  - The earlier advantage comes from rescaled units. In native units the direct
    fit is slightly ahead (second run 5,840 against 5,829). The smaller post-fix
    comparison is recorded below; it does not promote the two-stage route.
  - Engineering the route, later:
    - When the ordinary step errors, the PSD refit starts cold from the
      ordinary step's own start. A cold PSD start should always be the PSD
      default (FABIN3), whatever the ordinary step used. That is the main
      reason a layered ordinary step loses (153 of rule E's 344 losses).
    - Warm starts from rejected or improper endpoints: consider trying the
      cold start too and keeping the better certified fit. PORT's far-out
      endpoints (first run) and the layered endpoints (second run) are both
      poor warm starts.
    - The fallback already returns both attempts (`ordinary`, `psd`); the
      route as a `psd = TRUE` default needs a documented result shape for
      them.
- **S — the Newton check certifies an unidentified fit on an exact-fit ridge.**
  `r-package/examples/ml_psd_fallback.R` fails at `!e$converged` (before
  f10fdd84 too): `f =~ x1 + x2 + x3` with `auto_fix_first = FALSE` has 7
  parameters for 6 moments.
  - The PSD fit reaches the ridge with gradient about $10^{-10}$. There the
    Hessian's null direction has curvature of the gradient's order, so its
    condition number ($8 \cdot 10^{10}$) stays under the $10^{12}$ guard, and
    the Newton distance is $5 \cdot 10^{-10}$.
  - A larger guard is not the fix, since the condition number depends on how
    precisely the optimizer lands. A scale-free identification check (the
    rank of the moment Jacobian at the estimate) belongs next to the verdict.
  - It does not touch identified models, such as those of the
    covariance-honest paper.
- **S — PSD lift round-trip error in the std.lv equal-loadings CFA.**
  `fit_ml_psd` fails with "PSD lift finalization could not round-trip the
  terminal covariance links into ordinary partable coordinates" (decisions/01,
  lane psd-ml: 188 fits over all arms).
  - It occurs only in the two std.lv equal-loadings CFAs (three factors,
    fixed unit latent variances). Most cases have loadings .4, where it hits
    about 4% of fits at $N = 25$ and 0.5% at $N = 100$.
  - It occurs in native and $\times 0.01$ units about equally and never at
    $\times 100$.
  - The mechanism is unknown: fixed-diagonal $\Psi$ blocks, the equality
    links, or an absolute round-trip threshold.
- **S — PSD preconditioning clamp — FIXED; wider unit sensitivity remains
  (2026-09-27).** The information scale now has bounds `[1e-4*u, 1e4*u]`,
  where `u` is the coordinate's sample-derived unit, and zero information uses
  `u`. Original parameters reuse the shared identification/equality unit
  machinery; lifted Cholesky entries use their row variable's unit. The raw
  information scale retains its original arithmetic whenever unclamped.
  These are optimizer scales, not restrictions on estimated variances.
  - The former absolute clamp fails the transported-start coordinate test.
    Regression coverage now includes marker, std.lv with zero-information
    loadings, correlated residuals/equal variances, uniform ×0.001 to ×1000,
    and mixed observed units. The start eigenvalue floor is kept inactive
    for this test. A fitted PSD boundary witness also agrees under ×0.01
    and ×100. All 535 estimate tests pass.
  - A paired follow-up in decisions/01 uses three draws at each population/N,
    seed 202609290, explicit FABIN3-auto starts, L-BFGS ordinary and SLSQP
    constrained fits, diagonal PSD preconditioning. Both scale versions are
    evaluated on the same 1,356 problems per route. See
    `criteria/psd-scale-followup.md` and the report's PSD scaling follow-up.
    This is a diagnostic, not a route/defaults promotion. Final direct PSD
    certification is unchanged at 1,328/1,356 (eight gains, eight losses);
    two-stage changes 1,333→1,332. Under the corrected scale, two-stage matches
    1,326 best observed objectives versus direct's 1,298; native certification
    is 350 versus direct's 351. No route promotion follows.
  - **Remaining:** covariance-link residuals and their absolute tolerances
    still retain physical units; changing optimizer columns alone does not
    normalize those rows. The start covariance projection also floors
    eigenvalues at the absolute `start_eigen_floor`; when active it can change
    an otherwise transported starting covariance. Isolate these effects on
    the retained paired losses, especially the equality-constrained chains
    and the std.lv round-trip failure. Do not call every unit-sensitive failure
    a clamp effect. A retained witness is `eqchain_b20`, equal paths and
    disturbances, N=25 draw 3: direct native objective 1.555299 becomes
    2.198990 at ×100 although both certify; two-stage reaches 1.555299 at ×100.
    Preserve this as a unit-sensitive objective witness, not a proof about
    which solver component caused it. Keep the spectral-start question separate.
- **Full normalization before PSD fitting — unconstrained pilot completed
  (2026-09-27), not promoted.** `decisions/01-optimizer-defaults`,
  `criteria/psd-normalization-pilot.md`, seed 202609291: 300 model/sample
  combinations, four observed-unit transforms, marker and std.lv charts,
  direct and ordinary-then-PSD routes, current versus normalized inputs
  (9,600 attempts). Equality constraints and arbitrary nonzero fixed values
  are rejected. Markers stay fixed; estimates are backtransformed explicitly.
  - The sample-only normalization and parameter transport preserve every
    returned normalized point: maximum standardized covariance discrepancy
    5.94e-16 and objective discrepancy 2.71e-13. Correlated residuals and
    cross-loadings are included; a separate mean-transport check passes.
  - Direct marker PSD: 900 native-versus-rescaled comparisons go from nine
    verdict differences and two certified objective differences to zero of
    each. Direct std.lv certified objective differences go from nine to zero;
    five verdict differences remain. Two-stage marker objective differences
    go from 32 to one; two-stage std.lv from 20 to seven. Normalized inputs
    themselves differ by at most 4.45e-16 across observed-unit transforms.
  - This is improved unit consistency, not demonstrated basin selection.
    Direct marker certification rises 1,179→1,184/1,200 internally, but
    normalized two-stage marker best-observed matches fall 1,178→1,149.
    All paired losses and remaining unit exceptions are retained.
  - **Done: normalized PSD accuracy diagnostics (2026-09-27).** Single-group
    complete-data ML with equal labels and general linear equalities now audits PSD geometry and
    Newton accuracy in sample-normalized units, preserving requested markers
    and returning reusable full-space derivatives in caller coordinates.
    Accuracy thresholds are unchanged; `unit_normalized` exposes the scope.
    Re-auditing 4,742 saved points without refitting recovers all five earlier
    false failures, with zero original/normalized verdict or nullity/held-face
    disagreements. Two previously passing original-unit checks now detect
    nonpositive curvature; both fits already failed their internal check.
    The regression uses development data, not a fresh policy decision run.
    Linear equalities are transported into the internal model as weighted rows,
    including affine constants; original merge groups are cleared there to avoid
    imposing extra constraints. Constraint rank must be preserved. Tests cover
    equivalent mixed-unit models, interior and PSD-boundary geometry, feasible
    Newton steps, and rejection of an inaccurate point. Nonlinear equalities
    remain unsupported by this audit.
  - Follow the shared ML/PSD/barrier normalization programme at the top of this
    backlog. Inspect remaining two-stage sensitivity to roundoff and retain
    alternative local minima as search evidence. Equality/fixed-value transport
    remains outside this pilot; the core accuracy check now transports linear
    equalities. Complete-data ML/PSD fitting normalization has since landed;
    broader coverage and route adoption remain open.
- **Convergence bench — deferred (2026-09-26).** The judge and its scoring are
  built (the Newton check in every iterative estimator's verdict, also applied
  to other engines through `evaluate_at`; certified local minimum first, best
  known objective second). Default decisions now run in `experiments/decisions/`
  under a pre-registration standard (see `experiments/AGENTS.md`). The shared
  benchmark set, canary tier, problem classes and holdout protocol from
  [convergence-engineering.md](../design/convergence-engineering.md) are in the
  speculative backlog with their triggers.
- **High — remaining Newton-verdict controls and coverage.** The shipped
  contract is in [terminal-audit.md](../design/terminal-audit.md#authoritative-fit-verdict-2026-09-12);
  the rollout evidence is engineering/active/19-newton-verdict-migration. Remaining:
  - Adopt complete-data ML's stopping controls for FIML and the least-squares
    fitters after a corpus run (on the corpus they rescue 4 of 6 near misses
    and 4 of 5 FIML non-minima).
  - Build an analytic two-level Hessian, also replacing finite differences
    behind two-level observed standard errors. CatML stays first-order by decision.
  - Check the `.01` budget's sensitivity at `.003` and `.03` when the
    asymptotic covariance is estimated, especially for DWLS and ordinal fits.
  - Assess a final safeguarded Newton correction for positive-definite
    curvature just above budget as an optimizer-policy decision.
  - Deferred constraint/estimator extensions and R audit bindings remain in
    the [speculative backlog](speculative.md#convergence-audit-extensions),
    including nonlinear/callback equality Lagrangians, boxes interacting
    with singular PSD faces, FCSEM and implicit RBM. Retain first-order
    verdicts for unsupported cases; these are not new implementation commitments.
- **M — finish optimizer-control reporting and specialized-path inventory.**
  Explicit backend control blocks now cover NLopt L-BFGS/SLSQP/VAR2/TNEWTON/
  BOBYQA, PORT scalar/NLS, IPOPT, and Ceres estimator bridges, with legacy
  defaults preserved; see `project/reference/optimizer-controls.md`. Remaining:
  carry effective controls and raw backend stopping reasons through fitted
  results (separately from audit verdicts); inventory specialized scoring/EM/
  IRLS outer-loop controls and distinguish them from inner-solver controls.
  Decide a deprecation policy for legacy `gtol`/`history` without silently
  changing existing calls. Explicit `nlopt.vector_storage` now selects memory;
  legacy `history` remains ignored by NLopt for compatibility. Backend-specific
  blocks permit fallback configurations; unused blocks are inactive.
- **High priority / M — make domain recovery robust to parameter scale.**
  Reliable defaults are required even if PSD fitting becomes the default:
  ordinary fitting and the ordinary-first/PSD-recovery policy must remain
  numerically sound. PSD constraints do not excuse optimizer failures.
  NLopt's Luksan
  L-BFGS hard-codes ten line-search reductions. With the objective contract
  returning infinity outside the covariance domain, all trial evaluations can
  remain invalid and the solver aborts after 12 evaluations at its initial
  point. Increasing the overall evaluation budget does not change this limit.
  The standalone variance-likelihood probe
  `cpp/tests/checks/nlopt_lbfgs_domain.c` reproduces the mechanism without SEM.
  Provide configurable safeguarded backtracking or a portable adapter-level
  recovery policy; do not fix individual datasets by changing their units.
  Validate across units, equality-reduced models, box bounds, and invalid
  starts, checking the original objective and terminal audit. Preserve the
  ability to distinguish pure L-BFGS from an SLSQP recovery. The existing
  `nlopt-lbfgs-slsqp-fallback` is an available interim policy, but currently
  retries on optimizer failure/non-clean status, not every failed model-level
  stationarity verdict.
  The corrected-corpus study and production-adapter scalar probe confirm that
  extra evaluation budget and tighter tolerances do not remove early domain
  failures. PORT recovers cases missed by ordinary L-BFGS; bounded SLSQP can
  return a success code away from a known scalar optimum, where the terminal
  audit correctly fails. Any recovery policy must assess the original objective
  and independent audit, not merely the fallback backend's success flag.
  Do not substitute PSD fitting for ordinary optimization
  when diagnosing this numerical failure: that changes the feasible set.
  - **Clean reproduction (2026-09-23): Little's polynomial bullying growth
    model** (30 indicators, 16 latents, effect coding, higher-order growth;
    the covariance-honest-sem example).
    - **Setup.** Samples drawn from its accepted PSD population.
    - **Failure.** Default ML converges on the raw data. Multiplied by 3,
      the data fail on all 40 samples tried (N = 50 to 1132), with
      LineSearchFailed after 12 evaluations at the start objective. Factors
      2, 5 and 10 fail too on the one N = 300 sample tried.
    - **Others succeed.** lavaan (nlminb) and magmaan PORT converge on the
      ten ×3 samples at N = 300. SLSQP and the L-BFGS-to-SLSQP fallback
      converge on all 40 standardized samples, where default ML fails on
      all 40.
    - **The start is not scale-equivariant.** All 16 latent variances start
      at the constant 0.05 under every policy (scaled-fabin, fabin3,
      simple). The start objective grows from 43 (×1) to 200 (×10), while
      the optimum stays 0.64.
    - **Fixes to consider.** Scale constant latent-variance starts to the
      data, and fix the domain recovery above. The paper's pre-refresh bank
      (all 700 L-BFGS fits failing) was probably the same edge.
- **High priority / M — regular-interior curvature diagnostics for the
  full-model audit.** Author priority (2026-09-21): implement and understand
  regular identifiable interiors first; defer PSD-boundary extensions;
  practical verification by tighter refits, restarts, independent optimizers
  and held-out empirical calibration is lowest priority for later.
  The model-Frobenius dual L2 cutoff `1e-3` is uncalibrated and is not
  equivalent to lavaan's coordinatewise gradient criterion. First assess
  the reduced-coordinate observed Hessian, predicted Newton correction,
  and Newton decrement on the original objective scale. Distinguish a local
  accuracy approximation from a global error bound. Resolve linear equality
  constraints before checking positive definiteness; do not silently use a
  pseudoinverse in unidentified directions. Test scaling identities and
  analytic quadratic cases. For per-observation ML, record the relation
  between `N*g' H^{-1}g`, predicted twice-log-likelihood improvement, and
  information-standardized parameter displacement. Retain independent
  feasibility/objective checks and report this diagnostic unavailable when
  interior/curvature conditions fail. See `project/design/terminal-audit.md`.
  Advisory prototype and 3,402-fit paired L-BFGS control study are complete:
  `cpp/tests/checks/interior_newton/`; findings in
  `project/validation/interior-newton-audit.md`. Candidate audit budget is
  `d=sqrt(N*g' H^{-1}g) <= .01` (total EDM <=5e-5), with .003/.03
  sensitivity. Original conservative candidate backend stopping is ftol_rel=1e-12 and
  xtol_rel=1e-10, retaining internal tolg=1e-8 and automatic memory. These
  are recommendations, not deployed defaults. Extended-backtracking runs
  passed all 180 eligible interior cases at .003; stock backtracking still
  failed on small measurement units. One old-audit pass had d=.254 and
  stopped on xtol; reducing ftol alone did not help.
  Literature/accuracy rationale is now expanded in
  `project/validation/interior-newton-audit.md`: Dennis–Gay–Welsch's all-contrast
  criterion, the exact .01 conversion of MINUIT manual 94.1's nominal default,
  and conditional RMSE/Gaussian-reference interpretations. Available reference
  files and failed-download receipts are retained under ignored `external/refs/`.
  The author has accepted .01 as the regular-interior accuracy budget.
  Broader validation of predicted versus actual error remains separate.
  **Agreed tolerance rationale:** .01 is a numerical-error
  budget (one hundredth of an information-based SE in any linear contrast
  under the local quadratic approximation), not a literature-mandated cutoff
  or a value estimated by counting passes. Choose acceptable downstream
  accuracy first, then assess whether stopping controls deliver it at a
  reasonable cost. Report continuous d and sensitivity at .003/.01/.03;
  distinguish validation of the Newton approximation from the choice of an
  acceptable error budget. Immediate evaluation remains analytic quadratic
  identities, objective/sample-size normalization, coordinate invariance and
  curvature eligibility. Later, at the already agreed lowest priority,
  compare predicted displacement and likelihood improvement with accurately
  refined same-domain solutions, and check whether substantive conclusions
  change across budgets. A high pass rate alone validates neither the
  approximation nor the cutoff. For non-ML discrepancies, establish the
  estimator's objective and covariance scaling before claiming SE units;
  exposing its optimizer controls does not transfer the ML audit calibration.
  Intermediate option study (1,134 further paired fits) found that ftol_rel=1e-10,
  xtol_rel=1e-8 also passed all 180 eligible cases at .01, with 3.8% more total
  evaluations than legacy controls versus 17.0% for the conservative candidate.
  Evaluate this cheaper first-pass candidate with the separate audit; conditional
  polishing with tighter controls is a proposed next step, not yet tested.
  Same exploratory datasets and diagnostic extended backtracking: no default
  changed and no held-out reliability claim. Domain recovery remains prerequisite.
  Broader fixed-control validation is complete (720 fits): ten new settings,
  p=6–48, 12–162 free parameters, N=50/200/1000/10000, two replications,
  three unit scales. Adds large CFAs, a structural chain, cross-loadings,
  residual correlations and mixed indicator scales. Current/cheap/conservative
  controls pass 198/201/218 of 220 eligible fits; 20 noninterior endpoints per
  profile remain excluded. This supersedes any inference from the smaller
  panel that the cheap settings are near-settled defaults. Both conservative
  failures have mixed scales. Preserve the .01 target, investigate conditional
  polishing/scaling rather than relax it. Before a package-wide default claim,
  the subsequent panel below supplies multi-group invariance, growth, feedback
  and published fixtures; use fresh validation cases after tuning. See the research note for coverage.
  The previously missing growth, feedback, multi-group and published-fixture
  panel is now added (558 fits; same fixed profiles). Growth and all six
  published specifications pass under every profile; conservative controls pass
  all eligible multi-group cases. Feedback at scale 10 exposes five eligible
  conservative failures (d=.419–3.324), plus one nonpositive-curvature endpoint;
  one fit reaches the evaluation cap. Investigate that mechanism before defaults.
  Later, as requested, examine option performance by model family, p, free
  parameter count, n, group count/imbalance and measurement scale. Metadata is
  saved; use paired comparisons, keep failures/exclusions, account for dependent
  scale/specification variants, and validate any adaptive routing on fresh cases.
  Do not infer a routing policy or tune one from these small descriptive counts.
  Feedback start isolation is complete (72 fits): canonical FABIN retains
  scale-independent .05 latent variance starts from the simple scheme while
  residual starts scale with sample variances. Transporting unit-1 starts to
  scale 10 (latent variances 5 instead of .05) changes passes from 2/8 to 8/8;
  median evaluations fall from 3387.5 to 208.5. Transporting optimizer
  coordinates as well reduces median evaluations to 50, still 8/8 passes.
  High priority: design a data-derived scale-aware start/coordinate policy,
  respecting identification, fixed values, user hints and equalities; keep
  compatibility starts explicit. The experiment is model-specific and uses
  diagnostic backtracking, not a production fix. Validate on the full panel
  and fresh cases before selecting defaults; do not simply tighten tolerances.
  Matched marker/std.lv starts plus sample-derived coordinate scaling pilot
  complete (540 fits, five models, 90 cases). All acceptance uses the common
  marker-coordinate audit. Native marker passes 83/87 eligible; native std.lv
  87/87; transported std.lv starts with scaled marker coordinates also 87/87,
  with total evaluations 4253 versus 9740 for native std.lv. Weak-marker CFA
  remains a counterexample to uniform cost dominance (std.lv totals smaller).
  The approach uses current-sample SDs, not known unit multipliers. Promising
  candidate only: add full Guttman CFA starts under the same metric, handle
  fixed values/hints/equalities/means/groups, and validate fresh cases before
  defaults. Ambient chart domains differ outside positive disturbance variances.
  Broad frozen-policy validation complete: 15 synthetic models, six published
  specifications, five fresh replications per N=50/200/1000, three unit scales;
  693 cases, 3594 executed fits across stock and extended NLopt. Candidate uses
  transported std.lv FABIN starts where supported, otherwise native starts;
  sample-derived scaling acts in the unchanged equality-reduced coordinates.
  Stock baseline/candidate passes: 453/648 out of 693 attempts; extended:
  635/648. All 648 candidate eligible interiors pass; 45 noninterior endpoints
  are retained. Fixed-loading/equality fallback passes all 276 eligible cases.
  No baseline-only passes on either backend; not a universal guarantee.
  Diagnostic round complete. Complete-data ML defaults are now promoted:
  NLopt control profile, transported/native FABIN starts, ordinary reduced
  sample scaling and lifted PSD information scaling. Explicit user options
  and hints remain authoritative; ordinary scaling reports its applied branch.
  Remaining: production Newton diagnostic, extended constrained-coordinate
  scaling, detailed effective-control reporting and adaptive PSD recovery.
  Full Guttman-start comparison remains deferred. See the study note for costs,
  unsupported std.lv cases, invariance checks and source/result provenance.
  SLSQP ordinary/PSD smoke complete (648 fits, 108 cases per arm). All
  eligible interiors pass .01; improved starts plus sample scaling reduce
  ordinary evaluations 33009->6773. PSD improved starts plus existing lifted
  information scaling reduce 22950->8393; all boundary points pass the existing
  cone audit (not a .01 accuracy certificate). One native-start feedback PSD
  endpoint at N=200, scale 10 passes the cone audit with nullity 3 but has
  f=.21609 versus .11002 from improved starts. Preserve/inspect that endpoint;
  do not infer local/global optimality or fix it by tuning the audit cutoff.
  Frozen larger validation complete: 1692 fits, 423 cases per policy, fresh
  synthetic N=50/500/5000 samples. Improved ordinary and PSD policies each
  pass all 396 eligible interiors; accepted totals are ordinary 396/423 and
  PSD 420/423 (24 boundary cone passes), versus native 387 and 416.
  Keep ftol_rel=1e-12, xtol_rel=1e-10, maxeval=5000 and interior d<=.01.
  Use Newton accuracy for regular interiors, cone checks at PSD boundaries;
  retain the old interior cone residual as telemetry, not an extra veto.
  Research round provisionally closed; complete-data ML defaults promoted.
  Improved PSD fails on one weak-factor sample in three unit systems;
  paired native replay rescues one, leaving two. Consider native/unscaled
  retry during integration, without claiming a validated adaptive policy.
  Earlier structural PSD cost findings were different. Ordinary and
  lifted-PSD scaling remain distinct methods.
  **Done 2026-09-24:** the opt-in interior diagnostic is in core,
  `estimate::frontier::newton_accuracy_ml` and R `frontier_newton_accuracy()`,
  ported from the prototype and the covariance paper's bridge. On 86 Ernst
  fits (ordinary and PSD, N = 20 to 200) it matches the bridge's status in
  every case and its distance to 1.3e-8 relative. `cpp/tests/unit/
  newton_accuracy_test.cpp` covers the quadratic identities, an exact
  solution, perturbations with and without equalities, a boundary block and
  nonlinear constraints (unsupported). Multi-group fits are computed but not
  yet tested. Remaining immediate job:
  complete effective-control/stopping-reason reporting now that explicit
  NLopt step/tolg/vector-storage controls are available. Preserve terminal
  candidates independently of raw status.
  Deferred PSD work: dimensionless primal/dual/complementarity assessment,
  active-eigenvalue sensitivity, and metric projected-gradient alternatives.
  **Done 2026-09-24:** the boundary verdict is second order.
  `newton_accuracy_ml_psd` restricts the Newton step to the face of the PSD
  cone (positive-multiplier null directions held, sigma-term curvature), and
  PSD ML fits use it at every point. Tests: equality with the interior check
  at interior points, a Heywood boundary, and a curve inside the rank-one
  face of a correlated-factor model where predicted gain / actual gain is
  0.99978, 0.99989, 0.99994 at steps 2e-3, 1e-3, 5e-4. On the covariance
  paper's designs (2,400 small-sample fits, 80 bullying fits with a singular
  6 x 6 block) it agrees with the old cone rule at every boundary point
  except 28 std.lv drift fits (|beta| up to 9,500), which it passes and the
  unit-dependent cone residual happened to reject. Refitted bullying
  boundary fits gained what d^2/2 predicted (.153 against .154).
  A bounded refined-reference smoke is now complete: 84 eligible natural
  endpoints and 56 local probes, all with reference d<=1e-8. At nominal .01,
  measured/predicted lengths agree within .42%, with tiny hard-threshold
  disagreements documented in the study note. This is not a uniform error
  certificate; see `cpp/tests/checks/interior_newton/README.md` for reproduction.
  Remaining lowest-priority practical follow-up: broaden same-domain references,
  compare standardized parameter/moment changes, and freeze targets before
  held-out model/unit/rank validation. Neither the existing two-model pilot
  nor this prioritization establishes a production convergence policy.
- General runaway/nonattainment diagnostics are deferred to
  [the speculative backlog](speculative.md#runaway-estimates-and-nonattainment-diagnostics).
  Local stationarity does not establish attainment of a finite optimum.
### Convergence verdict and terminal verifier

The remaining explicit convergence-audit extensions are deferred to
[speculative.md](speculative.md#convergence-audit-extensions). The implemented
coverage and contracts live in [terminal-audit.md](../design/terminal-audit.md);
there is no scheduled expansion of this surface.

- **XL.** Design an optimizer terminal-point "ultimate verifier" track. Turn the
  provisional audit tolerance into an empirically justified convergence
  certificate rather than a hand-tuned cutoff. Build an offline verifier that
  records the backend-independent L1 residual
  `||projected_gradient||_inf / (1 + |f_recomputed|)`, objective/parameter gaps
  to lavaan or certified fixtures, cross-backend same-basin agreement, PD margins,
  active bounds, and constraint residuals over the
  Geiser/Mplus/Little/Newsom/paper corpora. Add a high-precision check mode
  (`long double`, MPFR/Boost.Multiprecision, or an R `Rmpfr` helper) that
  re-evaluates `f` and the gradient at terminal points and optionally performs a
  few high-precision local refinement steps. Use the resulting CSV/report to
  separate ordinary line-search noise-floor salvages from genuinely non-stationary
  same-objective points (e.g. Newsom `ex5_4`/`ex5_4c`; see
  [newsom-corpus-failures.md](newsom-corpus-failures.md)) and to justify any
  default `TerminalAuditOptions` tolerance change in
  `project/design/terminal-audit.md`.
- **M/L.** Decide whether `TerminalAuditOptions::stationarity_mode` should stay at
  Absolute (lavaan-matched) or switch to Relative once the verifier track above
  has data. v1 ships Absolute at `absolute_tol = 1e-3` to match lavaan's
  `check.gradient = TRUE` / `optim.dx.tol = 0.001` default; this is the first hard
  design call in magmaan and the calibration is genuinely unstable. The Relative
  code path is fully wired and unit-test-covered
  (`cpp/tests/unit/terminal_audit_test.cpp`), so the experiment is one option flip
  away once the data exists; see `project/design/terminal-audit.md` "Tolerance
  calibration".
- **M/L.** Complete the common-verdict rollout defined in
  `project/design/terminal-audit.md`. The authoritative C++ verdict and R
  TRUE/FALSE/NA convergence projection now use original-objective verification
  plus common full-model stationarity, with explicitly selected ambient/PSD
  geometry. Legacy/backend flags are not vetoes. Remaining:
  preserve evaluable terminal candidates uniformly across soft backend exits;
  migrate active research consumers that still gate on driven audits/backend
  statuses. Additional constraint and specialized-estimator audit coverage is
  deferred to [speculative.md](speculative.md#convergence-audit-extensions). Do not convert
  failed/unchecked returned fits into hard errors. Retain rank/conditioning
  and fallback diagnostics alongside the common verdict, and use stress/corpus
  comparisons to validate each coverage extension. The frozen SNLLS handoff's
  historical screen is retained only for reproducibility of its pinned run.

### Backend choice

- **M.** Compare NLopt L-BFGS/SLSQP/VAR2/TNEWTON/BOBYQA, PORT/PORT-NLS, Ceres
  trust-region, Ceres dense BFGS, and SNLLS only on semantically appropriate
  cases; include shallow or Heywood-prone LS cases so bounds and conditioning stay
  visible.

- **M/L.** Revisit the remaining complete-data default-backend choice once the
  optimizer comparison studies land. FIML now defaults to the NLopt
  L-BFGS-to-SLSQP fallback after the missing-data optimizer panel, and NLopt is
  a required dependency. The broader default still needs justification across
  ML, complete-data LS, bounded ordinal LS, direct optimizer callers,
  augmented-Lagrangian inner solves, and nonlinear-constraint paths (NLopt SLSQP
  and IPOPT). Document tolerance semantics (`gtol` vs NLopt `xtol_rel`),
  iteration/evaluation reporting, and bounded behavior before changing the
  remaining defaults.

## Estimation and inference follow-ups

### Categorical models: gaps found on the textbook corpus

The corpus holds 34 categorical (WLSMV) cases, each verified against its
book's output or the author's lavaan call. `textbook_ordinal_golden_test.cpp`
fits the 21 all-ordinal, covariate-free ones from lavaan's partable
(`cpp/tests/fixtures/textbook_ordinal/`) and matches lavaan on 19. The other
two sit in `kKnownGaps`: one is the residual mean-structure/delta-scale gap
below, the other a flat ridge where the optimizer stops early (Newsom 2024
ex1.3c, under
[Optimizer failures on the textbook corpus](#optimizer-failures-on-the-textbook-corpus)).
The evidence, with sources and run provenance, is logged in
[the translation audit](../validation/textbook-translation-audit.md#categorical-fits-against-lavaan-2026-09-25).

- **Done 2026-09-29 — ordinal partable semantics.** `prepare_ordinal_partable`
  unconditionally forced every ordinal indicator's residual-variance (`~~`)
  and intercept (`~1`) row to the single-group delta default (1 and 0), and
  never subtracted the mean structure (τ − ν − Λα, Λα ≡ μ since ν is always 0
  for an ordinal indicator) from the implied thresholds outside the
  Wu-Estabrook multigroup path. Three fixes, all in `cpp/src/estimate/
  ordinal.cpp`:
  1. **Mean structure.** `ordinal_residuals`/`ordinal_block_residual`/
     `ordinal_moment_jacobian_block`'s plain (single-group delta) branch now
     subtracts μ like the theta/released branches already did, and
     `ordinal_curvature_weights`'s `subtract_mu` gate (the analytic-Hessian
     second-order term) was generalized the same way — `add_lisrel_second_order`
     already consumed the bilinear Λα curvature generically, it just needed
     the weight populated. Fixes Mplus ex6.4/ex6.15 (partially — see #3) and
     Newsom ex7.2a/ex9.2 (both editions) outright.
  2. **Theta explicit free/fixed.** `prepare_ordinal_*_partable` now checks
     `spec::LatentNames::row_user` (threaded through as a new optional
     `row_user` parameter on `prepare_ordinal_partable`/
     `prepare_mixed_ordinal_partable`/`fit_ordinal_bounded`/
     `fit_mixed_ordinal_bounded`/`ordinal_ls_objective`, wired from
     `Model::names().row_user` in `api/sem.cpp`) and leaves a `~~` row the
     user's own model syntax explicitly resolved — free, or fixed at a
     non-default value — alone instead of overwriting it. Fixes Mplus ex6.5
     (residual variance freed at later occasions) and Newsom ex3.3a (fixed
     at 0, an equivalent rescaled fit) outright. `row_user` defaults to
     `nullptr` (old unconditional-forcing behavior) everywhere except the
     handful of call sites that now pass it, so partables synthesized in C++
     (nested-test H0/H1 pairs, PSD probes) are unaffected.
  3. **Delta scale.** magmaan never reads a `~*~` (response scale) *value* —
     δ is always derived analytically as `1/√Σ*ᵢᵢ` (see the Wu-Estabrook
     `block_released` branch) — so an explicitly-freed `~*~` row is
     translated into leaving the *sibling* `~~` row free instead: the same
     reparameterization (same fmin/χ²/df, different raw coordinates) that
     branch already implements, generalized from the multigroup-only release
     to any group. Since the translated `~~` row arrives *fixed* from
     lavaan's own imported partable (lavaan puts the free dimension on
     `~*~`), it needs an active new free-index assignment, not just an
     exemption from forcing — done in a pre-pass before `remove_free` is
     sized, mirroring the existing `intercepts_equal` pre-pass. Fixes Mplus
     ex6.4/ex6.15 fully (fmin/χ²/implied-Σ match lavaan to numerical
     precision) and clears the free-parameter-count mismatch on Newsom
     ex9.2, but that one case hits a narrower **remaining gap**: lavaan's
     free-delta optimum implies a *negative* residual variance for several
     occasions (Σ*ᵢᵢ > 0 throughout, but θᵢᵢ < 0 in magmaan's additive
     decomposition — a Heywood-looking point that's perfectly legitimate
     under free-delta, since only Σ*ᵢᵢ > 0 is truly required), which
     `fit_ordinal_bounded`'s default `variance_bounds` (θ ≥ 0, the deliberate
     Heywood-prevention default for ordinal fits) excludes. Left as
     `kKnownGaps` rather than relaxing that default for one fixture; a
     targeted per-row bound relaxation for delta-translated rows would clear
     it if a concrete need arises.

  All three landed together (the mean-structure and delta-scale fixes
  interact: ex6.4/ex6.15 needed both). Regression evidence: full `ordinal`/
  `estimate`/`inference`/`spec`/`api`/`parity`/`sphere_route_parity` ctest
  labels green (2600+ cases), plus the textbook-ordinal analytic-Hessian-vs-
  finite-difference golden test, which exercises the new Jacobian/curvature
  terms directly. Vendored into `r-package/src/` via `just vendor`; the R
  surface inherits the fix through the shared C++ core (no parallel R logic
  per the two-package design) and was smoke-checked via `just r-dev` +
  the existing R test/example suites, not re-verified against a hand-built
  Mplus-syntax growth model in R.
  **Remaining, separate from this fix (unrelated bugs, own backlog items
  below):** covariates in categorical models, `ordered` with `group:` blocks,
  and mixed ordinal/continuous fits stopping above lavaan's objective.
- **High — covariates in categorical models.** Eight corpus cases regress
  ordinal outcomes on observed covariates, which lavaan handles with
  `conditional.x`: Mplus ex3.4, 3.12, 3.13, 3.14, 5.16, 5.17, Muthén ex8.29_2
  and Newsom 2024 ex4.2b. magmaan has no conditional path, yet the R path fits
  the four single-group User's Guide models anyway and ends at saddles far above lavaan's
  objective (engineering/active/19-newton-verdict-migration): 2.6 on ex3.12 and ex3.13, 1.6 on ex3.14, and
  0.93 against 0 on the just-identified ex3.4, which any consistent setup fits
  exactly. So the R path's handling of the covariates (`fixed.x`, joint versus
  conditional moments) is wrong, not only missing. On the two single-group
  cases with one covariate (Muthén ex8.29_2, Newsom 2024 ex4.2b) the R path
  instead stops in stage 1 ("mixed ordinal stage-1 information matrix is not
  positive definite"). Implement the conditional moments or refuse such
  models.
- **S — `ordered` with `group:` blocks in the R interface.**
  `magmaanlab::fit_model()` with `ordered` and a model written in `group:`
  blocks stops in `data_ordinal_stats_from_df()` ("model/data group count
  mismatch"; UG ex5.19). engineering/active/19-newton-verdict-migration leaves out the multi-group
  categorical cases for this reason.
- **M — mixed ordinal/continuous fits stop above lavaan's objective.** Five
  corpus cases mix ordinal and continuous indicators without covariates. The
  R path fits all five and the Newton check accepts every endpoint, but only
  Mplus ex5.3 reaches lavaan's objective (0.000947 against 0.000946). The four
  Newsom models stop 12 to 62% higher: 0.121 against 0.074 on 2015 ex5.3a,
  0.185 against 0.146 on ex5.3b, 0.385 against 0.303 on ex5.7a and 1.216
  against 1.082 on 2024 ex5.8b. Accepted minima that high point at a different
  model or weight, not the optimizer; compare the setup (thresholds, means,
  scale and residual-variance semantics, the DWLS weight) with lavaan's. A
  fixture in the categorical lane would localize it; it needs the continuous
  means and variances in the NACOV, and the wider cases (up to 36 variables)
  a compact NACOV to stay under the 1 MB file limit.

### Score/inference adapter follow-ups

The first-class ML/FIML/NT-ML2S score primitives, R inference snapshots,
explicit calibration/resampling, and covariance/Wald composition are implemented;
see the roadmap's reusable-score entry and `r-package/examples/scores.R`.

- **M — extend shared inference beyond the continuous NTML path.** Complete-data
  structured expected-information score/GOF, exact empirical nested LR and
  covariance/Wald now share native contribution and geometry owners; see the
  roadmap and `r-package/examples/inference_reuse.R`. Extend persistent reuse to
  observed-bread score/covariance, delta-nesting and additional nested Gamma
  conventions, then FIML/ML2S evaluation-point-specific influence ingredients.
  Preserve each path's centering, group scaling and finite-sample conventions;
  do not equate score and LR spectra merely because statistics coincide.
- **L — add estimator-specific score/estimating-function adapters.** Ordinal,
  mixed, two-level and estimated-weight/regularized ML2S paths retain their
  existing interfaces. Extend the same reusable ingredients when their
  sensitivity, influence and normalization contracts are independently tested;
  do not label arbitrary estimating functions likelihood scores.
- **M — `compute_satorra2000` rejects well-identified models with badly scaled
  variables.** Its SPD and pivot checks on the pooled expected information `P`
  are relative to the largest eigenvalue/pivot (`1e-10 * max`), so they are
  not scale-invariant. Found by `experiments/research/evidence/52-robust-calibration-battery`
  on Kline's two-group Lynam path model (`kline_2023_ch12_lynam_indirect`),
  whose variable variances span a ratio near 300 (IQ scale against 0-1 scales).
  `robust_nested_lrt()` failed with `InfoMatrixSingular` ("rank 32/33", rcond
  ~1e-10) in about a fifth of normal draws at n = 300, while `policy_nested()`
  on the same fits was fine. Fix: judge rank on the diagonally scaled
  `D^{-1/2} P D^{-1/2}` (or on the whitened restriction problem), and add a
  test that rescales a variable by 100 and expects an identical statistic.
- **M — `score_components(ctx0, H1 = fit1)` nested score disagrees with
  `policy_nested()` and `lavTestScore()` for some constant restrictions.** On
  Kline's Worland SR model (`kline_2023_ch15_worland_sr_step2a`) with H0 adding
  `r1 == 0; r2 == 0` on the two `Risk` structural paths, the projected
  `score_components` statistic differs from both references (e.g. 3.68 against
  4.04 at n = 300, and a different two-point spectrum), while `policy_nested()`
  matches `lavTestScore()` to 1e-6. The same route agrees exactly for
  `cu == 0` residual-covariance restrictions (Little's 2x3 longitudinal CFA),
  equalities between parameters (growth residual variances, a cross-group
  path), and globally. Diagnose the H1-partable restriction map / nuisance
  projection in `score_components_impl` for structural-path constants.

### Uniform prepared-model/data interface in R

- **L — extend prepared R ownership to the remaining specialized paths.**
  Continuous ML/ULS/GLS/WLS/DWLS, FIML, ordinal ULS/DWLS/WLS and mixed DWLS/WLS
  now use `prepare_model/data/weight` plus `estimate`; see the
  [rollout status](../design/r-model-preparation.md). Extend the same ownership
  to ML2S/Stage 1, two-level/cluster summaries, FC-SEM, SAM and frontier methods
  before deprecating their legacy entry points. Migrate experiment callers
  without changing statistical procedures or benchmark timing boundaries.
- **M — reuse categorical stage-one work when preparing Gamma.** The data stage
  computes moments without weights, but the existing weight builders recompute
  moments while obtaining the score ingredients for Gamma. Expose reusable
  stage-one ingredients in core; do not duplicate that SEM logic in R. Profile
  model, data, weight and fit stages separately. The core also retains ordinal
  layout validation/preparation inside numerical fit composers.
- **M — reconcile estimate-only behavior and reusable post-fit work.** Audit
  automatic two-level SE/H1 calculations, the current FIML H1 attachment, and
  SAM's SE default against the explicit inference contract. Provide explicit
  ownership for reusable dataset-level H1/Stage-1 quantities; do not remove
  results relied on by existing consumers without a compatibility plan.


### Ordinal DWLS Gamma influence performance

The complete-data all-ordinal local diagonal assembly, direct influence, and
local finite differences landed on 2026-09-09. The numerical contract,
validation, and before/after timings are recorded in the
[roadmap](../architecture/roadmap.md#ordinal-dwls-gamma-performance). The
[benchmark](../../benchmarks/README.md#ordinal-dwls-gamma-influence-profiling)
now includes fitting and complete IJ reporting. At 18 binary indicators the
complete IJ takes about 2.6 ms at n=300 and 5.7 ms at n=1200, so the former
whole-diagonal/direct-influence bottleneck is resolved.

Remaining work, ordered by measured need:

- **M — profile the remaining multi-category cell-score cost before another
  derivative change.** At 18 four-category indicators, local FD still costs
  about 21 ms of a 26 ms complete IJ. Identify the cell-probability/threshold
  derivative work before choosing more cell reuse or analytic derivatives;
  retain probability-floor and correlation/threshold boundary behavior.
- **M — explicit reuse for repeated post-fit/profile calls.** Both Gamma
  channels are independent of the fitted SEM but are recomputed per public
  call. Add a caller-owned prepared result when repeated profile/score work
  warrants it, with ownership/invalidation tied to the data, category levels,
  thresholds, correlations, and derivative settings. The current private
  workspace caches within each call; no persistent cache or automatic
  inference during estimation has been introduced.
- **M/L — observed/missing, mixed, and full-WLS performance slices.** Measure
  their Gamma channels before transferring the complete-data optimization;
  support/overlap normalization and full-Gamma cross-pair entries need their
  own validation. Worker scaling is a separate measurement if simulation
  throughput remains limiting after the serial improvement.

### Ordinal weight storage and workspace cleanup

The structure-aware whitening fix is recorded in the
[roadmap](../architecture/roadmap.md#ordinal-weighted-ls-whitening-is-structure-aware).
Remaining memory and workspace work:

- **M — give `W_dwls` a diagonal storage type.** `data::OrdinalStats::W_dwls`
  and `OrdinalGammaCacheBlock::w_dwls` are still `Eigen::MatrixXd` holding a
  provably diagonal matrix (every construction site is `Zero(n,n)` with only
  `(k,k)` written), ~15 MB per block at p=50. About 100 references across 10
  files including `r-package/src/fit.cpp`, so it wants a quiet tree.
- **S — `build_joint_profiled_workspace` no longer needs `Ws` and `factors` as
  separate arguments** now that both are structure-aware; the weight is
  recoverable from the factor.

### Naive ordinal path builds and inverts a full NACOV it never uses — PARTLY FIXED

Found 2026-09-17 while re-measuring the above. `fit_model(model, data, estimator =
"DWLS", ordered = ...)` reached `data_ordinal_stats_from_df` without passing
`full_wls_weight`, so it took the default `TRUE` from the signature at
`model_data.R:313`. That builds the dense mdim x mdim NACOV **and inverts it**
to form the full WLS weight, for an estimator that needs only the diagonal.

The ordinal branch of `fit_model()` now passes
`full_wls_weight = identical(estimator, "WLS")`. Naive DWLS at p=50, N=1000 went
from 1396 ms to 541 ms. All ordinal R examples still pass, including
`ordinal_dwls_wls.R` (which exercises the WLS branch that does need the full
weight) and `profile_lrt_parameter_ordinal.R` (which reads `fit$ordinal_stats`
post-fit).

At p=50, N=1000, three-factor CFA:

- `data_ordinal_stats_from_df(full_wls_weight = TRUE)`  1270.8 ms
- `data_ordinal_stats_from_df(full_wls_weight = FALSE)`  501.9 ms
- staged equivalent, `prepare_data` + `prepare_weight(full = FALSE)`  81.3 ms

So the naive call wasted 769 ms on the unused inverse, and a further ~420 ms
building a dense Gamma that the diagonal materialization plan avoids. Total
naive `fit_model()` was 1396 ms against 178 ms for the staged path doing the same
work. This is why the *naive* ordinal speedup decayed (8.8x at p=12 to 1.2x at
p=50) while the pipeline speedup is flat.

Remaining: the other ~420 ms. `data_ordinal_stats_from_df(full_wls_weight =
FALSE)` still costs 501.9 ms at p=50 against 81.3 ms for `prepare_data` +
`prepare_weight(full = FALSE)`, because it materializes the full dense NACOV
(`NACOV = n * B_inv * INNER * B_inv'` at `cpp/src/data/ordinal.cpp:4075`) where DWLS
needs only its diagonal.

**Rerouting `fit_model()` through the staged handles was tried on 2026-09-17 and
rejected.** It works and it is a large win at high p, but it is a regression at
the sizes most models actually have, and it is not behaviour-preserving:

| p (N=1000) | direct | staged | |
|---|---|---|---|
| 6 | 5.7 ms | 6.5 ms | 1.14x **slower** |
| 12 | 13.4 ms | 13.9 ms | 1.04x slower |
| 18 | 22.3 ms | 20.6 ms | 0.92x |
| 30 | 72.0 ms | 46.7 ms | 0.65x |
| 40 | 200.1 ms | 82.0 ms | 0.41x |

Crossover is p ~ 14. The fixed cost is `prepare_model` (4.5 ms at p=6), which
replaces model building that `fit_dwls_ordinal` does internally, leaving ~0.6 ms
of extra object plumbing. Two further blockers, both verified:

- `audit$active_set` changes length, from the full free-parameter vector (24 at
  p=6) to the profiled one (6). All-zero in the cases tested, so nothing moved,
  but it is an observable reporting change — and it reflects a **pre-existing
  inconsistency between `estimate()` and `fit_dwls_ordinal()` that is worth
  fixing on its own terms**.
- `estimate()` rejects string bounds presets, while `fit_model(..., bounds =
  "pos.var")` works today through `bounds_arg`. Resolving a preset needs the
  augmented partable and, for `bounds_standard`, the sample statistics the fast
  path deliberately does not build. A fallback would leave two routes with
  different audit shapes selected by an argument.

So the fix belongs in the stats constructor, not the caller:

- **M — let `ordinal_stats_from_integer_data` assemble only the Gamma diagonal.**
  The efficient local diagonal assembly already exists (landed 2026-09-09) but is
  reachable only through the gamma-cache/prepared API, which consumes prepared
  polychorics rather than raw integer data. Note the naive `diag(B_inv INNER
  B_inv')` shortcut is still O(m³) and buys only ~2x; the real win needs the
  block structure, since `A22_inv` is diagonal and `A11_inv` is block-diagonal
  per item.
- **S — reconcile the `active_set` audit shape** between the staged and direct
  ordinal fit entry points.

`private/oslo-psychometric-gathering-2026/tools/check_naive_ordinal_route.R` (outside this
repository) is the behaviour gate used for the attempt: it snapshots whole fit objects across
DWLS/ULS/WLS, binary, multi-group and listwise cases and diffs them recursively.

### Continuous moment-quadratic weight follow-ups

Structured `BlockWeight`, scalar GLS, structured Stage-2 weights and the
whitened expected-information assembly are shipped; contracts, performance
checks and guard tests are in the
[roadmap](../architecture/roadmap.md#continuous-moment-quadratic-weights).
Remaining:

- **Continuous DWLS still arrives Dense from the R boundary.**
  `prepared_weight_impl` hands `prepare_weight(method = "DWLS")` across as a
  bare matrix. Pass its diagonal through the R glue and `prepared.hpp` so the
  fitter can retain diagonal whitening.
- **Converge `detail::WhitenFactor` and `gmm::BlockWeight` only with a clear
  factor/weight contract.** The first represents F and the second W; their
  same-named diagonal factories take different quantities. Preserve the
  working ordinal consumers during any consolidation.
- Batched triangular-solve optimization for the residual/Jacobian shape
  remains in the [speculative backlog](speculative.md).

### Other estimation and inference follow-ups

Small open items surfaced while fixing the standardized-solution and Kline/Guo
parity bugs (the fixes themselves are recorded in the test ledger; the ADF
`spectral_truncate` follow-up moved to [speculative.md](speculative.md)). The
post-fit covariance-admissibility audit is now part of the architecture
contract and fitted-result schema; experiment-specific simulation checkpoints
should copy `fit$diagnostics$admissibility$admissible` into their result rows
when they next change.

- **S — unexplained ULS statistic on Mplus `chapter6_ex6_10`.** lavaan and
  magmaan both report a ULS chi-square near zero for this corpus case, while
  lavaan's ULS test statistic is 38.3. Find which convention each number
  follows before gating either.

- **S — complete covariance-admissibility validation plumbing.** Add a
  deterministic lavaan warning-status fixture for an improper complete-data
  solution, and thread the audit flag into simulation result checkpoints when
  those experiments next change. Ordinal and native FC-SEM fits remain outside
  the ordinary `MatrixRep` finalizer and need a separate attachment decision.

- **M — harden the frontier PSD-ML slice after the correctness seed.**
  Completed 2026-09-24: an explicit core ordinary-first recovery entry point,
  `estimate::frontier::fit_ml_psd_fallback`, with thin R wrapper
  `frontier_fit_ml_psd_fallback()`. Accept ordinary L-BFGS only when accuracy
  and admissibility pass; otherwise try PSD-SLSQP once, warm-starting from
  usable ordinary estimates or using the original start after an error.
  Both attempts and failure reasons remain inspectable; neither rejected
  estimates nor failed PSD recovery are reported as accepted fits. Ordinary
  defaults are unchanged. C++ branch/acceptance tests and the R example
  cover the contract. Broader performance and reliability comparisons of
  this policy remain separate empirical work.

  `estimate::frontier::fit_ml_psd` and R's `frontier_fit_ml_psd()` now solve
  complete-data ML with Cholesky-lifted `Theta`/`Psi` blocks through SLSQP; the
  compact gates cover an interior ordinary-ML equivalence, a deterministic
  negative-residual Heywood case, a joint-indefinite 3x3 `Psi` whose pairwise
  checks all pass, fixed-zero reduced-LISREL structure, a shared residual
  variance, an optional SLSQP/IPOPT interior cross-check, and the R result
  contract.
  Experiment _archive/psd-ml-timing supplies the first timing panel for ordinary L-BFGS, direct
  PSD-SLSQP, and an ordinary-audit/warm-refit policy. On its deterministic
  interior CFA scaling cases, direct PSD fitting cost about 2.5x/6.9x/59x
  ordinary NTML at 3/6/12 indicators; an independent `p=12` replay gave 60x.
  Audit-first stayed at approximately 1.0x whenever the ordinary solution was
  admissible, while its five deliberately inadmissible fixtures cost about
  2.2x--3.6x because both fits ran. A three-repetition `p=24` direction probe
  reached roughly 711x and is not formal evidence. The first performance repair
  now compiles the structural covariance graph into independent Cholesky
  components and evaluates both the lifted objective and link Jacobian with
  sparse rank-two formulas. The matched 30-repetition rerun reduced direct-PSD
  overhead to about 1.8x/2.3x/3.7x ordinary NTML at 3/6/12 indicators, cutting
  the `p=12` PSD time by 15x; the directional `p=24` ratio fell to about 5.0x.
  Audit-first remains the practical policy. If more speed is needed, the next
  target is analytical elimination of duplicate original covariance coordinates
  and their links when no fixed/shared/general-equality semantics require them.
  The advisory continuous-corpus audit covered 97 checked-in Little, Newsom,
  Geiser, Mplus, Kline/Guo, and paper model/data summaries (before the
  2026-09-25 source-fidelity rebuild; on the corrected 102 fixtures only the
  two Geiser cases are inadmissible). Ordinary NTML was
  covariance-inadmissible in six: four Little and two Geiser cases. All 97
  PSD-ML refits converged and were admissible; the 91 interior cases agreed with
  ordinary NTML to \(3.21\times 10^{-6}\) in the parameters and
  \(7.18\times 10^{-12}\) in the objective. Keep ordinary NTML plus its audit as
  the core default and PSD-ML as an explicit warm refit. Do not add silent
  automatic fallback before there is a boundary-inference policy and a concrete
  R consumer. The runner is `cpp/tests/checks/psd_ml_corpus/`; the decision record
  is `psd_ml_corpus_audit.tex`.
  The De Jonckere--Rosseel / Ernst small-\(N\) benchmark now supplies the
  convergence stress evidence (`experiments/research/evidence/42-psd-ml-small-n-convergence/`):
  at \(N=10\), direct PSD-ML audit-converged in 96.8% of 1,000 replications
  versus 54.1% for ordinary L-BFGS NTML. The same-SLSQP comparison still
  strongly favors PSD-ML, ruling out a backend-only explanation. Do not
  promote it to an automatic small-sample cure: 86.1% of the \(N=10\) PSD
  fits were rank-boundary solutions and the raw structural coefficient had
  severe tails. Those tails are not themselves an optimizer defect: if a
  boundary solution is the constrained likelihood maximum, PSD-ML should
  report it. Bayesian or penalized stabilization is a separate project already
  recorded in [speculative.md](speculative.md). The subsequent validation
  question was whether the returned KKT point was the best attained maximum
  across a reasonable start portfolio, and whether equal-objective solutions
  had the same implied moments.
  Equality-constrained terminal auditing now uses primal feasibility and the
  Lagrangian KKT residual, and the experiment runner records the KKT residual,
  raw gradient, equality violation, and constraint-Jacobian rank. The full
  1,000-replication artifact was refreshed under that audit on 2026-07-30 and
  reproduced the \(N=10\) 96.8% versus 54.1% headline exactly. The two clean
  PSD solver returns rejected by the audit were equality-feasible and
  covariance-admissible but genuinely nonstationary.

  Ordered PSD-ML follow-up queue:

  1. **Completed 2026-07-30.** Central finite-difference gates now exercise
     the production lifted objective and constraint Jacobian for shared
     covariance coordinates, general-linear/nonlinear equalities,
     multi-group blocks, and mean structures. Deterministic exact-model fits
     cover each geometry in `cpp/tests/unit/psd_ml_test.cpp`.
  2. **Completed 2026-07-30.** The 10.8 KB
     `cpp/tests/fixtures/psd_ml/corpus_geometries.json` slice and
     `cpp/tests/unit/psd_ml_corpus_test.cpp` now gate the same-fit
     level/residual reallocation and the materially changed linear-growth fit
     (since 2026-09-25 labelled synthetic: they were mistranslations of
     Little's NegAFF growth models), the Geiser second-order negative
     disturbance, and the Geiser joint-indefinite quadratic growth covariance
     (regenerated from the corrected Geiser model). The full scan remains
     advisory and outside default CI.
  3. **Completed 2026-07-30.** The opt-in IPOPT extensions to experiments _archive/psd-ml-timing
     and research/42 compare the same lifted model across backends. In the
     30-repetition deterministic panel, SLSQP and IPOPT reached the same
     admissible optima in all nine cases, but direct IPOPT cost a median 75.7x
     ordinary NTML across the interior CFA cases, versus 2.37x for
     PSD-SLSQP—about 40x slower backend-to-backend. In the paired
     200-replication \(N=10\) stress cell, PSD-SLSQP was usable in 95.0% versus
     81.5% for IPOPT; IPOPT rescued 0.5% and harmed 14.0% relative to SLSQP at
     a median 57x time ratio. Thirty-one of its 37 failed returns were
     `Invalid_Number_Detected`, consistent with a formulation/backend mismatch:
     complete-data ML can be non-finite at singular observed-covariance or
     structural-transform trial points, NLopt can reject those points with an
     infinite objective, and IPOPT expects smooth finite callbacks throughout
     its bounded domain. Keep IPOPT as an optional deterministic cross-check,
     not a production PSD-ML default. Before reconsidering it, prototype a
     principled finite-domain formulation and reuse solver/factorization setup;
     exact Lagrangian Hessians plus callback-stage and reliable IPOPT iteration
     telemetry are secondary follow-ups.
  4. **Completed 2026-07-30.** Experiment engineering/evidence/12-psd-ml-basin-audit
     (`experiments/engineering/evidence/12-psd-ml-basin-audit/`) implements the self-contained
     multistart audit on the \(N=10,20,50\) stress cells. The pilot screened
     1,000 datasets per cell and replayed the prespecified 100-dataset random
     core plus default failures and extreme estimates with at most 13 starts.
     In the random cores, the default fit reached the best attained admissible
     KKT objective in 76%, 94%, and 100%; 22% and 5% at \(N=10,20\) were
     eligible but inferior stationary points rather than mere failed returns.
     Initial competing-basin rates were 51%, 23%, and 6%. Tight
     20,000-iteration representative restarts retained competing clusters in
     44, 20, and 3 of the 100 random-core datasets. All three initially flagged
     equal-objective/different-moment cases collapsed under that restart.
     Bidirectional fixed-\(\beta\) profiles found at least two minima in the
     feasible lower envelope for 9 of 13 targeted persistent cases. These are
     competing feasible KKT basins and strong local-maximum evidence, not a
     proof that the best attained member is global. Small-\(N\) convergence
     reports must therefore separate return/KKT success from basin hit.
  5. **Completed 2026-07-31.** Experiment research/43
     (`experiments/research/active/43-psd-ml-repair-risk/`) supplies a four-case repair-anatomy
     panel and a controlled primitive-boundary risk path without extending to
     another estimator. Its 4,000-dataset pilot keeps the observed covariance
     safely PD while varying the smallest latent-covariance eigenvalue over
     `0,.001,.01,.05,.20`. Ordinary NTML was inadmissible 1,540 times; every
     ordinary-warm PSD refit was KKT-converged and admissible. Conditional on
     those repairs, PSD-ML improved primitive-covariance error,
     implied-covariance error, and population Gaussian KL loss in all 1,540
     datasets, with mean improvements 0.00462, 0.00367, and 0.0354 and a median
     PSD/ordinary time ratio of 1.20. Treat this as transparent model-specific
     risk evidence for audit-first refitting, not a dominance theorem or an
     inferential result. The pilot is sufficiently decisive for this design;
     do not run the 1,000-replication full profile absent a new precision need.
  6. **Possible compact-multistart follow-up.** Design and validate a smaller
     deterministic portfolio on fresh held-out replications before exposing an
     optional methods-developer R audit/refit helper. In experiment engineering/evidence/12-psd-ml-basin-audit's pilot,
     the cumulative best-attained hit rates at \(N=10,20,50\) were
     76%/94%/100% for the default alone, 90%/97%/100% after adding the PSD
     restart and one moderate perturbation, 99%/100%/100% after eight starts,
     and 100%/100%/100% after nine. The eight-start portfolio cost roughly
     nine one-start fits but only 10--11 ms for this small model. These are
     in-sample portfolio-selection numbers, not a production guarantee; the
     next experiment should freeze a compact ordering, evaluate likelihood/AIC
     gaps and parameter changes out of sample, and measure scaling on larger
     models. Do not silently replace a one-start PSD fit: a successful KKT
     return is not a globality certificate, while boundary inference still
     needs an explicit policy. Parameter-space bounds or penalties remain a
     separate speculative estimator project.
  7. **Open (found 2026-09-24): PSD-SLSQP stalls, and a larger budget does
     not help.** Ernst design, marker identification, N = 10, 1,000 draws at
     `ba10b089` with the 2026-09-22 defaults (5,000 evaluations):
     - All 28 PSD-SLSQP failures exhaust the evaluation budget (66 of 68
       failures over N = 10 to 100).
     - With 50,000 evaluations all 28 return after 5,000 to 12,600
       evaluations, but none passes the cone or Newton certificate. Some
       stop at a worse objective than the optimum other routes certify
       (0.682 against 0.562). This looks like stalling in the Cholesky-lift
       coordinates near singular faces.
     - The same model under std.lv reaches a certified optimum in 18 of the
       28. The other 10 optima have Y's disturbance variance at zero, which
       std.lv cannot represent. The sphere PSD route (current HEAD)
       reports convergence on all 28 by magmaan's own verdict. The
       Newton/cone certificate was not applied to it.
     - Std.lv has the opposite weakness. It fails, or certifies a drifting
       point with |beta| of 340 to 13,554, whenever the optimum has a zero
       latent residual variance: 14.6% of N = 10 draws.
     - Warm-restart probe (2026-09-24, covariance paper
       `work/probes/stall_restart_probe.R`): first fit with 50,000
       evaluations, then up to five SLSQP restarts from the returned point.
       16 of the 28 pass the certificate. 13 of them keep an unchanged
       objective, so the restart only closes the tolerance. The other 3
       pass at clearly worse KKT points (for example 0.174 to 0.467),
       because the restart re-projects the start and SLSQP lands
       elsewhere. 12 never pass. Not a principled fix, and no default
       changed.
     - Next: characterize the stalled lift points (which L diagonals are
       near zero, and their multipliers). Decide whether a stall should
       trigger a refit in another identification or in the sphere.

- **M — multi-information penalty (frontier) follow-ups.** Landed
  2026-09-22: `estimate::frontier::fit_ml_multiinfo`,
  `estimate::fiml::frontier::fit_fiml_multiinfo`, and R
  `frontier_fit_ml_multiinfo()` / `frontier_fit_fiml_multiinfo()`. The
  penalty is `lambda * log det Corr(v_K)` over the complete latent-plus-observed
  vector. It has an analytic gradient through the RAM derivative, finite-
  difference, brute-force, closed-form, invariance, O(1/N), barrier,
  exclusion, fixed.x, nonrecursive, and FIML-equals-ML gates, and a default
  `lambda = 0.25` set by experiment research/47. Remaining, in order:
  1. Testing with a barrier estimate. Experiment research/47 (2026-09-25)
     settles the pairing: Browne's residual statistic
     (`inference::rls_chi2`, lavaan's `browne.residual.nt.model`) at any of ML,
     PSD-ML, or a barrier estimate gives the same rejection rate and is better
     calibrated at N = 50 than the LR test (3.8 to 6.5% against 5.1 to 12.6%).
     The unprojected RLS quadratic minus Browne's statistic
     (`frontier::nt_moment_quadratic` minus `rls_chi2`) is the score for the
     estimator's displacement and tests admissibility against the single-face
     law `(t(U) - U)^2` (PSD-ML: `U^2 1{U < 0}`), 3.8 to 6.2% on the face,
     conservative inside, 85% power at N = 50 against an improper residual
     variance that no unconstrained test sees. Improper pseudo-true values
     converge to the face under both barrier and PSD-ML, so the ordinary-ML
     audit is no longer needed as a companion. Remaining: a multi-face
     reference law, and wiring the pair as a frontier test entry point.
  2. Inference at the penalized estimate. Compare the current ordinary
     information SEs with a penalized-Hessian sandwich, `(H + lambda P'')^-1`
     bread, when `lambda / N` is not negligible (N <= 100 near the boundary).
  3. Latent-determinacy target. Landed 2026-09-24 as
     `PenaltyTarget::Determinacy` (R `target = "determinacy"`), with
     brute-force, identity, gradient, domain, manifest, exact-latent,
     invariance, and FIML gates. The research notes derive it from five
     requirements (distribution-only, gauge invariance, chain rule,
     complementarity to the likelihood, barrier at the reachable faces) and
     calibrate `lambda` by a local limit at a face: matching the posterior
     median gives `lambda` in [0.10, 0.43] for populations 3 to 0 standard
     errors inside the face, and `lambda = 0.25` is optimal at 1.6. Layer
     weights are not supported by that calibration, which replaces the
     two-weight direction. Experiment research/47 now carries the new target
     at the joint barrier's five weights on the same datasets, with truths on
     faces and improper truths added (2026-09-25). It is never less accurate
     than the joint barrier (30 design-weight cells at N = 50) and keeps
     better coverage at `lambda = 1`; the earlier "joint ahead on Heywood
     designs" came from scoring the minimum of two residual variances, a
     downward-biased functional, now replaced by the residual variance of x1.
     Against PSD-ML it wins with the truth inside and loses on the face by the
     predicted RMSE factor `sqrt(1 + 2 lambda)`; the local limit `t(u)` matches
     the simulated estimates within 0.19 SE at `lambda = 0.25`. Engineering
     (resolved 2026-09-25): the barrier fits that failed in NLopt L-BFGS line
     search converge under PORT, now the barrier default (experiment
     _archive/barrier-optimizer). The failures shared by every optimizer are Ernst
     marker-chart fits near a marker pole. Lane barrier-ml (decisions/02,
     2026-09-27) kept FABIN3 + PORT as the default against the layered start
     and L-BFGS. Remaining, for `papers/sem-barrier`: score it on the research/48
     posterior bank; run the research/48 path cells; then decide the default
     target. Experiment research/48's Jeffreys posterior
     median remained the most accurate estimator on every design at N = 50.
  4. Scope. LS/ordinal paths need a discrepancy-specific scaling in place of
     `l = -N * fmin`. Nonrecursive models have a proved barrier only when the
     zero-residual variables are sinks. The characterization in the header
     comment should be promoted to a note before any paper claim.

- **XL — covariance-honest SEM paper and optional uniform parameter
  inference.** The independent exploratory paper project is
  `papers/covariance-honest-sem/`. Its required contribution is the existing
  complete-data NTML estimator, admissibility contract, corpus evidence,
  small-sample convergence study, and basin audit; boundary-aware inference is
  explicitly optional. If inference proceeds, keep the first claim to correctly
  specified complete-data Gaussian NTML with a unique locally identified
  optimum and positive-definite implied observed covariance. Target coverage
  that is uniform across interior points, PSD-boundary rank strata, and
  local-to-boundary eigenvalue sequences, rather than merely plugging an
  estimated active set into pointwise cone asymptotics. First map the SEM
  parameter space and NTML local expansion into the assumptions of the recent
  proximal/uniform constrained-extremum results. Only then implement a compact
  comparison of ordinary Wald, plug-in cone, ordinary and directional
  bootstrap, and proximal/local-quadratic confidence sets. Full
  profile-likelihood intervals and likelihood-ratio tests remain deferred.

- **M, sem-psd corpus timings.** The supplement's accepted timing bank used the
  mistranslated corpus. Rerun it pinned on the corrected fixtures; the
  corrected 102-case PSD audit has two inadmissible ordinary fits (both
  Geiser), not six. `psd_ml_corpus_audit.tex` still reports the old counts.

- **M/L — extend covariance-honest point estimation estimator by estimator.**
  The lift is now internally estimator-neutral and the first non-ML slice is
  landed: `fit_gmm_psd` covers continuous ULS and caller-fixed WLS/ADF weights,
  `fit_gls_psd` builds the ordinary sample-based GLS weight once, and R has explicit ULS/GLS/WLS frontier
  wrappers. The correctness seed finite-differences
  the lifted fixed-weight objective and links, checks ordinary/PSD interior
  agreement for all three objectives, repairs one exact-fit ULS Heywood case,
  and gates the R result schema. The broader stress program is specified below;
  it reuses NTML only as an anchor rather than silently copying the much larger
  the archived PSD timing and retained basin/stress studies onto every estimator.

  Ordered computational queue (point estimation only). PSD two-level ML and
  native FC-SEM are deliberately outside this extension: they are independent
  large architecture projects, not blockers for the supported MatrixRep
  estimator family.

  2. **Done 2026-08-19 — FIML.**
     `estimate::fiml::frontier::fit_fiml_psd` feeds the lifted moments and
     analytic Jacobians through the existing immutable pattern cache and
     observed-pattern likelihood. Every `Sigma_oo` must be PD, and the existing
     fixed-x missingness policy remains authoritative. Focused C++ gates cover
     central-difference derivatives, all-observed reduction to PSD-NTML and
     ordinary FIML, agreement at an interior missing-data optimum, and repair
     of a missing-data negative-residual solution; the exported
     `frontier_fit_fiml_psd()` wrapper gates the R fit schema and retained raw
     missingness. Broader missing-pattern/corpus, timing, and basin validation
     remain part of the future estimator-specific validation discussion, and
     boundary inference remains explicitly out of scope.
  3. **Done 2026-08-19 — ML2S.**
     `estimate::fiml::frontier::fit_ml2s_psd` consumes the saturated EM moments
     unchanged. Its NT member dispatches Stage 2 to `fit_ml_psd`; ULS, DWLS,
     ADF, and fixed-`a` DLS freeze the existing Stage-2 weight and dispatch to
     `fit_gmm_psd`. The raw overload builds the same ordinary saturated first
     stage. Focused gates cover all five interior reductions, unchanged EM
     moments, and repair of an improper Stage-2 solution. R exposes
     `frontier_fit_ml2s_psd()` and retains the Stage-1/raw objects, but does not
     attach ordinary ML2S inference at a possible covariance boundary.
  4. **Done 2026-08-19 — ordinal and mixed-ordinal ULS/DWLS/WLS.**
     `estimate::frontier::fit_ordinal_psd` and `fit_mixed_ordinal_psd` apply the
     lift after their ordinary partable preparation for ULS, DWLS, and WLS,
     cover delta/theta, preserve threshold/scale and continuous-moment
     semantics, and return prepared ordinary parameter vectors with covariance
     diagnostics. Estimated polychorics/polyserials, continuous moments, and
     fixed NACOV-derived weights are consumed unchanged: PSD is a fitted-model
     constraint, not a silent Stage-1 repair. Central differences gate all
     transformed mixed moment derivatives; interior fits reduce to the
     ordinary estimator for all three weights, indefinite Stage-1 association
     matrices remain accepted by ULS, an improper mixed residual solution is
     repaired, and both R wrappers retain their original Stage-1 objects.
     Broader conditioning, multi-group, timing, and basin validation remains
     part of the future estimator-specific validation discussion.
  5. **Done 2026-08-19 — categorical ML.**
     `estimate::frontier::fit_catml` and `fit_catml_psd` implement the
     limited-information cML criterion: the normal-theory ML discrepancy on
     the Stage-1 polychoric correlation matrix under a saturated threshold
     structure. The analytic correlation-standardization Jacobian is shared by
     the ordinary and lifted objectives. The input polychoric matrix must be PD
     because its inverse/log determinant domain is intrinsic; it is rejected,
     never projected. The PSD fit constrains only fitted primitive covariance
     blocks. Central-difference, interior-reduction, non-PD-input, unchanged-
     Stage-1, and R-schema gates cover the slice. Broader boundary-geometry
     cases remain part of the future estimator-specific validation discussion.
  6. **L/XL — recover specialized algorithms where warranted.** The landed
     continuous slice scalarizes the least-squares objective for SLSQP/IPOPT.
     Ceres and the current SNLLS/Golub–Pereyra paths cannot accept the nonlinear
     Cholesky-link equalities by a nominal switch. A fast PSD-SNLLS or direct
     elimination formulation is a separate computational project and should be
     justified by scaling evidence.
  7. **S — robust-estimator naming and dispatch.** MLM/MLR/MLMV-style complete-
     data variants share the NTML point estimate, so they need no new point-fit
     optimizer; document and dispatch them to PSD-NTML when only estimation is
     requested. Missing-data robust ML instead depends on the PSD-FIML item.

  At singular-but-PSD component solutions, ordinary inverse-information
  SEs/tests are nonregular. Keep inference outside every automatic constrained
  fit until an explicit boundary-aware policy is available.

- **L — execute the broader covariance-honest point-estimation stress track.**
  The independent `experiments/engineering/active/13-psd-estimator-stress/` leaf now owns the
  common structural geometries, paired ordinary/PSD fits, selective restart
  logic, and result schema. Its first two-replication smoke completed 128 fit
  attempts across eight registered families: all 125 returned fits matched an
  independently recomputed objective and completed both geometric projections,
  all 69 cone-stationary PSD fits were covariance-admissible, all 28 ML2S
  Stage-1 fingerprints were unchanged, and
  both non-PD CatML sentinels were rejected as intended. Checkpoint/resume,
  task subsetting, metadata, paired summaries, and the smoke report are live.
  This validates the harness, not stochastic performance.

  The first 100-replication pilot tranche is also complete for continuous
  complete-data estimation. It covers 23 sparse one-axis cells over the
  one-factor, correlated-block two-factor, and latent-regression structures,
  including the `0.20, 0.05, 0.01, 0.001, 0` covariance-eigenvalue sequence,
  one joint boundary, and sample-size ratios `2, 5, 10, 50`. Its 20,700
  attempts produced 20,406 returned fits with no independent-objective
  failures. All 11,424 driven/lifted-audit-stationary covariance-honest fits
  were admissible;
  the 76 nonstationary constrained returns were confined to GLS in the
  correlated-block boundary cells. All 2,353 eligible interior pairs passed
  the objective and fitted-moment agreement gates. A terminal covariance-link
  round-trip defect revealed by the first run is fixed and promoted to a C++
  correlated-block GLS regression test.

  The second continuous tranche adds a misspecified two-factor model, anchor
  and near-boundary two-group models with a shared residual variance plus one
  general linear equality, and fixed-WLS weights with realized condition
  numbers `1, 1e4, 1e8`. Across the resulting 29 cells, 23,506 of 24,000 fits
  returned and every returned objective passed independent recomputation. All
  13,082 of 13,297 returned covariance-honest fits passed the common PSD-cone
  audit and every one was admissible, all 893 cone-stationary misspecified fits
  had nonzero discrepancy, all 1,797 returned
  grouped fits honored both equalities to at most `4.44e-16`, and all 3,253
  eligible interior pairs passed the agreement gates. Weight conditioning is
  the first sharp limit: ordinary fixed WLS returned in `100%, 3%, 0%` of the
  `1, 1e4, 1e8` cells; PSD WLS returned throughout but was cone-stationary in
  `100%, 100%, 6%`, compared with `100%, 100%, 8%` under the lifted audit. The
  PSD normal cone converted 2,824 ambient failures into stationary fits; 43
  returns passed the lifted audit but failed the common cone audit, with no
  reverse cases. All 23,506 returned fits had finite common-coordinate
  gradients and completed both normal projections. The weights retained their
  requested spectra and full effective ranks, ruling out silent rank
  truncation as the explanation.

  Next move from their existing deterministic smoke anchors to calibrated
  FIML/ML2S and categorical stress slices. Do not form a full Cartesian
  product: vary one stress axis at a time and add interactions only for
  prespecified risky pairs or failures seen in the pilot. This is computational
  validation, not an inference, coverage, or estimator-ranking study.

  The common structural panel is:

  1. an interior one-factor CFA with diagonal `Theta`;
  2. a two-factor CFA with a correlated-residual block and correlated `Psi`, so
     joint PSD rather than diagonal nonnegativity is material;
  3. a latent regression with a disturbance variance approaching the boundary
     and an observed mean structure;
  4. a focused two-group version with shared covariance coordinates and one
     general equality, used only at the anchor and near-boundary settings.

  For each applicable model, vary the target smallest primitive-covariance
  eigenvalue over `0.20, 0.05, 0.01, 0.001, 0`, and use sample-size-to-free-
  parameter ratios near `2, 5, 10, 50`; for categorical cells use
  `N = max(100, round(ratio * n_free))`. Construct the populations on a
  standardized scale so the eigenvalue sequence is comparable. The
  zero-eigenvalue population remains valid only when the implied observed
  covariance is PD. Move `Theta` and `Psi` toward the boundary one at a time,
  plus one joint-boundary interaction, so failure rates can be attributed to
  the owning block. Add one misspecified model per data family to ensure the
  constrained fit is not validated only at exact fit. Normal data are the
  baseline; skewed/heavy-tailed draws appear only in the empirical-ADF/DWLS/DLS
  weight slice, where higher moments affect the computation.

  Estimator-specific slices are mandatory rather than crossed indiscriminately:

  - **Continuous complete data:** ML is the anchor; pair ordinary/PSD ULS, GLS,
    one fixed nonidentity WLS weight, and fitted expected-information GMM.
    Separately perturb weight condition number over roughly `1, 1e4, 1e8` and
    record the effective rank/conditioning actually used.
  - **FIML:** cross complete, monotone MCAR, fragmented MCAR, and covariate-
    driven MAR patterns at approximately 20% and 40% missingness. Record the
    number and smallest frequency of observed patterns. The all-observed cell
    must reduce to the complete-data ML anchor.
  - **ML2S:** reuse the identical saturated-EM object for ordinary and PSD
    Stage 2 and exercise NT, ULS, DWLS, ADF, and fixed-`a` DLS in the anchor
    cells. Use NT plus the worst-conditioned empirical weight in the expanded
    missingness grid. Hash Stage-1 means, covariances, ACOV, warnings, and
    regularization diagnostics before fitting and require exact preservation.
  - **Ordinal and mixed ordinal:** cross 2/3/5 categories, balanced versus
    sparse thresholds, delta/theta parameterizations, and all-ordinal versus a
    50% ordinal mixed panel. Run ULS throughout; add DWLS/WLS on the anchor,
    sparse-threshold, and worst-conditioned NACOV cells. Record polychoric/
    polyserial and weight spectra separately from fitted-model covariance
    geometry.
  - **CatML:** use only PD polychoric inputs, stratified as well-conditioned,
    near-singular, and barely-PD, under its supported delta parameterization.
    A paired non-PD sentinel must be rejected as an expected Stage-1 domain
    outcome and must not be counted as optimizer failure or repaired. Recompute
    the correlation-standardized ML criterion independently from the returned
    implied covariance.

  Every fit row must retain the dataset seed and design identifiers; ordinary
  and PSD return codes; solver convergence; the driven/lifted terminal audit,
  common full-model ambient and PSD-cone residuals; equality violation;
  objective and an independently recomputed
  objective; minimum eigenvalue/rank for every primitive block and implied
  observed covariance; boundary status; parameter and implied-moment error;
  Stage-1-domain status/fingerprint where applicable; objective/gradient
  evaluations; elapsed time; and optimizer/backend version. Summaries must keep
  **solver return**, **driven/lifted stationary**, **cone stationary**,
  **covariance-admissible**, and **best-attained basin** rates separate.
  Report paired medians and tail
  quantiles, not only aggregate success rates, and attach binomial intervals to
  stochastic rates.

  Use four validation layers:

  1. **Deterministic sentinels:** add focused C++ fixtures for any structural
     geometry not already covered (especially mean-plus-multi-group and CatML
     boundary geometry). Hard gates are zero Stage-1 mutation, exact expected
     domain classification, finite audit telemetry, and covariance
     admissibility for every accepted PSD fit.
  2. **Smoke:** two fixed seeds per planned cell, used to validate checkpoint,
     resume, pairing, objective recomputation, and report generation. Smoke
     results are never scientific evidence.
  3. **Pilot:** 100 replications per one-axis cell. Interior ordinary/PSD pairs
     must agree to `1e-8 * (1 + abs(F))` in objective and `1e-5` in fitted
     moments; parameter comparisons use `1e-4` only in the same prepared chart.
     Any exceedance is inspected rather than averaged away. Any accepted but
     inadmissible PSD fit, Stage-1 mutation, or objective-recomputation failure
     is a correctness defect.
  4. **Targeted confirmation:** expand only cells with a material failure-rate,
     boundary-risk, conditioning, basin, or timing signal to at least 500
     replications. Replay a stratified 20-dataset core per flagged cell plus all
     default failures, nonstationary returns, boundary fits, and extreme
     estimates with a frozen compact start portfolio. A better feasible KKT
     objective gap above `1e-6 * (1 + abs(F))` is reported as a competing basin,
     not silently substituted or called an optimizer failure.

  Keep corpus breadth in a separate advisory
  `cpp/tests/checks/psd_estimator_corpus/` runner rather than letting experiment engineering/active/13-psd-estimator-stress
  depend on the tests leaf. Reuse the 97 continuous sample-statistic summaries
  for LS/GMM where applicable, the 17 FIML fixtures plus the two raw-data bfi
  parity cases, the 20 ordinal and two mixed-ordinal fixtures, and the existing
  real-data ordinal parity case. CatML runs only on corpus cases whose estimated
  polychoric matrix is PD; its domain exclusions are tabulated separately.
  This runner may promote a small deterministic failure slice into default CI,
  as the existing PSD-ML corpus audit does, but the full scan remains local.

  The independent checks are criterion-specific: lavaan/ordinary magmaan parity
  only where they estimate the same interior problem; direct value
  recomputation for every family; FIML-to-ML and all-observed NT-ML2S-to-ML
  reductions; groupwise objective additivity; and unchanged Stage-1 objects for
  ML2S, ordinal, mixed, and CatML. SLSQP is the production path. Use IPOPT only
  on a small regular interior cross-check because the archived PSD timing and retained basin studies already
  show that its non-finite callback handling is a poor match to the lifted ML
  formulation. Promote each newly discovered deterministic failure to the
  smallest suitable unit/golden fixture, and keep the stochastic/corpus scan
  advisory.

  The pilot report must end with estimator-by-estimator decisions: validated
  domain, expected rejections, numerical weak spots, basin evidence, median and
  upper-tail PSD/ordinary cost, and whether specialized PSD algorithms are
  justified. Treat a median cost above 5x on regular models with at most 12
  indicators, super-quadratic empirical scaling, or materially worse audited
  convergence as a trigger to profile the formulation; it is not by itself a
  correctness failure. Keep two-level ML, native FC-SEM, and all inference out
  of this track.

- **M/L — broaden and harden standardized score flips.** The frontier slice
  supports affine nested complete-data ML and direct-FIML pairs and is exercised
  by experiments research/32 and research/36. Its production probe path can now take H1 as a
  model tangent plus the H0 fit, and independently select asymptotic-only,
  effective, standardized, or full flip calibration, so future stress grids do
  not refit H1 or compute unused resampling references. Direct FIML uses
  casewise observed-pattern scores
  and conditional Fisher-information pattern strata; its all-observed reduction
  to complete ML is unit-gated. The tiny exact-enumeration kernel gate and independent dense
  per-case/hatted-nuisance oracle for the flip-specific covariance landed with
  experiment research/34; before treating the method as more than research, still probe
  near-singular nuisance information. The first expanded probe already crosses
  1/4/8 restrictions, balanced/1:3 groups, skew/t5/normal data, and several
  group-information geometries. Its next design question is to hold restriction
  df fixed while varying nuisance dimension and leverage, separating a genuinely
  high-dimensional nuisance effect from the joint-test-dimension effect seen at
  fixed 62-slot ambient dimension. Its sparse/dense power follow-up also found
  that the nominal FMG/SB power advantage vanished after matched empirical-null
  calibration: standardized/effective flips, MV-adjusted LR, pEBA4, and exact
  ALL were within roughly two percentage points on average. The next power
  design should therefore increase the null calibration budget and vary
  alternative alignment with the U-Gamma eigenvectors; raw nominal power is not
  an adequate ranking criterion while the comparator sizes differ this much.
  The direct efficient-meat sandwich comparator is now exposed and confirms the
  opposite finite-sample failure mode: full inversion is conservative and can
  be poorly conditioned even though its asymptotic reference is pivotal. The
  first multiplier-law replay now targets experiment research/32's eight difficult
  8-df severe-copula cells (N=60/200, homogeneous/geometry, VM/PL; 200
  replications and 499 draws). It exactly reproduced the old Rademacher
  p-values and their 12.4% average rejection. Gaussian weights supplied the
  intended numerical checksum against score-ALL (2.4% versus 2.7% rejection;
  mean absolute p-value gap 0.015), while centered exponential weights were
  much too conservative at 0.7%. Mammen's two-point weights were the only
  plausible initial correction, averaging 3.7% with cell rates 1.5--5.0%.
  The immediate mechanism probe is now complete on the same cells. The
  minimum-kurtosis two-point family
  (`Ew^3=gamma`, `Ew^4=1+gamma^2`) interpolated smoothly: gamma=.75 remained
  liberal at 6.1%, while gamma=.90 reached 4.8% but drifted from 3.9% at N=60
  to 5.6% at N=200. Within-information-stratum recentering was better:
  centered Mammen averaged 4.9%, ranged 3.5--6.0% across cells, and retained
  the theoretically motivated `Ew^3=1` rather than selecting a moment on the
  discovery grid. Per-draw `sum(w_i^2 u_i u_i')` studentization failed,
  rejecting 9.1% without and 16.4% with recentering; do not carry it forward.
  The independent confirmation is now complete on 12 fresh `q=8` cells
  crossing N=60/200, homogeneous/geometry information, and normal/t5/skew
  generators, with 500 null replications per cell. Centered Mammen rejected
  5.2% overall (5.4% at N=60, 4.9% at N=200; cell range 3.8--6.8%), versus
  4.5% for raw Mammen, 3.8% for direct sandwich, and 5.3% for pEBA4. A matched
  sparse loading-shift gate (`effect=.50`, 300 independent alternatives per
  cell) used each method's own empirical-null cutoff: centered Mammen had 35.6%
  power, raw Mammen 35.3%, pEBA4 36.0%, and sandwich 31.2%; the centered-minus-
  raw difference was only +0.4 percentage points and ranged -1.0 to +1.7
  points across cells. Score recentering therefore survives without a visible
  power cost, whereas gamma=.90 remains only a discovery-grid mechanism check
  and Gaussian, centered-exponential, and weighted-meat arms stay dropped.
  Keep centered Mammen experimental rather than default: it is a multiplier
  bootstrap without exact sign-randomization validity and is still
  effective-score-only because flip-specific standardization uses `w_i^2=1`.
  Do not spend another large grid on complete-data `q=8` calibration. An
  opt-in observed-sensitivity path has now landed for nested ML/FIML and global
  ML/FIML scores while preserving expected information as the historical
  default. It uses the realized likelihood Hessian in the nuisance projection
  and retains expected information as the positive-definite common quadratic,
  which is the required first-order correction at an observed-data pseudo-null
  under misspecification. It is deliberately
  effective/asymptotic-only and does not permit within-pattern centering; the
  remaining useful work is an asymptotic argument for any recentered
  estimated-nuisance rows, an explicit global-centering/stratum policy for FIML
  and ordinal/categorical scores, and a near-singular observed-sensitivity
  gate. In parallel, the bounded analytic
  question remains shrinkage of `G'B1G` itself, spanning SB's
  common-eigenvalue trace, pEBA-style spectral shrinkage, and the raw sandwich
  inverse. Research/49 now supplies a complete-data global score/LRT pilot of
  nonlinear covariance shrinkage, cycle-moment reconstruction (orders 4/6),
  and direct truncated-CGF tails, with explicit failures and an independent
  calibration-sample diagnostic. These remain experiment-local R prototypes.
  The focused `--mv` follow-up implements all-distinct U4 estimation of tau2
  with unknown means handled exactly for a fixed projection. Its 12,000-dataset
  null pilot substantially improves score MV but can make LRT liberal; moment
  feasibility violations remain frequent. Keep it experimental. The paired
  expected-versus-observed-H0 metric replay retains expected nuisance sensitivity:
  observed weighting makes corrected MV liberal (39.1–41.3% versus 4.9–5.6%
  at skewed p=12, n=100 on common usable fits) and adds 26 metric failures.
  Expected-score/LRT results reproduce exactly; keep expected weighting.
  The oracle/constraint gate is now run: the original latent-component design
  has a flat oracle spectrum, while severe Cholesky IG supplies weights up to
  5.20. In the 10,000-dataset smaller IG grid, at p=12/n=500 oracle score/LRT
  reject 4.2%/4.6%, versus corrected MV's 9.8%/9.6%. The lower-bound constraint
  does not repair that cell and stays close to SB overall. The fitted score
  trace and corrected tau2 average 92% and 61% of their oracle targets there.
  Next gates are isolating feasible-moment/nuisance bias on nonflat spectra
  and fit-selection effects (19–22% failures at severe Cholesky n=100), then
  independent calibration replication and matched-null power. Do not treat
  latent-component nonnormality or matched marginal skew/kurtosis alone as
  sufficient evidence of heterogeneous mixture weights.
  Before promotion: validate population-spectrum/resolvent recovery for
  transformed nonnormal rows, control same-sample projection error, stabilize
  high-order moments, and establish a valid tail law (a truncated polynomial
  is not automatically a CGF). Gate either proposal on null size, matched-null
  power, and meat conditioning rather than one aggregate rejection rate.
  Experiment research/44 has now closed the simple version of that gate. It exposed the
  projected score/metric/meat and compared raw and centered OPG, Hotelling, two
  predeclared vanishing meat shrinkages, and two analogous shrinkages of the
  observed sensitivity. At an exact three-df nonnormal-MAR pseudo-null the
  best shrinkage improved mean absolute size error only from 1.93 to 1.74
  percentage points, versus about 0.9 for score SB/pEBA4. At `n=200` in
  9/34/87-df latent models the raw observed-sensitivity sandwich was grossly
  liberal (10.2--99.7% across cells), and neither Hotelling nor shrinkage made
  it usable. A normal-complete diagnostic localized much of the problem to the
  noisy observed-Hessian nuisance projection: expected sensitivity gave
  4.2/5.6/2.8% rejection by df, versus 9.6/24.6/39.4% for observed sensitivity.
  Expected sensitivity is not an escape hatch because it rejects essentially
  zero times under strong nonnormal MAR. Do not tune a fixed shrinkage constant
  or carry direct-sandwich variants into the planned paper battery; retain the
  new surfaces as frontier diagnostics only.
  Experiment research/40 now supplies the first projected-Satterthwaite gate for the
  centered-meat pivotal GOF score. In a true five-indicator one-factor model
  (`df=5`, n=30/50/100/200), the centered-meat chi-square was liberal at low n
  (0.138 normal and 0.113 moderate VM at n=30), but matching the independently
  estimated score-meat fluctuation to a projected Wishart reference
  over-corrected badly: the normal-reference projected test rejected
  0.002/0.000, and the VM oracle effective df was at most `q-1` even at n=200.
  The latter needs eighth-order raw-data moments and remained visibly unstable
  across five independent 100,000-case references. Ordinary Hotelling
  (`eta=n-1`) was the simple winner in this gate; the same-sample projected
  plug-in looked competitive but estimated a shape orders of magnitude above
  the independent VM reference, so do not interpret that size as validation of
  the derivation. Also do not treat the raw-OPG chi-square's apparent
  calibration as independent evidence: algebraically
  `Q_raw = n Q_centered / (n - 1 + Q_centered)`, so its low-n shrinkage is
  mechanically tied to the numerator. The VM transition follow-up
  (`n=200/400/600/800/1200`, 2,500 disjoint target/donor pairs per cell)
  rules out beneficial same-sample coupling as the explanation for the
  empirical projected row: replacing each target's shape by an identically
  distributed independent-replication shape changed rejection by at most 0.24
  percentage points and with no consistent sign. Both empirical rules remained
  conservative near 3%, whereas ordinary Hotelling and the unadjusted centered
  score had transition-grid mean absolute size errors of 0.22 and 0.39 points.
  Moment convergence is the actual failure: at `n=1200` the median empirical
  shape was still 12.6 times the fixed VM reference (`eta` about 101 versus 8),
  and the fixed-oracle Hotelling tail rejected zero times wherever it existed.
  Do not promote either the fixed-geometry formula or its empirical plug-in,
  and do not extend this shape-matching route with more sample-size grids. A
  follow-up is justified only if it derives the full fitted-nuisance influence
  of the meat, or if the much simpler ordinary-Hotelling reference is
  replicated for nested robust scores; either route must report
  reference-moment convergence and `eta <= q-1` failures.
  Experiment research/41 now supplies the broader ordinary-Hotelling GOF gate:
  `q=2/5/9/20` crossed with `n=30/60/120/240`, normal and severe
  Vale--Maurelli data, and 2,000 replications per cell. Hotelling was closer to
  5% than the centered-score chi-square in all 32 cells. Its mean absolute size
  error was 1.59 percentage points under normality and 1.12 under VM, versus
  15.39 and 13.20 points for the chi-square reference; the remaining range
  (4.9--11.1% normal, 2.5--6.1% VM) says useful finite-sample approximation,
  not exactness. FMG pEBA4-RLS remained the strongest analytic comparator:
  better under normality (0.73-point error) but worse under VM (1.72 points).
  Do not count the raw-OPG chi-square as a second success: its rank-one
  shrinkage produced zero rejection in the most under-sampled cell. The first
  nested-score replay used the wrong linear map for the core's raw OPG.
  Correctly using
  `F=(n-q)Q_raw/[q(n-Q_raw)]` moves direct-sandwich rejection from 4.84% to
  5.00% in experiment research/32's dimension audit, 4.94% to 5.19% in experiment research/33,
  and 4.09% to 4.48% in experiment research/36. Thus ordinary Hotelling does not
  compound nested-score conservatism in those saved audits; the earlier
  2.59/3.50/2.97% replay was an algebra error. Keep the nested-score claim
  separate from the GOF grid because its score meat is raw rather than centered
  and its finite-sample behavior is still design-specific. The
  next publication-grade GOF gate, if needed, is a qualitatively different
  nonnormal generator or directional kurtosis, not a denser `n/q` grid.
  Experiment research/44 adds a direct-FIML global-GOF bridge in which a five-variable
  independence model is nested affinely in the saturated covariance model, so
  the existing nested score/flip machinery is exactly a global GOF test. Its
  second iteration crosses seven outcome laws (Gaussian, finite-moment heavy
  tail/skew/contamination, and `t(3)`) with complete, MCAR, two MAR, and MNAR
  mechanisms, explicitly separating 14 primary robustness scenarios from
  common-use MAR stress and deliberate moment/ignorability violations. At
  `n=120`, 10 df, and 500 null replications per scenario, the effective
  multiplier was nominal in every primary cell (.032--.068; mean absolute size
  error .009) and retained the highest primary size-adjusted power (.423).
  Score SB was the strongest simple analytic alternative (size .016--.060;
  adjusted power .391). MLR ranged from .068 to .829 in the same primary cells,
  pooling to .078 under Gaussian reference mechanisms and .317 under
  finite-moment nonnormality; the extreme result therefore does not depend on
  `t(3)` or MNAR. An expanded 875-dataset same-data oracle had zero
  finite/non-finite, df, or .05-decision mismatches against lavaan MLR (five
  paired non-finite p-values; maximum finite p difference .000281). Before a
  general claim, rerun a reduced primary grid at larger `n` and repeat the
  comparison in a latent SEM. The general H0 saturated-score projection now
  exists as `inference::frontier::global_score_flip_test`: direct saturated
  observed-data scores are projected off the fitted SEM tangent in the
  pattern-conditional Fisher metric, with no H1 fit or per-draw refit. Keep MLR
  as the target, effective multipliers as the lead, score
  SB/pEBA4 as analytic alternatives, and LRT pEBA4/robust Wald as comparators.
  Experiment research/44's third iteration now preflights that latent-SEM panel without
  pretending the projection exists: one-factor, two-factor, orthogonal
  bifactor, and linear-growth populations crossed with native normal/VM/IG
  generators and complete/30% MCAR data. At `n=120`, all 400 ten-rep smoke
  fits converged and 398 MLR results were finite. FMG was complete for the
  other three models but its strict projected-rank guard rejected 12/100
  bifactor fits, so that tolerance or its prespecified failure handling is a
  blocker for the large comparison. On four local workers the current
  null-plus-one-power panel projects to about 1.2, 6.0, and 23.8 minutes for
  100, 500, and 2,000 reps/cell, excluding the then-unimplemented general
  projected-score and multiplier cost. The model/generator budget is therefore
  acceptable; the next work is to measure the shipped projection cost, run its
  reduced null gate, and add matched power DGPs, not
  a large run of the incomplete method panel.
  The subsequent 1,000-rep null run completed all 40 cells (40,000 attempts) in
  541 seconds on four workers: 39,985 FIML fits converged, 39,894 MLR p-values
  were finite, and 39,156 FMG calls returned. MLR's mean cell rejection was
  .366 (range .062--.932). FMG `all` was materially better at .066 but still
  ranged .022--.168, with its liberal tail concentrated in severe nonnormal
  MCAR cells; SS was similar at .073, while SB and pEBA(4) remained clearly
  liberal. The FMG projected-rank guard rejected 829 otherwise-converged fits,
  including 797 bifactor fits. This makes the general multiplier comparator
  and explicit rank diagnostics the next scientific step; the existing FMG
  family is a useful comparator, not a solved replacement.
  Iteration 05 now gates that comparator on the full 40-cell design: all 120
  three-rep smoke calls returned finite 199-draw global multiplier p-values,
  every numerical df matched the prespecified model df, and mean test cost was
  .0219 seconds (maximum .209). Including fitting/MLR/FMG, the measured panel
  projects to about 7.2 minutes for 500 null replications per cell and 14.4
  minutes for 1,000 on four workers; null plus matched power doubles those
  figures. Iteration 06 completed the 500-replication gate (20,000 attempts) in
  446 seconds. The effective multiplier's mean cell rejection was .058, mean
  absolute size error .016, and range .018--.122, with 33/40 cells in
  [.025,.075]. It materially improved on FMG `all` (.065 mean, .018--.155,
  27/40 in-band), SS (.072 mean, .024--.173, 24/40 in-band), and MLR (.364
  mean, .068--.930). Nine FIML fits failed; every converged fit yielded a
  finite flip. Twenty-five of 19,991 finite calls had a numerically deficient
  tangent rank and 16 rejected, but requiring the prespecified rank changed
  mean cell rejection only from .0584 to .0577. Treat that geometry diagnostic
  as a separate fail-closed denominator. The next scientific gate is now the
  prespecified sparse/diffuse power panel with raw and size-adjusted power;
  another larger null run is for publication precision, not method triage.
  The 2026-08-20 paired extension adds the ML2S global multiplier and a
  15-indicator correlated three-factor CFA (48 free parameters, 87 df). A
  three-rep normal/VM/IG complete/MCAR smoke produced 300/300 converged paired
  FIML/ML2S fits, finite multiplier p-values, and prespecified tangent df.
  However, a five-rep p=15 timing audit found a heavy ML2S Stage-2 optimizer
  tail under MCAR (median total .26 s, 90th percentile 13.0 s, maximum 153 s).
  The checkpointed `n=200` continuation subsequently completed 500 null
  replications over five models, five normal/VM/IG laws, complete/MCAR/MAR, and
  paired FIML/ML2S fits (75,000 estimator rows; four fit failures). Restricting
  the primary interpretation to the 55 complete/MCAR plus normal-MAR cells,
  projected-score pEBA4 had mean absolute size error .0127 and range
  .014--.074, versus .0139 and .024--.094 for the expected-sensitivity
  Rademacher multiplier. Score SB
  was close (.0145, .034--.092). The statistic/correction interaction is the
  main result: score SB/pEBA4 were viable while score SS/ALL were conservative;
  LR SS/ALL were viable while LR SB/pEBA4 were liberal. ML2S worsened the useful
  methods under missingness and is now a secondary negative result.
  Iteration 07 sets up the FIML-only matched-power continuation. A normal-data
  audit found the realized observed-sensitivity projection extremely
  conservative at `n=200` in the higher-df models, so the identified-null
  comparison retains expected-sensitivity score tests as primary. A
  three-corner multiplier ablation (expected/Rademacher, expected/Mammen,
  observed/Mammen) separates weight-law and projection changes; score-FMG is
  recorded under both geometries. A subsequent 300-fit `n=120` smoke made the
  realized H0 information the quadratic and spectrum metric as well: 74 calls
  failed positive-definiteness and usable score-pEBA4 calls rejected 58%, so
  keep that full-observed construction diagnostic-only and do not add it to
  the planned primary battery. Sparse omitted-residual and diffuse omitted-method-factor
  alternatives are tuned per model to 50% asymptotic normal-theory LR power at
  `n=200`, then compared with method/cell-specific empirical-null cutoffs. The
  launched Modal design had 165 one-cell focus shards (55 mechanisms x
  null/sparse/diffuse,
  1,000 replications, 999 draws); 60 nonnormal-MAR shards are a separately
  labeled estimand stress. Do not interpret nonnormal-MAR rejection as ordinary
  Type-I error or add ML2S power unless a separate two-stage question emerges.
  Such a question has now emerged as patternwise NTML, but it is kept separate
  from the retired FIML-versus-ML2S power continuation. Iteration 15 implements
  the pattern-normal objective and its expected-information normal-MCAR
  inference, with exact complete-data gates and a 20-replication paired
  complete/MCAR plumbing smoke. If promoted beyond that smoke, run a dedicated
  normal-MCAR comparison of bias, RMSE, expected-information SE calibration,
  and null size. Study MAR separately: point consistency follows the saturated
  FIML target under ignorability, whereas the present pattern-count-only
  model-based inference is not a general MAR claim. An empirical Stage-1
  sandwich would be a separately named robust PNTML extension.
  The `power-v1-20260821` Modal continuation is retired rather than repaired:
  its stress phase combined 60/60 cells, but its focus phase stopped at 162/165
  shards because restarted calls tripped an empty-field configuration guard.
  The remaining 725 replications are not worth recovering before the setup
  changes. The 164,275 available focus rows and complete 30,000-row stress phase
  are collected locally; the preliminary report labels their power summary as
  incomplete and retired. The replacement design now defines H0 at the
  estimator level: a calibration DGP must place the saturated Gaussian-FIML
  pseudo-true target inside the fitted SEM. A generating full-data covariance
  in the SEM is not a null certificate under MAR; such off-pseudo-null arms are
  power/estimand-stress conditions. Preserve the remote files as provenance,
  exclude this run from publication tables, and build the replacement
  production run for Slurm only after the design is frozen. The prospective
  paper has three explicit design gates recorded in
  experiment research/44's `TODO.md`. First, replace the implementation-oriented pilot
  panel with five broad, sourceable models: one-factor, correlated three-factor,
  bifactor, latent mediation, and linear growth, all at 15 observed variables
  or fewer. Second, freeze literature-grounded MAR-L/MAR-L2/MAR-NL and
  few/many-pattern mechanisms, plus a separately labeled nonnormal-MAR
  pseudo-true-target stress. Third, predeclare and report a public-data corpus
  screen for one or two interpretable MLR-reject/score-pEBA4-nonreject examples.
  A targeted three-df nonnormal-MAR pseudo-null now isolates the information
  choice: expected-H0 score pEBA4 rejection falls to zero under strong tail
  selection through n=10,000, whereas observed-H0 sensitivity with the stable
  expected metric remains near nominal. Saturated-H1 observed geometry is now
  exposed as an opt-in comparator; it is PD and shares the H0 sensitivity limit,
  but converges prohibitively slowly in this stress design. Carry observed-H0
  sensitivity/expected metric into the source-based latent-model gate and do
  not treat expected Fisher as the default under nonnormal MAR.
  Multiplier development is dropped from the planned comparison; preserve its
  completed diagnostics but do not expand it. Do not launch the publication
  grid until model parameters, source adaptations, missingness mappings, and
  the empirical screening rule are frozen.
  The focused published-DGP follow-up (experiment research/33) adds an important negative
  result: with n=400 per homogeneous group, raising the weak-invariance rank to
  133 changed effective versus standardized decisions only twice in 3,200 null
  replications, while flip-specific standardization cost about 4.1 seconds per
  replication at that corner. Do not spend the next design on still more
  restrictions. Vary n/group, nuisance leverage/information heterogeneity, or
  the restriction-to-sample ratio while holding rank fixed, and use the returned
  variance-displacement diagnostics as the proposed effect modifier.
  The corresponding fixed-rank design and budget-guarded Modal harness now live
  in experiment research/34: `G=8`, `df=28`, p=5 versus p=20 nuisance dimension,
  n/group, allocation, information geometry, and normal/VM/IG/PL generators,
  followed by calibrated normal/PL power. Remaining work is to run and audit
  that experiment, then decide whether the standardization benefit clears its
  paired calibration/runtime gate. Do not infer a rank effect from its p
  contrast: tested rank is deliberately constant. The near-singular-information
  hardening test above remains a core prerequisite independent of the
  simulation outcome; exact enumeration and the dense covariance oracle are
  now gated.
  The expanded FIML null gate is now complete in experiment research/37. Its 240-cell,
  500-attempt screen crosses normal/severe VM/IG/PL, n1=50/100/200/400,
  complete/MCAR/paper-MAR/strong-MAR data, and ranks 1/3/5; an independent
  11-cell confirmation uses 2,000 attempts and 999 signs. Effective versus
  standardized rejection was 0.0693/0.0678 in the screen and 0.0808/0.0778 in
  the deliberately hard confirmation, with only 0.145%/0.301% paired decision
  disagreement. Score pEBA4 was 0.0496/0.0473, whereas nested pEBA4 was
  0.1102/0.1146. Standardization therefore does not rescue severe-nonnormal
  small-n flips at rank <=5, and at 999 signs its median 94.8-ms covariance
  phase dominates the 12.1-ms effective sign sums. Do not spend another null
  design on more missingness/rank cells. If power is pursued, use a small set
  of method/cell-specific empirical-null thresholds so the liberal flips and
  nested tests cannot win by construction. The bounded method question is now
  a genuinely small-sample correction to the effective flip under severe
  nonnormality, not more pattern standardization. Near-singular nuisance
  information remains a separate core-hardening prerequisite.
  Then decide other extensions separately: continuous LS and ordinal/mixed models need estimator-specific casewise
  influence contributions including estimated-weight effects; clustered data
  need cluster-level rather than row-level signs. A residual-based or robust-
  Wald flip test is a distinct method and should not be hidden behind the score-
  test entry point. The biased RLS H0-minus-H1 comparator now available through
  `fmg_nested(..., tests = "*_rls")` does not supply casewise RLS scores and is
  not such an extension. Experiment research/35 now supplies the first concrete
  residual/RLS GOF estimating-function derivation, but its 100-replication gate
  argues against immediate core promotion: the projected effective flip was
  plausible at df=5 and liberal at df=170 relative to n; raw empirical
  standardization was singular at df>=n and badly conditioned or more liberal
  where available. Before promoting it, run a larger low/moderate-df design
  that varies n/df directly. Treat any ridge/shrinkage rescue as a distinct
  tuned method with its own null and matched-power gate.

- **XL — establish whether raw-label permutation is valid for ordinal
  measurement-invariance tests.** Experiment research/31's continuous studentized-Wald
  path is motivated by Chung and Romano: raw observations need not be exactly
  exchangeable when the recomputed test statistic is asymptotically pivotal.
  Its focused robust-score arm (`p=9`, partial loading rank `q=1/4/8`) confirms
  that the same idea is computationally viable with one restricted fit per
  permutation. The balanced result alone was ambiguous, but the decisive
  unequal-allocation mirror favors the score pivot. With heterogeneous VM,
  `q=4/8`, 500 replications per cell, and `N=120/240/480/960`,
  score-permutation rejection was 4.9/5.3/5.1/5.7% under `1:3` and
  4.9/4.6/4.9/5.2% under `3:1`. Wald permutation was direction-sensitive:
  3.2% versus 6.3% at `N=120`, and did not settle near 5% in both directions
  until the larger samples. The score route cost about 12% more permutation
  time, so its win is calibration rather than speed. The analytical
  limiting-mixture reference converged extremely slowly: in a separate
  2,000-replication-per-q `1:3` ladder its rejection was
  2.9/3.3/3.8/4.3% at `N=480/960/1920/3840`. MV made virtually identical
  decisions, so the problem is finite-sample convergence to the mixture law,
  not chiefly MV tail approximation; SB reached 5.0% and pEBA4 4.8% at the
  largest endpoint by compensating that error. “Permutation Hotelling” remains
  exactly the same rank test as permutation `Q_raw`; do not build a separate
  Hotelling randomization API. The continuous score pivot also clears its
  bounded power gate, but as a calibration method rather than an efficiency
  win. Under the common-sign local loading shift `delta_N = 5 / sqrt(N)`,
  using method/cell-specific empirical-null thresholds with randomized
  boundaries, its power averaged 31.3% over both allocations, `q=4/8`, and the
  four sample sizes. SB, pEBA4, and the numerical mixture were
  30.2%/30.6%/30.3%; stratified bootstrap intervals for their roughly one-point
  deficits all included zero. Wald permutation averaged 27.0%, but its power
  beat score in every `1:3` cell and lost much more in every `3:1` mirror, so
  the pooled gap is allocation sensitivity rather than a universal score
  efficiency result. Retain the score permutation as the principled pivotal
  route when slow mixture convergence makes analytical calibration doubtful;
  retain SB/pEBA4 as simpler, competitive analytical defaults. Freeze the
  broad continuous grid here. Promotion from experiment code would now require
  a clean asymptotic argument and a deliberately small confirmatory alternative
  design, not more null cells or a claim of superior power.
  Ordinal DWLS/WLSMV is a separate question, not an automatic extension. A
  relabelled sample changes thresholds, polychoric/polyserial moments, their
  estimated NACOV, and possibly the pooled pseudo-true model; first establish
  whether the fully recomputed ordinal Wald pivot has the required common
  limiting law under heterogeneous but invariant groups. Only then add an
  ordinal lane to the experiment, with exact-exchangeable controls, balanced
  and unbalanced heterogeneous nulls, sparse-category diagnostics, and an
  ordinary/robust Wald and nested-test comparator. Do not use a fixed-statistics
  label shuffle as a surrogate: it is not the proposed test. The existing
  `mplus_wlsmv_invariance()` helper covers the fitting ladder, but it does not
  settle this resampling validity question.

- **M — finish streaming the continuous pairwise NT bread for observed-bread
  inference.** The pairwise empirical meat path is projection-first:
  `pairwise_casewise_contributions()` builds Van-Praag influence rows and
  `reduced_gamma_sample()` / `reduced_gamma_sample_streaming()` reduce them in
  df-space without forming the full pairwise covariance of moments. Expected
  pairwise bread is now also operator-first: `build_u_factor(... raw, pw,
  Information::Expected, WeightMoments::Pairwise)` builds a residual subspace
  from the model Jacobian, applies the `Gamma_NT^pw` metric with the existing
  pattern-grouped operator, and orthonormalizes only the reduced `df x df` Gram
  matrix.

  The remaining non-streaming piece is observed-bread pairwise inference. That
  path still calls `data::gamma_nt_pairwise(raw, pw)`, materializes each
  `p* x p*` pairwise NT Gamma, then takes an LLT because `ObservedHessian`
  stores `A = L_Gamma^{-1} Delta`. That is a separate pressure point from the
  FIML lavaan empirical-Gamma fix: it can still be O(p^4) memory/time and can
  fail on ill-conditioned pairwise overlap structures. Candidate routes:
  pattern-grouped linear solves for `Gamma_NT^pw^{-1} Delta`, blockwise
  factorization/update in pattern space, or a raw-metric observed-bread formula
  that bypasses explicit `L_Gamma` while matching the current
  `ObservedHessian` spectrum. Acceptance tests: complete-data collapse to the
  unstructured observed-bread path; missing-data agreement with materialized
  `gamma_nt_pairwise` on small fixtures; and stress cases report finite
  diagnostics instead of dense-matrix blowups.

- **M — establish which lavaan FIML robust convention, if any, magmaan's FIML
  `sb_ml` is supposed to match.** Under missing data the two engines make
  different choices in estimating the bread, the meat and the saturated-H1
  reference, and magmaan's defaults line up with neither lavaan variant
  consistently. Two-factor CFA, `N = 500`, 10% MCAR
  (`experiments/_archive/score-vs-lrt/diagnose_trace_sb_nested.R --with-fiml`):
  `fmg_tests(tests = "sb_ml")` sits 1.5e-04 from lavaan's `yuan.bentler.mplus`
  (the `estimator = "MLR"` test) but 9.2e-02 from `yuan.bentler` at `p = 20`,
  and 5.6e-03 from *both* at `p = 10`. So it is not simply "magmaan targets the
  Mplus variant" either. The `papers/fiml-fmg` gate reports 90/90 passing with
  `convention = "lavaan"`, so the likely resolution is a convention flag that
  `fmg_tests()` does not set rather than a defect, but until it is pinned down no
  lavaan FIML robust test is a usable oracle and no FIML SB parity or timing
  number should be quoted. Decide and document which convention each entry point
  targets, then gate it. Two comparator traps to encode in whatever gate results,
  because both yield plausible numbers: lavaan silently switches `missing` to
  `"listwise"` when `satorra.bentler` is requested at fit time (so
  `cfa(missing = "ml", test = "satorra.bentler")` is not a FIML fit), and
  `lavTest(test = "satorra.bentler")` called post hoc on a genuine FIML fit slips
  past that gate and returns Yuan-Bentler numbers under an SB label. Related but
  distinct from the saturated-H1 regularizer item below, which is about
  near-singular references rather than convention choice. Separately, lavaan's
  FMG refuses `missing != "listwise"` outright
  (`lav_test_fmg_check_missing()`), for global and nested tests alike, so no
  lavaan pEBA comparator exists for any FIML cell regardless of how the
  conventions are settled.
- **M — FIML Satorra-2000 nested test: calibrate the opt-in saturated-H1
  reference regularizer under missing data (frontier).** The scaled and mixture nested difference tests
  (`nestedTest(method = "satorra.2000")`, especially direct FIML) collapse to
  conservative (Type-I toward 0) under missing data at moderate model size. The
  saturated-H1 reference (the EM covariance and the moment acov Gamma) goes
  near-singular under missing data at p >= 16, N ~ 200 (cond(acov) toward Inf),
  so the difference spectrum (eigenvalues of `U*Gamma`) develops one giant plus
  a cascade of numerical-garbage eigenvalues. The Satorra-Bentler scale
  `c = trace(U*Gamma)/df` then over-scales by 30-70x (arbitrarily large once the
  reference is singular) and crushes the statistic. This is not a lavaan-parity
  bug (magmaan matches lavaan per dataset); it is a conditioning defect both
  share. A constant oracle scale `c = mean(T)/df` recalibrates every probed cell
  to nominal, so the dominant defect is the per-replicate reference scale/spectrum
  rather than the likelihood-ratio statistic.

  Core implementation landed 2026-07-03 as opt-in
  `nestedTest(..., h1_reference_regularization = ...)` for direct-FIML
  restriction-map tests. Defaults remain raw/lavaan-parity. When enabled,
  `TRUE` uses `condition_max = 1e6`; explicit lists can set `condition_max`,
  `min_eigenvalue`, and covariance/information sub-options. The lavaan
  convention transforms the saturated H1 covariance and propagates the
  transformation into its sandwich ACOV (the weight uses the fitted larger
  model covariance); the native eta-space and `ud_method = "2001"` FIML paths
  floor the saturated H1 information before inversion and form
  `Gamma_reg = H_reg^-1 J H_reg^-1` (or `H_reg^-1` for NT Gamma). ML2S remains
  controlled by fit-time `stage1_regularization`, not this nested-test option.

  Remaining work: calibrate candidate caps in `papers/fiml-fmg/dev/refcond/`
  before using the option in the paper grid. A post-hoc spectrum trim recovers
  small p but not the singular large-p regime, so it remains only a diagnostic
  proxy, not the implemented fix.

  Asymptotic accounting must be explicit. If the Stage-1 moments themselves are
  transformed, propagate the transformation by delta method
  (`Gamma_reg = J_g Gamma J_g'`), including the derivative of any data-adaptive
  intensity when it is part of the estimator, as in
  `data::frontier::shrink_mixed_ordinal_stats()`. If only the reference
  covariance/information used by the spectrum is regularized, treat it as a
  regularized variance estimator: it is first-order equivalent to the raw
  Satorra reference under fixed-p asymptotics only when the floor/intensity
  vanishes or the repair activates with probability tending to zero; fixed
  finite caps define a different frontier reference law that needs simulation or
  bootstrap calibration. Return diagnostics (`raw/regularized min eigen`,
  condition number, effective rank, lambda/floor, trace before/after) with every
  regularized spectrum.

  The related ML2S input-moment surface landed 2026-07-03 as opt-in
  `stage1_regularization`: it regularizes the saturated Stage-1 covariance before
  Stage 2, preserves `$stage1_raw`, and delta-propagates `$stage1$acov`. That
  does not solve the distinct direct-FIML H/Gamma reference-law problem here.
  Evidence, per-regime numbers, and a calibration gate are in the fiml-fmg
  invariance experiment (private paper). Use the ref-conditioning gate
  before/after implementation: p8 and p16, normal/vm2/ig2 x complete/MARnl, with
  regularization required to rescue missing cells without making complete-data
  controls liberal.

- **M — automatic robust fit-measure dispatch.** The lavaan-style robust/scaled
  fit-measure formulas are available for the core global-index family
  (`chisq.scaled`, scaled baseline, CFI/TLI, RMSEA CI/p-values) once the user
  and baseline scaling factors are supplied. Remaining work is estimator-specific
  plumbing: build the independence/baseline robust scaling for complete-data
  MLM/MLR and all-ordinal WLSMV baseline CATML ingredients, then let
  `fit_measures(fit, robust = "MLM"/"MLR"/"WLSMV")` compute those scalars
  automatically. FIML `robust = TRUE` / `"MLR"` is wired on the R surface
  through `estimate::fiml::fiml_corrected_fit_measures`, including the corrected
  `XX3` user/baseline reductions and robust CFI/TLI/RMSEA fields; lavaan's
  `MLR` fitMeasures fields are fixture references rather than the oracle for
  that explicit helper.
  The ML2S `robust.two.stage` robust/scaled CFI/TLI/RMSEA family is implemented
  as `estimate::fiml::two_stage_fit_measures` and lavaan-gated in the FIML
  golden fixtures.

- **Ordinal-SEM standardized solution / defined params / factor scores —
  mostly landed 2026-06.** Decided: `compute_defined` is valid for ordinal/mixed
  fits (a parameterization-agnostic delta-method transform) and its guard was
  removed at both the C++ api and the Rcpp binding. Ordinal/mixed factor scores
  now have their own categorical scorer instead of reusing the continuous
  regression/Bartlett path: EBM and ML are per-pattern Newton solves over the
  latent-response likelihood/posterior, and one-factor EAP uses the vendored
  QUADPACK infinite-interval integrator. Fixing the api flip surfaced a real bug:
  the ordinal api
  `Fit` carries the un-prepared structure while estimates live over the prepared
  (reduced) partable, so `standardize_lv`/`standardize_all`/`compute_defined`
  aborted on a theta-dimension mismatch — only the Rcpp path (which stores the
  prepared partable) worked. The api now reconstructs the prepared structure on
  demand (`prepared_structure` helper in `cpp/src/api/sem.cpp`). Coverage: C++ unit
  (`api_sem_test` ordinal case) and live lavaan value-parity in
  `r-package/examples/ordinal_dwls_wls.R` (ordinal `:=` value+SE to 5e-3; mixed
  std.all to 1e-3; all-ordinal std.all already covered). Remaining:
  - **Done 2026-06.** The oracle pin was realigned to `0.7-1.2691` and the
    checked C++ golden landed: `cpp/tests/golden/ordinal_golden_test.cpp`
    "ordinal/mixed standardized + := rows match lavaan" gates the `=~` loading
    std.lv/std.all rows (all-ordinal 5e-3, mixed 1e-3) and the
    `lprod := L2*L3` value+SE (5e-3) against the stored lavaan oracle
    (`ordinal/0015_defined_param_3cat_cfa` plus the per-fit
    `fits.DWLS.standardized` blocks emitted by `regen_oracle.R`).
  - **Done 2026-06-14.** Checked-in lavaan oracle for categorical factor scores
    landed. `regen_oracle.R` now emits `fits.DWLS.fscores` (via
    `ordinal_fscores_json` in `benchmarks/r/fixture_json.R`) for the single-group
    ordinal and mixed fixtures, and the golden
    `cpp/tests/golden/ordinal_golden_test.cpp` "ordinal/mixed factor scores (EBM/ML)
    match lavaan" gates `factor_scores_{ordinal,mixed_ordinal}` against
    `lavPredict()` at 5e-4 (the previously live-only parity from
    `r-package/examples/ordinal_dwls_wls.R`). The installed-lavaan method set is
    pinned by emitting only what lavaan's categorical `lavPredict()` supports:
    **EBM** single-group (all-ordinal and mixed; posterior mode, ~1e-5) and
    **ML mixed-only** (continuous indicators bound the mode). Deliberately not
    gated: all-ordinal ML (unbounded mode on extreme patterns) and EAP/precision
    (categorical `lavPredict()` rejects EAP, so no oracle; stays self-checked in
    `cpp/tests/unit/api_sem_test.cpp`).
  - **Resolved 2026-06-14 — not a magmaan bug; lavaan is the outlier.** The
    multi-group categorical EBM divergence (reference group matches lavaan ~2e-5,
    non-reference group drifts ~0.2 with `theta` matched to 1e-7) was root-caused
    to a **lavaan** defect: lavaan's multi-group categorical `lavPredict()`
    returns a *non-stationary* point for non-reference groups. Verified three
    ways: (1) magmaan's non-reference-group EBM matches an independent
    `optimize()`-based posterior-mode scorer built from lavaan's *own* extracted
    group-2 parameters to 1.9e-5; (2) at lavaan's group-2 score the posterior
    gradient is O(1) and the posterior density is *lower* than at magmaan's score
    (whose gradient is ~1e-7) — i.e. lavaan does not sit at the mode; (3) every
    ingredient lavaan feeds its scorer (prior `VETAx`, `THETA`, `TH(delta=FALSE)`,
    loadings, data, `th.idx`) is identical to magmaan's, consistent with lavaan's
    own `lav_predict.R` FIXME about categorical scores being "not identical (but
    close) to Mplus". So multi-group `fscores` are not emitted as a lavaan oracle.
    Instead the multi-group scorer is validated **transitively** against lavaan in
    `cpp/tests/golden/ordinal_golden_test.cpp`: for the unconstrained two-group
    fixture `0004` the per-group multi-group EBM equals an independent
    single-group fit on that group's data (machine-tight, ~3e-8), and the
    single-group EBM is lavaan-gated. No remaining work.
  - **M/L, only-when-needed.** Multi-factor EAP via adaptive Gauss-Hermite and
    non-diagonal residual-Theta orthant probabilities remain deferred in
    `speculative.md`; the landed scope is diagonal-Theta EBM/ML plus one-factor
    EAP for all-ordinal and mixed complete data.

- **Ordinal EAP reliability / posterior precision follow-up (2026-06).**
  Current state: `factor_score_precision()` reports one-factor ordinal/mixed
  EAP posterior variance/SE, sample-normalized PRMSE, and concrete ordinal
  reliability. Reference manifest and working notes live in
  [project/reference/ordinal_reliability.md](../reference/ordinal_reliability.md);
  the derivation note lives in
  maintainer working note `ordinal_factor_score_reliability.md`.
  Remaining:
  - **Done 2026-06-14.** One-factor ordinal EAP precision is now validated
    against simulated ground truth (not just plug-in self-consistency):
    `cpp/tests/unit/api_sem_test.cpp` "ordinal EAP factor-score precision tracks
    Monte-Carlo PRMSE" generates a five-indicator three-category one-factor
    model with retained latent `Z`, fits ordinal DWLS under `std.lv`, and pins
    (1) `pooled_prmse ≈ corr(Z, E[Z|Y])²`, (2) `mean Var(Z|Y) ≈` the realized
    EAP MSE, and (3) `concrete == 1 - mean Var(Z|Y)` exactly under unit latent
    variance (≈ `1 - MSE`). Gaps are ~1e-3 at n=8000 against 2e-2 gates. This
    surfaced and fixed a real bug: `std.lv` ordinal delta fits aborted in
    `compact_free_set` because `n_free()` (the max free index) dropped after the
    top response-scale variance was zeroed for elimination; the compaction now
    takes the original free count from the caller's `remove_free` metadata
    (`cpp/src/estimate/ordinal.cpp`, regression note at the fix site). Follow-up:
    - **Done 2026-06-15.** Checked-in lavaan oracle for `std.lv` ordinal CFA.
      `regen_oracle.R` now emits `ordinal/0016_std_lv_3cat_cfa` (model
      `f =~ NA*x1 + x2 + x3 + x4; f ~~ 1*f`, same data/structure as `0001` so it
      is a pure reparameterization: identical χ²/df, loadings rescaled by the
      latent SD). The existing delta-contract golden gates it directly —
      `magmaan` matches lavaan to χ² 3e-8 and all 12 free params (4 free
      loadings + 8 thresholds) to 4.6e-8, df=2 — and the standardized/EBM
      factor-score arms ride along. The threshold-profiled SNLLS arm is
      structurally N/A under std.lv delta (the conditionally-linear
      {Psi, Theta} block is empty: latent variance fixed to 1, response
      variances fixed by `~*~ 1`), so the SNLLS golden pins the expected
      "no conditionally linear free parameters" diagnostic rather than
      skipping; the bounded full-Newton fit carries the parity.
  - **M.** Add bootstrap CIs as the first inference surface, probably frontier
    or R-only first: parametric bootstrap from the fitted ordinal/mixed model,
    rebuild stats, refit, recompute both coefficients; optionally add
    nonparametric row bootstrap for robustness checks.
  - **L.** Analytic SEs require stable casewise ordinal moment influence
    plumbing for `theta_hat` plus finite-difference derivatives of posterior
    moments wrt free SEM parameters. Concrete reliability with fixed unit
    latent variance is the simplest analytic target; sample PRMSE needs the
    three-moment delta method. Follow Sung and Liu's IRT SE paper, replacing
    their item-parameter influence function with magmaan's ordinal SEM
    LS/GMM influence path.
  - **S/M, keep separate.** Do not collapse CTT reliability of the EAP score,
    EBM/ML determinacy summaries, and multi-factor EAP PRMSE into the same
    public coefficient; add them only with explicit names and a concrete
    downstream consumer.

- **Ordinal observed-score omega follow-ups (2026-07).**
  Landed first C++/R frontier slice:
  `estimate::frontier::ordinal_observed_omega` maps fitted all-ordinal
  thresholds + latent-response correlations to the observed integer category
  covariance, computes `OmegaTarget::Total` / `Hierarchical` through
  `measures::frontier::reliability`, and gets a misspecification-robust
  delta-method SE from `robust_ordinal_ij` (including DWLS/WLS estimated-weight
  influence). R exposure is
  `magmaan_core$measures_reliability_ordinal_observed_omega`; experiment
  `07-ordinal-observed-omega-dwls` is the smoke/calibration probe. Experiment
  `08-ordinal-omega-target-audit` now shows this is an observed-score
  covariance omega, not literally the direct one-factor ordinal true-score
  target: tau-equivalent equal-threshold cells match, ordinary equal-threshold
  congeneric cells are near-identical, but heterogeneous thresholds reached a
  0.0158 absolute difference in the population target. The simpler
  no-integration latent-response/polychoric coefficient is also exposed as
  `magmaan_core$measures_reliability_ordinal_polychoric_omega`: it computes
  `omega_multidim` on `stats$R[[group]]` and pads the ordinal `NACOV`
  correlation block into a full vech-correlation Gamma for the robust delta SE.
  Experiment `22-ordinal-polychoric-omega-coverage` probes that no-integration
  coefficient under a correctly specified one-factor ordinal-probit generator:
  at 200 reps per cell, `N in {50,100,250}`, and balanced vs threshold-extreme
  cuts, the 95% robust delta interval had coverage 0.915-0.965 among successful
  draws for the latent-response omega target 0.829. The only construction
  failures were 6/200 threshold-extreme `N=50` draws with an empty item category.
  Experiment `23-ordinal-polychoric-omega-stress` then uses large-sample
  pseudo-targets for skewed one-factor, heavy-tailed one-factor, and normally
  locally dependent ordinal DGPs. The same interval covers the pseudo-true
  polychoric omega target at 0.907-0.950 in local-dependence cells, but bends
  under skew/heavy tails: skew-balanced is 0.885-0.945, skew-extreme is
  0.815/0.903/0.935 for `N = 50/100/250`, and heavy-tail extreme falls to
  0.778/0.808/0.845 with 42/200 empty-category or singular-score failures at
  `N=50`.
  Remaining:
  - **M.** Add the direct conditional-mean ordinal omega target
    `Var(sum E[Y_j | eta]) / Var(sum Y_j)` as a separately named coefficient
    before calling anything Green-Yang/Flora omega.
  - **M.** Define and test the multi-group target before lifting the guard:
    pooled observed category covariance, sample-size-weighted group omegas, or
    explicit per-group output are different estimands.
  - **M/L.** Add fitted-model stress checks against the existing
    complete-sandwich DWLS/WLS machinery. A first all-ordinal parameter
    profile-LRT/CI seed now exists (`profile_lrt_parameter_ordinal` and
    `profile_lrt_ci_parameter_ordinal`, ordinary df-1 reference, fixed
    ULS/DWLS/WLS weight), and the no-integration model-implied
    latent-response/polychoric omega functional has the same seed surface
    (`profile_lrt_ordinal_polychoric_omega` and CI inversion). Both ordinal
    parameter and polychoric-omega profiles now have opt-in robust 1-df
    sandwich scaling and scaled CI inversion (`robust = TRUE` on the R
    frontier helpers), using the complete IJ ordinal sandwich at each
    constrained point. Observed-score / Green-Yang-like omega profiling and
    small-sample profile-LR corrections stay in the funLR lane until the
    calibrated-constant problem is solved. Experiment
    `25-ordinal-profile-lrt-calibration` now includes the first fitted-model
    misspecification stress for the no-integration profile target: a locally
    dependent ordinal latent-response DGP fitted by the same one-factor DWLS
    model. At 1000 reps/cell, ordinary DWLS still undercovers, while
    robust-scaled and misspecification-scaled references both cover near nominal
    for the pseudo-true fitted omega except in the sparse threshold-extreme
    `N=50` cell. The scalar misspecification arm is now reported as scaled; the
    previous mixture wording was only the singleton weighted-chi-square
    representation of the same one-df law. The first CI-inversion probe
    (`--ci`, 200 reps/cell) makes robust-scaled inversion the practical v1 CI
    lane: 3168 attempts, 38 failures, zero target-test/CI disagreements, and
    coverage 0.918-0.975. Misspecification-scaled CI inversion remains
    diagnostic only: 1201/3154 attempts failed, mostly at endpoint robust
    profile quadratics. A harder `--stress` mode now uses stronger local
    dependence and a two-factor DGP, smaller `N in {25,35,50,75}`, and sparse
    cuts. In the 300-rep pointwise stress run, sparse-threshold failures are
    mostly ordinal-stat construction failures, but the usable hard cells are
    threshold-extreme `N=50`: robust-scaled coverage was 0.932
    (strong-local) / 0.926 (two-factor), and the oracle-constant diagnostic
    moved those to 0.964 / 0.961. A 100-rep targeted CI pass on those two cells
    kept robust-scaled CI decisions connected (0 disagreements; coverage
    0.929 / 0.934 among successful inversions) while misspecification-scaled CIs
    failed 178/179 attempts. Next useful work: test candidate scalar
    finite-sample constants on the robust-scaled `N=50` stress cells.

- **Ordinal stats-construction perf headroom (2026-06-12 audit).** Workspace
  construction dominates ordinal/mixed wall time (fits are sub-3ms). Landed:
  cell-cached `ordinal_pair_scores`, an `x_tol` stop for the rho searches, and
  (second pass, same day) the two M items: the polychoric/polyserial ML rho
  searches are now safeguarded Newton on the closed-form score
  (`detail_rho_search.hpp`, ~6 grid evals vs ~44 golden-section), and per-rho
  bvn evaluation shares the `(K_i+1)(K_j+1)` corner grid
  (`ordinal_bvn_corner_{cdf,pdf,pdf_drho}`) instead of 4 `bvn_cdf` per cell.
  Construction-bench medians (n=900, reps=5): lazy ULS p16/c5 423→14 ms,
  lazy DWLS p16/c5 298→26 ms, legacy full stats p16/c5 376→68 ms.
  Third pass (2026-06-12, same day) closed the listed S items:
  - `bvn_cdf` is now the Genz (2004) refinement of Drezner–Wesolowsky (6/12/20
    Gauss–Legendre nodes by |rho| plus the complementary high-|rho| expansion,
    ~5e-16 absolute accuracy, unit-pinned against the asin closed form and a
    Simpson reference in `ordinal_test.cpp`); all-ordinal construction mins
    moved p16/c5 lazy ULS 11.4→4.1 ms, lazy DWLS 19.2→13.2 ms, p4/c3 lazy ULS
    1.0→0.24 ms. Lavaan parity suite unchanged.
  - `shared_ordinal_casewise_psi` is cell-cached (scaled per-cell scores
    scattered to rows via a transposed accumulator), and the shared-robust FD
    gradient and bread FD columns are restricted to the pairs touching the
    perturbed coordinate (`shared_ordinal_touched_pairs`) — exact, since
    untouched pairs difference to zero.
  - The shared-robust threshold+rho refinement and the joint polyserial DPD
    fit now delegate to the vendored NLopt L-BFGS (encoded coordinates are
    unconstrained) instead of hand-rolled steepest descent + Armijo; +inf is
    the barrier for invalid probes, and a hard optimizer failure keeps the ML
    starting values like the old bail-out. The remaining hand-rolled searches
    (1-D golden-section/bisection/Newton, Fisher scoring, EM) were audited
    2026-06-12 and are appropriate as-is.
  - `benchmarks/mixed_ordinal_construction_bench.cpp` covers the mixed
    workspace/stats paths (half-ordinal half-continuous designs). It shows
    mixed construction is dominated by the polyserial/Pearson branches:
    p16/c5 lazy ULS is ~21 ms vs ~4 ms all-ordinal at the same design.
  Remaining headroom:
  - **Done 2026-06-15 — closes this subsection.** The named target was the
    mixed polyserial casewise-influence pass. `polyserial_pair_scores` built the
    rho score column by a 3-point finite difference (three full conditional
    probability evaluations per case ≈ 6 `erfc`) plus an O(K) threshold-density
    sweep. It now uses the analytic boundary derivative
    `d log P(c|u)/drho = (top.d1 − bot.d1)/p`, reusing the two boundary normal
    densities the threshold columns already compute: one probability evaluation
    (2 `erfc`) plus at most two `normal_pdf` per case, O(1) in the category
    count. The rho score is now exact instead of a finite difference (the column
    shifts by ~1e-10 toward lavaan's analytic value); the threshold, mu, and var
    columns are bit-identical (`cpp/src/data/pairwise_mixed.cpp`). All 828 dev tests
    pass, including the mixed-DWLS NACOV/score fixtures and the bfi
    mixed-ordinal parity block. This roughly thirds the dominant `erfc` count of
    the score pass; on the score-assembly construction paths (legacy mixed
    stats, lazy DWLS/WLS workspace) wall time trends down ~15–30% at p≥12
    designs, though the i7-1355U thermal noise floor (visible as ±20–35% swings
    on the identical-code ULS control) is wide enough that the deterministic
    operation-count reduction is the solid result. The fit-only ULS path (rho
    ML search) is unchanged by design.

    No exact cell collapse exists for polyserial. The all-ordinal win came from
    both margins being discrete (a `(K_i+1)×(K_j+1)` count table where many
    cases share one cell → one set of transcendentals); the polyserial
    continuous margin `u_i` is distinct per case, so cases cannot collapse to
    cells and the rho ML search is transcendental-bound and already minimal.
    Binning `u` was considered and rejected: it would perturb the
    polyserial/polychoric estimate and break lavaan parity.

## Two-level (multilevel) SEM

V1 landed 2026-06: two-level normal-theory ML for random-intercept models over
a SHARED observed variable set, single group, complete data, no constraints,
observed + analytic-expected SE, optional box bounds, lavaan-parity. The
`level:` block header is a real `(group, level)` axis. Entry points:
`estimate::twolevel::fit_ml_twolevel`, `api::twolevel_ml()` +
`api::data_from_cluster()`, R `fit_twolevel`. Named R bounds presets compute
their boxes from the two-level H1 within/between covariance blocks, matching the
ordinary lavaan-style stabilizer scheme without changing the unbounded default.
Gated by `cpp/tests/golden/twolevel_golden_test.cpp` against
`lavaan::sem(model, data, cluster=)`; fixtures from
`cpp/tests/tools/regen_oracle_twolevel.R`, with an Mplus 9.1 Demo cross-check for
unbalanced cells via `cpp/tests/tools/regen_oracle_twolevel_mplus.R`.

Remaining work:

- **L (in progress).** Multi-group two-level: replicate the `(group, level)`
  block axis across groups, with cross-group equality via shared labels as in
  the single-level multi-group path.
- **L (deferred).** Between-only and within-only observed variables: the v1
  shared-observed-set restriction assumes every observed variable decomposes
  into a within and a between part. Lift it to support level-2-only covariates
  and within-only variables (lavaan's general `%WITHIN%` / `%BETWEEN%` blocks).
- **M — the Mplus two-level examples (User's Guide ex9.1a, 9.1b, 9.6, 9.11,
  9.12).** Not yet corpus cases. Mplus fits them by MLR. Four declare
  within-only or between-only variables and wait on the item above; ex9.11 is
  multi-group and waits on multi-group two-level. Hand translations to lavaan
  reproduce the H0 model; the printed statistics are harder:
  - Both programs under-converge the saturated two-level model at their
    default EM tolerance (Mplus Demo sweep: H1 −3502.966 at the default 1e-4,
    −3502.914 at 1e-7; lavaan −3502.941 by default, −3502.912 at 1e-8).
  - ex9.12 has an unexplained ~0.03 log-likelihood offset despite matching
    estimates.
  - Mplus's two-level MLR scaling (0.760 for ex9.6) is not reproduced
    (lavaan 0.96–1.06).

  The corpus side needs `%WITHIN%`/`%BETWEEN%` translation, the reserved
  `data.cluster` field, and a verification standard for these statistics.
- **M (deferred).** Constraints under two-level: `fit_ml_twolevel` currently
  rejects equality/inequality constraints; wire the linear-reduced /
  Jacobian-projected constraint machinery through the two-level fit.
- **XL.** Categorical / robust two-level: ordinal two-level estimators and
  robust (sandwich / scaled) two-level test statistics.
- **XL.** 3+ levels and random slopes: out of the current single-axis design;
  needs the `rv(...)` random-slope modifier (currently parser-rejected) and a
  multi-level block axis.

## Structural-after-measurement (SAM / LSAM) frontier

The C++ frontier core now has lavaan-parity SAM point estimates
(`estimate::frontier::fit_sam`) for local/global ML measurement fits, ML/GLS/ULS
mapping, Fuller lambda correction, classic `se = "standard"` /
`se = "twostep"` from `SampleStats`, and lavaan-style nonnormality-robust
`se = "twostep.robust"` from complete raw data for the single-group local
covariance-only scope. The current robust SE is not yet the misspecification /
estimated-weight sandwich used by the newer profile and residual surfaces.

Remaining work:

- **Done 2026-07-03.** R bindings / developer surface: expose a thin
  `magmaan_core$frontier_sam()` wrapper over the C++ SAM entry, plus exported
  `sam()` helper that parses/lavaanifies and builds `RawData` when robust SEs
  are requested. SAM stays out of `fit_model(estimator = ...)`; SE/test/fit-
  measure calls remain explicit.
- **L/XL.** Robustify SAM beyond lavaan `twostep.robust`: derive and implement
  a misspecification-robust / estimated-weight sandwich with observed-Hessian
  bread, data-dependent mapping / VETA influence terms, and diagnostics that
  show which pieces are active. This should share the moment-IJ accounting style
  used by continuous-LS and ordinal estimated-weight surfaces rather than
  silently reusing the lavaan nonnormality correction.
- **M.** Broaden robust parity fixtures beyond the current deterministic
  two-factor gate: overidentified measurement blocks, non-normal data, and
  optimizer backend variants that still stay inside single-group local
  covariance-only complete data.
- **L.** Extend `twostep.robust` only after matching lavaan conventions for the
  added scope: mean structures need the mean+vech empirical Gamma block, global
  SAM needs the corresponding step-1 Jacobian, and nonzero
  `alpha_correction` needs the lavaan SSC/weighted-SE convention documented
  before implementation.

## Sphere chart / global latent-scale gauge (frontier)

**Current investigation (2026-09-27):**
[engineering/active/15-sphere-reference-fits](../../experiments/engineering/active/15-sphere-reference-fits/report.qmd)
replaces the old local-convergence study's active protocol. Historical results
are preserved. The 60-dataset pilot yields repeated best-observed candidates on
46 ML and 59 PSD problems; nine ML problems have no screened reference, and
one has a screened lavaan witness that both sphere backends recover when started
there. These are exploratory candidates, not optimum or nonexistence proofs.
The start-design follow-up recovers five missing ML references and improves one
existing reference. On 60 fresh draws of the same populations, canonical starts
match the expanded best-observed reference on 37 problems; adding negative-X,
negative-Y and double-negative spectral starts reaches 55. The all-positive
spectral alternative adds no coverage. A witness ablation implicates loading
direction, not variance sign alone. These remain study-local ML starts, not PSD
starts or a production policy; both backends and the current screen are retained.
Expansion to 120 fresh draws in three new simple-structure model families gives
93 canonical versus 118 full-portfolio references. The positive spectral control
adds one result and should be retained; single-negative alternatives miss ten
references found by combined sign patterns. There are 97 repeated, 21 single-start
and two missing references, including one extent-only exclusion.

The direct ordinary-marker transfer probe now constructs starts from sample
moments alone and runs every recipe on every draw. On the retained fresh ML
batch, layered with information scaling matches 35/60 references (L-BFGS) and
36/60 (PORT); adding four spectral starts reaches 49/60 and 53/60, with 55
references available. No sphere fit or reference-selected retry enters this
portfolio. All available targets are marker-representable at 1e-6; no
fold/pole classification has been established. Ordinary PSD with FABIN3-auto
plus positive spectral reaches 57/60 development references without scaling
and 56/60 with diagonal scaling. The follow-up PSD reference run on the retained fresh batch supplies 60
references (58 repeated, two single-start). FABIN3-auto alone matches 47/60;
adding positive spectral reaches 51/60 under each scaling choice. Eight misses
have screened inferior candidates; one has no screened ordinary candidate.
Five exact geometry witnesses now cover an interior target, marker/std.lv
poles, an attained PSD face, and proved unrestricted-ML nonattainment. Both
alternative-marker and sphere routes recover the finite targets. Requested
std.lv fits can pass local checks without attaining the chart's limiting target.
No ML attempt passes the screen on the analytic nonattainment example; all
PSD attempts agree on a checked boundary candidate at positive discrepancy.

The nine missing original/fresh ML references have now been inspected across
nine markers, effect coding, positive std.lv and sphere with common transported
starts and tighter continuations. Two acquire candidates passing the original
screen; six have stable local candidates rejected by the extent cutoff; one
(development weak-marker N=100 draw 2) remains numerically delicate. All nine
have finite locally checked candidates. Two high-R2 extremes arise mainly from
standardization by small total latent variances. Do not call this collection
nonconvergent or infer nonattainment from its missing screened references.

**Seven non-near-pole cases investigated (2026-09-27):** independent 60-digit
ML refinement supplies finite positive-curvature local witnesses for all seven
in the original marker chart. Five refine directly. Fresh weak-marker N=20 draw 1
has a losing sphere path toward var(X)=0, but a better finite negative-X-variance
minimum exists and both sphere backends recover it when started there.
Development weak-marker N=100 draw 2 is resolved by a profile minimum at loading
740.4555345, confirmed at 90 digits; it remains ill-conditioned under the library
accuracy check. Thus six pass ordinary-precision accuracy and one remains a
numerical failure, rather than an established nonattainment example. All seven
remain covariance-inadmissible unrestricted-ML points, separate from PSD fits.

**Single-start selection checked (2026-09-27):** three retrospective sample-only
rules select one saved spectral fit: strongest absolute signed block eigenvalue,
the same rule retaining the constructor for positive signs, and lowest initial
spectral discrepancy. Across the retained 120 draws none gives a clear upgrade.
On the later 60, strongest signed improves ordinary ML 35→37 (L-BFGS) and
36→38 (PORT), each with four gains and two losses; the earlier 60 give no net
gain. Initial-discrepancy selection loses matches. All three miss the inspected
negative-X finite witness, choosing negative Y instead. Keep layered/native
ML and FABIN3-auto PSD starts unchanged; negative variance starts are not PSD
candidates. This is retrospective evidence under the frozen historical screen,
not a fresh policy decision or a claim that single starts cannot improve.

**Paired challenge normalization rerun (2026-09-27):** 120 retained draws,
960 single-start fits, normalization explicitly off/on with the current judge.
Layered/native ML (L-BFGS and PORT/information) and FABIN3-auto direct PSD
(SLSQP/diagonal) have unchanged target recovery. Later-batch direct PSD remains
47/60; two-stage falls 49→46/60. Investigate Ernst N=20 draws 1 and 3 and
high-R² N=20 draw 7: two PSD budget failures and one worse local solution.
Six of seven saved finite witnesses still pass requested-marker accuracy;
L-BFGS reaches one and PORT two, with no normalization gain. The flat loading-740
witness remains ill-conditioned. Both saved near-pole endpoints still fail the
study 1e-4 check but pass 1e-6; no production gate is implied. Preserve all
changed outcomes in engineering/active/15-sphere-reference-fits's normalization-revisit evidence and use its
executive summary for the current result. No spectral/multistart promotion.

**Two-stage regressions investigated (2026-09-27):** fixed ordinary endpoints
crossed with PSD normalization off/on isolate path sensitivity in Ernst N=20
draw 1, endpoint selection in Ernst draw 3, and both effects in high-R² draw 7.
Standalone PSD reproduces the wrapper from the same start; no wrapper mismatch
was found. Increasing the PSD budget 5,000→25,000 does not recover any of the
three targets and sends the two stalled paths to large-loading, worse endpoints.
A general certified-only warm-start rule checked on all 120 saved draws changes
ten handoffs: earlier 54→54 targets (one gain/one loss), later 46→47 (one gain).
Do not promote from this retrospective tradeoff; preserve the current handoff
and keep two-stage provisional. Evidence and loss identities are in
engineering/active/15-sphere-reference-fits's two-stage summary. No magnitude gate or multistart added.

**Exact independent-fallback policy checked (2026-09-27, user clarification):**
ordinary ML is accepted only when converged and PSD-admissible; otherwise PSD
constructs its own FABIN3-auto start, never reusing ordinary estimates. All 120
saved normalized challenge datasets were fitted. Thirty-four stop after ordinary
ML; 86 run PSD. Final acceptance, target recovery and returned objectives match
direct PSD (maximum objective discrepancy <1e-12). Recovery is 54/60 earlier,
47/60 later; against warm fallback there are two gains and one loss. This is a
reasonable candidate; warm starts are not required. No timing advantage or
held-out superiority established. **Remaining:** expose independent versus warm
fallback explicitly if adopting the user's intended policy; the current C++/R
fallback still warms from usable ordinary estimates. The exact test is
engineering/active/15-sphere-reference-fits's two-stage-cold evidence and updated two-stage summary.

**Two-stage timing completed (2026-09-27):** six balanced-order rounds,
five calls per timed block, serial single-thread math on the 120 retained cases.
Per-case median costs sum to 0.553 s direct, 0.703 s independent fallback and
0.839 s warm fallback per bank pass. Independent costs ~27% more than direct,
~16% less than warm. Its 34 ordinary-accepted cases average 0.35 ms versus
0.96 ms direct; the 86 fallback cases cost 8.03 versus 6.06 ms. Direct is fastest
in every round. End-to-end R workflows include errors and current orchestration;
no claim for a future integrated C++ cold-fallback API or other problem mixes.
Timing evidence supports direct PSD for this workload, not a speed argument
for ordinary-first. Independent fallback remains a reasonable policy candidate.

**Normalization timing extension (2026-09-27):** paired off/on timing,
11,520 calls on the same 120 challenge cases. Typical milliseconds are nearly
unchanged: ML 0.47→0.48, direct PSD 1.07→1.09. Total costs change by ~−5% ML,
+14% direct PSD/independent fallback, +34% warm fallback. Almost all direct-PSD
increase comes from later weak-marker N=20 draw 2 (1.46→68.76 ms, now a budget
failure; neither endpoint reaches its reference). Both-certified PSD cost is
unchanged; six failed normalized cases account for ~69% of total PSD time.
Treat this as changed optimizer-path cost, not a 14% scaling overhead on every
fit. The timing summary includes mean/median/tail costs; scope is prepared small
models, current R workflows, no inference. No policy change.

Remaining before a stronger reference claim:

- Separate the magnitude-only exclusion from reference eligibility in a
  future protocol revision; preserve the old frozen classifications. Inspect
  finite primitive contributions and latent-variance cancellation before
  labelling an endpoint runaway. Development weak-marker N=100 draw 2 now has
  a high-precision finite witness but remains a routine-accuracy failure; keep
  that distinction. The fresh weak-marker N=20 draw 1 witness isolates a missed
  basin from a losing path toward a singular regression representation. Positive std.lv only covers the positive disturbance
  sector; identification comparisons must preserve the feasible point/domain.
- **Requested-chart failure policy adopted 2026-09-27:** do not automatically
  change markers. Reject the retained fresh weak-marker N=100 draw 4 endpoint
  in its requested chart (loading about 184,000; chart level 5.44e-6), report
  a near-pole identification problem, and retain alternative-chart output only
  as diagnostic evidence. `scripts/check_requested_chart.R` freezes this
  required-failure witness: explicit study tolerance 1e-4 rejects it, while
  the current 1e-6 accepts it. Validate and implement a general requested-chart
  gate before promotion; do not turn this one example into an untested global
  tolerance change. A second endpoint (development weak-marker N=20 draw 4)
  also fails the study proximity check and already fails sphere accuracy.
  Keep chart failure, PSD feasibility, local accuracy and attainment claims
  separate. Reidentification remains an explicit user choice.
- Inspect the nine fresh PSD ordinary-portfolio misses and two single-start
  references; develop feasible direction/face alternatives only against the
  same PSD target. Frozen fresh evidence is now development evidence. A later
  decision study needs new draws, held-out families and criteria written first.
- A sphere-native accuracy assessment, separate from user-chart translation
  and polish. Reuse existing derivatives and geometry where possible. The
  pilot's strongest-indicator Newton cross-check is explicitly provisional.
- Preserve evaluable sphere endpoints when original-chart finalization fails;
  a chart condition alone must not assert that an optimum lies at a pole.
- Inspect the expansion's 21 single-start references and two unresolved cases
  before adding search machinery. Retain the positive spectral control: the
  new-family check refutes dropping it universally. Find a cheaper alternative
  to enumerating sign subsets only if it preserves useful coverage; single-factor
  changes alone already lose ten references. General SEM start construction
  beyond simple structure and at most three factors remains open. A retry only
  after convergence failure misses screened-but-inferior solutions. Keep
  ordinary-route performance and auxiliary reference discovery separate for PSD
  comparisons; the extent cutoff remains a heuristic.
- The sphere R wrapper currently rejects numeric `control$start` although the
  ordinary fitter accepts it. The pilot uses explicit partable start hints,
  verified as user starts. Unify this input contract when the wrapper is next
  changed; no core workaround is needed for the pilot.

The historical entries below retain their original measurements. Their
"local optimum", "runaway" and "no estimate" labels were heuristic and must
not be reused as mathematical classifications. Neither a failed portfolio nor
an objective below PSD establishes nonattainment. Broader sphere coverage is a
research direction; no default promotion is decided here.

Design and evidence: `papers/global-gauge-sem` (implementation plan of
2026-09-23). The C++ core landed: `analyze_gauge`, `reidentify`, the nonlinear
`optim::reparameterize(problem, ParameterMap)`, and `fit_ml_sphere`,
`fit_gmm_sphere`, `fit_gls_sphere`, `fit_fiml_sphere`, `fit_ml_psd_sphere`.
Every lavaan golden routed through the sphere (`sphere_route`,
`sphere_route_parity`) matches.

Remaining work, tiered:

- **Done 2026-09-23.** R surface: `frontier_fit_sphere()` (ML, ULS, GLS, WLS,
  FIML, `psd = TRUE` for ML), the classed `magmaan_user_chart_singular`
  condition, `frontier_reidentify()`, `fit$gauge`, and sphere refits in
  `case_rerun()` / `modification_indices_lrt()`. (`robust_nested_lrt()` takes
  two fits and never refits.)
- **Done 2026-09-23.** Deterministic sanity experiment
  (`experiments/_archive/sphere-chart-sanity`):
  - **Recovery.** 37 models / 68 fits across ML, ULS, GLS, WLS, FIML and
    PSD-ML reproduce the ordinary fit, including SEs, robust SEs, the
    standardized solution, fit measures, MIs and `:=`. Both routes are equally
    close to lavaan. The gauge classification matched the design in 68/68.
  - **Population recovery.** Exact population moments, including populations
    outside the marker, std.lv or effect-coding charts, are recovered in every
    chart that holds them (sphere ML 24/24, PSD 21/21) and flagged in every
    chart that does not (8/8, 7/7).
  - **Chart invariance.** 48/48 translations between identifications match
    the direct fit.
  - **Bug fixed.** It found and fixed a sphere bug: std.lv-type units were not
    checked against `pole_tol`, so a vanishing endogenous residual variance
    returned a "converged" std.lv fit with a slope near 3.7e4.
- **Open.** Promote to `magmaan(chart = "sphere")` once the author decides the
  surface (plan decision D1). The FIML goldens run through the sphere seam
  too; FIML case 0014 is flat enough that the sweep uses a 1e-4 theta
  tolerance there (same objective to 1e-10). Found by experiment _archive/sphere-chart-sanity and to
  settle before promotion:
  - **Done 2026-09-23 — sphere-native (canonical) start.** The only
    effective dependence of the sphere route on how the user wrote the
    identification was the start. `x0` came from the user partable's start
    policy and was carried onto the sphere, and marker, std.lv and
    effect-coding starts were different points.
    - **Orientation was checked and dropped.** Each unit's internal axes come
      from the user's constraints, but with a shared start, marker- and
      effect-coding-oriented fits agree to about 1e-15 under L-BFGS, PORT and
      SLSQP.
    - **What the start is now.** `SphereOptions::start = Canonical` (the
      default) runs FABIN on the gauge-free internal model. Each unit there
      is identified by a data-chosen marker, fixed in every tied block: the
      unit's indicator with the largest sum of absolute correlations with
      the others, never one the span excludes. That model is the same for
      every identification.
    - **Why a data-chosen marker.** Fixing a variance instead makes FABIN
      reference the first indicator, which re-imports the marker pole: a
      pure-noise first indicator sent GLS to a collapsed point under every
      identification.
    - **Least squares starts from sphere ML.** ULS, GLS and WLS then start
      from the sphere ML solution, which is canonical too.
      - **Why.** LS has no log-determinant barrier. On the sphere every
        indicator is equally reachable, including points where a factor
        collapses onto one indicator with a Heywood residual. The marker
        chart keeps collapse onto a non-marker indicator at infinity.
      - **Evidence.** From FABIN, sphere GLS on two-group HS stopped at such
        a point (fmin 0.182 against 0.163), with either start. From the ML
        solution it reaches 0.163. The golden `0002_multigroup_3f_school`
        GLS now passes through the sphere.
    - **Staged refinement.** The sphere run uses the caller's tolerances,
      then refines from that solution two orders tighter. The refinement is
      kept only if it succeeds without raising the objective.
      - **Why.** The polish cannot improve a point that already meets the
        ordinary stopping rule. The HS real-data ML golden needed the
        refinement to stay within 1e-6.
      - **Why staged.** Tight tolerances from the start raised the Ernst
        N = 20 error rate from 5% to 14%.
    - **Fallback.** Explicit `start()` values or `control$start` select the
      user start. `fit$gauge$start` records which start was used.
    - **Evidence (experiment _archive/sphere-chart-sanity rerun).**
      - Sphere ML, ULS and GLS recover 24/24 populations in the
        identifications that hold them. PSD recovers 21/21.
      - The earlier user-start run missed once each in ULS and GLS.
      - All population translations are correct.
      - Across four identifications, sphere solutions agree to 1e-14, and
        to 5e-7 on the flat PoliticalDemocracy objective, whose objective
        values agree to 3e-12.
    - **Scope.** Invariance covers sphere units only. Passthrough latents
      keep the user's identification by design.
  - **Bounds switch the sphere off.** `bounds = "standard"` box-bounds every
    loading, so every latent stays in the user's chart. Decide whether sphere
    fits translate loading boxes (they are not scale-invariant) or document
    the interaction.
  - **Coverage for a default.** The sphere route refuses ordinal data, ML2S,
    two-level models and PSD for non-ML estimators. Composites run through
    the FC-SEM route (`magmaan_fcsem()`), which has no sphere. A default
    needs either those siblings or a documented ordinary-route fallback.
  - **The switch itself (D1).** Proposed: `magmaan()` uses the sphere by
    default (`chart = "sphere"`, with `chart = "ordinary"` as the opt-out).
    Inputs the sphere does not cover (the coverage list above) fall back to
    the ordinary route automatically, and the fit records the fallback and
    its reason. Switching the default must never break a model that
    `magmaan()` fits today. Also decide:
    - what `fit$gauge` shows on the ordinary route;
    - whether a fit outside the user's identification should be an error
      (the `magmaan_user_chart_singular` condition) or a warning plus the
      sphere solution when it is the default.
  - **Scalar invariance alongside.** Fix the continuous `group.equal`
    intercepts release around the same time (the "M/L — continuous scalar
    `group.equal` release" entry in the ordinal invariance section: group
    2+ latent means stay fixed, npar 60 against lavaan's 63). Invariance
    users are a main audience of a sphere default, and experiment _archive/sphere-chart-sanity had to
    free those means by hand.
  - **Done 2026-09-23 — the driven ML run is scaled like `fit_ml`.**
    - **The bug.** `fit_ml_sphere` switched off the sample-based coordinate
      scaling for its driven problem, while `fit_ml` scales the
      constraint-reduced coordinates by `ml_coordinate_scale`.
    - **Evidence.** On Little's bullying growth model, which has no sphere
      units, the sphere route with default L-BFGS failed on all 40 raw-data
      samples tried (N = 50 to 1132). The ordinary route converged on all.
    - **The fix.** The driven run now takes the same scale for the rest
      coordinates, read off `ml_coordinate_scale` of the internal model per
      kernel column. Unit directions stay unscaled. The scale applies under
      the same conditions as in `fit_ml`, and it also covers the ML stage of
      the least-squares fits.
      - The point, gradient norm and terminal audit are reported in
        unscaled coordinates. `SphereReport::driven_scaled` and
        `fit$gauge$driven_scaled` record the choice.
      - The bullying sphere route now converges on all 40 samples.
    - **Test.** `gauge_test` has a three-wave effect-coded growth model with
      no units. At data scale 0.1 the unscaled driven run fails, while the
      scaled one matches `fit_ml`.
    - **Unchanged elsewhere.**
      - Experiment _archive/sphere-chart-sanity is unchanged: recovery 68/68 and every population
        recovered or flagged. Cross-identification agreement is 1e-12, and
        1e-7 on PoliticalDemocracy, where it was 6e-7.
      - The Ernst probe moves by at most 2.5 points in any cell.
      - Identifications now agree to about 5e-9 along the flattest
        direction, with the same objective to 1e-15.
  - **What the remaining Ernst ML errors are (2026-09-23).** At N = 10,
    27 to 35.5% of sphere ML fits fail (a line-search error or an uncertified
    stop), and 13.5 to 14% at N = 20. Of those 180 failures:
    - 178 sit below the PSD-ML minimum, so the likelihood keeps improving
      into the improper region;
    - L-BFGS, SLSQP and PORT all fail on 163. The uncertified stops examined have
      parameters up to 3.4e3. These suggest escape paths beyond identification
      poles, but do not establish nonexistence; PSD is a different domain.
    - PORT certifies 15, so those are optimizer failures.
    - Against the unscaled run, 39 draws flipped from converged to failed
      and 36 the other way, almost all on such ridges. Whether a ridge stop
      certifies is a matter of the stopping rule.
    - One case where a finite optimum exists and the sphere misses it:
      N = 10, draw 125, SEM form. Ordinary ML converges to an improper
      point (negative disturbance variance, largest parameter 2.8), and the
      CFA form's sphere fit finds it too. The SEM form's sphere fit runs off
      along a ridge at a higher objective.
  - **Models outside the v1 analyzer.** In the bullying growth model all 16
    latents pass through:
    - the five wave factors per process share loading labels across
      latents within one group;
    - the growth factors load on latents.
    Supporting it needs a unit spanning several tied latents together with
    their higher-order factors (literature scope below).
  - **Multiple local optima (probe of 2026-09-23).** Single-start success
    is overstated when judged against the best of a few fits.
    `papers/global-gauge-sem/work/probes/psd_multistart.R` reruns the Ernst
    probe draws (N = 10 and 20, 200 each) against 20 random free-sign
    starts.
    - **PSD-ML.** The canonical sphere reaches the best optimum in 83.5 to
      93.5% of draws, where the earlier probe reported 97 to 100%. The
      ordinary PSD marker route reaches it in 76 to 89%.
      - About half the N = 10 draws (47%) and 28% of N = 20 draws have two
        or more distinct optima.
      - The optima sit on different boundary faces: which indicator gets a
        zero residual (the factor collapses onto it), a zero factor
        variance, or a factor correlation of ±1. 89 of 92 misses are on a
        different face from the best.
      - These are candidate constrained local minima: the polish does not
        move them, and the marker route finds them too. Those observations
        alone do not certify local minimality.
      - Fit gaps are small: median Δχ² 0.6, 90% under 3.1, max 7.2. The
        estimates differ completely, though.
    - **Cheap remedies fall short.** Adding the user start gains 1 to 2.5
      points. Nine sphere fits, one for every marker placement, gain nothing
      more. The face is set by the sign and scale pattern of the start, not
      by the reference indicator. Twenty random starts reach 98 to 100%.
    - **ML.** Against the same reference, the canonical ML sphere succeeds
      in 49.5 to 68.5% of draws (marker 42.5 to 62.5%).
      - Converged canonical fits miss the best in 16 to 21.5% of draws
        (median Δχ² 0.14).
      - Unverified whether those better points are local minima or ridge
        points without established attainment.
      - At N = 10, no fit from any start converged in 14 to 15% of draws.
    - **Parked (author, 2026-09-23).** Global optimality is a later,
      separate question. Experiment engineering/active/15-sphere-reference-fits judges convergence to a local
      optimum. The probe stays in the paper repo as the seed of a future
      experiment. When it is picked up, decide:
      - whether a single start's miss rate is acceptable for a default
        (the marker route misses more);
      - whether sphere fits get an opt-in multistart (random free-sign
        starts, best objective);
      - for PSD, whether a face-aware start is feasible (a research
        question).
- **Done 2026-09-23 — experiment engineering/active/15-sphere-reference-fits, small version**
  (`experiments/engineering/active/15-sphere-reference-fits`, about 70 s).
  - **Setup.** ML and PSD-ML; the Ernst, weak-marker and high-R² populations;
    N = 10 to 100; 200 replications. Ordinary and sphere routes, each under
    marker and std.lv.
  - **Success.** A certified local optimum, or on the sphere route a flagged
    one with a stationary sphere run, whose standardized solution stays
    within 10.
  - **Results.**
    - The two sphere columns agree within about one point.
    - The sphere beats the ordinary marker route by up to 10 points (weak
      marker). It is level with ordinary std.lv from N = 20 and up to 5.5
      points ahead at N = 10.
    - Many failures remain unresolved. All four routes fail on the
      same draw in up to 38.5% of Ernst draws and 61% of weak-marker draws
      at N = 10, still 38% for the weak marker at N = 100.
    - Optimizer failures (another optimizer finds a certified local
      optimum) are rare: 51 draws for ordinary marker, 22 for sphere marker,
      5 for ordinary std.lv and 21 for sphere std.lv, of 2400 each. These
      are recoveries under the historical numerical screen.
    - Routes differ in how they fail on ridges. Ordinary marker errors. The
      ordinary std.lv route reports convergence at a runaway point in up to
      54% of draws. The sphere sits in between.
    - With high R², the sphere flags 40 to 52% of draws at every N as lying
      outside the std.lv identification. On 352 of those 366 draws, ordinary
      std.lv reports a certified but worse local optimum: the best its
      identification can reach.
    - **PSD-ML, added the same evening.**
      - The sphere PSD-ML fit reaches a certified local optimum in 99.8% of
        all 2400 draws (lowest cell 99%). Every draw has an estimate.
      - Ordinary PSD-ML stalls at its identification's pole. The marker
        route drops to 75% with the weak marker. The std.lv route drops to
        50% with high R², where the PSD optimum has Y's disturbance variance
        at zero in half the draws. 494 of its 500 failures are draws the
        sphere flags as outside std.lv.
      - The full grid runs in about 70 s.
  - **Two issues found.** Both are listed below.
- **v1, S. Sphere flags at runaway stops.** The
  `magmaan_user_chart_singular` condition is raised without checking the
  driven optimizer status. With PORT on the std.lv identification (Ernst
  and weak-marker draws), stops reported as `noisy_objective`, with
  sphere-chart parameters of 500 to 84,000, were flagged. Require a clean
  or audited driven stop before flagging, and otherwise report
  non-convergence. `fit$gauge$driven_stationary` now exposes the driven
  audit.
- General runaway-ridge detection from both routes is deferred to
  [the speculative backlog](speculative.md#runaway-estimates-and-nonattainment-diagnostics);
  the concrete sphere-status reporting fix above remains active.
- **v1, M. Step two: expand experiment engineering/active/15-sphere-reference-fits to the full Ernst-type
  simulation.** This follows experiment _archive/sphere-chart-sanity and uses the canonical start.
  - **Designs:**
    - the Ernst SEM and CFA forms (3 indicators, loadings 1/.8/.6, beta in
      {0, .25, .5});
    - a weak marker (lambda_1 in {.1, .3});
    - a high-R2 endogenous latent (R2 in {.9, .98}), where experiment research/46's
      std.lv reversal sits;
    - a small two-group metric-invariance design.
  - **Grid:**
    - N in {10, 20, 50, 100, 200};
    - ML, plus at most one least-squares estimator (the author's call on
      2026-09-23, possibly ML alone). GLS is the candidate, since the
      collapse points showed up there. PSD-ML, ULS and FIML drop out of
      the grid.
    - marker and std.lv identifications.
  - **Outcomes, per draw.** The first business is convergence to a local
    optimum, not global optimality (author, 2026-09-23). Success means
    reaching a certified local optimum:
    - no error;
    - the domain-specific stationarity audit passes;
    - either in the user's chart or correctly flagged outside it.
    Also recorded:
    - the error type. The protocol from the classification above: refit
      with PORT and SLSQP; a certified optimum there means an optimizer
      failure; otherwise an objective below the PSD-ML minimum with growing
      parameters means a nonexistence ridge;
    - a silent wrong answer (reported converged, audit fails);
    - flagged outside the identification;
    - largest parameter and wall time.
    - For LS, how often the sphere and ordinary fits land in different basins
      from the same data.
  - **Cost.** Estimate from a `--smoke` timing trial before choosing reps,
    with Modal or Saga if the grid needs it.
- **Idea (author, 2026-09-24): name the cause of a failed fit.** Within the
  closedness theorem's scope (simple structure, free latent covariance, S
  positive definite) the PSD sphere fit always has a minimum to find. A fit
  that fails in the user's identification can then be explained rather than
  retried. Compare the ordinary fit with the sphere fits:
  - **Sphere fits, marker fit fails.** The marker's loading is about zero at
    the optimum. Choose another marker, or the item barely measures the
    factor.
  - **Sphere fits, std.lv fit fails in a regression.** The endogenous
    factor's disturbance variance is about zero. The factor is perfectly
    predicted, so the two factors are not distinct (discriminant validity).
  - **PSD fit on a singular face.** An exact Heywood case. Name the zero
    variance, or the factor correlation at ±1.
  - **Sphere ML fails, sphere PSD fits.** No unconstrained estimate exists.
    The likelihood keeps improving toward an improper solution.
  - **Sphere PSD fails.** Outside the theorem (structural cases such as the
    mediation chain), or a genuine optimizer failure.
  - **Evidence (experiment engineering/active/15-sphere-reference-fits).** Refitting in the other ordinary
    identification is not a complete remedy. In 26 of 2400 PSD draws (weak
    marker and high R²) neither the ordinary marker nor the ordinary std.lv
    fit succeeds, while the sphere does.
  - **Caveat.** At small N a near-zero marker loading is often sampling
    noise: the Ernst marker's true loading is 1. The message is "these data
    cannot set the scale through that item", not "wrong item".
  - **Caveat: a failed fit can be a stall.** The diagnosis must check
    whether the sphere optimum lies inside the user's identification. In
    the Ernst PSD design at N = 10, all 28 marker failures were SLSQP stalls
    at optima the marker identification contains (marker shares 0.005 to
    0.84). Only then is the cause the optimizer, not the identification
    (PSD-ML queue item 7).
  - **v1, M (after D1).** Report the cause on failed or flagged fits. The
    `magmaan_user_chart_singular` condition already names the unit that left
    the identification. Add the cause and the parameter at the boundary.
    Decide whether this lives in `fit$gauge` or a separate diagnostic call.
  - **Paper.** A selling point for the sphere paper: a fit that explains its
    own failures. Check prior art before claiming novelty. "Try std.lv or
    another marker" is common advice, the reference-indicator literature
    concerns invariance testing, and empirical underidentification
    (Rindskopf 1984) covers the weak-marker case. It is not known to be
    systematized as a diagnostic.
- **Literature scope.** PSD LS/FIML siblings, ML2S, multi-information penalty,
  two-level slots, higher-order loadings when the lower-order factor is itself
  a unit, units spanning several latents tied within a group (longitudinal
  invariance, as in Little's bullying growth model), partial invariance via nested spheres, SNLLS (sphere in the nonlinear
  block), ordinal, sphere-chart Wald SEs through the constrained information.
- **Research questions.** Closure of translated nonlinear restrictions,
  Henseler-Ogasawara composites under the sphere, existence for SNLLS on the
  compact outer problem, mediator-chain structural nonexistence.

## Misspecification-robust SE for the moment-quadratic family (frontier)

Observed-Hessian bread, complete/MCAR estimated-weight IJ adapters, the ML2S
Stage-1 Gamma influence, profile RMSEA/LRT and scalar parameter-profile
references are implemented. All-ordinal and mixed fit-index bundles and FIML
observed/robust covariance dispatch are exposed in R. The
[roadmap](../architecture/roadmap.md#ordinal-and-mixed-categorical-ls) owns
capabilities and validation boundaries; the
[test ledger](../validation/test_ledger.md) records cross-subsystem guards.
Completed rollout entries are not another work queue.

### Reduced-bias estimation

The reduced-space parts and explicit/implicit drivers are shipped; their
independent FD guards are in `rbm_fd_test.cpp`. Remaining:

- **Deferred — performance and regular-region checks (2026-09-22).**
  RBM has been dropped from the current inference-study expansion at the
  author's request. The optimization prototype was reverted; no new RBM
  implementation or search restriction is shipped. Reopen only for an
  explicitly renewed RBM consumer.
  Complete-data implicit ML finite-differences the full adjusted objective in
  full theta coordinates: `1 + 2*n_free` parts rebuilds per optimizer
  value/gradient callback, even when equalities reduce the search dimension.
  Each rebuild includes observed information and raw empirical scores; the
  base likelihood's analytic gradient is computed and discarded. Add a
  value-only ML penalty/parts interface so external derivative audits need
  not invoke a complete explicit correction to read the penalty. Retain the
  analytic base gradient, differentiate the adjustment in reduced coordinates,
  and investigate model/sample-feature reuse and analytic trace derivatives.
  For complete-data Gaussian scores, cache centered linear/quadratic data
  features (or their empirical cross-product when smaller) once per fit;
  preserve empirical meat rather than replacing it by a normal expectation.
  Independently verify off-optimum scores, means, equality reduction and
  small/large-N branches before accepting that optimization.
  The existing trace solve checks LU invertibility, not positive curvature;
  finite indefinite-information endpoints can have large negative trace
  adjustments. Specify conditioning/curvature acceptance and a documented
  regular-region search policy before treating such returned endpoints as
  validated inference estimates. Preserve the distinction between search
  restrictions, estimator definition, and solver return status. Benchmark
  fitting and independent audit separately before scaling up simulations.
- **M — magmaan-owned paper rerun.** Experiment
  `06-jamil-rosseel-2026-rbm-sem` now reproduces the SEM bias-reduction
  paper's main two-factor and growth-curve examples from the authors' OSF
  result objects. Remaining work is the independent magmaan rerun: implement
  the bounded-estimation choices needed for the paper design, rerun the
  normal/non-normal cells from generated data, and compare magmaan ML/eRBM/iRBM
  to the OSF summaries.

### Estimated-weight and hybrid validation

- Continue finite-sample calibration and stress checks for singular full-WLS
  Gamma on mixed hybrid observed-data fits. Existing hybrid construction keeps
  ordinal moments pairwise and obtains continuous moments/influence from
  saturated FIML; its MCAR/MAR efficiency checks do not establish this coverage.
- Keep the ordinary NT ML2S robust-score path distinct from the
  moment-quadratic GLS IJ correction. Robust/experimental mixed missing-data
  Stage-1 variants (WMA/DPD/Huber polyserial) remain in the speculative backlog
  until a recipe and consumer are selected.
- New IJ adapters require fixed-weight reduction, independent weight-influence
  derivatives, a deterministic resampling-jackknife check and misspecification
  simulation with ULS/fixed-weight negative controls. Existing analytic
  Hessian/Gamma channels retain finite differences as validation references.

### Profile and fit-index follow-ups

- **TODO: survey the multigroup CRMR/SRMR pooling convention** (short lit
  survey). Multi-group is implemented, but the CRMR point has two defensible
  forms that diverge for G>1: lavaan / `ordinal_crmr` use the size-weighted
  *mean of per-group roots* `Σ_b (n_b/N)·√(‖r_b‖²/k)` (oracle-pinned for the
  core fit measure), while `ordinal_crmr_misspec_inference` reports the literal
  RMS, the *root of the size-weighted mean* `√(Σ_b (n_b/N)‖r_b‖²/k)`, forced by
  the `T = N·G` statistic its CI is built on; by Jensen the former ≤ the latter,
  equal at G=1 (pinned in `ordinal_test.cpp`). Two questions: (1) mean-of-roots
  vs root-of-mean (root-of-mean is the literal "root mean square residual" and
  coheres with the CI; lavaan's is the nonstandard form); (2) the weighting
  itself: CRMR is not a discrepancy (identity metric, no `W`), so `π_g = n_g/N`
  is not derived; it enters legitimately only via the pseudo-true
  `θ_0 = argmin Σ_g π_g F_g`, whereas the explicit `π_g`-weighting of the pooled
  residuals is inherited convention plus the independence-additivity that keeps
  `Var(T) = Σ_g n_g(·)` a clean block sum. Unweighted stacked-RMS
  `√(Σ_b‖r_b‖²/Σ_b k_b)` is a third, equally-legit target. Survey what
  lavaan / Mplus / EQS and the SRMR literature actually do, then decide which
  the frontier index should report (current: weighted root-of-mean) and whether
  to expose a `crmr_lavaan` companion. From the 2026-06-22 audit discussion.

- Assess whether the small-pencil `max|nu_j-1|` diagnostic can skip negligible
  dense profile-curvature work; keep dense `Q Gamma` when actual mixture
  weights are needed. An a-priori analytic sign count from model structure
  remains separate from the implemented inertia identity/eigendecomposition.
- Stabilize the CFI-to-TLI scale ratio when the user model's signed trace is
  small. The existing TLI variance is conservative at strong misfit; the
  multi-group extension did not resolve this quality gap.
- **Deferred:** close-fit/boundary calibration and the classical-noncentral
  RMSEA comparator require an explicit validation design.
- **Deferred CRMR/SRMR coverage:** threshold-inclusive variants, residual-map
  curvature, and lavaan CRMR oracle/golden checks. Preserve the distinction
  between the core mean-of-roots point and the frontier root-of-mean CI target.
- Per-index R bindings remain deferred because the bundles already expose
  them. Main `fit_measures()` scaled/robust table integration remains separate.
- Expose the continuous-LS `bread` choice for `vcov(fit, regime=)` at the R
  boundary, and consolidate `ordinal_block_residual` into one implementation.

## Robust score / modification-index tests (frontier)

`inference::frontier::{modification_indices,score_tests}_robust` has landed for
complete-data ML (both breads), continuous ULS/GLS/WLS/DWLS (the
`gmm::Weight` overloads, expected bread, moment-metric sandwich), FIML/MLR
(`{modification_indices,score_tests}_fiml_robust`, observed bread + casewise
meat), and — as
`estimate::frontier::{modification_indices,score_tests}_{ordinal,mixed_ordinal}_robust`
— all-ordinal ULS/DWLS/WLS and mixed DWLS/WLS over the polychoric NACOV meat.
The continuous ML/LS, ordinal/mixed-ordinal, and FIML/MLR tiers are single- or
multi-group; a df>1 total-release (`score_tests_robust_joint`, mean-scaled +
imhof mixture) covers joint releases.
Validated by `cpp/tests/unit/score_robust_test.cpp`,
`cpp/tests/golden/score_robust_golden_test.cpp` fixtures 0006-0012, the R-internals
oracle from `cpp/tests/tools/regen_robust_score.R`, and the advisory
`cpp/tests/checks/robust_score/`. Remaining work:

- **Done 2026-06-13.** Multi-group robust MI/score tests now cover the
  continuous ML/LS tiers, the ordinal/mixed-ordinal tiers, and the FIML/MLR tier.
  The ordinal guard in `estimate::frontier` and the FIML guard in
  `inference::frontier` are removed after validating the same block-stacked
  nuisance projection: per-block `n_b/N` sandwich over polychoric NACOV with the
  two-group WLSMV golden 0012, exact WLS reductions for all-ordinal MI/score and
  mixed-ordinal MI/score, and a two-group FIML raw/pack parity regression over
  unequal raw block sizes and MCAR patterns.
- **R wrappers done 2026-06-14.** `modification_indices_robust()` /
  `score_tests_robust()` (R/`context.R`, exported) now wrap the LS and ordinal
  robust tiers: new Rcpp glue `inference_{modification_indices,score_tests}_robust`
  (`r-package/src/fit.cpp`) dispatches ordinal/mixed -> `estimate::frontier`
  (intrinsic NACOV-meat scaling) and continuous ML/ULS/GLS/WLS ->
  `inference::frontier`, reconstructing the LS weight in glue (ULS identity / GLS
  normal-theory / WLS explicit) and taking `bread`/`moments`/`cov` plus the raw
  fitting data for the empirical meat. The `bread/moments/cov` -> `InferenceSpec`
  parsers moved to the shared `r-package/src/internal.hpp`. The original engineering demonstration was removed on 2026-09-30;
  its ordinal DWLS/continuous GLS scaling and exact `c -> 1` reductions are
  maintained in `score_robust_test.cpp` and `score_robust_golden_test.cpp`. Still open / only-when-needed: a C++ `api::frontier` entry point taking
  `api::Fit` for the LS tiers (the api `Fit`'s `EstimatorSpec.weight` is empty for
  GLS/DWLS, so it would have to recompute the weight; the ML `api::frontier`
  robust overloads already exist). The R path sidesteps this by reconstructing the
  weight in glue, so no consumer needs the C++ api entry point yet.
- **Done 2026-06-22 — estimated-weight ("complete-sandwich") robust MI.** The
  per-direction robust scaling `c = gᵀB1g / gᵀA1g` can now use the complete
  (Hall-Inoue infinitesimal-jackknife) meat — with the data-dependent-weight
  `IF(Ŵ)` term — instead of the fixed-weight `Δ'WΓ̂WΔ`, via a new
  `estimated_weight` flag. This is the per-parameter denominator lavaan never
  builds (it scales MI only by the global SB scalar `c`). New core primitives
  `estimate::weighted_param_space_sandwich_ij` and the IJ adapters
  `continuous_ls_param_space_sandwich_ij` (continuous GLS/WLS/DWLS/DLS, via the
  shared `build_continuous_ls_ij_blocks`) and `ordinal_param_space_sandwich_ij`
  (all-ordinal DWLS/WLS, via the shared `build_ordinal_ij_blocks` factored out of
  `robust_ordinal_ij` so the SE and MI paths build identical meat). Threaded
  through `inference::frontier::RobustScoreOptions.{estimated_weight,
  ij_weight_mode}` and the ordinal/mixed `estimate::frontier` entry points
  (mixed-ordinal errors: not yet wired); ML/FIML reject the flag (no estimated
  second-stage weight). R: `modification_indices_robust()` /
  `score_tests_robust()` gain `estimated_weight = FALSE`
  (`r-package/examples/estimated_weight_modindices.R`). The correction is
  leading-order only under misspecification (Hall-Inoue order-promotion; ULS has
  a fixed weight and is unaffected). Reduction anchors and a misspec-shift check
  in `cpp/tests/unit/score_robust_test.cpp`. Open follow-ups: mixed-ordinal IJ meat;
  an MC calibration cell in `cpp/tests/checks/robust_score/` showing the
  estimated-weight `mi.scaled` is better-calibrated than fixed-weight under DWLS
  misspecification; these new IJ sandwich primitives ride the pending
  `<domain>::frontier` retier below.

## Residual summary (lavResiduals parity)

- **Done 2026-06-23 — `lavResiduals(fit)$summary` table.** `measures::
  standardized_residuals` now fills a per-block `ResidualSummary` (`cov`, and
  `mean`/`total` under a mean structure), the analogue of
  `lavResiduals(fit)$summary` with the default `type = "cor.bentler"`: the SRMR
  family (SRMR, its asymptotic SE, the exact-fit z-test against 0) and the
  bias-corrected USRMR with a close-fit CI and a close-fit z-test against 0.05.
  The cor.bentler residual ACOV is the existing raw-metric `acov_res`
  (`Q·acov_obs·Qᵀ` in `fill_residual_z`) congruence-scaled by the sample SDs
  (Ogasawara 2001 eq. 13; verified `GG·acov_raw·GG == lavaan cor.bentler acov` to
  machine zero), so no new projection — just the RMS summary matching lavaan's
  `lav_residuals_summary_rms`. R: `lav_residuals(fit)` (exported) and
  `residuals(fit, standardized = TRUE)$summary` carry it; one C++ call.
  Gated full-precision vs lavaan 0.7.1.2691 in `cpp/tests/unit/residuals_test.cpp`
  (no-mean cov column; meanstructure cov/mean/total) and end-to-end vs live
  lavaan in `r-package/examples/lav_residuals_summary.R` (single- and
  multi-group, ~1e-6/1e-5). Note: only the default cor.bentler/SRMR type is
  built; the raw (RMR) and cor.bollen (CRMR, needs the `lav_deriv_cov2cor_b`
  Jacobian) summary types are not — add if a consumer appears. Ordinal residual
  summaries are a separate object (`OrdinalMisspecFitMeasures`), left untouched.

- **Done 2026-06-23 — estimated-weight ("complete-sandwich") standardized
  residuals (frontier).** `measures::frontier::standardized_residuals_estimated_
  weight` (new `measures::frontier` namespace) gives the residual SE/z and the
  `$summary` inference an estimated-weight residual ACOV for continuous LS fits
  (GLS/WLS/DWLS/DLS), instead of the NT projection + Γ_NT that lavaan and the NT
  path use. The residual influence function is `IF_r = M·Qᵀ − C·P` (residual-
  maker `Q = I − P·W`, `P = n_b·Δ_b·A⁻¹·Δ_bᵀ`, `A` the pooled LS bread, `C =
  weight_correction` the Hall-Inoue IF(Ŵ) rows); the ACOV is the empirical
  covariance of those rows. New core primitive `estimate::continuous_ls_residual_
  acov_ij` reuses the Stage-A `build_continuous_ls_ij_blocks`; `fill_residual_z`'s
  standardization was factored into a shared `apply_residual_standardization` the
  override path also calls. With a fixed weight (`C = 0`) it collapses to
  `Q·(Γ̂/n)·Qᵀ`, gated EXACTLY in `cpp/tests/unit/residuals_test.cpp` against a hand-
  built projector (1e-8); a DWLS weight-correction shift test isolates the `C`
  term. R: `residuals(fit, standardized = TRUE, estimated_weight = TRUE, data =)`
  / `lav_residuals(fit, estimated_weight = TRUE, data =)`
  (`r-package/examples/estimated_weight_residuals.R`); ML/FIML/ordinal reject the
  flag. Advisory bootstrap MC `cpp/tests/checks/residual_estimated_weight/` confirms
  the calibration: under t₆ heavy tails the estimated-weight SE / bootstrap-SD
  ratio ≈ 1.02 while the NT ratio ≈ 0.68 (NT under-states the residual
  variability by ~⅓). Open follow-ups: multi-group couples blocks only through
  the shared bread (per-block meat, the lavaan residual-ACOV convention);
  mixed-ordinal / categorical estimated-weight residuals are not wired; these
  primitives ride the pending `<domain>::frontier` retier below.

## Case-level influence diagnostics (semfindr parity)

The **exact leave-one-out engine and the approximate parameter-change engine
landed 2026-06-23** (`r-package/R/case_influence.R`: `case_rerun` /
`est_change_raw` / `est_change` (+`gcd`) / `fit_measures_change` /
`mahalanobis_rerun` exact; `est_change_raw_approx` / `est_change_approx`
one-step), single-group continuous ML/ULS/GLS, semfindr-format, validated by
`r-package/examples/case_influence_semfindr.R` and frozen fixtures under
`cpp/tests/fixtures/case_influence/`. The one-step engine rides a new core accessor
`inference::casewise_scores` (the N×n_free per-case score matrix; bound as
`infer_casewise_scores_fit`, == `lavaan::lavScores`); `information_cross_products`
is now its Gram. Remaining:

- **Upstream PR to semfindr (est_change_approx scaling).** semfindr 0.2.0
  `est_change_approx()` applies `N/(N-1)` twice to DFTHETAS and once (too few)
  inside `gcd_approx`; both are spurious O(1/N) factors with no first-principles
  basis (`est_change_raw_approx` is correct). magmaan uses the correct scaling
  and gates transitively up to the documented factors; see the
  `project/validation/oracle-defects.md` entry. File a PR / issue against
  `sfcheung/semfindr` with the influence-function derivation. **S**
- **`fit_measures_change_approx` — NOT supported (by design).** semfindr's
  no-refit fit-measure change exists to dodge expensive refits; magmaan's refits
  are cheap summary-stat fits and are already reused from `case_rerun`, so the
  exact `fit_measures_change` covers this leg at machine precision for no extra
  cost. The approximation would be lossier (first-order `2·(ll1_i − ll0_i)`) and
  its CFI/TLI would inherit the `fixed.x` baseline gap. It is implementable in
  pure R (casewise MVN log-density at model/saturated/baseline moments — no core
  accessor needed); build only if a concrete large-N, approximate-only consumer
  ever appears.
- **Done 2026-06-23 — `fixed.x` baseline df gap.** `fit_measures()` reported
  `baseline.df` one too high per exogenous covariance vs lavaan (path / `pa_dat`:
  magmaan 6, lavaan 5), so CFI/TLI diverged on `fixed.x` models (chisq/rmsea were
  fine). The core correction already existed (`measures::baseline_chi2(pt, samp)`
  frees the exo (co)variances); the R `fit_measures()` was just calling the
  partable-unaware `infer_baseline(ss)`. Fixed by adding `infer_baseline_fit(fit)`
  (calls the partable-aware overload; a no-op without exo, so safe for all fits)
  and pointing `fit_measures()` at it. Path-model CFI/TLI now match lavaan to
  machine precision; the case-influence example and fixture use all four
  measures; full `just r-examples` clean.
- **Done 2026-06-23 — multiple-group `case_rerun` / `est_change`.** Pass the
  original `data` frame to `case_rerun(fit, data = …)`; it refits on `data[-i, ]`
  through the canonical `df_to_data` pipeline for any number of groups (case ids
  = data rows, as in semfindr). Columns are suffixed with the group *label*
  (magmaan and lavaan order groups differently — magmaan by factor level, lavaan
  by data appearance — so a label keeps comparisons tool-independent).
  `mahalanobis_rerun(fit, data)` computes per-group distances placed at the
  original rows. Gated vs semfindr in `case_influence_semfindr.R` (2-group HS
  CFA, `meanstructure = FALSE` to match): est_change_raw 7e-6, est_change 5e-5,
  fit_measures_change 2e-9, Mahalanobis exact. The one-step `*_approx` engine is
  still single-group (errors clearly on multigroup; the block-stacked scores
  would need original-row reordering — extend if needed).
- **Done 2026-06-23 — robust-regime `est_change`.** `est_change(rerun, se = )`
  takes `"standard"` (default, naive ML), `"robust.sem"` (Satorra-Bentler /
  expected-bread sandwich), or `"robust.huber.white"` (MLR / observed-bread
  sandwich); the robust regimes standardize and form gCD from the per-refit
  robust vcov (`robust_se_raw_fit(refit, refit$raw_data$X, bread)`, reusing the
  raw data every fit carries). Gated vs semfindr on robust-SE fits in
  `case_influence_semfindr.R` (~1e-5).
- **Done 2026-06-23 — misspecification-robust ("complete-sandwich") case
  influence (frontier).** The casewise dual of the estimated-weight SE: the
  one-step leave-one-out change carries the per-case data-dependent-weight term
  the naive (semfindr / Pek-MacCallum) influence drops by treating the estimator
  weight as fixed. New core accessor
  `estimate::continuous_ls_casewise_influence_ij` returns the `N×q` per-case
  influences `c_i = (1/N)·K·A⁻¹·(Δ_b K)ᵀ·v_i` (observed bread; `v_i = g_i·W + IF(Ŵ)`)
  plus the fixed-weight `influence_naive`; by construction `Σ_i c_i c_iᵀ`
  reproduces the `robust_continuous_ls_*_ij` vcov exactly (pinned at 1e-9 in
  `weighted_inference_test.cpp`). Reuses the shared `build_continuous_ls_ij_blocks`
  from the estimated-weight stream (no new IJ math). Bound as
  `infer_casewise_influence_ij_fit`; R surface
  `est_change_raw_approx(fit, type = "estimated.weight")` /
  `est_change_approx(..., type = "estimated.weight")` (continuous GLS/WLS/ULS),
  the raw output carrying `"naive"` / `"weight_diagnostic"` (`Δ'W'_d`) attributes.
  Self-validated against the EXACT GLS leave-one-out engine (which re-estimates
  the weight per drop) in `r-package/examples/case_influence_estimated_weight.R`:
  the complete one-step tracks the exact refit at RMSE ~3e-4 in both regimes,
  while the naive error grows with misfit (2.8e-3 at cfi 0.94 → 1.3e-2 at cfi
  0.85) and the complete error stays flat — the Hall-Inoue order promotion
  (`O_p(N⁻¹)` at the null → `O_p(N^{-1/2})` off it). Writeup in
  `papers/estimated-weight-se` (the casewise-dual section).
- **Done 2026-06-23 — full estimator/group coverage for the case-influence
  one-step engine.** Both regimes (`standard` and `estimated.weight`) are now
  **multiple-group**: the bindings already block-stack the per-group cases, so
  the R surface passes the full per-group raw list and labels rows `g{b}_{row}`
  with group-suffixed columns (the standard multigroup one-step tracks the exact
  ML refit at RMSE 3.6e-3, cor 0.93). The per-case row extraction was factored
  into the shared `estimate::casewise_influence_from_ij_blocks(blocks, K, bread)`
  (the per-case dual of `robust_weighted_moment_ij`), reused by both the
  continuous accessor and the new **ordinal** `estimate::ordinal_casewise_influence_ij`
  — the categorical (DWLS/WLSMV/ULSMV) headline cell, riding the existing
  `build_ordinal_ij_blocks` + `ordinal_observed_bread_analytic`, bound as
  `infer_ordinal_casewise_influence_ij_fit` and routed automatically when
  `fit$ordinal`. Continuous WLS/ADF (`SampleEmpiricalWls`) is also covered. Every
  cell self-checks `Σ_i c_i c_iᵀ ≡ robust_{continuous_ls,ordinal}_*_ij` vcov to
  1e-8/1e-9 (`weighted_inference_test.cpp`, `ordinal_test.cpp`), and the R
  example exercises multigroup + ordinal.
- **Done 2026-06-23 — two-stage (ML2S) case influence**, the missing-data member
  of the estimated-weight family. `estimate::fiml::two_stage_casewise_influence_ij`
  decomposes the ML2S complete sandwich: each case influences θ̂ through the
  Stage-1 saturated-moment influence (`saturated_em_moment_influence`) AND the
  Stage-2 data-dependent-weight term (`ml2s_weight_correction_block`). It factors
  the same IJ-block assembly the missing-data SE sandwich uses (`build_ml2s_ij_blocks`)
  and ends in the shared `casewise_influence_from_ij_blocks`; non-NT complete data
  routes through `continuous_ls_casewise_influence_ij` (the SE's complete-data
  route). The NT weight (lavaan robust.two.stage) treats the weight as fixed, so
  its correction is exactly zero (complete == naive); the non-NT Stage-2 weights
  (DWLS/ADF/DLS) carry the live term (HS missing+misspecified DWLS: naive-diff
  0.036). Bound `infer_ml2s_casewise_influence_ij_fit`, routed by the `^ML2S`
  estimator label (Stage-2 weight parsed from `ML2S_{ULS,DWLS,ADF,DLS,WLS}`).
  Self-check `Σ_i c_i c_iᵀ ≡` the observed-bread ML2S IJ vcov to 1e-8
  (`fiml_test.cpp`); R example exercises NT (zero) vs DWLS (live).
- **Remaining (case-influence one-step):** the per-drop exact leave-one-out
  *simulation figures* for the categorical/DWLS and ML2S cells (paper
  deliverables, not library code — `case_rerun` is ML/ULS/GLS; re-estimating
  polychorics / re-running the Stage-1 EM per drop is expensive; the GLS
  exact-LOO already validates the one-step semantics and the self-checks pin the
  math). Mixed ordinal-continuous case influence stays blocked on its
  estimated-weight SE (not yet wired). Ties to the model-free DOCR reference
  (`external/refs/jaffari-2024-model-free-case-influence-sem.pdf`).

## Local hardening and validation tooling

- **S — two R examples fail on their own assertions.** `r-package/examples/ml_psd_fallback.R`
  stops at `!e$converged` and `score_flip_test.R` at
  `a$mean_variance_relative_shift == 0`. Both fail identically on the package
  built before the 2026-09-25 package rename, so the cause predates it.

Local-first safety tooling for an AI-assisted repo. Design note:
[project/validation/local_hardening.md](../validation/local_hardening.md). The test
ledger, risk map, regression-note convention, and JUnit/health recipes have
landed; remaining open items:

- **Done 2026-06-15 — coverage lane calibrated.** A full `just coverage` run
  (clang/llvm-cov 21, all nine `magmaan_test_*` binaries) was interpreted and
  written up in
  [project/validation/local_hardening.md](../validation/local_hardening.md)
  ("Interpreting a coverage run"). Findings: (1) the `171 functions have
  mismatched data` warning is benign — header-defined inline/template functions
  carry different structural hashes across test TUs; the count scales with the
  number of combined binaries (0/19/171 for smoke/ordinal/all-nine) and is
  emitted at load time, so it cannot be suppressed via `--ignore-filename-regex`
  and must not be piped away; line/region coverage of out-of-line `cpp/src/*.cpp`
  (one consistent hash from the single static lib) is accurate, only the function
  denominator is mildly understated. (2) Object list (the nine targets) and
  ignore list (`/_deps/|/third_party/|/tests/|/usr/`) reviewed and correct;
  no changes. (3) Domain-level snapshot table added with dark-room reading —
  `api`/`compat` read artificially low because they are R-boundary-validated
  (lane is C++-test-only), while `robust`/`sim` low spots track known backlog
  gaps. Refresh the snapshot after major test/source changes; the ordering, not
  the absolute %, is the durable output.
- **M, audit.** Audit test tolerances so the suite is not "faked" by loose gates
  that pass a known divergence instead of pinning the right answer. Template
  already fixed: continuous GLS/WLS chisq was gated at `max(5e-2, 2·N·fmin·2e-3)`
  (≈1.2) in `lavaan_parity_golden_test` and `ls_golden_test`, absorbing the whole
  `(N−G)/N` multiplier gap; both now pin `magmaan·(N−G)/N == lavaan` to 5e-3.
  The ordinal goldens are now swept too (2026-06: same multiplier found under
  the 8e-2 ordinal chisq gates; rescaled and tightened to 5e-3, with the
  per-group `(n_g−1)/n_g` estimator-weighting consequence documented in
  numerical-conventions exception 4 and the test ledger). Follow-up audit:
  the real-data bfi ordinal block in `lavaan_parity_golden_test` had the same
  stale multiplier gate (and used relative tolerances); it now rescales to
  lavaan's statistic and pins χ²-family quantities to 5e-3, and finite lavaan
  fit-measure fields now fail if magmaan returns non-finite values, which
  surfaced and fixed the saturated-model TLI convention (`df_user = 0` → TLI =
  1). Soft-gap hygiene pass (2026-06): ptable tolerated divergences and the
  matrix-rep deferred set now have exact count checks, and the skipped
  Little/Newsom broad corpus check no longer aborts on Little's null mean
  placeholders; forcing it with `--no-skip` reaches the single documented
  `newsom/ex5_5b` backend failure. The remaining golden soft sets are now
  swept too: `fit_implied`, `fit_measures`, `fit_theta`, `inference`,
  `test_stats`, and `observed_inference` count-pin their skip/defer/
  needs-regen/no-oracle buckets, and the old silent `test_stats` `pe_z`
  pre-regen path is now recorded and pinned empty. The self-referential-fixture
  audit came back clean: all fixture families are anchored to lavaan,
  lavaan-internals, PearsonDS, SuppDists, rvinecopulib, robcat, or analytic
  references; `paper_corpus`'s misleading `export_magmaan.R` name still runs
  `lavaan::sem()`. Observed-inference currently has no no-oracle fixtures
  under the regenerated lavaan-backed corpus; that bucket is literal-count
  pinned at zero so future structural missing-oracle cases must be explicit.
- **M, calibration parity for robust statistics.** Single-dataset value parity
  cannot detect a miscalibrated scaling *convention*: a robust or scaled statistic
  can match the oracle on one draw (or be exempted as a "known convention
  difference") yet carry the wrong sampling distribution and mis-size the test.
  The nested Satorra-2000 eta-space scaling did exactly this — it passed value
  parity as a KNOWN divergence ("magmaan believed correct" on normal-data
  evidence) while over-rejecting 5x under nonnormality (.45 vs lavaan .085 at the
  strict rung), caught only by a paired rejection-rate Monte Carlo and fixed by
  the `convention = "lavaan"` selector (`55eeb77`). Write-up:
  [project/validation/calibration-parity.md](../validation/calibration-parity.md).
  Two follow-ups: (1) add a Monte Carlo calibration check (advisory, under
  `cpp/tests/checks/`) that pins magmaan rejection rates to lavaan **and** to nominal,
  in the regime the statistic exists to handle (nonnormal / missing / ordinal),
  for the robust-statistic family: Satorra-Bentler scaled and scaled-shifted GOF,
  Satorra-2000 nested (FIML and ML2S), FMG/pEBA spectrum, robust SE / Wald, ML2S,
  ordinal WLSMV. The fiml-fmg paper's `analysis/parity_calibration.R` is the
  prototype to port and generalize. (2) Audit every existing "known divergence" /
  "believed correct" / convention exemption in the parity suite and in
  [oracle-defects.md](../validation/oracle-defects.md): each must carry a
  calibration proof in the target regime or convert to a tracked bug. This raises
  the oracle-defects standard of proof.
- **M — grow CI past its first lanes.** `.github/workflows/ci.yml` runs the
  layering check, vendor sync plus two R testthat files, a GCC 13 library
  build, and a clang 19 (apt.llvm.org) Debug build with the full `ctest` on
  ubuntu-24.04. Runs from 2026-06-02 on all failed: stock ubuntu-24.04 clang
  18 cannot compile libstdc++'s `<expected>`, and two corpus golden checks
  required the gitignored corpus mount. The workflow now takes clang 19 and
  those checks skip without the mount. Remaining: `R CMD check` in place of the two hand-picked testthat
  files, sanitizer validation on main or a schedule, heavy parity/optional
  optimizer lanes less often, and coverage as an artifact before considering
  badges. Avoid coverage-percentage gates until the report has been calibrated
  by real maintenance work.
- **S — export the two corpus-mount golden cases.** The at-θ implied-moment
  checks for `newsom_2015_ex9_3` and `little_2013_ch3_fig_3_6_1indicator`
  (`cpp/tests/golden/textbook_corpus_golden_test.cpp`) still read the optional
  `external/textbook-corpus` mount and skip without it. They need no data, only
  model syntax, options, and lavaan's θ and implied moments: add them to
  `cpp/tests/tools/regen_textbook_case_fixtures.R` from a machine with the corpus
  mounted and point the tests at the checked-in export.

## Simulation primitives

The `magmaan::sim` surface (NORTA / IG / Vale-Maurelli-Fleishman / PLSIM /
t-copula / Archimedean copula / C-vine / elliptical generators, marginal
families, and observed-variable projection) is summarized in the roadmap; its
work queue and decision log live in [`simulation.md`](simulation.md). Keep this
section as a high-level index for cross-domain planning — including the Johnson
SL exposure decision, the special-functions dependency policy, NORTA calibration
hardening, group-specific projection/population metadata, and model-implied
simulation lowering — and put detailed generator, marginal, and fixture
decisions in the simulation backlog.

## Model syntax

- **S — `sqrt` in defined parameters.** The grammar allows only the unary
  `exp` and `log` calls in `:=` expressions (`function_call` in
  `project/grammar/grammar.ebnf`). lavaan also takes `sqrt`, which the Mplus
  User's Guide twin models use (ex5.21 and ex5.22, e.g.
  `a := sqrt(2*(covmz - covdz))`); the corpus carries those rows. Extend the
  EBNF first, then the parser and the defined-parameter derivatives.

## Documentation

- **M, later — decide which research-note derivations become supplementary
  docs.** The derivation notes moved on 2026-09-24 to the private
  `private/research-notes` repository (outside this repository), alongside
  paper-track work such as the robust RMSEA material that may take a long time
  to publish, if ever. Derivations behind shipped behaviour (for example
  two-level ML, RMSEA asymptotics, two-stage weighting) are not esoteric and
  should be documented in magmaan, as supplementary derivation docs, well
  before any slow journal route. Per note: "level up and leanify" it into a
  clear, self-contained derivation checked against the implementation, then
  promote it; keep paper-track ideas private until their paper decides. Nothing
  blocks on this.

## API and R boundary

### Decompose `EstimatorSpec` into its actual axes — NOT STARTED

The current "estimator" surface mixes several axes in one string. The
development tiers require covariance domain and penalty to be explicit alongside
discrepancy, moments, algorithm and post-fit inference. C++ emits
multiple `fit$estimator` labels (`ML ULS GLS WLS DWLS FIML ML2S
ML-Fisher ML-Fisher-SNLLS ML-IRLS ML-IRLS-SNLLS ULS-SNLLS GLS-SNLLS WLS-SNLLS
SAM FCSEM-ML RBM-*×5 noniterative*×3`) while `api::EstimatorKind` has
**7** entries. The gap is a cross-product:

1. **Discrepancy** — what F is: `MomentQuadratic(W)`, normal likelihood,
   pattern likelihood (FIML), two-level likelihood, catML, FC-SEM ML. ~5 things.
2. **Moments** — what F eats: complete sample stats / ordinal (polychoric) /
   EM-saturated / pairwise / raw. This is why `FIML` (= ML on raw),
   `ML2S` (= any estimator on EM-completed moments) and `GLSpw` (= GLS on
   pairwise stats) look like estimators. **FIML genuinely earns estimator
   status** — per-pattern likelihood really is a different F. ML2S and GLSpw do
   not: same F, different moments. The tell is that R has to synthesize
   cross-product label strings (`ML2S_DWLS`, `ML2S_ADF`, `ML2S_DLS`).
3. **Covariance domain** — ordinary or PSD-constrained. PSD lifting implements
   the domain restriction; it does not define a separate estimator family.
4. **Penalty** — none or a specified multi-information barrier, including its
   weight and applicability contract. This changes the objective.
5. **Algorithm** — direct gradient, Fisher scoring, IRLS, SNLLS
   (Golub-Pereyra), lifted optimization, ridge continuation, two-stage EM. Currently
   encoded *in the function name* (`fit_ml_irls_snlls`) and re-encoded in the
   label string.
6. **Post-fit correction** — MLM/MLR/SB. Already correctly excluded; R
   hard-errors on `estimator = "MLM"`. This axis is the part that is right.

So `ML-IRLS-SNLLS` is axis 1 × axis 5, and `FIML` is axis 1 × axis 2, and
nothing in the type system says so.

**Why this is not a C++-only change.** The R side is where the drift actually
bites: five independent `estimator=` validation schemes, two disagreeing
allow-lists (`model_data.R:2022` has `ML2S`, `prepared.R:162` does not),
`evaluate_at()` defaulting to **ULS** while `fit_model()` defaults to **ML**, and
the label stored twice (`fit$estimator` vs `fit$options$estimator`) with
downstream readers picking inconsistently — some read one, some read `%||%`
both. Continuous `DWLS` is reachable via `estimate()` but rejected by every
continuous-LS post-fit gate. Doing the C++ half alone would leave all of that.

**Also collapse the six overlapping estimator enums**: `api::EstimatorKind`,
`estimate::Estimator`, `estimate::OrdinalWeightKind`, `data::OrdinalEstimatorKind`
(the same three values as the previous one, in a *different declaration
order* — a silent bug surface if either is ever cast), `robust::frontier::
Discrepancy`, `fiml::TwoStageWeight`. Likewise `estimate::OrdinalParameterization`
duplicates `data::OrdinalMomentParameterization`. And add `estimator_from_string`
/ `estimator_name` mirroring the existing `estimate/backend_strings.hpp`, so R
stops hand-rolling the mapping in three places.

**Adjacent cleanups this would enable.** `api::OptimizerKind` is a strict
5-of-11 subset of `estimate::Backend`, so PORT — the nlminb-parity backend,
default-ON — is unreachable through the staged API. `api::Fit` hard-codes
`fiml_pack_` / `fiml_h1_` members on the generic class, so a new estimator with
cross-call state has no slot. The main `switch (estimator.kind)` at
`cpp/src/api/sem.cpp:875` lists `FIML/DWLS/TwoLevelML` as empty `break` cases
relying on unreachability comments, so a new enumerator falls through with
`est` uninitialized rather than failing to compile.

**Groundwork already done.** The weight is now a structured type
(`gmm::BlockWeight`), so `DWLS` is `diagonal(Γ)` rather than a name, and
`EstimatorSpec::ordinal_moments` is explicit rather than inferred from
`weight.empty()` — the first slice of the `moments` axis. See the
continuous-whitening entry above.

- **S — document the trace-form robust-test path so it is findable.** SB,
  mean-variance-adjusted and scaled-shifted need only `df`, `tr(M)` and
  `tr(M²)`, and magmaan has had that path for a long time:
  `weighted_chisq_moments_from_M` in core, exposed to R as
  `magmaan_core$robust_test_moments_both_breads_{zc,gamma}`, with the closed-form
  scaling applied in R (the `sb_from_moments` helper copied across experiments
  07/15/16/19). It is exact, not an approximation: on complete-data continuous
  ML it reproduces `fmg_tests(tests = "sb_ml")` to 7e-16..1e-14
  (`experiments/_archive/score-vs-lrt/diagnose_trace_sb_nested.R`), because the
  reduced `M = BᵀΓ̂B` is exactly `df × df` so `Σλ = tr(M) = tr(UΓ̂)`.
  The problem is purely discoverability. Today the only pointer is a three-line
  comment in `r-package/R/zzz_core.R:266-268` referring to "the Maydeu
  experiment", nothing in `r-package/README.md` or the roadmap mentions it, and
  the obvious-looking entry points do not lead there: `fmg_tests()` always
  eigensolves (`r-package/R/fmg.R:207` calls `infer_fmg_ugamma_spectra()` before
  inspecting which method was requested, so even `tests = "std_ml"` pays it),
  and `magmaan_core$robust_satorra_bentler` takes `eigvals`, not moments. Two
  separate audit passes re-derived "magmaan eigensolves where lavaan traces"
  from scratch before noticing the trace path already existed. Fix the docs
  first: a named section in the R README plus a roadmap pointer, and cross-refer
  from `fmg_tests()`'s help. Two real caveats to record with it. (1) The FMG
  path truncates negative eigenvalues before averaging (`cpp/src/robust/fmg.cpp:165-172`),
  so its SB scale is `mean(max(λ,0))` while the trace is untruncated `tr(M)/df`;
  the two must disagree exactly when `n_truncated > 0` (the rank-deficient
  `p=20, N=100` case). (2) Only complete-data continuous ML has the `Zc` route;
  FIML already scales from its own trace inside the estimator.
  Whether to also give `fmg_tests()` a method-aware branch that skips the
  eigensolve for trace-only requests is a *separate*, lower-value decision: it
  would have to return `NULL` in the `eigenvalues` column that
  `.fmg_rows_to_df` currently publishes (`r-package/R/fmg.R:252`), and it buys
  nothing whenever a pEBA/pOLS test is requested alongside SB, which is the
  usual batched case. Do not bundle the two.
- **S/M.** Add or rename R wrappers only when the methods-developer workflow
  exposes a concrete gap in the staged API; the current `magmaan_core`,
  `magmaan_fit`, and post-fit wrapper surface is otherwise sufficient for the
  next R exploration pass.
- **S/M, experiment-motivated.** Decide whether the Deng-Chan reliability-
  difference test (`experiments/research/active/03-deng-chan-2017-alpha-omega`) earns a home in
  core. The experiment already runs against the current surface (ML fits plus
  `infer_gamma_nt`/`infer_empirical_gamma`; Cronbach's alpha falls out as the
  omega of a ULS tau-equivalent fit) and diagnoses a genuine non-regularity:
  because `omega ≥ alpha` with equality only at equal loadings, `omega − alpha`
  is a second-order (1/N) statistic with a weighted-chi-square null, so a naive
  Wald z under-rejects and the fix references `2n(omega − alpha)` against its
  Imhof tail. If promoted, the natural shape is a `measures::frontier`
  reliability module (alpha, omega, the joint `omega − alpha` SE, and its
  second-order Imhof calibration) plus a thin R wrapper. First slice landed
  2026-06-26: `measures::frontier::reliability` now provides covariance-only
  alpha, Guttman's lambda6, and Spearman-Guttman covariance omega with
  delta-method SEs, plus the exploratory R primitive
  `magmaan_core$measures_reliability_cov` and
  `experiments/research/banked/16-reliability-lambda6`. The non-regular joint
  `omega - alpha` test and Imhof calibration remain unpromoted. Not required.
  Prior-art oracle for the Spearman-Guttman covariance omega is Hancock & An
  (2020) (closed-form single-factor omega; see
  the private paper eval `2020-hancock-closed-form-omega.md`): their
  Spearman-1927 ratio-of-sums loading aggregation is more numerically stable than
  the average-of-ratios communality and is the parity target for exp research/16. Their
  256-cell sim is the validation oracle. Open lane beyond Hancock & An (single
  factor only): the multi-factor / multi-group / weighted `ω_G(σ;w)` form and
  omega-hierarchical via a second-stage Schmid-Leiman centroid on `Φ_G` (k>=3),
  derived in the `guttman_cfa_asymptotics.tex` note.
- **Ordinal workspace and invariance follow-ups.** The workspace split,
  threshold maps, all-ordinal/mixed SNLLS, keyword theta invariance and R
  nested-test wrappers are implemented and documented in the
  [roadmap](../architecture/roadmap.md#ordinal-and-mixed-categorical-ls) and
  [workspace contract](../design/ordinal-snlls-gamma-architecture.md). Remaining:
  - Add R/API polish only when a concrete caller needs it. Reduced-Gamma
    robust products remain speculative with a size-driven build-if trigger.
  - Extend the ordinal release to mixed models; the current keyword release
    is all-ordinal only.
  - Decide the supported mixed-pairwise input contract before extending
    `mplus_wlsmv_invariance()`, which currently rejects those inputs.
  - **S — review `continuous_invariance()`'s explicit mean-syntax insertion**
    now that `spec::build` supplies the release. Preserve explicit user mean
    specifications and the metric-to-scalar delta restriction map.
  Paper-specific reruns belong to the paper's own planning, not this library
  backlog. Existing C++/R fixtures and the deliberate theta versus released-
  delta validation boundary remain recorded in the roadmap.
- **M/L.** Optional h-weighted polyserial path: a polyserial-only h-weighted
  moment builder — continuous-ordinal h objective, casewise threshold/rho
  estimating functions, bread/influence/Gamma construction, and splicing into the
  mixed moment stack so `NACOV`/`W_dwls`/`W_wls` rebuild. The all-ordinal h-score
  variants already have the generic Gamma machinery; the missing piece is the
  mixed polyserial estimating-equation design.
- **FIML FMG follow-ups.** The landed FMG R wiring (single-model, FIML/missing,
  and the nested restriction-map route; see roadmap) leaves: multi-group FIML
  fits need explicit start/convergence care, and pairwise-data FMG remains
  deferred. Nonlinear equality tangent-space support for the single-model FIML
  UGamma spectrum and nested FIML restriction-map route landed 2026-06. The
  high-level `fit_model(estimator = "FIML")` path now auto-enables mean structure
  for syntax-backed models and rejects explicit `meanstructure = FALSE`.
  Saturated-EM reuse extended to the FIML nested path (2026-06):
  `lr_test_satorra2000/2001_fiml_from_data` take an optional `sm_precomputed` and
  `infer_fiml_lr_test_satorra2000` reuses a fit's `$stage1`, and
  `fit_ml2s(stage1=)` skips the rung-independent Stage-1 EM; this is what
  `experiments/research/active/06-fiml-invariance-fmg-power` uses to build one saturated EM per
  masked dataset and thread it through both estimators, all four ladder rungs,
  every FMG battery, and every nested test (kills the ~72s-at-p=30 FIML-nested
  rebuild and the 4x ML2S Stage-1 redundancy; bit-identical, verified). Deferred
  residual: `fit_fiml` still recomputes its cheap mu/Sigma-only `fiml_h1_moments`
  per rung; eliminating it needs a `fit_fiml` h1-injection arg + a FIMLH1-from-R
  reconstructor (the structured optimization dominates, so it is low priority).
- **Measurement-invariance nested run (exp research/06) gaps.** Two items block the full
  Brace-Savalei-style invariance Type-I run; the overnight run uses the p=6
  one-factor model with the weak + strict nested steps only.
  (1) **metric->scalar ("strong") nested test is not nestable**: the scalar model
  ties intercepts AND frees the group-2 latent mean, so `npar(scalar) = 37 >
  npar(metric) = 36` and the exact Satorra-2000 route
  (`restriction_alpha_from_K`) rejects `K_H1.npar != K_H0.npar`. Needs a
  mean-structure-aware nested path (reference both models to the saturated model
  the way lavaan does), the same item the pEBA-nested paper reviewers raise.
  (2) **Brace-Savalei multi-factor populations** (two-factor, p in {8,16,30}) are
  not implemented; `build_population()` errors for p != 6. The p=30 cell is
  pEBA's home turf and the high-dimensional head-to-head; it is the build-out.
  Note: pEBA-on-the-difference IS already exposed (`fmg_nested()`, FIML + ML2S),
  so the spectrum nested battery is harvested now; only the two items above
  remain.
- **Done 2026-06-24 — structured-h1 FIML FMG ditched.**
  `estimate::FIMLH1Information::Structured` (the FIML FMG U·Γ-spectrum knob that
  evaluated the U-side information `V` at the model-implied moments `ξ(θ̂)`
  instead of the saturated `ξ̃`) is removed: the `FIMLH1Information` enum, the
  `h1_information` parameter on the `fiml_ugamma_spectrum` overloads, the
  `FIMLUGammaSpectrum::h1_information` field, the `infer_fiml_fmg_spectrum` Rcpp
  arg, and the R `fmg_tests(h1_information=)` / `fmg_pvalues()` argument. `V` is
  now always the saturated observed H1 information — the FMG-spectrum convention,
  PD by second-order optimality. Rationale: structured-h1 was *not*
  asymptotically advantageous (both consistent under H₀; Satorra-Bentler varies
  only `U⁰`, and Xia et al. 2016 via Savalei & Rosseel 2022 mildly prefer the
  saturated/unstructured information), and the model-implied curvature is not
  guaranteed PD off H₀ with nothing guarding the sign of the retained spectrum
  fed to the FMG/pEBA mixture (which assumes `λᵢ ≥ 0`). No new sim was needed:
  experiment research/04 already carried both variants across its grid, so the
  keep-or-ditch evaluation was in hand. Two things survive the removal: the
  `fiml_structured_h1_information` helper stays (the ML2S Stage-2 `W*` uses it),
  and magmaan's MLR/Yuan-Bentler reproduction is the independent
  `fiml_robust_mlr` (`estimate_fiml_robust_mlr`), which never used this knob.
  Experiment research/04's structured columns were dropped from the harness and the
  variant is marked legacy in its report; `examples/fmg.R` and the
  `fiml_ugamma_spectrum` C++ test no longer exercise it.
- **Ordinal/polychoric FMG (`papers/ordinal-fmg/` Paper 2).** Core gate **landed
  2026-06-13** (commit 9989c9d): `fmg_tests_ordinal()` / `fmg_tests_mixed_ordinal()`
  apply the FMG eigenvalue-tail transforms to the `robust_ordinal()` /
  `robust_mixed_ordinal()` polychoric UGamma spectrum (`eigvals` + `chisq_standard`
  + `df`), single- and multi-group, `_ml`/`_ug` rejected, anchored by the ordinal
  C++ FMG test and `r-package/examples/fmg_ordinal.R` (single-group plus
  two-group all-ordinal/mixed SB parity); no new C++ production code (see
  roadmap). The nested-test gate **also landed 2026-06-13**:
  all-ordinal DWLS/WLS `nestedTest(..., data = ordinal_stats,
  method = "satorra.2000")` now builds the direct ordinal Satorra-2000 reduced
  spectrum for exact or delta restrictions, including two-group
  configural-vs-metric invariance, with lavaan WLSMV parity in
  `r-package/examples/nested_test_ordinal.R`. What Paper 2 still lacks, to
  write the paper:
  - **Scaffold + reframed design landed 2026-06-13.** The paper-local harness
    exists at `papers/ordinal-fmg/` (its own nested git repo, gitignored by the
    outer repo), mirroring the `papers/fiml-fmg/` scaffold. **Thesis correction:**
    the paper is NOT about robustness to non-normality. For ordinal DWLS/ULS the
    fit statistic is weighted-chi-square (`T → Σ λ_j χ²₁`, `λ_j ≠ 1`) *even at the
    correct normal model*, because the diagonal weight is not `Γ⁻¹` — so the
    correction is on-model, and a Gaussian-copula non-normal generator
    (Vale-Maurelli, NORTA) is invisible after thresholding (collapses to normal
    theory). The harness therefore generates **normal** underlying data and varies
    the spectrum-shaping axes: categories {2,3,5} × model size p {6,12} × N. It
    carries the single-group GOF arm (omitted residual covariance) and the
    two-group nested arm (configural-vs-metric via `nestedTest(satorra.2000)`),
    comparing naive / SB / WLSMV mean-variance / WLSMV scaled-shifted / FMG
    (pEBA, pOLS). Smoke-verified: pipeline runs, lavaan WLSMV parity holds (UGamma
    spectrum ~3.8e-9, DWLS chi-square ~1e-6 after the documented `(N-G)/N`
    rescale). The `sim_vm_*` core exposure stays (useful for the continuous
    papers) but is unused here. Remaining (author, in the paper repo's
    `dev/todo.md`): the full `just parity` run, and the live empirical question —
    do the FMG transforms beat the WLSMV mean-variance adjustment anywhere (only
    possible where the spectrum spreads: binary items, larger models), or is
    WLSMV hard to beat for ordinal?
  - **Chen-style pairwise-missing scalar probe (experiment research/11, 2026-06).**
    Full-spectrum p-values do not repair the WLSMV_PD inflation once the
    pairwise-missing ordinal summary statistics are already distorted. In a
    Mplus-free 10-indicator invariant scalar test (`N=1000`, symmetric
    thresholds, overlap Gamma, 200 reps), scaled-shifted rejects 2.5% / 11.0% /
    31.0% at 0% / 30% / 50% missing; `all`/mixture is only slightly lower
    (2.5% / 10.5% / 30.0%); pEBA4 is more liberal (4.0% / 12.5% / 35.0%).
    Treat this as evidence that the Chen WLSMV_PD failure is not primarily a
    low-moment tail approximation problem; the missing-data moment/Gamma
    construction is upstream of the FMG transform.
    **MCAR vs MAR decomposition (added 2026-06-19, `--missing-mechanism`).**
    Re-running the same design under MCAR isolates the cause: every method holds
    nominal Type-I even at 50% missing (scaled-shifted 5.5% / 3.5% at 30% / 50%
    MCAR; whole battery .035-.050), while MAR inflates to ~.30-.35. The
    difference-statistic center is flat across MCAR (37.3 -> 37.4) and climbs
    across MAR (37.6 / 41.4 / 50.2) with the scale factor unchanged (~0.98), so
    the inflation is a bias-driven non-centrality from pairwise deletion being
    inconsistent under MAR, not a calibration/tail-approximation defect. Confirms
    the failure is the missing-data technique (PD), not the WLSMV estimator or
    the p-value family; the fix is full information or multiple imputation.
  - **Done 2026-06-13.** Direct ordinal UGamma-spectrum oracle: the paper
    parity pipeline now emits an explicit `ordinal_wlsmv_ugamma_spectrum_maxabs`
    row comparing magmaan's public DWLS + `robust_ordinal()` eigenvalues against
    lavaan WLSMV `lavInspect(., "UGamma")`. The core C++ fixture gate already
    compares `robust_ordinal().eigvals` to lavaan's stored UGamma eigenvalues in
    `cpp/tests/golden/ordinal_golden_test.cpp`; the paper row makes that provenance
    visible alongside the FIML parity table.
  - **Decided 2026-06-13.** Separate `papers/ordinal-fmg/` folder (not a second
    part of `papers/fiml-fmg/`).
  - **Done 2026-06-13.** Multi-group ordinal/mixed GOF FMG is first-class at the
    R boundary: the wrappers reuse the multi-block robust ordinal sandwich and
    `r-package/examples/fmg_ordinal.R` checks two-group all-ordinal and mixed
    SB parity against `robust_ordinal()` / `robust_mixed_ordinal()`.
  - **Only-when-needed.** A C++ methods-developer convenience entry point is not
    needed for the paper, since the spectrum-once-then-loop R orchestration
    avoids a per-method `robust_ordinal` recompute. (`_ug`/unbiased-Gamma is N/A
    for polychoric: the NACOV is already the asymptotic Gamma.)
## Benchmarks

Advisory local tooling, not a substitute for parity fixtures. Full design:
[project/validation/benchmark_plan.md](../validation/benchmark_plan.md). The imhof
integrator is now QUADPACK-backed (~3x faster, ~2e-16 parity; roadmap), which was
the dominant cost of every FMG/pEBA/pOLS p-value. Deferred unbiased-spectrum perf
work lives in [`speculative.md`](speculative.md). Open work:

- **S.** Keep the build-loop timings table in
  [project/architecture/roadmap.md](../architecture/roadmap.md) current after major
  workflow changes.
- **S. Two fixture fields are written but read by nothing** — found while bumping
  the oracle pin to lavaan 0.7-2, because both changed and *no test noticed*:

  - `cpp/tests/fixtures/twolevel/*.json` → `sampstat.{within_cov,between_cov,between_mean}`.
    Read by nothing in C++. **Correcting an earlier version of this entry: these
    are NOT sufficient statistics and are not worth wiring up.** The name is
    literal — `lavInspect(fit, "sampstat")` on a two-level fit returns lavaan's
    `S.PW.start` / `S.B.start`, i.e. *starting values* for the iteration. The
    two-level ML likelihood is evaluated per cluster on the raw data; there is no
    within/between covariance pair that it consumes. magmaan's actual two-level
    sufficient statistics (SSW, grand mean, per-size cluster-mean sums) *are*
    lavaan-gated, in `cluster_stats_test.cpp`.

    What the shift actually propagated into is worth knowing, though, and is
    recorded here because it bounds how tightly the two-level goldens can ever be
    held. Across the 0.7-1 → 0.7-2 bump the **model** side is unmoved — `fmin`
    identical exactly, model `logl` to 1e-12, `est` identical on `twolevel_1f4`,
    `se` to 1.6e-9 — while the **saturated (h1) log-likelihood** moved 1.0e-4
    (1f4), 3.8e-4 (2f6), 8.3e-5 (ri3). χ² = 2·(ll_h1 − ll_model) then inherits
    exactly twice that: 2.07e-4 and 7.53e-4, which is 2× the h1 shifts to three
    figures. Cause: the unstructured two-level model has no closed form, so
    lavaan fits it iteratively and converges it only to ~1e-4; different starting
    values land it in a slightly different place.

    Consequence: magmaan's two-level χ² is gated against an oracle number lavaan
    itself only pins to ~1e-4, and `twolevel_golden_test`'s `epsilon(1e-4)` on a
    χ² of 6.37 allows ~6.4e-4 — about 3× the observed oracle noise. That
    tolerance is therefore set by the *oracle's* convergence slop, not by
    magmaan's precision, and tightening it would make the test fail on lavaan's
    own irreproducibility. Do not tighten it without an independent reference for
    the saturated two-level likelihood.
  - `cpp/tests/fixtures/fiml/*.json` → `mlr_*_robust` / `ml2s_*_robust`. 0.7-2 now
    returns NA for robust CFI/TLI on `0001_one_factor_hs_fiml` (a perfect-fit
    model where the correction is degenerate; it was 0.999999999997366 before),
    and `fit/0015_start_call` gained a `se_robust_huberwhite` where 0.7-1 gave
    `null`. Both invisible to the suite.

  Neither is a magmaan bug. The point is that a fixture field nothing asserts on
  is not parity coverage, and the pin bump is what exposed it.
- **S.** `measures::fit_measures` costs ~3.2 ms at p=96 (about 10% of the whole
  fit) and is both non-monotone in p and sensitive to n (2.96× over n
  200→50000), which is impossible for a pure function of three scalars plus
  `samp`. Cause confirmed by reading: the RMSEA confidence interval runs
  `bisect_zero` (`cpp/src/measures/fit_measures.cpp:53`) over
  `noncentral_chisq_cdf` for each bound, so the iteration count tracks the
  chi-square value rather than the dimension. **Deliberately not fixed here.**
  RMSEA CI bounds are lavaan-gated, so any change to the root-finder tolerance,
  its iteration cap, or the noncentral-chi-square algorithm has to be gated
  against the lavaan RMSEA fixtures first; a speedup that shifts the third
  decimal of a published CI is a regression, not a win. Pick this up only with
  the parity fixtures in hand.
- **M. Public speed report and lavaan cost attribution.** The first experiment
  implements matched raw/prepared/post-fit ML workloads on HS CFA and
  PoliticalDemocracy, one shared R batch timer, fresh serial sessions, and
  output/adapter gates. The full
  [design](../validation/benchmark_plan.md#proposed-public-speed-report-2026-09-20)
  remains the publication target. The pilot is experiment showcases/08 (benchmark,
  active), indexed in the experiment collection; it is not a public speed
  claim. Native lavaan 0.7-2 `peba4_ml` and magmaan disagree in the HS tail
  even with identical statistic/eigenvalues (about 7% relative p-value
  difference). Follow-up: the cause is lavaan's default `1e-6` absolute Imhof
  tolerance on a `1.53e-7` tail. Tightening to `1e-13` agrees with magmaan
  and an independent 70-digit Erlang-chain calculation. This is numerical
  tail accuracy, not an information-matrix convention or method mismatch.
  Time an accuracy-matched route before accepting those ratios; do not
  retroactively substitute a corrected p-value into the default timing rows.
  SB and covariance checks are separate. Next: matched native
  evaluator replay/common driver, reliable exclusive stage attribution,
  broader cases and independent reproduction. Pinned SNLLS bundles and old
  runners remain untouched until their replacements cover their consumers.
- **S/M.** Retire the hand-rolled timing loops now that
  `benchmarks/timing/timing.hpp` (C++) and `benchmarks/r/timing.R` (R,
  `time_paired()`: batch auto-calibration, arm rotation, median reporting) both
  exist. `experiments/showcases/02-lavaan-speed-bench` and
  `experiments/showcases/06-speed-attribution` are migrated to
  `benchmarks/r/timing.R`. Still hand-rolled: `benchmarks/score_primitives.R`
  and `benchmarks/inference_reuse.R` (mean-of-15 over `proc.time()`/`system.time()`,
  whose ~0.5 ms quantisation is a quarter of the ~2 ms workloads they time),
  `benchmarks/r/bench_mi_lrt.R` (3 reps, no warmup), `benchmarks/r/run_benchmark.R`
  (`bench::mark`, whole-fit only), `experiments/_archive/psd-ml-timing`
  (already batch-calibrated with `--repeats/--warmups`, just not on the shared
  helper), and `experiments/research/banked/08-ordinal-stage2-pairwise`. Convert the
  C++ benches to include `timing/timing.hpp` first — that is mechanical. Do
  not add another hand-rolled copy; migrate to `benchmarks/r/timing.R` instead.
- **M.** Track objective value, gradient norm, iteration count, wall time, and
  agreement with lavaan-backed estimates where applicable.
- **S/M.** Continue extending benchmark coverage beyond the current
  lavaan-backed complete-data ML, controlled-missingness FIML, and continuous
  ULS/GLS smoke cases to WLS, ordinal DWLS/WLS, and mixed categorical models.
- **S/M.** Remaining ordinal SNLLS / Gamma workspace benchmark polish: lazy
  mixed WLS and mixed theta rows for the estimators landed 2026-06, and any
  R/API wrapper polish justified by the paper results. (All-ordinal delta
  fit-only, fit-plus-inference cache reuse, threshold-constraint,
  construction-boundary, raw-to-SNLLS legacy/lazy, delta/theta, mixed-DWLS
  lazy, two-group invariance, naive corr-block-WLS rows, and experimental
  `OrdinalStats` stage-2 weight reuse helpers for ULS/DWLS/WLS/NT/DLS have landed
  across `experiments/_archive/06/10/11/12/13` and the benchmark itself; the
  literature-grade `q ≤ 12` grid now runs from `papers/ordinal-snlls/`.)
- **Landed.** Two-stage EM / saturated-covariance missing-data path. Stage 1
  (`estimate::fiml::saturated_em_moments` / `estimate_saturated_em_moments`) and
  Stage 2 (`estimate_two_stage_em(partable, raw_data, kind = c("ml","gls"))`)
  have landed and feed the MSE comparator in
  `experiments/research/active/01-pairwise-gls-efficiency/`. The packaged ML2S path has also
  landed: `fit_ml2s()` / `fit_model(..., estimator = "ML2S")` run Stage-2 ML on
  the saturated EM moments and attach Savalei-Bentler-style corrected SEs plus
  scaled chi-square from the Stage-1 `(H, J, ACOV)` ingredients. The C++ post-fit
  layer also exposes lavaan's `robust.two.stage` scaled/robust CFI/TLI/RMSEA
  family, including baseline scaling and RMSEA interval/p-value variants. The
  GLS branch remains an explicit research comparator, not a named high-level
  estimator.
  `fmg_tests()` now accepts an ML2S fit: the eigenvalue-tail family (SS/all/pall/
  pEBA/pOLS, plus the SB/SS/SF/MV low-moment matches, where MV = `mv` =
  Satterthwaite mean.var.adjusted, a new `FmgMethod` matching lavaan to ~1e-12)
  is applied to the df-dim two-stage UGamma spectrum + Stage-2 ML base on
  `fit$ml2s`. The two-stage scaling and SEs match lavaan's
  `missing="robust.two.stage"` (sandwich ACOV) convention to machine precision
  (≲1e-4 across the exp-research/05 grid, typically ~1e-7; EM/optimizer-limited), and the
  robust/scaled global indices are lavaan-gated to fixture tolerances. This is
  not the plain `missing="two.stage"`
  (normal-theory ACOV) scaling, which collapses under non-normality; the base
  matches both. `trace(UGamma) = E[T]` (normal-data ncp ~ 0) is an independent
  first-principles check. Calibration study + lavaan parity oracle:
  `experiments/research/active/05-fiml-twostage-fmg-chisq`; unit gate:
  `two_stage_em_ml_inference` self-consistency in `cpp/tests/unit/fiml_test.cpp` and
  the `ml2s_*` rows of `cpp/tests/golden/fiml_golden_test.cpp`.
  - **Done 2026-06-28 — lavaan-like H1 edge behavior.** The saturated H1 EM now
    keeps the EM path but treats near-singular covariance updates and iteration
    caps as diagnostic conditions: covariance updates are diagonally floored,
    cap hits return the last iterate, and `FIMLH1`/`SaturatedMoments` carry
    warnings. The saturated information matrix used for Stage-1 ACOV is
    symmetrized and floored only when needed before inversion, preventing
    high-dimensional sparse-missing p30 cases from aborting on a singular H1
    information matrix. The non-routine lavaan comparison/stress harness lives
    in `cpp/tests/checks/fiml_h1_edge`.
  - **Structured/unstructured weight axis carries to ML2S (resolved 2026-06-17).**
    The Satorra-Bentler U-metric weight choice - `WeightMoments::Structured`
    (model-implied Σ̂(θ̂)) vs `Unstructured` (sample/saturated h1, =
    `h1.information="unstructured"`) - applies to the two-stage path just as it
    does to FIML/robust.sem. ML2S now uses **Unstructured** for both the test
    spectrum and the SEs (`two_stage_em_ml_inference` in `cpp/src/estimate/fiml.cpp`),
    matching lavaan `robust.two.stage` (which hard-forces unstructured) and
    magmaan's own FIML FMG convention. It was briefly `Structured`, which left a
    1-3% trace/SE gap that grew with non-normality. Note: on complete data this is
    `robust.two.stage`, NOT `robust.sem`/MLM - those differ on the same axis when
    Σ̂ ≠ Σ(θ̂).
  - **M — verify the Savalei–Falk (2014) observed-information configurations.**
    Their simulation did not use today's MLR trace-difference or lavaan
    `robust.two.stage` defaults. Build independent equation-level references for
    the paper's analytic observed structured-H0 FIML correction and its
    observed-information two-stage correction; gate the intermediate A/B/Omega,
    derivative, residual-projector, trace, scale, statistic, and df before any
    Monte Carlo comparison. Add explicitly named paper-era post-fit routes only
    after those same-data gates pass, preserving current MLR, FIML FMG, and ML2S
    semantics. `experiments/replications/08-savalei-falk-2014-test-map` records the source map,
    deterministic routing witness, and staged replication plan. Experiment replications/09
    now supplies the exhaustive diagnostic surface: 48 direct-FIML and 32 ML2S
    matrix-estimation combinations are evaluated on identical generated
    samples. Its targeted run identifies the two-stage candidate (11.5% versus
    the published 10.0%) but exposes a direct-FIML conflict: the literal
    all-structured observed row rejects only 36.5% among 935 usable tests, while
    the closest saturated-expected row rejects 61.9% versus the published 63.3%.
    Before Experiment research/44 consumes a direct-FIML route, compare the analytic
    structured observed-H1 matrix against an independent EQS/symbolic oracle and
    determine whether EQS applied `SE=EXACT` to the robust test or only its
    standard errors. The ML2S candidate can proceed through the remaining
    equation-level gates above without waiting for that FIML resolution.
- **Landed; remainder in speculative.** The Van-Praag pairwise covariance
  machinery (`data::pairwise_sample_stats`,
  `robust::pairwise_casewise_contributions`, `data::gamma_nt_pairwise`,
  `estimate::fit_gls_pairwise`, the inference-side `WeightMoments::Pairwise` bread
  plus `robust::reduced_gamma_nt_pairwise` meat, and the matching R surface) has
  landed for `papers/pairwise-robust-sem/` and `experiments/08`/`09`. The
  remaining pairwise μ ACOV and pairwise Browne-unbiased items live in
  [`speculative.md`](speculative.md).
- **Closed 2026-09-30.** The covariance-continuation study is archived at
  `experiments/_archive/near-singular-ml-continuation/`: its tested target/profile
  grid added cost and did not improve convergence. No further run is queued;
  reconsider only for a concrete failure under the current starts and verdict.

- **Done 2026-06-15.** All three Geiser GLS/ULS parity exceptions are closed
  (`cpp/tests/golden/geiser_golden_test.cpp`, regression note in the test ledger):
  (1) manifest fixed.x path models (`manifest_regression`, `manifest_path`,
  `manifest_path_non_saturated`) resolve exogenous observed moments from the
  sample before the implied-moment check and gate Σ/μ against lavaan; (2) the
  latent AR cross-lagged family is rescued by a multi-start recipe (`best_start`:
  take the lower-objective of a `simple_start_values` fit and an ML-warm-started
  fit), reaching lavaan's Σ/μ to ~1e-7, with the scalar objective gate relaxed to
  `max(2e-4, 2.5e-3·|fx|)` (GLS) / `max(5e-3, 5e-3·|fx|)` (ULS) to absorb the
  documented ~1/N mean-structure scale convention; (3) the two *manifest* fixed.x
  cross-lagged path models (`manifest_ar_cross_lagged`, `…_extended`) now gate
  Σ/μ against lavaan to ~1e-8. The earlier "worse global optimum" symptom was
  **not** a core propagation bug: it was an observed-order mismatch in the golden
  harness. magmaan orders observed variables `[ov.y, ov.x]` (classify) and
  `SampleStats` is name-free / positionally aligned to that `ov_order`, but the
  fixtures supply `sample_cov`/`sample_mean` and lavaan's Σ/μ in their own data
  column order. The two cross-lagged models are the only Geiser cases where the
  exogenous variables (`d11`, `c11`) are not already last in the data, so
  `resolve_fixed_x` read exogenous moments from the wrong sample positions and
  the objective compared mis-ordered matrices. The harness now reconciles by
  variable name (`perm_to_magmaan` over `rep.ov_names`), reordering the sample
  and the lavaan moments into magmaan's `ov_order` (identity for the other
  cases). Still open from before: the Geiser per-parameter GLS comparison surface
  is implied-moment-based, not θ̂/SE-keyed.
- **S/M.** Per-parameter θ̂/SE parity for the Kline/Guo measurement-invariance
  corpus. The order-free chisq/df parity is gated; per-parameter parity needs a
  lavaan→magmaan free-parameter-order map (the submodule oracle stores
  `theta`/`se` in lavaan's free-parameter order).
- **S — textbook corpus coverage.** Corpus v3.2.0 holds 398 cases from eight
  books, each verified against its source; the per-book evidence is in the
  corpus's `docs/audit/`. ESEM is out of scope by decision. Not yet ingested:
  - Little's second-edition Mplus material. 15 CH3 and 16 CH5 inputs repeat
    first-edition models, and 25 CH5 CarpThesis inputs are new data but the
    same model families. The 19 ch9 CLPM/RI-CLPM inputs have no `.out`; they
    need the second edition's printed tables.
  - Inputs waiting on source material: the Muthén (2017) inputs whose data
    are not distributed (`bengt.062911.dat`, Monte Carlo replication lists)
    and Little's ch9 Homcov/Omit runs, which need Table 9.2.
  - The Mplus two-level examples; see
    [Two-level](#two-level-multilevel-sem).
- **S/M.** Extend the Mplus SEM corpus beyond the strict growth tranche.
  `external/textbook-corpus/raw/mplus_sem` retains 26 examples verified against
  their Mplus `.out` (28 more lack a data file in the archives), and the tracked
  fixtures gate seven continuous growth cases across ML/ULS/GLS/WLS.
  Remaining: extend the translator to the out-of-scope Mplus features recorded
  in the corpus audit (WLSMV categorical growth, MODEL INDIRECT; ESEM is out
  of scope by decision), decide how to test
  observed-only path models without exercising the saturated observed-path abort,
  and add categorical fixtures only for models that match magmaan's ordinal/mixed
  LS surface rather than Mplus logistic/probit response models.
- **S/M.** Extend the Little/Newsom tracked fixtures. The builders verify 106
  of 108 Little LISREL inputs against LISREL's output and retain 101 Newsom
  first-edition fit calls that reproduce the author's script; the consolidated
  `magmaan_textbook_corpus_v1` manifest indexes these alongside Geiser and Mplus
  SEM, with an advisory overlap graph for future paper mining. Remaining: a
  multi-group fixture format for Little's 21 multi-group cases and a split or
  compact format for the 37 wider or bounded verified cases above the 1 MB
  file limit. The categorical cases have their own lane
  (`textbook_ordinal/`; see
  [Categorical models](#categorical-models-gaps-found-on-the-textbook-corpus)).
- **S/M.** Promote the remaining first paper-corpus seed and broaden the
  paper-corpus fixture surface. `external/paper-corpus` owns scouting, minimal
  derived lavaan cases, validation, and magmaan JSON exports; magmaan consumes
  copied snapshots under `cpp/tests/fixtures/paper_corpus/`. `zxqvn` is promoted as a
  core complete-data ML point-estimate fixture. **Added 2026-09-24:** 14
  aggregate-only examples now gate ordinal DWLS fits and MI/EPC/equality
  releases, continuous/FIML fits, and a controlled missingness variant through
  the lavaan partable boundary. Fixed-row ordinal MI double counting is fixed.
  Raw categorical moment estimation and casewise robust FIML inference are not
  covered by this aggregate batch. Remaining: promote `hwkem`,
  document license/data-handling for that richer source, extract supported lavaan
  model/data pairs, classify RI-CLPM pieces outside the core parity surface, and
  decide whether clustered-SE handling should become a later paper-corpus
  inference fixture.
- **S/M.** Add a small OpenMx tutorial corpus as an offline second-oracle
  cross-check, not as a runtime input format. Start with the dormant
  `openmx_mimic` case in `benchmarks/r/cases.R` and the OpenMx RAM examples noted
  in `benchmark_zoo.md`; hand-translate each retained
  model to lavaan syntax, harvest golden values from OpenMx (`mxRun`,
  `omxGetParameters`, `model$output`, `mxGetExpected(., "covariance")`), and tag
  those fixtures with `_meta.tool = "OpenMx"`. Keep the curated tier small
  (roughly 3-6 lavaan-expressible SEM/CFA cases, including a mixed
  continuous/ordinal CFA if licensing and fixture shape are clear), assert
  magmaan's implied moments against OpenMx's implied covariance, and keep
  `mxModel`/`mxPath` parsing plus a runtime RAM frontend out of scope.
- **S/M.** Refine benchmark use of optimizer diagnostics now that fit results
  expose `optimizer_status` and final gradient norms. Benchmark scripts should
  distinguish clean convergence from line-search salvage or singular PORT
  convergence, and still avoid interpreting backend-specific missing iteration
  counts as real zero-iteration solves.
- **S/M.** SNLLS follow-up experiments after the contract repairs. The R
  bounds rejection, full-coordinate common audit, corrected conditioning
  rationale, direct inner normal-residual screen, and explicit Kaufman
  Jacobian contract are implemented and regression-gated. Still evaluate an
  exact residual derivative for PORT-NLS/Ceres and a cheaper scalar-only
  gradient as separate performance experiments; neither is a prerequisite
  correctness repair for the documented approximation. Condition/rank
  telemetry and broader corpus calibration remain useful before freezing
  new speed claims. Details and the bounded publication/handoff proposal:
  the SNLLS handoff review (`private/snlls-handoff/REVIEW.md`, outside this
  repository).
- **S/M.** Extend the frozen common-verdict SNLLS handoff to the larger paper
  grids. The handoff bundle (`private/snlls-handoff/handoff-current`, outside
  this repository) pins core `5f2ebe10`, including
  theta specialization and input guards, with fresh clean-source diagnostic
  comparisons and a manuscript evidence inventory. Full paper grids still need
  a rerun with balanced timing order and the common verdict; reconcile the
  continuous corpus's 280-versus-289 model counts and the Ernst replication
  provenance before carrying over manuscript headlines. Preserve the earlier
  frozen `b8bd95d9` bundle as its own historical snapshot.
- **S/M.** SNLLS audit follow-up: released-scale delta is now explicitly
  rejected by both all-ordinal SNLLS entry points, and the shared GP classifier
  rejects nonlinear equalities for every caller. New support needs a nonlinear
  standardized-moment profile, plus an audit of the adjacent cache-aware
  bounded delta path; the full-moment objective alone does not establish
  profiled-path support. Performance follow-ups include mixed theta threshold
  elimination, Fisher Schur factor reuse, and clearer ordinal/IRLS inner-solve
  telemetry. See `private/snlls-handoff/common-verdict-rerun-2026-09-13/`
  `IMPLEMENTATION-AUDIT.md` (outside this repository).
- **S/M.** Broaden theta threshold-profile eligibility only when needed.
  The independently free, unbounded-threshold/no-active-equality fast path is
  implemented, with ULS/DWLS direct reconstruction, a cached WLS QR Schur
  factor, full-model audits, generic-path equivalence and fallback regression
  checks. Fixed/shared thresholds, constrained models, and mixed moments retain
  the generic path. Expanded cases need reduced-gradient, reconstruction and
  common-verdict checks before timing claims. See
  `THETA-INVESTIGATION.md` and `THETA-FAST-PATH.md` in
  `private/snlls-handoff/common-verdict-rerun-2026-09-13/` (outside this
  repository) for the investigation and implementation results.
- **S/M.** Extend the paper-local SNLLS benchmark package in
  `papers/snlls-constrained/r-package/` with the remaining defensible real cases
  (especially a Geiser/Eid LST covariance input and a documented MTMM variant)
  plus one Boomsma-style simulation design. Keep the runner reporting setup time,
  fit time, whole time, iterations, objective values, and errors.
- **S/M.** For the SNLLS paper, add a narrow finite-difference-gradient mechanism
  probe rather than another broad simulation grid. NLopt's gradient-based
  algorithms (`LD_*`, including L-BFGS) expect caller-supplied gradients; the
  derivative-free `LN_*` algorithms (e.g. BOBYQA) are not the same as
  finite-differencing a BFGS gradient. If the paper needs an empirical
  Kreiberg-style explanation, add an explicit benchmark wrapper that
  finite-differences the existing scalar LS objective for ordinary LS and SNLLS
  under the same line-search optimizer, or report objective-evaluation accounting
  plus measured objective/Jacobian costs.
- **S.** In the SNLLS paper, spell out the implementation problem and solution
  more explicitly: the hard part is not proving separability but turning the
  profiled objective into residual/Jacobian calls an optimizer can trust. Explain
  why published tensor-gradient derivations are useful as a code blueprint and
  correctness check, while keeping the main text focused on the projection
  identity, the affine constraint split, and the fact that magmaan reuses the
  ordinary LISREL moment Jacobian instead of hand-writing pages of tensor
  products.

## Ordinal/SNLLS research

- **M/L.** Robust ordinal SEM paper track. Follow
  maintainer working note `robust_ordinal_sem_paper_plan.md`:
  build a paper-local simulation runner that emits tidy CSV for the Welz bivariate
  contamination design, the Welz/Foldnes-Gronneberg five-variable robust
  polychoric matrix design, an ordinal CFA downstream design, a Clayton copula
  stress test, and a small computation benchmark comparing ML, WMA hard cap,
  smooth h, and Huberized Pearson-residual moments. First milestone: reproduce the
  known Welz qualitative pattern for Designs 1-2 before adding SEM fits or broad
  copula grids.
  - **Robust-variant cleanup / evaluation harness (decide which to keep).** The
    full variant family is wired end-to-end (C++ core + R `data_ordinal_stats_-
    from_raw(robust=, h_kind=, clip=, ...)`), and `ros_method_specs()` now
    sweeps all 8: `ml`, `wma_hard_cap`, `smooth_cap`, `exp_cap`, `dpd`,
    `hard_huber`, `pseudo_huber`, `tukey_biweight`. Open question is which earn
    a main-text line vs. supplement vs. removal: we don't necessarily want all
    of them, but the call should come from a proper comparison, not a hunch.
    Build the evaluation setup before pruning: contamination sweep (bias/RMSE/SE
    calibration/convergence/Γ-conditioning/runtime) AND the copula
    distributional stress (Welz et al. evaluated on copulas too — Clayton main
    text, Gumbel/t supplement) so we separate "robust to tail contamination"
    from "robust to nonnormal copula." Prior signal: a limited pilot
    (`robust_ordinal_pilot_n1000_r5.csv`) had WMA hard cap doing best, and hard
    cap's φ is closed-form (the others — smooth/exp — carry Gauss-Legendre
    quadrature in `phi_from_h`). Hard cap alone is a sufficient paper; the
    cleanup decides whether the rest add enough to report.
    - **DONE 2026-06-24: copula distributional stress (Design 4).**
      `robust_ordinal_copula.R` runs the full 8-estimator
      family on a faithful Welz §8.2 replica (Clayton/Gumbel/Frank × rho_G
      {0.3,0.9}, N=1000, 5000 reps); our WMA hard cap reproduces robcat and ML
      reproduces the MLE to Table-5 precision. Findings in
      maintainer working note `robust_ordinal_copula_results.md`:
      WMA's robustness is *directional* (wins on Clayton, over-corrects on
      Gumbel rho=0.9 to worse-than-ML); the Huber/Tukey residual-clip family is
      dominated everywhere (Tukey SEs unusable); DPD is the only cross-copula-
      stable robust recipe. Still to run: the contamination sweep (Designs 1-2
      at scale) and the SEM-downstream design (Design 3).
- **M/L.** Ordinal SNLLS follow-up research. The all-ordinal delta ULS/DWLS/WLS
  path covers free, fixed, merged (including cross-group invariant), and
  general linearly constrained thresholds through both the threshold-profiled
  and full-threshold SNLLS routes; the all-ordinal theta path covers
  cache-aware bounded/SNLLS point estimation. Mixed continuous/ordinal delta
  and theta SNLLS exist for the materialized full-threshold DWLS/WLS path,
  mixed fit-only DWLS has a lazy workspace path, and mixed robust scaled tests
  now match lavaan at all-ordinal tightness. Use experiments to decide whether
  the next paper-facing C++ work should be lazy mixed WLS construction or
  reduced-Gamma inference plumbing.

## Composite models

The single-group ML slice has landed end to end: `<~` parsing (`Op::Composite`),
the historical Henseler-Ogasawara expansion and the native FC-SEM spec/evaluator
path (`CompositeMode::FcSem`, `model::FcSemEvaluator`, `estimate::fit_ml_fcsem`),
native expected SEs, standardization, fit measures, df, the `api::frontier`
surface, and the R frontier mirror (`fcsem_model_spec()`,
`fit_ml_fcsem()`/`magmaan_fcsem()`, `fcsem_standard_errors()`,
`fcsem_fit_measures()`, `fcsem_standardized_rows()`). Details are in the roadmap.
Single-group native ML lavaan parity for the pure-composite,
composite-plus-factor, and composite-structural HS fixture trio is now gated by
`cpp/tests/golden/composite_golden_test.cpp`.
Remaining:

- **M/L.** Multi-group composites are in scope only after the single-group ML and
  R frontier slices stay green and only if lavaan handles them cleanly, including
  `group.equal = "composite.weights"`.
- **S, after parity fixtures are green.** Add composite benchmark cases.
- **S — `fit_model()` with `<~` fails opaquely.** Composite syntax passed to
  `fit_model()` (and so to `frontier_fit_sphere()`) ends in a non-finite
  objective from the optimizer. It should route to, or point at,
  `magmaan_fcsem()` (found by experiment _archive/sphere-chart-sanity).

Deferred beyond the lavaan-validated single-group ML slice: ordinal composites,
FIML/LS composites, robust corrections for composites, and composite
mean-structure rows.

## Core/frontier layout follow-ups

Deferred from the first core/frontier separation pass, which introduced
`api::frontier` and retiered FMG, DLS, pairwise-composite, and shrinkage helpers
into `<domain>::frontier`. Canonical public headers now live under
`<domain>/frontier/`, with old public paths kept as forwarding shims. See
[project/design/ideas.md](../design/ideas.md) for the tier model.

- **M/L.** Retier the remaining `data/` research cluster (`h_score`,
  `pairwise_ordinal`, `pairwise_mixed`) into `data::frontier`. Blocked: core
  `data/ordinal.{hpp,cpp}` is entangled with these headers - `ordinal.cpp` defines
  `pairwise_ordinal_stats_from_integer_data` and uses `eval_polychoric_h_score`,
  the core ordinal options embed `PolychoricHScoreOptions`, and `ordinal.hpp`
  `#include`s `pairwise_mixed.hpp`. Moving the headers naively inverts the
  dependency (core -> frontier). This work must first untangle `data/ordinal` -
  separating the core polychoric path from the research builders - then retier.
  `data::frontier::shrinkage` is already retiered, and `r-package/src/fit.cpp`
  calls the frontier namespace while keeping the R surface stable.
- **L.** Relocate the misplaced `estimate/` files to `spec/`: `constraints.hpp`
  (24 includers), `nl_constraints.hpp`, `expr_eval.hpp`, `resolve_fixed_x.hpp`
  (13). A structural relayering — its own pass, with a design note settling
  whether constraint *evaluation* is `spec` or `estimate`.
- **S/M.** Settle whether `cfa_utils.hpp` belongs in `spec` or `model`; depends on
  the start-values decision below.
- **M.** Gather the five start-value producers (`start_values.hpp`) into an
  `estimate::starts` sub-namespace; `start_values.hpp` has 16 includers and
  `spec::Starts` is part of the lavaanified-model triple, so this needs care.
- **M/L.** Retier the moment-quadratic misspecification research surface into
  `<domain>::frontier`. Currently in plain core namespaces: the estimated-weight
  fit-index inference (`ordinal_{crmr,rmsea,cfi_tli,fit_measures}_misspec_inference`
  in `estimate/ordinal`), the profile-Hessian primitives
  (`weighted_moment_profile_*`, `continuous_ls_profile_*`, `ml_profile_*`,
  `observed_moment_bread_fd` in `robust/weighted_inference`; `fiml_profile_*` /
  `two_stage_nt_profile_*` in `estimate/fiml`; `compute_profile_contrast_spectrum`
  in `robust/satorra2000`), the pre-existing exp-research/12 misspec-SE machinery
  (`robust_continuous_ls`, the Hall-Inoue `*_ij` family, `weighted_param_space_sandwich`
  and its estimated-weight MI counterparts `weighted_param_space_sandwich_ij` /
  `continuous_ls_param_space_sandwich_ij` / `ordinal_param_space_sandwich_ij`,
  also in `weighted_inference`/`ordinal`), and the older `OrdinalCatmlDwlsRmsea` probe. All
  are non-lavaan research yet sit in `magmaan::{estimate,robust,estimate::fiml}`.
  This must be one deliberate pass, not piecemeal: `weighted_inference.{hpp,cpp}`
  interleaves the new profile primitives with the widely-used `robust_continuous_ls`
  (callers across `api/`, tests, R glue), so wrapping only the audited stream would
  leave a half-migrated header. Do it with the `<domain>/frontier/` directory move
  + forwarding shims used by the first separation pass, updating R glue
  (`r-package/src/{robust,fit}.cpp`), `experiments/36`, the unit tests, and the
  vendored mirrors (`just vendor`). Flagged by the 2026-06-22 audit of the
  estimated-weight fit-index stream.

## Speculative research lanes (pointers)

Deferred research projects live in [project/backlog/speculative.md](speculative.md)
(a trigger list, not a parallel roadmap). Signposts to the lanes with landed
experiment evidence, so they are visible from the active backlog. Detailed
findings and next steps stay in `speculative.md`; nothing here is scheduled core
work until a concrete downstream consumer appears.

- **funLR — Functional profile-LR CI** (small-sample reliability CIs, JASA-target;
  `experiments/research/active/20-profile-lr-reliability-ci`). Generic-`g` profile-LR (test-inversion)
  engine validated vs `semlbci`; coverage + Bartlett characterized (omega near-nominal,
  bifactor maximal reliability `rho*` collapses to 0.61 @ N=50). **Key finding:** the
  small-sample factor must be a stable model-level **constant** — an analytic Lawley
  factor misses the near-boundary mass (~20-25% recovery), and a per-dataset bootstrap
  (one-level or double) is anti-correlated with need (caps ~0.85, and is `B`-invariant).
  Analytic functional gradients landed (11x faster constrained fits), and the
  C++/R seed now covers ordinary, robust-scaled, misspec-scaled, and
  misspec-mixture parameter profiles for complete-data ML, continuous fixed-weight GMM, direct FIML, ML2S-NT, ordinal, and mixed ordinal, plus
  ordinal polychoric-omega profiles.
  The ordinal-specific calibration lane now has a working note
  (`ordinal_profile_lrt_finite_sample_calibration.tex`) and experiment research/25
  (`ordinal-profile-lrt-calibration`) to separate LR inflation, robust/misspec
  reference scaling, sparse ordinal summaries, and omega point bias before
  choosing a correction. Next: run the full experiment-research/25 grid, then decide
  whether a cell/model-level constant is worth pursuing. See speculative.md for
  the full list.
- **Small-sample DF coverage (Kauermann-Carroll Wald sibling)** — `experiments/44`;
  variance-of-variance `t`-on-effective-df correction for covariance functionals. See
  speculative.md.
- **Guttman own-composite regression: rerun the aligned-map studies** (2026-09-29).
  The aligned map now reads loadings from the own-composite regression
  `K_if = (HB)_if / Q_ff` instead of truncating the multiple regression
  `HB Q^-1`, reports `psi = diag(S) - diag(H)` everywhere, and imposes
  within-factor loading restrictions (including fixed non-marker loadings and
  tau-equivalence) as an exact projection before the marker rescaling. A toy
  3x4 check had the new map ahead in 90-99% of replications at factor
  correlation .6-.8 with a far lighter error tail, so the high-correlation
  failure regime reported by the research/29 paper simulation
  may largely be an artifact of `Q^-1`. On 2026-09-29 the Guttman studies
  (research/15, 24, 26, 27, 28, 29, 30, engineering/10 and the
  constraint-charts archive) were retired from `experiments/`: research/29 (the
  paper simulation) and research/24's runner moved into the guttman-inference
  paper, and the others' findings are recorded in that paper's notes. The rerun
  (Modal, ~$6 at 2000 reps, with a multiple-vs-own-composite arm) is the
  paper's next step and no longer tracked here.
  Remaining C++ work (do not build while another agent holds the build tree):
  (1) `fit_noniterative_cfa` / `noniterative_cfa_theta` (configural entry)
  still hard-code the `triad_wls` communality; thread a `CommunalityMethod`
  through them and the configural Jacobian so the paper's estimator
  (`extended_triad_ls`) no longer needs the restricted entry as a proxy.
  (2) Optional, if the paper ships it: a tau-equivalent communality rule
  `h_i = cbar` (mean off-diagonal covariance) for the restricted map, under
  which the tau-equivalent fit with unit composites returns coefficient alpha
  exactly.
- **Non-iterative CFA inference** — the `estimate::frontier` / `robust::frontier`
  GOF/LRT/SE machinery for closed-form CFA estimators landed (2026-07; Guttman
  1952, delta-method via the map Jacobian; derivations in
  `noniterative_cfa_tests` (guttman-inference paper,
  `papers/guttman-inference/dev/notes/`) and the shared
  `guttman_cfa_asymptotics.tex`).
  **Validated** on the legacy map by the retired research/24 study, whose runner
  and findings moved to the guttman-inference paper on 2026-09-29 (SE coverage + GOF /
  difference-test Type-I / power vs magmaan ML across normal / independent-
  component / ordinal-as-continuous generators, Dhaene-Rosseel 2024 style): the
  headline is that the empirical Gamma is calibrated everywhere while the
  normal-theory Gamma gives *asymptotically* wrong SEs on non-normal data
  (coverage flat at ~0.79 on the independent generator), tracking robust ML;
  the residual GOF brackets nominal (NT over-rejects, empirical conservative at
  small N) with near-full power. A **reliability sweep** (0.3/0.5/0.7) shows the
  efficiency gap widens as reliability drops and spreads from loadings to the
  structural parameters (loadings RMSE ratio ~1.1 at 0.7 to ~1.2-1.25 at 0.3,
  factor covariances similar; residual variances stay at parity), while the
  empirical-Gamma inference stays calibrated throughout; at low reliability x
  small N the closed form throws fewer Heywood cases than ML and never fails to
  converge. A **heterogeneity sweep** (one weak indicator per factor at matched
  average reliability 0.5) shows that a single weak indicator inflates the gap as
  much as making all indicators weak, and -- counterintuitively -- the loss lands
  on the weak indicator's factor-mates, not the weak indicator itself (Guttman's
  triad communality divides by the *other* indicators' correlation, which the
  weak one makes small and noisy); inference stays calibrated even for the weak
  loading, and the Heywood-robustness advantage reverses (under heterogeneity the
  closed form throws more improper solutions than ML). Later slots for the enum:
  FABIN2/Bentler/James-Stein/MIIV maps
  (loadings producers already exist) and a polychoric-ADF Gamma path.
  **H-diagonal communality rules landed** (2026-07):
  `estimate::frontier::estimate_h_communalities` and R `guttman_h()` expose AR,
  RS, triad least squares, anchor triad least squares, blockwise triad-GMM, and
  full selected-triad-GMM for a fixed simple-structure indicator block vector,
  returning `h2`, `diag(H)`, and the filled `H`. `extended_triad_ls` is the
  identity-weighted extended anchor-triad rule: one anchor stays in the target
  indicator's block, while the other may be cross-block.
  The retired research/27 smoke study (8 reps; findings in the
  guttman-inference paper's notes) ranked extended triad LS first. The promoted point-estimator lane landed as
  `guttman_aligned`: blockwise triad-GMM for the H diagonal plus the aligned
  score reconstruction, exposed through the non-iterative CFA C++ and R paths.
  The composite-weight axis also landed: `auto` resolves to `unit` for legacy
  `guttman_lavaan` and `standardized` for `guttman_aligned`,
  `unit` uses incidence weights, `standardized` uses `diag(S)^-1/2 Z` and is
  included in the map Jacobian, and the adaptive compatibility selector was removed on 2026-09-30.
  **Communality and score admissibility machinery landed; constants remain a
  calibration task.**
  The aligned map supports explicit `raw`, `hard`, and smooth `soft`
  communality policies on the symmetric correlation-scale box. `raw` is the
  production default and is bit-for-bit behavior preserving; clamp derivatives
  are included in analytic SEs, and R fits record tuning values and per-block
  activation counts. The engineering/10 screen (below) found no clamp or
  repair worth promoting, so the production default stays `raw` and no
  `(margin, beta0, rate)` is recommended. Making the configural and
  restricted Raw-path improper-split guards consistent remains a separate,
  announced behavior change.
  Score covariance conditioning is separately opt-in and `raw` by default.
  For aligned `unit`/`standardized` composites, `hard` and `soft` repair
  `Q=B'HB` by diagonal shrinkage on its normalized correlation scale, preserve
  `diag(Q)`, guarantee a positive-definite factor covariance at the requested
  `delta_n=floor0*n^-rate`, and include the spectral repair in analytic SEs.
  Hard activation/tie boundaries retain the finite-difference fallback; soft
  repeated eigenvalues use the invariant soft projector. Legacy
  `guttman_lavaan` rejects non-raw conditioning. C++ and
  R fit/inference surfaces retain the exact configuration and per-block
  raw/repaired score eigenvalues, normalized eigenvalues, intensity, floor
  violation, minimum score variance, and marker diagnostic. Existing R fits
  without these fields reconstruct raw conditioning, and nested pseudo-LRTs
  require matching configurations. Configural, metric, and restricted fits now
  all report the real communality-clamp activation counts.
  Experiment engineering/10 (retired 2026-09-29; gate table, arm grid and
  findings kept in the guttman-inference paper's notes; all runs predate the
  own-composite map) crossed the four communality-clamp finalists with hard/soft
  score rates `{0.5,1}` and `delta_50 in {0.01,0.025,0.05,0.1}`, screens them
  under the predeclared success/PD/coverage/tail/benign/runtime gate, and can
  confirm the best two survivors. The completed 24-cell/300-rep screen on
  2026-07-10 had **no survivor**: raw H generated improper communality splits,
  while some clamped-H score arms had non-positive score variances that a
  fixed-diagonal Q repair cannot change. The gate therefore correctly blocks a
  confirmation run and leaves production conditioning raw. The next recorded
  feasibility branch is fixed-diagonal PSD repair of H itself (hard/soft
  normalized spectral shrinkage before aligned score construction), initially
  point-estimation-only with its own 18-cell stress screen. That screen is
  complete (2026-07-10; 100 reps): repaired-H arms achieved 100% point-fit
  success and PD Phi but all worsened loading RMSE relative to their matching
  clamp-only arm; median shrinkage was about 6--12 and hard/soft saturated to
  the same practical map. **Do not build analytic/FD post-fit support or promote
  H repair.** Retain the opt-in point-fit implementation and experiment mode as
  a negative result; the R post-fit surface rejects it explicitly. With the
  study retired, the result is final for the pre-2026-09-29 map. The
  own-composite map no longer inverts `Q`, so if clamps are revisited, rerun
  against it before relying on the old gate.
  **Analytic Jacobians and SE-only inference landed**: `estimator_map_jacobian`
  now uses the regular-interior analytic derivative for configural Guttman
  maps, including the correlation-standardization, triad-GMM communality,
  fixed-rank Moore-Penrose-weight, composite-weight, inverse, and
  marker-scaling chains. The promoted configural path batches the
  block-GMM communality Jacobian and downstream score-regression derivative over
  all covariance coordinates. Restricted estimator-side maps differentiate the
  active communality KKT system and loading projection in the regular interior;
  central differences remain only as the boundary/rank-change fallback. The
  `noniterative_se*` primitives compute just `Omega = J Gamma J'/N`; empirical
  SEs stream casewise moment rows in parameter space, while full
  `noniterative_inference*` remains responsible for the residual GOF projector
  and weighted-chi2 spectrum. A p = 25 configural probe of the research/29 (now the paper sim)
  shape (`normal`, five factors, five indicators each) initially showed
  remaining 5 ms fixed Jacobian cost, not empirical meat: NT and empirical SE
  timings were nearly identical at n = 300, while n-scaling only made the
  casewise contraction visible at much larger n. Follow-up profiling on
  2026-07-09 traced the cost to `triad_wls_h2_jacobian()` building
  full-global-width `dGamma` rows for block-local derivatives and materializing
  dense `dW` matrices when the GMM derivative needs only `A' dW e`. The
  configural path now accumulates block-active derivative columns directly,
  applies the fixed-rank pseudo-inverse derivative as a `dW e` action, and uses
  a guarded full-column-rank pseudo-inverse fast path for `M Gamma M'`.
  Repeated-call p = 25 timings dropped to roughly 1.5 ms for `unit` and
  `standardized`, and 2.3 ms for `adaptive`; the remaining extra
  `adaptive` cost is mostly the downstream data-dependent composite-weight
  derivative.
  The estimator-side residual-restricted Guttman map now has an explicit
  communality axis for the four LS-form H rules (`triad_ls`,
  `extended_triad_ls`, `triad_wls`, `triad_wls_joint`), defaulting to `triad_wls` and
  threading the same choice, along with the selected composite weight, through
  restricted Jacobians and grouped inference; AR/RS are rejected there because
  they do not supply a linear residual-constraint system. A second restricted
  timing pass found that the regular path was analytic but still assembled one
  covariance coordinate at a time: p = 25 residual-restricted SE calls were
  about 166 ms. The default single-block `triad_wls` restricted path now
  batches the constrained h2 KKT RHS with the same block-active `dW e` action
  as configural, solves the KKT system once over all RHS columns, batches the
  downstream score-regression derivative from the constrained `dH` diagonal,
  and applies the loading-projection derivative as an action of
  `(I - C R) W^-1 dW a` rather than materializing inverse derivatives for every
  covariance column. Repeated-call p = 25 timings
  are now roughly 1.8 ms for residual-only restrictions, 2.8 ms for
  loading-only restrictions, and 3.2 ms for both under `unit`/`standardized`;
  `adaptive` is about 2.7 / 3.7 / 3.9 ms. A follow-up fixed-fit empirical
  restricted probe put `unit`/`standardized` at about 1.5x ULS robust SE and
  `adaptive` at about 1.7x in the same p = 25 cell. A later no-constraint
  restricted-proxy speed fix bypassed the stacked KKT system whenever
  `R_h2` is empty, evaluates the selected H diagonal directly, and feeds its
  batched `dH` columns into the existing score-regression/loading-projection
  derivative. The same pass changed correlation-standardization Jacobian rows
  from dense `p*` additions to three-coordinate scatters. This specifically
  removed the `extended_triad_ls` + `standardized` proxy artifact: repeated-call
  p = 15/25/50 fit+empirical-SE timings are now roughly 0.3 / 0.9 / 8.0 ms,
  not 7.3 / 91 / 2142 ms. True residual-communality constraints still use the
  stacked KKT path above, and `triad_wls_joint` remains analytic but
  direction-wise until the joint selected-triad GMM weight derivative is
  batched. The old `guttman` selector is retained as the legacy lavaan-like
  Spearman/incidence map.
  **Deferred: Guttman vocabulary cleanup phase.** The 2026-07 rename
  (commit e873d1d) made the gmm-free names canonical (communality
  `triad_mean`/`triad_pooled`/`triad_ls`/`extended_triad_ls`/`triad_wls`/
  `triad_wls_joint`; estimator `guttman_lavaan`/`guttman_aligned`;
  composite `adaptive`); the adaptive selector was subsequently removed on 2026-09-30
  alias and left loose ends. A later cleanup pass should: (1) decide a
  removal point for the deprecated aliases (`ar`, `rs`, `gmm_block`,
  `gmm_full`, `anchor_triad_ls`, `ilm`, `anchor_ilm`, `guttman`,
  `guttman_gls_aligned`, `gls_aligned`); (2) refresh remaining descriptive
  prose still on old terms (this file's "AR/RS/anchor/triad-GMM" wording,
  the roadmap); (3) align the self-contained research sims
  `guttman_triad_gmm_*.R`, which keep their own
  internal `gmm_block`/`gmm_full` labels; (4) resolve two naming warts held
  deliberately: bare `guttman` still resolves to legacy `guttman_lavaan`
  (decide whether it should instead point at the recommended
  `guttman_aligned`). The aligned default is now `standardized`; explicit
  The retired `adaptive` selector was removed on 2026-09-30. Not in scope:
  the `estimate::gmm` namespace and
  "GMM" solver prose stay (they name the weighted solve, not a method).
  Trigger: when a downstream compatibility window is chosen for alias removal.
  **Quality verdict (2026-07-09): the aligned map is a decent success on
  RMSE; speed is now mostly an SE-coverage caveat, not a blocker.** Bad-boi
  worst-draw sanity (q=3, m=5,
  rho=0.8, weak, n=150; single draw, the paper sim has the across-draws numbers):
  the recommended `extended_triad_ls` + `standardized` recipe lands loading
  RMSE at ~1.1x NTML (a dead heat on the parameters, in closed form on the
  worst draw), and `extended_triad_ls` beats `triad_wls` in every composite.
  The honest cost is implied-Sigma *fit*: ~3x NTML, because the map is a
  covariance functional, not a discrepancy minimizer (fine for a reliability
  functional; disclose for SRMR-minded users). The `adaptive` composite is
  catastrophic here (~22-30x NTML on loadings, loadings to ~65) which is why
  it is retired; legacy `guttman_lavaan` is ~3.65x (the AR/`triad_mean`
  instability, see `papers/closed-form-omega/dev/notes/anti-ar-example.md`).
  Good enough for most purposes. **Speed: the fit wins big, and the old
  extended-via-restricted proxy bottleneck is gone.** Head-to-head timing
  (2026-07-09, n=300, ms/call, p=q*5):
  ```
  p    guttman_fit  guttman_fit+SE   ntml_fit  ntml_fit+SE  uls_fit+SE  ext_via_restricted+SE
  15      0.10          0.63           0.54        0.83        0.74            0.3
  25      0.19          1.74           1.73        3.08        3.27            0.9
  50      0.68         13.4           14.9        55.3        61.5            8.0
  ```
  Read-out: (1) the point fit is 5x-22x faster than NTML and scales better
  (0.68 vs 14.9 ms at p=50); (2) the configural fit+SE fast path already
  BEATS NTML+SE and pulls ahead with p (0.76x -> 0.57x -> 0.24x), so the
  recommended recipe's speed is already a win; (3) the no-constraint
  restricted proxy for extended is now on the direct-H fast path and is faster
  than current configural fit+SE at p=50 in this probe (8.0 vs 13.4 ms), so
  Part B is now first-class configural API wiring rather than a speed rescue;
  (4) within configural Guttman the SE is ~20x the fit (p=50: 0.68 ms fit,
  13.4 ms fit+SE), so any remaining speed work is mostly SE work; (5) the
  remaining C++ speed caveats are true residual-communality constraints, which
  still use the stacked KKT path, and methods whose SE is still direction-wise
  or finite-difference fallback (`triad_wls_joint`, `triad_mean`,
  `triad_pooled`).
  **Paper-sim lane (2026-07-11; the study moved to the guttman-inference paper on
  2026-09-29 and these numbers predate the own-composite map):** the completed 1,728-cell by
  150-replication configural screen now includes raw and soft-clamped aligned
  maps. Factor correlation and loading strength interact sharply: under
  moderate loadings, the raw map's median common-covariance RMSE ratio versus ML
  is about 1.01 / 1.13 / 1.56 / 4.27 at `rho=0/.35/.60/.80`; under weak
  loadings it is about .99 / 1.95 / 39 / 892. The smooth clamp
  (`margin=.001`, `beta0=4`, `rate=1`) lowers the average improper-return rate
  only from 13.4% to 12.7% and does not repair the high-correlation RMSE failure.
  Conditional empirical-SE coverage remains near 95%, but fit and improper
  rates must accompany it. Score and full-H spectral conditioning remain raw.
  The paper grid now makes correct restrictions co-primary: tau-equivalent,
  equal-residual population cells fit both configural and estimator-side
  restricted Guttman/ML/ULS maps on the same draws, with restricted ML as the
  paired RMSE reference. Full runs include that regime by default; the completed
  configural result set contains no paper-grade restricted evidence, so the
  paired restricted run remains the next required result.
  Future point-estimator lane (largely moot for the aligned map since
  2026-09-29: the own-composite regression needs only `Q_ff > 0`, not `Q^-1`,
  so it remains relevant to legacy `guttman_lavaan` only): a
  **boundary-complete Guttman map** (not
  "robust") that always returns a well-labeled object when the composite
  correlation \(P\) is singular or nearly singular. Policy sketch: ordinary
  inverse in the interior; Moore-Penrose inverse for compatible PSD-singular
  cases \(L = LP^+P\), with uniqueness claimed only for the fitted common
  covariance \(LP^+L'\) and an explicit boundary warning for the
  marker-scaled parameter representative; separate, named regularized fallback
  (ridge / PSD repair) for indefinite or incompatible systems, because that is
  no longer the pure Guttman map. Record the derivation before implementation
  (the paper's estimator-criterion note was retired on 2026-09-29, when the
  paper's least-squares definition superseded it).

  **Measurement invariance landed** (2026-07): multi-group blocks with a
  block-diagonal `Omega` for configural maps and a full stacked restricted
  Jacobian when equality constraints couple groups; mean structure (free
  intercepts, latent means fixed at 0); the estimator-side metric map
  `fit_noniterative_cfa_metric` (common standardized loading shape estimated
  inside the Sigma/H reconstruction); the estimator-side restricted map
  `fit_noniterative_cfa_restricted` (separable loading/residual constraints,
  with residual rows imposed in the selected LS-form H/communality step and the
  chosen composite-weight map reused for inference); the `Omega`-metric
  minimum-distance projection onto linear `group.equal` constraints for
  Wald/inference (exact chi2_k); grouped pseudo-LRTs comparing actual
  estimator-side H0/H1 fits; and true (free-latent-mean) scalar invariance via
  the reference-group mean map (linearized pseudo-inverse Wald).
  C++ `estimate::frontier::fit_noniterative_cfa_{metric,restricted}` and
  `robust::frontier::noniterative_{se,inference_grouped,difference_test,
  constrained_fit,scalar_invariance}`; R
  `fit_noniterative_cfa_{metric,restricted}` and
  `magmaan_core$noniterative_cfa_{se,grouped_inference,pseudo_lrt,
  constrained,scalar}_impl`;
  note `constrained_noniterative_cfa` (guttman-inference paper; removed from its
  notes on 2026-09-29, recoverable from that repo's history); validated on the legacy map by the
  retired research/26 study (findings in the paper's notes; metric Wald tracks the ML LRT, the
  empirical Gamma restores the level under non-normality, the scalar Wald is
  nominal and does true scalar in one step where the ML nested test cannot).
  The retired constraint-charts archive study added the marker-chart sanity
  check for the estimator-side metric map: configural and metric-constrained
  implied covariances are marker-invariant at roundoff even off the metric
  surface; theta coordinates differ by chart, as expected.
  Remaining: a **pooled-loading** scalar refinement (test given metric, more
  efficient than the reference-group loadings), an **ordinal** mean-structure
  invariance path (needs threshold modeling, not treated-as-continuous moments),
  and **nonlinear / inequality** constraints (both leave closed form: the first
  needs linearize-and-iterate, the second an active set).

  **Post-fit magmaan-parity interface landed** (2026-07, pure R in
  `r-package/R/noniterative_postfit.R`): the non-iterative fits are now
  first-class `magmaan_fit` objects, so `standardized`, `residuals` /
  `lav_residuals` (SRMR), `factor_scores`, `composite_weights`,
  `compute_defined`, `vcov` (regime `model` -> NT, `robust` -> empirical), and a
  general exported `parameter_table` (est/se/z/p/CI) all dispatch on them.
  `fit_measures` routes to the new exported `fit_measures_noniterative`:
  residual-GOF CFI/TLI/RMSEA + SRMR under the NT (`ntml`) or ULS discrepancy,
  each against a same-discrepancy independence baseline, naive and
  robust/mixture-scaled (user `scale_c` + closed-form baseline `c_0`, NT
  `c_0 = 1`). Fit-index hardening landed 2026-07-09: empirical `c_0` is
  checked against raw cross-product variances, multi-group fits route through
  grouped residual inference and have hand-summed baseline tests, and
  likelihood information criteria (`logl`/AIC/BIC/BIC2) are `NA` by policy
  because the closed-form map is not an ML estimator. The ML score / LRT
  machinery (`modification_indices`,
  `score_tests`, `*_robust`, `*_lrt`, `case_rerun`, `nestedTest`) is guarded off
  for these fits (a closed-form map is not a gradient-zero minimizer). Notes
  `noniterative_fit_indices` and `noniterative_modification_indices`
  (guttman-inference paper); validated by
  `r-package/examples/noniterative_postfit.R`.
  Residual-based modification-index diagnostics landed as the explicit
  `noniterative_cfa_modification_indices()` surface: fixed-zero and absent
  candidates get raw residual scores, tangent-residualized residual scores, and
  local discrepancy-drop/EPC diagnostics, while the ordinary ML score/LRT names
  remain guarded. Still deferred: exact augmented-map/refit releases for
  candidates outside the simple-structure Guttman map, grouped/mean-structure
  MI diagnostics, and continuous **reliability omega from a non-iterative fit** (the omega
  primitive has no non-iterative `weight`; would use the delta-method Omega).

  ### Non-iterative CFA — heavy remaining work (consolidated)

  A single scannable list of the substantial items left after the post-fit
  parity pass. Gathers the pieces scattered in the prose above plus the new
  gaps; ordered roughly by heaviness. The light adapters (standardized,
  residuals, factor scores, composite weights, defined params, parameter table,
  vcov, fit indices) are **done**.

  **Post-fit measures still missing**
  - [x] **Residual-based modification-index diagnostics.** Landed 2026-07-09 as
    `noniterative_cfa_modification_indices()`: raw residual score,
    tangent-residualized residual score, and local discrepancy-drop/EPC
    variants are reported side by side for fixed-zero and absent loading /
    covariance candidates. The generic ML score/LRT MI names stay guarded.
    Remaining work in this lane is exact augmented-map/refit validation where a
    candidate is inside the estimator map, and a grouped/mean-structure
    extension.
  - [ ] **Reliability omega from a non-iterative fit.** `measures_reliability_
    omega_from_fit` has no non-iterative `weight`; wire the delta-method Omega as
    the covariance so omega_total / omega_h / H carry SEs. Natural fit since
    Guttman is itself a reliability-family estimator.

  **Ergonomics / API**
  - [ ] A friendly **one-call entry** (`api::frontier` / a `magmaan_noniterative
    (model, data, ...)` convenience) so the fit + SE + fit-measures are not
    hand-staged. Plus a `summary.magmaan_fit`-style dump.

  **Estimator lanes** (heavy; some already noted in the prose above)
  - [ ] **Boundary-complete Guttman map** — well-labeled object when the
    composite correlation `P` is singular / near-singular (note-first; policy
    sketch already in the prose above).
  - [ ] **Other estimator maps**: FABIN2, Bentler-1982, James-Stein, MIIV-2SLS.
    The `NonIterativeEstimator` enum seam and the loadings producers exist;
    each needs a complete `(Lambda, Phi, psi)` map plus its Jacobian so the
    residual inference bundle consumes it unchanged.
  - [ ] **Invariance remainders**: pooled-loading scalar refinement (test given
    metric, more efficient than reference-group loadings); ordinal
    mean-structure invariance; nonlinear / inequality constraints (both leave
    closed form).

  **Model-form extensions** (heaviest; extend the estimator, not just inference;
  candidates for `speculative.md` with a build-if trigger once a consumer
  appears)
  - [ ] **Polychoric-ADF Gamma path** (ordinal data). The Gamma seam already
    takes any Gamma; the *map* is continuous-moment only, so this needs a
    threshold/polychoric map, not just a new Gamma.
  - [ ] **FIML / missing data** for the closed-form path (complete data only
    today).
  - [ ] **Structural part** — regressions among latents (`~` paths), which also
    unlocks `measures::effects` (indirect / total effects). Needs Reduced-LISREL
    variable-table support.
  - [ ] **Cross-loadings, residual covariances, std.lv identification** — the map
    rejects all three today (simple-structure, marker-only).
- **S. `pEBA-k` silently clamps `k` to `df_diff`; decide whether it should warn.**
  On a nested pair with `df_diff = 1`, `fmg_nested(tests = "peba4_rls")` returns
  the pEBA-1 value, which is exactly the scaled-shifted statistic — verified
  identical to `ss_rls` to 1e-12 in `r-package/examples/fmg.R`. semTests ≤ 0.6
  degenerated the same way (which is why parity used to pass), but **1.0.0 now
  hard-errors**: *"pEBA cannot use more blocks than the test degrees of freedom"*.

  So magmaan is now deliberately the lenient one. Clamping is defensible, but
  returning the `ss` number under a `peba4_rls` label with no signal is the kind
  of thing that silently pollutes a simulation grid — a cell labelled pEBA-4 that
  is not pEBA-4. Options: (a) keep clamping silently, (b) attach a warning to the
  returned row, (c) match upstream and refuse. (b) is the cheap middle and does
  not break callers. Not urgent; the behaviour is now asserted in `fmg.R` so it
  cannot drift unnoticed.

  Resolved in passing while investigating (all 54 examples now pass):
  - `fmg.R` passed `tests = as.list(parity_tests)`; 1.0.0's `validate_tests()`
    requires `is.character(tests)`. One word. Non-nested parity: max|Δp| =
    2.0e-11 over 50 cells.
  - `fmg.R` **nested** parity was comparing magmaan's default `A.method =
    "exact"` restriction map against semTests' delta-based `a`
    (`nested_factor_2000` → `get_a_matrix`). Apples-to-oranges: up to 13% per
    eigenvalue, ~6% in trace, ~4e-4 in the nested tails — which the old
    *absolute* `< 1e-3` tolerance hid completely once the p-values were
    themselves ~1e-4 (e.g. `sb_ml` 3.96e-4 vs 2.15e-4, an 84% relative miss
    passing a 1e-3 absolute gate). Passing `A.method = "delta"` takes the grid to
    **2.5e-9**, and the tolerance is now `1e-7`, which has teeth in the tail.
    The parity pair also moved to a `df_diff = 5` H0 so the `peba4_*` cells are
    comparable at all.
  - `nested_test_2001.R` — `semTests:::ugamma_nested(., method = "2001")` is gone
    and **not renamed**: 1.0.0's NEWS.md says *"Nested method 2001 is withdrawn
    because of its poor performance"*, and both survivors
    (`lav_ugamma_nested_2000`, `ugamma_nested_reference`) are method-2000
    constructions. The parity block was dropped rather than aimed at a
    differently-defined internal, and replaced with self-consistency checks.
    magmaan **keeps** `ud_method = "2001"`: it is the documented fallback when
    SB2010 cannot run (method 2000 needs a same-parameter restriction map, 2001
    needs only the two fits — see `cpp/src/robust/lr_test_satorra.cpp`), and it is
    gated transitively per AGENTS.md — `U0`/`U1` come from the lavaan-gated
    single-model spectrum machinery (`ugamma_eigvals_nt`,
    `mlr_trace_ugamma{,_h0,_h1}`) and the difference plus eigen-solve is checked
    against an independent dense `Eigen::EigenSolver` oracle in
    `cpp/tests/unit/satorra2000_test.cpp`. Upstream's verdict is about power, not
    correctness, so prefer `"2000"` when both apply.
- **Completed 2026-09-20: SEM scaling / PSD optimization defaults (experiment _archive/sem-total-variance).**
  Historical decision, superseded for complete-data ML defaults on 2026-09-22
  by the broader fixed-control start/scaling validation described above. The
  original native-start comparison favored no preconditioning; retain these
  findings as a scope limitation, not the current default policy. No automatic chart
  selector or targeted restart is adopted. Across six structures/ten settings,
  all 220 fits passed audits in 2.83s; diagonal scaling increased median
  evaluations in every setting. 109/110 paired objectives agreed within `6e-12`;
  the default unscaled fit attained the lower objective in the discrepant
  mediation pair. Its boundary identification mechanism and a successful
  model-specific restart are documented in the completed experiment report.
  First-order acceptance is not a local/global optimum certificate at singular
  representations. This limitation is retained, not a blocker for the default
  decision. Further candidates and explicit reopening conditions live in
  `speculative.md` under "SEM PSD optimization beyond the engineering baseline".
- **S. Test whether per-chart start values explain the std_lv optimizer-work
  advantage beyond chart geometry.** Findings and framework in
  [project/design/parameterization-geometry.md](../design/parameterization-geometry.md);
  measured by `experiments/research/banked/46-latent-metric-geometry`, which supersedes
  `experiments/_archive/latent-metric-identification`.

  exp research/46's cost arm has `std_lv` at roughly a third of `marker`'s `f_evals`, and
  the gap **persists at `lambda1 = 0.9`** (66 vs 19 at p=12) where marker is
  competitive on both conditioning (25.1 vs 28.0) and parameter-effects curvature
  (0.82 vs 0.57). Geometry does not account for that, so magmaan's per-chart start
  heuristics are the remaining suspect. Worth chasing because it is free speed on
  the default (marker) path, which is the one users actually take. Add a
  start-value arm to exp research/46 (fixed common starts projected into each chart vs each
  chart's native heuristic) before touching `spec/start`.

  Scope of the earlier findings:
  - *Is there a better convention?* Still open outside the measured CFA models.
    `std_lv`'s PE curvature sits below intrinsic curvature there, but that does
    not prove numerical optimality; completed experiment _archive/sem-total-variance compared structural alternatives without changing the PSD default.
  - *Should the numerics lever be the chart?* No, the optimizer's metric. Full
    Newton / Fisher scoring is affine-invariant, so linear conditioning is free;
    nothing absorbs PE curvature. This is why `effect_coding` beating `std_lv` on
    conditioning while losing on curvature is not worth acting on.
  - *exp _archive/latent-metric-identification's back-convert wash.* **Superseded.** exp _archive/latent-metric-identification's timings came from
    `system.time()`, which quantises to ~1ms on Linux, applied once per
    sub-millisecond fit (ten timings of the same 225µs fit give min 0.000, median
    0.001, max 0.007). With batched timing at 1.7% relative IQR, the fit ratio
    exp _archive/latent-metric-identification put at 0.839 is **0.427** at p=24 under nlopt-lbfgs and improves with p.
    Back-conversion is exact to 1e-15, costs 0.07% of the fit for point estimates
    at p=48, and ~3% including the dense O(p³) `J V J'` vcov transform that exp _archive/latent-metric-identification
    never counted. So roughly 3% overhead against a 30–57% saving: the
    internal-chart substitution pays.
  - *Does conditioning cost convergence at small n?* Yes, and strictly. At p=12,
    n=50, nlopt-lbfgs, marker fails 2.04% while std_lv and effect coding fail 0%,
    and across 2000 matched draws marker-fails-std_lv-succeeds happens 7 times
    against 0 the other way. A dominance relation rather than a rate difference.
  - *Does `std_lv` fix Heywood cases?* No. It relocates them from latent to
    observed variances (exp _archive/heywood-box-constraints: 0/3 admissible without bounds; marker + `pos.var`
    is the winner at 2/3). exp research/46's detector therefore reads every estimated
    variance — **but exp research/46's population produces zero improper solutions**, so its
    convergence arm cannot corroborate or contradict exp _archive/heywood-box-constraints. Loadings of 0.7 with
    ψ=0.51 are not extreme enough. Wiring a near-Heywood population into exp research/46
    (weak loadings, tiny ψ, n≈50) is the cheap way to put both results on one grid.
- **S. (resolved 2026-09-29) standardized composite evidence.** The
  guttman-inference paper now proves that standardized composites make the
  Guttman map scale-equivariant and unit composites do not, and the paper
  simulation (moved there from research/29) carries the indicator-scale slice
  if a practical-size check is wanted.
