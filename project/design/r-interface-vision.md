# magmaan vision: an opinionated package over a research lab

Status: adopted direction, 2026-09-25. The package split has landed: the
compiled package is `magmaanlab` in `r-package/`, and the pure-R `magmaan`
package is in `r-magmaan/`. The policy composer (`api::policy_inference_ml`)
covers single-level complete-data ML; other estimators fit but report their
inference as `unsupported_model`. The remaining work is tracked in
[todo.md](../backlog/todo.md#primary-inference-workflows).

## Intention

magmaan serves two audiences over one C++ core.

- **Ordinary users** write one call. It estimates the model and computes
  inference under a single policy that magmaan chooses and justifies. The
  package has a few familiar options and no compatibility conventions:
  no MLR, no WLSMV, no switches for information matrices, standard-error types
  or test corrections. Choosing well is magmaan's job, not the user's.
- **Power users** (methods developers, including magmaan's authors) get the
  full composable surface: every estimator path, every information and Gamma
  convention, compatibility routes such as lavaan's MLR, and the frontier
  methods the research depends on.

This supersedes two earlier rules: that `magmaan()` performs estimation only,
and that the audience is methods developers rather than end users. It keeps the
rule that R owns argument handling and presentation while C++ owns the
statistics, including the ordinary-user policy itself. Neither package holds a
second SEM implementation.

The [statistical scope](../scope.md) owns the policy's target and sampling law:
population approximation parameters under joint observation sampling, including
random covariates. Conditional fitting can serve this target. General
fixed-design inference under mean misspecification is banked, as is categorical
conditional-moment expansion. For models with fixed observed covariates, the
existing `fixed.x = TRUE` fitting default does not establish ordinary-policy
support; unsupported inference remains
explicit until a separately validated route exists. This scope decision changes
no fitting default or compatibility contract.

## Two packages

| Package | Audience | Contents | Promise |
| --- | --- | --- | --- |
| `magmaan` | Ordinary users | Pure R. `magmaan()`, its result class and a few methods | Small and stable |
| `magmaanlab` | Power users | The compiled package in `r-package/`: all bindings, primitives and frontier methods | Changes freely, like `frontier` |

- `magmaan` imports `magmaanlab` and has no compiled code, so the package
  boundary enforces the no-SEM-logic rule.
- Power users attach both, so no name may be exported by both with different
  meanings. The lab's estimate-only convenience is `fit_model(model, data,
  estimator, ...)`, formerly `magmaan()`, and it now takes `psd = TRUE` to
  dispatch to the PSD-constrained fitters. The prepared `estimate()` path does
  not replace it: it needs prepared objects and has no ML2S or two-level route.
- An ordinary fit converts to a lab fit with `as_lab_fit()`, so any
  alternative inference runs in the lab without refitting.
- The policy composer lives in C++ under `api::`, so C++ callers get the same
  opinionated call and the R package calls one lab binding.
- Layout: `r-package/` holds `magmaanlab` (unchanged path, so vendoring, the
  fast dev loop and cluster installs are unaffected) and `r-magmaan/` holds
  `magmaan`. Both are T2 in the layering rules.
- `meanstructure = "default"` follows lavaan: a mean structure for multiple
  groups, ordered variables, FIML and ML2S, and syntax with an intercept. With
  that rule the lab reproduces lavaan's parameter rows in each case.
- The 0.1.0 simulation prerelease accepts lavaan syntax and saved specs carrying
  lavaan syntax. Ordered variables, parameterization and grouping are inherited
  from a spec before validation; explicit conflicts error. EQS and partable-only
  specifications stay lab-only, with ordinary EQS adoption undecided.
- Defined estimates are retained during fit reconstruction independently of
  inference. The ordinary README and help pages specify simulation extraction;
  consumers pin the ordinary package and compiled dependency together.

## The ordinary-user call

```r
magmaan(model, data,
        estimator = "ML",
        ordered = NULL,
        group = NULL, group.equal = NULL, group.partial = NULL,
        cluster = NULL,
        identification = "marker",
        parameterization = "delta",
        meanstructure = "default", fixed.x = TRUE,
        missing = "listwise",
        psd = FALSE,
        start = "default",
        inference = TRUE)
```

Naming rule: use lavaan's name where the concept is identical, and a new name
only where lavaan's name is poor or magmaan's meaning differs. A lavaan user's
`group = "school", group.equal = "loadings", ordered = c("y1", "y2")` then
works as typed. lavaan's `ordered = TRUE` (every endogenous observed variable)
is deferred; the scaffold asks for the names. This makes the ordinary package dot-case while the lab stays snake_case,
a deliberate trade for users moving from lavaan.

### Requested identification is part of the result contract

Adopted 2026-09-27: do not automatically change the user's marker indicators
to turn a failed requested identification into a successful fit. If a candidate
is too close to a pole of the requested chart, fail that fit and explain the
identification problem. Another marker or the sphere may be used internally
for diagnosis, but its estimates must not replace the requested result.
Retain the diagnostic point and implied covariance where available; choosing
a different identification is an explicit user action.

The concrete promotion witness is the retained fresh weak-marker N=100 draw 4
in engineering/active/15-sphere-reference-fits: a marker loading near 184,000 becomes 1 under a different
marker, while the fitted covariance is unchanged. This endpoint must fail in
the requested chart. It is a numerical near-pole case, not a proof of an exact
pole or of nonattainment in every chart. The general rejection criterion is
still to be validated: the existing pole tolerance 1e-6 admits this saved point;
a study check at 1e-4 rejects it. No library tolerance is changed by this policy
record. Accuracy failures and covariance inadmissibility remain separate causes.

### Estimators

Development follows the [estimator tiers](../architecture/roadmap.md#estimator-development-priorities)
adopted 2026-09-27 and narrowed 2026-10-01: NTML, FIML and all-ordinal DWLS
come first, with PSD developed alongside them. Barrier-specific completion
follows in 0.0.2. Priority includes inference and interface completion; it does
not itself expose a method here or make it a default.
The broader estimator list below records the intended surface, not equal
implementation priority or complete policy-inference availability.

Mixed continuous/ordered completion is assigned to 0.0.2. Noniterative
development is indefinitely postponed; existing lab APIs and regression gates
remain, with reopening requiring an explicit user scope decision. MI and
release-score work across the remaining estimator/weight families is tracked
in the [completion matrix](../backlog/todo.md#mi-and-release-score-completion-001).

`estimator` names only the estimator. Continuous data: ML, FIML, ML2S, GLS,
ULS and WLS. Variables declared in `ordered`: DWLS, WLS and ULS. The data type
is declared by `ordered`, never inferred from the estimator. Normal-theory ML
on Likert items means not declaring them ordered. Invalid pairs error with a
message naming the valid alternatives.

lavaan names that bundle an estimator with a correction (MLM, MLR, MLMV, WLSM,
WLSMV, ULSM, ULSMV) are rejected with a pointer to the plain estimator, since
inference is automatic. This matches the backlog item on decomposing
`EstimatorSpec` into its actual axes.

### Options kept, renamed and dropped

| lavaan or current option | Ordinary package | Reason |
| --- | --- | --- |
| `missing = "ml"` | `estimator = "FIML"` | It is an estimator |
| `missing = "two.stage"` | `estimator = "ML2S"` | Same |
| `missing = "listwise"` | Default, stated in the output | Every estimator is defined under listwise deletion |
| `missing = "pairwise"` | Kept where supported (ordinal) | |
| `std.lv = TRUE` | `identification = "std.lv"` | One option with room for more conventions |
| `groups` (current magmaan) | `group` | lavaan spelling |
| MLM, MLR, MLMV, WLSM, WLSMV, ULSM, ULSMV | Rejected | Estimator plus correction in one name |
| `se`, `test`, `information`, `h1.information`, `observed.information`, `likelihood`, `bootstrap` | Dropped | Set by the policy |
| `bounds` | `psd` | The principled replacement |
| `start` | `"default"`, `"fabin3"`, a previous fit or a parameter table (added 2026-09-26) | FABIN3 was the ML and GLS start before the layered start; it reproduces earlier results and gives every fit, ordinary and PSD, the same start. A fit or table sets the start of each matching free parameter, as lavaan's `start = fit`. Other start constructors stay in the lab |
| `optimizer`, `control`, `W`, `stage2_weight`, `dls_a`, `stage1_regularization`, `pd_gamma` | Lab only | Expert tuning |
| `orthogonal`, `auto.*`, `int.ov.free` and similar | Lab only | Expressible in model syntax or `model_spec()` |
| `sample.cov`, `sample.nobs` | Lab only | The policy needs raw data |

Missing data: listwise deletion is automatic for every estimator not designed
for missing data. The fit records the rows used and deleted per group, and the
summary reports both. An estimator path that cannot yet delete listwise is a
backlog gap, not a reason to change the default.

Raw data only: the sandwich covariance and the SB and PEBA4 calibrations need
casewise contributions. Covariance-matrix input stays in the lab, where the
user chooses what replaces the missing empirical moments.

## Inference policy

`inference = TRUE` is the default. With `inference = FALSE`, the call only
estimates; `infer(fit)` later runs the same composer without refitting, reusing
retained data and geometry. Repeated `summary()`, `vcov()` and `confint()`
read retained results.

Parameter uncertainty:

- Covariance: the sandwich with observed-information bread and empirical score
  covariance, H^-1 J H^-1 / n. It is consistent for the pseudo-true parameter
  under misspecification in the scope's regular joint-sampling regime; an
  expected-information bread is not generally consistent there. For estimators
  whose weight is estimated from the data (GLS, WLS, DWLS), the covariance
  includes the weight-estimation influence, which vanishes under a correct
  model but not under misspecification.
- Standard errors, Wald z-tests and symmetric Wald intervals from that
  covariance, and defined parameters by the delta method. A contrast such as
  `d := a - b` gives the one-df Wald test of `a == b`.
- `confint()` takes `test = "wald"`, the only value so far. Inverting the
  robust likelihood-ratio test (`"lr"`) is planned there. Wald stays the
  default until a pre-registered decision study says otherwise (backlog).

Global tests against the saturated model:

- The score test and the likelihood-ratio test, each calibrated with SB and
  PEBA4. The likelihood-ratio statistic and its df are reported because
  classical readers look for them; the normal-theory p-value is not shown.
- For fixed-weight estimators (GLS, ULS, WLS, DWLS) the objective is exactly
  quadratic in the saturated moments, so the global score statistic equals the
  fit-function statistic n F. It is reported once, labelled as the fit-function
  statistic, since it is not a likelihood ratio.
- Profiling: when the score uses the estimator's own weight, the nuisance part
  of the score is zero at the estimate (the first-order condition), so the
  effective (profile) score vector equals the raw score. The statistic is that
  vector's quadratic form in the inverse of its covariance, and the covariance
  depends on the sensitivity used for the nuisance projection. When the
  projection uses the metric's own information the statistic reduces to the
  unprojected form; with a different sensitivity (observed with an expected
  metric, say) the statistic changes as well as the spectrum. When the score
  uses a different weight from the estimator, the nuisance part no longer
  vanishes, and the test must use the effective score explicitly. Which
  sensitivity and score weight the policy uses is settled by experiment
  (_archive/complete-ml-global-test-geometry for complete-data ML; see the backlog).
- Statistic and calibration are labelled separately; the spectrum or trace
  and numerical diagnostics are retained.
- Geometry of the global tests, by estimator:
  - Complete-data ML: expected information for the score sensitivity, the score
    metric and the LR spectrum, with the empirical Gamma. Experiment
    _archive/complete-ml-global-test-geometry (FMG 2024 two-factor designs, 14,000 fits) found
    expected-information score PEBA4 within 2.0 to 7.2% rejection in all 32
    cells; observed sensitivity drove the score test to 0% rejection as p grew,
    an observed score metric was often not positive definite and far too
    liberal, and the LR spectrum's information choice did not matter.
  - FIML: observed-H0 sensitivity with the expected (pattern-conditional
    Fisher) metric in the score test, per research/44: expected sensitivity
    fails under non-normal MAR and an observed metric is unstable. Under MAR the
    score and LR tests are then not asymptotically equivalent, so each keeps its
    own spectrum and both are reported.

Nested comparisons take an explicit second model, `anova(fit0, fit1)`, and
report the analogous score and likelihood-ratio (or fit-function difference)
tests with SB and PEBA4. The comparison checks that data, estimator and nesting
agree.

Complete-data ML nested geometry (`api::policy_nested_ml`, 2026-09-26):

- **Nesting.** The restricted model is the other model plus equality
  constraints on the same parameters (shared labels, or `b == 0` on a labeled
  parameter). The restriction map is exact. A restriction written as a fixed
  value (`0*`) changes the parameter slots and is not yet recognized (backlog).
- **Likelihood ratio.** The difference of the two normal-theory fit statistics.
  Its spectrum uses the expected information at the larger model with the
  empirical Gamma: Satorra (2000) with the exact restriction map. SB equals
  lavaan's `lavTestLRT(method = "satorra.2000", A.method = "exact",
  scaled.shifted = FALSE)` on MLM fits. Evidence for SB and PEBA on this
  statistic: FMG (2026) Study 2, reproduced in
  `experiments/replications/07-foldnes-moss-gronneberg-2026-study2/`.
- **Score.** Evaluated at the restricted fit: casewise likelihood scores along
  the restriction directions, projected against the restricted model's own
  directions with its expected information (the efficient score), and the
  expected-information metric. The projection keeps the statistic meaningful
  where the restricted fit is not stationary, as at a PSD boundary. It mirrors
  the global score test (_archive/complete-ml-global-test-geometry); a calibration study of its own is
  still missing (backlog).

Where observed information matters: under the global null the observed and
expected Hessians differ by O_p(n^-1/2), so the global tests have the same
asymptotic law either way and the choice is a finite-sample one. For parameter
covariance under misspecification, and for nested tests whose larger model is
misspecified (the usual invariance-testing case), the choice is first order.
The nested geometry must therefore be specified component by component before
implementation: nuisance sensitivity, quadratic metric, evaluation point,
moment covariance, centering and normalization. Do not mechanically replace
every matrix by an observed Hessian.

A misspecification-robust covariance and an exact-fit test answer different
questions. The global tests still test the null model; the robust covariance
does not make that null true.

### Availability

- A successful fit survives unavailable inference. Each unavailable component
  carries a typed reason that the summary shows.
- No silent substitution of expected information, a different correction or a
  different estimator.
- Structural gaps (an estimator path whose policy is not yet implemented) are
  reported as unavailable, not refused, so the estimates remain usable.
- With `psd = TRUE`, every converged fit gets full inference, including one
  whose estimate lies on the boundary of the covariance space (since
  2026-09-26). When the population is interior, the PSD and ordinary
  estimators coincide with probability tending to one, so the regular limits
  apply; a boundary estimate is a finite-sample event. The output says that
  the inference assumes an interior population. Improper ordinary fits get
  inference under the same assumption. A population on the boundary (a
  hypothesis that a growth factor has no variance, or a correlation of one)
  has chi-bar-square limits and stays out of scope. Evidence at interior
  populations near the boundary: the covariance-honest paper's
  interior-inference study, with normal-theory tests so far; its rerun with
  the policy components is planned. The default stays `psd = FALSE` pending
  the separate decision on the default PSD estimation policy.

## Validation

The ordinary package's output as a whole matches no lavaan call, by design.
Correctness rests on three kinds of evidence:

1. **Checked components.** Where lavaan computes the same quantity, the C++
   golden tests keep gating it: the observed-bread sandwich is lavaan's
   `robust.huber.white` covariance, SB on expected information is lavaan's
   `satorra.bentler`, and the fit-function statistics are lavaan's standard
   tests. Users never see those lavaan names.
2. **Identical composition.** The automatic policy and the equivalent explicit
   composition of lab primitives agree exactly, and `inference = FALSE`
   followed by `infer()` equals the default call.
3. **Evidence for each policy choice.** Every default cites the experiment or
   paper that supports it. For example, SB and PEBA4 replace lavaan's MLR
   Yuan-Bentler-Mplus test because of the recorded calibration evidence. FIML
   evidence and its distinct pseudo-null/geometry limits are in
   `experiments/research/active/44-fiml-global-tests/report.qmd` and
   `experiments/research/active/06-fiml-invariance-tests/report.qmd`; the older
   ten-replication FIML/MLR lane cannot independently justify a default.

## Frontier methods in the ordinary package

A research method enters `magmaan` when all three hold:

1. A paper or experiment in this repository backs it.
2. It fits as a value of an existing argument or as one post-fit function.
3. Its inference availability is defined.

Otherwise it stays in the lab.

- **In now:** `psd = TRUE` (PSD fits exist for ML, FIML, ML2S, ULS, GLS, WLS,
  ordinal and mixed data) and PEBA4, which is part of the policy.
- **Candidates:** `identification = "sphere"`, the closed-form CFA estimator
  and the retained correlation-target ML capability currently called catML.
  Its shared fitting/metadata replacement is near-term lab work; ordinary-user
  exposure still requires the gates above.
- **Priority frontier, currently lab-only:** the multi-information barrier.
  Ordinary-user exposure requires its argument and inference contracts plus
  the evidence above; development priority alone does not promote it.
- **Other lab-only methods:** robust ordinal estimation, FC-SEM, flip tests
  and simulation.

The proposed `covariance = "unrestricted" | "psd" | "barrier"` spelling is an
open naming and compatibility decision, not an available argument. Internal
domain constraints and penalties remain independent. Correlation-ML fitting
retains ordinal sampling provenance; it does not inherit Gaussian raw-data
likelihood or automatic continuous-ML inference. See
[shared composition](../architecture/roadmap.md#shared-fitting-composition).

## Starting point

Checked against the source on 2026-09-25.

| Surface | Today | Consequence |
| --- | --- | --- |
| `magmaan()` in `model_data.R` | Estimate-only; `se` and `test` must be `"none"`; many estimator-specific controls | Renamed to the lab's `fit_model()`; the ordinary `magmaan()` is new |
| S3 methods on `magmaan_fit` | `print`, `residuals` and `vcov` only | `summary`, `coef`, `confint` and `anova` are new; the parameter table is `coef(summary(fit))` |
| `vcov.magmaan_fit()` in `context.R` | `regime = "model"` means inverse observed information for FIML but an expected-bread sandwich with empirical meat for complete ML; continuous fits stop without `data` although fits retain `fit$raw_data` | Retained data already exists; the regime naming is inconsistent across estimators |
| `prepare_inference()` in `scores.R` | ML, FIML and fixed-NT ML2S; rejects active bounds | The ordinary policy needs every listed estimator |
| `inference_quadratic()`, `inference_covariance()`, `calibrate_quadratic()` | Score and LR, each with SB and PEBA4, plus the robust covariance, on structured expected-information geometry (`man/inference_reuse.Rd`) | The bundle's composer exists; the geometry changes |
| `score_components()` | Separate sensitivity and metric, including observed options | Supplies the distinctions the nested geometry needs |
| `robust_se()` in C++ with observed bread | Gated against lavaan's `robust.huber.white` in `inference_golden_test.cpp` | The complete-data ML covariance is already checked |
| `nested_score_test()` / `score_flip_test()` | ML and FIML only | The nested score test for the least-squares estimators is new work |
| `fmg_tests()` | Fit-function statistic with SB and PEBA4 for ML, FIML, ML2S, continuous LS and ordinal fits, with several default test sets | Supplies the least-squares global tests |
| Listwise deletion in the data constructors | Applied, but the count of deleted rows is not recorded | Recording is new work |

## Deferred

These are decided later and are not part of the first release: `ordered =
TRUE`, `fit_measures()`
(including which statistic feeds CFI and RMSEA; PEBA4 has no index analogue),
modification indices under the policy, factor scores through `predict()`,
a `control` option for non-converging fits, and summary-statistic input.

## Implementation sequence

The steps below retain the package-split history. For remaining work, the
[development priorities](../architecture/roadmap.md#estimator-development-priorities)
supersedes their original ordering: finish NTML gaps and extend policy inference
to FIML and all-ordinal DWLS first, with PSD work alongside them. Mixed
continuous/ordered and barrier-specific completion follow in 0.0.2.
Other least-squares estimators and ML2S extend for a concrete consumer or
inexpensive shared benefit. Two-level, SAM and composite expansion is parked;
the historical steps below do not schedule their policy work. Broader PSD/barrier
applicability is consumer-gated, with estimation and inference validated for each
combination.

1. Build the C++ policy composer for complete-data ML: observed sandwich,
   global score and likelihood-ratio tests, SB and PEBA4. Gate it against lab
   primitives and the lavaan components above.
2. Extend it to the least-squares estimators: the global fit-function identity,
   the weight-influence covariance, and the new nested least-squares score test.
3. Make listwise deletion available and recorded on every estimator path.
4. Rename the compiled package to `magmaanlab`, remove its `magmaan()`, and
   create the pure-R `magmaan` package with `magmaan()`, `infer()`,
   `as_lab_fit()`, `print`, `summary`, `coef`, `vcov`, `confint`, `fitted`
   and `anova()`. Done on 2026-09-25; `anova()` on 2026-09-26 for complete-data
   ML. The same day the exported `parameters()` became `coef(summary(fit))`,
   and `fitted()` (model-implied moments, as lavaan's) arrived with support
   for fully specified models, so populations can be written in model syntax.
5. Extend the policy to FIML, ML2S, ordinal and mixed, and two-level fits,
   with a component-level capability table.
6. Migrate experiments, examples, the vendoring scripts and the cluster install
   notes to the two package names, timing preparation, estimation and inference
   separately.
