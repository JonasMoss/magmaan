# lavaan tutorial parity

This document audits magmaan against the [lavaan tutorial](https://lavaan.ugent.be/tutorial/),
section by section. It is the reference for *what magmaan can do* relative to
the tutorial and *where that is tested*. It is a living checklist — update it
when capability or coverage changes.

For the current implementation state and architecture see
[docs/architecture/roadmap.md](../architecture/roadmap.md); for the remaining-work backlog see
[docs/backlog/todo.md](../backlog/todo.md); for the magmaan-vs-lavaan speed
comparison on these models see
[experiments/05-lavaan-speed-bench](../../experiments/05-lavaan-speed-bench/).

## Scope

The tutorial has 18 sections. Sections 1–3 (Overview, Before you start,
Installation) are non-technical. The technical sections are 4–18.

- **In scope** — sections 4–16: model syntax, CFA, SEM, meanstructures,
  multiple groups, growth curves, categorical data, covariance-matrix input,
  estimators, mediation, modification indices, extracting information.
- **Out of scope** — section 17 (Multilevel SEM) and section 18 (ESEM/EFA):
  magmaan is a linear-SEM library and does not target these.

magmaan deliberately exposes an *estimate-then-explicit-post-fit* API rather
than lavaan's `cfa()`/`sem()`/`growth()`/`summary()` one-shot ergonomics (see
roadmap: "end-user lavaan replacement ergonomics" is out of scope). So a
"can do" verdict below means *the quantity is computable through the documented
API*, not that a same-named convenience function exists.

## Three layers

Each section is assessed across the three layers the project ships:

- **core** — the C++ numeric library (`include/magmaan`, `src`).
- **api** — the staged C++ facade `magmaan::api` (`include/magmaan/api`).
- **R** — the R package (`r-package/`): `magmaan()`, `model_spec()`,
  `compute_defined()`, `nestedTest()`, and the `magmaan_core` primitive
  environment. The R package is a methods-developer interface and exposes a
  *subset* of the C++ api: estimation, information / vcov / SE / z / Wald /
  χ² / df / baseline / fit measures, robust SEs and Satorra-Bentler tests,
  defined parameters, nested tests, model-implied moments, the standardized
  solution, and modification indices / score tests.

Legend — Cap: ✓ supported · ◐ partial · ✗ not supported. Test: which checked-in
test gates it (parity fixture / corpus golden / unit / none).

## Section-by-section audit

### §4 — Model syntax 1 (`=~`, `~`, `~~`, `~1`)

The four core operators: latent definition, regression, (co)variance,
intercept. Plus multi-line continuation, `;`, comments.

- **core ✓ / api ✓ / R ✓.** Full recursive-descent parser
  (`src/parse/`), grammar in `docs/grammar/grammar.ebnf`. `~1` is synthesized
  by the parser. Lavaanify (`src/spec/build.cpp`) projects to the model triple.
- **Test:** `lexer_golden_test` / `parser_golden_test` / `lavaanify_golden_test`
  (fixtures `tests/fixtures/{lexer,flat,ptable}`), unit `lexer_test` /
  `parser_test` / `lavaanify_test`, corpus cases `0001`–`0020`.
- **Gap:** none.

### §5 — A CFA example

3-factor Holzinger-Swineford CFA via `cfa()`; `summary(fit, fit.measures=TRUE,
standardized=TRUE)`.

- **core ✓ / api ✓ / R ✓.** CFA fits through ML; SEs, χ², fit measures, and
  standardized solution are explicit post-fit calls.
- **Test:** parity `hs_3factor_cfa` (real-data ML, n=301), corpus `0001`/`0002`,
  `inference_golden_test`, `fit_measures_golden_test`, `standardized_golden_test`.
- **Gap:** no `cfa()` wrapper and no `summary()` — by design. The example is
  reproduced with `magmaan()` + explicit post-fit calls.

### §6 — A SEM example

Bollen's Political Democracy industrialization/democracy model: latent
regressions and correlated residuals.

- **core ✓ / api ✓ / R ✓.** Latent regressions (`~` → Beta), residual
  covariances (`~~`), observed-variable path models via phantom latents.
- **Test:** parity `bollen_democracy_sem`, corpus `0011` (regression),
  `0019` (path), `0020` (CFA + structural).
- **Gap:** none.

### §7 — Model syntax 2

Fixing parameters (`2*x`), freeing (`NA*x`), starting values (`start()*x`),
labels, simple equality (shared label), `equal()` modifier, `orthogonal=TRUE`,
`std.lv=TRUE`, nonlinear equality constraints (`b1 == (b2+b3)^2`), inequality
constraints (`b1 > exp(...)`), multiple modifiers on one term.

- **core ✓ / api ✓ / R ✓.** Fixing, `NA*`, `start()`, labels, simple equality,
  linear equality (`a == 2*b+c`), `std.lv`, multi-modifier terms (corpus
  `0004`–`0008`, `0013`, `0015`, `0018`; `fit_lincon` fixtures;
  `lin_constraint_golden_test`).
  - **`equal()` modifier ✓** — `equal("visual=~x2")*x3` parses and lavaanifies
    into a shared-label tie (`parser_test`, `lavaanify_test`).
  - **`orthogonal=` ✓** — `model_spec(..., orthogonal = TRUE)` fixes the auto
    latent covariances at 0, mirroring lavaan's partable (`lavaanify_test`).
  - **Nonlinear equality constraints ✓** — `b1 == (b2+b3)^2` is compiled and
    enforced by the IPOPT fit path; the constrained vcov / df are
    Jacobian-projected. `exp()` / `log()` function calls are supported in
    constraint and `:=` expressions, and a model carrying *both* a linear and
    a nonlinear equality constraint fits (IPOPT runs in the linear-reduced
    α-space). Lavaan-parity unit tests in `constraints_test`.
- **Gap:**
  - **Inequality constraints `<` / `>` ✗** — out of scope: estimation would be
    fine but active-bound (chi-bar-squared) inference needs machinery magmaan
    does not have. They fail with an explicit, specific early error.

### §8 — Meanstructures

`~1` intercepts, `meanstructure=TRUE`, latent means, multi-group means.

- **core ✓ / api ✓ / R ✓.** Observed intercepts (Nu) and latent means (Alpha),
  `meanstructure` build flag, `effect.coding` for loadings.
- **Test:** corpus `0012` (intercept), `0026` (two-factor meanstructure),
  parity `demo_growth_linear` and `bollen_democracy_sem` (mean structures).
- **Gap:** mean-structure effect coding (`Σν == 0`) is not implemented (a
  loadings-only effect-coding contract exists). Minor.

### §9 — Multiple groups

`group=`, `group.equal=c("loadings","intercepts")`, `group.partial=`,
measurement invariance, χ²-difference tests.

- **core ✓ / api ✓ / R ✓.** Per-group parameter replication, cross-group
  equality via shared labels (configural / metric / scalar / partial
  invariance), the `c(...)` per-group modifier, LR / Satorra nested tests.
- **Test:** corpus `0016`, `0021`–`0025`, parity `hs_3factor_ls_mg_configural`
  / `hs_3factor_ls_mg_metric`, Kline Guo configural/weak/strong/partial-strong
  checked-in textbook export, `multigroup_inference_golden_test`; examples
  `holzinger_invariance.R`, `holzinger_2group_satorra_bentler.R`,
  `c_modifier_per_group.R`.
- **Gap:** no `group.equal=` / `group.partial=` convenience arguments —
  invariance is expressed with explicit shared labels. Capability is complete;
  only the shorthand is missing.

### §10 — Growth curves

`growth()`: latent intercept/slope, time-varying covariates, regressions on
the growth factors.

- **core ✓ / api ✓ / R ✓.** `model_spec(..., model_type = "growth")` applies
  lavaan's `growth()` defaults: observed intercepts fixed at 0, latent
  growth-factor means free, growth-factor covariance auto-added.
- **Test:** parity `demo_growth_linear` (hard-gated, `magmaan_aligned = true`).
- **Gap:** no `growth()` wrapper — by design; `model_type = "growth"` is the
  documented path.

### §11 — Categorical data

`ordered=`, WLSMV (lavaan's ordinal default), thresholds, polychoric
correlations, `parameterization=`, mixed continuous/ordinal.

- **core ✓ / api ✓ / R ✓.** All-ordinal and mixed continuous/ordinal DWLS/WLS,
  thresholds (`|`), response scale (`~*~`), polychoric/polyserial moments,
  delta and theta parameterizations, robust scaled tests.
- **Test:** parity `bfi_ordinal_dwls`, ordinal corpus
  (`tests/fixtures/{ordinal,mixed_ordinal}`), `ordinal_golden_test`,
  `robcat_parity_golden_test`; example `ordinal_dwls_wls.R`.
- **Gap:** CFI/TLI/RMSEA/SRMR are not yet produced for ordinal fits — the
  categorical independence-model baseline is a tracked follow-up, and
  `api::fit_measures` fails explicitly for an ordinal fit. Mixed
  continuous/ordinal NACOV/weight parity is looser than all-ordinal (both
  recorded in the backlog).

### §12 — Covariance matrix input

`sample.cov=` + `sample.nobs=` instead of raw data.

- **core ✓ / api ✓ / R ✓.** Sample statistics are magmaan's *default* fit
  input; `data::SampleStats` drives every complete-data estimator. R accepts
  `magmaan(model, data = list(S = , nobs = , mean = ))`.
- **Test:** exercised indirectly everywhere; no dedicated tutorial-shaped test.
  *Phase 2 adds one (`12_cov_input.R`).*
- **Gap:** none functionally.

### §13 — Estimators and more

ML, GLS, WLS, DWLS, ULS, DLS, PML; robust ML (MLM, MLMV, MLR, MLF, MLMVS);
robust DWLS/ULS (WLSMV, ULSMV, …); `se=`/`test=` robust and bootstrap;
`missing="ML"`; `likelihood="wishart"`.

- **core ◐ / api ◐ / R ◐.**
  - ML, GLS, WLS, DWLS, ULS: **✓** (`fit/`, `ls/`, `ordinal/` fixtures).
  - FIML (`missing="ML"`): **✓** (parity `bfi_fiml`, `bfi_fiml_5factor`;
    no textbook-corpus FIML lane yet because the canonical corpus currently
    exports no `expected/lavaan_fiml.json` snapshots).
  - Robust MLM/MLR/WLSMV: **✓ as post-fit choices** — magmaan estimates, then
    Satorra-Bentler / Yuan-Bentler scaling and sandwich SEs are explicit
    post-fit calls. This is an architecture difference from lavaan's fit-time
    `estimator="MLR"` string, not a capability gap.
  - DLS: ◐ (explicit weight-matrix builder, not an `estimator=` string).
  - **Non-robust SEs + fit measures ✓** — `standard_errors()` and
    `fit_measures()` dispatch per estimator: complete-data ML, FIML, and
    continuous ULS/GLS/WLS all yield information-inverse SEs and
    CFI/TLI/RMSEA/SRMR (FIML SRMR is computed against the saturated EM
    moments). Ordinal fit measures remain a follow-up (see §11).
- **Test:** parity LS / FIML-MLR / ordinal-robust families; `robust_test`,
  `weighted_inference_test`, `api_sem_test`; examples
  `constraints_and_satorra_bentler.R`, `observed_information_se.R`.
- **Gaps:** PML, MLF, MLMVS ✗; `likelihood="wishart"` ✗. Bootstrap
  (`se`/`test="bootstrap"`) is deliberately deferred as simulation-engine
  scope rather than tracked as active tutorial parity work.

### §14 — Mediation

Indirect effects via defined parameters: `ab := a*b`, `total := c + a*b`,
delta-method SEs, optional bootstrap SEs.

- **core ✓ / api ✓ / R ✓.** `:=` defined parameters with dependency ordering,
  chained definitions, `.pN.` plabel resolution, delta-method SEs
  (`compute_defined`).
- **Test:** corpus `0008`, `effects_test`; example `defined_parameters.R`.
- **Gap:** none for delta-method defined parameters. Bootstrap SEs for indirect
  effects are deliberately deferred with the broader simulation-engine question.

### §15 — Modification indices

`modindices(fit, sort=TRUE, minimum.value=)`: `mi`, `epc`, `sepc.all`.

- **core ✓ / api ✓ / R ✓.** The C++ core and `magmaan::api` compute
  fixed-parameter modification indices, equality-release score tests,
  absent cross-loading/covariance enumeration, and EPC — raw plus
  standardized (`sepc.lv` / `sepc.all`, via `fill_standardized_epc`) — across
  the ML / FIML / LS / ordinal paths. The R package exposes them as the
  friendly `modification_indices()` wrapper plus the
  `magmaan_core$inference_modification_indices` / `inference_score_tests`
  primitives.
- **Test:** `score/` fixtures, `score_golden_test`, `score_test` (C++ only).
- **Gaps:** none; the R binding has no dedicated R-side parity test.

### §16 — Extracting information

`coef()`, `parameterEstimates()`, `standardizedSolution()`, `fitted()`,
`resid()`/`residuals()`, `lavResiduals()`, `vcov()`, `AIC()`/`BIC()`,
`fitMeasures()`, `lavInspect()`, `lavPredict()`.

- **core ◐ / api ◐ / R ◐.**
  - `coef` (θ̂), `vcov`, `parameterEstimates` (partable + SE + z, composable),
    `fitMeasures` (`measures_fit`), `AIC`/`BIC`, `fitted` (model-implied
    moments via `model_implied`): **✓** at all three layers.
  - `standardizedSolution` (`standardize_lv` / `standardize_all`): **core ✓ /
    api ✓ / R ✓** through `magmaan_core$measures_standardize_lv()` and
    `magmaan_core$measures_standardize_all()`.
  - `residuals()` ✓ / `lavResiduals()` ✓ / `lavPredict()` ✓ (core ✓ / api ✓ /
    R ✓):
    - `residuals()` — `measures::residuals` / `api::residuals` expose the raw
      moment residuals `S − Σ̂(θ̂)` (and the mean residuals under a mean
      structure).
    - `lavResiduals()` — `measures::standardized_residuals` / `api::…` expose
      the raw and correlation-metric (Bentler) residual matrices, residual
      SE/z-statistics (`$cov.z` for the default continuous path), and the SRMR.
    - `lavPredict()` — `measures::factor_scores` / `api::factor_scores`
      compute regression (Thurstone) and Bartlett factor scores.
- **Gaps:**
  - `lavInspect()` — no single inspector; model matrices are reachable via
    `model_matrix_rep`.
  - R-package convenience gaps: the current accessors are exploratory
    `magmaan_core` primitives rather than polished friendly wrappers.

## Deferred backlog

Genuine gaps, recorded here and tracked in [docs/backlog/todo.md](../backlog/todo.md). None block
tutorial reproduction for the in-scope sections except where noted.

| Item | Tutorial § | Effort | Notes |
|---|---|---|---|
| Inequality constraints + active-bound (chi-bar-squared) inference | 7 | XL | Out of scope; fails with an explicit early error. |
| CFI/TLI/RMSEA/SRMR for ordinal | 11 | L | FIML / continuous-LS landed; ordinal needs the polychoric independence-model baseline. |
| `group.equal=` / `group.partial=` convenience args | 9 | S | Invariance works via explicit labels. |
| PML, MLF, MLMVS estimators; `likelihood="wishart"` | 13 | L | Not implemented. |
| Mean-structure effect coding (`Σν == 0`) | 8 | S | Loadings-only effect coding exists. |

## Informal tests

Each in-scope section has a runnable script under
`r-package/examples/tutorial/` that reproduces the tutorial example end-to-end
on the `magmaan()` + explicit-post-fit API and cross-checks against live
`lavaan`. See `r-package/examples/tutorial/README.md`. These are informal
verification; formal property/boundary tests are separate later work.
