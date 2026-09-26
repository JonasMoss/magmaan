# Numerical defaults: staged promotion plan (2026-09-26)

Status: plan. Stage 1 is ready to execute; later stages need evidence first.
It applies [convergence-engineering.md](convergence-engineering.md) to the
three default choices engineering/17 studied: the start constructor, the
optimizer, and the optimizer coordinates. Coordinates are already promoted
(information coordinates, 2026-09-26); this plan covers the start and the
optimizer, route by route.

## Rules every stage follows

- **The judge** is the library verdict (`fit_verdict`, R `fit$converged`), the
  Newton check for every iterative estimator. A reference engine is judged by
  evaluating the same verdict at its estimate (`evaluate_at`), never by its own
  flag.
- **Scoring.** Primary: certified local minimum. Secondary: certified at the
  best known objective of the problem; a certified endpoint above it is
  another local minimum and is reported with its gap. The secondary rate ranks
  start policies; it does not by itself block a default.
- **Problem classes.** Main comparisons use standard problems only. Nonlinear
  equality constraints are their own lane (today one case, Mplus ex6.17).
- **Losses are named.** A change of default lists every problem the old
  default certified (and every best-known optimum it reached) that the new one
  does not. Allowed only with an explanation.
- **Out of sample.** The layered start was developed against the corpus, so a
  promotion needs at least one problem set it was not tuned on.
- **Record.** Each decision gets an entry in the defaults register
  (`project/validation/defaults-register.md`, created in stage 0) with the
  slice, the judge commit, the criteria written before the run, the losses, and
  what would reopen it.

## Evidence so far (engineering/17, 2026-09-26)

Standard pairs: 303 ML and 303 GLS (textbook corpus v3.2.0, continuous,
complete data). All magmaan runs in information coordinates.

| Configuration | ML certified / best | GLS certified / best |
|---|---|---|
| today's default: current start, L-BFGS | 283 / 282 | 283 / 283 |
| layered start, L-BFGS | 303 / 300 | 303 / 303 |
| layered start, PORT | 303 / 300 | 303 / 303 |
| lavaan defaults (same judge) | 294 / 292 | 297 / 295 |
| lavaan from the layered start | 303 / 300 | 303 / 303 |

- Layered start with PORT against today's default: 40 certified and 38
  best-known gained over both estimators, none lost.
- From the layered start, PORT and L-BFGS reach identical outcomes on every
  pair; PORT needs slightly fewer evaluations (median 24 against 25.5).
- Every observed variable multiplied by a power of ten between 0.01 and 100,
  on the 120 ML and 119 GLS unit-invariant pairs: the layered start certifies
  119 and 119 with either optimizer; today's start 101 to 103 and 95 to 98;
  lavaan's defaults 99 and 86 (its flag claims 19 and 26 more).
- The three ML pairs that miss the best known optimum (Little's two Chapter 8
  ALT models, Kline's Worland step 2a) are certified minima that lavaan from
  the layered start also settles in.
- The current start is not unit-equivariant (its start objective moves on 148
  of 239 invariant pairs under rescaling); the layered start moves on 32.

Report: `experiments/engineering/17-corpus-optimizer-recovery/report.qmd`,
sections "How fits are scored" and "Optimizer coordinates"; summaries in its
`results/full/scaling_*.csv`.

## Stage 0: register and bench seed

- Create `project/validation/defaults-register.md` from the table in
  convergence-engineering.md, plus the entries settled on 2026-09-26 (scoring,
  optimizer coordinates). Mark every entry settled, provisional or open.
- Lift the engineering/17 prototypes needed for the stages below into
  `benchmarks/convergence/`: the corpus slice with problem classes, the
  configuration runner, the unit-rescaling transform with its invariance check,
  the reference-engine runner under the shared judge, and the scoring summary.
  Keep engineering/17 as the study that produced the first entries.

Evidence expected: none new. Done when the stage 1 runs can be driven from
the bench rather than from experiment scripts.

## Stage 1: complete-data ML and GLS (ready)

**Choices.**

| Choice | Candidates | Recommendation |
|---|---|---|
| start | current (auto-transported FABIN3 for ML, native FABIN3 for GLS); layered | layered |
| optimizer | NLopt L-BFGS; PORT; L-BFGS with SLSQP fallback | PORT |
| nonlinear equalities | error; route to SLSQP (IPOPT when built) and report it | route |
| compatibility starts | keep `simple`, `fabin3`, `scaled-fabin` selectable | keep, for lavaan parity and studies |
| namespace | layered start in `estimate::frontier`; move to core | move to core with the promotion |

PORT over L-BFGS rests on robustness, not on the corpus counts: its trust
region backs off from infeasible trial points, while NLopt's L-BFGS aborts on
them and can run along flat ridges (see the backlog's L-BFGS first-step item).

**Evidence still required, before the flip.**

1. Simulated families the layered start was not developed on: the research/47
   populations, Ernst, weak marker, high `R^2`, collinear latent predictors,
   the engineering/13 stress cells and the bullying population, at a small and
   a moderate `N`, a handful of replications each, native units and one
   rescaled transform. Expected: the candidate (layered, PORT) certifies at
   least as many fits as today's default in every family; any family with a
   net loss is named and explained before promotion. Poles and folds, where no
   minimum exists, are scored as refusals, not failures.
2. The equivalent-spelling and identification transforms on the corpus
   (marker, std.lv, effect coding; a latent regression written as a
   higher-order loading). Expected: the same certified outcome and objective
   across forms, or a named exception.
3. After the flip, a confirmation corpus run that reads the defaults from the
   library (no explicit arms). Expected: it reproduces the layered/PORT row
   above.

**Implementation.**

- Move `layered_start_values` / `layered_start_report` to core.
- ML: `ml_start_values` and `api::ml()`'s `ml_starts()` use the layered start;
  R `fit_ml` default `start` becomes `"layered"`. GLS: R `fit_gls` and
  `api::gls()` likewise.
- Default backend `Backend::Port` for `fit_ml` (C++ default argument in
  `fit.hpp`), `fit_gls`, `api::ml()` and `api::gls()`, and R's
  `backend_from_optimizer_arg` for these entries.
- Nonlinear equalities: when the selected backend cannot take them, switch to
  SLSQP and record the backend used in the fit.
- Update optimizer-controls.md, the roadmap, the register entries, and the
  tests that pin the old defaults. The ordinary-user package inherits the
  change through `api::ml()`.

Done when: the out-of-sample families show no unexplained loss, the
confirmation run matches, both test suites and the R checks pass, and the
register entries are settled.

## Stage 2: continuous least squares (ULS, DWLS, WLS) and pairwise GLS

Choices as in stage 1. No corpus evidence exists yet: engineering/17 fit only
ML and GLS. Evidence needed: the corpus under ULS, DWLS and WLS (raw-data cases
only, since DWLS and WLS need fourth moments), with the same arms, judge and
rescaling; the stage 1 families. Expected: the same ordering as ML/GLS; ULS is
not scale-invariant, so its rescaled lane is descriptive only.

## Stage 3: FIML

Choices: start (native FABIN3 on the start moments; layered on the same
moments); optimizer (L-BFGS with SLSQP fallback, today; PORT, which needs
wiring into the FIML dispatch); stopping controls (generic, today; complete-data
ML's profile, which the Newton-verdict work already recommends).
Evidence so far: 13 single-group continuous FIML corpus cases, information
coordinates certify 11 native and 8 rescaled (raw coordinates: 12 and 0); the
native miss is a stopping-precision near miss under the generic controls.
Evidence needed: the multi-group FIML cases, simulated MCAR and MAR missingness
over the stage 1 families, and the layered start on FIML start moments.
Expected: ML's stopping profile removes the near misses; PORT or the layered
start helps most in rescaled units.

## Stage 4: ordinal, mixed ordinal, CatML

Blocked on the categorical model semantics in the backlog (theta residual
variances, delta scale factors, means entering the thresholds, covariates):
until those are fixed, convergence counts measure a different model. Then:
wire the coordinate layer (mixed models keep continuous columns in data
units), design a start for thresholds and polychoric structure (the layered
start does not apply as is), and decide the optimizer on the 34 WLSMV corpus
cases and simulated ordinal families.

## Stage 5: the remaining routes

Two-level ML, SAM, SNLLS (outer loading block), IRLS inner solves, the
fitted-weight loop, RBM and the pairwise likelihood: wire the coordinate layer
where it applies (each with a unit-rescaling test, as `coordinates_test.cpp`
does for ML), then decide starts and optimizers per route. PSD ML and the
sphere route keep their own scaling unless a study shows otherwise.

## Cross-cutting items (backlog)

- L-BFGS first-step domain aborts and flat-ridge runaways; a fallback that
  triggers on the verdict instead of the backend status.
- Unit equivariance of the layered start on the 32 corpus pairs where its start
  objective moves.
- Score tests that admit unidentified candidates by rounding.
- Multistart as an opt-in policy for the second-optimum cases (speculative).
