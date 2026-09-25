# Convergence engineering: vision (2026-09-25)

Status: vision for discussion. Nothing here is built yet. It proposes one way
to decide magmaan's numerical defaults (starting values, optimizers, optimizer
options, scaling, and fallbacks) so that each decision rests on a shared
benchmark set, a shared judge, and a written record, instead of on a study
that invents its own problems and its own notion of success.

## Why

The evidence behind the current defaults is spread over at least ten
engineering and research studies, several design notes, and many backlog
items. Each study chose its own problems, its own acceptance rule, and its own
reference answer. Some consequences:

- `engineering/17-corpus-optimizer-recovery` and
  `engineering/18-barrier-optimizer` both asked whether PORT beats NLopt
  L-BFGS. They used different problems, and each defined success against the
  best objective among the optimizers it compared, with tolerances
  `1e-6 (1 + |f|)` and `1e-7` absolute. One changed a default, the other did
  not.
- The Ernst design is copied into engineering/03, 15 and 18. The research/47
  populations are copied into 18.
- An earlier result that L-BFGS was the most reliable optimizer was scored
  under an older acceptance rule. Nothing records which rule a result was
  scored under, so the comparison cannot be repeated.
- The vocabulary for problems whose optimum does not exist (chart poles, folds
  of the model closure, flat ridges) lives in paper notes, not in the
  repository.

## Scope

The question this machinery answers: **from its defaults, does a fit reach an
accepted local minimum of the objective it declares, on the problems users
bring, and how cheaply?**

In scope:

- convergence at all: stalls, line-search failures, budget exhaustion,
  domain errors, non-finite starts;
- stationary points that are not minima (saddles, degenerate starts);
- starting values: constructors, identification transport, structural
  initialization, scaling of starts;
- optimizer choice and options: backend, tolerances, budgets, memory,
  coordinate scaling, fallback and recovery policies;
- a benchmark set of easy and hard problems that all of the above are judged
  on.

Out of scope:

- **global optimality.** Which local minimum a fit reaches is not the
  question. A fit that reaches a different accepted local minimum than another
  configuration is recorded as a different endpoint, not as a failure.
  Multistart portfolios stay an opt-in policy (engineering/12).
- statistical properties of estimators (bias, coverage, tests). Those are
  research. Changing the estimator to remove a problem (PSD-ML, the
  determinacy barrier, the sphere chart) is a research decision, even when it
  also fixes convergence.
- published speed comparisons with lavaan, which follow
  [benchmark_plan.md](../validation/benchmark_plan.md). Cost appears here only
  as a secondary criterion among fits that converged.
- lavaan parity, which the fixtures gate.

## Precedents

The design borrows from established practice in numerical optimization and
statistical software testing:

- **NIST StRD** (nonlinear regression): datasets graded by difficulty, each
  with a far and a near start and certified answers. McCullough (1998, 1999)
  used it to grade statistical packages. Borrowed: starts are part of the
  problem, and difficulty is graded.
- **Moré, Garbow and Hillstrom (1981):** every test problem comes with starts
  `x0`, `10 x0` and `100 x0`. Borrowed: test robustness by moving the start.
- **CUTEst:** every problem carries a classification string (objective type,
  constraints, smoothness, origin, size). Borrowed: filterable problem tags.
- **COCO/BBOB:** functions grouped by property (ill-conditioned, multimodal,
  and so on), each with random instances, results reported per group.
  Borrowed: families with instances and per-class reporting.
- **Dolan and Moré (2002)** performance profiles; **Beiranvand, Hare and
  Lucet (2017)** on comparing optimizers. Borrowed: compare cost only among
  successes, fix the criteria before running, report every problem.
- **SMT-LIB:** each benchmark has a known status, and a wrong answer costs
  more than no answer. Borrowed: "no minimum is attained" is a legitimate
  answer.
- **posteriordb** (Stan): model, data, and reference results for judging
  changes to inference algorithms. **lme4 `allFit()`**: refit with every
  optimizer to tell optimizer trouble from model trouble. **OpenMx** keeps a
  suite of test models against which default optimizers were changed.
- **Architecture decision records** (Nygard 2011): one record per decision,
  with context, choice, rejected alternatives, and what supersedes it.

## The four parts

### 1. The benchmark set

A manifest of problems. A problem is a model with data, and says which
estimators it applies to. Problems come from four sources:

- **Textbook corpus** (optional mount): the source-verified continuous cases
  (304 ML and 304 GLS fits in engineering/17) and, later, the categorical
  lane. The real-model backbone.
- **Simulated families:** a population, an `N`, and a seed rule, so a family
  yields as many instances as a study needs. Existing families: the
  research/47 populations, Ernst, weak marker, high `R^2`, collinear latent
  predictors, the engineering/13 stress cells, and the bullying-model
  population.
- **Constructed problems** that isolate one mechanism: the scalar variance
  likelihood at small units (`cpp/tests/checks/nlopt_lbfgs_domain.c`) and the
  equivalent-spelling pair from engineering/17's start-semantics probe.
- **The existing benchmark cases** in `benchmarks/cases.yml`, which keep their
  timing role.

**Transforms** turn one problem into several that must give the same answer,
as COCO's instances do. A configuration should be invariant to each:

- rescaling observed variables (the bullying model at three times its units
  failed on 40 of 40 samples in the 2026-09-23 reproduction);
- reordering variables;
- identification: marker choice, std.lv, effect coding, the sphere chart;
- equivalent spellings: a latent regression written as a higher-order loading
  (engineering/17 found the simple start gives 0 to one and 0.7 to the other);
- moving the start: the constructor's start, a rescaled start, a perturbed
  start, and a known solution.

**Classes** are tags, and a class belongs to a problem together with an
estimator and a chart, not to the data alone. The same dataset can be a fold
for ML and a regular problem for the determinacy barrier. Proposed classes,
each with an operational definition:

| Class | Definition | Current example |
|---|---|---|
| regular | an interior local minimum that every chart reaches | research/47 f2_r90 |
| near-face | an interior minimum next to an admissibility face, ill-conditioned | one-factor Heywood-prone cells at `N = 50` |
| face | the constrained minimum sits on a face (PSD-ML) | PSD-ML on improper draws |
| degenerate start | the default start sits where a direction has no first-order effect, and the fit ends at a non-minimum | Little phantom model (zero scale paths) |
| scale-fragile | fails under a change of units | Little bullying model, times 3 |
| pole | no minimum in this chart, but the implied covariance converges to a model point another chart attains | part of Ernst marker at `N = 10` |
| fold or ridge | no minimum in any chart: the implied covariance converges to a boundary point of the model closure | Ernst std.lv with `R^2 -> 1` |
| flat | the minimum is not unique | Little ch8 fig3a |
| structured | equality or nonlinear constraints, several groups, missing data, large parameter count | corpus invariance models |

Poles and folds both send a parameter to infinity. They differ in where the
implied covariance goes, which does not depend on the chart. Fitting the same
data in several charts, and with PSD-ML, tells them apart.

**Tiers** of the set:

- **canary:** about 30 problems covering every class, well under a minute,
  run by a `just` recipe. Protects decisions against regressions.
- **standard:** the decision set, a few thousand fits, minutes to an hour on
  four local cores.
- **stress:** full families across `N` and replications, tens of thousands of
  fits, run on Modal.

### 2. Configurations

A configuration is everything that decides how a fit is computed, for a given
estimator:

> start constructor, identification transport, structural start pass,
> chart, optimizer, optimizer options, coordinate scaling, fallback or
> recovery policy.

Start questions and optimizer questions are then the same kind of experiment:
a comparison of configurations on the same problems. Options belong to the
configuration because they do not mean the same thing across algorithms
(engineering/17). The library's current default is always one of the
configurations, read from the library, so the benchmark tests what users get.

### 3. The judge

The judge is the library's fit verdict (`estimate::fit_verdict`, R
`fit$converged`; [terminal-audit.md](terminal-audit.md)), recorded with the
commit it ran at. No study defines its own acceptance tolerance.

Each fit gets one outcome:

- **converged:** the verdict passes.
- **refused:** the verdict fails. The cause is recorded: stall, budget,
  nonpositive curvature, a gradient too large, and so on.
- **error:** non-finite objective at the start, unsupported feature, crash.
- **timeout.**

Whether a failure counts against a configuration depends on the problem:

- A problem is **attainable** when some configuration reaches an accepted
  local minimum. A witness can be a known solution used as a start (the
  corpus stores verified solutions). A failure on an attainable problem
  counts against the configuration.
- On a problem with no attained minimum (pole, fold, ridge), refusing is the
  correct answer. Acceptance there is an escape along a divergent path and is
  flagged. This is the known weakness of the verdict recorded in
  [speculative.md](../backlog/speculative.md) (runaway estimates and
  nonattainment).

Every failure gets a cause, from a fixed attribution rerun: the same
optimizer from the witness start, the other optimizers from the same start,
and the same configuration under the transforms. The cause is one of
optimizer, start, options (budget or tolerance), problem (no configuration
succeeds), or unexplained. engineering/17's crossed starts and optimizers are
this protocol done by hand.

**Every iterative estimator uses the Newton metric (decided and landed
2026-09-25, except two-level; CatML stays first-order).** Until then the
Newton check, which also rejects stationary points with nonpositive curvature,
was part of the default verdict only for complete-data ML
([newton-verdict-plan.md](newton-verdict-plan.md), engineering/19).
engineering/17 showed the cost: of the twelve Little models where ML's
check rejects a PORT endpoint for nonpositive curvature, nine are accepted
under GLS at a worse objective. A judge that differs by estimator makes
convergence rates incomparable across estimators, so a uniform judge is a
prerequisite for everything else here.

The principle that makes one budget serve every estimator: take the Newton
step of the fitted objective, `delta = H^{-1} g` with `H` that objective's
Hessian (whose positive definiteness is the curvature check), and measure it
in standard-error units, `d^2 = N delta' V^{-1} delta`, where `V` is the
estimator's asymptotic covariance. For likelihood estimators (ML, FIML,
two-level ML) and for normal-theory GLS, `V^{-1}` is the information and `H`
estimates it, so `d` is exactly the current ML check and the `d <= .01` budget
carries over. For least-squares estimators (ULS, DWLS, WLS, ordinal) `V` is
the sandwich, which makes `d` free of units. For penalized fits (the
barriers), `H` includes the penalty and `V` is the ML information.

### 4. The defaults register

One file with an entry per numerical default. Each entry records:

- the current value and the paths it covers;
- the alternatives considered, including rejected ones;
- the acceptance criteria, written before the run;
- the evidence: benchmark set version, slice, judge commit, and the study
  that ran it;
- status: settled, provisional (a default without recorded evidence), or open
  (evidence says it should change);
- what would reopen it;
- history.

This formalizes what the experiments README already asks of engineering
studies. The first entries, from existing evidence:

| Default | Current value | Status |
|---|---|---|
| acceptance rule | fit verdict, Newton check for complete-data ML | settled 2026-09-24 |
| complete-data ML optimizer | NLopt L-BFGS | open: PORT gained 48 corpus cases and lost none (engineering/17) |
| complete-data ML controls | `ftol_rel` 1e-12, `xtol_rel` 1e-10, 5000 evaluations | settled 2026-09-22 (interior Newton audit) |
| GLS, ULS, WLS optimizer | NLopt L-BFGS | open: PORT gains, loses Kline's Worland step 2b under GLS |
| FIML optimizer | L-BFGS with SLSQP fallback | provisional |
| PSD-constrained fits | SLSQP; PSD as an explicit refit after an ordinary fit fails its audit | settled (engineering/11, 12, 13, research/42, 43) |
| barrier fitters | PORT | settled 2026-09-25 (engineering/18) |
| IRLS | PORT NLS | settled, IRLS itself no gain (engineering/03) |
| two-level, ML2S, SAM, ordinal optimizers | NLopt L-BFGS | provisional |
| ML starts | FABIN3, transported from std.lv to the marker chart | settled for measurement, open for structural paths |
| other starts | native FABIN3 | provisional |
| structural start pass | none | open: 10 of 12 ML curvature failures recovered by nonzero latent-path starts (engineering/17) |
| sphere start | canonical | settled (engineering/14) |
| fallback trigger | backend status, not the verdict | open |
| rejected | covariance continuation (engineering/02), Fisher scoring or IRLS as default (engineering/03), multistart as default (engineering/12) | settled |

## Rules for a decision study

1. Write the register entry's criteria, the slice, and the configurations
   before running.
2. Report by class. Never pool across classes: a configuration can be the
   fastest on regular problems and the worst near faces, and a pooled rate
   only reflects the mix.
3. The primary measure is the convergence rate on attainable problems.
   Secondary: cost among converged fits (objective evaluations first, time
   second, as performance profiles), and the invariance rate under transforms.
4. A change of default lists every attainable problem the old default solved
   and the new one does not. Losses are allowed if they are named.
5. The canary passes before and after the change.
6. Different accepted endpoints across configurations are reported as a
   diagnostic, not scored.

## Where it lives

Experiments may not depend on each other, so the shared parts sit below them:

- benchmark set, transforms, judge wrapper, and harness:
  `benchmarks/convergence/`, next to the existing case manifest;
- the defaults register: `project/validation/defaults-register.md`, next to
  the oracle-defects ledger;
- decision studies: `experiments/engineering/`, now thin. A study picks a
  slice and configurations, runs, reports, and updates the register. Once its
  decision is in the register and its problems are in the set, the study is
  archived.

## Plan

### Build now (v1)

0. The Newton metric in the default verdict of every iterative estimator,
   as above, with analytic Hessians. Plan:
   [newton-verdict-plan.md](newton-verdict-plan.md).
1. The register, filled from existing evidence as in the table above, with
   provisional and open entries marked honestly.
2. Benchmark set v0 from problems that already exist: the corpus cases of
   engineering/17, the research/47, Ernst, weak-marker, high-`R^2`,
   collinear-predictor and stress families, the constructed problems, and the
   benchmark cases. Classes are tagged from existing evidence, and left
   untagged where there is none.
3. Transforms: unit rescaling, variable reordering, identification, equivalent
   spellings, start moves.
4. The harness: run named configurations over a slice, write one outcome
   record per fit, run the attribution rerun on failures, and render a
   per-class report.
5. The canary tier with a `just` recipe (advisory, like `cpp/tests/checks`).
6. First uses, in order:
   - rerun engineering/17 and 18 on the set; they should agree;
   - decide the complete-data ML optimizer (open);
   - validate the structural start pass that the backlog already prioritizes
     (equivalent spellings, units, identification, the full corpus).

### Deferred, methods known

- Performance and data profiles in the report template.
- Safeguarded backtracking or domain-aware line search for NLopt L-BFGS
  (the high-priority backlog item on domain recovery).
- A fallback that triggers on the verdict, not on backend status.
- Multistart as an opt-in policy.
- The ordinal, two-level and ML2S paths in the set.

### Open research questions

- Deciding automatically whether a failed fit has no attained minimum, and
  whether the cause is a pole or a fold. The verdict can pass along divergent
  paths (speculative backlog).
- Predicting a problem's class cheaply before fitting, so that a policy can
  choose a configuration per problem.
- Start constructors that are equivariant under units and equivalent
  spellings by construction.
- Whether one optimizer can serve every estimator, or defaults must differ by
  estimator.

## Questions to settle before building

- Whether the register is one file or one file per entry.
- How strict "attainable" must be: a witness solution is enough for v1, but
  corpus solutions come from other software and are starts, not proofs.
