# Benchmark Plan

This is the benchmark design: workload contracts, attribution, datasets, and
public reporting. The 2026-09-20 proposal below is the next implementation
slice; the broader coverage inventory follows it. Execution tasks live in
[the active backlog](../backlog/todo.md), not in a separate roadmap.

The existing case registry and R harness provide lavaan-backed smoke comparisons.
The newer [C++ timing harness](../../benchmarks/timing/README.md) provides modular
complete-data measurements. The first matched-workload slice is now an experiment, with a shared R timer,
raw/prepared/post-fit boundaries, native lavaan pEBA-4 checks, and an experiment-local
report. Publication and controlled evaluator/backend attribution remain open;
no existing benchmark files have been retired.

## Proposed public speed report (2026-09-20)

### Scope and questions

Start with complete-data continuous ML. Ordinal, FIML, and SNLLS get separate
extensions with their own setup and statistical contracts. The initial report
should answer three questions independently:

1. How long does the same requested analysis take through each R interface?
2. Where does each implementation spend that time?
3. How much of the difference comes from numerical evaluations, optimizer work,
   and interface/setup/finalization work?

Do not call the third quantity a pure language effect. R matrix operations
already call compiled numerical libraries, and R's `nlminb` uses compiled PORT
routines ([R documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/nlminb.html)).
The defensible attribution is to these implementations, algorithms, data
representations, and call boundaries. Compiler/BLAS differences remain part of
the recorded environment.

### What the inspection established

- `benchmarks/timing/` already covers parsing, model construction, sample
  statistics, starts, constraints, evaluator/objective construction, implied
  moments/Jacobians, objective/gradient, L-BFGS, and conventional post-fit work.
  Its `pipeline_total` and `staged_fit_ml` are **sums of stage medians**, not
  directly measured complete analyses. Its per-evaluation probes use theta-hat;
  SB, pEBA-4, and robust Wald intervals are missing from that stage catalog.
- The talk's `benchmark_talk_speed.R` distinguishes prepared and raw-input
  workloads, but a call supplying lavaan slots still runs through lavaan's fit
  orchestration. It is not an optimizer-only measurement.
- The installed lavaan 0.7-2 has native FMG tests. A local HS three-factor CFA
  probe with `se="robust.sem"`, `test=c("satorra.bentler","peba4_ml")`, and
  `baseline=FALSE` returned both requested tests and a converged fit. Prefer
  this direct comparator; the talk's older lavaan + semTests path is historical.
- That installation exposes `lav_step*` functions, `lav_model_objective`,
  `lav_model_grad`, and an object `timing` slot. These are candidates for a
  version-pinned adapter, not stable cross-version APIs. The old talk's slot
  argument spelling also differs from the current formal arguments.
- `benchmarks/inference_reuse.R` and `score_primitives.R` cover useful staged
  inference, but use older timing loops. They should contribute workloads to
  the common runner rather than become additional public timing authorities.

### Matched statistical outputs

Use named workload contracts, each returning a small common result schema.
Run every workload directly from syntax and raw data, and separately from
prepared model/data inputs where reuse is meaningful.

| Workload | Required outputs |
|---|---|
| `fit` | estimates, normalized objective, convergence/acceptance diagnostics |
| `wald_nt` | fit + conventional covariance, SEs, 95% normal Wald intervals |
| `sb_wald` | fit + SB-scaled ML GOF (statistic, df, scaling, p) + robust sandwich covariance and 95% Wald intervals |
| `peba4_wald` | fit + ML-based pEBA-4 GOF + the same robust Wald outputs |
| `sb_peba4_wald` | fit + both GOF calibrations + robust Wald outputs, reusing shared inference ingredients |

Use explicit `peba4_ml`, not suffixless `peba4` or `fmg`: the reference base
statistic must be identical. Score/RLS GOF is a different workload and can be
added separately. Restrict the pEBA-4 headline cells to df >= 4. Match covariance
normalization, mean structure, fixed-x treatment, information/bread choice,
empirical meat, finite-sample factors, and interval convention. In particular,
SB GOF and robust covariance are distinct outputs; requesting one does not
stand in for the other. Lavaan's MLM pairs robust SEs with SB GOF
([lavaan estimator documentation](https://lavaan.ugent.be/tutorial/est.html)).

No baseline model, CFI/TLI/RMSEA, standardized estimates, or printing is required
by these contracts. Disable unrequested work where supported. If a requested
output forces additional reference work, retain it in the interface total and
identify it in the stage account. Add fit indices later as an explicit workload.
Native defaults may be shown in a separate convenience panel, never mixed with
matched-output ratios.

### Timing boundaries and setup decomposition

Record independently measured boundaries:

- **Raw to report:** syntax/raw observations to all contracted outputs.
- **Prepared to report:** retained model and sample/data structures to all
  outputs; charge starts, optimization, finalization, and any inference setup
  that is not explicitly retained.
- **Fit to inference:** an estimate-only fit plus the retained fitting data to
  covariance, intervals, and GOF. Do not start from a fit already carrying those
  results. Include first-use/lazy computation and state exactly what is cached.
- **Repeated calibration:** already prepared quadratic/spectrum to p-values;
  useful as a marginal cost, never presented as full GOF cost.

Within a traced call, use mutually exclusive top-level stages: input/options
and parsing; data conversion/grouping; sample statistics; partable/model
construction; starts/bounds/constraints; evaluator/cache/problem preparation;
optimization; terminal validation and fit-object assembly; inference; requested
output extraction. Keep H1/baseline work separate if it is actually performed.

Inference sub-stages should identify fitted geometry/Jacobian, bread,
casewise contributions or empirical moment covariance, sandwich assembly,
GOF projection, trace or eigenspectrum, calibration, and interval formatting.
Shared inputs are charged once in `sb_peba4_wald`; SB-only should not be forced
to compute a pEBA spectrum. Measure the complete bundle as well as isolated
primitives; overlapping public functions are not additive stages.

Use native timing/trace information first to identify boundaries, then a
benchmark-local, version-pinned adapter for finer probes. Check installed
function signatures, inputs, and results before using an adapter. Never patch
the user's installed lavaan. Instrumentation runs and uninstrumented headline
runs are separate; quantify instrumentation perturbation and retain an
unattributed residual. Sampling profiles can locate work but do not price
sub-millisecond stages reliably.

Only stack exclusive durations from the **same call**. Keep nested callback
measurements nested under optimization. A whole-call time minus the sum of
independently measured medians is not an identified object-construction cost;
it also reflects caching, allocation, sharing, and measurement differences.
Retain such reconstructions as diagnostics with explicit labels. The existing
C++ `pipeline_total` should become `pipeline_sum_of_medians` when its consumers
are migrated, with a new direct timer for the real pipeline.

### Separating evaluation cost from optimizer behavior

“Cost per iteration” is insufficient: an iteration can make multiple function,
gradient, or joint calls. Report iterations when available, actual callback
counts, and callback cost separately.

**Fixed-input evaluator experiment.** Export a canonical parameter map and a
bank of valid points: common start, early/middle optimization points, and final
fit, drawn from both trajectories. Replay identical points through both
implementations. Time implied moments, full moment Jacobian, scalar objective,
gradient, and combined objective/gradient where available. A gradient path
that avoids materializing the full Jacobian remains the production path; do
not force it through an artificial Jacobian to make the implementations alike.
Count joint callbacks distinctly to avoid double charging shared work.
Validate values/derivatives after coordinate and objective-scale alignment.
Measure cache-hit repetition separately from changing-theta replay.

**Common-driver experiment.** Drive both evaluators with the same R `nlminb`,
analytic gradients, common coordinates, starts, bounds, scaling, and controls.
The C++ evaluator needs a small benchmark-only bridge with a retained native
context, outside the public R API. Context creation is a separate setup cost.
Both callbacks cross an R function boundary; the C++ callback also crosses
`.Call`. Record bridge-only probes, but do not subtract a no-op timing and
claim a pure language speedup. Batch replay inside C++ additionally gives
native evaluation cost without the per-call R bridge.

**Native-driver experiment.** Hold the C++ evaluator fixed and compare the
R-driven callback route, native PORT, and native NLopt L-BFGS. R `nlminb` versus
native PORT is not automatically an identical-driver experiment: audit PORT
variants, scaling, bounds, tolerances, and stopping behavior. If they cannot be
matched, label the result a backend/adapter comparison. Differences between
PORT and L-BFGS include algorithmic work, not merely dispatch overhead.

Finally compare production defaults separately, including native start
heuristics. Keep common-start and native-start arms distinct. For constrained
models, use an explicit common reduced-coordinate map or exclude them from
the first common-driver study. Use the same independent terminal objective and
stationarity screen for both engines; backend success alone does not establish
comparable accuracy. Report callback counts, convergence/admissibility failures,
and objective discrepancies even when the corresponding speed ratio is absent.
There is no generally valid additive or multiplicative decomposition of the
headline speedup from these interventions.

### Cases, repetitions, and correctness

Pilot: HS three-factor CFA and PoliticalDemocracy SEM, with paired raw input,
prepared input, and inference bundles. Revalidate both first; the old zoo
README lists historical optimizer failures that should not silently exclude
hard cases or be treated as current results.

Then use the existing synthetic catalog for p = 6/12/24/48/96 where supported,
with a limited N ladder and several fixed datasets per cell. Start with marker
identification; show other charts as a separate controlled study. Record p,
latent dimension, free/reduced parameter count, group count, df, and model
structure. Include a bounded multi-group/equality validation extension before
making broader claims. Normal and finite-moment nonnormal draws exercise the
robust inference route; repeat timings on a dataset separately from drawing
new datasets.

For an N-cost isolation experiment, hold moments and theta fixed or replicate
rows with the chosen normalization documented. Redrawing at each N can change
conditioning and optimizer work, so differing fit time is not automatically
a timing defect. Empirical robust inference requires the raw-data contribution
work even when complete-data estimation consumes only sample moments.

Use `timing/timing.hpp` for native probes and one shared `r/timing.R` for R
workloads: warmup, adaptive batching, balanced arm order, raw batch samples,
and explicit GC policy. Start with at least seven paired batches in each of
three fresh process sessions; extend unstable cells rather than multiplying
the entire design. Calibrate R batches to roughly 100 ms rather than timing
single sub-millisecond fits. Retain natural GC inside whole workflows; report
allocation separately. Launch with one BLAS/OpenMP thread before loading R.
Keep compilation, data generation/download, validation, profiling, and report
rendering outside the timing run. Package startup is a separate optional cost.

Isolate dissimilar stages in fresh/reset workers for absolute stage costs.
Rotate comparable arms within a stage, not every stage against every other
stage: the existing harness documents allocator interference from such a
rotation. Fix and report cache-reset semantics. Save process/run/batch IDs,
call counts, clock/batch duration, arm order, seeds, options, input hashes,
source/library hashes, CPU, compiler flags, BLAS, and package versions.

Predeclare mixed absolute/relative tolerances per quantity using existing
parity contracts. Check named estimates, implied moments, normalized objective,
stationarity, covariance/SEs, interval endpoints, GOF statistic/df/scaling, and
p-values. Compare spectrum/trace ingredients when available. Keep failures and
unsupported cells in the report denominator. Validate adapters outside timing,
and spot-check timed results. A checksum prevents elimination; it is not a
correctness test. Publish paired ratios/distributions and session variability;
any uncertainty interval from timing repeats describes timing uncertainty,
not sampling uncertainty across statistical datasets.

### Public artifact and ownership

Proposed entry point: `benchmarks/report/speed.qmd`, rendering a standalone HTML
report from **its own** `benchmarks/report/results/<run-id>/`. A runner writes
canonical inputs, raw samples, validation, counts, metadata, and summaries
there; rendering performs no benchmarking. Existing ignored exploratory
`benchmarks/results/` remains scratch. Track a small redistributable frozen
report snapshot and provide the full run bundle with checksums for reproduction.
The report must not source scripts or read results from a paper, experiment,
or talk. Extract reusable harness code downward into `benchmarks/`; retain
historical talk assets unchanged.

The public report needs four views:

1. Absolute time and paired ratio for each complete workload, with correctness
   and failure counts immediately visible.
2. Exclusive setup/fit/inference stage account and direct total, with unresolved
   time visible rather than assigned to “R overhead”.
3. Callback cost versus callback count, followed by controlled backend results.
4. Incremental SB, pEBA-4, and Wald costs, plus the measured shared bundle.

Put environment, precise contracts, reproducible commands, and full numerical
checks behind those views. Avoid a single pooled “X times faster” claim.
A first local report may be marked machine-specific; satisfy the independent
reproduction criterion below before promoting a general public headline.

### Cleanup proposal

Do cleanup after replacement workload smoke checks, preserving reproduction
paths. This design does not delete or move benchmark code.

| Existing material | Proposed disposition |
|---|---|
| `timing/`, case registry, `cases/`, R data/reference helpers | Keep as reusable infrastructure; extend the timer/output contracts |
| `r/run_benchmark.R` | Keep the zoo smoke entry point; use the common timer and separate public workload runner |
| `score_primitives.R`, `inference_reuse.R`, `r/bench_mi_lrt.R` | Migrate useful workloads to shared runner; then retire duplicate timing loops, retaining thin entry points only if consumed |
| Top-level `*_bench.cpp` and memory-profiler helpers | Inventory distinct workload/consumer first; migrate active diagnostics under `micro/` and memory tools under `memory/`, preserving CMake target names |
| `snlls-current/`, `snlls-handoff/`, `snlls-handoff-current/` | Done 2026-09-24: moved out of this repository into the private SNLLS handoff repository |
| Talk timing scripts/data | Moved out with the talk (2026-09-24); promote reusable logic into benchmarks without importing talk dependencies |
| Old cpp/build/install commands and stale “R-first scaffold only” text | Update alongside the replacement entry points and a clear active/historical index |

The SNLLS packaging scripts embed directory paths, so a directory move is a
functional change, not tidying. Audit tracked and local artifact references,
update consumers, and smoke packaging before any such move. Delete only
superseded duplicate implementations after their distinct checks/outputs are
accounted for. Keep historical conclusions with their source revisions; never
mix those numbers into a new report.

### Implementation order

1. Shared R timing/output schema and two-case matched-workload pilot, including
   direct totals and correctness gates for native lavaan pEBA-4.
2. Lavaan stage adapter, comparable magmaan stage account, and direct inference
   bundles; quantify instrumentation and closure/cache effects.
3. Fixed-point/trajectory replay and common-driver bridge, then native backend
   and native-start comparisons.
4. Consolidate benchmark entry points and retire duplicate loops with consumer
   checks; populate and render the public report from a frozen run bundle.
5. Broaden cases and reproduce elsewhere before promoting headline claims.

## Goals

magmaan should eventually have three benchmark layers with different audiences:

1. Public lavaan comparison: honest end-to-end timings for supported workflows
   where magmaan and lavaan compute the same statistical outputs.
2. Internal regression suite: repeatable timings and diagnostics that make
   optimizer, sample-statistic, matrix-representation, and inference changes
   measurable.
3. Backend comparison suite: magmaan-only comparisons of NLopt, PORT, Ceres,
   and profiled/SNLLS variants where those backends are statistically
   equivalent and semantically appropriate.

The public claim should be narrow and defensible: speed for supported linear
SEM workflows under lavaan-equivalent results, not a general claim that every
lavaan call is faster or that unsupported features are covered.

## Benchmark Principles

- Compare workloads, not function names. A lavaan call often computes or stores
  information that a magmaan primitive may not need. Public comparisons must
  define the requested statistical report first, then run both packages until
  that report is complete.
- Correctness gates timing. Every timed row must first pass tolerances for
  estimates and requested post-fit statistics against lavaan, or against an
  already accepted magmaan reference for magmaan-only backend comparisons.
- Keep point estimation separate from reporting. Report timings for
  point-estimate-only, standard inference, robust inference, fit measures, and
  standardized/defined-parameter extras separately.
- Pin versions and options. Record magmaan commit, lavaan version, R version,
  compiler, BLAS/LAPACK, CPU, OS, CMake preset, package install flags, and all
  lavaan/magmaan options.
- Do not hide setup cost unless the label says so. Public end-to-end runs
  should include parse/lavaanify, data preparation, sample statistics,
  optimization, and requested reporting. Internal component benchmarks can time
  narrower primitives.
- Use real data for headline and representative suites. Simulated or resampled
  data can be used for scaling studies, but should be clearly labeled as stress
  tests rather than headline real-data evidence.
- Keep benchmark artifacts reproducible. Data downloads, cleaning, model
  syntax, package versions, and result summaries should be scripted and cached.

## Matching Statistical Workloads

Public lavaan comparisons should be grouped by requested output level. Each
level must have a small compatibility adapter for both packages that returns a
common result object and validates equality before timing.

### Workload Levels

- Parse/data/model setup: parse syntax, lavaanify/model-spec creation, select
  variables, handle groups, construct sample statistics or raw-data structures.
  This is useful internally, but should not be the headline comparison unless
  clearly labeled.
- Estimate only: fit the model and return estimates, objective value, degrees
  of freedom, optimizer status, and iteration count when available. Lavaan
  should be called with options that avoid unnecessary SE/test/baseline work
  where possible, for example `se = "none"` and `test = "none"` when those
  options are valid for the estimator.
- Standard report: estimate plus conventional SEs, z tests, chi-square, df,
  p-value, and fit measures that require H1 and baseline models.
- Robust report: estimate plus robust SEs and scaled/shifted test statistics.
  The exact requested robust report must be named, for example ML with MLR or
  MLM, ordinal DWLS with sandwich SE plus SB-family tests, or FIML MLR.
- Full methods report: estimate, standard or robust report, fit measures,
  standardized solution, defined parameters, Wald/nested tests, and any other
  public-facing result claimed in the comparison.

### Fairness Rules

- If magmaan reports robust SEs, lavaan must be timed until the matching robust
  SEs are materialized. If lavaan computes fit measures by default but the
  workload is estimate-only, disable them where lavaan allows it.
- If lavaan computes H1/baseline models for fit indices, magmaan must compute
  equivalent H1/baseline ingredients before claiming a fit-measures timing.
- If magmaan uses sample moments while lavaan is given raw data, that benchmark
  is only fair if the workload explicitly excludes sample-stat construction for
  both packages or includes it for both packages.
- If lavaan exposes a statistical result only as part of a larger object, that
  extra object-building cost is part of lavaan's real user-facing workflow, but
  the label must say which result forced it.
- Do not compare a magmaan C++ primitive directly with a lavaan R front-end in
  public claims. Keep such timings in the internal suite.
- For categorical LS, distinguish sample-stat construction from fitting.
  Polychoric/polyserial/NACOV construction can dominate the workload and should
  be reported separately and together.
- For FIML, compare raw-data workflows and track missingness pattern counts.
  Complete-data ML and FIML-with-no-missing should be separate checks because
  they answer different implementation questions.

## What To Benchmark

The suite should be a matrix of model family, estimator/data path, and report
level. It does not need to run every cell on every commit; use tiers.

### Model Families

- CFA: one-factor, three-factor, five-factor, high-indicator models, with and
  without mean structures.
- Proper SEM: latent regressions, observed regressions, mediation/path models,
  equality constraints, and defined parameters.
- Multi-group: configural, metric, scalar, partial invariance, unequal group
  sizes, and group-specific missingness.
- Growth: linear latent growth with time scores, predictors of intercept/slope,
  and time-varying covariates where supported.
- Composite/formative-style models: include once the supported syntax and
  lavaan-equivalent semantics are explicit. Until then, keep them out of
  public lavaan comparisons and use only internal experimental timings.
- Boundary cases: shallow/Heywood-prone LS models, near-collinear indicators,
  sparse ordinal categories, many thresholds, and equality constraints across
  blocks.

### Estimator And Data Paths

- Complete-data continuous ML.
- Continuous FIML with natural missingness and with controlled missingness
  masks for scaling studies.
- ULS, GLS, and WLS on continuous sample statistics.
- Profiled/SNLLS variants for ULS/GLS/WLS, compared against the non-profiled
  magmaan fit and lavaan where lavaan has an equivalent estimator.
- Ordinal DWLS/WLS using thresholds, polychorics, NACOV, diagonal/full weights,
  and robust ordinal reporting.
- Mixed continuous/ordinal DWLS/WLS using thresholds, means/variances,
  polychorics, polyserials, covariance moments, NACOV, and weights.
- Robust normal-theory reporting: MLM, MLR, observed-bread variants, SB-family
  tests, Satorra-2000 nested tests where supported.

### Measurements

Record at least:

- wall time distribution: min, median, p90, p95, and interquartile range;
- memory allocation and peak RSS where practical;
- estimate agreement: max absolute and relative differences;
- requested statistic agreement: SEs, robust SEs, chi-square variants, df,
  fit measures, standardized solution, defined parameters, nested tests;
- objective value, gradient norm, convergence code/message, iterations;
- objective/gradient/Jacobian/profile evaluation counts when magmaan exposes
  them;
- data dimensions: N, number of observed variables, number of groups, number
  of free parameters, number of thresholds, moment-vector length, missingness
  pattern count, and weight-matrix dimension.

## Dataset Plan

Prefer data available from CRAN packages or stable public repositories, with a
local cache and citation/license notes. Candidate sources should be audited
before implementation, but the following set covers the intended shape of the
suite.

### Core Small And Medium Data

- `lavaan::HolzingerSwineford1939`: real classic CFA data with 301 rows and
  school/sex grouping variables. Use for 1F/3F CFA, multi-group invariance,
  mean structures, FIML masks, and quick smoke benchmarks.
- `lavaan::PoliticalDemocracy`: real Bollen political democracy SEM example.
  Use for proper SEM with latent regressions and indirect structural paths.
- `psych::bfi`: 25 Likert personality items from the SAPA project, documented
  as 2800 respondents. Use for five-factor ordinal CFA, missing-data FIML
  stress if treated as continuous, group splits, and mixed models with age,
  gender, or education covariates.
- `nlme::Orthodont`: real repeated-measures dental growth data. Use for a
  compact latent growth model with time scores and sex grouping.

### Larger Real-Data Candidates

- GSS: the General Social Survey provides mixed continuous, binary, and
  ordered survey variables with natural missingness. Use a pinned extract for
  mixed ordinal/polyserial benchmarks and FIML/missingness stress. Prefer a
  cache built from a documented source such as GSS Data Explorer, `gssr`, or a
  CRAN package extract after license review.
- ESS: the European Social Survey offers larger cross-national ordinal survey
  data and explicit country groups. Use a pinned wave/country subset for
  large-N ordinal/mixed multi-group benchmarks after confirming download and
  redistribution rules.
- NLSY-derived data: useful for growth/path examples with developmental
  variables if a stable package dataset has enough repeated measures for the
  intended latent growth model. Otherwise keep NLSY as a path/regression
  candidate, not the primary growth benchmark.

Known documentation sources checked while drafting this plan include the
`lavaan` manual for `HolzingerSwineford1939`, `PoliticalDemocracy`, and
`Demo.growth`; the `psych` documentation for `bfi`; CRAN documentation for
GSS/NLSY extracts; and ESS documentation stating free data access. Re-check
licenses, current package versions, and download stability before turning
candidates into fixtures.

### Scaling Variants

Use three deterministic scaling mechanisms, each labeled separately:

- Subsample real large datasets at fixed seeds for N ladders such as 250, 1k,
  5k, 25k, and 100k where available.
- Row-replicate real data only for algorithmic N-scaling checks where the
  distribution is intentionally fixed. This is not a headline real-data result.
- Simulate from fitted real-data models only for stress tests that need
  dimensions not found in public data. Label these as model-based simulations
  and keep them out of headline claims unless the text is explicit.

## Benchmark Tiers

Tier 0 should run quickly and frequently:

- HS 3F CFA, complete ML, estimate-only and standard report.
- HS 3F CFA by school, ML multi-group standard report.
- HS 3F CFA, ULS/GLS/WLS with and without SNLLS, magmaan-only backend
  comparison.
- Small bfi ordinal CFA, DWLS estimate-only plus robust ordinal report.

Tier 1 should run before performance-sensitive merges:

- PoliticalDemocracy SEM, ML standard and robust report.
- HS FIML with controlled missingness, MLR report.
- bfi 25-item ordinal five-factor CFA, DWLS/WLS sample stats plus fit.
- Orthodont growth model, complete ML standard report.
- Mixed bfi or GSS subset with polyserial/polychoric moments.

Tier 2 should run for releases or benchmark-page refreshes:

- Large GSS/ESS ordinal and mixed benchmarks across N ladders.
- Multi-group ESS/GSS models with unequal groups and missingness.
- Stress tests with many thresholds, high missingness pattern counts, and
  shallow LS cases.
- Full methods report workloads with fit measures, standardized solution,
  defined parameters, robust tests, and nested tests.

## Execution Methodology

The eventual harness should be R-first for public comparisons because the
public comparison is against lavaan's R interface and the magmaan user-facing
benchmark path is the R package. It should still call into C++ diagnostics for
internal metrics.

Recommended structure:

- `benchmarks/r/`: R harness, workload definitions, dataset preparation, and
  result validation.
- `benchmarks/data/`: cache metadata and scripts, not large downloaded data
  unless redistribution is clearly allowed.
- `benchmarks/results/`: ignored local raw results plus a checked-in small
  summary for documented runs.
- `benchmarks/cpp/`: optional C++ microbenchmarks for evaluator, discrepancy,
  gradient, sample-stat, and optimizer primitives.

The current scaffold follows this layout. `benchmarks/cases.yml` is the
human-readable manifest, `benchmarks/r/cases.R` is the executable case
registry, and `benchmarks/cases/<case_id>/` holds model syntax, source
metadata, and small reference summaries. Ignored local cache directories under
`benchmarks/data/` hold prepared CSV files and any fetched raw data whose
redistribution terms are not yet pinned.

Use `bench` or a similarly robust R benchmarking tool rather than raw
`system.time()` for public runs. Use `callr` or a similar process-isolation
tool for cold-start/end-to-end timings when package load, JIT warm-up, or
cache state matters. For hot-loop timings, explicitly perform warm-up runs and
then time only the labeled workload.

Control the environment:

- set BLAS/OpenMP thread counts to 1 for default public results;
- run optimized non-sanitized magmaan builds for speed claims;
- report whether Ceres is enabled;
- disable CPU frequency scaling where practical, or at least record the CPU
  governor and machine load;
- run enough iterations to stabilize medians, with fewer repetitions for
  large Tier 2 workloads;
- random seeds govern only subsampling/masking, not optimizer behavior unless
  a backend actually uses randomness.

Each workload function should:

1. Build a common input object from the model, data, estimator, options, and
   requested report level.
2. Run lavaan and magmaan adapters.
3. Validate estimates and requested statistics.
4. Return a compact result row with timings, dimensions, correctness deltas,
   and diagnostics.

## Public Reporting

The GitHub/README headline should be concise and hard to misread. Recommended
shape:

- one small table of headline real-data workloads;
- one bar or dot plot of median wall-time speedups with error bars or IQR;
- one companion correctness column such as max absolute estimate difference
  and max requested-statistic difference;
- a footnote with lavaan version, magmaan commit, hardware, R version,
  estimator/report level, and whether fit measures/robust SEs were included.

Suggested headline workloads:

- HS 3F CFA, ML, standard report.
- PoliticalDemocracy SEM, ML or robust report.
- bfi ordinal five-factor CFA, DWLS robust ordinal report.
- Orthodont growth, ML standard report.
- One larger GSS/ESS mixed ordinal benchmark once stable.

Avoid "magmaan is X times faster than lavaan" as a global sentence. Prefer
"On these supported real-data workloads, median end-to-end R timings were ..."
and show the workloads. If magmaan is slower on a fair cell, include it or
exclude the cell with a clear reason; hidden unfavorable cases will make the
benchmark page fragile.

## Internal Optimization Benchmarks

Internal benchmarks can and should be more detailed than lavaan comparisons.
They answer engineering questions rather than marketing questions.

Track:

- parser/lavaanify time;
- matrix representation construction;
- sample-stat construction for continuous, ordinal, and mixed data;
- model-implied moment evaluation;
- analytic gradient and finite-difference Jacobian costs;
- objective evaluation counts and gradient norms;
- LS residual/Jacobian construction;
- FIML pattern compression and per-pattern likelihood/gradient costs;
- robust U-Gamma construction, reduced gamma, eigenvalue computation;
- H1/baseline fit-measure components;
- R-to-C++ conversion overhead.

For optimizer/backend decisions, benchmark only appropriate comparisons:

- ML continuous: NLopt L-BFGS, PORT, and other ML-compatible backends.
- Bounded LS: NLopt L-BFGS, PORT, and Ceres where semantically appropriate.
- Profiled LS: non-profiled vs SNLLS for ULS/GLS/WLS, with both point
  agreement and profile diagnostics.
- Ordinal/mixed LS: DWLS/WLS with full sample-stat construction separated from
  fitting.

Internal benchmarks should keep historical baselines as JSON/CSV and fail
softly on large regressions in optional CI or pre-release checks. They should
not block normal correctness CI until the noise profile is well understood.

## Acceptance Criteria Before Publishing Benchmarks

- R package has stable wrappers for estimate-only and explicit post-fit
  reporting workflows.
- Each public workload has lavaan parity fixtures (`cpp/tests/fixtures/parity/`,
  gated by `cpp/tests/golden/lavaan_parity_golden_test.cpp`) or a benchmark-local
  correctness gate with documented tolerances.
- Dataset cache scripts are reproducible and license-compatible.
- Benchmark output records all versions, options, dimensions, and hardware.
- At least one non-author machine has reproduced the Tier 0 and headline
  results.
- The README/GitHub summary links to full methodology and raw result files.

## Reference Links

- `lavaan` manual: <https://cran.r-universe.dev/lavaan/doc/manual.html>
- `psych::bfi` documentation:
  <https://www.rdocumentation.org/packages/psych/topics/bfi>
- GSS mixed-scale CRAN extract example:
  <https://search.r-project.org/CRAN/refmans/GGMnonreg/html/gss.html>
- GSS package/cumulative-data candidate: <https://kjhealy.r-universe.dev/gssr>
- ESS data access notes:
  <https://www.europeansocialsurvey.org/methodology/ess-methodology/data-and-documentation-availability>
- NLSY CRAN extract example:
  <https://search.r-project.org/CRAN/refmans/heplots/html/NLSY.html>
