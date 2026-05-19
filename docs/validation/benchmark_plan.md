# Benchmark Plan

This is the full benchmarking design — tiers, datasets, workload levels, and
public-reporting shape. The scaffold described under Execution Methodology now
exists (`benchmarks/cases.yml`, the `benchmarks/r/` harness, and active
lavaan-backed cases spanning complete-data ML, controlled-missingness FIML,
and continuous ULS/GLS smoke paths); the actionable near-term slice is tracked
in `docs/backlog/todo.md` §2. The lavaan-parity test layer
(`tests/golden/lavaan_parity_golden_test.cpp`, `tests/fixtures/parity/`) is the
concrete realization of the correctness gating this document calls for. Public
headline benchmarking still waits on a more mature R package API.

## Goals

magmaan should eventually have three benchmark layers with different audiences:

1. Public lavaan comparison: honest end-to-end timings for supported workflows
   where magmaan and lavaan compute the same statistical outputs.
2. Internal regression suite: repeatable timings and diagnostics that make
   optimizer, sample-statistic, matrix-representation, and inference changes
   measurable.
3. Backend comparison suite: magmaan-only comparisons of LBFGS, LBFGS-B,
   Ceres, and profiled/SNLLS variants where those backends are statistically
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

- ML continuous: LBFGS and any future ML-compatible backends.
- Bounded LS: LBFGS-B and Ceres.
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
- Each public workload has lavaan parity fixtures (`tests/fixtures/parity/`,
  gated by `tests/golden/lavaan_parity_golden_test.cpp`) or a benchmark-local
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
