# magmaan SEM Zoo Benchmarks

## Numerical acceptance

New fit comparisons use `estimate::fit_verdict(est).status == FitCheck::Passed`
in C++, or `isTRUE(fit$converged)` in R. Preserve the component verdict and
backend termination separately; neither a backend success code nor a legacy
`fit$audit` flag substitutes for the common full-model check. Admissibility and
cross-method objective/parameter agreement are additional, separately reported
comparison requirements. Unchecked fits are not accepted by default.

The ordinal SNLLS driver exports `common_verdict`, its objective/stationarity
components, the declared domain, stationarity residual/tolerance, and
admissibility fields. Its `status=ok` means a fit returned, not that it passed
the numerical screen. The frozen `snlls-handoff` scripts/results retain the
policy of their pinned source revision for reproduction; use the common
verdict contract for new studies. See `docs/design/terminal-audit.md`.

This directory is the staging area for repeatable benchmark cases. The harness
is R-first because the public comparison target is lavaan and the exploratory R
package is the user-facing path for now.

The repository tracks case metadata, model syntax, source notes, and small
reference summaries. Downloaded raw data, prepared CSV files, and raw timing
results stay in ignored local cache directories until redistribution terms are
explicitly clear.

## Layout

- `cases.yml` is the human-readable case manifest.
- `cases/<case_id>/model.lav` stores lavaan-style model syntax.
- `cases/<case_id>/source.yml` records source, status, license notes, and
  feature requirements.
- `r/` contains the preparation and reference-generation scripts.
- `data/` is an ignored local cache for raw and prepared data.
- `results/` is an ignored local cache for timing outputs.

## First Commands

```sh
Rscript benchmarks/r/check_sources.R
Rscript benchmarks/r/fetch_data.R
Rscript benchmarks/r/prepare_case.R all
Rscript benchmarks/r/make_lavaan_reference.R all
Rscript benchmarks/r/run_benchmark.R all
```

`prepare_case.R` and `make_lavaan_reference.R` accept explicit case ids or
`all` (every case marked `supported_now` in `r/cases.R`). `run_benchmark.R`
fits each active case with the magmaan R package and with lavaan, validates
magmaan against the lavaan oracle (estimate drift gate plus reported SE and
chi-square diagnostics where available), times both at the estimate-only
workload level for that case's estimator/data path, and writes a result
summary into the ignored `results/` cache. The default estimate tolerance is
`1e-3`; cases may declare a wider estimator-specific tolerance in `r/cases.R`
when the advisory benchmark is looser than the parity fixtures.

External cases such as Stata Press and Mplus examples use `fetch_data.R` to
download raw data into the ignored cache. Cases whose terms are unclear should
stay as metadata and manual-download notes until they are audited.

## Complete-data modular timing

`timing/` holds `magmaan_timing_bench`, the reusable harness for comparing
algorithms, reparameterizations, and backends on complete-data linear SEM. It
times each pipeline stage separately (spec, data reduction, problem
construction, per-iteration primitives, optimize, post-fit) over a crossed
structure x parameterization x p x n design and writes tidy long CSV. Seven
model structures, p up to 96, populations constructed in closed form so the
fitted model is correctly specified by construction.

`timing/timing.hpp` is the shared timer — batch auto-calibration, arm rotation,
median reporting, dead-code-elimination barriers. New timing work should include
it rather than hand-rolling an eighth copy; see the retirement item in
[docs/backlog/todo.md](../docs/backlog/todo.md#benchmarks). Full documentation in
[timing/README.md](timing/README.md).

```sh
cmake --preset opt -DMAGMAAN_BUILD_BENCH=ON
cmake --build --preset opt --target magmaan_timing_bench
OMP_NUM_THREADS=1 ./build/opt/benchmarks/magmaan_timing_bench --p 12 --n 1000 --out /tmp/t.csv
Rscript benchmarks/timing/summarize.R /tmp/t.csv
```

## C++ memory profiling

`magmaan_mem_profile` is a standalone C++ harness that measures the peak heap of
four Gamma computations: the three reduced ones in `src/robust/robust.cpp` —
`reduced_gamma_sample` (batched streaming), `reduced_gamma_sample_materialized`
(the q×q reference), and `reduced_gamma_sample_streaming` (row-by-row) — plus a
`dense` path that forms the full q×q Gamma and U and eigendecomposes the q×q
product, the computation standard SEM software performs.

It is gated behind `MAGMAAN_BUILD_BENCH` and not built by default:

```sh
cmake -S . -B build/bench -DMAGMAAN_BUILD_BENCH=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/bench --target magmaan_mem_profile
./build/bench/benchmarks/magmaan_mem_profile {dense|materialized|batched|rowwise} [n] [p]
```

The harness builds a synthetic one-factor CFA of size `(n, p)`, fits it with ML,
forms the U-factor and casewise-contribution matrix, runs one chosen
computation, and prints the peak heap it allocated plus the process peak RSS.

`malloc_peak.{hpp,cpp}` installs the peak counter at the `malloc` layer via
linker `--wrap`. This is necessary because Eigen's dynamic matrices allocate
through `aligned_malloc` → `std::malloc`, bypassing `operator new`; an
`operator new` counter would miss them. Cross-check against Valgrind Massif
(`valgrind --tool=massif ./magmaan_mem_profile ...`) — the two agree to within
about 0.02%.

## Ordinal DWLS Gamma influence profiling

`magmaan_ordinal_gamma_influence_bench` isolates the two estimated-weight
influence channels used by complete-data ordinal DWLS IJ. It calls the
production statistics builder (full Gamma, without its WLS inverse), direct
Gamma-diagonal influence, Gamma-diagonal finite-difference Jacobian, and
`moment_influence * D.transpose()` separately. For at least three indicators it
also times a DWLS fit and the complete `robust_ordinal_ij` calculation. The
fitted one-factor model fixes its second loading to 0.4 (first loading 1), away
from its generating ratio, so the weight-influence correction matters under
population misspecification. Other loadings and thresholds remain free. Parsing,
model construction, and start preparation are outside these timings.

```sh
cmake --preset opt -DMAGMAAN_BUILD_BENCH=ON
cmake --build --preset opt --target magmaan_ordinal_gamma_influence_bench
build/opt/benchmarks/magmaan_ordinal_gamma_influence_bench 300 18 2 5 23260716
build/opt/benchmarks/magmaan_ordinal_gamma_influence_bench 1200 18 2 5 23260716
build/opt/benchmarks/magmaan_ordinal_gamma_influence_bench 300 18 4 5 23260716
```

Arguments are sample size, indicator count, category count, timed repetitions,
and seed; the first example gives the defaults. The synthetic data come from a
normal one-factor model with loadings 0.55–0.75 and equiprobable marginal
categories. The seed fixes the sample within a given C++ standard-library
implementation; this is not a reproduction of a paper's data generator.

Each stage receives one untimed warm-up and then repeats on the same data.
CSV stdout reports median/minimum/maximum wall milliseconds, median process
CPU milliseconds, and a result checksum. Stderr reports the proposed
item/pair-local Jacobian support and the largest derivative outside it, plus
the largest absolute column mean of the complete Gamma influence. These are
diagnostics, not replacements for the case-weight derivative tests.

Run serially without competing simulation workers. The `opt` library disables
Eigen threading. Record compiler/build flags, CPU, library identity, and any
background load with saved timings under ignored `benchmarks/results/`.
Memory and worker scaling need separate measurements.

The before/after timing plan is deliberately small:

1. Preserve an executable built before the change, then build the new executable
   with the same `opt` settings. Record the source revision and library hashes.
2. Run the three commands above against both executables, serially with the same
   seed. Each reports five repetitions after one warm-up. Stop competing builds
   and simulation workers while timing.
3. Compare median direct influence, Jacobian, and complete IJ time. Also report
   statistics + fit + IJ as the sum of their medians, clearly distinguished from
   a separately timed raw-data pipeline. Use the individual min/max columns to
   detect an unstable run rather than adding a large performance test grid.
4. Require numerical agreement first: unchanged statistics/fit checksums, close
   Gamma and complete-covariance checksums, and passing dense-reference,
   case-weight, and ordinal parity tests. Performance ratios are advisory, not
   timing assertions in CI.

Completed measurements and numerical contracts live in the
[roadmap](../docs/architecture/roadmap.md#ordinal-dwls-gamma-performance);
remaining work lives in the
[active backlog](../docs/backlog/todo.md#ordinal-dwls-gamma-influence-performance).

## Outstanding

- License audit: the Stata Press and Mplus example datasets are fetched into
  the local `data/` cache for local benchmarking only. Their redistribution
  terms have not been audited -- do this before vendoring any raw or cleaned
  data into the repository.
- `stata_higher_order` and `stata_correlated_uniqueness` ship as Stata
  summary-statistics (`ssd`) files, not raw observations; activating them needs
  covariance-matrix extraction plus model syntax pinned from the Stata SEM
  Reference Manual.
- `stata_growth` has real raw data cached but its `model.lav` is still a
  placeholder; the latent-curve syntax must be pinned from source.
- magmaan-side smoke gaps surfaced by `run_benchmark.R`: ML L-BFGS fails the
  line search on `bollen_democracy_sem` and does not converge on `bfi_5factor`;
  magmaan has no `growth()` equivalent, so `demo_growth_linear` does not match
  the lavaan growth parameterization.


## Reusable score inference

`score_primitives.R [output.csv]` is a bounded R benchmark for the staged score
API. It times context preparation, components, projection, spectrum and
calibration separately on two two-factor CFAs (10/20 variables, N=400), for
complete ML, 10% MCAR FIML, and NT-ML2S. It compares the combined staged
inference call with the legacy score-flip wrapper plus the same SB/pEBA2/pEBA4
calibrations. Fits are outside all timers, each function is warmed, and 15
calls are averaged. Component timers are independent measurements, not an
exact additive profile. The reference wrapper has already benefited from the
shared-core refactor; its historical exact-mixture diagnostic still runs.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 Rscript benchmarks/score_primitives.R /tmp/score-primitives.csv
```

No full simulation is run and no historical run artifacts are overwritten.

### Shared NTML inference

`Rscript benchmarks/inference_reuse.R` compares separate and shared global and
nested score/LR pipelines at N=400, p=10/20 (15 warmed repetitions, no simulation
rerun). Fresh shared timings include preparation; repeated timings cover four
calibrations plus a cached covariance. The script asserts that repeated calls
leave construction counts unchanged. Run with one BLAS/OpenMP thread and an
updated R development installation. Timings are advisory and depend on dimension.
