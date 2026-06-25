# AGENTS.md

Working rules for coding agents in the magmaan repo.

## What magmaan is

A C++23 library that ports lavaan's behavior for **linear SEM under
complete-data normal-theory estimators**. The audience is methods developers,
not end users. [docs/architecture/roadmap.md](docs/architecture/roadmap.md) is the current state and
architecture summary; [docs/backlog/todo.md](docs/backlog/todo.md) is the active backlog of
remaining SEM/parser/estimation work; [docs/backlog/simulation.md](docs/backlog/simulation.md)
is the simulation-specific backlog and decision log. Read the roadmap and the
relevant backlog before structural changes. The old external `latva` plan is
historical archaeology, not current guidance.

## Non-negotiables

- **C++23**, built with `-fno-exceptions -fno-rtti`. Eigen runs under
  `EIGEN_NO_EXCEPTIONS`. Failures are values: `std::expected<T, Error>`. The
  C++23 floor is carried by exactly that one feature — `std::expected` is the
  error model — a deliberate single-feature dependency (GCC 13 / Clang 17).
- **No virtual functions on the hot path.** Extension is via free function
  templates over structural (duck-typed) interfaces: `Discrepancy`,
  `Optimizer`, `StandardErrorMethod`, and `FitIndex` are documented
  member-function conventions a type must provide, not C++ `concept`s. magmaan
  deliberately uses no `concept`/`requires` constraints.
- **Lavaan is the oracle.** Parser, partable, point estimates, SEs, and
  chi-square statistics match installed lavaan output to documented
  tolerances. New fixtures are regenerated via `tests/tools/regen_oracle.R`; CI
  itself never invokes R. The rare exception: when lavaan (or another oracle) is
  *provably* wrong, do not gate against its output — gate transitively or by an
  independent reference, and record the case in
  [docs/validation/oracle-defects.md](docs/validation/oracle-defects.md) with
  the required standard of proof. A bare "magmaan differs" is almost always a
  magmaan bug, not an oracle defect; clear the high bar in that file before
  claiming otherwise. This is mostly relevant for less-popular features
  (multi-group categorical scores, exotic test/SE combinations) where the oracle
  is least exercised.
- **The lavaanified model is the contract.** Held in memory as a triple:
  `LatentStructure` (what to estimate, name-free, modulo estimator and
  identification convention), `LatentNames` (the verbal model: variable names,
  user labels, group var/levels, `.pN.` plabels), and `Starts` (free-param
  start hints). `to_lavaan_partable()` / `from_lavaan_partable()`
  (`compat/lavaan/partable_view.hpp`) project to/from the familiar lavaan-shaped SoA
  (`LavaanParTable`), which is what the R data.frame and golden `parTable()`
  fixtures compare against. Adding a feature means deciding what each of the
  three carries and what `matrix_rep` / `fit` honor.
- **`docs/grammar/` is the parser source of truth.** `grammar.ebnf` is
  normative; if the parser disagrees with the EBNF, the parser is wrong. Every
  parser/lexer function carries a `// production: name = ...` back-reference
  comment. When the grammar changes, edit the EBNF first, then the code, then
  regenerate fixtures.

## Where things live

- `include/magmaan/` - public headers (stable surface).
- `src/` - implementations plus private `detail_*.hpp`.
- `tests/unit/` - focused unit tests, including finite-difference
  Jacobian/property checks (`property_test.cpp`).
- `tests/golden/` - fixture-based parity checks against lavaan.
- `tests/fixtures/` - checked-in JSON. Regenerate via `tests/tools/regen_oracle.R`.
- `tests/tools/` - maintainer-only fixture-generation scripts (R, etc.).
- `tests/checks/` - advisory local simulation checks, outside the default test suite.
- `benchmarks/` - advisory benchmark harness; ignored data/results caches stay local.
- `docs/research/` - tracked research notes and simulation scripts, not vendored PDFs.
- `docs/research/paper-evals/` - short relevance-scored reads of recent SEM
  papers (the PDFs live in ignored `external/refs/`). Each eval ends in a
  verdict; actionable ones graduate to `backlog/speculative.md`, `todo.md`, or a
  paper/experiment. See its `README.md` for the template and the `/eval-paper` skill.
- `docs/reference/` - policy for ignored external resources and source mirrors.
- `docs/grammar/` - `grammar.ebnf` (normative), `lexer.md`, `grammar.md`.
- `docs/architecture/roadmap.md` - current implementation state and design contracts.
- `docs/backlog/todo.md` - active human-readable backlog and remaining milestones.
- `docs/backlog/simulation.md` - simulation-specific TODO and decision log for
  `magmaan::sim`, marginal generators, and simulation fixture policy.
- `docs/backlog/speculative.md` - deferred may-never-build items; each entry
  names the gap, the cheaper alternative that already covers it, and the
  explicit build-if trigger. Promote to `todo.md` only when a concrete
  downstream consumer appears.
- `external/` - ignored single "development help" folder: source mirrors for reading
  upstream code plus `external/refs/` reference PDFs. Never built. (Replaces the former
  separate `resources/` folder.)
- `third_party/` - tracked vendored third-party sources that participate in the
  build. Each subdirectory holds the verbatim upstream sources plus the upstream
  LICENSE files and a vendor README.md documenting source URL, commit, license,
  and any local patches. Currently: `third_party/port/` (PORT optimizer
  routines, AMPL/ASL + Fermi-LAT, BSD-3) — wired into the build via
  `cmake/PortVendor.cmake`.
- `r-package/` - exploratory R bindings (Rcpp). Self-contained and portable:
  the C++ core (plus `third_party/port` + `third_party/quadpack`) is **vendored**
  into `r-package/src/{core,magmaan,third_party}/` by `dev/vendor-cpp.sh`
  (`just vendor`) so `R CMD INSTALL` / `remotes::install_github` builds it with
  no CMake and no prebuilt library. NLopt is resolved from a system install
  (pkg-config) or, failing that, from the `nloptr` CRAN package, which bundles
  and self-builds NLopt (`LinkingTo`/`Imports: nloptr`) so no system NLopt module
  is needed. The vendored copies carry an `@generated` banner
  and must never be hand-edited; edit canonical `src/`/`include/` and re-vendor
  (`just vendor-check` guards drift). The **fast dev loop is `just r-dev`**, which
  compiles only the Rcpp glue and links the prebuilt `opt` `libmagmaan.a` via a
  throwaway `build-rdev/` mirror with `dev/r-makevars-dev` swapped in.

## Dependency layering

Dependencies flow strictly downward; **leaves are sinks**. Each item depends
only on strictly-lower tiers plus the one sanctioned shared sibling at its tier.

- **T0 inputs**: `third_party/` (built), `external/` (ignored), `corpus/`
  (submodule data).
- **T1 core**: `include/`, `src/` - depend on T0 only.
- **T2**: `r-package/` (depends on core only); `experiments/_support/` (the
  `magmaan.experiments` harness package: depends on core/r-package only, carries
  **no SEM logic** and **no paper/experiment-specific references**); `benchmarks/`
  (shared benchmark harness that experiments may consume).
- **T3 leaves / sinks**: each `papers/<name>/`, each `experiments/<NN>-*/`, and
  `tests/`. A leaf consumes only lower tiers, is referenced by nothing, and never
  references a sibling leaf.

Invariants (enforced by `tests/tools/check_layering.sh`, run via
`just check-layering`, folded into `just check`, and a hard-failing CI job):

1. **Core never reaches up**: nothing in `include/`, `src/`, or `r-package/`
   references `papers/`, `experiments/`, `benchmarks/`, or `tests/`.
2. **Papers are private**: `papers/A/**` is referenced only from within
   `papers/A/` (each paper is its own nested git repo, gitignored by the outer
   repo).
3. **Experiments are endpoints**: an `experiments/<NN>/` references no paper and
   no other experiment; the only shared experiment sibling is
   `experiments/_support`. Experiments may consume `benchmarks/` and the corpus
   submodule.
4. **No sibling-leaf edges**: paper-to-paper, experiment-to-experiment (except
   `_support`), paper-to-experiment, and tests-to-(papers/experiments) are all
   forbidden.
5. **Shared code flows down, never sideways**: code two leaves both need goes
   into core, `r-package`, `experiments/_support`, or `benchmarks` - never
   sourced/loaded/included across a sibling boundary. (Running a built artifact,
   e.g. `build/<preset>/benchmarks/<bin>`, is allowed; it is execution, not a
   source dependency.)
6. **Reports read only from their own `results/`**.

The checker scans code files only (`*.R/*.cpp/*.hpp/*.h/CMakeLists.txt/*.cmake/
*.sh/justfile`), with comments stripped, so prose and comments never trip it.

## Build

```sh
cmake --preset dev
cmake --build --preset dev
ctest --preset dev

cmake --preset opt
cmake --build --preset opt
ctest --preset opt

cmake --preset ceres
cmake --build --preset ceres
ctest --preset ceres
```

`dev` is the local debug build (clang++, AddressSanitizer + UBSan). `opt` is
the local performance-comparison build (Release, no sanitizers,
`-O3 -DNDEBUG -march=native`). `ceres` is the optional optimized build with the
Ceres and NLopt backends enabled; PORT remains enabled by default. Existing
presets such as `release` and `ubsan` remain available for compatibility.

There's a `justfile` at the repo root wrapping the common loops: `just build`,
`just test` (aliases for the `dev` build plus ctest), `just opt`,
`just test-opt`, `just r-dev` (the fast R-bindings dev loop; links the prebuilt
`libmagmaan.a`), `just vendor` (refresh the vendored C++ in `r-package/src/`),
`just r-install` (the portable self-contained build, as `install_github`/Saga
would do — slower), `just r-check` (reinstall via `r-dev` plus run
`r-package/examples/*.R` vs lavaan), `just regen-oracle`, and `just check`
(everything). `just` with no recipe lists them. For the iterative
loop, `just test-area <area>` builds and runs a single test executable
(`smoke`, `spec`, `estimate`, `inference`, `ordinal`, or `parity`) — with an
optional test-name filter as a second arg — and `just test-quick` runs
everything except the heavy real-data `parity` tests.

The fast dev loop `just r-dev` links the prebuilt non-sanitized `opt`
`libmagmaan.a` (via the `build-rdev/` mirror + `dev/r-makevars-dev`), so a C++
header change forces the glue to recompile against the new ABI; a full rebuild of
the vendored core only happens on the portable `just r-install`. After editing
`src/`/`include/`, run `just vendor` to refresh the vendored copies. If you ever
see an `undefined symbol` at R load time, `just r-clean` and reinstall. To
install on a cluster, see [dev/saga/README.md](dev/saga/README.md).

## R Package Direction

The R package is a methods-developer interface over the C++ library, not a
second implementation and not a lavaan replacement UI. Prefer thin exported R
wrappers around one C++ entry point, with C++ argument structure kept visible.
Small R helpers are fine when they compose existing wrappers, validate R-shaped
inputs, or preserve names/groups for inspection; they should not contain
parallel SEM logic.

The intended high-level convenience is `magmaan(model, data, estimator,
groups)`: parse/lavaanify, build sample statistics, and estimate parameters in
one call. That function should do estimation only. Standard errors,
information matrices, Wald/z tests, robust corrections such as MLM and
Satorra-Bentler, fit measures, defined parameters, and nested tests remain
explicit post-fit calls so methods work can choose and inspect each step.

Partables exposed from R are compatibility/projection objects. When a model is
built with the R helpers, the partable returned from a fitted magmaan object
must match the corresponding lavaan call under the same options, modulo
documented numeric tolerances and any explicitly unsupported rows. Do not paper
over partable mismatches in R formatting code; fix the lavaanified model
triple, the projection helpers, or the fit result reconstruction. Mean-structure
rows are especially sensitive: fixed-zero intercept/mean parameters
must appear or disappear exactly as lavaan would for the requested model.

## Namespace Layout

Each top-level namespace owns a domain; the directory under `include/magmaan/`
matches the namespace.

- `parse` - lexer, parser, operator enums.
- `spec` - model specification: lavaanify (`build`), start hints, linear
  constraints, composite expansion.
- `compat::lavaan` - lavaan-shaped partable projection and oracle matching.
- `data` - raw data and sample statistics.
- `model` - the numeric face of the lavaanified model: LISREL matrix
  representation (`MatrixRep`) and evaluation (`ModelEvaluator`). Shared by
  `estimate`, `inference`, `measures`, and `robust`.
- `estimate` - fit orchestration, ML/FIML/GMM/GLS discrepancies, bounds,
  constraints, start values, SNLLS; `estimate::gmm` for the moment-quadratic
  weight machinery.
- `optim` - optimizer interfaces and backends.
- `inference` - post-fit information, scores, standard errors.
- `robust` - robust tests (Satorra-Bentler family, weighted inference).
- `measures` - fit indices, residuals, standardization, effects, factor scores.
- `api` - the friendly staged entry points.

### Core vs frontier

The public API is tiered (see [docs/design/ideas.md](docs/design/ideas.md)). `core` is the
stable, lavaan-parity surface. `frontier` is the research / non-lavaan methods
surface: it nests **per domain** - `estimate::frontier`, `data::frontier`,
`robust::frontier`, and so on - never a single top-level `frontier` namespace.
Friendly frontier entry points live under `api::frontier`. A `frontier` symbol
carries no deprecation-cycle promise. Put new non-lavaan methods in their
domain's `frontier` sub-namespace.

So far `api::frontier`, `estimate::frontier`, `robust::frontier`, and
`measures::frontier` exist; their headers still sit in the domain directory
rather than a `<domain>/frontier/` subdirectory, and the `data/` research
headers are not yet retiered. See `docs/backlog/todo.md`.

## Conventions

- Lowercase `snake_case` for filenames and free functions; `CamelCase` for
  types; `kCamelCase` is not used; constants are `snake_case`
  (`version_major`, etc.).
- Public headers include with `#include "magmaan/foo.hpp"`.
- Private headers under `src/.../detail_*.hpp` include with relative paths.
- Comments only when the why is non-obvious. The roadmap and lavaan reference
  together cover the what.
- Keep `docs/backlog/todo.md` as the main active backlog and
  `docs/backlog/simulation.md` as the simulation-specific active backlog.
  Treat `docs/backlog/simulation.md` as the design roadmap for the simulation
  sublibrary's generator/projection/calibration stack; keep only cross-domain
  summaries in the main roadmap/TODO.
  Non-committed may-never-build ideas live in `docs/backlog/speculative.md`,
  not in `todo.md`; it is a trigger list, not a parallel roadmap.
  Remove or fold stale finished planning docs into `docs/architecture/roadmap.md`,
  `docs/backlog/todo.md`, or `docs/backlog/simulation.md` when a phase
  completes; do not create parallel roadmaps.
- Keep `docs/architecture/roadmap.md` current whenever a change alters implementation
  state, architecture, contracts, boundaries, or validation expectations.
- Keep `docs/backlog/todo.md` current whenever a change completes a milestone,
  changes priorities, or reveals new remaining work.
- Keep `docs/backlog/simulation.md` current whenever a change completes a
  simulation milestone, changes generator priorities, or reveals new
  simulation-specific work.
- Commit every finished user request as a coherent completed change before
  handing back, unless the user explicitly asks not to commit. Keep work on the
  current branch unless explicitly asked to branch.

## Working with the lavaan reference

`external/` is ignored and optional. It may hold local source mirrors such as
lavaan or robcat for reading implementation details, but normal tests and CI do
not depend on it. Fixture regeneration uses installed R packages at the pinned
versions and writes checked-in JSON; C++ tests consume those fixtures only.
When implementing a step, read the formulas (Bollen 1989, Mulaik 2009,
Yuan-Bentler), not the R source. Use package output, not vendored code, as the
oracle. See `docs/reference/external_resources.md`.

When a parity check fails, assume a magmaan bug first. If — and only if — you can
clear the standard of proof in
[`docs/validation/oracle-defects.md`](docs/validation/oracle-defects.md) (an
independent reference magmaan matches plus a first-principles property the oracle
violates), record the case there and gate the affected test transitively or
self-consistently rather than against the bad output. That ledger is also where
to look before re-investigating a suspicious divergence, and the place to grow a
list we can later turn into upstream PRs / bug reports.
