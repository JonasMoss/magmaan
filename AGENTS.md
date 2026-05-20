# AGENTS.md

Working rules for coding agents in the magmaan repo.

## What magmaan is

A C++23 library that ports lavaan's behavior for **linear SEM under
complete-data normal-theory estimators**. The audience is methods developers,
not end users. [docs/architecture/roadmap.md](docs/architecture/roadmap.md) is the current state and
architecture summary; [docs/backlog/todo.md](docs/backlog/todo.md) is the active backlog of
remaining work. Read both before structural changes. The old external `latva`
plan is historical archaeology, not current guidance.

## Non-negotiables

- **C++23**, built with `-fno-exceptions -fno-rtti`. Eigen runs under
  `EIGEN_NO_EXCEPTIONS`. Failures are values: `std::expected<T, Error>`.
- **No virtual functions on the hot path.** Extension is via concepts
  (`Discrepancy`, `Optimizer`, `StandardErrorMethod`, `FitIndex`) and free
  function templates.
- **Lavaan is the oracle.** Parser, partable, point estimates, SEs, and
  chi-square statistics match installed lavaan output to documented
  tolerances. New fixtures are regenerated via `tests/tools/regen_oracle.R`; CI
  itself never invokes R.
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
- `tests/unit/` - focused unit tests.
- `tests/golden/` - fixture-based parity checks against lavaan.
- `tests/property/` - finite-difference Jacobian checks, etc.
- `tests/fixtures/` - checked-in JSON. Regenerate via `tests/tools/regen_oracle.R`.
- `tests/tools/` - maintainer-only fixture-generation scripts (R, etc.).
- `tests/checks/` - advisory local simulation checks, outside the default test suite.
- `benchmarks/` - advisory benchmark harness; ignored data/results caches stay local.
- `docs/research/` - tracked research notes and simulation scripts, not vendored PDFs.
- `docs/reference/` - policy for ignored external resources and source mirrors.
- `docs/grammar/` - `grammar.ebnf` (normative), `lexer.md`, `grammar.md`.
- `docs/architecture/roadmap.md` - current implementation state and design contracts.
- `docs/backlog/todo.md` - active human-readable backlog and remaining milestones.
- `external/` - ignored optional source mirrors for reading upstream code, never built.
- `third_party/` - tracked vendored third-party sources that participate in the
  build. Each subdirectory holds the verbatim upstream sources plus the upstream
  LICENSE files and a vendor README.md documenting source URL, commit, license,
  and any local patches. Currently: `third_party/port/` (PORT optimizer
  routines, AMPL/ASL + Fermi-LAT, BSD-3) — wired into the build via
  `cmake/PortVendor.cmake`.
- `r-package/` - exploratory R bindings (Rcpp); consumes the prebuilt
  `libmagmaan.a`, separate from and not part of the C++ build.

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
Ceres backend enabled. Existing presets such as `asan`, `release`, and `ubsan`
remain available for compatibility.

There's a `justfile` at the repo root wrapping the common loops: `just build`,
`just test` (aliases for the `dev` build plus ctest), `just opt`,
`just test-opt`, `just r-install`, `just r-check` (reinstall the R bindings
plus run `r-package/examples/*.R` vs lavaan), `just regen-oracle`, and
`just check` (everything). `just` with no recipe lists them. For the iterative
loop, `just test-area <area>` builds and runs a single test executable
(`smoke`, `spec`, `estimate`, `inference`, `ordinal`, or `parity`) — with an
optional test-name filter as a second arg — and `just test-quick` runs
everything except the heavy real-data `parity` tests.

The R bindings (`r-package/`) link the prebuilt non-sanitized `opt`
`libmagmaan.a`; `src/Makevars` makes the package objects depend on it, so a
C++ header change correctly forces the R glue to recompile against the new ABI.
If you ever see an `undefined symbol` at R load time anyway, `just r-clean`
(or `rm -f r-package/src/*.o`) and reinstall.

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
- `optim` - optimizer concepts and backends.
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

So far `api::frontier`, `estimate::frontier`, and `robust::frontier` exist;
their headers still sit in the domain directory rather than a
`<domain>/frontier/` subdirectory, and the `data/` research headers are not yet
retiered. See `docs/backlog/todo.md`.

## Conventions

- Lowercase `snake_case` for filenames and free functions; `CamelCase` for
  types; `kCamelCase` is not used; constants are `snake_case`
  (`version_major`, etc.).
- Public headers include with `#include "magmaan/foo.hpp"`.
- Private headers under `src/.../detail_*.hpp` include with relative paths.
- Comments only when the why is non-obvious. The roadmap and lavaan reference
  together cover the what.
- Keep `docs/backlog/todo.md` as the single active backlog. Remove or fold stale
  finished planning docs into `docs/architecture/roadmap.md` or `docs/backlog/todo.md` when a phase
  completes; do not create parallel roadmaps.
- Keep `docs/architecture/roadmap.md` current whenever a change alters implementation
  state, architecture, contracts, boundaries, or validation expectations.
- Keep `docs/backlog/todo.md` current whenever a change completes a milestone,
  changes priorities, or reveals new remaining work.
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
