# AGENTS.md

Working rules for coding agents in the magmaan repo.

## What magmaan is

A C++23 library that ports lavaan's behavior for **linear SEM under
complete-data normal-theory estimators**. The audience is methods developers,
not end users. [docs/roadmap.md](docs/roadmap.md) is the current state and
architecture summary; [docs/todo.md](docs/todo.md) is the active backlog of
remaining work. Read both before structural changes. The old external `latva`
plan is historical archaeology, not current guidance.

## Non-negotiables

- **C++23**, built with `-fno-exceptions -fno-rtti`. Eigen runs under
  `EIGEN_NO_EXCEPTIONS`. Failures are values: `std::expected<T, Error>`.
- **No virtual functions on the hot path.** Extension is via concepts
  (`Discrepancy`, `Optimizer`, `StandardErrorMethod`, `FitIndex`) and free
  function templates.
- **Lavaan is the oracle.** Parser, partable, point estimates, SEs, and
  chi-square statistics match `external/lavaan/` outputs to documented
  tolerances. New fixtures are regenerated via `tools/regen_oracle.R`; CI
  itself never invokes R.
- **The lavaanified model is the contract.** Held in memory as a triple:
  `LatentStructure` (what to estimate, name-free, modulo estimator and
  identification convention), `LatentNames` (the verbal model: variable names,
  user labels, group var/levels, `.pN.` plabels), and `Starts` (free-param
  start hints). `to_lavaan_partable()` / `from_lavaan_partable()`
  (`partable/lavaan_view.hpp`) project to/from the familiar lavaan-shaped SoA
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
- `tests/fixtures/` - checked-in JSON. Regenerate via `tools/regen_oracle.R`.
- `tools/` - maintainer-only scripts (R, etc.).
- `docs/grammar/` - `grammar.ebnf` (normative), `lexer.md`, `grammar.md`.
- `docs/roadmap.md` - current implementation state and design contracts.
- `docs/todo.md` - active human-readable backlog and remaining milestones.
- `external/lavaan/` - reference source, never built.
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
`just check` (everything). `just` with no recipe lists them.

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

The current public namespace layout is transitional. Prefer target headers for
new code:

- `spec` for model specification, lavaanify, start hints, and linear
  constraints.
- `lavaan` for lavaan-shaped partable projection.
- `estimate` for fit orchestration, bounds, fixed.x, constraints, and SNLLS.
- `optim` for optimizer concepts and implementations.
- `nt` for normal-theory ML, inference, robust tests, measures, effects, and
  standardization.
- `gls` for ULS/GLS/WLS discrepancies.
- `data` for raw data and sample statistics.

Old `fit/*` and `partable/*` headers remain as compatibility shims during the
namespace transition.

## Conventions

- Lowercase `snake_case` for filenames and free functions; `CamelCase` for
  types; `kCamelCase` is not used; constants are `snake_case`
  (`version_major`, etc.).
- Public headers include with `#include "magmaan/foo.hpp"`.
- Private headers under `src/.../detail_*.hpp` include with relative paths.
- Comments only when the why is non-obvious. The roadmap and lavaan reference
  together cover the what.
- Keep `docs/todo.md` as the single active backlog. Remove or fold stale
  finished planning docs into `docs/roadmap.md` or `docs/todo.md` when a phase
  completes; do not create parallel roadmaps.
- Keep `docs/roadmap.md` current whenever a change alters implementation
  state, architecture, contracts, boundaries, or validation expectations.
- Keep `docs/todo.md` current whenever a change completes a milestone,
  changes priorities, or reveals new remaining work.
- Commit every finished user request as a coherent completed change before
  handing back, unless the user explicitly asks not to commit. Keep work on the
  current branch unless explicitly asked to branch.

## Working with the lavaan reference

`external/lavaan/` is the R package. Treat it as read-only spec material. When
implementing a step, read the formulas (Bollen 1989, Mulaik 2009,
Yuan-Bentler), not the R source. Use lavaan output, not its code, as the
oracle.
