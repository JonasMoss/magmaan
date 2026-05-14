# AGENTS.md

Working rules for coding agents in the magmaan repo.

## What magmaan is

A C++23 library that ports lavaan's behavior for **linear SEM under
normal-theory ML**. The audience is methods developers, not end users.
The plan is in `/home/jmoss/.claude/plans/hello-i-want-to-eager-dream.md`.
Read it before structural changes.

## Non-negotiables

- **C++23**, built with `-fno-exceptions -fno-rtti`. Eigen runs under
  `EIGEN_NO_EXCEPTIONS`. Failures are values: `std::expected<T, Error>`.
- **No virtual functions on the hot path.** Extension is via concepts
  (`Discrepancy`, `Optimizer`, `StandardErrorMethod`, `FitIndex`) and
  free function templates.
- **Lavaan is the oracle.** Parser, partable, point estimates, SEs, and
  χ² match `external/lavaan/` outputs to documented tolerances. New
  fixtures are regenerated via `tools/regen_oracle.R`; CI itself never
  invokes R.
- **The lavaanified model is the contract.** Held in memory as a triple:
  `LatentStructure` (what to estimate — name-free, modulo estimator and
  identification convention), `LatentNames` (the verbal model — variable
  names, user labels, group var/levels, `.pN.` plabels), and `Starts`
  (free-param start hints). `to_lavaan_partable()` / `from_lavaan_partable()`
  (`partable/lavaan_view.hpp`) project to/from the familiar lavaan-shaped
  SoA (`LavaanParTable`) — that's what the R data.frame and the golden
  `parTable()` fixtures compare against. Adding a feature means deciding
  what each of the three carries and what `matrix_rep` / `fit` honor.
- **`docs/grammar/` is the parser source of truth.** `grammar.ebnf` is
  normative; if the parser disagrees with the EBNF, the parser is wrong.
  Every parser/lexer function carries a `// production: name = ...`
  back-reference comment. When the grammar changes, edit the EBNF first,
  then the code, then regenerate fixtures.

## Where things live

- `include/magmaan/` — public headers (stable surface).
- `src/` — implementations + private `detail_*.hpp`.
- `tests/unit/` — focused unit tests.
- `tests/golden/` — fixture-based parity checks against lavaan.
- `tests/property/` — finite-difference Jacobian checks, etc.
- `tests/fixtures/` — checked-in JSON. Regenerate via `tools/regen_oracle.R`.
- `tools/` — maintainer-only scripts (R, etc.).
- `docs/grammar/` — `grammar.ebnf` (normative), `lexer.md`, `grammar.md`.
- `docs/roadmap.md` — what's left for full-data normal-theory ML feature-completeness
  and active architectural cleanup.
- `external/lavaan/` — reference source, never built.
- `r-package/` — exploratory R bindings (Rcpp); consumes the prebuilt
  `libmagmaan.a`, separate from and not part of the C++ build.

## Build

```sh
cmake --preset dev
cmake --build --preset dev
ctest --preset dev

cmake --preset opt
cmake --build --preset opt
ctest --preset opt
```

`dev` is the local debug build (clang++, AddressSanitizer + UBSan). `opt` is
the local performance-comparison build (Release, no sanitizers,
`-O3 -DNDEBUG -march=native`). Existing presets such as `asan`, `release`,
`ubsan`, and `ceres` remain available for compatibility.

There's a `justfile` at the repo root wrapping the common loops — `just build`,
`just test` (aliases for the `dev` build + ctest), `just opt`, `just test-opt`,
`just r-install`, `just r-check` (reinstall the R bindings + run
`r-package/examples/*.R` vs lavaan), `just regen-oracle`, `just check`
(everything). `just` with no recipe lists them.

The R bindings (`r-package/`) link the prebuilt non-sanitized `opt` `libmagmaan.a`;
`src/Makevars` makes the package objects depend on it, so a C++ *header* change
correctly forces the R glue to recompile against the new ABI. If you ever see an
`undefined symbol` at R load time anyway, `just r-clean` (or `rm -f
r-package/src/*.o`) and reinstall.

## Conventions

- Lowercase `snake_case` for filenames and free functions; `CamelCase`
  for types; `kCamelCase` is **not** used; constants are `snake_case`
  (`version_major`, etc.).
- Public headers include with `#include "magmaan/foo.hpp"`.
- Private headers under `src/.../detail_*.hpp` include with relative paths.
- Comments only when the *why* is non-obvious. The plan and the lavaan
  reference together cover the *what*.
- Remove or fold stale finished planning docs into `docs/roadmap.md` when a
  phase completes; do not keep old plan files around as parallel roadmaps.
- Commit coherent completed changes when there is a natural checkpoint; keep
  work on the current branch unless explicitly asked to branch.

## Working with the lavaan reference

`external/lavaan/` is the R package. Treat it as read-only spec material.
When implementing a step, read the formulas (Bollen 1989, Mulaik 2009,
Yuan-Bentler) — not the R source. Use lavaan output, not its code, as
the oracle.
