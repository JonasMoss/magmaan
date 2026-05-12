# AGENTS.md

Working rules for coding agents in the latva repo.

## What latva is

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
- **The ParTable is the contract.** SoA struct mirroring lavaan's columns.
  Adding a feature means deciding what columns to add and what existing
  columns to honor.
- **`docs/grammar/` is the parser source of truth.** `grammar.ebnf` is
  normative; if the parser disagrees with the EBNF, the parser is wrong.
  Every parser/lexer function carries a `// production: name = ...`
  back-reference comment. When the grammar changes, edit the EBNF first,
  then the code, then regenerate fixtures.

## Where things live

- `include/latva/` — public headers (stable surface).
- `src/` — implementations + private `detail_*.hpp`.
- `tests/unit/` — focused unit tests.
- `tests/golden/` — fixture-based parity checks against lavaan.
- `tests/property/` — finite-difference Jacobian checks, etc.
- `tests/fixtures/` — checked-in JSON. Regenerate via `tools/regen_oracle.R`.
- `tools/` — maintainer-only scripts (R, etc.).
- `docs/grammar/` — `grammar.ebnf` (normative), `lexer.md`, `grammar.md`.
- `docs/roadmap.md` — what's left for full-data normal-theory ML feature-completeness
  (the Tier 1/2/3 gap list); `docs/p8_inference.md` — the detailed P8/inference plan.
- `external/lavaan/` — reference source, never built.
- `r-package/` — exploratory R bindings (Rcpp); consumes the prebuilt
  `liblatva.a`, separate from and not part of the C++ build.

## Build

```sh
cmake --preset asan
cmake --build --preset asan
ctest --preset asan
```

`asan` is the canonical CI build (combines AddressSanitizer + UBSan).

## Conventions

- Lowercase `snake_case` for filenames and free functions; `CamelCase`
  for types; `kCamelCase` is **not** used; constants are `snake_case`
  (`version_major`, etc.).
- Public headers include with `#include "latva/foo.hpp"`.
- Private headers under `src/.../detail_*.hpp` include with relative paths.
- Comments only when the *why* is non-obvious. The plan and the lavaan
  reference together cover the *what*.

## Working with the lavaan reference

`external/lavaan/` is the R package. Treat it as read-only spec material.
When implementing a step, read the formulas (Bollen 1989, Mulaik 2009,
Yuan-Bentler) — not the R source. Use lavaan output, not its code, as
the oracle.
