# C++ core

Canonical public headers are `include/magmaan/`; implementations and private
`detail_*.hpp` files are in `src/`. Tests are leaves under `tests/unit/` and
`tests/golden/`; `tests/fixtures/` holds checked-in oracle JSON, `tests/tools/`
maintainer generators, and `tests/checks/` advisory checks outside the default
suite. Licensed vendored build dependencies live in `third_party/`; each has
upstream sources/licenses and a README recording provenance and local patches.

## Implementation contracts

C++23 is required specifically for the `std::expected` error model. The compiler
probe in `cmake/MagmaanCompilerCheck.cmake` enforces GCC 13, Clang 19 with
libstdc++, or Clang 17 with libc++. Preserve `-fno-exceptions -fno-rtti` and
`EIGEN_NO_EXCEPTIONS`.

`Discrepancy`, `Optimizer`, `StandardErrorMethod` and `FitIndex` are documented
structural member-function interfaces used by free-function templates, not C++
concepts. Use no `concept`/`requires` constraints or hot-path virtual dispatch.
Public includes use `#include "magmaan/foo.hpp"`; private detail includes are
relative. Constants use `snake_case`, not `kCamelCase`.

Keep the model triple and lavaan partable projection consistent. Check names,
labels, starts, grouping and mean rows as well as numeric estimates. Parser
changes follow the root's EBNF-first rule. Use `magmaan-oracle-validation` for
fixture changes or parity investigations and `magmaan-r-bindings` when changing
the exposed R surface. After any canonical header/source change, run `just vendor`.
Never hand-edit `r-package/src/{core,magmaan,third_party}/`.

## Domain ownership

Directory names under `include/magmaan/` match namespaces:

| Namespace | Domain |
| --- | --- |
| `parse` | Lexer, parser, operator enums |
| `spec` | Lavaanify/build, starts, constraints, composites |
| `compat::lavaan` | Lavaan-shaped partable projection and matching |
| `data` | Raw observations and sample statistics |
| `model` | LISREL matrix representation and model evaluation |
| `estimate` | Fit orchestration, discrepancies, bounds, starts, SNLLS; `estimate::gmm` owns moment-quadratic weights |
| `optim` | Optimizer interfaces and backends |
| `inference` | Post-fit information, scores and SEs |
| `robust` | Robust tests and weighted inference |
| `measures` | Fit indices, residuals, standardization, effects, factor scores |
| `sim` | Population construction, reusable calibration, generators and projection |
| `api` | Friendly staged entry points |

The lavaan-parity core is the intended stable surface; frontier methods carry no
deprecation-cycle promise. Put new non-lavaan methods in the owning domain's
`frontier` namespace, with friendly entry points in `api::frontier`, never a
single top-level frontier namespace. Consult the roadmap and active backlog for
the current header layout and remaining retiering; do not assume the directory
already mirrors the namespace tier.

## Build and validation

Run root recipes from the repository root, or direct presets from `cpp/`:

- `just build` / `just test`: `fast` Debug build without sanitizers.
- `just test-area <area> [regex]`: one fast test executable. Areas include
  smoke, spec, estimate, inference, ordinal, api, sim, parity and robcat;
  inspect `justfile` and CMake targets for the current mapping.
- `just test-quick`: fast tests excluding heavy real-data parity.
- `just test-dev`: `dev`, AddressSanitizer and UBSan.
- `just opt` / `just test-opt`: `opt`, Release with native CPU tuning.
- `ceres` / `ipopt`: optional optimized backend presets; PORT is enabled by
  default. Portable `default` / `release` and legacy `ubsan` remain available.

Use focused unit/property/finite-difference checks for changed numerical
contracts and golden tests for parity. C++ tests read fixtures and never require
R/lavaan at runtime. `just check` also runs structural guards, vendor refresh/drift
checking, C++ tests and R checks; it is broader than an ordinary focused loop.
Root recipes cap build jobs by available RAM; preserve that cap unless explicitly
overridden. Avoid importing tests or experiment code into the library.
