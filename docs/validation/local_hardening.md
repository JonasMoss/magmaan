# Local Hardening Program

This note is the validation/tooling program for keeping magmaan navigable as an
AI-first methods-development repo. It is intentionally local-first: no GitHub
Actions, no badges, and no global coverage gate are required for the first
slice. The active implementation checklist lives in
[`docs/backlog/todo.md`](../backlog/todo.md).

The goal is not to prove correctness with one number. The goal is to make the
current safety net visible enough that a maintainer or coding agent can tell
which areas are protected, which areas are dark, and what kind of proof belongs
next.

## Principles

- **Coverage is a map, not a grade.** Use line/function coverage to find dark
  rooms by domain, not to optimize one repository-wide percentage.
- **Local reports come first.** Generated artifacts belong under ignored build
  directories such as `build/coverage/`; CI can consume the same commands later
  if they prove useful.
- **Lavaan parity remains the contract.** Coverage reports do not replace
  golden fixtures, fixture regeneration, or benchmark correctness gates.
- **Different behavior gets different proof.** Lavaan-facing behavior wants
  golden parity. Algebraic internals want property, finite-difference, or
  invariant tests. R/API glue wants smoke examples and shape checks.
- **Reports should orient future work.** A good report answers "where am I
  unsafe?" faster than it answers "what percentage did I score?"

## Local Tools

### LLVM coverage

The clang-based local toolchain uses LLVM source coverage without changing
magmaan library code. The initial local lane is wired through
`CMakePresets.json` and `justfile`:

- the `coverage` CMake preset injects
  `-fprofile-instr-generate -fcoverage-mapping -O0 -g`;
- `just coverage` builds/runs the coverage tree and prints a terminal summary;
- `just coverage-html` writes `build/coverage/html/index.html`;
- raw profiles and merged `.profdata` files stay under `build/coverage/`;
- generated/dependency/test-harness paths such as `_deps/`,
  `third_party/`, and `tests/` when the question is core source coverage.

The useful unit of interpretation is domain-level coverage:

```text
parse/spec/model/estimate/inference/robust/measures/api
```

### Test reports

CTest already supports JUnit XML. The local report recipes make this
discoverable:

```sh
just test-report
just test-quick-report
```

This is useful for local dashboards, editor integrations, and eventual CI
without changing the test suite.

### One-command health check

A `just health` command is the maintainer cockpit. The first version stays
boring:

```text
just test-quick-report
just coverage
```

Later versions can add sanitizer or parity lanes deliberately, but the command
should stay understandable and locally runnable.

### Test ledger

Add a small `docs/validation/test_ledger.md` that describes protection by area:

```text
Area        Oracle        Test kind             Known gaps
Parser      EBNF/goldens  unit + golden         grammar-coverage walk
ML fit      lavaan JSON   golden + property     weak-id edge cases
Ordinal     lavaan/robcat golden + unit         mixed robust tolerances
R API       lavaan        examples smoke        mean-structure boundaries
```

The ledger is not a second backlog. It is the map from subsystem to validation
surface, oracle, and known blind spots.

### Risk map

For high-risk areas, add a short "protected by" note in the ledger rather than
scattering comments through source files. Example:

```text
src/estimate/fiml.cpp
Protected by:
- tests/unit/fiml_test.cpp
- tests/golden/fiml_golden_test.cpp
- tests/unit/api_sem_test.cpp
Known weak spots:
- missing observed exogenous variables under fixed.x
```

This is especially useful for AI-assisted edits because it turns "run the
tests" into a concrete validation path.

### Regression notes

When a bug fix adds or changes a test, leave a concise note either near the
test or in the ledger:

```text
Regression: repeated lhs/op/rhs terms produced duplicate matrix-cell rows.
Guard: lavaanify_test.cpp checks merge semantics; corpus parity still pending.
```

Tests become more useful when their intent survives the person who wrote them.

### CI and badges later

CI can be layered on top after the local commands are useful:

1. `test-quick` on pushes and pull requests.
2. Sanitizer validation on main or on a schedule.
3. Heavy parity and optional optimizer lanes less frequently.
4. Coverage as an artifact/report first, badge second.

Badges should stay narrow: "main is green" is usually more honest than a broad
coverage number. A coverage badge is only worth adding if maintainers are
actually using the report to guide test work.
