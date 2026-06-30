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
- **Scaled statistics need calibration parity, not just value parity.** A robust
  or scaled test statistic can match the oracle on one dataset, or look like a
  defensible convention difference, yet carry the wrong sampling distribution and
  mis-size the test. Single-dataset parity cannot see that; a null
  rejection-rate Monte Carlo against the oracle, in the regime the statistic is
  meant for, can. See [calibration-parity.md](calibration-parity.md).

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
- generated, vendored, and test-harness paths are dropped from the report via
  `--ignore-filename-regex='(/_deps/|/third_party/|/tests/|/usr/)'`: `_deps/`
  (fetched doctest / nlohmann / NLopt), `third_party/` (vendored PORT), `tests/`
  (the test sources themselves), and `/usr/` (system Eigen and libstdc++). What
  remains is magmaan's own `include/magmaan/**` and `src/**`, header-defined
  inline/template code included.

The useful unit of interpretation is domain-level coverage. The report lists one
row per file; group those rows by their `include/magmaan/<domain>/` or
`src/<domain>/` path segment:

```text
parse spec model data estimate inference robust measures sim optim api compat
```

plus a small `(top-level)` bucket for the shared `src/detail_*.{hpp,cpp}` numeric
helpers and `src/util/`. (The earlier short list omitted `data`, `sim`, `optim`,
and `compat`, which are real domains the report covers.)

### Interpreting a coverage run

Calibrated against a full `just coverage` run on 2026-06-15 (clang/llvm-cov 21,
all nine `magmaan_test_*` binaries). Two things need interpretation before the
numbers are trustworthy: the `mismatched data` warning and the domain map.

#### The `mismatched data` warning is benign

A clean run prints, just before the table:

```text
warning: 171 functions have mismatched data
```

This is expected and does not corrupt the report. Cause: header-defined inline
and template functions (in `include/magmaan/**`, the `src/**/detail_*.hpp`
headers, and Eigen) are compiled into several test translation units, and their
structural coverage-mapping hashes differ across those TUs. `llvm-profdata merge`
keeps one record per `(name, hash)`; when `llvm-cov` then matches a binary's
function record against the single merged profile and finds the name under a
different hash than the profile retained, it drops that function record and
counts it as "mismatched".

Three facts pin this down (all reproducible without rebuilding, by re-running
`llvm-cov report` against `build/coverage/coverage.profdata`):

- The count scales with how many binaries are combined: the trivial `smoke`
  binary alone reports `0`, `magmaan_test_ordinal` alone against the merged
  profile reports `19`, and all nine together report `171`. More binaries means
  more independently-hashed copies of the same header code to reconcile.
- It is emitted at **load time**, before `--ignore-filename-regex` runs. Adding
  `/include/` and `\.hpp$` to the ignore regex still prints `171` (it only drops
  ~109 header rows from the *summary*, 1757 → 1648 reported functions). So the
  warning cannot be silenced by changing the ignore list, and it should not be
  piped to `/dev/null` — that would also hide genuine load errors.
- The out-of-line functions defined in `src/*.cpp` are compiled once into the
  single static `libmagmaan.a` and linked identically into every test binary, so
  they carry one consistent hash and are reported accurately. The dropped records
  are header glue.

Consequence: the **function** column's denominator is mildly understated (a
fraction of header inline/template instantiations are omitted), but the
**region** and **line** coverage of magmaan's compiled source — the part the map
is for — is sound. Read the line/region columns, not the function count.

#### Object list and ignore list need no changes

Reviewed during the calibration run; both are correct as-is:

- **Object list.** The recipe passes exactly the nine `magmaan_test_*` targets
  (`smoke spec estimate inference ordinal api sim parity robcat`), which is the
  complete set defined in `tests/CMakeLists.txt`. Nothing else is built into a
  coverage-instrumented binary, so there is nothing to add. The R-package
  examples and the benchmarks/experiments/papers leaves are *not* compiled into
  these binaries and correctly never appear.
- **Ignore list.** `(/_deps/|/third_party/|/tests/|/usr/)` drops vendored, system,
  and test code while keeping `include/magmaan/**` and `src/**` (including the
  `detail_*.hpp` headers, which are real magmaan code we want mapped). The report
  file list contains only those two trees — no spurious entries leak in.

#### Domain map (2026-06-15 snapshot)

`just coverage` line/region coverage aggregated to domains. Treat this as a map
of where the dark rooms are, not a grade:

```text
domain          lines  miss  lineCov  regions  regCov
spec             1589    74    95.3%     1073    95.1%
parse            1049    98    90.7%      759    92.6%
data             6432   669    89.6%     3888    90.1%
model            1205   198    83.6%      883    85.4%
inference        2592   527    79.7%     1694    81.5%
optim             841   176    79.1%      459    75.4%
measures         2547   555    78.2%     1684    79.9%
estimate        12292  2737    77.7%     8124    81.4%
sim              6810  1934    71.6%     4506    79.8%
robust           3700  1143    69.1%     2290    75.6%
compat            282   100    64.5%      160    45.0%
api              1556   654    58.0%      839    66.2%
(top-level)       399    66    83.5%      239    84.1%
TOTAL           41297  8931    78.4%    26599    82.0%
```

How to read the dark rooms:

- **`api` (58%) and `compat` (64.5%, regions 45%) read artificially dark.** Both
  are validated primarily through the R boundary — the staged `api::*` entry
  points and the `from_lavaan_partable` reverse projection are exercised by
  `r-package/examples/*.R` (`just r-check`), which this lane does not instrument.
  Before treating either as untested, cross-reference the R-boundary row of
  [`test_ledger.md`](test_ledger.md). This is a lane limitation, not a coverage
  signal; the coverage lane is C++-test-only by design.
- **`robust` (69%) and `sim` (72%)** track known backlog gaps, not blind spots:
  `robust/weighted_chisq.cpp` (~40%) and the FMG/Satorra-2000 tails carry
  deferred tiers (see the test ledger), and `sim/population.cpp` (~41%) and
  `model_implied.cpp` (~57%) are the open model-implied-simulation and
  population-metadata items in [`../backlog/simulation.md`](../backlog/simulation.md).
- **`spec`, `parse`, `data` (≥90%)** are the well-mapped core: parser, lavaanify,
  and the ordinal/pairwise moment builders. Coverage there is a real safety net,
  consistent with their golden + property protection.

Re-run and refresh this snapshot after major test or source changes; the absolute
percentages drift, but the *ordering* (which domains are darkest, and why) is the
durable output.

### Test reports

CTest already supports JUnit XML. The local report recipes make this
discoverable:

```sh
just test-report
just test-quick-report
```

This is useful for local dashboards, editor integrations, and eventual CI
without changing the test suite.

First calibration run: CTest resolves relative `--output-junit` paths from the
build tree, not the repository root. The report recipes therefore pass absolute
paths so the artifacts land at:

```text
build/fast/test-results.xml
build/fast/test-quick-results.xml
```

### One-command health check

A `just health` command is the maintainer cockpit. The first version stays
boring:

```text
just test-quick-report
just coverage
```

Later versions can add sanitizer or parity lanes deliberately, but the command
should stay understandable and locally runnable.

First calibration run: `just health` successfully runs `test-quick-report`
first and then the full LLVM coverage lane. That means it is a real local
maintenance check, not a smoke test; coverage includes the parity and robcat
targets because the coverage report is meant to map the whole checked surface.

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
