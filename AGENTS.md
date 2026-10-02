# magmaan agent instructions

magmaan is a C++23 library for linear SEM, checked component by component
against lavaan. Ordinary users get the opinionated pure-R `magmaan` package;
methods developers get the compiled `magmaanlab` package and composable C++ API.
See [the R-interface vision](project/design/r-interface-vision.md).

## Start with the relevant guidance

Read applicable nested `AGENTS.md` files before editing their directories:
[C++](cpp/AGENTS.md), [compiled R bindings](r-package/AGENTS.md),
[ordinary-user R](r-magmaan/AGENTS.md), [experiments](experiments/AGENTS.md),
and [papers](papers/AGENTS.md). `CLAUDE.md` files import the corresponding
`AGENTS.md`; keep one source of instructions per directory.

[project/scope.md](project/scope.md) owns statistical targets and sampling-law
boundaries. [project/architecture/roadmap.md](project/architecture/roadmap.md)
owns current
implementation state and contracts. [project/backlog/todo.md](project/backlog/todo.md)
owns accepted SEM/parser/estimation work; [simulation.md](project/backlog/simulation.md)
owns simulation state, priorities and its decision log. Before structural changes,
read the relevant sections; use headings and search rather than loading both
large documents in full. The old external `latva` plan is historical only.

Task-specific workflows live in [.agents/skills](.agents/skills):

- `magmaan-oracle-validation`: investigate parity failures or regenerate fixtures.
- `magmaan-r-bindings`: change bindings, generated exports, or vendored core code.
- `magmaan-experiment`: create, extend, run, or report a repository experiment.

Each skill describes its inputs, outputs and verification. Load its references
only for the operation at hand. Skills support these rules; they do not replace
the user's scope or the applicable directory instructions.

## Non-negotiables

- C++23, `-fno-exceptions -fno-rtti`, and `EIGEN_NO_EXCEPTIONS`.
  Failures are values: `std::expected<T, Error>`. No `concept`/`requires`
  constraints; no virtual functions on the hot path. Extension uses free-function
  templates over documented member-function conventions.
- Lavaan is the component oracle for parser, partable, estimates, SEs and tests,
  with documented numeric tolerances. C++ tests read checked-in fixtures and
  need no R; R integration tests, including CI, also compare installed lavaan.
  Assume a magmaan bug first. An oracle exemption requires the independent
  reference and first-principles proof in
  [oracle-defects.md](project/validation/oracle-defects.md), including target-regime
  calibration where required. Record the proof and an independent or transitive
  gate before exempting a comparison.
- The ordinary-user inference policy deliberately differs from lavaan defaults.
  Its choices need recorded experimental or published evidence, not just parity.
- The lavaanified model is the contract: `LatentStructure` (name-free estimands),
  `LatentNames` (names, labels, groups, `.pN.` plabels), and `Starts` (start hints).
  `compat::lavaan::to_lavaan_partable()` / `from_lavaan_partable()` project the
  triple to/from `LavaanParTable`. A feature must specify what each component
  carries and what `matrix_rep` / `fit` honor. Fix partable mismatches in this
  contract or its projections, never by concealing them in R formatting.
- [project/grammar/grammar.ebnf](project/grammar/grammar.ebnf) is normative.
  Every parser/lexer function carries a `// production: name = ...` reference.
  Change the EBNF first, then code, then regenerate affected fixtures.
- Both R packages compose the same C++ implementation. Keep wrappers thin;
  inference policy belongs in C++. No exported name has different meanings in
  the two packages.

## What belongs in this repository

Track the library, bindings, tests, fixtures, experiments, benchmarks and public
maintainer docs. [.agents/skills](.agents/skills) holds versioned repository
workflows; agent configuration and session state do not belong there.

Manuscripts belong in `papers/<name>/`; non-magmaan research notes, talks,
handoffs, proposals, applications and paper evaluations belong in `private/<name>/`.
Each is its own independent Git repository, ignored by the outer repository;
see [private/README.md](private/README.md). Ask when ownership is genuinely unclear.

Until 0.2.0 is released, papers are out of scope unless the user asks for paper
work. Do not read, run, migrate or report on paper code, and do not let paper
callers constrain library or API changes. Papers stay on their pinned 0.1.0
packages and migrate after the release.

Third-party papers, supplements, data and source mirrors live in ignored
`external/` (`external/refs/` for PDFs), never in commits. Implement from formulas;
describe oracle agreement as behavior, not as a source-code port. Exceptions:
licensed build dependencies under `cpp/third_party/` (PORT and QUADPACK), and
public teaching datasets listed in [DATASETS.md](cpp/tests/fixtures/DATASETS.md).
Textbook-derived fixtures contain derived summaries only.

Archives, serialized objects, office documents, PDFs, build products, and
IDE/session state stay untracked. Experiment instructions define the narrow
exceptions for frozen CSV evidence and generated GitHub-facing `report.md`.
`just check-tracked` checks the index: known top-level entries, only allowlisted
convention files inside ignored mounts, prohibited artifacts, and the 1 MB
limit (with an explicit logo exception). New top-level entries need a deliberate
update to both these instructions and the checker.

## Dependency layering

Dependencies flow down; each paper, experiment and the test tree is a leaf.

- Inputs: built `cpp/third_party/` and optional ignored external resources.
- Core: `cpp/include/` and `cpp/src/` depend only on inputs.
- Interfaces/harnesses: `r-package/` depends on core; `r-magmaan/` on
  `r-package/`; `experiments/_support/` on core/bindings, with no SEM logic or
  paper/experiment-specific references; `benchmarks/` is the shared benchmark harness.
- Leaves: each `papers/<name>/`, each numbered experiment (research and engineering add
  `active/`, `banked/`, or `evidence/`), and `cpp/tests/`. No sibling-leaf
  imports or reads. Experiments may consume `_support`, benchmarks and the
  optional textbook corpus. Shared statistical primitives belong in core or
  bindings; shared harness mechanics belong in `_support` or benchmarks.
- Core and both R packages never reach into papers, experiments, benchmarks
  or tests. Nothing tracked depends on `private/`; private repositories may
  use magmaan freely. No active code depends on frozen `papers/_archive/`.
  Running a built executable under `cpp/build/` is allowed.

`just check-layering` scans R/C++/CMake/shell/just code with comments stripped;
it runs in CI and `just check`. Ignored papers are checked locally when present;
their archive is excluded. Reports read only their own `results/`; this rule is
reviewed manually because `.qmd` files are outside the checker. Documentation
may cite evidence without introducing a code dependency.

## Completion and Git

Use `snake_case` filenames/functions/constants and `CamelCase` types. Comments
explain non-obvious reasons. Use `just` from the root; `just build` / `just test`
use `fast`, `just test-dev` runs sanitizers, `just test-opt` runs Release/native.
Choose checks appropriate to the changed surface; nested guidance gives details.
Refresh vendors with `just vendor` after canonical C++ changes.

Update the roadmap when implementation state, architecture, contracts or
validation expectations change; capability detail lives in the area files under
`project/architecture/capabilities/`, and the roadmap holds current state,
contracts and the area index. Update the relevant active backlog when a
milestone finishes, priorities change or concrete work is discovered. Keep
simulation detail in its own backlog and only cross-domain summaries elsewhere.
May-never-build ideas belong in [speculative.md](project/backlog/speculative.md),
with a cheaper existing alternative and an explicit build-if trigger. Fold stale
finished plans into the maintained documents rather than creating parallel roadmaps.

Preserve unrelated changes and staged work. Stage explicit paths, never `git add .`
or `git add -A`; check new files belong here. Commit each finished user request as
a coherent change on the current branch unless the user asks otherwise. A
read-only review creates no artificial commit or session-state document.
