# Fixture regeneration

Resolve paths from the magmaan root, identified by `cpp/CMakeLists.txt` and
`r-package/DESCRIPTION`. Inspect the existing fixture and its generating script
before selecting a command. `cpp/tests/tools/regen_oracle.R` is the main lavaan
generator, exposed as `just regen-oracle`; specialized features also have tools
in `cpp/tests/tools/`. Read the relevant generator's actual arguments rather
than assuming a feature filter or `--check` mode exists.

The main generator checks installed lavaan against
`cpp/tests/fixtures/lavaan_version.txt`, normalizing hyphen/dot spellings. Do not
change the pin merely to bypass a mismatch. An intentional version update
requires reviewing its affected fixtures and comparison behavior.

Preserve unrelated fixture edits. A broad generator may rewrite more than the
target feature: inventory its outputs first, review all diffs, and stage only
intentional changes. Prefer a temporary checkout/output when supported by the
generator if unrelated working-tree data would be overwritten. New fixtures
need reproducible model/options, source/version provenance, and a C++ consumer
test. Do not hand-adjust oracle values to match magmaan.

Store third-party PDFs/data/source mirrors under ignored `external/`. Dataset
exceptions and provenance are in `cpp/tests/fixtures/DATASETS.md`; copyrighted
textbook fixtures carry summary statistics only. CI's C++ tests consume frozen
JSON, while selected R integration tests can compare live installed lavaan.

Validate the affected parity component with its focused test area and inspect
numeric/row changes within the documented tolerances. Run structural guards
when adding artifacts; refresh C++ vendors if canonical code changed. Report
the actual generator and oracle version used and unexplained differences.
