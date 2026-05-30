# External Resources Policy

The repository keeps the core source, tests, fixtures, benchmark manifests,
and maintainer notes in Git. Local reference material stays outside history.

## `external/` — the single "development help" folder

All outside material you read while developing lives under `external/`, which is
ignored (except its `README.md` convention doc). It is not part of the build, not
part of CI, and never a required test input. This replaces the former split
between `external/` and a separate `resources/` folder — there is now one place.

It holds:

- **Source mirrors** — `lavaan`, `robcat`, `kreiberg`, and similar checkouts read
  for implementation details.
- **`external/refs/`** — local PDFs (papers, textbooks), textbook companion files,
  and other reference artifacts whose redistribution terms are unclear or simply
  unnecessary for the build. Do not link tests or scripts to files that must exist
  under `external/refs/`.

Tracked research notes and simulation scripts live under `docs/research/`.

Textbook-corpus material (raw Mplus, Brown, Geiser, Kline, Little, Newsom
bundles) does **not** live here — it lives under the textbook-corpus submodule
at `corpus/textbook-corpus/raw/<book>/`, which is itself gitignored. The
parent-repo build/regen scripts under `tests/tools/` read from there.

`external/paper_corpus` is a special ignored nested Git repository for curated
paper-corpus work. It owns raw downloads, tracked minimal derived data/models,
raw-to-derived validation, and magmaan-facing JSON exports. magmaan consumes
only copied export snapshots under `tests/fixtures/paper_corpus/`; default
C++ tests do not read the nested repository.

Fixture regeneration uses installed R packages at the pinned versions recorded
in `tests/fixtures/`, then writes self-contained JSON fixtures. The checked-in
C++ tests consume those fixtures only; they do not run R and do not read
`external/`.

For regeneration:

- lavaan fixtures require the installed lavaan version to match
  `tests/fixtures/lavaan_version.txt`.
- robcat fixtures require the installed robcat version to match
  `tests/fixtures/robcat_version.txt`. If the pinned version is not available
  from a package repository, install it from a local source checkout or source
  archive outside Git, for example `R CMD INSTALL external/robcat` when that
  ignored checkout exists.

`external/lavaan` and `external/robcat` are therefore useful for source checks,
but the oracle is always generated package output, not vendored source code.
