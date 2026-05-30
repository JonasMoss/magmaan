# `external/` — development help

Everything under `external/` is **development help**: source mirrors and reference
material you read while developing magmaan. It is **ignored by Git** (this README
is the one tracked exception) — never built, not part of CI, and never a required
test or build input. Anything here can be deleted and re-fetched at will.

This folder replaces the old split between `external/` and `resources/`; there is
now a single place for "outside material you read."

## Layout

```
external/
├── README.md        ← this file (tracked)
├── lavaan/          ← source mirror: read lavaan's R implementation
├── robcat/          ← source mirror: robust polychoric R package
├── kreiberg/        ← reference: Kreiberg's Matlab SNLRLS/SNLLS code (cited in src comments)
├── paper_corpus/    ← nested git repo for curated paper-corpus work (see below)
└── refs/            ← reference PDFs + textbook companions you read, not source code
```

You won't have all of these on a fresh clone — add only what you need.

## What goes where

- **Source mirrors** (`lavaan/`, `robcat/`, `kreiberg/`) — upstream code checked out
  locally to read implementation details. The mirror paths are referenced by name
  from R fixture-regen tooling (`tests/tools/regen_*_fixtures.R`) and a source
  comment in `include/magmaan/estimate/fit.hpp`, so keep these top-level names
  stable if present.
- **`paper_corpus/`** — a special ignored *nested* Git repository owning raw paper
  downloads, derived data, and magmaan-facing JSON exports. magmaan consumes only
  the copied export snapshots under `tests/fixtures/paper_corpus/`; the nested repo
  itself is never read by the C++ tests.
- **`refs/`** — reference PDFs (papers, textbooks) and textbook companion files
  whose redistribution terms are unclear or simply unneeded for the build. Read-only
  reference; nothing in code may depend on a file existing here.

## What does NOT go here

- **Textbook-corpus material** (raw Mplus/Brown/Geiser/Kline/Little/Newsom bundles)
  lives under the `corpus/textbook-corpus` submodule at `corpus/textbook-corpus/raw/<book>/`.
- **Self-generated test material** (smoke goldens, fixtures) belongs under `tests/`
  — it is not "external." The live smokes are `tests/golden/` (C++) and
  `r-package/examples/` (R).

See `docs/reference/external_resources.md` for the full policy and the oracle model
(fixtures come from installed R packages, not vendored source).
