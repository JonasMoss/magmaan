# Optional real-data textbook corpus

`external/textbook-corpus/` is the mount point for an **optional**
collection of real SEM datasets and cases curated from textbooks and software
manuals (Kline, Little, Newsom, Geiser, the Mplus User's Guide, etc.). The
datasets are publicly downloadable from the books' companion sites. The corpus
is kept out of this repository because it holds those original files as
distributed; the repository carries only what is derived from them.

## What the repository carries instead

The C++ golden tests read **checked-in JSON fixtures** built from the corpus:
`cpp/tests/fixtures/{little,newsom,geiser,mplus_sem,textbook_corpus}/`.
They hold model syntax, *derived summary statistics* (`sample_cov`,
`sample_mean`, `n_obs`) and lavaan's fitted results, never raw casewise rows.
See [`cpp/tests/fixtures/DATASETS.md`](../../cpp/tests/fixtures/DATASETS.md).

## What depends on the mount

- **Two golden checks** in `cpp/tests/golden/textbook_corpus_golden_test.cpp`
  (implied moments at lavaan's θ for `newsom_2015_ex9_3` and
  `little_2013_ch3_fig_3_6_1indicator`) read case files straight from the
  mount. They need no data, only model syntax, options, and lavaan's θ and
  implied moments, and they **skip** when the mount is absent. Exporting the two
  cases with `cpp/tests/tools/regen_textbook_case_fixtures.R` would let them run
  everywhere.
- **Fixture regeneration**: the maintainer-only regenerators under
  `cpp/tests/tools/` (`build_*_corpus.R`, `regen_*_corpus*.R`,
  `regen_little_newsom_fixtures.R`, `regen_textbook_case_fixtures.R`, etc.).
- **Research experiments**: `experiments/00-lavaan-parity/` and
  `experiments/01-complete-data-estimator-speed/`. These entry points detect
  the corpus via `experiments/_support/R/helpers.R::corpus_available()` and skip
  (or fail with a clear message) when it is absent.

## Providing the corpus

Mount the corpus at `external/textbook-corpus/` (a working copy, symlink, or git
checkout). It is gitignored so it is never committed to this repository.
Expected layout: a top-level `manifest.csv` plus `cases/<book>/<case_id>/` and
`raw/<book>/` trees, as consumed by `experiments/00-lavaan-parity/R/corpus.R`.
`raw/<book>/` keeps each book's original downloads with SHA-256 sums
(`archives.json`, `SHA256SUMS`).
