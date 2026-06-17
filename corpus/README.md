# corpus/ — optional private real-data corpus

This directory is the mount point for `textbook-corpus`, a **private, optional**
collection of real SEM datasets and cases curated from textbooks and software
manuals (Kline, Little, Newsom, the Mplus User Guide, etc.). It is **not part of
the public magmaan repository** because those datasets are third-party
copyrighted material that magmaan may not redistribute.

## What depends on it

Nothing in the default build or test suite. The C++ golden tests
(`tests/golden/textbook_corpus_golden_test.cpp`,
`tests/golden/paper_corpus_golden_test.cpp`) read **checked-in JSON fixtures**
under `tests/fixtures/` that carry only *derived summary statistics*
(`sample_cov`, `sample_mean`, fitted results) — never raw casewise rows — so the
suite stays green without this corpus.

The corpus is required only to **regenerate** those fixtures or to run the
corpus-dependent research experiments:

- `experiments/00-lavaan-parity/`, `experiments/01-complete-data-estimator-speed/`
- the maintainer-only regenerators under `tests/tools/` (`build_*_corpus.R`,
  `regen_*_corpus*.R`, `regen_little_newsom_fixtures.R`, etc.)

Those entry points detect the corpus via
`experiments/_support/R/helpers.R::corpus_available()` and skip (or fail with a
clear message) when it is absent.

## Providing the corpus

Mount your private corpus at `corpus/textbook-corpus/` (a working copy, symlink,
or a privately-hosted git checkout). It is gitignored so it is never committed
to this repository. Expected layout: a top-level `manifest.csv` plus
`cases/<book>/<case_id>/` and `raw/<book>/` trees, as consumed by
`experiments/00-lavaan-parity/R/corpus.R`.
