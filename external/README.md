# External material

Optional local inputs, ignored by magmaan except this guide:

- `textbook-corpus/`: textbook datasets, cases, and raw downloads. See
  [the corpus guide](../project/reference/textbook-corpus.md).
- `paper-corpus/`: independent Git repository owning paper ingestion, derived
  cases, validation, and exports.
- `paper-downloads/`: source archives used by experiments showcases/06 (Etzel) and 32
  (Schlechter). These are still required by those experiments.
- `refs/`: reference papers and books.
- Other source mirrors may be placed here for reference; they are not built.

C++ tests normally consume committed snapshots in `cpp/tests/fixtures/`.
Two optional textbook checks also read the mounted case collection and skip
when it is absent. Paper-corpus exports are copied by the bridge scripts in
`cpp/tests/tools/`; moving the collections does not regenerate them.
