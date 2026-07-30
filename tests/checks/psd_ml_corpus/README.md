# PSD-ML corpus audit

Advisory complete-data NTML check over the checked-in continuous corpus
fixtures. It asks which ordinary fits violate the primitive `Theta`/`Psi`
covariance cone and how the estimates and objective change under PSD-ML.

Run after installing the current R package:

```sh
just -f tests/checks/psd_ml_corpus/justfile audit
```

The default refits every case with PSD-ML, which checks interior equivalence as
well as inadmissible repairs. `audit-first` refits only cases rejected by the
ordinary post-fit audit. Generated CSV files stay under `results/`.

The interpretation and July 2026 decision record are in
`docs/research/notes/psd_ml_corpus_audit.tex`.

Four geometry-diverse findings are also checked in as the compact default-suite
fixture `tests/fixtures/psd_ml/corpus_geometries.json`, exercised by
`tests/unit/psd_ml_corpus_test.cpp`. Regenerate that slice from the two source
fixtures after installing the current package:

```sh
Rscript tests/tools/regen_psd_ml_corpus_geometries.R
```

This regeneration runs only the four promoted cases; it does not replace the
full advisory audit.
