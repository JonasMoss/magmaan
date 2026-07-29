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
