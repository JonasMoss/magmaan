# Mplus SEM Fixtures

Derived lavaan oracles for the ignored local Mplus SEM corpus in
`external/textbook-corpus/raw/mplus_sem`. Regenerate with:

```sh
Rscript cpp/tests/tools/build_mplus_sem_corpus.R
Rscript cpp/tests/tools/regen_mplus_sem_fixtures.R
```

The builder uses the textbook corpus's shared Mplus translator
(`external/textbook-corpus/ingest/_mplus_helpers.R`) and retains an example
only when lavaan's fit of the translation reproduces the shipped Mplus `.out`:
analysed N, free parameters, df, chi-square, H0 log-likelihood and every
printed estimate. The source corpus may contain Mplus `.inp`, `.out`, and data
files, but those stay under the ignored `external/textbook-corpus/raw/` landing
zone. This directory commits only classification metadata, sample statistics,
weight matrices, and lavaan fit outputs.

Current strict coverage is the continuous growth tranche in
`continuous_reference.json`: seven Mplus User's Guide chapter 6 cases (ex6.1,
6.8, 6.9, 6.10, 6.11, 6.13, 6.14), each fitted with ML, ULS, GLS, and WLS
through `lavaan::sem` on explicit syntax, with x variables fixed
(`fixed_x = true`) and Mplus's mean structure. Retained cases outside that
tranche (observed-variable path models, multi-group, FIML, and nonlinear
constraints) are listed in `manifest.json`. The ordinal and mixed files are
empty: no categorical Mplus example passes the source-fidelity screen.
