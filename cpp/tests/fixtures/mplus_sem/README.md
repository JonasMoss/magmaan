# Mplus SEM Fixtures

Derived lavaan oracles for the ignored local Mplus SEM corpus in
`external/textbook-corpus/raw/mplus_sem`. Regenerate with:

```sh
Rscript cpp/tests/tools/build_mplus_sem_corpus.R
Rscript cpp/tests/tools/regen_mplus_sem_fixtures.R
```

The source corpus may contain Mplus `.inp`, `.out`, and `.dat` files, but those
stay under the ignored `external/textbook-corpus/raw/` landing zone. This
directory commits only classification metadata, sample statistics, weight
matrices, and lavaan fit outputs.

Current strict coverage is the continuous growth tranche in
`continuous_reference.json`: six cases, each fitted with ML, ULS, GLS, and WLS.
The ordinal and mixed files record retained categorical examples that are not
v1-tested because they are observed-response/logistic or otherwise outside
magmaan's current ordinal/mixed LS model surface.
