# Third-party datasets in test fixtures

Some checked-in fixtures under `cpp/tests/fixtures/` embed **raw casewise data** from
well-known, publicly distributed SEM teaching datasets. These datasets are the
de facto standards used across the SEM literature and ship with widely-used R
packages. They are reproduced here, unmodified, solely to validate magmaan
against lavaan on the same inputs. magmaan itself is MIT-licensed; the datasets
below retain the terms of their original distributions, noted per entry.

If you redistribute this repository, keep this file with it.

## Datasets

### HolzingerSwineford1939

- **Fixtures:** every `*_hs*` / `*_school*` case under `cpp/tests/fixtures/fit/`,
  `cpp/tests/fixtures/fiml/`, `cpp/tests/fixtures/ls/`, `cpp/tests/fixtures/composite/`, and
  `cpp/tests/fixtures/parity/hs_3factor_*` (n = 301; multigroup splits Pasteur
  n = 156, Grant-White n = 145; 9 cognitive-test scores `x1`–`x9`).
- **Source:** Holzinger, K. J., & Swineford, F. (1939). *A study in factor
  analysis: The stability of a bi-factor solution.* Supplementary Educational
  Monographs, No. 48. University of Chicago.
- **Obtained from:** the `HolzingerSwineford1939` data object in the
  [lavaan](https://lavaan.ugent.be) R package.
- **License:** lavaan is distributed under the GPL (>= 2). The underlying 1939
  data are in the public domain by age in most jurisdictions.

### PoliticalDemocracy

- **Fixtures:** `cpp/tests/fixtures/parity/bollen_democracy_sem/` (n = 75, 11
  observed variables).
- **Source:** Bollen, K. A. (1989). *Structural Equations with Latent
  Variables.* New York: Wiley. Industrialization and political democracy in 75
  developing countries (1960 / 1965).
- **Obtained from:** the `PoliticalDemocracy` data object in the lavaan R
  package.
- **License:** lavaan is distributed under the GPL (>= 2).

### Demo.growth

- **Fixtures:** `cpp/tests/fixtures/parity/demo_growth_linear/` (n = 400).
- **Source:** a simulated latent-growth demonstration dataset created by the
  lavaan authors.
- **Obtained from:** the `Demo.growth` data object in the lavaan R package.
- **License:** lavaan is distributed under the GPL (>= 2).

### bfi (Big Five / SAPA)

- **Fixtures:** `cpp/tests/fixtures/parity/bfi_*` (n = 2800 / 2436; 25 personality
  items `A1`–`O5`, five each for Agreeableness, Conscientiousness, Extraversion,
  Neuroticism, Openness).
- **Source:** 25 self-report items from the International Personality Item Pool
  (IPIP), collected through the Synthetic Aperture Personality Assessment (SAPA;
  https://sapa-project.org) project.
- **Obtained from:** the `bfi` data object in the
  [psych](https://personality-project.org/r/psych/) / `psychTools` R packages
  (W. Revelle).
- **License:** psych / psychTools are distributed under the GPL (>= 2). The SAPA
  / IPIP items are made freely available for research and teaching.

### Demo.twolevel

- **Fixtures:** `cpp/tests/fixtures/twolevel/` (the first 80 clusters, n = 1000;
  see `cpp/tests/tools/regen_oracle_twolevel.R`).
- **Source:** a simulated two-level demonstration dataset created by the lavaan
  authors.
- **Obtained from:** the `Demo.twolevel` data object in the lavaan R package.
- **License:** lavaan is distributed under the GPL (>= 2).

### semfindr `pa_dat`

- **Fixtures:** `cpp/tests/fixtures/case_influence/path_pa.json` (n = 100, 4
  observed variables). Its sibling `cfa_hs.json` uses HolzingerSwineford1939
  (above).
- **Obtained from:** the `pa_dat` example dataset in the
  [semfindr](https://CRAN.R-project.org/package=semfindr) R package; see
  `cpp/tests/tools/regen_semfindr_fixtures.R`.
- **License:** semfindr is distributed under the GPL (>= 3).

## Fixtures that are *not* third-party data

- **Synthetic fixtures.** `cpp/tests/fixtures/ordinal/`,
  `cpp/tests/fixtures/mixed_ordinal/`, and `cpp/tests/fixtures/score/` use data
  simulated specifically for these tests (descriptive case names, round sample
  sizes). They carry no third-party rights.
- **Summary-statistic-only fixtures.** `cpp/tests/fixtures/little/`,
  `cpp/tests/fixtures/newsom/`, `cpp/tests/fixtures/geiser/`,
  `cpp/tests/fixtures/mplus_sem/`, `cpp/tests/fixtures/textbook_corpus/`, and
  `cpp/tests/fixtures/paper_corpus/` carry model syntax, *derived* summary
  statistics (`sample_cov`, `sample_mean`, `n_obs`) and fitted lavaan results
  for their cases, never raw casewise rows. The textbook data those statistics
  were computed from are publicly downloadable from the books' companion sites;
  the original files live in the optional corpus mount (see
  [`project/reference/textbook-corpus.md`](../../project/reference/textbook-corpus.md)), not in this repository.
  `paper_corpus/` is derived from a public OSF project.
