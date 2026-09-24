# Third-party datasets in test fixtures

Some checked-in fixtures under `tests/fixtures/` embed **raw casewise data** from
well-known, publicly distributed SEM teaching datasets. These datasets are the
de facto standards used across the SEM literature and ship with widely-used R
packages. They are reproduced here, unmodified, solely to validate magmaan
against lavaan on the same inputs. magmaan itself is MIT-licensed; the datasets
below retain the terms of their original distributions, noted per entry.

If you redistribute this repository, keep this file with it.

## Datasets

### HolzingerSwineford1939

- **Fixtures:** every `*_hs*` / `*_school*` case under `tests/fixtures/fit/`,
  `tests/fixtures/fiml/`, `tests/fixtures/ls/`, `tests/fixtures/composite/`, and
  `tests/fixtures/parity/hs_3factor_*` (n = 301; multigroup splits Pasteur
  n = 156, Grant-White n = 145; 9 cognitive-test scores `x1`–`x9`).
- **Source:** Holzinger, K. J., & Swineford, F. (1939). *A study in factor
  analysis: The stability of a bi-factor solution.* Supplementary Educational
  Monographs, No. 48. University of Chicago.
- **Obtained from:** the `HolzingerSwineford1939` data object in the
  [lavaan](https://lavaan.ugent.be) R package.
- **License:** lavaan is distributed under the GPL (>= 2). The underlying 1939
  data are in the public domain by age in most jurisdictions.

### PoliticalDemocracy

- **Fixtures:** `tests/fixtures/parity/bollen_democracy_sem/` (n = 75, 11
  observed variables).
- **Source:** Bollen, K. A. (1989). *Structural Equations with Latent
  Variables.* New York: Wiley. Industrialization and political democracy in 75
  developing countries (1960 / 1965).
- **Obtained from:** the `PoliticalDemocracy` data object in the lavaan R
  package.
- **License:** lavaan is distributed under the GPL (>= 2).

### Demo.growth

- **Fixtures:** `tests/fixtures/parity/demo_growth_linear/` (n = 400).
- **Source:** a simulated latent-growth demonstration dataset created by the
  lavaan authors.
- **Obtained from:** the `Demo.growth` data object in the lavaan R package.
- **License:** lavaan is distributed under the GPL (>= 2).

### bfi (Big Five / SAPA)

- **Fixtures:** `tests/fixtures/parity/bfi_*` (n = 2800 / 2436; 25 personality
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

- **Fixtures:** `tests/fixtures/twolevel/` (the first 80 clusters, n = 1000;
  see `tests/tools/regen_oracle_twolevel.R`).
- **Source:** a simulated two-level demonstration dataset created by the lavaan
  authors.
- **Obtained from:** the `Demo.twolevel` data object in the lavaan R package.
- **License:** lavaan is distributed under the GPL (>= 2).

### semfindr `pa_dat`

- **Fixtures:** `tests/fixtures/case_influence/path_pa.json` (n = 100, 4
  observed variables). Its sibling `cfa_hs.json` uses HolzingerSwineford1939
  (above).
- **Obtained from:** the `pa_dat` example dataset in the
  [semfindr](https://CRAN.R-project.org/package=semfindr) R package; see
  `tests/tools/regen_semfindr_fixtures.R`.
- **License:** semfindr is distributed under the GPL (>= 3).

## Fixtures that are *not* third-party data

- **Synthetic fixtures.** `tests/fixtures/ordinal/`,
  `tests/fixtures/mixed_ordinal/`, and `tests/fixtures/score/` use data
  simulated specifically for these tests (descriptive case names, round sample
  sizes). They carry no third-party rights.
- **Summary-statistic-only fixtures.** `tests/fixtures/little/`,
  `tests/fixtures/newsom/`, `tests/fixtures/geiser/`,
  `tests/fixtures/mplus_sem/`, `tests/fixtures/textbook_corpus/`, and
  `tests/fixtures/paper_corpus/` carry model syntax, *derived* summary
  statistics (`sample_cov`, `sample_mean`, `n_obs`) and fitted lavaan results
  for their cases, never raw casewise rows. The textbook data those statistics
  were computed from are publicly downloadable from the books' companion sites;
  the original files live in the optional corpus mount (see
  [`corpus/README.md`](../../corpus/README.md)), not in this repository.
  `paper_corpus/` is derived from a public OSF project.
