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

## Fixtures that are *not* third-party data

- **Synthetic fixtures.** `tests/fixtures/ordinal/`,
  `tests/fixtures/mixed_ordinal/`, and `tests/fixtures/score/` use data
  simulated specifically for these tests (descriptive case names, round sample
  sizes). They carry no third-party rights.
- **Summary-statistic-only fixtures.** `tests/fixtures/textbook_corpus/` and
  `tests/fixtures/paper_corpus/` carry only *derived* summary statistics
  (`sample_cov`, `sample_mean`, fitted results) for their cases — never raw
  casewise rows. The raw textbook corpora those statistics were computed from are
  a private, optional dependency (see [`corpus/README.md`](../../corpus/README.md))
  and are not included in this repository.
