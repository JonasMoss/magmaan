# Categorical SEM Simulation Scaffolds

This folder keeps paper-adjacent simulation resources for future informal
magmaan x lavaan parity work. The PDFs are reference material; the R scripts in
[`r/`](r/) extract the data-generating designs into small, callable generators.

These scripts are intentionally not a committed simulation study. They do not
run thousands of replications by default, and they do not establish support
claims for magmaan. They are meant to make later MSE-style checks easy once the
relevant categorical estimators are implemented and hardened.

## Scripts

- `r/common.R` shared normal/categorical data generation, lavaan/magmaan fitting
  adapters, and MSE summaries.
- `r/li_2016_ordinal_sem.R` Li (2016): five-factor all-ordinal SEM with 4-7
  categories, three response distributions, and seven sample sizes.
- `r/li_2021_mixed_sem.R` Li (2021): mixed continuous/categorical three-factor
  CFA and five-factor SEM with 2-7 categories, symmetric/slightly asymmetric
  categorical variables, and three sample sizes.
- `r/lei_shiverdecker_2020_missing_ordinal_cfa.R` Lei & Shiverdecker (2020):
  two-factor all-ordinal CFA with missing-data mechanisms, exact threshold and
  MAR-slope tables from the article, and 2-5 categories.
- `r/rhemtulla_2012_categorical_cfa.R` Rhemtulla et al. (2012): two-factor CFA
  with 10 or 20 indicators, 2-7 categories, normal or nonnormal latent response
  variables, five threshold-pattern conditions, and four sample sizes.

## Example

From the repo root:

```r
source("research/sims/r/li_2016_ordinal_sem.R")

one <- li2016_generate(n = 300, categories = 5,
                       distribution = "slight", seed = 11)
lav <- sim_fit_lavaan(one$model, one$data, ordered = one$ordered,
                      estimator = "DWLS")
```

The fitting helpers are best-effort adapters. They load `lavaan` or `magmaan`
only when called, so the generators can be sourced without either package
attached.

## Notes

Rhemtulla et al. (2012) say the exact asymmetric threshold table is in the
supplemental materials rather than in the PDF. The script keeps the published
factorial design and uses deterministic threshold-pattern approximations for
the moderate/extreme and alternating conditions. Replace those thresholds if the
supplemental table is added later.
