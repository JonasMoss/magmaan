# Standardized score-flip probe

This experiment probes the first `score_flip_test()` frontier implementation on
the configural-to-metric continuous ML invariance comparison. It crosses
`n = 30, 50, 100` per group, normal versus standardized `t(5)` observations,
and equal versus threefold factor-variance heterogeneity. One local loading-
noninvariance cell is included as a coarse power check.

The comparison separates basic score flips, nuisance-effective flips,
flip-specifically standardized effective flips, three asymptotic score
references, and Satorra-2000 scaled/mixture difference tests. Comparator
failure does not discard an otherwise valid flip result; fit failures and
method-specific available-replication counts are recorded separately.

```sh
Rscript experiments/62-standardized-flip-score/run_experiment.R --smoke
Rscript experiments/62-standardized-flip-score/run_experiment.R --probe
quarto render experiments/62-standardized-flip-score/report.qmd
```

The default probe uses 300 replications per cell and 499 random flips. Its
result is a calibration signal, not a validation or support claim.

Method sources:

- Hemerik, Goeman & Finos (2020), *JRSS B*,
  <https://doi.org/10.1111/rssb.12369>.
- De Santis, Goeman, Hemerik, Davenport & Finos (2024),
  <https://doi.org/10.48550/arXiv.2209.13918>.
