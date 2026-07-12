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

The expansion asks when standardization is worth its cost. It holds a
two-factor, 62-slot ambient model fixed and crosses 1/4/8 loading restrictions,
total `N = 60/100/200`, 1:1 versus 1:3 allocation, three group-information
geometries, and normal, `t(5)`, and skewed innovations. Its power mode adds
sparse and dense equal-Euclidean-norm loading violations at two effect sizes.
The comparator battery includes Satorra-2000 unscaled/SB/MV-adjusted/
scaled-shifted/exact-mixture tests and FMG SB/MV/SS/SF, EBA2/4/6,
pEBA2/4/6, PALL, pOLS, and ALL. The report presents both nominal power and
power after method- and design-cell-specific empirical-null calibration.

The timing runner uses warm serial calls to separate setup, basic/effective
resampling, flip-standardization, and asymptotic-comparator cost. The expansion
also records the two-fit, nested-test, and full FMG-battery latency separately.

```sh
Rscript experiments/62-standardized-flip-score/run_experiment.R --smoke
Rscript experiments/62-standardized-flip-score/run_experiment.R --probe
Rscript experiments/62-standardized-flip-score/run_expansion.R --probe
Rscript experiments/62-standardized-flip-score/run_expansion.R --power
Rscript experiments/62-standardized-flip-score/run_timing.R
quarto render experiments/62-standardized-flip-score/report.qmd
```

The original probe uses 300 replications per cell; the null expansion uses 200
per cell and the power expansion uses 150. All use 499 random flips. `--smoke`
runs two representative cells, `--probe` runs the null grid, `--power` runs the
alternative grid, and `--full` runs both grids at 150 replications per cell
unless `--reps` overrides it. Results are calibration signals, not validation
or support claims.

Method sources:

- Hemerik, Goeman & Finos (2020), *JRSS B*,
  <https://doi.org/10.1111/rssb.12369>.
- De Santis, Goeman, Hemerik, Davenport & Finos (2024),
  <https://doi.org/10.48550/arXiv.2209.13918>.
