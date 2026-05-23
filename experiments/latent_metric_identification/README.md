# Latent Metric Identification Experiment

This experiment compares two equivalent latent-scale identification conventions:

- `marker`: first loading fixed, latent variance free.
- `std_lv`: latent variance fixed, first loading free.

It runs complete-data ML fits over strict continuous textbook-corpus fixtures and
records optimizer diagnostics, timing, and model-implied moment agreement.

## Run

Install the R package first if needed:

```sh
just r-install
```

Quick smoke:

```sh
Rscript experiments/latent_metric_identification/run_experiment.R --reps 1 --backends nlopt-lbfgs
```

Default experiment:

```sh
Rscript experiments/latent_metric_identification/run_experiment.R
```

Render the report:

```sh
(cd experiments/latent_metric_identification && quarto render report.qmd)
```

Generated CSV files live under `results/` and are intentionally ignored by git;
`report.html` is also ignored.

## Outputs

- `results/fits.csv`: one row per attempted fit.
- `results/pairs.csv`: marker-vs-`std_lv` paired comparisons.
- `results/summary.csv`: grouped summary by backend, source, and family.

The comparison treats implied-moment agreement as a validity check. Differences
in raw parameter values are expected because the two conventions use different
latent scales.
