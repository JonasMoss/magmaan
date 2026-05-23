# Latent Metric Identification Experiment

This experiment compares two equivalent latent-scale identification conventions:

- `marker`: first loading fixed, latent variance free.
- `std_lv`: latent variance fixed, first loading free.

It runs complete-data ML fits over continuous textbook-corpus fixtures and
records optimizer diagnostics, timing, and model-implied moment agreement.
The case filter is `measurement_kind == "continuous"`, not `observed_only`,
and the model contains at least one `=~` operator. Cases where the two
parameterizations would not produce the same number of free parameters
(e.g. multi-factor models with cross-loading equality constraints) are
skipped automatically.

## Run

Install the R package first if needed:

```sh
just r-install
```

Quick smoke:

```sh
Rscript experiments/02-latent_metric_identification/run_experiment.R --reps 1 --cases ^geiser
```

Default experiment (10 reps × `nlopt-lbfgs,port` × all eligible cases):

```sh
Rscript experiments/02-latent_metric_identification/run_experiment.R
```

The runner prints a headline summary at the end (median elapsed ratio,
% std_lv faster, per-backend medians, skip/error counts) so the result is
readable without rendering the report.

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
