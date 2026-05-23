# Complete-Data Estimator Timing

This small experiment compares magmaan's complete-data estimators across the
textbook corpus:

- `NT`: normal-theory ML, via `magmaan_core$fit_ml()`.
- `ULS`: unweighted least squares, via `magmaan_core$fit_uls()`.
- `GLS`: generalized least squares, via `magmaan_core$fit_gls()`.

The comparison is curiosity-grade rather than a correctness claim. It records
wall time, optimizer diagnostics, parameter-estimate drift from NT, implied
moment drift from NT, and a naive chi-square-like statistic computed as
`magmaan_core$infer_chi2_stat(sample_stats, fit$fmin)`.

## Run

Install the R package first if needed:

```sh
just r-install
```

Quick smoke:

```sh
Rscript experiments/complete_data_estimators/run_experiment.R --reps 1 --cases '^geiser::latent_regression$'
```

Default experiment:

```sh
Rscript experiments/complete_data_estimators/run_experiment.R
```

Render the report:

```sh
(cd experiments/complete_data_estimators && quarto render report.qmd)
```

Generated CSV files live under `results/` and are ignored by git; `report.html`
is also ignored.

## Outputs

- `results/fits.csv`: one row per attempted estimator fit.
- `results/pairs_vs_nt.csv`: ULS/GLS paired against NT by case and replicate.
- `results/case_summary.csv`: label-sorted case-level timing and drift summary.
