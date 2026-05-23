# Complete-Data Estimator Timing

This small experiment compares magmaan's complete-data estimators across the
continuous complete-data textbook fixtures:

- `NT`: normal-theory ML, via `magmaan_core$fit_ml()`.
- `ULS`: unweighted least squares, via `magmaan_core$fit_uls()`.
- `GLS`: generalized least squares, via `magmaan_core$fit_gls()`.

The comparison is curiosity-grade rather than a correctness claim. The report
focuses on wall time and timing ratios, using geometric means for aggregate
comparisons. Categorical and mixed fixtures are outside this experiment because
these estimator paths expect complete-data continuous sample statistics.

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
- `results/case_summary.csv`: label-sorted case-level timing summary.
- `results/group_summary.csv`: total, subcorpus, and tag timing summaries.
