# Complete-Data Estimator Timing

This small experiment compares magmaan's complete-data estimators across the
checked-out textbook corpus at `corpus/textbook-corpus`:

- `NT`: normal-theory ML, via `magmaan_core$fit_ml()`.
- `ULS`: unweighted least squares, via `magmaan_core$fit_uls()`.
- `GLS`: generalized least squares, via `magmaan_core$fit_gls()`.

The comparison is curiosity-grade rather than a correctness claim. The report
focuses on wall time and timing ratios, using geometric means for aggregate
comparisons. Ordered and non-complete-data cases are excluded up front because
these estimator paths expect complete-data continuous sample statistics; cases
that still fail during model construction or fitting are summarized in the
report.

## Run

Install the R package first if needed:

```sh
just r-install
```

Quick smoke:

```sh
Rscript experiments/01-complete_data_estimator_speed/run_experiment.R --reps 1 --cases 'brown_2015_tab4_1_neuroticism_extraversion'
```

Default experiment:

```sh
Rscript experiments/01-complete_data_estimator_speed/run_experiment.R
```

Render the report:

```sh
(cd experiments/01-complete_data_estimator_speed && quarto render report.qmd)
```

Generated CSV files live under `results/` and are ignored by git; `report.html`
is also ignored.

## Outputs

- `results/fits.csv`: one row per attempted estimator fit.
- `results/pairs_vs_nt.csv`: ULS/GLS paired against NT by case and replicate.
- `results/case_summary.csv`: label-sorted case-level timing summary.
- `results/group_summary.csv`: total, subcorpus, and tag timing summaries.
- `results/coverage.csv`: corpus rows considered, timed candidates, and
  pre-timing skip reasons.
