# Near-Singular ML Continuation Experiment

This experiment probes a first, deliberately simple convergence aid for
complete-data normal-theory ML when the sample covariance is nearly singular:
fit a sequence of easier covariance problems,

```text
S_alpha = (1 - alpha) S + alpha diag(S),
```

warm-starting each stage from the previous estimate, and finish at
`alpha = 0` so the endpoint is the ordinary ML problem when `S` is positive
definite.

The C++ primitive lives in `estimate::frontier::fit_ml_ridge_continuation()`.
The R surface is exposed as
`magmaan_core$frontier_fit_ml_ridge_continuation()`.

## Run

Install the R package first if needed:

```sh
just r-install
```

Quick smoke:

```sh
Rscript experiments/04-near_singular_ml_continuation/run_experiment.R --reps 3
```

Default run:

```sh
Rscript experiments/04-near_singular_ml_continuation/run_experiment.R
```

Render the report:

```sh
(cd experiments/04-near_singular_ml_continuation && quarto render report.qmd)
```

## Outputs

Generated files live under `results/` and are ignored by git:

- `fits.csv`: one row per replicate and method.
- `summary.csv`: method-level convergence and cost summary.

The comparison is research tooling, not a default recommendation. Continuation
spends more optimizer calls than a single ML fit, so use `total_iterations`,
`total_f_evals`, and `total_g_evals` when comparing cost.

The runner defaults to `optimizer = "port"` because the trust-region backend is
stable enough to make the path diagnostics visible. Passing
`--optimizer nlopt-lbfgs` is useful for studying the line-search failures that
motivated this experiment.
