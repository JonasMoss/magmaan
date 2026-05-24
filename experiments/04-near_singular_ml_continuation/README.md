# Near-Singular ML Continuation Experiment

This experiment probes a first, deliberately simple convergence aid for
complete-data normal-theory ML when the sample covariance is nearly singular:
fit a sequence of easier covariance problems,

```text
S_lambda = (1 - lambda) S + lambda diag(S),
```

warm-starting each stage from the previous estimate, and finish at
`lambda = 0` so the endpoint is the ordinary ML problem when `S` is positive
definite.

The C++ primitive lives in `estimate::frontier::fit_ml_ridge_continuation()`.
The R surface is exposed as
`magmaan_core$frontier_fit_ml_ridge_continuation()`.
The default path is `0.50, 0.20, 0.10, 0.05, 0.01, 0.00`; the last endpoint is
appended by the C++/R wrapper when `include_endpoint = TRUE`.
Available targets are `diagonal`, `scaled_identity`, and `identity`.

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

Convergence-note comparison:

```sh
Rscript experiments/04-near_singular_ml_continuation/run_convergence_note_comparison.R
```

By default, the convergence-note runner crosses targets
`diagonal,scaled_identity,identity` with four lambda profiles:

- `legacy`: `0.50,0.20,0.10,0.05,0.01,0`
- `light`: `0.10,0.05,0.02,0.01,0.005,0`
- `endpoint`: `0.20,0.10,0.05,0.025,0.0125,0.006,0.003,0.001,0`
- `dense`: `0.50,0.35,0.25,0.18,0.13,0.09,0.06,0.04,0.025,0.015,0.008,0`

Render the report:

```sh
(cd experiments/04-near_singular_ml_continuation && quarto render report.qmd)
```

## Outputs

Generated files live under `results/` and are ignored by git:

- `fits.csv`: one row per replicate and method.
- `summary.csv`: method-level convergence and cost summary.
- `convergence_note_fits.csv`: one row per convergence-note replicate and
  method.
- `convergence_note_summary.csv`: convergence-note method summary by design.
- `convergence_note_paired.csv`: paired baseline-vs-continuation outcomes by
  design.

The comparison is research tooling, not a default recommendation. Continuation
spends more optimizer calls than a single ML fit, so use `total_iterations`,
`total_f_evals`, and `total_g_evals` when comparing cost.
The `converged` columns in the generated CSVs use magmaan's terminal audit,
not the raw backend stop code: the final point must satisfy the lavaan-style
projected-gradient check (`||Pg||_inf <= 0.001` by default). The raw optimizer
verdict is retained separately as `optimizer_converged` / `optimizer_status`.

The runner defaults to `optimizer = "port"` because the trust-region backend is
stable enough to make the path diagnostics visible. Passing
`--optimizer nlopt-lbfgs` is useful for studying the line-search failures that
motivated this experiment.
