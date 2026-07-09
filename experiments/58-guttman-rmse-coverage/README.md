# Experiment 58: Guttman RMSE and Coverage

This experiment is an overnight-style RMSE, coverage, failure, and timing grid
for the current Guttman CFA estimators.

The categorical arm is intentionally **not polychoric**. It uses
`magmaan::sim_ordcorr_calibrate(metric = "pearson_codes")` to generate ordinal
category scores whose observed Pearson-code correlation is the CFA target. The
Guttman, NTML, and ULS arms then all consume the same ordinary sample covariance
of those numeric category scores.

## What It Compares

Configural cells:

- `guttman_std`: GLS-aligned Guttman map with standardized composite weights.
- `guttman_gls`: GLS-aligned Guttman map with data-dependent aligned weights.
- `guttman_unit`: GLS-aligned Guttman map with unit incidence weights.
- `guttman_legacy`: legacy Spearman/incidence Guttman map.
- `ml_emp`: normal-theory ML point estimates with empirical sandwich SE.
- `uls_emp`: ULS point estimates with empirical sandwich SE.

Constrained cells:

- `guttman_std_restricted`: estimator-side residual/loading restricted Guttman.
- `guttman_gls_restricted`: same with GLS-aligned composites.
- `ml_emp_restricted`: iterative NTML on the same restricted partable.
- `uls_emp_restricted`: iterative ULS on the same restricted partable.

The constrained population is true tau-equivalent within each factor with equal
within-factor residual variances, so the restricted estimators are evaluated on
their intended target rather than under misspecification.

The Guttman arms use `noniterative_cfa_se()` for empirical SE timing. The full
`noniterative_cfa_inference()` GOF bundle is intentionally not called here,
because this RMSE/coverage grid only consumes standard errors.

## Default Full Grid

`--full` currently expands to:

- generators: `normal`, `ig`, `ordinal`
- factors: `2`, `3`, `5`
- indicators per factor: `3`, `5`, `u` (`u` = unbalanced 3/6 alternation)
- latent correlations: `0`, `0.35`, `0.6`, `0.8`
- loading strength: `moderate`, `weak` (`weak` scales loadings to ~0.62x)
- sample sizes: `50`, `100`, `300`, `800`
- configural scales: `equal`, `unequal`
- configural loading pattern: `mixed`
- restricted cells: `tau_resid` with equal scales and tau loadings

That is 2592 cells: 1728 configural plus 864 constrained. The
independent-generator stress arm uses a mild Tukey g-and-h target, skewness
`0.2` and excess kurtosis `0.5`, because stronger targets failed calibration for
the 25-variable cells. The higher-rho / weak-loading / unbalanced arms are new,
aimed at finally stressing the aligned maps' cross-factor information and the
communality estimators where average-of-ratios should hurt. Trim any axis with
its flag (e.g. `--strength moderate`, `--indicators 3,5`, `--n 100,300,800`) or
lower `--reps` to shrink the grid.

## How To Start It

From the repo root, run a small smoke check first:

```sh
Rscript experiments/58-guttman-rmse-coverage/run_experiment.R \
  --probe \
  --results-dir experiments/58-guttman-rmse-coverage/results/probe
```

Then run the overnight grid:

```sh
Rscript experiments/58-guttman-rmse-coverage/run_experiment.R \
  --full \
  --reps 150 \
  --cores 8 \
  --results-dir experiments/58-guttman-rmse-coverage/results
```

The prior 324-cell run (rho `0`/`0.35`) took about 75 minutes at `--reps 150`
on 8 cores. This grid is 2592 cells (8x larger), so at `--reps 150` budget on
the order of 8-10 hours at 8 cores with the current SE path; the inference-speed
fix should cut that several-fold. For a quicker pass drop to the full-mode
default `--reps 50` (roughly a third of the time) or trim an axis.

If launching detached, keep logs in the ignored results directory:

```sh
nohup Rscript experiments/58-guttman-rmse-coverage/run_experiment.R \
  --full \
  --reps 150 \
  --cores 8 \
  --results-dir experiments/58-guttman-rmse-coverage/results \
  > experiments/58-guttman-rmse-coverage/results/run.log 2>&1 &
echo $! > experiments/58-guttman-rmse-coverage/results/run.pid
```

## Outputs

The script writes summaries incrementally after each cell, so partial runs are
inspectable:

- `design.csv`: requested condition grid.
- `population.csv`: population diagnostics for each design.
- `estimate_summary.csv`: parameter RMSE, bias, empirical-SE coverage.
- `estimate_joint_summary.csv`: same-replication RMSE ratios against NTML.
- `whole_summary.csv`: common-covariance, Sigma, common-diagonal, and residual
  diagonal whole-estimator RMSE.
- `whole_joint_summary.csv`: same-replication whole-estimator RMSE ratios
  against NTML.
- `diagnostic_summary.csv`: fit success, SE success, convergence, improper
  residual/Sigma diagnostics.
- `timing_summary.csv`: fit, SE, and total timing by estimator and condition.
- `progress.csv`: per-cell elapsed time and draw status.

Raw draw-level estimate tables are omitted by default to avoid memory and disk
pressure. Add `--keep-draws` if they are needed.

## Report

Render the report after results exist:

```sh
quarto render experiments/58-guttman-rmse-coverage/report.qmd
```

To render a probe or alternate result directory:

```sh
quarto render experiments/58-guttman-rmse-coverage/report.qmd \
  -P results_dir:results/probe
```

The `results_dir` parameter is resolved relative to the experiment directory.
