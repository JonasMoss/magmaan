# Heywood Box-Constraint Experiment

This experiment studies whether variance box constraints turn the hard
random-start Study 3 MSST cases from unconstrained Heywood stationary points
into admissible boundary solutions.

The motivating cases come from
`papers/convergence-note/random-start-cartography`: random starts often find
stationary ML solutions, but those solutions usually have negative observed or
latent variances. This experiment asks the next question:

> If variance parameters are constrained to be nonnegative, do the same sample
> covariances produce usable boundary optima?

## Current Scope

Both the C++ ML fitting path and the R `magmaan(..., estimator = "ML")` surface
accept bounds. The runner fits lavaan and magmaan under the same named bound
presets (`none`, `pos.var`, `standard`, `wide`) and the same identification
conditions (`marker`, `std.lv`) so the boundary geometry can be inspected
directly.

The bounded problem is a different optimization problem from lavaan parity:
active variance bounds should be judged with KKT/active-bound diagnostics, not
with the ordinary unconstrained gradient alone.

The report is meant to answer two narrow questions. First, if unbounded ML
converges to negative variances but `pos.var` converges with no negative
variances and active lower bounds, the problem is not simply an optimizer
failure. It is evidence that the unconstrained SEM optimum lies outside the
admissible variance region, while the bounded problem has a boundary optimum.
Second, if `std.lv` without bounds still has negative observed variances, the
Heywood behavior is not merely a marker-identification artifact. These results
matter for engineering convergence checks; they do not by themselves justify
turning bounds or `std.lv` into statistical defaults. The `standard` and `wide`
lavaan boxes are included as supporting probes; `wide` is deliberately loose
and can still admit negative variance estimates.

## Run

Install the R package first if needed:

```sh
just r-install
```

Quick smoke:

```sh
Rscript experiments/03-heywood_box_constraints/run_experiment.R --rstarts 0 --cases study3_msst:997
```

Default run:

```sh
Rscript experiments/03-heywood_box_constraints/run_experiment.R
```

Render the report:

```sh
(cd experiments/03-heywood_box_constraints && quarto render report.qmd)
```

## Outputs

Generated files live under `results/` and are ignored by git:

- `lavaan_bounds.csv`: lavaan ML fits by case, bound mode, and random-start count.
- `magmaan_bounds.csv`: magmaan ML fits by case and bound mode.
- `summary.csv`: compact per-case/per-bound summary.
- `magmaan_summary.csv`: compact magmaan summary by case and bound mode.
- `engine_comparison.csv`: lavaan-vs-magmaan objective/admissibility comparison,
  with magmaan's full-discrepancy `fmin` compared to `2 * lavaan_fmin`.
- `surface_gap.csv`: current magmaan R bounds-surface status.

All fit-output files include an `identification` column with `marker` and
`std.lv` conditions.

## Defaults

- Cases: `study3_msst:997,study3_msst:805,study3_msst:592`.
- Identification: `marker,std.lv`.
- Bound modes: `none,pos.var,standard,wide`.
- Lavaan random-start counts: `0,150`.
