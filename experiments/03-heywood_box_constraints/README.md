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

The C++ ML fitting path accepts bounds, but the current R ML wrapper does not
thread `bounds` through for `estimator = "ML"`. Until that R surface is exposed,
the runner uses lavaan bounded ML as a reference probe and records a magmaan
unbounded baseline.

The bounded problem is a different optimization problem from lavaan parity:
active variance bounds should be judged with KKT/active-bound diagnostics, not
with the ordinary unconstrained gradient alone.

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
- `magmaan_unbounded.csv`: magmaan unbounded ML baseline for the same cases.
- `summary.csv`: compact per-case/per-bound summary.
- `surface_gap.csv`: current magmaan R bounds-surface status.

## Defaults

- Cases: `study3_msst:997,study3_msst:805,study3_msst:592`.
- Bound modes: `none,pos.var,standard,wide`.
- Lavaan random-start counts: `0,150`.

