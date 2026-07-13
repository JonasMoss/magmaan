# Flip calibration frontier

This is a focused follow-up to the illustrative Foldnes–Moss–Grønneberg probe.
It is not a replication. The design keeps `G = 8` and the tested rank fixed at
28 while separating sample size, group allocation, information geometry,
nonnormality, and the number of free nuisance loadings.

The two model sizes deliberately test the same restrictions: equality of the
non-marker loadings for `x2:x5` across eight groups. With `p = 5` those are all
non-marker loadings; with `p = 20`, `x6:x20` remain group-specific nuisance
parameters. This makes a clean standardized-versus-effective flip comparison:
restriction count and degrees of freedom do not change when nuisance dimension
does.

## Design

- Broad null screen: 128 cells, 1000 replications, 199 random flips.
- Focused null/power study: 32 cells, 2000 replications, 999 random flips.
- Severe nonnormal generators target marginal skewness 3 and excess kurtosis
  21. The screen uses normal, Vale–Maurelli, independent-generator, and
  polynomial-logistic generators; the focus uses normal and polynomial-logistic.
- Primary methods: effective flip, standardized flip, and score pEBA4.
- Diagnostics: basic flip; score chi-square, SB, MV, SS, pEBA2/4/6, PALL, ALL,
  and sandwich; nested ML and a small RLS FMG set.
- Rejection is `p <= .05`, which is the exact Monte Carlo convention. A
  `p < .05` sensitivity column makes the 199-flip grid effect visible.

Power deviations are the centered published `G = 8` loading patterns on
`x2:x5`. The multiplier is calibrated separately for `p = 5` and `p = 20` to
roughly 50% effective-flip power at `n_avg = 100` under homogeneous balanced
normal data. The `n_avg = 50` alternative scales by `sqrt(100 / 50)`.

## Local commands

```sh
Rscript experiments/64-flip-calibration-frontier/run_experiment.R --smoke

Rscript experiments/64-flip-calibration-frontier/calibrate_power.R \
  --output experiments/64-flip-calibration-frontier/results/calibration/power_calibration.csv

Rscript experiments/64-flip-calibration-frontier/run_experiment.R --focus \
  --power-calibration-file experiments/64-flip-calibration-frontier/results/calibration/power_calibration.csv
```

Completed chunks are resumed only if the saved manifest and run configuration
match. Use a new results directory to change replications, flips, seeds,
calibration, filters, or shard layout.

## Modal

The Modal harness defaults to a dry run and has no retries. It fans out one
container per design cell (128 screen, 32 focus) so every cell finishes well
inside its timeout; the runner shards on the filtered row index, so shard `j` is
design row `j`. The launch gate is a measured expected-cost estimate (~$8.7 with
calibration, ~$8.1 without) against a $14 in-code guard, with the absolute
worst case printed for context. Actual billing tracks the expected estimate:
containers stop when their cell finishes. The guard is not a hard billing limit;
create a dedicated Modal environment with an account-side budget first.

```sh
cd experiments/64-flip-calibration-frontier/modal
modal run app.py                                       # dry run, no spend
modal run app.py --mode preflight --run-id july13      # image + 2-rep smoke
modal run --detach app.py --mode all --run-id july13   # substantive run
```

The preflight builds the image, runs a two-replication calibration diagnostic,
and executes one screen cell. A substantive launch is intentionally separate.

A substantive `--mode all` (or `--mode screen` / `--mode focus`) hands the whole
pipeline to a server-side `drive` function and returns immediately, so the
multi-hour run does not depend on the launching process. Launch it with
`modal run --detach` so the app persists after the entrypoint exits; watch it
with `modal app logs <app-id>` and pull with `modal volume get
exp64-flip-results july13 ./july13`. Pass `--skip-calibration` to reuse an
existing `power_calibration.csv` on the volume.

## Interpretation contract

Do not report a pooled rejection rate by itself. The report shows cellwise
Wilson intervals and summarizes the distribution across cells with RMSE, mean
absolute deviation, extrema, and the proportions below 2.5%, within 2.5–7.5%,
and above 7.5%. Power is shown both at the nominal threshold and at the matched
null empirical critical value. Effective and standardized flips are paired by
replication, including decision disagreement and incremental timing.

