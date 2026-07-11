# Permutation Measurement Invariance

Experiment 61 evaluates fully recomputed, studentized Wald permutation tests
for multi-group CFA measurement invariance. Every permutation relabels cases,
refits the unrestricted model for the tested invariance step, rebuilds the
empirical sandwich covariance, and recomputes the Wald statistic. The same
permutation also fits the constrained model and records the ordinary NT
likelihood-ratio difference, reproducing the `semTools::permuteMeasEq()`
comparison without putting robust nested-test corrections inside the loop.

## Studies

The executable has two study modules.

1. **Reference-law study.** Two groups and two correlated factors with `p = 8`
   or `16`, total `N = 220, 440, 1760`, and `1:1` or `1:3` allocation. Normal,
   Pearson independent-generator `(skew, excess kurtosis) = (2,7)` and `(3,21)`,
   and ordinal-probit integer-code generators are supplied by magmaan. The
   ordinal lane crosses 3/5/7 categories with symmetric or asymmetric
   thresholds and analyzes the codes by continuous ML. Heterogeneous invariant
   nulls differ in factor means and factor/residual covariance; exact-
   exchangeability controls are separate cells. This lane is intentionally
   misspecified: latent-response invariance does not guarantee equality of the
   continuous-ML pseudo-parameters after nonlinear discretization, so that
   pseudo-parameter drift is reported rather than hidden.
2. **Unbalanced-data study.** Chen and Chao's one-factor, six-indicator model,
   with group A fixed at 200, group B in `{400,600,800,1000,1400,2000,3000}`,
   and total-N-matched balanced cells. The original metric/scalar affected-item
   sets and small/large/mixed/non-uniform changes are reproduced. Strict
   invariance uses the same patterns as proportional residual-variance changes.

Both modules report the adjacent metric/configural, scalar/metric, and
strict/scalar tests. Configural is an internal reference, not a global-fit
outcome. The Chen--Chao MWT is the model-based multivariate Wald column in the
unbalanced module; the experiment adds sandwich and permutation-studentized
versions of the same restriction vector. The ordinary permutation LRT is the
established SEM baseline; the studentized permutation Wald is the proposed
asymptotically pivotal method under heterogeneous nuisance distributions.

## Run profiles

```sh
# Both wiring/timing smokes: one replication and nine permutations per cell.
Rscript run_experiment.R --smoke

# One smoke only.
Rscript run_experiment.R --study reference --smoke
Rscript run_experiment.R --study imbalance --smoke

# Representative intermediate grids.
Rscript run_experiment.R --study reference --modal --cores 10
Rscript run_experiment.R --study imbalance --modal --cores 10

# High-precision reference-law null slice: normal/IG, p=8, N=440,
# both allocations, all steps, heterogeneous and exchangeable nulls.
Rscript run_experiment.R --sensitivity --cores 10

# Paper grids. Use calibrated deltas for the reference alternatives.
Rscript calibrate_alternatives.R --reps 200 --permutations 199 --cores 10
Rscript run_experiment.R --study reference --full --cores 10 \
  --calibration-file results/alternative-calibration-deltas.csv
Rscript run_experiment.R --study imbalance --full --cores 10

# Split a full study over ten jobs sharing the same checkpoint tree.
Rscript run_experiment.R --study reference --full --shard-count 10 \
  --shard-index 1 --cores 10 --calibration-file results/alternative-calibration-deltas.csv
```

`--reps`, `--permutations`, and the cell filters shown by `--help` can turn any
profile into a targeted timing or calibration run. The default reference
alternative deltas are starting values for modal/pilot work, not a claim that
50% power has already been achieved.

`calibrate_alternatives.R` uses common random seeds and a bracketed search for
50% studentized-permutation power in the normal, balanced, `p = 8`, `N = 440`
reference cells. Its `*-deltas.csv` output can be passed directly to the runner;
`--smoke` exercises that calibration pipeline cheaply without producing usable
power estimates.

## Checkpoints and outputs

Each study/profile writes to `results/<study>-<profile>/`. Replications are
checkpointed in stable-cell/chunk files under `raw/cell_NNNN/`; rerunning the
same command skips completed replications. A changed configuration is rejected
instead of silently mixing results, so use another `--results-dir` for a new
run. Shards share the manifest and raw tree; each shard summarizes every cell
available when it finishes, so the last finishing shard produces the complete
summary.

Each output contains:

- `manifest.csv`: the exact requested design cells and stable IDs;
- `run_config.csv` and `metadata.csv`: run settings and elapsed time;
- `raw/cell_NNNN/chunk_*.csv`: restartable replication records;
- `summary.csv`: rejection, availability, failure, and timing summaries;
- `failures.csv`: observed-fit failures, when present.

Render `report.qmd` after either smoke. It selects full, sensitivity, modal,
then smoke results independently for each study and labels the selected
profiles.
