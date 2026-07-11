# Resource application draft: methods research for structural equation modeling

> Draft for the Sigma2 / NRIS MAS application. Methods-developer voice, terse.
> Fill the TODOs (PI, institution, exact numbers from `compute-budget.md`).

## Project title

High-performance simulation for structural equation modeling methods (magmaan)

## Summary

We develop and validate new estimators and test statistics for structural
equation models (SEM). Every new method must be checked against theory by large
Monte-Carlo simulation: Type-I error, power, and confidence-interval coverage,
measured across many data-generating conditions and thousands of replicates. The
work is organized around `magmaan`, a C++23 SEM engine that reproduces the
reference software (lavaan) to documented tolerance and adds research-tier
methods on top. The simulations are embarrassingly parallel - one independent
model fit per (condition, replicate) cell - and map directly onto Slurm array
jobs. We request Saga CPU time to run the simulation pipeline behind roughly ten
active method papers.

## Scientific background and objectives

SEM is the dominant latent-variable framework in psychology, education, and the
social and health sciences. Its inferential guarantees (test calibration,
standard-error validity) are asymptotic, and hold or fail in finite samples in
ways only simulation can establish. Our program targets the cases where the
standard tooling is known to be unreliable:

- **Robust difference tests for measurement invariance.** Scaled chi-square
  difference tests (Satorra-Bentler / Yuan-Bentler family) are poorly calibrated
  under non-normality, missingness, and group-size imbalance. We study
  permutation and studentized alternatives that recover finite-sample validity.
- **Non-normal and incomplete data.** Full-information (FIML) and two-stage
  estimation with robust corrections (the Foldnes-Moss-Gronneberg / pEBA
  eigenvalue family), evaluated across calibrated non-normal generators.
- **Ordinal and polychoric SEM.** Categorical estimation (DWLS/ULSMV,
  polychoric-based methods) whose finite-sample behavior is expensive to map.
- **New estimators.** Closed-form (Guttman) CFA, separable nonlinear least
  squares (SNLLS), composite/Henseler-Ogasawara models, and reliability
  inference, each requiring calibration and coverage studies.

The common objective is honest finite-sample characterization: for each method,
the design grid crosses data generator, sample size, model size, and (where
relevant) invariance level or missingness mechanism, at 1,000-5,000 replicates
per cell. This routinely means hundreds of independent cells per study.

## Computational method and why HPC

Each replicate is one or more SEM fits (millisecond-to-second scale for complete
data; iterative and far heavier under FIML, bootstrap, and permutation
resampling). Cells and replicates are fully independent, so the natural unit is a
Slurm array task per cell (or per replicate chunk), each writing a small CSV that
a reduce step concatenates. There is no inter-task communication and no GPU need;
the workload is pure CPU throughput. We already run this shape on a laptop and on
paid cloud (Modal, one container per cell), which caps our scale and cost. Saga
lets us run the full grids, and the resampling-based methods (permutation,
bootstrap) that are otherwise infeasible.

Worked example (measured): one complete-data measurement-invariance permutation
grid - 963 design cells, 1,000 replicates, ~200 permutation refits each - is
about 700 CPU-core-hours. Its FIML counterpart, which the invariance program
needs, is 10-100x heavier because each permutation refits an iterative
full-information model. `compute-budget.md` scales this anchor across the
portfolio.

## Resource request

- **Compute:** see `compute-budget.md`. Headline request: TODO CPU-core-hours for
  the 1 Oct 2026 - 31 Mar 2027 period, dominated by the FIML/ordinal/permutation
  studies.
- **Storage:** modest. Inputs are checked-in JSON fixtures; outputs are
  rectangular CSVs. A few hundred GB of project storage (NIRD) is ample; no
  large primary datasets.

## Software and technical readiness

`magmaan` is self-contained and already builds on Saga (documented in
`dev/saga/`). It is C++23 (GCC 13+) with a small R interface; third-party code
(the PORT optimizer, QUADPACK) is vendored, and NLopt is resolved through the
`nloptr` CRAN package, so there is no exotic dependency chain and no CMake needed
for the R build. Runs are reproducible: deterministic per-cell seeding, pinned
package versions, and result CSVs with stable schemas. The team maintains the
engine, the simulation harness, and the fan-out tooling.

## Track record and group

TODO: PI name, Norwegian-institution affiliation, coauthors, and the list of
in-progress papers (see `compute-budget.md` rows). Note the existing validated
simulation experiments in the repository as evidence of an efficient, working
pipeline.
