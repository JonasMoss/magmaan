# Resource application draft: SEM-SIM, a simulation program for structural equation modeling methods

> Draft for the Sigma2 / NRIS MAS application. Methods-developer voice, terse,
> no em-dashes or semicolons. Fill the TODOs (PI, institution, final numbers
> from `compute-budget.md`). Work-package blurbs are grounded in the current
> paper abstracts under `papers/`.

## Project title

SEM-SIM: high-throughput simulation for structural equation modeling methods

## Summary

We develop and validate new estimators and test statistics for structural
equation models (SEM). A recurring pattern across this work is that the hard part
of a method is a reference distribution or a finite-sample property that only
large-scale simulation can establish: weighted chi-square eigenvalue laws,
non-regular second-order limits, misspecification bias that does not vanish with
sample size, permutation calibration, and closed-form confidence-interval
coverage. These studies were previously impractical in standard R tooling. We run
them on `magmaan`, a C++23 SEM engine that reproduces the reference software
(lavaan) to documented tolerance and, for several methods, is non-iterative or
spectrum-cheap. A replicate that costs a multi-second iterative fit in R becomes
milliseconds, and resampling schemes that need many refits per replicate become
feasible. The simulations are embarrassingly parallel, one independent fit per
(condition, replicate) cell, and map directly onto Slurm array jobs. We request
Saga CPU time for the simulation program behind roughly ten active method papers.

## The new capability

Standard practice validates a SEM method by simulating a few hundred replicates
across a handful of conditions in R, because each lavaan fit is an iterative
optimization and a full design grid times a bootstrap or permutation loop is out
of reach. magmaan changes the unit cost in three ways that this program exploits:

1. **Speed and parity.** It is a compiled, lavaan-faithful engine, so the same
   design grid runs one to two orders of magnitude faster, and grids that were
   too large to contemplate become routine.
2. **Non-iterative estimators.** Closed-form (Guttman) CFA and closed-form
   reliability map the sample covariance directly to parameters, with no
   optimization and no convergence failures, so per-replicate cost collapses and
   resampling is affordable.
3. **Cheap spectra.** The eigenvalue-based robust tests (Foldnes-Moss-Gronneberg
   family, ordinal reference laws) get the full limiting spectrum per fit at low
   cost, which is exactly the object several of these papers must simulate.

The consequence is that the bottleneck moves from "can we run this at all" to raw
CPU throughput, which is what we are asking Saga to supply.

## Work packages (one per active paper)

Each is an independent, embarrassingly parallel simulation study. Compute
footprints are in `compute-budget.md`.

1. **Robust standard errors under misspecification** (`estimated-weight-se`).
   Least-squares SEM estimators (DWLS, WLS, GLS, ADF, two-stage) minimise a
   quadratic form whose weight is estimated from the same data, and the standard
   robust SE treats that weight as fixed, so coverage stays wrong as the sample
   grows under misspecification. Establishing the bias and validating the
   complete sandwich that corrects it needs coverage grids across estimators,
   misspecification types, and sample sizes, each a many-replicate refit.

2. **Measurement invariance with missing, non-normal data** (`fiml-fmg`). Robust
   chi-square difference tests correct only low-order moments of the non-normal
   reference law, while spectrum-based tests use the limiting eigenvalues
   directly. Testing whether the spectrum tests improve invariance inference
   under incomplete non-normal data requires direct full-information (FIML) fits
   across many non-normality, missingness, and invariance conditions. FIML at
   this scale is the central previously-infeasible case.

3. **Permutation measurement invariance** (experiment 61, paper in progress).
   The naive normal-theory difference test permuted over group labels is not
   pivotal when groups are heterogeneous, and a studentized permutation is.
   Mapping its calibration and power against the robust and asymptotic tests,
   first on complete data and then under FIML, needs a full permutation loop per
   replicate over the whole invariance grid.

4. **Reference laws for ordinal robust tests** (`ordinal-fmg`). For ordinal SEM
   fit by polychoric diagonally weighted least squares (WLSMV), even correctly
   specified data give a weighted chi-square goodness-of-fit and difference law,
   because the diagonal weight is not the efficient one. Characterising the
   correction across category counts, threshold shapes, and model sizes is a
   large categorical-estimation study.

5. **Closed-form multidimensional reliability** (`closed-form-omega`).
   Coefficient omega is normally read off a fitted CFA with a bootstrap interval,
   but the loadings can be read straight from the covariances, giving omega and
   its inference in closed form for correlated factors and weighted composites.
   Validating the closed-form intervals against the bootstrap across factor
   structures, sample sizes, and non-normality is a large coverage study that the
   non-iterative estimator makes cheap per replicate.

6. **Inference for non-iterative CFA** (`guttman-inference`). Closed-form CFA maps
   the covariance directly to parameters and never fails to converge, but has
   lacked a full inferential apparatus, which we supply by the delta method with
   a fourth-moment matrix. Validating the standard errors, goodness-of-fit, and
   difference tests needs large size and coverage grids.

7. **Non-regular reliability contrasts** (`second-order-reliability-gaps`). Some
   reliability comparisons, such as coefficient alpha against omega under equal
   loadings, are non-regular at the null, with a one-sided second-order
   quadratic-form limit rather than a normal one, so they need an Imhof-type
   reference. Pinning down the finite-sample behaviour of the corrected reference
   is a many-null-draw simulation.

8. **Native maximum likelihood for composite SEM** (`composite-ml`).
   Factor-and-composite SEM brings composite indicators into the SEM likelihood
   by treating composite weights and within-composite moments as parameters.
   Validating the estimator, its expected-information inference, and lavaan-style
   output requires parity and coverage simulations across composite designs.

9. **Optimizer robustness for least-squares SEM** (`snlls-continuous`,
   `ordinal-snlls`). Whether an estimate reproduces, and how fast it computes,
   depends on the optimizer, not only the estimator. Benchmarking separable
   nonlinear least squares against full optimization, for continuous and ordinal
   models across a hard model corpus, is itself a large compute task that
   measures hard-failure rates.

10. **Polychoric SEM** (`h-polychorics-sem`). TODO: one-line motivation from the
    paper lead. Categorical fits are slow, so its grid benefits directly from the
    engine speed.

## Computational method and why HPC

Each replicate is one or more SEM fits. Cells and replicates are fully
independent, so the natural unit is a Slurm array task per cell or per replicate
chunk, each writing a small CSV that a reduce step concatenates. There is no
inter-task communication and no GPU need. We already run this shape on a laptop
and on paid cloud (Modal, one container per cell), which caps our scale. Worked
example, measured: one complete-data invariance permutation grid of 963 cells at
1,000 replicates and about 200 permutation refits each is roughly 700
CPU-core-hours. Its FIML counterpart (work package 2) is 10 to 100 times heavier.

## Resource request

- **Compute:** see `compute-budget.md`. Headline request: TODO CPU-core-hours for
  the 1 Oct 2026 to 31 Mar 2027 period, dominated by the FIML, ordinal, and
  permutation work packages.
- **Storage:** modest. Inputs are checked-in JSON fixtures, outputs are
  rectangular CSVs. A few hundred GB of NIRD project storage is ample, with no
  large primary datasets.

## Software and technical readiness

`magmaan` is self-contained and already builds on Saga (documented in
`dev/saga/`). It is C++23 (GCC 13+) with a small R interface. Third-party code
(the PORT optimizer, QUADPACK) is vendored, and NLopt is resolved through the
`nloptr` CRAN package, so there is no exotic dependency chain and no CMake needed
for the R build. Runs are reproducible: deterministic per-cell seeding, pinned
package versions, and result CSVs with stable schemas.

## Track record and group

TODO: PI name and Norwegian-institution affiliation, coauthors, and prior output.
The repository already holds a large body of validated simulation experiments and
lavaan-parity checks, which is direct evidence of an efficient, working pipeline.
