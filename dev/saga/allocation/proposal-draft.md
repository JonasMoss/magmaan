# Resource application draft: SEM-SIM, a simulation program for structural equation modeling methods

> Draft for the Sigma2 / NRIS MAS application. Terse, no em-dashes or semicolons.
> Keep it short: Sigma2 wants a tight project description plus a credible
> resource justification, not a ten-page grant. Fill the TODOs (PI, institution,
> funding source, final numbers from `compute-budget.md`). The four flagship
> blurbs are grounded in the current paper abstracts under `papers/`.

## Project title

SEM-SIM: high-throughput simulation for structural equation modeling methods

## Summary

We develop and validate new estimators and test statistics for structural
equation models (SEM). The hard part of these methods is almost always a
reference distribution or a finite-sample property that only large-scale
simulation can establish: weighted chi-square eigenvalue laws, non-regular
second-order limits, misspecification bias that does not vanish with sample size,
and permutation calibration. These studies were impractical in standard R tooling.
We run them on `magmaan`, a C++23 SEM engine that reproduces the reference
software (lavaan) to documented tolerance and, for several methods, is
non-iterative or spectrum-cheap. A replicate that costs a multi-second iterative
fit in R becomes milliseconds, and resampling schemes that need many refits per
replicate become feasible. The work is embarrassingly parallel, one independent
fit per (condition, replicate) cell, and maps directly onto Slurm array jobs.

## Why the capability is new

magmaan changes the per-replicate cost in three ways this program exploits: it is
a compiled, lavaan-faithful engine, so full grids run one to two orders of
magnitude faster; its closed-form (Guttman) CFA and reliability estimators are
non-iterative, so per-replicate cost collapses and resampling is affordable; and
its eigenvalue-based robust tests return the full limiting spectrum per fit at low
cost, which is the object several of these studies must simulate. The bottleneck
moves from "can we run this at all" to raw CPU throughput.

## Flagship studies

Four representative studies, each embarrassingly parallel. Footprints in
`compute-budget.md`.

1. **Robust standard errors under misspecification** (`estimated-weight-se`, the
   anchor). Least-squares SEM estimators (DWLS, WLS, GLS, ADF, two-stage)
   minimise a quadratic form whose weight is estimated from the same data, and
   the standard robust SE treats that weight as fixed, so its coverage stays
   wrong as the sample grows under misspecification. We derive and validate the
   complete sandwich that restores it. The validation is a coverage program
   across estimators, misspecification types, and sample sizes, and it extends
   directly to measurement-invariance difference testing, where misspecified
   nuisance structure is the rule rather than the exception.

2. **Measurement invariance with missing, non-normal data** (`fiml-fmg`, plus a
   permutation extension). Robust chi-square difference tests correct only
   low-order moments of the non-normal reference law, while spectrum-based tests
   use the limiting eigenvalues directly. We test whether the spectrum tests, and
   a studentized permutation alternative, improve invariance inference under
   incomplete non-normal data, using direct full-information maximum likelihood
   (FIML). FIML across many non-normality, missingness, and invariance
   conditions, with permutation resampling per replicate, is the central
   previously-infeasible case and the heaviest single study.

3. **Reference laws for ordinal robust tests** (`ordinal-fmg`). For ordinal SEM
   fit by polychoric diagonally weighted least squares (WLSMV), even correctly
   specified data give a weighted chi-square goodness-of-fit and difference law,
   because the diagonal weight is not the efficient one. We characterise the
   eigenvalue correction across category counts, threshold shapes, and model
   sizes, a large and slow categorical-estimation study.

4. **Inference for non-iterative CFA** (`guttman-inference`). Closed-form CFA maps
   the sample covariance directly to parameters and never fails to converge, but
   has lacked a full inferential apparatus, which we supply by the delta method
   with a fourth-moment matrix. Validating the standard errors, goodness-of-fit,
   and difference tests needs large size and coverage grids that the
   convergence-free estimator makes painless to run at scale.

## Broader relevance

These four are a subset. The recurring need behind magmaan is broader: any
resampling-heavy inference (bootstrap intervals, permutation tests, jackknife and
cross-validation for case influence) and any robust or spectrum-based test whose
reference law must be simulated. Both are core to magmaan and both are
simulation-bound, so studies of exactly this shape recur across the group's work.
We therefore apply for the program, not a single study. Additional active studies
drawing on the same allocation include closed-form multidimensional reliability
(`closed-form-omega`), native maximum likelihood for composite SEM
(`composite-ml`), separable least squares for continuous and ordinal SEM
(`snlls-continuous`, `ordinal-snlls`), non-regular reliability contrasts
(`second-order-reliability-gaps`), and polychoric SEM (`h-polychorics-sem`).

## Resources, software, readiness

Compute request: see `compute-budget.md`, dominated by the FIML, ordinal, and
permutation work. Storage is modest, a few hundred GB of NIRD project storage,
inputs are checked-in JSON fixtures and outputs are rectangular CSVs. `magmaan`
already builds on Saga (documented in `dev/saga/`): it is C++23 (GCC 13+) with a
small R interface, third-party code is vendored, and NLopt comes through the
`nloptr` CRAN package, so there is no exotic dependency chain. Runs are
reproducible with deterministic per-cell seeding, pinned package versions, and
stable result schemas.

## PI, funding, track record

TODO: PI name and Norwegian-institution affiliation, funding source (mandatory in
MAS), coauthors, and prior output. The repository already holds a large body of
validated simulation experiments and lavaan-parity checks, direct evidence of an
efficient working pipeline.
