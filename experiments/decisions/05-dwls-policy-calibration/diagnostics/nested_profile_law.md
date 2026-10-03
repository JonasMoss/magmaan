# DWLS nested profile-law diagnosis

Question: why does the estimated-weight nested profile spectrum have many more
terms than the number of restrictions, and does a parameter-space law resolve
that discrepancy? This is exploratory production-seed replay, not confirmation
and not a policy change.

## Short answer

The excess rank is a finite-sample plug-in cancellation artifact, not an
arithmetic error in the implemented Schur complement. Separate fitted points
leave a slowly vanishing, indefinite layer; positive-tail truncation inflates
its trace. Using its positive-term count in pEBA additionally dilutes the real
restriction eigenvalues. This is unsuitable as the current nested first-order
reference construction. The observed parameter-space IJ law removes that layer.
Its common-point profile counterpart agrees to relative error below 4e-14 on
all 300 draws, and both have exactly ten positive terms per draw.

Recommend registering confirmation of the **observed-Hessian parameter-space
estimated-weight IJ law**, with SB and pEBA4 retained as named candidate tails
and exact-mixture/shifted tails as comparators. Fix cells and fresh seeds before
running, include both correct and invariant misspecified models, binary and
five-category indicators, and increasing N. This diagnosis does not establish
release calibration and does not change the adopted policy. If confirmation
cannot be completed before 0.2.0, typed unavailability is preferable to claiming
the separate-point profile reference is calibrated.

## Evidence

All three cells completed 100 production-seed replicates with zero failures.
The final replay took 227.1 seconds on two workers with one math thread each.
Rows below give raw T's Monte Carlo moments and the average theoretical moments
of each positive weighted chi-square law; common-point results equal the
parameter-space results to numerical precision and are omitted from the table.

| Cell | Law | MC mean T | MC variance T | Mean trace | Mean 2 sum(lambda²) | Positive terms | Terms above relative 0.001 |
|---|---|---:|---:|---:|---:|---:|---:|
| 53 | Separate profile | 14.37 | 39.40 | 18.14 | 50.71 | 83.08 | 55.27 |
| 53 | Fixed Satorra-2000 | 14.37 | 39.40 | 13.20 | 37.34 | 10 | 10 |
| 53 | Parameter IJ | 14.37 | 39.40 | 14.18 | 44.35 | 10 | 10 |
| 75 | Separate profile | 11.24 | 19.26 | 12.07 | 24.31 | 183.37 | 98.28 |
| 75 | Fixed Satorra-2000 | 11.24 | 19.26 | 10.70 | 23.70 | 10 | 10 |
| 75 | Parameter IJ | 11.24 | 19.26 | 10.80 | 24.24 | 10 | 10 |
| 83 | Separate profile | 11.20 | 20.55 | 12.34 | 25.45 | 183.02 | 100.17 |
| 83 | Fixed Satorra-2000 | 11.20 | 20.55 | 10.72 | 23.77 | 10 | 10 |
| 83 | Parameter IJ | 11.20 | 20.55 | 11.09 | 25.51 | 10 | 10 |

The separate profile's signed traces are 15.25, 10.81 and 11.08; discarded
negative mass is respectively 2.89, 1.26 and 1.26. In the five-category cells,
that discarded mass almost exactly explains the difference between the profile
positive trace and the parameter-space trace. Relative Euclidean discrepancy
between the profile's top ten eigenvalues and the parameter spectrum averages
7.71%, 2.86% and 3.27%. The many smaller terms are therefore not additional
restrictions.

At nominal 5%, SB / pEBA4 / exact rejection rates for the separate profile are
0% / 5% / 1% (53), 1% / 18% / 1% (75), and 2% / 14% / 2% (83).
For parameter IJ they are 6% / 5% / 5%, 5% / 5% / 5%, and 6% / 6% / 6%.
The fixed-weight comparator gives 7%, 5% and 6% for each of those three tails.
The full family, including per-arm availability denominators, is in
[summary.csv](results/summary.csv). Every arm was available on every draw.

A 5% rejection estimate from 100 draws has approximately 2.2 percentage points
of Monte Carlo standard error. Means have standard errors around 0.44–0.63;
variance estimates are noisier. The comparison is paired but exploratory and
covers only these three cells. This replay identifies the cancellation
mechanism, not its empirical rate of disappearance with N; the
O_p(N^-1/2) residual-difference layer follows the regular nested-null expansion.
It cannot prove broad calibration or select a tail from small observed size
differences. Earlier debugging runs, including an incorrect covariance
contraction and nominal-df FMG truncation, are excluded from all retained
summaries.

## Construction and cancellation

`ordinal_dwls_profile_lrt` calls `ordinal_dwls_profile_rmsea` once with each
model's own fitted parameter vector. Each call evaluates its moment Jacobian,
residual and analytic observed bread at that point. The joint NACOV of sample
moments and estimated diagonal NACOV entries is shared, since it depends on
the same first-stage data. `weighted_moment_profile_lrt` then subtracts the two
already constructed matrices numerically; it does not cancel their shared
terms analytically.

For each group, the direct extended-moment metric has blocks
`[W, diag(r/gamma^2); diag(r/gamma^2), diag(r^2/gamma^3)]`.
The top-left weight block cancels, but the residual-dependent blocks need not
cancel at separately fitted points. The analytic formula is the profile Schur
complement `V0 - Wstar D K (K' H K)^-1 K' D' Wstar`.
An indefinite contrast is expected from this construction: the primitive records
signed and negative traces but uses only positive eigenvalues for its tails.
Those small positive terms contribute to the SB scale, and counting them in
pEBA changes the partition of the spectrum.

At one common point the direct metrics cancel exactly. The remainder factors
through the restriction space and has rank at most the number of restrictions.
For observed sensitivity H, IJ score meat B and restriction map A its nonzero
spectrum is that of

`(A H^-1 A')^-1 A H^-1 B H^-1 A'`.

The internal lab accessor returns the exact total ordinal Newton Hessian, H1's
constraint basis K, and `embed_nested_null`'s exact A. The runner divides the
Hessian by total N, contracts the IJ covariance with K's left inverse, and
multiplies it by N; this supplies `H^-1 B H^-1` without separately reconstructing
B. A Cholesky whitening of `A H^-1 A'` gives a symmetric eigenproblem. Pure-merge
K is not necessarily orthonormal, so using K' alone to contract covariance
would be incorrect.

For the common-point comparator the accessor embeds the null's constraint
tangent into H1's structure and evaluates both profiles at H1's fitted theta.
That theta need not satisfy the null: this is a curvature diagnostic, not a
restricted fit or a new statistic. Raw T always remains the actual fitted
`N (F_H0 - F_H1)` difference. No objective, policy or default is changed.

## Replay and conventions

The three cells are 75 (five categories, total N=1000, correct, threshold-to-metric
step), 83 (its omitted cross-loading .3 sibling) and 53 (binary, total N=400,
configural-to-metric step), all two-factor, two-group theta fits. Replicates use
the original generator and `817150001 + 10000 * cell_id + replicate`, matching
production. Summary CSVs are frozen small diagnostic evidence; all replicate
rows and four per-replicate positive spectra remain ignored in `results/raw.rds`.

SB uses all positive spectrum mass and the nominal restriction df, matching
production policy SB. The remaining FMG methods use `max(df, spectrum length)`
as their dimension, matching production pEBA's full-spectrum convention:
passing nominal df to FMG would silently retain only its largest df terms.
Scaled-shifted, mean-variance, scaled F, All, penalized All, EBA and pEBA at
j=2/4/6, and pOLS at gamma=4 use existing lab primitives. The exact comparator
calls the existing weighted chi-square upper-tail primitive through an internal
accessor. Successful denominators for every arm are saved separately. The
non-negligible count uses eigenvalues above `1e-3 * max(eigenvalues)`.

## Reproduce

From the repository root, install the lane lab library with the opt core, then:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
R_LIBS=~/.cache/magmaan-rlib/lane-e nice -n 10 Rscript \
experiments/decisions/05-dwls-policy-calibration/diagnostics/nested_profile_law.R \
--reps 100 --workers 2
```

The runner refuses to overwrite a completed output directory. Preserve or move
`diagnostics/results/` first when rerunning. `--reps 1` is the smoke path.
Metadata records source/native hashes, package version, source commit, command,
seeds, workers and elapsed time. The runner checkpoints batches and stops before
starting another batch after 270 seconds. Debugging runs are ignored and are
excluded from the retained evidence.
