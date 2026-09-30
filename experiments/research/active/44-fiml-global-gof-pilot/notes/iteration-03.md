# Iteration 03: representative SEM preflight

## Purpose

This is a timing and execution preflight, not a calibration simulation. It
checks whether a compact latent-SEM panel can be generated and fitted before
spending thousands of replications per cell. The existing affine independence
bridge remains the only design in this experiment with a correctly labelled
effective multiplier global score test.

## Design

The four null populations are a six-indicator one-factor CFA (9 df), the exact
ten-indicator two-factor CFA used in the Foldnes--Moss--Grønneberg setup
(34 df), a twelve-indicator orthogonal bifactor model (42 df), and a five-wave
linear growth model (10 df). The bifactor uses four indicators per specific
factor because its nine-indicator version triggered the FMG projected-rank
guard more often at `n = 120`.

Each model is crossed with five native magmaan generators: normal, VM and IG
at target marginal skewness/excess-kurtosis `(2, 7)`, and VM and IG at `(3, 21)`.
IG uses a symmetric covariance root and Pearson independent generators. Each
law is observed completely or with 30% MCAR on every variable except the first,
which is always observed. The retained run used `n = 120`, ten replications per
cell, and four cell workers: 40 cells and 400 FIML fits.

The runner records the ordinary likelihood-ratio p-value, the dedicated FIML
MLR/Yuan--Bentler result, and SB, SS, pEBA(4), and `all` FMG corrections. It does
not compute the effective multiplier score. General CFA-versus-saturated FIML
projection is not exposed by the current score machinery, so using the affine
flip statistic here would give the wrong test a persuasive label.

## Execution result

All 20 native generator calibrations succeeded. All 400 FIML fits converged and
their degrees of freedom matched 9, 34, 42, and 10. MLR was finite in 398/400
replications. FMG was available in 388/400: the one-factor, two-factor, and
growth models were 300/300, while the bifactor was 88/100. Every FMG failure was
the same strict rank diagnostic: a nominally projected-out eigenvalue of about
`1e-6` to `1e-4` was not accepted as numerical zero. This is a useful preflight
finding, not evidence about statistical calibration.

The final smoke took 3.6 seconds of simulation wall time after R startup and
generator setup. Using the observed 3.43x four-worker speedup, the current
40-cell null-only panel projects to approximately 36 seconds for 100 reps/cell,
3.0 minutes for 500, and 11.9 minutes for 2,000. Doubling to one matched power
condition projects to 1.2, 6.0, and 23.8 minutes respectively (8,000, 40,000,
and 160,000 total fits). These are machine-local, small-run extrapolations and
exclude the not-yet-implemented general projected score/multiplier work.

## Decision

The model and generator panel is cheap enough to keep. Before a large inferential
run, resolve or explicitly prespecify handling of the bifactor FMG rank guard,
implement the general FIML saturated-moment H0 score projection, and define
matched misspecifications for power. A 500-rep-per-cell null-plus-power pilot is
only about six projected minutes on this machine; 2,000 reps/cell is still under
half an hour at the present model sizes, before the new score cost is added.
