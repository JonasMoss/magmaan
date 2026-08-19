# Iteration 05: curved-SEM global effective multiplier

Date: 2026-08-20

## Decision

Ship the direct saturated-score construction as the production global test.
For each fitted ML/FIML model it evaluates casewise saturated likelihood scores
at the null-implied moments, constructs the observed-pattern conditional Fisher
metric, and removes the fitted SEM tangent. The effective multiplier reference
then reweights the projected rows without fitting an H1 model or refitting the
null.

The saturated-EM moment influence remains an independent validation route. It
is not used to define the production statistic.

## Initial contract

- continuous single- or multi-block ML/FIML;
- random-X only;
- affine equality constraints allowed through the tangent map;
- no active parameter bounds, inequality constraints, or nonlinear equality
  constraints;
- missing-data models require a mean structure;
- effective multiplier and asymptotic calibration only.

The result reports saturated moment dimension, tangent rank and conditioning,
model-tangent score stationarity, multiplier Monte Carlo error, asymptotic
mixture/mean-scaled/direct-sandwich comparators, and component timings.

## Validation gates

1. Direct saturated case scores times the model moment Jacobian reproduce the
   ordinary model-parameter score row-for-row under missing-pattern FIML.
2. An all-observed FIML mask and the complete-data representation produce the
   same statistic, eigenvalue spectrum, and multiplier p-value.
3. Covariance-only complete ML has the expected saturated dimension and df.
4. A multi-pattern FIML call is deterministic under fixed seed and supports
   within-pattern multiplier-score centering.
5. The friendly C++ API and R wrapper are exercised separately.

All focused debug tests passed: 54 assertions in the three global-score cases,
271 assertions in the full score-flip group, and 23 assertions in the frontier
API case.

## Representative-SEM timing smoke

Command:

```sh
Rscript experiments/77-fiml-global-gof-pilot/run_sem_models.R --reps 3 \
  --flips 199 --cores 4 \
  --results-dir experiments/77-fiml-global-gof-pilot/results/sem-global-flip-smoke
```

The 40 cells produced 120/120 converged fits, finite MLR values, and finite
global multiplier p-values. Every global-test df matched the model's expected
df. Mean global-test cost was 0.0219 seconds per fit (maximum 0.209 seconds).
Mean costs by model were about 0.003--0.005 seconds for the one-factor/growth
models, 0.019--0.031 seconds for the two-factor model, and 0.051--0.053 seconds
for the bifactor. Total wall time including fitting, MLR, and FMG was 2.6
seconds on four workers.

The measured full-panel projection is 7.2 minutes for 500 null replications per
cell and 14.4 minutes for 1,000. A matched power panel doubles those times.
Three replications per cell are not used for calibration claims.
