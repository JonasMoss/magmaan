# Simulation study principles

Adopted 2026-10-09 with the user (TASK-103) for magmaan's decision and
calibration studies. The fit-index study (`experiments/decisions/09-fit-indices`)
is the first to use them. A study's committed criteria may depart from a
default here only with a stated reason.

## 1. A tiered, shared basis

- **Tier 1: representative models.** The sem-score set of representative
  models, held as plain data in `experiments/_support` (syntax and population
  values; no SEM logic there). Each model carries one natural misspecification:
  a single parameter that the generating model has and the fitted model omits
  (a cross-loading, residual covariance or cross-lag, whichever is natural for
  that model). Misfit size is set by solving for that parameter's value so the
  population index hits prespecified levels, for example RMSEA 0, .03, .05 and
  .08 in the ML metric.
- **Tier 2: study-specific additions**, each with a stated reason (for example
  frozen textbook populations for an ordinal estimator, multigroup cells for a
  pooling convention, small-df models where truncation dominates).
- **Tier 3, later: richer mechanisms**, such as populations with many small
  misfits (Cudeck-Browne or Wu-Browne style) or complex missingness, after
  their own design discussion.

## 2. Targets

- Each estimator is judged against the population quantity it is consistent
  for: its own discrepancy at the population moments, or for FIML under MAR the
  missingness-dependent population discrepancy.
- When that differs from what users mean by the quantity (for example the
  complete-data misfit under MAR), report both; the estimator's own target
  judges and the other is descriptive.
- Compute targets exactly where possible (population moments, closed forms),
  otherwise from large population draws, and state which.

## 3. Missing data

The base tier is simple MAR in the Savalei-Falk style: missingness on half of
the variables, driven by one fully observed variable, at rates 0, 15% and 30%.
Richer mechanisms belong to tier 3.

## 4. Generators and sample sizes

- Continuous data: normal, Vale-Maurelli, independent-generator and discretized
  generators, as in the sem-score study. Ordinal data: thresholded latent
  responses with Gaussian and skewed latents.
- Sample sizes include a small N where boundary or truncation effects matter,
  typically 200, 500, 1000 and 4000.

## 5. Fitting and judging

- Library defaults, judged by `fit$converged`; failures stay in the attempted
  denominators; no study-specific acceptance tolerance (the decision-study rules
  in `experiments/AGENTS.md`).
- Report per family and cell, never pooled, and list every loss.
- Default margins: a point estimator is "best" only if its RMSE is more than
  10% (relative) below every competitor's; otherwise the candidates tie, and the
  current default or the simpler candidate stays. Interval coverage is flagged
  outside 87.5-92.5% for 90% intervals and 92.5-97.5% for 95% intervals; width
  and availability are reported beside coverage.

## 6. Process

- Planner drafts stay unregistered until the principles and the study's
  criteria are agreed with the user.
- Criteria are committed before the decision run; a pilot that prices the run
  uses fresh seeds distinct from production and is not confirmatory.
- Production runs on simbox with the user's explicit authorization, in
  resumable per-cell outputs.
